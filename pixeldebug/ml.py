"""Experimental frozen SigLIP2/Gemma bridge. No pretrained quality claim.

Reference tokens, current tokens, and signed token differences are pooled to
bounded sequences and projected into the language embedding space. Teacher
forcing supervises JSON completion tokens only. Frozen LM still passes gradients
to the visual prefix; wrapping it in no_grad would break projector training.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    GemmaForCausalLM,
    Siglip2VisionModel,
)
from PIL import Image

PROMPT = "Compare the reference and current screenshot. Return JSON with a defects array containing category and bbox (x,y,width,height).\n"


@dataclass
class ModelConfig:
    vision_id: str = "google/siglip2-base-patch16-224"
    language_id: str = "google/gemma-2b"
    pooled_tokens: int = 16
    seed: int = 42


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class VisualBridge(nn.Module):
    def __init__(
        self,
        vision: Siglip2VisionModel,
        language: GemmaForCausalLM,
        config: ModelConfig,
    ):
        super().__init__()
        self.vision, self.language, self.bridge_config = vision, language, config
        for parameter in list(vision.parameters()) + list(language.parameters()):
            parameter.requires_grad_(False)
        self.projector = nn.Sequential(
            nn.LayerNorm(vision.config.hidden_size),
            nn.Linear(vision.config.hidden_size, language.config.hidden_size),
            nn.GELU(),
            nn.Linear(language.config.hidden_size, language.config.hidden_size),
        )
        self.roles = nn.Parameter(torch.zeros(3, 1, language.config.hidden_size))

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        self.language.eval()
        return self

    def prefix(
        self, reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
    ):
        with torch.no_grad():

            def encode(inputs):
                tokens = self.vision(**inputs).last_hidden_state
                mask = inputs["pixel_attention_mask"].bool()
                return torch.stack(
                    [
                        nn.functional.adaptive_avg_pool1d(
                            t[m].T.unsqueeze(0), self.bridge_config.pooled_tokens
                        )[0].T
                        for t, m in zip(tokens, mask)
                    ]
                )

            a, b = encode(reference), encode(current)
        parts = [
            self.projector(t.to(self.roles.dtype)) + self.roles[i]
            for i, t in enumerate([a, b, b - a])
        ]
        return torch.cat(parts, dim=1)

    def forward(self, reference, current, input_ids, attention_mask, labels):
        prefix = self.prefix(reference, current)
        text = self.language.get_input_embeddings()(input_ids)
        embeddings = torch.cat([prefix.to(text.dtype), text], dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2], dtype=attention_mask.dtype, device=input_ids.device
        )
        full_labels = torch.cat(
            [
                torch.full(
                    prefix.shape[:2], -100, device=labels.device, dtype=labels.dtype
                ),
                labels,
            ],
            dim=1,
        )
        return self.language(
            inputs_embeds=embeddings,
            attention_mask=torch.cat([prefix_mask, attention_mask], 1),
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, reference, current, prompt_ids, max_new_tokens=256):
        self.eval()
        prefix = self.prefix(reference, current)
        ids = prompt_ids
        generated = []
        for _ in range(max_new_tokens):
            text = self.language.get_input_embeddings()(ids)
            embeddings = torch.cat([prefix.to(text.dtype), text], dim=1)
            logits = self.language(
                inputs_embeds=embeddings,
                attention_mask=torch.ones(
                    embeddings.shape[:2], device=ids.device, dtype=torch.long
                ),
            ).logits
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            ids = torch.cat([ids, token], dim=1)
            if (token == self.language.config.eos_token_id).all():
                break
        return torch.cat(generated, dim=1)

    def save_checkpoint(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(
            json.dumps(asdict(self.bridge_config), indent=2)
        )
        from safetensors.torch import save_file

        save_file(
            {
                **{
                    f"projector.{k}": v.detach().cpu().contiguous()
                    for k, v in self.projector.state_dict().items()
                },
                "roles": self.roles.detach().cpu().contiguous(),
            },
            str(folder / "adapter.safetensors"),
        )

    def load_checkpoint(self, folder: Path):
        saved = json.loads((folder / "config.json").read_text())
        if saved != asdict(self.bridge_config):
            raise ValueError("Checkpoint model configuration mismatch")
        from safetensors.torch import load_file

        state = load_file(str(folder / "adapter.safetensors"))
        self.projector.load_state_dict(
            {
                k.removeprefix("projector."): v
                for k, v in state.items()
                if k.startswith("projector.")
            }
        )
        with torch.no_grad():
            self.roles.copy_(state["roles"])


def load_pretrained(config: ModelConfig):
    seed_all(config.seed)
    vision = Siglip2VisionModel.from_pretrained(config.vision_id)
    language = GemmaForCausalLM.from_pretrained(config.language_id)
    model = VisualBridge(vision, language, config)
    return (
        model,
        AutoImageProcessor.from_pretrained(config.vision_id),
        AutoTokenizer.from_pretrained(config.language_id),
    )


def encode_example(row, root, processor, tokenizer, device):
    with (
        Image.open(root / row["reference"]) as a,
        Image.open(root / row["defective"]) as b,
    ):
        pixels = processor(
            images=[a.convert("RGB"), b.convert("RGB")], return_tensors="pt"
        ).to(device)
    prompt = tokenizer(PROMPT, add_special_tokens=True)["input_ids"]
    target = tokenizer(
        json.dumps(row["diagnosis"], separators=(",", ":")), add_special_tokens=False
    )["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > 1024:
        raise ValueError("Training example exceeds 1024 token limit")
    ids = torch.tensor([prompt + target], device=device)
    labels = torch.tensor([[-100] * len(prompt) + target], device=device)
    return (
        {k: v[:1] for k, v in pixels.items()},
        {k: v[1:] for k, v in pixels.items()},
        ids,
        torch.ones_like(ids),
        labels,
    )


def train(root: Path, output: Path, epochs=1, device="cpu"):
    config = ModelConfig()
    model, processor, tokenizer = load_pretrained(config)
    model.to(device)
    rows = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not train_rows or not validation:
        raise ValueError(
            "Both family-disjoint train and validation splits are required"
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    history = []
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_rows)
        losses = []
        for row in train_rows:
            optimizer.zero_grad()
            loss = model(*encode_example(row, root, processor, tokenizer, device)).loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            values = [
                float(
                    model(*encode_example(r, root, processor, tokenizer, device)).loss
                )
                for r in validation
            ]
        val = sum(values) / len(values)
        history.append(
            {
                "epoch": epoch + 1,
                "train_example_mean_nll": sum(losses) / len(losses),
                "validation_example_mean_nll": val,
            }
        )
        if val < best:
            best = val
            model.save_checkpoint(output)
        (output / "training.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("dataset", type=Path)
    training.add_argument("--output", type=Path, default=Path("checkpoints/bridge"))
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--device", default="cpu")
    inference = sub.add_parser("infer")
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("reference", type=Path)
    inference.add_argument("current", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(args.dataset, args.output, args.epochs, args.device)
    else:
        config = ModelConfig(
            **json.loads((args.checkpoint / "config.json").read_text())
        )
        model, processor, tokenizer = load_pretrained(config)
        model.load_checkpoint(args.checkpoint)
        with Image.open(args.reference) as a, Image.open(args.current) as b:
            width, height = a.size
            pixels = processor(
                images=[a.convert("RGB"), b.convert("RGB")], return_tensors="pt"
            )
        prompt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
        completion = tokenizer.decode(
            model.generate(
                {k: v[:1] for k, v in pixels.items()},
                {k: v[1:] for k, v in pixels.items()},
                prompt,
            )[0],
            skip_special_tokens=True,
        )
        try:
            from pixeldebug.diagnostics import validate_diagnosis

            result = validate_diagnosis(completion, width, height)
            print(json.dumps({"valid_json": True, "diagnosis": result}))
        except (json.JSONDecodeError, ValueError):
            print(
                json.dumps(
                    {
                        "valid_json": False,
                        "raw_output": completion,
                        "warning": "Invalid model output; do not treat as diagnosis",
                    }
                )
            )
