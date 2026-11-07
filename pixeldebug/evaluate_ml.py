"""Offline pretrained-checkpoint evaluation; not run without Gemma access.

Invalid JSON counts as an error, not a discarded example. One-to-one greedy
matching reports detection precision/recall and category-correct recall at IoU .5.
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from pixeldebug.diagnostics import validate_diagnosis
from pixeldebug.evaluate import iou
from pixeldebug.ml import ModelConfig, load_pretrained, PROMPT


def score_predictions(predictions: list[dict], truth: list[dict]) -> dict:
    matched = set()
    hits = correct = 0
    for prediction in predictions:
        candidates = [
            (iou(prediction["bbox"], target["bbox"]), index)
            for index, target in enumerate(truth)
            if index not in matched
        ]
        overlap, index = max(candidates, default=(0, -1))
        if overlap >= 0.5:
            matched.add(index)
            hits += 1
            correct += prediction["category"] == truth[index]["category"]
    return dict(
        hits=hits, category_hits=correct, predicted=len(predictions), targets=len(truth)
    )


def evaluate(checkpoint: Path, root: Path, split="test"):
    config = ModelConfig(**json.loads((checkpoint / "config.json").read_text()))
    model, processor, tokenizer = load_pretrained(config)
    model.load_checkpoint(checkpoint)
    rows = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise ValueError("Selected split has no examples")
    totals = dict(hits=0, category_hits=0, predicted=0, targets=0)
    valid = 0
    for row in rows:
        with (
            Image.open(root / row["reference"]) as a,
            Image.open(root / row["defective"]) as b,
        ):
            width, height = a.size
            pixels = processor(
                images=[a.convert("RGB"), b.convert("RGB")], return_tensors="pt"
            )
        prompt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
        tokens = model.generate(
            {k: v[:1] for k, v in pixels.items()},
            {k: v[1:] for k, v in pixels.items()},
            prompt,
        )
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        try:
            diagnosis = validate_diagnosis(text, width, height)
            valid += 1
            scores = score_predictions(
                diagnosis["defects"], row["diagnosis"]["defects"]
            )
        except ValueError:
            scores = dict(
                hits=0,
                category_hits=0,
                predicted=0,
                targets=len(row["diagnosis"]["defects"]),
            )
        for key in totals:
            totals[key] += scores[key]
    return {
        "split": split,
        "examples": len(rows),
        "valid_json_rate": valid / len(rows),
        "precision_iou_0.5": totals["hits"] / max(totals["predicted"], 1),
        "recall_iou_0.5": totals["hits"] / max(totals["targets"], 1),
        "category_correct_recall_iou_0.5": totals["category_hits"]
        / max(totals["targets"], 1),
        "counts": totals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.dataset, args.split), indent=2))
