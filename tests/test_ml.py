"""Actual tiny randomly initialized HF architectures, not mocked model outputs."""
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import Siglip2VisionConfig, Siglip2VisionModel, GemmaConfig, GemmaForCausalLM
from pixeldebug.ml import VisualBridge, ModelConfig, seed_all


def tiny():
    seed_all(7)
    vision = Siglip2VisionModel(Siglip2VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8))
    language = GemmaForCausalLM(GemmaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    return VisualBridge(vision, language, ModelConfig(vision_id='tiny-random-siglip2', language_id='tiny-random-gemma', pooled_tokens=2, seed=7))


def test_training_gradient_checkpoint_and_greedy_inference(tmp_path):
    model = tiny()
    pixels = {'pixel_values': torch.rand(1,4,192), 'pixel_attention_mask': torch.ones(1,4,dtype=torch.long), 'spatial_shapes': torch.tensor([[2,2]])}
    current = {**pixels, 'pixel_values': pixels['pixel_values']*.5}
    ids = torch.tensor([[1,4,5,6,2]])
    labels = torch.tensor([[-100,-100,5,6,2]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.train()
    first = float(model(pixels,current,ids,torch.ones_like(ids),labels).loss.detach())
    for _ in range(12):
        optimizer.zero_grad()
        loss = model(pixels,current,ids,torch.ones_like(ids),labels).loss
        loss.backward()
        assert model.projector[1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.vision.parameters())
        assert all(p.grad is None for p in model.language.parameters())
        optimizer.step()
    final = float(model(pixels,current,ids,torch.ones_like(ids),labels).loss.detach())
    assert final < first
    model.save_checkpoint(tmp_path)
    restored = tiny()
    restored.load_checkpoint(tmp_path)
    model.eval(); restored.eval()
    assert torch.allclose(model.prefix(pixels,current),restored.prefix(pixels,current))
    a = model.generate(pixels,current,ids[:,:2],max_new_tokens=4)
    b = restored.generate(pixels,current,ids[:,:2],max_new_tokens=4)
    assert torch.equal(a,b)
    assert a.shape[1] <= 4
    print(f'tiny-random training loss: {first:.4f} -> {final:.4f}')
