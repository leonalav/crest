import torch

from crest.config import CRESTConfig
from crest.model import CRESTModel
from crest.state import detach_state


def tiny_cfg():
    return CRESTConfig(vocab_size=128, max_seq_len=8, max_steps=16, n_layers=2, d_model=32, n_heads=4, d_ffn=64, memory_slots=5, tie_embeddings=True)


def test_state_shape_stable_across_steps():
    model = CRESTModel(tiny_cfg())
    state = model.init_state(3)
    for t in range(4):
        logits, state, aux = model(torch.randint(0, 128, (3, 8)), state=state, step_idx=t)
        assert logits.shape == (3, 8, 128)
        assert len(state) == 2
        assert all(s.shape == (3, 5, 32) for s in state)
        assert 0.0 <= aux.gate_mean.item() <= 1.0


def test_zero_state_produces_zero_state_read_output_path():
    cfg = tiny_cfg()
    model = CRESTModel(cfg)
    model.set_diagnostics_enabled(True)
    x = torch.randn(2, 4, cfg.d_model)
    s = torch.zeros(2, cfg.memory_slots, cfg.d_model)
    y, probs = model.layers[0].state_read(model.layers[0].x_norm1(x), model.layers[0].s_norm(s))
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
    assert probs is not None
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-6)


def test_within_step_causality_no_future_tokens():
    torch.manual_seed(0)
    model = CRESTModel(tiny_cfg()).eval()
    state = model.init_state(1)
    tokens_a = torch.randint(0, 128, (1, 8))
    tokens_b = tokens_a.clone()
    tokens_b[:, 5:] = torch.randint(0, 128, (1, 3))
    with torch.no_grad():
        logits_a, _, _ = model(tokens_a, state=state, step_idx=0)
        logits_b, _, _ = model(tokens_b, state=state, step_idx=0)
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5)


def test_detach_state_removes_graph():
    model = CRESTModel(tiny_cfg())
    logits, state, _ = model(torch.randint(0, 128, (2, 8)), step_idx=0)
    loss = logits.mean()
    assert any(s.requires_grad for s in state)
    detached = detach_state(state)
    assert all(not s.requires_grad for s in detached)


def test_state_changes_after_write():
    model = CRESTModel(tiny_cfg())
    state = model.init_state(2)
    _, next_state, _ = model(torch.randint(0, 128, (2, 8)), state=state, step_idx=0)
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(state, next_state))
