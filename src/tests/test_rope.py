import torch

from crest.rope import apply_rope, rope_frequencies


def test_rope_norm_preserved():
    x = torch.randn(2, 4, 8, 16)
    cos, sin = rope_frequencies(16, 8, 10000.0, x.device)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_relative_equivariance_one_pair():
    q = torch.randn(2)
    k = torch.randn(2)
    cos, sin = rope_frequencies(2, 8, 10000.0)
    xq = q.view(1, 1, 1, 2).expand(1, 1, 8, 2)
    xk = k.view(1, 1, 1, 2).expand(1, 1, 8, 2)
    rq = apply_rope(xq, cos, sin)[0, 0]
    rk = apply_rope(xk, cos, sin)[0, 0]
    assert torch.allclose((rq[1] * rk[3]).sum(), (rq[2] * rk[4]).sum(), atol=1e-5)
