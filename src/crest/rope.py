from __future__ import annotations

import torch


def rope_frequencies(head_dim: int, seq_len: int, base: float, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin tables for RoPE.

    Citation: RoFormer, arXiv:2104.09864, derives the rotary matrix and the
    relative inner-product identity; see docs/suite/2104.09864v5 lines 148-183
    and 337-358. CREST v0 applies this only to intra-step local Q/K tokens.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [B, H, L, D]."""
    seq_len = x.size(-2)
    cos = cos[:seq_len].to(device=x.device, dtype=x.dtype).view(1, 1, seq_len, -1)
    sin = sin[:seq_len].to(device=x.device, dtype=x.dtype).view(1, 1, seq_len, -1)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
