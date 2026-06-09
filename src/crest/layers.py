from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention_policy import AttentionBackendPolicy
from .config import CRESTConfig
from .rope import apply_rope, rope_frequencies


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Citation: RMSNorm, arXiv:1910.07467, defines RMSNorm(a)_i = a_i/RMS(a)*g_i;
    see docs/suite/1910.07467v1 lines 64-68 and 84-87.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Citation: GLU variants improve Transformer, arXiv:2002.05202, defines SwiGLU
    Transformer FFNs; see docs/suite/2002.05202v1 lines 42-71.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


@dataclass
class LayerAux:
    gate: torch.Tensor
    state_read_entropy: torch.Tensor
    write_entropy: torch.Tensor


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with optional causal masking.

    Citation: Attention Is All You Need, arXiv:1706.03762, defines
    softmax(QK^T/sqrt(d_k))V and causal decoder masking; see docs/suite/1706.03762v7
    lines 83-93 and 111-118.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, backend: str = "auto") -> None:
        super().__init__()
        self.n_heads = n_heads
        self.backend_policy = AttentionBackendPolicy(backend)
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, *, causal: bool = False, rope: bool = False, rope_base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._split(self.q(query))
        k = self._split(self.k(key_value))
        v = self._split(self.v(key_value))
        if rope:
            cos, sin = rope_frequencies(self.head_dim, q.size(-2), rope_base, q.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        # Use is_causal=True (FlashAttention-eligible) when query and key lengths match
        # (i.e. local self-attention). Only fall back to an explicit float mask when
        # causal=True but lq != lk (cross-attention with causal masking).
        use_causal_kernel = causal and q.size(-2) == k.size(-2)
        bool_mask = None
        if causal and not use_causal_kernel:
            lq, lk = q.size(-2), k.size(-2)
            bool_mask = torch.ones(lq, lk, device=q.device, dtype=torch.bool).triu(1)
            mask = torch.zeros(lq, lk, device=q.device, dtype=q.dtype).masked_fill(bool_mask, float("-inf"))
        else:
            mask = None
        with self.backend_policy.context():
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=use_causal_kernel)
        probs = None
        if not self.training:
            # Diagnostic-only: compute attention probabilities for entropy logging.
            # Runs under no_grad to avoid building an autograd graph and to avoid
            # wasting GPU memory on a second set of attention logits during training.
            with torch.no_grad():
                logits = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) / math.sqrt(self.head_dim)
                if causal:
                    # Reuse bool_mask if already computed; otherwise build it for
                    # the equal-length (use_causal_kernel) case.
                    if bool_mask is None:
                        lq, lk = q.size(-2), k.size(-2)
                        bool_mask = torch.ones(lq, lk, device=q.device, dtype=torch.bool).triu(1)
                    logits = logits.masked_fill(bool_mask, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
        y = y.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        return self.o(y), probs


class StateWriter(nn.Module):
    """GRU-like recurrent state update for CREST.

    Citation: GRU, arXiv:1406.1078, uses h_t = z_t*h_{t-1} + (1-z_t)*h~_t;
    see docs/suite/1406.1078v3 lines 96-129. Pascanu et al., arXiv:1211.5063,
    support recurrent Jacobian-product analysis and gradient clipping; see lines
    53-101 and 194-206.
    """

    def __init__(self, cfg: CRESTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.backend_policy = AttentionBackendPolicy(cfg.attention_backend)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.step_embed = nn.Embedding(cfg.max_steps, cfg.d_model)
        # Learned slot identities break permutation symmetry between zero-initialized
        # memory slots. They are used for write addressing/gating only, so step-0
        # state reads from a zero state remain exactly zero as required by tests.
        self.slot_embed = nn.Embedding(cfg.memory_slots, cfg.d_model)
        self.gate = nn.Sequential(nn.Linear(cfg.d_model * 4, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.write_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        nn.init.constant_(self.gate[-1].bias, cfg.gate_retention_bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, state: torch.Tensor, tokens: torch.Tensor, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        slot_ids = torch.arange(state.size(1), device=state.device)
        slot = self.slot_embed(slot_ids).unsqueeze(0).expand_as(state)
        q = self._split(self.q(state + slot))
        k = self._split(self.k(tokens))
        v = self._split(self.v(tokens))
        with self.backend_policy.context():
            update = F.scaled_dot_product_attention(q, k, v)
        probs = None
        if not self.training:
            # Diagnostic-only: compute write attention probabilities for entropy
            # logging. Wrapped in no_grad to avoid a duplicate autograd graph and
            # wasted memory bandwidth from recomputing the attention matrix.
            with torch.no_grad():
                logits = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) / math.sqrt(self.head_dim)
                probs = torch.softmax(logits, dim=-1)
        update = update.transpose(1, 2).contiguous().view(state.size(0), state.size(1), -1)
        step = self.step_embed(step_idx.clamp_min(0).clamp_max(self.step_embed.num_embeddings - 1)).unsqueeze(1).expand_as(state)
        retention = torch.sigmoid(self.gate(torch.cat([state, update, step, slot], dim=-1)))
        next_state = retention * state + (1.0 - retention) * self.write_norm(update)
        return next_state, retention, probs


class CRESTLayer(nn.Module):
    def __init__(self, cfg: CRESTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.x_norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.s_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.local_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.attention_backend)
        self.state_read = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.attention_backend)
        self.fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.x_norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn, cfg.dropout)
        self.write = StateWriter(cfg)

    def forward(self, x: torch.Tensor, state: torch.Tensor, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, LayerAux]:
        x_norm = self.x_norm1(x)
        s_norm = self.s_norm(state)
        local, _ = self.local_attn(x_norm, x_norm, causal=True, rope=self.cfg.use_local_rope, rope_base=self.cfg.rope_base)
        if self.cfg.use_state_read:
            read, read_probs = self.state_read(x_norm, s_norm, causal=False, rope=False)
        else:
            read = torch.zeros_like(local)
            read_probs = None if self.training else torch.full((x.size(0), self.cfg.n_heads, x.size(1), state.size(1)), 1.0 / state.size(1), device=x.device, dtype=x.dtype)
        fuse_gate = torch.sigmoid(self.fuse(torch.cat([local, read], dim=-1)))
        x = x + fuse_gate * local + (1.0 - fuse_gate) * read
        x = x + self.ffn(self.x_norm2(x))
        if self.cfg.use_state_write:
            next_state, gate, write_probs = self.write(s_norm, x, step_idx)
        else:
            next_state = state
            gate = torch.ones_like(state)
            write_probs = None if self.training else torch.full((x.size(0), self.cfg.n_heads, state.size(1), x.size(1)), 1.0 / x.size(1), device=x.device, dtype=x.dtype)
        if read_probs is None:
            read_entropy = x.new_zeros(())
        else:
            read_probs = read_probs.clamp_min(1e-9)
            read_entropy = -(read_probs * read_probs.log()).sum(dim=-1).mean()
        if write_probs is None:
            write_entropy = x.new_zeros(())
        else:
            write_probs = write_probs.clamp_min(1e-9)
            write_entropy = -(write_probs * write_probs.log()).sum(dim=-1).mean()
        return x, next_state, LayerAux(gate=gate, state_read_entropy=read_entropy, write_entropy=write_entropy)
