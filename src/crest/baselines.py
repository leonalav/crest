from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import CRESTConfig
from .layers import RMSNorm, SwiGLU


class FullAttentionBaseline(nn.Module):
    """Dense full-episode causal Transformer baseline.

    Citation: Attention Is All You Need, arXiv:1706.03762, supplies dense causal
    decoder self-attention. This baseline intentionally scales as O((T*L)^2 d)
    and is used only for small ablations against fixed-state CREST.
    """

    def __init__(self, cfg: CRESTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ffn,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_steps * cfg.max_seq_len, cfg.d_model)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embedding.weight

    def forward(self, episode_input_ids: torch.Tensor) -> torch.Tensor:
        b, t, l = episode_input_ids.shape
        flat = episode_input_ids.reshape(b, t * l)
        pos = torch.arange(t * l, device=episode_input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.embedding(flat) + self.position_embedding(pos)
        n = x.size(1)
        mask = torch.ones(n, n, device=x.device, dtype=torch.bool).triu(1)
        x = self.layers(x, mask=mask)
        return self.head(self.norm(x)).view(b, t, l, self.cfg.vocab_size)


class LocalOnlyCRESTConfig:
    """Marker class retained for ablation reports; use ablation.local_only_config."""
