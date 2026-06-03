from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import CRESTConfig
from .layers import CRESTLayer, RMSNorm
from .state import init_state


@dataclass
class CRESTAux:
    gates: list[torch.Tensor]
    state_read_entropy: torch.Tensor
    write_entropy: torch.Tensor
    hidden: torch.Tensor
    final_state: list[torch.Tensor]

    @property
    def gate_mean(self) -> torch.Tensor:
        return torch.stack([g.mean() for g in self.gates]).mean()


class CRESTModel(nn.Module):
    def __init__(self, cfg: CRESTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([CRESTLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def init_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
        return init_state(self.cfg.n_layers, batch_size, self.cfg.memory_slots, self.cfg.d_model, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, state: list[torch.Tensor] | None = None, step_idx: torch.Tensor | int = 0, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor], CRESTAux]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        b = input_ids.size(0)
        if state is None:
            state = self.init_state(b, device=input_ids.device, dtype=self.token_embedding.weight.dtype)
        if len(state) != self.cfg.n_layers:
            raise ValueError("state length must equal n_layers")
        if isinstance(step_idx, int):
            step_idx = torch.full((b,), step_idx, device=input_ids.device, dtype=torch.long)
        elif step_idx.ndim == 0:
            step_idx = step_idx.expand(b).to(device=input_ids.device, dtype=torch.long)
        else:
            step_idx = step_idx.to(device=input_ids.device, dtype=torch.long)
        x = self.token_embedding(input_ids)
        next_state: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        read_entropies = []
        write_entropies = []
        for layer, layer_state in zip(self.layers, state, strict=True):
            x, s_next, aux = layer(x, layer_state, step_idx)
            next_state.append(s_next)
            gates.append(aux.gate)
            read_entropies.append(aux.state_read_entropy)
            write_entropies.append(aux.write_entropy)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        return logits, next_state, CRESTAux(gates=gates, state_read_entropy=torch.stack(read_entropies).mean(), write_entropy=torch.stack(write_entropies).mean(), hidden=hidden, final_state=next_state)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
