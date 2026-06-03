from __future__ import annotations

import torch


def init_state(n_layers: int, batch_size: int, memory_slots: int, d_model: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
    return [torch.zeros(batch_size, memory_slots, d_model, device=device, dtype=dtype) for _ in range(n_layers)]


def detach_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [s.detach() for s in state]


def state_metrics(state: list[torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        stacked = torch.stack([s.float().norm(dim=-1).mean() for s in state])
        return {"state_norm_mean": float(stacked.mean().item()), "state_norm_max": float(stacked.max().item())}
