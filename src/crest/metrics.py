from __future__ import annotations

from .config import CRESTConfig
from .model import CRESTModel


def count_parameters(model: CRESTModel) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_step_flops(cfg: CRESTConfig) -> dict[str, int]:
    l, m, d = cfg.max_seq_len, cfg.memory_slots, cfg.d_model
    local = 2 * l * l * d
    state_read = 2 * l * m * d
    state_write = 2 * m * l * d
    ffn = 6 * l * d * cfg.d_ffn
    gate = 6 * m * d * d
    total = local + state_read + state_write + ffn + gate
    return {"local_attention": local, "state_read": state_read, "state_write": state_write, "ffn": ffn, "gate_mlp": gate, "total": total}
