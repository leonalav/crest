from __future__ import annotations

from .config import CRESTConfig
from .model import CRESTModel


def count_parameters(model: CRESTModel) -> int:
    return sum(p.numel() for p in model.parameters())


def component_parameter_counts(model: CRESTModel) -> dict[str, int]:
    groups = {"embedding": 0, "local_attention": 0, "state_read": 0, "state_write": 0, "gate_mlp": 0, "ffn": 0, "norm": 0, "lm_head": 0, "other": 0}
    for name, param in model.named_parameters():
        n = param.numel()
        if "token_embedding" in name:
            groups["embedding"] += n
        elif "local_attn" in name:
            groups["local_attention"] += n
        elif "state_read" in name:
            groups["state_read"] += n
        elif ".write.gate" in name:
            groups["gate_mlp"] += n
        elif ".write" in name:
            groups["state_write"] += n
        elif "ffn" in name:
            groups["ffn"] += n
        elif "norm" in name:
            groups["norm"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        else:
            groups["other"] += n
    return groups


def estimate_step_flops(cfg: CRESTConfig) -> dict[str, int]:
    """Estimate forward FLOPs for one episode step across all CREST layers.

    The CREST plan accounts local attention, state read/write, FFN, and gate MLP
    per layer, so every layer-internal term is multiplied by cfg.n_layers. The
    LM head is applied once after the final layer.
    """
    l, m, d = cfg.max_seq_len, cfg.memory_slots, cfg.d_model
    per_layer_local = 2 * l * l * d
    per_layer_state_read = 2 * l * m * d
    per_layer_state_write = 2 * m * l * d
    per_layer_ffn = 6 * l * d * cfg.d_ffn
    per_layer_gate = 6 * m * d * d
    local = cfg.n_layers * per_layer_local
    state_read = cfg.n_layers * per_layer_state_read
    state_write = cfg.n_layers * per_layer_state_write
    ffn = cfg.n_layers * per_layer_ffn
    gate = cfg.n_layers * per_layer_gate
    logits = 2 * l * d * cfg.vocab_size
    total = local + state_read + state_write + ffn + gate + logits
    return {"local_attention": local, "state_read": state_read, "state_write": state_write, "ffn": ffn, "gate_mlp": gate, "lm_head": logits, "total": total}


def estimate_episode_flops(cfg: CRESTConfig, episode_steps: int, backward: bool = False) -> dict[str, int]:
    step = estimate_step_flops(cfg)
    multiplier = episode_steps * (3 if backward else 1)
    return {k: v * multiplier for k, v in step.items()}
