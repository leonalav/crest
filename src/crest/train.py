from __future__ import annotations

import math

import torch
from torch.optim import AdamW

from .config import CRESTConfig, TrainingConfig
from .losses import gate_target_loss, lm_loss
from .model import CRESTModel
from .state import detach_state


def build_optimizer(model: CRESTModel, cfg: TrainingConfig) -> AdamW:
    """AdamW optimizer with common no-decay parameter grouping.

    Citation: AdamW, arXiv:1711.05101, establishes decoupled weight decay for
    adaptive optimizers; see docs/suite/1711.05101v3 lines 5-18 and 101-105.
    Gate/norm/bias exclusions are CREST heuristics from cautions.md, not paper claims.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name or "gate" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.learning_rate)


def cosine_warmup_lr(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * float(step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (1.0 + math.cos(math.pi * progress))


def train_episode_batch(model: CRESTModel, batch, optimizer: AdamW, cfg: TrainingConfig) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    b, t, _, = batch.input_ids.shape
    state = model.init_state(b, device=batch.input_ids.device, dtype=next(model.parameters()).dtype)
    total_loss = None
    last_aux = None
    chunks = 0
    for start in range(0, t, cfg.tbptt_k):
        chunk_loss = None
        for offset in range(start, min(t, start + cfg.tbptt_k)):
            logits, state, aux = model(batch.input_ids[:, offset], state=state, step_idx=batch.step_idx[:, offset])
            loss = lm_loss(logits, batch.labels[:, offset]) + gate_target_loss(aux, cfg.gate_regularization_weight, cfg.gate_target)
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            last_aux = aux
        assert chunk_loss is not None
        (chunk_loss / cfg.tbptt_k).backward()
        state = detach_state(state)
        total_loss = chunk_loss.detach() if total_loss is None else total_loss + chunk_loss.detach()
        chunks += 1
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    optimizer.step()
    return {"loss": float((total_loss / max(1, chunks)).item()), "gate_mean": float(last_aux.gate_mean.item()) if last_aux is not None else 0.0}


def make_model(cfg: CRESTConfig) -> CRESTModel:
    return CRESTModel(cfg)
