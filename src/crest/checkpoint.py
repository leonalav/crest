from __future__ import annotations

from pathlib import Path

import torch

from .model import CRESTModel


def save_checkpoint(path: str, model: CRESTModel, optimizer: torch.optim.Optimizer, step: int, extra: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "extra": extra or {}}, target)


def load_checkpoint(path: str, model: CRESTModel, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    # Full-head and adaptive-head checkpoints are structurally incompatible
    # (lm_head.weight vs lm_head.head.weight/lm_head.tail.*), and a checkpoint
    # trained under one token permutation is meaningless under another. Fail
    # loudly instead of letting strict=True produce a confusing key dump.
    ckpt_keys = set(ckpt["model"].keys())
    ckpt_adaptive = any(k.startswith("lm_head.tail") or k.startswith("lm_head.head") for k in ckpt_keys)
    model_adaptive = model.cfg.head_type == "adaptive"
    if ckpt_adaptive != model_adaptive:
        raise ValueError(
            f"checkpoint head_type ({'adaptive' if ckpt_adaptive else 'full'}) does not match "
            f"model head_type ({model.cfg.head_type}); adaptive and full-head runs cannot resume each other"
        )
    if ("token_perm" in ckpt_keys) != (model.token_perm is not None):
        raise ValueError(
            "checkpoint and model disagree on token permutation presence; "
            "a run must keep the same token_perm_path for its whole lifetime"
        )
    # Compare BEFORE load_state_dict overwrites the model's buffer, otherwise
    # the check trivially passes against the just-loaded values.
    if model.token_perm is not None and "token_perm" in ckpt_keys:
        if not torch.equal(model.token_perm.cpu(), ckpt["model"]["token_perm"].cpu()):
            raise ValueError("checkpoint token permutation differs from the model's; refusing silent resume")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))
