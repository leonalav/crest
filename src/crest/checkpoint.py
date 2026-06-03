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
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))
