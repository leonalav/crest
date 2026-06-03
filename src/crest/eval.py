from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from .losses import lm_loss
from .model import CRESTModel


@torch.no_grad()
def evaluate(model: CRESTModel, loader: DataLoader, device: torch.device | str = "cpu", max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    total_loss = 0.0
    total_batches = 0
    gate_sum = 0.0
    read_entropy_sum = 0.0
    write_entropy_sum = 0.0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        step_idx = batch.step_idx.to(device)
        state = model.init_state(input_ids.size(0), device=device, dtype=next(model.parameters()).dtype)
        for t in range(input_ids.size(1)):
            logits, state, aux = model(input_ids[:, t], state=state, step_idx=step_idx[:, t])
            loss = lm_loss(logits, labels[:, t])
            total_loss += float(loss.item())
            total_batches += 1
            gate_sum += float(aux.gate_mean.item())
            read_entropy_sum += float(aux.state_read_entropy.item())
            write_entropy_sum += float(aux.write_entropy.item())
            valid = labels[:, t] != -100
            if torch.any(valid):
                pred = logits.argmax(dim=-1)
                correct += int((pred[valid] == labels[:, t][valid]).sum().item())
                total += int(valid.sum().item())
    mean_loss = total_loss / max(1, total_batches)
    return {
        "eval_loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "recall_accuracy": correct / max(1, total),
        "gate_mean": gate_sum / max(1, total_batches),
        "state_read_entropy": read_entropy_sum / max(1, total_batches),
        "write_entropy": write_entropy_sum / max(1, total_batches),
    }
