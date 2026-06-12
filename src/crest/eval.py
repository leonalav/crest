from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from .model import CRESTModel


@torch.no_grad()
def evaluate(model: CRESTModel, loader: DataLoader, device: torch.device | str = "cpu", max_batches: int | None = None, micro_batch_size: int = 0) -> dict[str, float]:
    model.eval()
    # Enable attention diagnostics for the duration of evaluation so that read
    # and write attention probabilities are produced exactly once on the
    # forward path. The previous implementation always recomputed a second
    # attention matmul outside of SDPA, which doubled eval attention cost.
    prev_diag = getattr(model, "_diagnostics_enabled", False)
    if hasattr(model, "set_diagnostics_enabled"):
        model.set_diagnostics_enabled(True)
    device = torch.device(device)
    total_loss_weighted = 0.0
    total_valid_tokens = 0
    total_step_records = 0
    gate_sum = 0.0
    read_entropy_sum = 0.0
    write_entropy_sum = 0.0
    correct = 0
    total = 0
    boundary_correct = 0
    boundary_total = 0
    boundary_loss_weighted = 0.0
    boundary_valid_tokens = 0
    early_correct = 0
    early_total = 0
    mid_correct = 0
    mid_total = 0
    late_correct = 0
    late_total = 0
    try:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            step_idx = batch.step_idx.to(device)
            mb_size = micro_batch_size or input_ids.size(0)
            for mb_start in range(0, input_ids.size(0), mb_size):
                mb_end = min(input_ids.size(0), mb_start + mb_size)
                mb_inputs = input_ids[mb_start:mb_end]
                mb_labels = labels[mb_start:mb_end]
                mb_steps = step_idx[mb_start:mb_end]
                state = model.init_state(mb_inputs.size(0), device=device, dtype=next(model.parameters()).dtype)
                for t in range(mb_inputs.size(1)):
                    # return_logits=False avoids materializing the [B, L, V]
                    # tensor; head_eval computes exact per-target log-probs and
                    # argmax predictions through the configured head.
                    _, state, aux = model(mb_inputs[:, t], state=state, step_idx=mb_steps[:, t], return_logits=False)
                    step_labels = mb_labels[:, t]
                    valid = step_labels != -100
                    num_valid = int(valid.sum().item())
                    token_lp, pred = model.head_eval(aux.hidden, step_labels)
                    # Token-weighted loss: -log p summed over valid positions,
                    # divided at the end by the total valid-token count, gives
                    # the exact corpus-level mean NLL. token_lp is exactly 0 at
                    # ignored positions, so the masked sum is just the sum.
                    if num_valid > 0:
                        total_loss_weighted += float((-token_lp[valid]).sum().item())
                        total_valid_tokens += num_valid
                    total_step_records += 1
                    gate_sum += float(aux.gate_mean.item())
                    read_entropy_sum += float(aux.state_read_entropy.item())
                    write_entropy_sum += float(aux.write_entropy.item())
                    if num_valid > 0:
                        hits = pred[valid] == step_labels[valid]
                        correct += int(hits.sum().item())
                        total += num_valid
                        progress = t / max(1, mb_inputs.size(1) - 1)
                        if progress < 1 / 3:
                            early_correct += int(hits.sum().item())
                            early_total += num_valid
                        elif progress < 2 / 3:
                            mid_correct += int(hits.sum().item())
                            mid_total += num_valid
                        else:
                            late_correct += int(hits.sum().item())
                            late_total += num_valid
                    if t < mb_inputs.size(1) - 1:
                        boundary_labels = step_labels[:, -1]
                        boundary_valid_mask = boundary_labels != -100
                        b_valid = int(boundary_valid_mask.sum().item())
                        if b_valid > 0:
                            boundary_loss_weighted += float((-token_lp[:, -1][boundary_valid_mask]).sum().item())
                            boundary_valid_tokens += b_valid
                            boundary_correct += int((pred[:, -1][boundary_valid_mask] == boundary_labels[boundary_valid_mask]).sum().item())
                            boundary_total += b_valid
    finally:
        if hasattr(model, "set_diagnostics_enabled"):
            model.set_diagnostics_enabled(prev_diag)
    mean_loss = total_loss_weighted / max(1, total_valid_tokens)
    return {
        "eval_loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "recall_accuracy": correct / max(1, total),
        "boundary_loss": boundary_loss_weighted / max(1, boundary_valid_tokens),
        "boundary_accuracy": boundary_correct / max(1, boundary_total),
        "early_recall_accuracy": early_correct / max(1, early_total),
        "mid_recall_accuracy": mid_correct / max(1, mid_total),
        "late_recall_accuracy": late_correct / max(1, late_total),
        "gate_mean": gate_sum / max(1, total_step_records),
        "state_read_entropy": read_entropy_sum / max(1, total_step_records),
        "write_entropy": write_entropy_sum / max(1, total_step_records),
    }
