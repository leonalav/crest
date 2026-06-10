from __future__ import annotations

import torch
from torch.nn import functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import CRESTAux


def lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Autoregressive token loss: -mean log p(target | prefix, recurrent state).

    If a micro-step has no supervised target tokens, the mathematically correct
    contribution is the empty sum, i.e. zero. PyTorch's mean-reduced
    cross_entropy returns NaN in that case because it divides by zero valid
    targets, so handle the empty set explicitly while preserving autograd.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits[valid], flat_labels[valid])


def chunked_lm_head_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    chunk_size: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy that only materializes lm_head logits for token chunks."""
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    if not torch.any(valid):
        return flat_hidden.sum() * 0.0

    valid_hidden = flat_hidden[valid]
    valid_labels = flat_labels[valid]
    if chunk_size <= 0 or valid_hidden.size(0) <= chunk_size:
        return F.cross_entropy(lm_head(valid_hidden), valid_labels)

    loss_sum = valid_hidden.new_zeros(())
    for start in range(0, valid_hidden.size(0), chunk_size):
        end = min(start + chunk_size, valid_hidden.size(0))
        logits = lm_head(valid_hidden[start:end])
        loss_sum = loss_sum + F.cross_entropy(logits, valid_labels[start:end], reduction="sum")
    return loss_sum / valid_hidden.size(0)


def gate_target_loss(aux: "CRESTAux", weight: float = 0.0, target: float = 0.5) -> torch.Tensor:
    if weight == 0.0:
        return aux.gate_mean.new_zeros(())
    return weight * (aux.gate_mean - target).pow(2)


def dpo_loss(policy_w: torch.Tensor, policy_l: torch.Tensor, ref_w: torch.Tensor, ref_l: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """DPO preference loss over trajectory log-probabilities.

    Citation: DPO, arXiv:2305.18290, derives preference optimization from
    Bradley-Terry plus KL-regularized RL; see docs/suite/2305.18290v3 lines
    45-55, 59-79, and 88-101. For CREST, callers must compute policy/ref
    log-probs with separate recurrent state rollouts on identical prefixes.
    """
    logits = beta * ((policy_w - ref_w) - (policy_l - ref_l))
    return -F.logsigmoid(logits).mean()
