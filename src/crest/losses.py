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


def adaptive_lm_head_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    adaptive_head: torch.nn.Module,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Exact cluster-factored cross-entropy via nn.AdaptiveLogSoftmaxWithLoss.

    Citation: Grave et al., arXiv:1609.04309. The factorization
        p(y|h) = p_head(y|h)                        for y in the head cluster
        p(y|h) = p_head(c_i|h) * p_tail_i(y|P_i h)  for y in tail cluster i
    is exactly normalized: sum_y p(y|h) = sum_head p_head + sum_i p_head(c_i)*1 = 1.
    The returned value is therefore a true mean NLL, directly comparable with
    full-softmax eval losses.

    PyTorch's module has NO ignore_index handling — its loss is the plain mean
    of -output over every row passed in — so invalid (-100) positions must be
    filtered out HERE, mirroring chunked_lm_head_loss. Labels must already be
    in frequency-rank space (CRESTModel._map_labels does this).
    """
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    if not torch.any(valid):
        # Empty supervised set: the mathematically correct contribution is the
        # empty sum (zero), kept on-graph for autograd/DDP parity. Touch the
        # head parameters so DDP does not flag them as unused on this rank.
        # Guard the degenerate 0-row case: mean over an empty batch is NaN,
        # and NaN * 0.0 = NaN would poison the loss.
        if flat_hidden.size(0) == 0:
            return flat_hidden.sum() * 0.0
        dummy = adaptive_head(flat_hidden[:1], flat_labels[:1].clamp_min(0)).loss
        return dummy * 0.0
    return adaptive_head(flat_hidden[valid], flat_labels[valid]).loss


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
