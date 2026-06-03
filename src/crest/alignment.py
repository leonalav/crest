from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import dpo_loss, lm_loss
from .model import CRESTModel


def trajectory_logprob(model: CRESTModel, input_ids: torch.Tensor, labels: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
    """Return per-episode log probability under separate recurrent rollout.

    Citation: DPO, arXiv:2305.18290, optimizes log-probability ratios. CREST
    extension required for recurrent models: policy and reference must roll their own states
    on identical prefixes; this helper enforces that separation by taking one model.
    """
    b = input_ids.size(0)
    state = model.init_state(b, device=input_ids.device, dtype=next(model.parameters()).dtype)
    totals = torch.zeros(b, device=input_ids.device)
    for t in range(input_ids.size(1)):
        logits, state, _ = model(input_ids[:, t], state=state, step_idx=step_idx[:, t])
        log_probs = F.log_softmax(logits, dim=-1)
        valid = labels[:, t] != -100
        safe_labels = labels[:, t].clamp_min(0)
        token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        totals = totals + (token_lp * valid).sum(dim=-1)
    return totals


def sft_loss_for_batch(model: CRESTModel, batch) -> torch.Tensor:
    b = batch.input_ids.size(0)
    state = model.init_state(b, device=batch.input_ids.device, dtype=next(model.parameters()).dtype)
    total = None
    for t in range(batch.input_ids.size(1)):
        logits, state, _ = model(batch.input_ids[:, t], state=state, step_idx=batch.step_idx[:, t])
        loss = lm_loss(logits, batch.labels[:, t])
        total = loss if total is None else total + loss
    assert total is not None
    return total / batch.input_ids.size(1)


def dpo_loss_for_batch(policy: CRESTModel, reference: CRESTModel, batch, beta: float = 0.1) -> torch.Tensor:
    policy_w = trajectory_logprob(policy, batch.winner_input_ids, batch.winner_labels, batch.winner_step_idx)
    policy_l = trajectory_logprob(policy, batch.loser_input_ids, batch.loser_labels, batch.loser_step_idx)
    with torch.no_grad():
        ref_w = trajectory_logprob(reference, batch.winner_input_ids, batch.winner_labels, batch.winner_step_idx)
        ref_l = trajectory_logprob(reference, batch.loser_input_ids, batch.loser_labels, batch.loser_step_idx)
    return dpo_loss(policy_w, policy_l, ref_w, ref_l, beta=beta)
