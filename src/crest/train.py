from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .auxiliary import StateReconstructionHead
from .checkpoint import load_checkpoint, save_checkpoint
from .config import CRESTConfig, DataConfig, TrainingConfig
from .data import build_dataset, collate_episodes
from .distributed import DistributedInfo
from .eval import evaluate
from .logging_utils import JsonlLogger
from .losses import gate_target_loss, lm_loss
from .metrics import component_parameter_counts, count_parameters, estimate_episode_flops
from .model import CRESTModel
from .precision import autocast_context, make_grad_scaler
from .state import detach_state


def build_optimizer(model: CRESTModel, cfg: TrainingConfig) -> AdamW:
    """AdamW optimizer with common no-decay parameter grouping.

    Citation: AdamW, arXiv:1711.05101, establishes decoupled weight decay for
    adaptive optimizers; see docs/suite/1711.05101v3 lines 5-18 and 101-105.
    Gate/norm/bias exclusions are CREST implementation heuristics, not AdamW paper claims.
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


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def run_training(
    model_cfg: CRESTConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    *,
    device: str | torch.device | None = None,
    distributed: DistributedInfo | None = None,
) -> dict[str, Any]:
    """Full 125M-capable CREST training harness.

    Implements CREST requirements: explicit recurrent state, truncated BPTT,
    AdamW, gradient clipping, checkpoint/resume, eval loop, JSONL logging,
    parameter/FLOP startup report, and mixed-precision policy. FSDP wrapping is
    handled by the caller so this function also works in unit tests.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    distributed = distributed or DistributedInfo(enabled=False)
    torch.manual_seed(train_cfg.seed + distributed.rank)
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CRESTModel(model_cfg).to(device)
    aux_head = StateReconstructionHead(model_cfg.d_model, train_cfg.aux_state_dim).to(device) if train_cfg.aux_state_weight > 0 else None
    if aux_head is not None:
        model.aux_state_head = aux_head
    optimizer = build_optimizer(model, train_cfg)
    start_step = 0
    if train_cfg.resume_from:
        start_step = load_checkpoint(train_cfg.resume_from, model, optimizer, map_location=device)

    train_ds = build_dataset(data_cfg, "train")
    eval_ds = build_dataset(data_cfg, "eval")
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers, collate_fn=collate_episodes, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers, collate_fn=collate_episodes)

    logger = JsonlLogger(train_cfg.output_dir, train_cfg.run_name) if distributed.is_main else None
    report = {
        "parameters": count_parameters(model),
        "component_parameters": component_parameter_counts(model),
        "episode_flops_forward": estimate_episode_flops(model_cfg, data_cfg.episode_steps),
        "model_config": asdict(model_cfg),
        "data_config": asdict(data_cfg),
        "training_config": asdict(train_cfg),
    }
    if logger:
        logger.log(start_step, {"event": "startup", **report})
    if distributed.is_main:
        print(
            f"[startup] run={train_cfg.run_name} device={device} precision={train_cfg.precision} "
            f"params={report['parameters']:,} train_episodes={len(train_ds)} eval_episodes={len(eval_ds)} "
            f"max_steps={train_cfg.max_steps} output_dir={train_cfg.output_dir}",
            flush=True,
        )

    scaler = make_grad_scaler(train_cfg.precision, device)
    step = start_step
    iterator = iter(train_loader)
    while step < train_cfg.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = type(batch)(input_ids=batch.input_ids.to(device), labels=batch.labels.to(device), step_idx=batch.step_idx.to(device))
        lr = cosine_warmup_lr(step, train_cfg)
        set_lr(optimizer, lr)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        b, t, _ = batch.input_ids.shape
        state = model.init_state(b, device=device, dtype=next(model.parameters()).dtype)
        total_loss = None
        last_aux = None
        with autocast_context(train_cfg.precision, device):
            for start in range(0, t, train_cfg.tbptt_k):
                chunk_loss = None
                denom = min(train_cfg.tbptt_k, t - start)
                for offset in range(start, start + denom):
                    logits, state, aux = model(batch.input_ids[:, offset], state=state, step_idx=batch.step_idx[:, offset])
                    loss = lm_loss(logits, batch.labels[:, offset]) + gate_target_loss(aux, train_cfg.gate_regularization_weight, train_cfg.gate_target)
                    if aux_head is not None:
                        loss = loss + train_cfg.aux_state_weight * aux_head(aux.final_state[-1], aux.hidden)
                    chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                    last_aux = aux
                assert chunk_loss is not None
                total_loss = chunk_loss.detach() if total_loss is None else total_loss + chunk_loss.detach()
                scaler.scale(chunk_loss / denom).backward()
                state = detach_state(state)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % train_cfg.log_every == 0:
            train_metrics = {"event": "train", "loss": float(total_loss.item() / max(1, t)), "loss_sum": float(total_loss.item()), "lr": lr, "grad_norm": float(grad_norm), "gate_mean": float(last_aux.gate_mean.item()) if last_aux else 0.0}
            if logger:
                logger.log(step, train_metrics)
            if distributed.is_main:
                print(
                    f"[train] step={step}/{train_cfg.max_steps} loss={train_metrics['loss']:.4f} "
                    f"lr={lr:.3e} grad_norm={train_metrics['grad_norm']:.3f} gate_mean={train_metrics['gate_mean']:.3f}",
                    flush=True,
                )
        if step > 0 and step % train_cfg.eval_every == 0:
            metrics = evaluate(model, eval_loader, device=device, max_batches=8)
            if logger:
                logger.log(step, {"event": "eval", **metrics})
            if distributed.is_main:
                print(
                    f"[eval] step={step} loss={metrics['eval_loss']:.4f} ppl={metrics['perplexity']:.3f} "
                    f"recall={metrics['recall_accuracy']:.3f} gate={metrics['gate_mean']:.3f} "
                    f"read_H={metrics['state_read_entropy']:.3f} write_H={metrics['write_entropy']:.3f}",
                    flush=True,
                )
        if distributed.is_main and step > 0 and step % train_cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{step}.pt"
            save_checkpoint(str(ckpt_path), model, optimizer, step, extra=report)
            print(f"[checkpoint] step={step} path={ckpt_path}", flush=True)
        step += 1

    if distributed.is_main:
        final_path = output_dir / "checkpoint_final.pt"
        save_checkpoint(str(final_path), model, optimizer, step, extra=report)
        print(f"[done] step={step} final_checkpoint={final_path}", flush=True)
    return {"step": step, **report}
