from __future__ import annotations

import argparse
import copy
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

import yaml

from .ablation import memory_sweep_configs, no_state_config
from .baselines import FullAttentionBaseline
from .config import CRESTConfig, DataConfig, TrainingConfig
from .data import build_dataset, collate_episodes
from .eval import evaluate
from .losses import lm_loss
from .train import build_optimizer, cosine_warmup_lr, run_training, set_lr

T = TypeVar("T")


def load_dataclass(cls: type[T], path: str) -> T:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def run_full_attention_baseline(model_cfg: CRESTConfig, data_cfg: DataConfig, train_cfg: TrainingConfig) -> dict:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullAttentionBaseline(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    train_loader = DataLoader(build_dataset(data_cfg, "train"), batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_episodes, drop_last=True)
    eval_loader = DataLoader(build_dataset(data_cfg, "eval"), batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_episodes)
    iterator = iter(train_loader)
    for step in range(train_cfg.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = type(batch)(input_ids=batch.input_ids.to(device), labels=batch.labels.to(device), step_idx=batch.step_idx.to(device))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids)
        loss = lm_loss(logits.reshape(batch.input_ids.size(0), -1, logits.size(-1)), batch.labels.reshape(batch.labels.size(0), -1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        set_lr(optimizer, cosine_warmup_lr(step, train_cfg))
        optimizer.step()
    # Full baseline eval focuses on token metrics; recurrent-specific metrics are not applicable.
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_batches = 0
        correct = 0
        total = 0
        for batch in eval_loader:
            batch = type(batch)(input_ids=batch.input_ids.to(device), labels=batch.labels.to(device), step_idx=batch.step_idx.to(device))
            logits = model(batch.input_ids)
            loss = lm_loss(logits.reshape(batch.input_ids.size(0), -1, logits.size(-1)), batch.labels.reshape(batch.labels.size(0), -1))
            total_loss += float(loss.item())
            total_batches += 1
            pred = logits.argmax(dim=-1)
            valid = batch.labels != -100
            correct += int((pred[valid] == batch.labels[valid]).sum().item())
            total += int(valid.sum().item())
    return {"eval_loss": total_loss / max(1, total_batches), "recall_accuracy": correct / max(1, total), "model_type": "full_attention"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CREST ablations")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--memory-sweep", default="4,8,16,32")
    parser.add_argument("--out", default="runs/ablations/results.jsonl")
    args = parser.parse_args()

    model_cfg = load_dataclass(CRESTConfig, args.model)
    data_cfg = load_dataclass(DataConfig, args.data)
    train_cfg = load_dataclass(TrainingConfig, args.training)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, CRESTConfig]] = [("crest", model_cfg), ("no_state", no_state_config(model_cfg))]
    for name, cfg in memory_sweep_configs(model_cfg, [int(x) for x in args.memory_sweep.split(",") if x]).items():
        variants.append((name, cfg))

    with out_path.open("w", encoding="utf-8") as f:
        for name, cfg in variants:
            tcfg = replace(train_cfg, output_dir=str(Path(train_cfg.output_dir) / name), run_name=f"{train_cfg.run_name}_{name}")
            result = run_training(cfg, data_cfg, tcfg)
            row = {"variant": name, "model_type": "crest", "memory_slots": cfg.memory_slots, "parameters": result["parameters"], **result.get("last_eval", {})}
            f.write(json.dumps(row) + "\n")
        baseline = run_full_attention_baseline(model_cfg, data_cfg, replace(train_cfg, max_steps=min(train_cfg.max_steps, 200)))
        f.write(json.dumps(baseline) + "\n")
    print(f"[ablate] wrote results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
