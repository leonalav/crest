from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

import yaml

from .ablation import memory_sweep_configs, no_state_config
from .config import CRESTConfig, DataConfig, TrainingConfig
from .distributed import init_distributed
from .train import run_training

T = TypeVar("T")


def load_dataclass(cls: type[T], path: str) -> T:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CREST training variant")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--variant", default="crest", help="crest, no_state, or M<int>")
    parser.add_argument("--streaming", action="store_true", help="Ignore prepared shards and tokenize the data manifest online with HF streaming=True")
    parser.add_argument("--raw-text", action="store_true", help="Train from bounded raw JSONL files referenced by the data config")
    args = parser.parse_args()
    if args.streaming and args.raw_text:
        raise ValueError("Use either --streaming or --raw-text, not both")

    model_cfg = load_dataclass(CRESTConfig, args.model)
    data_cfg = load_dataclass(DataConfig, args.data)
    if args.streaming:
        data_cfg = replace(data_cfg, task="streaming_text", path=args.data)
    if args.raw_text:
        data_cfg = replace(data_cfg, task="raw_text")
    train_cfg = load_dataclass(TrainingConfig, args.training)
    variant = args.variant
    if variant == "crest":
        cfg = model_cfg
    elif variant == "no_state":
        cfg = no_state_config(model_cfg)
    elif variant.startswith("M") and variant[1:].isdigit():
        cfg = memory_sweep_configs(model_cfg, [int(variant[1:])])[variant]
    else:
        raise ValueError("variant must be crest, no_state, or M<int>")

    dist = init_distributed()
    tcfg = replace(train_cfg, output_dir=str(Path(train_cfg.output_dir) / variant), run_name=f"{train_cfg.run_name}_{variant}")
    run_training(cfg, data_cfg, tcfg, distributed=dist)


if __name__ == "__main__":
    main()
