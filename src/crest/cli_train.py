from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import TypeVar

import yaml

from .config import CRESTConfig, DataConfig, TrainingConfig
from .distributed import init_distributed, maybe_wrap_fsdp
from .model import CRESTModel
from .train import run_training

T = TypeVar("T")


def load_dataclass(cls: type[T], path: str) -> T:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {f.name for f in fields(cls)}
    data = {k: v for k, v in data.items() if k in allowed}
    return cls(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CREST training")
    parser.add_argument("--model", required=True, help="Path to model YAML")
    parser.add_argument("--data", required=True, help="Path to data YAML")
    parser.add_argument("--training", required=True, help="Path to training YAML")
    parser.add_argument("--streaming", action="store_true", help="Ignore prepared shards and tokenize the data manifest online with HF streaming=True")
    args = parser.parse_args()
    model_cfg = load_dataclass(CRESTConfig, args.model)
    data_cfg = load_dataclass(DataConfig, args.data)
    if args.streaming:
        data_cfg = replace(data_cfg, task="streaming_text", path=args.data)
    train_cfg = load_dataclass(TrainingConfig, args.training)
    dist = init_distributed()
    # FSDP is exposed for launch scripts; run_training constructs a non-wrapped model
    # for clean checkpoint naming. Full external FSDP users can import lower-level APIs.
    run_training(model_cfg, data_cfg, train_cfg, distributed=dist)


if __name__ == "__main__":
    main()
