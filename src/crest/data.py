from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch
from pathlib import Path

from torch.utils.data import Dataset

from .config import DataConfig


@dataclass(frozen=True)
class Episode:
    input_ids: torch.Tensor
    labels: torch.Tensor
    step_idx: torch.Tensor


class SyntheticKeyValueDataset(Dataset[Episode]):
    """Episodic key-value recall curriculum from the CREST project blueprint.

    Tokens are synthetic and intentionally simple: each step is either a write
    statement or a query whose target label is the remembered value token.
    """

    PAD = 0
    WRITE = 1
    QUERY = 2
    ANSWER = 3

    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        self.cfg = cfg
        self.size = cfg.train_episodes if split == "train" else cfg.eval_episodes
        self.rng = random.Random(cfg.seed + (0 if split == "train" else 10_000))
        self.key_offset = 10
        self.value_offset = self.key_offset + cfg.num_keys

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Episode:
        rng = random.Random(self.cfg.seed + index)
        memory: dict[int, int] = {}
        steps: list[list[int]] = []
        labels: list[list[int]] = []
        for _ in range(self.cfg.episode_steps):
            do_query = bool(memory) and rng.random() < self.cfg.query_probability
            if do_query:
                key = rng.choice(list(memory.keys()))
                value = memory[key]
                toks = [self.QUERY, self.key_offset + key, self.ANSWER, self.PAD]
                labs = [-100, -100, -100, self.value_offset + value]
            else:
                key = rng.randrange(self.cfg.num_keys)
                value = rng.randrange(self.cfg.num_values)
                memory[key] = value
                toks = [self.WRITE, self.key_offset + key, self.value_offset + value, self.PAD]
                labs = [-100, -100, -100, -100]
            toks = toks[: self.cfg.step_length] + [self.PAD] * max(0, self.cfg.step_length - len(toks))
            labs = labs[: self.cfg.step_length] + [-100] * max(0, self.cfg.step_length - len(labs))
            steps.append(toks)
            labels.append(labs)
        return Episode(input_ids=torch.tensor(steps, dtype=torch.long), labels=torch.tensor(labels, dtype=torch.long), step_idx=torch.arange(self.cfg.episode_steps, dtype=torch.long))


def collate_episodes(batch: list[Episode]) -> Episode:
    return Episode(input_ids=torch.stack([x.input_ids for x in batch]), labels=torch.stack([x.labels for x in batch]), step_idx=torch.stack([x.step_idx for x in batch]))


class SyntheticMultiHopDataset(SyntheticKeyValueDataset):
    """Synthetic A->B, B->C, ask A->? episodic curriculum from the CREST project blueprint."""

    LINK = 4

    def __getitem__(self, index: int) -> Episode:
        rng = random.Random(self.cfg.seed + index)
        edges: dict[int, int] = {}
        steps: list[list[int]] = []
        labels: list[list[int]] = []
        for t in range(self.cfg.episode_steps):
            do_query = len(edges) >= 2 and rng.random() < self.cfg.query_probability
            if do_query:
                start = rng.choice(list(edges.keys()))
                mid = edges[start]
                target = edges.get(mid, mid)
                toks = [self.QUERY, self.key_offset + start, self.LINK, self.ANSWER]
                labs = [-100, -100, -100, self.value_offset + target % self.cfg.num_values]
            else:
                a = rng.randrange(self.cfg.num_keys)
                b = rng.randrange(self.cfg.num_keys)
                edges[a] = b
                toks = [self.WRITE, self.key_offset + a, self.LINK, self.key_offset + b]
                labs = [-100, -100, -100, -100]
            steps.append(toks[: self.cfg.step_length] + [self.PAD] * max(0, self.cfg.step_length - len(toks)))
            labels.append(labs[: self.cfg.step_length] + [-100] * max(0, self.cfg.step_length - len(labs)))
        return Episode(torch.tensor(steps), torch.tensor(labels), torch.arange(self.cfg.episode_steps))


class SyntheticToolTraceDataset(SyntheticKeyValueDataset):
    """Synthetic tool-call/return traces for agentic step-boundary training."""

    CALL = 5
    RETURN = 6

    def __getitem__(self, index: int) -> Episode:
        rng = random.Random(self.cfg.seed + index)
        steps, labels = [], []
        accumulator = 0
        for _ in range(self.cfg.episode_steps):
            arg = rng.randrange(self.cfg.num_values)
            accumulator = (accumulator + arg) % self.cfg.num_values
            toks = [self.CALL, self.value_offset + arg, self.RETURN, self.PAD]
            labs = [-100, -100, -100, self.value_offset + accumulator]
            steps.append(toks[: self.cfg.step_length] + [self.PAD] * max(0, self.cfg.step_length - len(toks)))
            labels.append(labs[: self.cfg.step_length] + [-100] * max(0, self.cfg.step_length - len(labs)))
        return Episode(torch.tensor(steps), torch.tensor(labels), torch.arange(self.cfg.episode_steps))


class JsonlEpisodicDataset(Dataset[Episode]):
    """Natural episodic pretraining loader.

    Expects JSONL rows with `steps`, each step containing `input_ids` and optional
    `labels`. Labels default to next-token SFT labels supplied by the data builder.
    This enforces the CREST no-cross-episode packing rule: each row is one episode.
    """

    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        if cfg.path is None:
            raise ValueError("JsonlEpisodicDataset requires DataConfig.path")
        path = Path(cfg.path)
        if path.is_dir():
            path = path / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Episodic JSONL dataset file not found at '{path}'. "
                f"Please download/prepare the dataset first by running: "
                f"python -m crest.cli_prepare_text --out {cfg.path}"
            )
        self.rows = path.read_text(encoding="utf-8").splitlines()
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Episode:
        row = json.loads(self.rows[index])
        steps = row["steps"][: self.cfg.episode_steps]
        input_steps, label_steps = [], []
        for step in steps:
            ids = list(step["input_ids"][: self.cfg.step_length])
            if "labels" in step:
                labels = list(step["labels"][: self.cfg.step_length])
            else:
                labels = ids[1:] + [-100]
            ids += [0] * max(0, self.cfg.step_length - len(ids))
            labels += [-100] * max(0, self.cfg.step_length - len(labels))
            input_steps.append(ids)
            label_steps.append(labels)
        while len(input_steps) < self.cfg.episode_steps:
            input_steps.append([0] * self.cfg.step_length)
            label_steps.append([-100] * self.cfg.step_length)
        return Episode(torch.tensor(input_steps), torch.tensor(label_steps), torch.arange(self.cfg.episode_steps))


def build_dataset(cfg: DataConfig, split: str = "train") -> Dataset[Episode]:
    if cfg.task in {"key_value_recall", "overwrite_recall"}:
        return SyntheticKeyValueDataset(cfg, split)
    if cfg.task == "multi_hop":
        return SyntheticMultiHopDataset(cfg, split)
    if cfg.task == "tool_trace":
        return SyntheticToolTraceDataset(cfg, split)
    if cfg.task == "jsonl_episodic":
        return JsonlEpisodicDataset(cfg, split)
    raise ValueError(f"unknown data task {cfg.task!r}")
