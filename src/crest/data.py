from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import DataConfig


@dataclass(frozen=True)
class Episode:
    input_ids: torch.Tensor
    labels: torch.Tensor
    step_idx: torch.Tensor


class SyntheticKeyValueDataset(Dataset[Episode]):
    """Episodic key-value recall curriculum from verdict.md.

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
