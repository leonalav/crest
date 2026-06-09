from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Iterator

import torch
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset

from .config import DataConfig
from .cli_prepare_manifest import iter_windows, load_hf_dataset, tokenizer_meta
from .cli_prepare_text import encode_text, load_tokenizer


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


class SyntheticExactTwoHopDataset(SyntheticKeyValueDataset):
    """Synthetic curriculum where every query has an exact two-hop answer."""

    LINK = 4

    def __getitem__(self, index: int) -> Episode:
        rng = random.Random(self.cfg.seed + index)
        chains: list[tuple[int, int, int]] = []
        edges: dict[int, int] = {}
        steps: list[list[int]] = []
        labels: list[list[int]] = []
        for _ in range(self.cfg.episode_steps):
            do_query = bool(chains) and rng.random() < self.cfg.query_probability
            if do_query:
                a, _, c = rng.choice(chains)
                toks = [self.QUERY, self.key_offset + a, self.LINK, self.ANSWER]
                labs = [-100, -100, -100, self.value_offset + c % self.cfg.num_values]
            else:
                a = rng.randrange(self.cfg.num_keys)
                b = rng.randrange(self.cfg.num_keys)
                c = rng.randrange(self.cfg.num_keys)
                edges[a] = b
                edges[b] = c
                chains.append((a, b, c))
                if rng.random() < 0.5:
                    toks = [self.WRITE, self.key_offset + a, self.LINK, self.key_offset + b]
                else:
                    toks = [self.WRITE, self.key_offset + b, self.LINK, self.key_offset + c]
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


class ArrowEpisodicDataset(Dataset[Episode]):
    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        if cfg.path is None:
            raise ValueError("ArrowEpisodicDataset requires DataConfig.path")
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError("Arrow episodic datasets require `pip install datasets`.") from exc
        path = Path(cfg.path)
        if path.is_dir() and (path / split).exists():
            path = path / split
        if not path.exists():
            raise FileNotFoundError(f"Episodic Arrow dataset not found at '{path}'. Run: python -m crest.cli_prepare_text --out {cfg.path}")
        self.ds = load_from_disk(str(path))
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Episode:
        row = self.ds[int(index)]
        steps = row["steps"][: self.cfg.episode_steps]
        input_steps, label_steps = [], []
        for step in steps:
            ids = list(step["input_ids"][: self.cfg.step_length])
            labels = list(step["labels"][: self.cfg.step_length]) if "labels" in step else ids[1:] + [-100]
            ids += [0] * max(0, self.cfg.step_length - len(ids))
            labels += [-100] * max(0, self.cfg.step_length - len(labels))
            input_steps.append(ids)
            label_steps.append(labels)
        while len(input_steps) < self.cfg.episode_steps:
            input_steps.append([0] * self.cfg.step_length)
            label_steps.append([-100] * self.cfg.step_length)
        return Episode(torch.tensor(input_steps), torch.tensor(label_steps), torch.arange(self.cfg.episode_steps))


def episode_row_to_tensors(row: dict, cfg: DataConfig) -> Episode:
    steps = row["steps"][: cfg.episode_steps]
    input_steps, label_steps = [], []
    for step in steps:
        ids = list(step["input_ids"][: cfg.step_length])
        labels = list(step["labels"][: cfg.step_length]) if "labels" in step else ids[1:] + [-100]
        ids += [0] * max(0, cfg.step_length - len(ids))
        labels += [-100] * max(0, cfg.step_length - len(labels))
        input_steps.append(ids)
        label_steps.append(labels)
    while len(input_steps) < cfg.episode_steps:
        input_steps.append([0] * cfg.step_length)
        label_steps.append([-100] * cfg.step_length)
    return Episode(torch.tensor(input_steps), torch.tensor(label_steps), torch.arange(cfg.episode_steps))


class StreamingTextDataset(IterableDataset[Episode]):
    """Tokenize Hugging Face manifest sources online instead of using prepared shards."""

    def __init__(self, cfg: DataConfig, split: str = "train") -> None:
        if cfg.path is None:
            raise ValueError("StreamingTextDataset requires DataConfig.path pointing to a manifest YAML")
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Streaming text datasets require `pip install PyYAML`.") from exc
        manifest = yaml.safe_load(Path(cfg.path).read_text(encoding="utf-8")) or {}
        items = manifest.get("datasets")
        if not isinstance(items, list) or not items:
            raise ValueError("streaming_text manifest must contain a non-empty `datasets` list")
        self.items = items
        self.cfg = cfg
        self.split = split
        tokenizer_name = cfg.metadata.get("tokenizer", manifest.get("metadata", {}).get("tokenizer", "byte"))
        self.tokenizer = load_tokenizer(str(tokenizer_name))
        self.pad_token_id = int(tokenizer_meta(self.tokenizer)["pad_token_id"])

    def __iter__(self) -> Iterator[Episode]:
        # Shard documents across DataLoader workers so each worker tokenizes only
        # its own slice. Without this, num_workers copies of the tokenizer each
        # process every document, multiplying CPU overhead by num_workers.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        window = self.cfg.episode_steps * self.cfg.step_length + 1
        stride = int(self.cfg.metadata.get("stride_tokens", self.cfg.episode_steps * self.cfg.step_length))
        for item in self.items:
            stream_item = dict(item)
            stream_item["streaming"] = True
            if self.split == "eval" and "eval_split" in stream_item:
                stream_item["split"] = stream_item["eval_split"]
            ds = load_hf_dataset(stream_item)
            text_field = stream_item.get("text_field", "text")
            max_documents = stream_item.get("max_documents")
            doc_idx = 0
            docs = 0
            for row in ds:
                text = row.get(text_field)
                if not text:
                    continue
                # Each worker takes every num_workers-th document starting at worker_id.
                if doc_idx % num_workers != worker_id:
                    doc_idx += 1
                    continue
                doc_idx += 1
                docs += 1
                ids = encode_text(self.tokenizer, str(text))
                for chunk in iter_windows(ids, window, stride, self.pad_token_id):
                    yield episode_row_to_tensors(self._episode_from_window(chunk), self.cfg)
                if max_documents is not None and docs >= int(max_documents // max(1, num_workers)):
                    break

    def _episode_from_window(self, tokens: list[int]) -> dict:
        needed = self.cfg.episode_steps * self.cfg.step_length
        tokens = tokens[: needed + 1]
        if len(tokens) < needed + 1:
            tokens = tokens + [self.pad_token_id] * (needed + 1 - len(tokens))
        inputs = tokens[:-1]
        labels = tokens[1:]
        steps = []
        for step_idx in range(self.cfg.episode_steps):
            start = step_idx * self.cfg.step_length
            end = start + self.cfg.step_length
            label_chunk = [-100 if tok == self.pad_token_id else tok for tok in labels[start:end]]
            steps.append({"input_ids": inputs[start:end], "labels": label_chunk})
        return {"steps": steps}


class RawTextDataset(IterableDataset[Episode]):
    """Tokenize bounded local raw Arrow or JSONL files during training.

    Worker splitting: when used with DataLoader num_workers > 0, each worker
    processes every num_workers-th document (round-robin by document index).
    This eliminates duplicate tokenization — without get_worker_info() sharding
    all workers iterate the entire file and tokenize every document independently,
    multiplying tokenizer CPU cost by num_workers.
    """

    def __init__(self, cfg: DataConfig, split: str = "train", file_format: str = "arrow") -> None:
        if cfg.path is None:
            raise ValueError("RawTextDataset requires DataConfig.path")
        path = Path(cfg.path)
        suffix = ".jsonl" if file_format == "jsonl" else ".arrow"
        self.path = path if path.suffix else path / f"{split}{suffix}"
        if not self.path.exists() and "episodic_arrow" in path.parts:
            fallback = Path(*[("raw_text" if part == "episodic_arrow" else part) for part in path.parts]) / f"{split}{suffix}"
            if fallback.exists():
                self.path = fallback
        if not self.path.exists():
            raise FileNotFoundError(f"Raw text split not found at '{self.path}'")
        self.cfg = cfg
        self.file_format = file_format
        tokenizer_name = cfg.metadata.get("tokenizer", "byte")
        self.text_field = cfg.metadata.get("text_field", "text")
        self.tokenizer = load_tokenizer(str(tokenizer_name))
        self.pad_token_id = int(tokenizer_meta(self.tokenizer)["pad_token_id"])

    def __iter__(self) -> Iterator[Episode]:
        if self.file_format == "jsonl":
            yield from self._iter_jsonl()
            return
        yield from self._iter_arrow()

    def _worker_slice(self) -> tuple[int, int]:
        """Return (worker_id, num_workers) for round-robin document sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def _iter_jsonl(self) -> Iterator[Episode]:
        # Each DataLoader worker owns every num_workers-th document starting at
        # worker_id. This prevents each worker from re-tokenizing the entire file.
        worker_id, num_workers = self._worker_slice()
        window = self.cfg.episode_steps * self.cfg.step_length + 1
        stride = int(self.cfg.metadata.get("stride_tokens", self.cfg.episode_steps * self.cfg.step_length))
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            doc_idx = 0
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get(self.text_field)
                if not text:
                    continue
                if doc_idx % num_workers == worker_id:
                    ids = encode_text(self.tokenizer, str(text))
                    for chunk in iter_windows(ids, window, stride, self.pad_token_id):
                        yield episode_row_to_tensors(self._episode_from_window(chunk), self.cfg)
                doc_idx += 1

    def _iter_arrow(self) -> Iterator[Episode]:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError("Raw Arrow training requires `pip install pyarrow`.") from exc
        worker_id, num_workers = self._worker_slice()
        window = self.cfg.episode_steps * self.cfg.step_length + 1
        stride = int(self.cfg.metadata.get("stride_tokens", self.cfg.episode_steps * self.cfg.step_length))
        with pa.memory_map(str(self.path), "r") as source:
            reader = pa.ipc.open_file(source)
            text_col = reader.schema.get_field_index(self.text_field)
            if text_col < 0:
                raise ValueError(f"Raw Arrow file '{self.path}' has no column {self.text_field!r}")
            doc_idx = 0
            for batch_idx in range(reader.num_record_batches):
                batch = reader.get_batch(batch_idx)
                for text in batch.column(text_col).to_pylist():
                    if not text:
                        continue
                    if doc_idx % num_workers == worker_id:
                        ids = encode_text(self.tokenizer, str(text))
                        for chunk in iter_windows(ids, window, stride, self.pad_token_id):
                            yield episode_row_to_tensors(self._episode_from_window(chunk), self.cfg)
                    doc_idx += 1

    def _episode_from_window(self, tokens: list[int]) -> dict:
        needed = self.cfg.episode_steps * self.cfg.step_length
        tokens = tokens[: needed + 1]
        if len(tokens) < needed + 1:
            tokens = tokens + [self.pad_token_id] * (needed + 1 - len(tokens))
        inputs = tokens[:-1]
        labels = tokens[1:]
        steps = []
        for step_idx in range(self.cfg.episode_steps):
            start = step_idx * self.cfg.step_length
            end = start + self.cfg.step_length
            label_chunk = [-100 if tok == self.pad_token_id else tok for tok in labels[start:end]]
            steps.append({"input_ids": inputs[start:end], "labels": label_chunk})
        return {"steps": steps}


def build_dataset(cfg: DataConfig, split: str = "train") -> Dataset[Episode]:
    if cfg.task in {"key_value_recall", "overwrite_recall"}:
        return SyntheticKeyValueDataset(cfg, split)
    if cfg.task == "multi_hop":
        return SyntheticMultiHopDataset(cfg, split)
    if cfg.task == "exact_two_hop":
        return SyntheticExactTwoHopDataset(cfg, split)
    if cfg.task == "tool_trace":
        return SyntheticToolTraceDataset(cfg, split)
    if cfg.task == "jsonl_episodic":
        return JsonlEpisodicDataset(cfg, split)
    if cfg.task == "arrow_episodic":
        return ArrowEpisodicDataset(cfg, split)
    if cfg.task == "streaming_text":
        return StreamingTextDataset(cfg, split)
    if cfg.task == "raw_text":
        return RawTextDataset(cfg, split)
    if cfg.task == "raw_jsonl":
        return RawTextDataset(cfg, split, file_format="jsonl")
    raise ValueError(f"unknown data task {cfg.task!r}")
