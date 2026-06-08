from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterator

import yaml

from .cli_prepare_text import encode_text, load_tokenizer, save_arrow_split


def tokenizer_meta(tokenizer) -> dict:
    try:
        vocab_size = int(len(tokenizer))
    except TypeError:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0))
     pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    return {"vocab_size": vocab_size, "pad_token_id": int(pad_id), "eos_token_id": eos_id, "bos_token_id": bos_id}


def episode_from_window(tokens: list[int], episode_steps: int, step_length: int, pad_token_id: int) -> dict:
    needed = episode_steps * step_length
    tokens = tokens[: needed + 1]
    if len(tokens) < needed + 1:
        tokens = tokens + [pad_token_id] * (needed + 1 - len(tokens))
    inputs = tokens[:-1]
    labels = tokens[1:]
    steps = []
    for s in range(episode_steps):
        start = s * step_length
        end = start + step_length
        input_chunk = inputs[start:end]
        label_chunk = labels[start:end]
        label_chunk = [-100 if tok == pad_token_id else tok for tok in label_chunk]
        steps.append({"input_ids": input_chunk, "labels": label_chunk})
    return {"steps": steps}


def iter_windows(ids: list[int], window: int, stride: int, pad_token_id: int) -> Iterator[list[int]]:
    if len(ids) < 2:
        return
    if len(ids) <= window:
        yield ids + [pad_token_id] * (window - len(ids))
        return
    for start in range(0, len(ids) - 1, stride):
        chunk = ids[start : start + window]
        if len(chunk) < 2:
            break
        yield chunk
        if start + window >= len(ids):
            break


def load_hf_dataset(item: dict):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HuggingFace dataset loading requires `pip install datasets`.") from exc
    repo = item["repo"]
    subset = item.get("subset")
    split = item.get("split", "train")
    streaming = bool(item.get("streaming", True))
    print(f"[prepare_manifest] loading repo={repo} subset={subset} split={split} streaming={streaming}", flush=True)
    return load_dataset(repo, subset, split=split, streaming=streaming) if subset else load_dataset(repo, split=split, streaming=streaming)


def maybe_cleanup_cache(item: dict, default_cleanup: bool) -> None:
    cleanup = bool(item.get("cleanup_cache", default_cleanup))
    if not cleanup:
        return
    cache_dir = item.get("cache_dir")
    if cache_dir:
        target = Path(cache_dir).expanduser()
        if target.exists():
            shutil.rmtree(target)
            print(f"[prepare_manifest] removed cache_dir={target}", flush=True)
        return
    try:
        from datasets import config
    except ImportError:
        return
    root = Path(config.HF_DATASETS_CACHE)
    repo = item["repo"].replace("/", "___")
    if not root.exists():
        return
    for child in root.iterdir():
        if repo in child.name:
            shutil.rmtree(child, ignore_errors=True)
            print(f"[prepare_manifest] removed dataset cache {child}", flush=True)


def split_for_episode(index: int, eval_fraction: float, seed: int) -> str:
    if eval_fraction <= 0:
        return "train"
    rng = random.Random(seed + index)
    return "eval" if rng.random() < eval_fraction else "train"


def prepare_manifest(args, items: list[dict], tokenizer) -> dict:
    meta = tokenizer_meta(tokenizer)
    pad_token_id = meta["pad_token_id"]
    window = args.episode_steps * args.step_length + 1
    stride = args.stride_tokens or args.episode_steps * args.step_length
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": [], "eval": []}
    total_tokens = 0
    total_episodes = 0
    split_counts = {"train": 0, "eval": 0}
    source_stats = []

    for item in items:
        ds = load_hf_dataset(item)
        text_field = item.get("text_field", "text")
        max_documents = item.get("max_documents", args.max_documents)
        max_tokens = item.get("max_tokens", args.max_tokens_per_dataset)
        source_tokens = 0
        source_docs = 0
        source_episodes = 0
        for row in ds:
            text = row.get(text_field)
            if not text:
                continue
            ids = encode_text(tokenizer, str(text))
            if max_tokens is not None and source_tokens >= int(max_tokens):
                break
            remaining_source = None if max_tokens is None else max(0, int(max_tokens) - source_tokens)
            remaining_global = None if args.max_tokens is None else max(0, int(args.max_tokens) - total_tokens)
            remaining_limits = [x for x in [remaining_source, remaining_global] if x is not None]
            if remaining_limits:
                remaining = min(remaining_limits)
                if remaining <= 0:
                    break
                ids = ids[:remaining]
            source_tokens += len(ids)
            total_tokens += len(ids)
            source_docs += 1
            for chunk in iter_windows(ids, window, stride, pad_token_id):
                if args.max_episodes is not None and total_episodes >= args.max_episodes:
                    break
                ep = episode_from_window(chunk, args.episode_steps, args.step_length, pad_token_id)
                split = split_for_episode(total_episodes, args.eval_fraction, args.seed)
                split_rows[split].append(ep)
                split_counts[split] += 1
                total_episodes += 1
                source_episodes += 1
            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
            if max_documents is not None and source_docs >= int(max_documents):
                break
            if args.max_tokens is not None and total_tokens >= int(args.max_tokens):
                break
        source_stats.append({"repo": item["repo"], "subset": item.get("subset"), "split": item.get("split", "train"), "documents": source_docs, "tokens": source_tokens, "episodes": source_episodes})
        print(f"[prepare_manifest] source done docs={source_docs} tokens={source_tokens} episodes={source_episodes}", flush=True)
        maybe_cleanup_cache(item, args.cleanup_cache)
        if args.max_episodes is not None and total_episodes >= args.max_episodes:
            break
        if args.max_tokens is not None and total_tokens >= int(args.max_tokens):
            break

    if total_episodes == 0:
        raise RuntimeError("No episodes produced from manifest")
    for split, rows in split_rows.items():
        save_arrow_split(rows, out_dir / split)
    return {"episodes": total_episodes, "train": split_counts["train"], "eval": split_counts["eval"], "tokens": total_tokens, "sources": source_stats, **meta}


def prepare_raw_text(args, items: list[dict], tokenizer) -> dict:
    meta = tokenizer_meta(tokenizer)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    total_tokens = 0
    total_docs = 0
    split_counts = {"train": 0, "eval": 0}
    source_stats = []
    with train_path.open("w", encoding="utf-8") as train_f, eval_path.open("w", encoding="utf-8") as eval_f:
        for item in items:
            ds = load_hf_dataset(item)
            text_field = item.get("text_field", "text")
            max_documents = item.get("max_documents", args.max_documents)
            max_tokens = item.get("max_tokens", args.max_tokens_per_dataset)
            source_tokens = 0
            source_docs = 0
            for row in ds:
                text = row.get(text_field)
                if not text:
                    continue
                ids = encode_text(tokenizer, str(text))
                if max_tokens is not None and source_tokens >= int(max_tokens):
                    break
                remaining_source = None if max_tokens is None else max(0, int(max_tokens) - source_tokens)
                remaining_global = None if args.max_tokens is None else max(0, int(args.max_tokens) - total_tokens)
                remaining_limits = [x for x in [remaining_source, remaining_global] if x is not None]
                if remaining_limits and min(remaining_limits) <= 0:
                    break
                if remaining_limits and len(ids) > min(remaining_limits):
                    break
                source_tokens += len(ids)
                total_tokens += len(ids)
                split = split_for_episode(total_docs, args.eval_fraction, args.seed)
                target = eval_f if split == "eval" else train_f
                target.write(json.dumps({"text": str(text)}, ensure_ascii=False, separators=(",", ":")) + "\n")
                split_counts[split] += 1
                total_docs += 1
                source_docs += 1
                if max_documents is not None and source_docs >= int(max_documents):
                    break
                if args.max_tokens is not None and total_tokens >= int(args.max_tokens):
                    break
            source_stats.append({"repo": item["repo"], "subset": item.get("subset"), "split": item.get("split", "train"), "documents": source_docs, "tokens": source_tokens})
            print(f"[prepare_manifest] raw source done docs={source_docs} tokens={source_tokens}", flush=True)
            maybe_cleanup_cache(item, args.cleanup_cache)
            if args.max_tokens is not None and total_tokens >= int(args.max_tokens):
                break
    if total_docs == 0:
        raise RuntimeError("No raw text documents produced from manifest")
    return {"documents": total_docs, "train": split_counts["train"], "eval": split_counts["eval"], "tokens": total_tokens, "sources": source_stats, **meta}


def update_manifest_config(manifest_path: Path, out_dir: Path, tokenizer_name: str, episode_steps: int, step_length: int, meta: dict) -> None:
    cfg = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    cfg["format"] = "arrow"
    cfg["task"] = "arrow_episodic"
    cfg["path"] = str(out_dir).replace("\\", "/")
    cfg["episode_steps"] = episode_steps
    cfg["step_length"] = step_length
    cfg["train_episodes"] = int(meta["train"])
    cfg["eval_episodes"] = int(meta["eval"])
    cfg["vocab_size"] = int(meta["vocab_size"])
    cfg["metadata"] = {"tokenizer": tokenizer_name, "pad_token_id": meta.get("pad_token_id"), "eos_token_id": meta.get("eos_token_id")}
    manifest_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[prepare_manifest] updated manifest config {manifest_path}", flush=True)


def update_raw_manifest_config(manifest_path: Path, out_dir: Path, tokenizer_name: str, episode_steps: int, step_length: int, meta: dict) -> None:
    cfg = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    cfg["format"] = "raw_text"
    cfg["task"] = "raw_text"
    cfg["path"] = str(out_dir).replace("\\", "/")
    cfg["episode_steps"] = episode_steps
    cfg["step_length"] = step_length
    cfg["train_episodes"] = 0
    cfg["eval_episodes"] = 0
    cfg["vocab_size"] = int(meta["vocab_size"])
    cfg["metadata"] = {"tokenizer": tokenizer_name, "pad_token_id": meta.get("pad_token_id"), "eos_token_id": meta.get("eos_token_id"), "text_field": "text"}
    manifest_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[prepare_manifest] updated raw manifest config {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one or more HF datasets into CREST episodic Arrow shards")
    parser.add_argument("--manifest", required=True, help="YAML manifest with datasets list")
    parser.add_argument("--out", required=True, help="Output directory containing train/eval Arrow datasets")
    parser.add_argument("--name", default="default", help="Deprecated; no YAML files are created")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--episode-steps", type=int, default=16)
    parser.add_argument("--step-length", type=int, default=128)
    parser.add_argument("--stride-tokens", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--max-documents", type=int, default=None, help="Default per-dataset document cap")
    parser.add_argument("--max-tokens-per-dataset", type=int, default=None, help="Default hard token cap for each dataset")
    parser.add_argument("--max-tokens", type=int, default=None, help="Global hard token cap across all datasets")
    parser.add_argument("--max-episodes", type=int, default=None, help="Global hard episode cap")
    parser.add_argument("--cleanup-cache", action="store_true", help="Remove HF dataset cache for each source after it is processed")
    parser.add_argument("--raw-text-only", action="store_true", help="Download bounded raw text JSONL files instead of pretokenized Arrow shards")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    items = manifest.get("datasets")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest must contain a non-empty `datasets` list")
    tokenizer = load_tokenizer(args.tokenizer)
    out_dir = Path(args.out)
    if args.raw_text_only:
        meta = prepare_raw_text(args, items, tokenizer)
        metadata = {"format": "raw_text", "schema": {"text": "raw document text"}, "tokenizer": args.tokenizer, "manifest": manifest, "episode_steps": args.episode_steps, "step_length": args.step_length, **meta}
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        update_raw_manifest_config(Path(args.manifest), out_dir, args.tokenizer, args.episode_steps, args.step_length, meta)
        print(f"[prepare_manifest] wrote {meta['documents']} raw documents, {meta['tokens']} tokens to {out_dir}", flush=True)
        return
    meta = prepare_manifest(args, items, tokenizer)
    metadata = {"format": "arrow", "schema": {"steps": [{"input_ids": "list[int]", "labels": "next-token list[int], pad labels masked as -100"}]}, "tokenizer": args.tokenizer, "manifest": manifest, "episode_steps": args.episode_steps, "step_length": args.step_length, **meta}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_manifest_config(Path(args.manifest), out_dir, args.tokenizer, args.episode_steps, args.step_length, meta)
    print(f"[prepare_manifest] wrote {meta['episodes']} episodes, {meta['tokens']} tokens to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
