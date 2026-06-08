from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


class ByteTokenizer:
    """Dependency-free byte tokenizer for enwik8/text8/PTB/WikiText smoke runs."""

    vocab_size = 260
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    offset = 3

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] + [b + self.offset for b in text.encode("utf-8", errors="ignore")] + [self.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        data = bytes(max(0, tok - self.offset) for tok in ids if tok >= self.offset)
        return data.decode("utf-8", errors="ignore")


def load_tokenizer(name: str):
    if name == "byte":
        return ByteTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Non-byte tokenizers require `pip install transformers`.") from exc
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def iter_texts(input_path: Path, text_field: str = "text") -> Iterable[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                yield str(row.get(text_field, ""))
        else:
            yield input_path.read_text(encoding="utf-8", errors="ignore")
        return
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".text", ".jsonl", ".tokens"}:
            yield from iter_texts(path, text_field=text_field)


def iter_hf_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, max_documents: int | None) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HuggingFace dataset loading requires `pip install datasets`.") from exc
    ds = load_dataset(dataset_name, dataset_config, split=split) if dataset_config else load_dataset(dataset_name, split=split)
    count = 0
    for row in ds:
        text = row.get(text_field)
        if text:
            yield str(text)
            count += 1
            if max_documents is not None and count >= max_documents:
                break


def encode_text(tokenizer, text: str) -> list[int]:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return list(tokenizer.encode(text, add_special_tokens=True))


def tokens_to_episode(tokens: list[int], episode_steps: int, step_length: int, pad_token_id: int = 0) -> dict | None:
    needed_inputs = episode_steps * step_length
    if len(tokens) < 2:
        return None
    tokens = tokens[: needed_inputs + 1]
    if len(tokens) < needed_inputs + 1:
        tokens = tokens + [pad_token_id] * (needed_inputs + 1 - len(tokens))
    inputs = tokens[:-1]
    labels = tokens[1:]
    steps = []
    for s in range(episode_steps):
        start = s * step_length
        end = start + step_length
        label_chunk = [-100 if tok == pad_token_id else tok for tok in labels[start:end]]
        steps.append({"input_ids": inputs[start:end], "labels": label_chunk})
    return {"steps": steps}


def build_episodes(texts: Iterable[str], tokenizer, episode_steps: int, step_length: int, stride_tokens: int) -> list[dict]:
    episodes: list[dict] = []
    window = episode_steps * step_length + 1
    stride = stride_tokens or episode_steps * step_length
    for text in texts:
        ids = encode_text(tokenizer, text)
        for start in range(0, max(1, len(ids) - 1), stride):
            chunk = ids[start : start + window]
            ep = tokens_to_episode(chunk, episode_steps, step_length, getattr(tokenizer, "pad_token_id", 0) or 0)
            if ep is not None:
                episodes.append(ep)
            if start + window >= len(ids):
                break
    return episodes


def save_arrow_split(rows: list[dict], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Arrow shard writing requires `pip install datasets`.") from exc
    if path.exists():
        import shutil
        shutil.rmtree(path)
    Dataset.from_list(rows).save_to_disk(str(path))


def write_split(episodes: list[dict], out_dir: Path, eval_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_eval = max(1, int(len(episodes) * eval_fraction)) if len(episodes) > 1 else 0
    splits = {"eval": episodes[:n_eval], "train": episodes[n_eval:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        save_arrow_split(rows, out_dir / split)
    meta = {"format": "arrow", "episodes": len(episodes), "train": len(splits["train"]), "eval": len(splits["eval"]), "schema": {"steps": [{"input_ids": "list[int]", "labels": "next-token list[int], pad labels masked as -100"}]}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw text into CREST episodic Arrow shards")
    parser.add_argument("--input", default=None, help="Raw text file, JSONL with text field, or directory")
    parser.add_argument("--hf-dataset", default=None, help="Optional HuggingFace dataset name, e.g. Salesforce/wikitext")
    parser.add_argument("--hf-config", default=None, help="Optional HuggingFace dataset config, e.g. wikitext-103-raw-v1")
    parser.add_argument("--hf-split", default="train", help="HuggingFace split")
    parser.add_argument("--kaggle-dataset", default=None, help="Optional Kaggle dataset name, e.g. vadimkurochkin/wikitext-103")
    parser.add_argument("--mock-download", action="store_true", help="Generate mock wikitext files instead of downloading from Kaggle")
    parser.add_argument("--text-field", default="text", help="Text field for JSONL/HuggingFace rows")
    parser.add_argument("--max-documents", type=int, default=None, help="Optional document cap for cheap smoke prep")
    parser.add_argument("--out", required=True, help="Output directory containing train/eval Arrow datasets")
    parser.add_argument("--tokenizer", default="byte", help="byte or HuggingFace tokenizer name, e.g. gpt2")
    parser.add_argument("--episode-steps", type=int, default=16)
    parser.add_argument("--step-length", type=int, default=128)
    parser.add_argument("--stride-tokens", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.input is None and args.hf_dataset is None and args.kaggle_dataset is None:
        args.kaggle_dataset = "vadimkurochkin/wikitext-103"
    tokenizer = load_tokenizer(args.tokenizer)
    if args.hf_dataset:
        texts = iter_hf_texts(args.hf_dataset, args.hf_config, args.hf_split, args.text_field, args.max_documents)
        source = f"hf:{args.hf_dataset}/{args.hf_config or ''}:{args.hf_split}"
    elif args.kaggle_dataset:
        if args.mock_download:
            from crest.downloader import download_wikitext103
            mock_dir = Path(args.out) / "raw_mock_tokens"
            input_path = download_wikitext103(dest_dir=mock_dir, mock=True)
        else:
            import kagglehub
            print(f"[prepare_text] downloading {args.kaggle_dataset} via kagglehub...", flush=True)
            input_path = Path(kagglehub.dataset_download(args.kaggle_dataset))
        texts = iter_texts(input_path, text_field=args.text_field)
        source = f"kaggle:{args.kaggle_dataset}"
    else:
        input_path = Path(args.input)
        texts = iter_texts(input_path, text_field=args.text_field)
        source = str(input_path)
    episodes = build_episodes(texts, tokenizer, args.episode_steps, args.step_length, args.stride_tokens)
    if not episodes:
        raise RuntimeError(
            "No episodes produced. Source had no usable text or documents were too short. "
            f"source={source!r}, tokenizer={args.tokenizer!r}, episode_steps={args.episode_steps}, step_length={args.step_length}. "
            "If using WikiText locally, verify files exist with `find data/raw/wikitext103 -type f | head`. "
            "Or use `--hf-dataset Salesforce/wikitext --hf-config wikitext-103-raw-v1`."
        )
    write_split(episodes, Path(args.out), args.eval_fraction, args.seed)
    print(f"[prepare_text] wrote {len(episodes)} episodes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
