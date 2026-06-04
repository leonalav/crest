from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


class ByteTokenizer:
    """Dependency-free byte tokenizer for enwik8/text8/PTB/WikiText smoke runs."""

    vocab_size = 260
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    offset = 3

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] + [b + self.offset for b in text.encode("utf-8", errors="ignore")] + [self.eos_token_id]


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


def iter_texts(input_path: Path) -> Iterable[str]:
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                yield str(row.get("text", ""))
        else:
            yield input_path.read_text(encoding="utf-8", errors="ignore")
        return
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".text", ".jsonl"}:
            yield from iter_texts(path)


def encode_text(tokenizer, text: str) -> list[int]:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return list(tokenizer.encode(text, add_special_tokens=True))


def tokens_to_episode(tokens: list[int], episode_steps: int, step_length: int) -> dict | None:
    needed_inputs = episode_steps * step_length
    if len(tokens) < 2:
        return None
    tokens = tokens[: needed_inputs + 1]
    if len(tokens) < needed_inputs + 1:
        tokens = tokens + [0] * (needed_inputs + 1 - len(tokens))
    inputs = tokens[:-1]
    labels = tokens[1:]
    steps = []
    for s in range(episode_steps):
        start = s * step_length
        end = start + step_length
        steps.append({"input_ids": inputs[start:end], "labels": labels[start:end]})
    return {"steps": steps}


def build_episodes(texts: Iterable[str], tokenizer, episode_steps: int, step_length: int, stride_tokens: int) -> list[dict]:
    episodes: list[dict] = []
    window = episode_steps * step_length + 1
    stride = stride_tokens or episode_steps * step_length
    for text in texts:
        ids = encode_text(tokenizer, text)
        for start in range(0, max(1, len(ids) - 1), stride):
            chunk = ids[start : start + window]
            ep = tokens_to_episode(chunk, episode_steps, step_length)
            if ep is not None:
                episodes.append(ep)
            if start + window >= len(ids):
                break
    return episodes


def write_split(episodes: list[dict], out_dir: Path, eval_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_eval = max(1, int(len(episodes) * eval_fraction)) if len(episodes) > 1 else 0
    splits = {"eval": episodes[:n_eval], "train": episodes[n_eval:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    meta = {"episodes": len(episodes), "train": len(splits["train"]), "eval": len(splits["eval"]), "schema": {"steps": [{"input_ids": "list[int]", "labels": "next-token list[int]"}]}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw text into CREST episodic JSONL")
    parser.add_argument("--input", required=True, help="Raw text file, JSONL with text field, or directory")
    parser.add_argument("--out", required=True, help="Output directory containing train.jsonl/eval.jsonl")
    parser.add_argument("--tokenizer", default="byte", help="byte or HuggingFace tokenizer name, e.g. gpt2")
    parser.add_argument("--episode-steps", type=int, default=16)
    parser.add_argument("--step-length", type=int, default=128)
    parser.add_argument("--stride-tokens", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    episodes = build_episodes(iter_texts(Path(args.input)), tokenizer, args.episode_steps, args.step_length, args.stride_tokens)
    if not episodes:
        raise RuntimeError("No episodes produced; check input path and text length.")
    write_split(episodes, Path(args.out), args.eval_fraction, args.seed)
    print(f"[prepare_text] wrote {len(episodes)} episodes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
