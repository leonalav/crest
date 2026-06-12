from __future__ import annotations

"""Phase-1 vocabulary frequency audit for the adaptive softmax head.

Counts supervised target-token frequencies over a prepared episodic dataset,
derives the frequency permutation required by nn.AdaptiveLogSoftmaxWithLoss
(most frequent token -> rank 0), and reports cumulative head-coverage so
adaptive_cutoffs can be chosen from MEASURED mass instead of Zipf folklore.

The tokenizer is never modified. The permutation is an internal bijective
relabeling pi: original_id -> rank; CRESTModel applies pi to inputs/labels and
pi^{-1} to predictions, which is likelihood-invariant for the model class.

Counting labels (not inputs) is deliberate: the head cost is paid per
SUPERVISED token, so the partition must match the label distribution. For
standard next-token data the two distributions differ by one token per step.

Usage:
  PYTHONPATH=src python -m crest.cli_vocab_freq \
    --data data/episodic_arrow/default \
    --vocab-size 128256 \
    --out data/episodic_arrow/default/token_freq.pt \
    --candidate-cutoffs 4096,8192,16384,32768,65536

Output .pt payload (torch.save):
  perm        LongTensor [V]: perm[original_id] = frequency rank
  perm_inv    LongTensor [V]: perm_inv[rank] = original_id
  counts      LongTensor [V]: raw label counts, original-id order
  total       int: total supervised tokens counted
  coverage    dict cutoff -> cumulative probability mass of top-`cutoff` ranks
"""

import argparse
import json
from pathlib import Path

import torch


def count_label_frequencies(data_path: Path, vocab_size: int, split: str = "train", max_episodes: int | None = None) -> tuple[torch.Tensor, int]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Frequency audit requires `pip install datasets`.") from exc
    path = data_path
    if path.is_dir() and (path / split).exists():
        path = path / split
    if not path.exists():
        raise FileNotFoundError(f"Episodic Arrow dataset not found at '{path}'")
    ds = load_from_disk(str(path))
    counts = torch.zeros(vocab_size, dtype=torch.long)
    n = len(ds) if max_episodes is None else min(len(ds), max_episodes)
    episodes_seen = 0
    for i in range(n):
        row = ds[int(i)]
        for step in row["steps"]:
            labels = torch.as_tensor(step["labels"] if "labels" in step else step["input_ids"][1:], dtype=torch.long)
            valid = labels[(labels >= 0) & (labels < vocab_size)]
            if valid.numel():
                counts.scatter_add_(0, valid, torch.ones_like(valid))
        episodes_seen += 1
        if episodes_seen % 5000 == 0:
            print(f"[freq] {episodes_seen}/{n} episodes", flush=True)
    return counts, episodes_seen


def build_permutation(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """perm[original_id] = rank, stable so ties keep original-id order.

    torch.argsort(descending=True, stable=True) on counts gives
    perm_inv[rank] = original_id; invert for perm. Bijectivity holds by
    construction (argsort of V distinct positions is a permutation).
    """
    perm_inv = torch.argsort(counts, descending=True, stable=True)
    perm = torch.empty_like(perm_inv)
    perm[perm_inv] = torch.arange(perm_inv.numel(), dtype=torch.long)
    return perm, perm_inv


def coverage_table(counts: torch.Tensor, cutoffs: list[int]) -> dict[int, float]:
    total = int(counts.sum().item())
    if total == 0:
        raise ValueError("no supervised tokens counted; check --data path and split")
    sorted_counts, _ = torch.sort(counts, descending=True)
    cumulative = torch.cumsum(sorted_counts, dim=0)
    return {c: float(cumulative[min(c, counts.numel()) - 1].item()) / total for c in cutoffs}


def expected_head_flops(d_model: int, vocab_size: int, cutoffs: list[int], coverage: dict[int, float], div_value: float = 4.0) -> dict[str, float]:
    """E[FLOPs/token] = 2d(k_h + n_clusters) + sum_i p_i (2 d d_i + 2 d_i |V_i|).

    Head matmul is always paid; tail cluster i is paid only when the target
    lands there (probability p_i from measured coverage). 1 MAC = 2 FLOPs.
    """
    k_h = cutoffs[0]
    edges = cutoffs + [vocab_size]
    n_clusters = len(edges) - 1
    flops = 2.0 * d_model * (k_h + n_clusters)
    detail = {"head": flops}
    prev_cov = coverage[k_h]
    for i in range(n_clusters):
        lo, hi = edges[i], edges[i + 1]
        cov_hi = coverage.get(hi, 1.0) if hi < vocab_size else 1.0
        p_i = max(0.0, cov_hi - prev_cov)
        prev_cov = cov_hi
        d_i = max(1, int(d_model / (div_value ** (i + 1))))
        cluster_flops = p_i * (2.0 * d_model * d_i + 2.0 * d_i * (hi - lo))
        detail[f"tail_{i}"] = cluster_flops
        flops += cluster_flops
    detail["total"] = flops
    detail["full_head_total"] = 2.0 * d_model * vocab_size
    detail["reduction_factor"] = detail["full_head_total"] / flops
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="CREST vocabulary frequency audit (Phase 1, adaptive softmax)")
    parser.add_argument("--data", required=True, help="Path to prepared episodic Arrow dataset directory")
    parser.add_argument("--vocab-size", type=int, required=True, help="Tokenizer vocab size (e.g. 128256 for Llama 3)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, help="Output .pt path for the permutation payload")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for a faster approximate audit")
    parser.add_argument("--candidate-cutoffs", default="4096,8192,16384,32768,65536", help="Comma-separated cutoff candidates for the coverage report")
    parser.add_argument("--d-model", type=int, default=256, help="Used only for the expected-FLOP report")
    parser.add_argument("--div-value", type=float, default=4.0)
    parser.add_argument("--report-cutoffs", default=None, help="Comma-separated cutoffs (e.g. 16384,49152) for a 2-cluster FLOP estimate; defaults to best single candidate split")
    args = parser.parse_args()

    candidates = sorted({int(c) for c in args.candidate_cutoffs.split(",") if c.strip()})
    counts, episodes = count_label_frequencies(Path(args.data), args.vocab_size, args.split, args.max_episodes)
    total = int(counts.sum().item())
    perm, perm_inv = build_permutation(counts)
    coverage = coverage_table(counts, candidates)
    nonzero = int((counts > 0).sum().item())

    print(f"[freq] episodes={episodes} supervised_tokens={total} distinct_tokens_seen={nonzero}/{args.vocab_size}")
    print("[freq] cumulative coverage (top-k ranks):")
    for c in candidates:
        print(f"  top {c:>7,}: {coverage[c]:.5f}")

    if args.report_cutoffs:
        cutoffs = sorted({int(c) for c in args.report_cutoffs.split(",") if c.strip()})
    else:
        head = next((c for c in candidates if coverage[c] >= 0.90), candidates[-1])
        cutoffs = [head, min(args.vocab_size - 1, head * 3)]
    cov_for_cutoffs = coverage_table(counts, cutoffs)
    flops = expected_head_flops(args.d_model, args.vocab_size, cutoffs, cov_for_cutoffs, args.div_value)
    print(f"[freq] suggested adaptive_cutoffs: {cutoffs} (head coverage {cov_for_cutoffs[cutoffs[0]]:.4f})")
    print(f"[freq] expected head FLOPs/token: {flops['total'] / 1e6:.2f}M vs full {flops['full_head_total'] / 1e6:.2f}M "
          f"-> {flops['reduction_factor']:.1f}x head reduction (d_model={args.d_model}, div_value={args.div_value})")
    if cov_for_cutoffs[cutoffs[0]] < 0.90:
        print("[WARNING] head coverage < 0.90: measured distribution is flatter than assumed. "
              "Raise the first cutoff or expect a smaller speedup (Volkov gate from the plan).")

    # Tail-cluster probabilities for honest FLOP reporting in metrics.py.
    edges = cutoffs + [args.vocab_size]
    cluster_probs = []
    prev = cov_for_cutoffs[cutoffs[0]]
    for hi in edges[1:]:
        cov_hi = coverage_table(counts, [hi])[hi] if hi < args.vocab_size else 1.0
        cluster_probs.append(max(0.0, cov_hi - prev))
        prev = cov_hi

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "perm": perm,
            "perm_inv": perm_inv,
            "counts": counts,
            "total": total,
            "coverage": {int(k): float(v) for k, v in coverage.items()},
            "suggested_cutoffs": cutoffs,
            "suggested_cluster_probs": cluster_probs,
            "vocab_size": args.vocab_size,
            "split": args.split,
            "episodes": episodes,
        },
        out_path,
    )
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "supervised_tokens": total,
                "distinct_tokens": nonzero,
                "coverage": {str(k): v for k, v in coverage.items()},
                "suggested_cutoffs": cutoffs,
                "suggested_cluster_probs": cluster_probs,
                "expected_head_flops_per_token": flops,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[freq] wrote {out_path} and {summary_path}")
    print("[freq] model YAML additions:")
    print("  head_type: adaptive")
    print(f"  adaptive_cutoffs: {cutoffs}")
    print(f"  adaptive_cluster_probs: {[round(p, 5) for p in cluster_probs]}")
    print(f"  token_perm_path: {out_path}")
    print("  tie_embeddings: false")


if __name__ == "__main__":
    main()
