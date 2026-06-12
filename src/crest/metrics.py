from __future__ import annotations

from .config import CRESTConfig
from .model import CRESTModel


def count_parameters(model: CRESTModel) -> int:
    return sum(p.numel() for p in model.parameters())


def component_parameter_counts(model: CRESTModel) -> dict[str, int]:
    groups = {"embedding": 0, "local_attention": 0, "state_read": 0, "state_write": 0, "gate_mlp": 0, "ffn": 0, "norm": 0, "lm_head": 0, "other": 0}
    for name, param in model.named_parameters():
        n = param.numel()
        if "token_embedding" in name:
            groups["embedding"] += n
        elif "local_attn" in name:
            groups["local_attention"] += n
        elif "state_read" in name:
            groups["state_read"] += n
        elif ".write.gate" in name:
            groups["gate_mlp"] += n
        elif ".write" in name:
            groups["state_write"] += n
        elif "ffn" in name:
            groups["ffn"] += n
        elif "norm" in name:
            groups["norm"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        else:
            groups["other"] += n
    return groups


def estimate_lm_head_flops(cfg: CRESTConfig, tokens: int) -> int:
    """Forward FLOPs of the output head for `tokens` supervised positions.

    full head:  2 * tokens * d * V exactly (one dense matmul row per token).

    adaptive head (Grave et al., arXiv:1609.04309), expectation per token:
        2d(k_h + n_clusters) + sum_i p_i * (2 d d_i + 2 d_i |V_i|)
    where k_h = cutoffs[0], d_i = floor(d / div_value^(i+1)), |V_i| is the
    cluster size, and p_i is the probability the target falls in tail cluster
    i. p_i comes from cfg.adaptive_cluster_probs (measured by
    crest.cli_vocab_freq); without measurements we use the conservative
    fallback p_i = |V_i| / V... which UNDERSTATES the savings on Zipfian data
    but never overstates them, keeping the startup report honest.
    """
    d = cfg.d_model
    if cfg.head_type != "adaptive":
        return 2 * tokens * d * cfg.vocab_size
    cutoffs = [int(c) for c in cfg.adaptive_cutoffs]
    edges = cutoffs + [cfg.vocab_size]
    n_clusters = len(edges) - 1
    per_token = 2.0 * d * (cutoffs[0] + n_clusters)
    if cfg.adaptive_cluster_probs is not None:
        probs = [float(p) for p in cfg.adaptive_cluster_probs]
    else:
        probs = [(edges[i + 1] - edges[i]) / cfg.vocab_size for i in range(n_clusters)]
    for i in range(n_clusters):
        cluster_size = edges[i + 1] - edges[i]
        d_i = max(1, int(d / (cfg.adaptive_div_value ** (i + 1))))
        per_token += probs[i] * (2.0 * d * d_i + 2.0 * d_i * cluster_size)
    return int(per_token * tokens)


def estimate_step_flops(cfg: CRESTConfig) -> dict[str, int]:
    """Estimate forward FLOPs for one episode step across all CREST layers.

    Counts every dense matmul in the layer graph using the standard
    "2 * output * input" multiply-accumulate convention (one MAC = 2 FLOPs):
      * local self-attention   = Q/K/V/O projections + QK^T + softmax @ V
      * state read             = Q on tokens, K/V on state, O projection,
                                 plus the cross-attention matmuls
      * state write            = Q on state, K/V on tokens, plus the two
                                 attention matmuls (StateWriter has no O proj)
      * fuse                   = Linear(2d, d) over L tokens
      * gate MLP               = Linear(4d, d) + Linear(d, d) over M slots
                                 (= 10 M d^2)
      * FFN SwiGLU             = three (d <-> d_ffn) projections over L tokens
      * LM head                = (L, d) -> (L, V) projection (once per step)

    Element-wise ops, RMSNorms, residuals, gating sigmoids, embedding lookups,
    and softmax exponentials are intentionally omitted; they are negligible
    next to the matmul terms above for any non-toy d.
    """
    l, m, d = cfg.max_seq_len, cfg.memory_slots, cfg.d_model
    d_ffn = cfg.d_ffn
    # Local self-attention: 4 * 2 L d^2 (Q,K,V,O) + 2 * 2 L^2 d (scores + AV).
    per_layer_local = 8 * l * d * d + 4 * l * l * d
    # State read: Q (L tokens) and O (L tokens) cost 2 * 2 L d^2; K and V on M
    # state slots cost 2 * 2 M d^2; cross-attention matmuls cost 2 * 2 L M d.
    per_layer_state_read = 4 * l * d * d + 4 * m * d * d + 4 * l * m * d
    # State write: Q on M slots = 2 M d^2; K and V on L tokens = 2 * 2 L d^2;
    # cross-attention matmuls = 2 * 2 M L d. No output projection.
    per_layer_state_write = 2 * m * d * d + 4 * l * d * d + 4 * m * l * d
    # Fusion linear: Linear(2d, d) applied to L tokens => 2 * L * 2d * d.
    per_layer_fuse = 4 * l * d * d
    # Gate MLP per slot: Linear(4d, d) + Linear(d, d) => 2*M*4d*d + 2*M*d*d.
    per_layer_gate = 10 * m * d * d
    # FFN SwiGLU: three (d <-> d_ffn) projections over L tokens.
    per_layer_ffn = 6 * l * d * d_ffn
    local = cfg.n_layers * per_layer_local
    state_read = cfg.n_layers * per_layer_state_read
    state_write = cfg.n_layers * per_layer_state_write
    fuse = cfg.n_layers * per_layer_fuse
    gate = cfg.n_layers * per_layer_gate
    ffn = cfg.n_layers * per_layer_ffn
    # LM head is applied once per step on L tokens. For the adaptive head this
    # is the EXPECTED cost under the configured/measured cluster probabilities;
    # see estimate_lm_head_flops for the formula and its conservative fallback.
    logits = estimate_lm_head_flops(cfg, l)
    total = local + state_read + state_write + fuse + gate + ffn + logits
    return {
        "local_attention": local,
        "state_read": state_read,
        "state_write": state_write,
        "fuse": fuse,
        "gate_mlp": gate,
        "ffn": ffn,
        "lm_head": logits,
        "total": total,
    }


def estimate_episode_flops(cfg: CRESTConfig, episode_steps: int, backward: bool = False) -> dict[str, int]:
    step = estimate_step_flops(cfg)
    multiplier = episode_steps * (3 if backward else 1)
    return {k: v * multiplier for k, v in step.items()}
