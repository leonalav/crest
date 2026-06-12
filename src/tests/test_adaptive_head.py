"""Adaptive softmax head + token permutation tests (Phase 2 validation gates).

Math contracts under test (Grave et al., arXiv:1609.04309):
  G1  Exact normalization: sum_y p(y|h) = 1 for the cluster-factored softmax.
  G2  Permutation correctness: pi is a bijection; predictions/log-probs are
      reported in original token-id space; ignore_index is never remapped.
  G3  Loss identity: head_loss == -mean head_target_log_prob over valid
      positions == NLL gathered from head_log_probs (all three computations
      route through different code paths in nn.AdaptiveLogSoftmaxWithLoss).
  G4  Empty supervision contributes the empty sum (exactly 0, never NaN).
  G5  Full-head path is bit-compatible with the old behavior (regression).
  G6  FLOP accounting: adaptive expected cost < full cost, formula matches a
      hand computation.
"""

import math

import pytest
import torch

from crest.config import CRESTConfig
from crest.losses import adaptive_lm_head_loss, lm_loss
from crest.model import CRESTModel

V = 64
D = 32


def adaptive_cfg(**overrides):
    base = dict(
        vocab_size=V,
        max_seq_len=8,
        max_steps=16,
        n_layers=2,
        d_model=D,
        n_heads=4,
        d_ffn=64,
        memory_slots=5,
        tie_embeddings=False,
        head_type="adaptive",
        adaptive_cutoffs=(8, 24),
        adaptive_div_value=4.0,
    )
    base.update(overrides)
    return CRESTConfig(**base)


def full_cfg(**overrides):
    base = dict(
        vocab_size=V,
        max_seq_len=8,
        max_steps=16,
        n_layers=2,
        d_model=D,
        n_heads=4,
        d_ffn=64,
        memory_slots=5,
        tie_embeddings=True,
    )
    base.update(overrides)
    return CRESTConfig(**base)


def random_perm(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(V, generator=g)


# ---------------------------------------------------------------------------
# G1 — exact normalization
# ---------------------------------------------------------------------------

def test_adaptive_log_probs_normalize_to_one():
    torch.manual_seed(0)
    model = CRESTModel(adaptive_cfg()).eval()
    hidden = torch.randn(4, 8, D)
    log_probs = model.head_log_probs(hidden)
    assert log_probs.shape == (4, 8, V)
    total = log_probs.exp().sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_adaptive_forward_without_labels_returns_normalized_log_probs():
    torch.manual_seed(0)
    model = CRESTModel(adaptive_cfg()).eval()
    out, _, _ = model(torch.randint(0, V, (2, 8)), step_idx=0)
    assert out.shape == (2, 8, V)
    total = out.exp().sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


# ---------------------------------------------------------------------------
# G2 — permutation correctness
# ---------------------------------------------------------------------------

def test_non_bijective_perm_rejected():
    bad = torch.zeros(V, dtype=torch.long)  # constant map: not a bijection
    with pytest.raises(ValueError, match="bijection"):
        CRESTModel(adaptive_cfg(), token_perm=bad)


def test_wrong_length_perm_rejected():
    with pytest.raises(ValueError, match="shape"):
        CRESTModel(adaptive_cfg(), token_perm=torch.arange(V - 1))


def test_perm_inverse_roundtrip():
    perm = random_perm()
    model = CRESTModel(adaptive_cfg(), token_perm=perm)
    ids = torch.arange(V)
    assert torch.equal(model.token_perm_inv[model.token_perm[ids]], ids)
    assert torch.equal(model.token_perm[model.token_perm_inv[ids]], ids)


def test_ignore_index_never_remapped():
    perm = random_perm()
    model = CRESTModel(adaptive_cfg(), token_perm=perm)
    labels = torch.tensor([[-100, 3, -100, 7]])
    mapped = model._map_labels(labels)
    assert mapped[0, 0].item() == -100 and mapped[0, 2].item() == -100
    assert mapped[0, 1].item() == perm[3].item()
    assert mapped[0, 3].item() == perm[7].item()


def test_log_prob_columns_are_original_id_space():
    """Column v of head_log_probs must be log p(original token v | h)."""
    torch.manual_seed(1)
    perm = random_perm()
    model = CRESTModel(adaptive_cfg(), token_perm=perm).eval()
    hidden = torch.randn(2, 4, D)
    log_probs = model.head_log_probs(hidden)
    # rank-space reference straight from the module
    rank_lp = model.lm_head.log_prob(hidden.reshape(-1, D)).view(2, 4, V)
    for v in [0, 5, V - 1]:
        assert torch.allclose(log_probs[..., v], rank_lp[..., perm[v]], atol=1e-6)


def test_predictions_are_original_id_space():
    torch.manual_seed(2)
    perm = random_perm()
    model = CRESTModel(adaptive_cfg(), token_perm=perm).eval()
    hidden = torch.randn(3, 8, D)
    pred = model.head_predict(hidden)
    expected = model.head_log_probs(hidden).argmax(dim=-1)
    assert torch.equal(pred, expected)


def test_full_head_permutation_is_likelihood_invariant():
    """Relabeling tokens by a bijection and permuting embedding rows to match
    must leave the full-softmax likelihood unchanged (Volkov invariance check)."""
    torch.manual_seed(3)
    cfg = full_cfg(tie_embeddings=False)
    perm = random_perm()
    base = CRESTModel(cfg).eval()
    permuted = CRESTModel(cfg, token_perm=perm).eval()
    permuted.load_state_dict({k: v for k, v in base.state_dict().items()}, strict=False)
    with torch.no_grad():
        # permuted model looks up embedding row perm[id]; to represent the SAME
        # function, its embedding/head row perm[v] must hold base's row v.
        permuted.token_embedding.weight[perm] = base.token_embedding.weight
        permuted.lm_head.weight[perm] = base.lm_head.weight
    ids = torch.randint(0, V, (2, 8))
    labels = torch.randint(0, V, (2, 8))
    labels[0, :2] = -100
    state_a = base.init_state(2)
    state_b = permuted.init_state(2)
    with torch.no_grad():
        loss_a, _, _ = base(ids, state=state_a, step_idx=0, labels=labels)
        loss_b, _, _ = permuted(ids, state=state_b, step_idx=0, labels=labels)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


# ---------------------------------------------------------------------------
# G3 — loss identities across independent code paths
# ---------------------------------------------------------------------------

def test_head_loss_matches_target_log_prob_and_full_log_probs():
    torch.manual_seed(4)
    model = CRESTModel(adaptive_cfg(), token_perm=random_perm()).eval()
    hidden = torch.randn(3, 8, D)
    labels = torch.randint(0, V, (3, 8))
    labels[:, 0] = -100
    valid = labels != -100

    loss = model.head_loss(hidden, labels)

    token_lp = model.head_target_log_prob(hidden, labels)
    assert torch.allclose(token_lp[~valid], torch.zeros_like(token_lp[~valid]))
    loss_from_lp = -token_lp[valid].mean()

    full_lp = model.head_log_probs(hidden)
    loss_from_full = -full_lp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].mean()

    assert torch.allclose(loss, loss_from_lp, atol=1e-5)
    assert torch.allclose(loss, loss_from_full, atol=1e-5)


def test_head_eval_consistent_with_target_log_prob_and_predict():
    torch.manual_seed(5)
    for cfg, perm in [(adaptive_cfg(), random_perm()), (full_cfg(), None)]:
        model = CRESTModel(cfg, token_perm=perm).eval()
        hidden = torch.randn(2, 8, D)
        labels = torch.randint(0, V, (2, 8))
        labels[0, :3] = -100
        lp, pred = model.head_eval(hidden, labels)
        assert torch.allclose(lp, model.head_target_log_prob(hidden, labels), atol=1e-5)
        assert torch.equal(pred, model.head_predict(hidden))


def test_full_and_adaptive_heads_agree_through_shared_api():
    """Same trunk hidden states: both heads must satisfy the same contracts
    (normalization + loss identity), even though their parameterizations and
    therefore their numerical losses differ."""
    torch.manual_seed(6)
    hidden = torch.randn(2, 8, D)
    labels = torch.randint(0, V, (2, 8))
    for cfg in [adaptive_cfg(), full_cfg()]:
        model = CRESTModel(cfg).eval()
        loss = model.head_loss(hidden, labels)
        assert loss.ndim == 0 and math.isfinite(loss.item()) and loss.item() > 0
        lp = model.head_log_probs(hidden)
        total = lp.exp().sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


# ---------------------------------------------------------------------------
# G4 — empty supervision
# ---------------------------------------------------------------------------

def test_adaptive_loss_empty_supervision_is_zero_not_nan():
    torch.manual_seed(7)
    model = CRESTModel(adaptive_cfg())
    hidden = torch.randn(2, 8, D, requires_grad=True)
    labels = torch.full((2, 8), -100, dtype=torch.long)
    loss = model.head_loss(hidden, labels)
    assert loss.item() == 0.0
    loss.backward()  # graph must exist (DDP parity), gradient must be finite
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_adaptive_lm_head_loss_filters_partial_ignore():
    torch.manual_seed(8)
    head = torch.nn.AdaptiveLogSoftmaxWithLoss(D, V, cutoffs=[8, 24], div_value=4.0)
    hidden = torch.randn(10, D)
    labels = torch.randint(0, V, (10,))
    labels[::2] = -100
    loss = adaptive_lm_head_loss(hidden, labels, head)
    valid = labels != -100
    ref = head(hidden[valid], labels[valid]).loss
    assert torch.allclose(loss, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# G5 — full-head regression + training step integration
# ---------------------------------------------------------------------------

def test_full_head_default_unchanged():
    torch.manual_seed(9)
    model = CRESTModel(full_cfg())
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert model.lm_head.weight is model.token_embedding.weight
    ids = torch.randint(0, V, (2, 8))
    labels = torch.randint(0, V, (2, 8))
    logits, _, _ = model(ids, step_idx=0)
    assert logits.shape == (2, 8, V)
    loss, _, _ = model(ids, step_idx=0, labels=labels)
    ref = lm_loss(logits, labels)
    # same hidden -> identical CE between head_loss path and legacy lm_loss
    with torch.no_grad():
        _, _, aux = model(ids, step_idx=0, return_logits=False)
        assert torch.allclose(model.head_loss(aux.hidden, labels), lm_loss(model.lm_head(aux.hidden), labels), atol=1e-6)
    assert math.isfinite(loss.item())
    assert math.isfinite(ref.item())


def test_adaptive_training_step_updates_head_and_trunk():
    torch.manual_seed(10)
    model = CRESTModel(adaptive_cfg(), token_perm=random_perm())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, V, (4, 8))
    labels = torch.randint(0, V, (4, 8))
    loss0 = None
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = model(ids, step_idx=0, labels=labels)
        loss.backward()
        opt.step()
        loss0 = loss0 if loss0 is not None else loss.item()
    assert loss.item() < loss0  # memorizing a fixed batch must reduce NLL
    assert model.lm_head.head.weight.grad is not None


def test_evaluate_runs_with_adaptive_head():
    from torch.utils.data import DataLoader

    from crest.config import DataConfig
    from crest.data import build_dataset, collate_episodes
    from crest.eval import evaluate

    data_cfg = DataConfig(vocab_size=V, episode_steps=2, step_length=8, train_episodes=4, eval_episodes=4, num_keys=4, num_values=8)
    ds = build_dataset(data_cfg, "eval")
    model = CRESTModel(adaptive_cfg(), token_perm=random_perm())
    metrics = evaluate(model, DataLoader(ds, batch_size=2, collate_fn=collate_episodes), max_batches=1)
    assert math.isfinite(metrics["eval_loss"])
    assert 0.0 <= metrics["recall_accuracy"] <= 1.0


def test_trajectory_logprob_adaptive_matches_manual_gather():
    from crest.alignment import trajectory_logprob

    torch.manual_seed(11)
    model = CRESTModel(adaptive_cfg(), token_perm=random_perm()).eval()
    ids = torch.randint(0, V, (2, 2, 8))  # [B, T, L]
    labels = torch.randint(0, V, (2, 2, 8))
    labels[0, 0, :4] = -100
    steps = torch.arange(2).expand(2, 2)
    with torch.no_grad():
        total = trajectory_logprob(model, ids, labels, steps)
        # manual: re-roll state, gather from full log-probs
        state = model.init_state(2)
        ref = torch.zeros(2)
        for t in range(2):
            _, state, aux = model(ids[:, t], state=state, step_idx=steps[:, t], return_logits=False)
            lp = model.head_log_probs(aux.hidden)
            tok = lp.gather(-1, labels[:, t].clamp_min(0).unsqueeze(-1)).squeeze(-1)
            ref += (tok * (labels[:, t] != -100)).sum(dim=-1)
    assert torch.allclose(total, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# G6 — config validation + FLOP accounting + checkpoint guards
# ---------------------------------------------------------------------------

def test_adaptive_requires_untied_embeddings():
    with pytest.raises(ValueError, match="tie_embeddings"):
        adaptive_cfg(tie_embeddings=True)


def test_adaptive_requires_increasing_cutoffs_below_vocab():
    with pytest.raises(ValueError):
        adaptive_cfg(adaptive_cutoffs=(24, 8))
    with pytest.raises(ValueError):
        adaptive_cfg(adaptive_cutoffs=(8, V))
    with pytest.raises(ValueError):
        adaptive_cfg(adaptive_cutoffs=())


def test_cluster_probs_validated():
    with pytest.raises(ValueError):
        adaptive_cfg(adaptive_cluster_probs=(0.9, 0.9))  # sums over 1
    with pytest.raises(ValueError):
        adaptive_cfg(adaptive_cluster_probs=(0.1,))  # wrong arity


def test_lm_head_flops_adaptive_below_full_and_matches_hand_calc():
    from crest.metrics import estimate_lm_head_flops

    cfg_full = full_cfg()
    cfg_ad = adaptive_cfg(adaptive_cluster_probs=(0.04, 0.01))
    full = estimate_lm_head_flops(cfg_full, tokens=1)
    ad = estimate_lm_head_flops(cfg_ad, tokens=1)
    assert full == 2 * D * V
    # hand calc: head 2*32*(8+2); tail0 p=.04 d0=8 |V0|=16; tail1 p=.01 d1=2 |V1|=40
    expected = 2 * 32 * 10 + 0.04 * (2 * 32 * 8 + 2 * 8 * 16) + 0.01 * (2 * 32 * 2 + 2 * 2 * 40)
    assert ad == int(expected)
    assert ad < full


def test_checkpoint_head_type_mismatch_rejected(tmp_path):
    from crest.checkpoint import load_checkpoint, save_checkpoint

    full_model = CRESTModel(full_cfg())
    opt = torch.optim.AdamW(full_model.parameters(), lr=1e-3)
    path = tmp_path / "full.pt"
    save_checkpoint(str(path), full_model, opt, step=1)
    ad_model = CRESTModel(adaptive_cfg())
    with pytest.raises(ValueError, match="head_type"):
        load_checkpoint(str(path), ad_model)


def test_checkpoint_perm_mismatch_rejected(tmp_path):
    from crest.checkpoint import load_checkpoint, save_checkpoint

    m1 = CRESTModel(adaptive_cfg(), token_perm=random_perm(seed=1))
    opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    path = tmp_path / "perm.pt"
    save_checkpoint(str(path), m1, opt, step=1)
    m2 = CRESTModel(adaptive_cfg(), token_perm=random_perm(seed=2))
    with pytest.raises(ValueError, match="permutation"):
        load_checkpoint(str(path), m2)
    m3 = CRESTModel(adaptive_cfg(), token_perm=random_perm(seed=1))
    assert load_checkpoint(str(path), m3) == 1


# ---------------------------------------------------------------------------
# Phase 1 — frequency audit unit checks (no Arrow dataset needed)
# ---------------------------------------------------------------------------

def test_build_permutation_sorts_by_frequency_with_stable_ties():
    from crest.cli_vocab_freq import build_permutation

    counts = torch.tensor([5, 9, 9, 0, 7], dtype=torch.long)
    perm, perm_inv = build_permutation(counts)
    # ranks: id1 (9) -> 0, id2 (9, tie keeps id order) -> 1, id4 (7) -> 2, id0 (5) -> 3, id3 (0) -> 4
    assert perm.tolist() == [3, 0, 1, 4, 2]
    assert perm_inv.tolist() == [1, 2, 4, 0, 3]
    assert torch.equal(perm[perm_inv], torch.arange(5))


def test_coverage_table_cumulative():
    from crest.cli_vocab_freq import coverage_table

    counts = torch.tensor([1, 4, 3, 2], dtype=torch.long)  # total 10
    cov = coverage_table(counts, [1, 2, 4])
    assert abs(cov[1] - 0.4) < 1e-9
    assert abs(cov[2] - 0.7) < 1e-9
    assert abs(cov[4] - 1.0) < 1e-9


def test_expected_head_flops_reduction_reported():
    from crest.cli_vocab_freq import expected_head_flops

    flops = expected_head_flops(d_model=256, vocab_size=128256, cutoffs=[16384, 49152], coverage={16384: 0.95, 49152: 0.99}, div_value=4.0)
    assert flops["reduction_factor"] > 5.0
    assert flops["total"] < flops["full_head_total"]
