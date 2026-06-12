from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import CRESTConfig
from .layers import CRESTLayer, RMSNorm
from .losses import adaptive_lm_head_loss, chunked_lm_head_loss
from .state import init_state


@dataclass
class CRESTAux:
    gates: list[torch.Tensor]
    state_read_entropy: torch.Tensor
    write_entropy: torch.Tensor
    hidden: torch.Tensor
    final_state: list[torch.Tensor]

    @property
    def gate_mean(self) -> torch.Tensor:
        return torch.stack([g.mean() for g in self.gates]).mean()


def _load_token_perm(path: str, vocab_size: int) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    perm = payload["perm"] if isinstance(payload, dict) else payload
    return perm.to(torch.long)


def _validate_perm(perm: torch.Tensor, vocab_size: int) -> None:
    """A token relabeling must be a bijection on {0, ..., V-1}.

    Composing a label permutation with the matching embedding-row permutation
    is likelihood-invariant for the full-softmax model class; for the adaptive
    head it determines which tokens land in the cheap head cluster, so a
    non-bijective map would silently corrupt the output distribution.
    """
    if perm.ndim != 1 or perm.numel() != vocab_size:
        raise ValueError(f"token permutation must have shape [{vocab_size}], got {tuple(perm.shape)}")
    sorted_vals, _ = torch.sort(perm)
    if not torch.equal(sorted_vals, torch.arange(vocab_size, dtype=perm.dtype)):
        raise ValueError("token permutation is not a bijection on [0, vocab_size)")


class CRESTModel(nn.Module):
    def __init__(self, cfg: CRESTConfig, token_perm: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([CRESTLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

        # --- Token relabeling (tokenizer untouched; internal bijection) ---
        # token_perm[original_id] = frequency rank; token_perm_inv[rank] = original_id.
        if token_perm is None and cfg.token_perm_path:
            token_perm = _load_token_perm(cfg.token_perm_path, cfg.vocab_size)
        if token_perm is not None:
            token_perm = token_perm.to(torch.long)
            _validate_perm(token_perm, cfg.vocab_size)
            perm_inv = torch.empty_like(token_perm)
            perm_inv[token_perm] = torch.arange(cfg.vocab_size, dtype=torch.long)
            self.register_buffer("token_perm", token_perm)
            self.register_buffer("token_perm_inv", perm_inv)
        else:
            self.token_perm = None
            self.token_perm_inv = None
        if cfg.head_type == "adaptive" and self.token_perm is None:
            import warnings

            warnings.warn(
                "head_type='adaptive' without a token permutation: "
                "nn.AdaptiveLogSoftmaxWithLoss assumes token IDs are sorted by "
                "descending frequency (most frequent = 0). Llama 3 IDs are not. "
                "Provide token_perm_path from `python -m crest.cli_vocab_freq` "
                "for the expected speed/quality tradeoff.",
                stacklevel=2,
            )

        # --- Output head factorization ---
        if cfg.head_type == "adaptive":
            # Exact cluster-factored softmax (Grave et al., arXiv:1609.04309):
            #   p(y|h) = p_head(y|h)                      for y in head cluster
            #   p(y|h) = p_head(c_i|h) * p_tail_i(y|P_i h) for y in tail cluster i
            # Sum over y is exactly 1 (each tail softmax normalizes to 1 and is
            # weighted by its cluster symbol's head probability).
            self.lm_head = nn.AdaptiveLogSoftmaxWithLoss(
                cfg.d_model,
                cfg.vocab_size,
                cutoffs=list(int(c) for c in cfg.adaptive_cutoffs),
                div_value=float(cfg.adaptive_div_value),
                head_bias=bool(cfg.adaptive_head_bias),
            )
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.token_embedding.weight

        self._diagnostics_enabled = bool(cfg.compute_attention_diagnostics)
        self.set_diagnostics_enabled(self._diagnostics_enabled)

    # ------------------------------------------------------------------
    # Permutation helpers (no-ops when no permutation is registered)
    # ------------------------------------------------------------------
    def _map_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.token_perm is None:
            return input_ids
        return self.token_perm[input_ids]

    def _map_labels(self, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        if self.token_perm is None:
            return labels
        # Never remap the ignore sentinel: clamp for a safe gather, then
        # restore -100 where it was. torch.where keeps this fully vectorized.
        remapped = self.token_perm[labels.clamp_min(0)]
        return torch.where(labels == ignore_index, labels, remapped)

    def _unmap_predictions(self, ranks: torch.Tensor) -> torch.Tensor:
        if self.token_perm_inv is None:
            return ranks
        return self.token_perm_inv[ranks]

    def _head_autocast_off(self, hidden: torch.Tensor):
        """fp32 region for the adaptive head under autocast.

        log-softmax over a >=16k head cluster in fp16 risks overflow/underflow;
        the head is ~10% of total FLOPs after factorization, so fp32 is cheap.
        """
        return torch.autocast(device_type=hidden.device.type, enabled=False)

    # ------------------------------------------------------------------
    # Head API: loss / log-probs / target log-probs / argmax predictions.
    # All label/prediction tensors at this boundary are in ORIGINAL token-id
    # space; rank space is an internal detail.
    # ------------------------------------------------------------------
    def head_loss(self, hidden: torch.Tensor, labels: torch.Tensor, *, ce_chunk_size: int = 0) -> torch.Tensor:
        """Mean NLL over valid (label != -100) positions; 0 if none are valid."""
        labels = self._map_labels(labels)
        if self.cfg.head_type == "adaptive":
            with self._head_autocast_off(hidden):
                return adaptive_lm_head_loss(hidden.float(), labels, self.lm_head)
        return chunked_lm_head_loss(hidden, labels, self.lm_head, chunk_size=ce_chunk_size)

    def head_log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Exact log p(. | h) over the full vocabulary, original-id order.

        O(V d) per position for both head types — use only when the full
        distribution is genuinely required (e.g. sampling, distillation).
        """
        if self.cfg.head_type == "adaptive":
            with self._head_autocast_off(hidden):
                flat = hidden.reshape(-1, hidden.size(-1)).float()
                log_probs = self.lm_head.log_prob(flat)
            log_probs = log_probs.view(*hidden.shape[:-1], self.cfg.vocab_size)
        else:
            log_probs = torch.log_softmax(self.lm_head(hidden), dim=-1)
        if self.token_perm is not None:
            # column v (original id) lives at rank column token_perm[v]
            log_probs = log_probs[..., self.token_perm]
        return log_probs

    def head_target_log_prob(self, hidden: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """log p(label | h) per position; exactly 0.0 where label == -100.

        For the adaptive head this touches only the head cluster plus the
        target's own tail cluster — O(d * (cutoff_0 + n_clusters)) expected —
        never the full vocabulary.
        """
        labels = self._map_labels(labels, ignore_index)
        flat_labels = labels.reshape(-1)
        valid = flat_labels != ignore_index
        out = torch.zeros(flat_labels.shape, device=hidden.device, dtype=torch.float32)
        if torch.any(valid):
            flat_hidden = hidden.reshape(-1, hidden.size(-1))[valid]
            if self.cfg.head_type == "adaptive":
                with self._head_autocast_off(hidden):
                    out_valid = self.lm_head(flat_hidden.float(), flat_labels[valid]).output
            else:
                logits = self.lm_head(flat_hidden)
                out_valid = torch.log_softmax(logits.float(), dim=-1).gather(-1, flat_labels[valid].unsqueeze(-1)).squeeze(-1)
            out = out.masked_scatter(valid, out_valid.to(out.dtype))
        return out.view(labels.shape)

    def head_eval(self, hidden: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (target_log_prob, argmax_prediction) per position.

        target_log_prob is exactly 0.0 at ignored positions; predictions are
        in original token-id space. For the full head this uses a single
        lm_head matmul for both quantities. For the adaptive head the loss
        side touches only head + target clusters; predict touches the head
        plus every tail cluster once per call (still ~5x cheaper than the
        dense 128k head at the default cutoffs).
        """
        if self.cfg.head_type == "adaptive":
            return self.head_target_log_prob(hidden, labels, ignore_index), self.head_predict(hidden)
        mapped = self._map_labels(labels, ignore_index)
        logits = self.lm_head(hidden)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        lp = log_probs.gather(-1, mapped.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        lp = torch.where(mapped == ignore_index, torch.zeros_like(lp), lp)
        return lp, self._unmap_predictions(logits.argmax(dim=-1))

    def head_predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Argmax token id (original-id space) per position."""
        if self.cfg.head_type == "adaptive":
            with self._head_autocast_off(hidden):
                flat = hidden.reshape(-1, hidden.size(-1)).float()
                pred = self.lm_head.predict(flat).view(hidden.shape[:-1])
        else:
            pred = self.lm_head(hidden).argmax(dim=-1)
        return self._unmap_predictions(pred)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        self._diagnostics_enabled = enabled
        for layer in self.layers:
            layer._propagate_diagnostics(enabled)

    def init_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
        return init_state(self.cfg.n_layers, batch_size, self.cfg.memory_slots, self.cfg.d_model, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: list[torch.Tensor] | None = None,
        step_idx: torch.Tensor | int = 0,
        labels: torch.Tensor | None = None,
        ce_chunk_size: int = 0,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], CRESTAux]:
        """Run one CREST step.

        Returns, as the first tuple element:
          * labels given           -> scalar mean NLL over valid positions
          * labels None, full head -> logits [B, L, V] (original-id order)
          * labels None, adaptive  -> exact log-probs [B, L, V] (original-id
                                      order; O(V d) — prefer head_* methods)
          * return_logits=False    -> None (use aux.hidden + head_* methods)
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")
        b = input_ids.size(0)
        if state is None:
            state = self.init_state(b, device=input_ids.device, dtype=self.token_embedding.weight.dtype)
        if len(state) != self.cfg.n_layers:
            raise ValueError("state length must equal n_layers")
        if isinstance(step_idx, int):
            step_idx = torch.full((b,), step_idx, device=input_ids.device, dtype=torch.long)
        elif step_idx.ndim == 0:
            step_idx = step_idx.expand(b).to(device=input_ids.device, dtype=torch.long)
        else:
            step_idx = step_idx.to(device=input_ids.device, dtype=torch.long)
        x = self.token_embedding(self._map_input_ids(input_ids))
        next_state: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        read_entropies = []
        write_entropies = []
        for layer, layer_state in zip(self.layers, state, strict=True):
            x, s_next, aux = layer(x, layer_state, step_idx)
            next_state.append(s_next)
            gates.append(aux.gate)
            read_entropies.append(aux.state_read_entropy)
            write_entropies.append(aux.write_entropy)
        hidden = self.norm(x)
        aux_out = CRESTAux(
            gates=gates,
            state_read_entropy=torch.stack(read_entropies).mean(),
            write_entropy=torch.stack(write_entropies).mean(),
            hidden=hidden,
            final_state=next_state,
        )
        if labels is not None:
            loss = self.head_loss(hidden, labels, ce_chunk_size=ce_chunk_size)
            return loss, next_state, aux_out
        if not return_logits:
            return None, next_state, aux_out
        if self.cfg.head_type == "adaptive":
            return self.head_log_probs(hidden), next_state, aux_out
        logits = self.lm_head(hidden)
        if self.token_perm is not None:
            logits = logits[..., self.token_perm]
        return logits, next_state, aux_out

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        state: list[torch.Tensor] | None = None,
        step_idx: torch.Tensor | int = 0,
        *,
        ce_chunk_size: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], CRESTAux]:
        return self(input_ids, state=state, step_idx=step_idx, labels=labels, ce_chunk_size=ce_chunk_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
