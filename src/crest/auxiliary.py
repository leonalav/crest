from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StateReconstructionHead(nn.Module):
    """JEPA-style state prediction auxiliary head.

    CREST plan defines L_aux = lambda ||Dec_phi(S_t) - Enc_psi(C_{t-K:t})||_F^2.
    This module implements Dec_phi and a simple target encoder over token states.
    The target path is stop-gradient by default, so the auxiliary loss trains the
    recurrent state to predict recent context summaries rather than collapsing both sides.
    """

    def __init__(self, d_model: int, aux_dim: int) -> None:
        super().__init__()
        self.state_decoder = nn.Linear(d_model, aux_dim)
        self.context_encoder = nn.Linear(d_model, aux_dim)

    def forward(self, state: torch.Tensor, token_hidden: torch.Tensor) -> torch.Tensor:
        pred = self.state_decoder(state).mean(dim=1)
        target = self.context_encoder(token_hidden.detach()).mean(dim=1)
        return F.mse_loss(pred, target)
