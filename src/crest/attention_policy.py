from __future__ import annotations

from contextlib import nullcontext

import torch


class AttentionBackendPolicy:
    """PyTorch SDPA backend policy for exact attention.

    Citation: FlashAttention, arXiv:2205.14135, is exact tiled attention, not an
    approximation; see docs/suite/2205.14135v2 lines 90-110 and 149-181. This
    policy only selects exact SDPA kernels. It never changes the mathematical
    attention operator softmax(QK^T/sqrt(d_k))V.
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "math", "flash", "mem_efficient"}:
            raise ValueError("attention backend must be auto|math|flash|mem_efficient")
        self.backend = backend

    def context(self):
        if self.backend == "auto" or not torch.cuda.is_available():
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            mapping = {
                "math": SDPBackend.MATH,
                "flash": SDPBackend.FLASH_ATTENTION,
                "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            }
            return sdpa_kernel(mapping[self.backend])
        except Exception:
            return nullcontext()
