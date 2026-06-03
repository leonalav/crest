from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistributedInfo:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedInfo:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistributedInfo(enabled=False)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return DistributedInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def maybe_wrap_fsdp(model: torch.nn.Module, use_fsdp: bool) -> torch.nn.Module:
    if not use_fsdp:
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return FSDP(model)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
