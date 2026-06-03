from __future__ import annotations

from dataclasses import replace

from .config import CRESTConfig


def no_state_config(cfg: CRESTConfig) -> CRESTConfig:
    return replace(cfg, use_state_read=False, use_state_write=False)


def memory_sweep_configs(cfg: CRESTConfig, memory_slots: list[int]) -> dict[str, CRESTConfig]:
    return {f"M{m}": replace(cfg, memory_slots=m) for m in memory_slots}


def local_only_config(cfg: CRESTConfig) -> CRESTConfig:
    return replace(cfg, use_state_read=False, use_state_write=False)
