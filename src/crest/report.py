from __future__ import annotations

from .config import CRESTConfig
from .metrics import component_parameter_counts, estimate_episode_flops, estimate_step_flops
from .model import CRESTModel


def model_report(model: CRESTModel, cfg: CRESTConfig, episode_steps: int) -> dict[str, int | dict[str, int]]:
    return {
        "total_parameters": model.count_parameters(),
        "component_parameters": component_parameter_counts(model),
        "step_flops_forward": estimate_step_flops(cfg),
        "episode_flops_forward": estimate_episode_flops(cfg, episode_steps),
    }
