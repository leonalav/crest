from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CRESTConfig:
    vocab_size: int = 1024
    max_seq_len: int = 64
    max_steps: int = 128
    n_layers: int = 2
    d_model: int = 128
    n_heads: int = 4
    d_ffn: int = 256
    memory_slots: int = 16
    dropout: float = 0.0
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    gate_retention_bias: float = -2.0
    tie_embeddings: bool = True
    pad_token_id: int = 0
    use_state_read: bool = True
    use_state_write: bool = True
    use_local_rope: bool = True
    attention_backend: str = "auto"
    compute_attention_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.memory_slots <= 0 or self.max_seq_len <= 0:
            raise ValueError("memory_slots and max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "debug"
    batch_size: int = 8
    micro_batch_size: int = 64
    max_steps: int = 1000
    tbptt_k: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    gate_regularization_weight: float = 0.0
    gate_target: float = 0.5
    precision: str = "bf16"
    seed: int = 1337
    output_dir: str = "runs/debug"
    run_name: str = "crest_debug"
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    resume_from: str | None = None
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_fsdp: bool = False
    compile_model: bool = False
    compile_mode: str = "default"
    fused_optimizer: bool = True
    aux_state_weight: float = 0.0
    aux_state_dim: int = 128
    ce_chunk_size: int = 256


@dataclass(frozen=True)
class DataConfig:
    suite: str = "key_value_recall_debug"
    task: str = "key_value_recall"
    vocab_size: int = 1024
    num_keys: int = 16
    num_values: int = 64
    episode_steps: int = 8
    step_length: int = 16
    train_episodes: int = 1024
    eval_episodes: int = 128
    query_probability: float = 0.5
    path: str | None = None
    seed: int = 1337
    metadata: dict[str, Any] = field(default_factory=dict)


MODEL_PRESETS: dict[str, CRESTConfig] = {
    "debug": CRESTConfig(vocab_size=1024, max_seq_len=64, n_layers=2, d_model=128, n_heads=4, d_ffn=256, memory_slots=16),
    "small": CRESTConfig(vocab_size=32000, max_seq_len=128, n_layers=8, d_model=512, n_heads=8, d_ffn=1368, memory_slots=64),
    "base": CRESTConfig(vocab_size=64000, max_seq_len=256, n_layers=12, d_model=768, n_heads=12, d_ffn=2048, memory_slots=128),
    "research_125m": CRESTConfig(vocab_size=50304, max_seq_len=256, n_layers=12, d_model=768, n_heads=12, d_ffn=2048, memory_slots=128),
    "scale_1b": CRESTConfig(vocab_size=128000, max_seq_len=512, n_layers=20, d_model=1536, n_heads=16, d_ffn=4096, memory_slots=512),
}


def get_model_config(name: str) -> CRESTConfig:
    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown CREST preset {name!r}; available={sorted(MODEL_PRESETS)}") from exc
