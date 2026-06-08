# CREST Repo Manual

This manual explains how to use the CREST implementation suite from setup through synthetic memory tests, real text preparation, and training with byte or Llama 3 tokenization.

CREST is a recurrent-state decoder model. Each episode is split into steps. Within a step, tokens use local causal attention. Across steps, the model uses a fixed-size learned state that is read and written by attention/gated recurrence.

## 1. Environment

Run all commands from the repo root:

```bash
cd /root/crest
```

Install dependencies:

```bash
pip install -U pip
pip install -r src/requirements.txt
```

For Hugging Face datasets and Llama tokenizers:

```bash
pip install -U transformers datasets huggingface_hub sentencepiece tiktoken
```

If using gated Meta tokenizers such as `meta-llama/Meta-Llama-3-8B`, login first:

```bash
huggingface-cli login
```

Check GPU:

```bash
nvidia-smi
```

Check Python syntax:

```bash
PYTHONPATH=src python -m py_compile src/crest/*.py
```

Run tests when PyTorch is installed locally:

```bash
PYTHONPATH=src pytest -q src/tests
```

## 2. Repo Layout

Important paths:

```text
src/crest/
  model.py                  CRESTModel forward API
  layers.py                 local attention, state read, state write, RMSNorm, SwiGLU
  data.py                   synthetic, Arrow, JSONL, and streaming episodic datasets
  train.py                  training loop, DDP, mixed precision, checkpointing
  eval.py                   eval metrics
  cli_train.py              train standard config
  cli_train_variant.py      train one variant: crest, no_state, M<int>
  cli_ablate.py             run ablation suites
  cli_prepare_text.py       prepare one raw/HF/Kaggle text source
  cli_prepare_manifest.py   prepare a list of HF datasets from YAML manifest

src/configs/models/
  default.yaml
  debug.yaml
  debug_plus.yaml
  debug_wide.yaml
  small.yaml
  base.yaml
  research_125m.yaml
  scale_1b.yaml

src/configs/data/
  synthetic task configs
  legacy episodic JSONL configs

src/configs/data_manifests/
  manifest YAMLs for multi-source Hugging Face prep and streaming training

src/configs/training/
  debug and hardware-specific training configs
```

## 3. Core Concepts

### Episodes And Steps

CREST real-text data is stored as episodic rows. New prep writes Hugging Face Arrow datasets under `train/` and `eval/`. Legacy JSONL still loads, but new real-text runs should use Arrow or direct streaming.

Each row is one episode:

```json
{"steps":[{"input_ids":[1,2,3],"labels":[2,3,4]}]}
```

Shapes during training:

```text
input_ids: [B, T, L]
labels:    [B, T, L]
step_idx:  [B, T]
```

Where:

```text
B = batch size
T = episode steps
L = tokens per step
```

The model receives one step at a time:

```python
logits, next_state, aux = model(input_ids[:, t], state=state, step_idx=step_idx[:, t])
```

### State Update

The recurrent state update is:

```text
S_next = G * S + (1 - G) * U
```

Where:

```text
G near 1 means retain old state.
G near 0 means overwrite with new content.
```

Useful diagnostics:

```text
gate_mean              average retention gate
state_read_entropy     how sharp state reads are
write_entropy          how sharp state writes are
grad_norm              gradient health
recall_accuracy        token accuracy over labeled positions
```

For `M=16`, uniform read entropy is:

```text
log(16) = 2.773
```

Values below this mean state reads are becoming selective.

## 4. Precision And Hardware

Use `fp16` on T4, RTX 3090, and most consumer GPUs:

```yaml
precision: fp16
```

Use `bf16` on A100/H100 or hardware where BF16 tensor cores are known to be fast.

Do not use BF16 on T4.

On a single GPU, use plain Python:

```bash
PYTHONPATH=src python -m crest.cli_train_variant ...
```

On multiple visible GPUs, use `torchrun`:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 -m crest.cli_train_variant ...
```

Do not use `--nproc_per_node=2` if `nvidia-smi` shows only one GPU.

## 5. Model Configs

### Byte-Tokenizer Debug Plus

Use this for byte-tokenized data such as WikiText byte runs:

```text
src/configs/models/debug_plus.yaml
```

It uses:

```yaml
vocab_size: 1024
n_layers: 4
d_model: 256
n_heads: 8
d_ffn: 768
memory_slots: 16
```

### Default Llama 3 Tokenizer Model

Use this for `meta-llama/Meta-Llama-3-8B` tokenizer data:

```text
src/configs/models/default.yaml
```

It uses:

```yaml
vocab_size: 128256
n_layers: 4
d_model: 256
n_heads: 8
d_ffn: 768
memory_slots: 16
```

The larger vocab greatly increases embedding/head parameters. This is expected.

### Debug Wide

Use this when you want a small scale-up from `debug_plus`:

```text
src/configs/models/debug_wide.yaml
```

It uses:

```yaml
d_model: 320
d_ffn: 896
n_layers: 4
n_heads: 8
```

Recent experiments showed `debug_wide` did not clearly beat `debug_plus` on WikiText at the same step budget, so `debug_plus M16` remains the default until stronger evidence says otherwise.

## 6. Training Variants

Use `cli_train_variant.py` for focused experiments:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/wikitext103_episodic.yaml \
  --training src/configs/training/debug_plus_2500.yaml \
  --variant M16
```

Valid variants:

```text
crest       base model config as provided
no_state    disables state read/write
M4          memory_slots=4
M8          memory_slots=8
M16         memory_slots=16
M32         memory_slots=32
```

Use `M16` as the current default.

Use `no_state` only for comparison. In current code, disabled state modules may still exist as unused parameters; this is acceptable for a functional ablation but not a perfect active-parameter-matched baseline.

## 7. Synthetic Diagnostics

Synthetic tasks are memory probes, not the final goal.

### Key-Value Recall

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/key_value_recall_debug.yaml \
  --training src/configs/training/debug_plus_fast.yaml \
  --variant M16
```

### Medium Multi-Hop

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/synthetic_multi_hop_medium.yaml \
  --training src/configs/training/debug_plus_fast.yaml \
  --variant M16
```

Expected by step 500-600:

```text
recall_accuracy >= 0.29
```

### Exact Two-Hop Medium

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/synthetic_exact_two_hop_medium.yaml \
  --training src/configs/training/debug_plus_fast.yaml \
  --variant M16
```

This is a stricter compositional probe. It is not a blocker for text training.

## 8. Ablations

Use `cli_ablate.py` only when you need reports. It is slower because it trains multiple variants.

Fast M8/M16 ablation without dense baseline:

```bash
PYTHONPATH=src python -m crest.cli_ablate \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/synthetic_multi_hop_medium.yaml \
  --training src/configs/training/debug_plus_fast.yaml \
  --memory-sweep 8,16 \
  --skip-full-attention \
  --out runs/ablations/debug_plus_multi_hop_medium_final.jsonl
```

Avoid broad sweeps during active development. Use focused `cli_train_variant.py` runs.

## 9. Preparing WikiText-103

Kaggle WikiText-103 prep:

```bash
PYTHONPATH=src python -m crest.cli_prepare_text \
  --kaggle-dataset vadimkurochkin/wikitext-103 \
  --out data/wikitext103_episodic \
  --tokenizer byte \
  --episode-steps 16 \
  --step-length 128 \
  --eval-fraction 0.2
```

Train CREST on WikiText byte episodes:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/wikitext103_episodic.yaml \
  --training src/configs/training/debug_plus_2500.yaml \
  --variant M16
```

No-state comparison:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/debug_plus.yaml \
  --data src/configs/data/wikitext103_episodic.yaml \
  --training src/configs/training/debug_plus_compare.yaml \
  --variant no_state
```

Matched-step WikiText evidence so far:

```text
CREST M16 step 600 eval_loss:    ~1.922
no_state step 600 eval_loss:     ~1.970

CREST M16 step 900 eval_loss:    ~1.745
no_state step 900 eval_loss:     ~1.827
```

CREST is modestly but consistently better on WikiText at debug-plus scale.

## 10. Preparing FineWeb-Edu With Llama 3 Tokenizer

Use manifest-based prep for real-world datasets.

Manifest:

```text
src/configs/data_manifests/default.yaml
```

Contents:

```yaml
datasets:
  - repo: HuggingFaceFW/fineweb-edu
    subset: sample-10BT
    split: train
    text_field: text
    max_documents: 50000
    max_tokens: 10000000
    streaming: true
```

Prepare:

```bash
PYTHONPATH=src python -m crest.cli_prepare_manifest \
  --manifest src/configs/data_manifests/default.yaml \
  --out data/episodic_arrow/default \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --episode-steps 16 \
  --step-length 128 \
  --stride-tokens 2048 \
  --eval-fraction 0.02 \
  --max-tokens 10000000 \
  --cleanup-cache
```

`max_tokens` is a hard token cap. For example, FineWeb-Edu stops being pretokenized once 10M tokenizer tokens have been consumed. With multiple manifest entries, each source can also define its own `max_tokens`, and the prep tool moves to the next dataset after the cap is reached. `--cleanup-cache` removes matching Hugging Face dataset cache directories after each source finishes.

This writes:

```text
data/episodic_arrow/default/train/
data/episodic_arrow/default/eval/
data/episodic_arrow/default/metadata.json
```

It also updates `src/configs/data_manifests/default.yaml` in place with:

```yaml
format: arrow
task: arrow_episodic
path: data/episodic_arrow/default
vocab_size: 128256
metadata:
  tokenizer: meta-llama/Meta-Llama-3-8B
```

Do not create a separate data YAML for this path. The manifest is the data config.

Train:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/default.yaml \
  --data src/configs/data_manifests/default.yaml \
  --training src/configs/training/default.yaml \
  --variant M16
```

Train without pretokenization by streaming Hugging Face data and tokenizing online:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/default.yaml \
  --data src/configs/data_manifests/default.yaml \
  --training src/configs/training/default.yaml \
  --variant M16 \
  --streaming
```

`--streaming` changes the loaded data task to `streaming_text`, forces each manifest source to `streaming=True`, and does not read or write Arrow/JSONL shards. The same option works with `crest.cli_train`.

If you want lower RAM and a local disk cache instead of live streaming, materialize bounded raw JSONL files first:

```bash
PYTHONPATH=src python -m crest.cli_prepare_manifest \
  --manifest src/configs/data_manifests/default.yaml \
  --out data/raw_text/default \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --episode-steps 16 \
  --step-length 128 \
  --eval-fraction 0.02 \
  --max-tokens 10000000 \
  --raw-text-only
```

This writes:

```text
data/raw_text/default/train.jsonl
data/raw_text/default/eval.jsonl
data/raw_text/default/metadata.json
```

Then train from the raw files without pretokenization:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/default.yaml \
  --data src/configs/data_manifests/default.yaml \
  --training src/configs/training/default.yaml \
  --variant M16 \
  --raw-text
```

`--raw-text` changes the loaded data task to `raw_text` and reads local `train.jsonl` / `eval.jsonl` files directly.

No-state comparison:

```bash
PYTHONPATH=src python -m crest.cli_train_variant \
  --model src/configs/models/default.yaml \
  --data src/configs/data_manifests/default.yaml \
  --training src/configs/training/default.yaml \
  --variant no_state \
  --streaming
```

If Llama tokenizer access fails, use `gpt2` as an open fallback, but then use a GPT-2 vocab-sized model config.

## 11. Adding More Datasets To A Manifest

Edit the manifest:

```yaml
datasets:
  - repo: HuggingFaceFW/fineweb-edu
    subset: sample-10BT
    split: train
    text_field: text
    max_documents: 50000
    streaming: true
  - repo: wikimedia/wikipedia
    subset: 20231101.es
    split: train
    text_field: text
    max_documents: 30000
    streaming: true
  - repo: codeparrot/github-code
    split: train
    text_field: code
    max_documents: 30000
    streaming: true
```

Then either run `cli_prepare_manifest.py` with `--out data/episodic_arrow/default` to refresh Arrow shards, or train directly with `--streaming` or `--raw-text`. Do not create a new YAML unless there is a real separate experiment that needs to be preserved.

## 12. Generation Status

The mechanism is ready in principle for generation because the model exposes:

```python
logits, next_state, aux = model(input_ids, state=state, step_idx=step_idx)
```

But a dedicated generation CLI is not yet documented here. Expected generation flow:

```text
1. Load checkpoint.
2. Load matching tokenizer.
3. Initialize recurrent state.
4. Encode prompt into one or more CREST steps.
5. Repeatedly sample next token from logits.
6. Carry state forward across step boundaries.
```

Current small checkpoints can produce toy text but should not be expected to behave like a capable language model.

Important: byte-token checkpoints are not compatible with Llama-tokenizer models. Tokenizer and `vocab_size` must match.

## 13. Current Recommended Path

The current recommended default is:

```text
model: default M16
precision: fp16
gate_retention_bias: -2.0
tokenizer for real training: meta-llama/Meta-Llama-3-8B
first real dataset: FineWeb-Edu sample-10BT
```

Run order:

```text
1. Train default M16 with `--streaming` for immediate runs, or prepare Arrow shards when you want deterministic reusable data.
2. Train default M16.
3. Run no_state comparison.
4. Run synthetic medium multi-hop regression occasionally.
5. Only scale after CREST beats no_state on real text.
```

## 14. Troubleshooting

### DDP Unused Parameter Error

If `no_state` crashes under DDP with unused parameters, use plain Python on a single GPU:

```bash
PYTHONPATH=src python -m crest.cli_train_variant ... --variant no_state
```

Or ensure DDP uses unused-parameter detection.

### `grad_norm=inf` At Step 0

This can happen in FP16 and usually recovers with GradScaler. If it persists after warmup, lower LR:

```yaml
learning_rate: 0.0002
warmup_steps: 200
grad_clip_norm: 0.5
```

### Llama Tokenizer Fails

Run:

```bash
huggingface-cli login
```

Confirm you have access to `meta-llama/Meta-Llama-3-8B`.

### Embedding Index Error

Tokenizer vocab and model vocab do not match.

Use:

```text
debug_plus.yaml              for byte tokenizer
default.yaml                 for Llama 3 tokenizer
```

### Dataset Produces No Episodes

Use more documents or lower episode size:

```bash
--max-documents 50000 --episode-steps 16 --step-length 128
```

### VRAM OOM

Lower batch size:

```yaml
batch_size: 64
```

For Llama 3 tokenizer models, the vocab head is large. Start with lower batch sizes.

## 15. Success Criteria

For real text training, useful signs are:

```text
eval_loss decreases smoothly
CREST eval_loss < no_state eval_loss at matched steps
state_read_entropy drops below uniform log(M)
write_entropy drops below uniform log(step_length)
synthetic multi-hop regression remains above target
```

Bad signs:

```text
CREST worse than no_state
gate stuck near 0.12 after hundreds of steps
read entropy stuck at log(M)
repeated NaN/Inf after warmup
synthetic memory probes regress sharply
```

## 16. Mathematical Status

CREST is conditionally validated at this stage.

Verified empirically so far:

```text
State beats no_state on key-value recall.
State beats no_state on multi-hop diagnostics.
State improves WikiText byte-token text loss modestly.
Gate and entropy diagnostics show the state path becomes active.
```

Still empirical:

```text
Whether state helps large real-world corpora.
Whether M16 remains sufficient under Llama-tokenized FineWeb-Edu.
Whether exact two-hop improves with longer training or better curriculum.
Whether larger models improve the frontier under limited compute.
```

Do not treat one good text run as proof of long-horizon memory. Use no-state comparisons and synthetic regressions as guardrails.
