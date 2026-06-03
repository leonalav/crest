# CREST Implementation Suite

This folder contains the first mathematically faithful CREST implementation suite:

- `crest/`: PyTorch model, recurrent state, attention backend policy, losses, alignment, baselines, distributed utilities, checkpointing, logging, synthetic/natural data, training/eval utilities, and metrics.
- `configs/models/`: separate YAML model parameter presets.
- `configs/data/`: separate YAML episodic data suites, including synthetic recall, multi-hop, tool traces, and JSONL natural episodes.
- `configs/training/`: training YAML presets, including debug and 125M-scale research runs.
- `tests/`: unit tests for RoPE, causality, state mechanics, losses, data, training, eval, ablations, baselines, reports, and alignment surfaces.

Ground-truth references are `../docs/crest_plan.md`, `../verdict.md`, `../cautions.md`, and the Markdown paper suite under `../docs/suite`.

Install runtime/test dependencies with:

```bash
pip install -r src/requirements.txt
```

Run tests with:

```bash
PYTHONPATH=src pytest -q src/tests
```

Run a debug training harness:

```bash
PYTHONPATH=src python -m crest.cli_train \
  --model src/configs/models/debug.yaml \
  --data src/configs/data/key_value_recall_debug.yaml \
  --training src/configs/training/debug.yaml
```

Run the 125M research harness:

```bash
PYTHONPATH=src python -m crest.cli_train \
  --model src/configs/models/research_125m.yaml \
  --data src/configs/data/synthetic_multi_hop.yaml \
  --training src/configs/training/research_125m.yaml
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="A:\crest_lm\src"; pytest -q "A:\crest_lm\src\tests"
```
