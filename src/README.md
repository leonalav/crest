# CREST Implementation Suite

This folder contains the first mathematically faithful CREST implementation suite:

- `crest/`: PyTorch model, recurrent state, losses, synthetic data, training utilities, and metrics.
- `configs/models/`: separate YAML model parameter presets.
- `configs/data/`: separate YAML episodic data suites.
- `configs/training/`: training YAML presets.
- `tests/`: unit tests for RoPE, causality, state mechanics, losses, data, and training smoke paths.

Ground-truth references are `../docs/crest_plan.md`, `../verdict.md`, `../cautions.md`, and the Markdown paper suite under `../docs/suite`.

Install runtime/test dependencies with:

```bash
pip install -r src/requirements.txt
```

Run tests with:

```bash
PYTHONPATH=src pytest -q src/tests
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="A:\crest_lm\src"; pytest -q "A:\crest_lm\src\tests"
```
