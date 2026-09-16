# Dynamics-informed selective structural routing for cryptic binding sites prediction from apo structures

This repository provides the reference implementation for residue-level
cryptic binding-site prediction from apo protein structures. The method uses
sequence representations, local backbone geometry, residue flexibility cues,
and selective structural routing with a compact vector-quantized bottleneck.

## Repository contents

- `model.py` — the complete model, data loader, training loop, and evaluation code.
- `config.py` — paths, dataset presets, and all experiment hyperparameters.
- `requirements-model.txt` — minimal dependencies for training and evaluation.
- `data_preprocessing/` — data preparation utilities and a concise preprocessing overview.
- `docs/` — representation provenance and reproduction notes.
- `docs/dataset_scale.tex` — manuscript-ready dataset-scale table.

Large data files, precomputed representations, caches, and checkpoints are not
tracked in this repository. Their sources and expected layout are documented in
`docs/` and `data_preprocessing/`.

## Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-model.txt
```

## Configure and run

Edit `config.py` if the prepared data are stored outside the default `data/`
layout. First verify the resolved configuration without loading data:

```bash
python model.py --dry_run --out_dir runs/dry-run
```

Then run training:

```bash
python model.py --train_subset apo --test_preset same \\
  --epochs 20 --seed 42 --out_dir runs/apo-seed42
```

The output directory contains the trained weights, resolved configuration,
epoch histories, and validation/test metrics.

## Citation

If you use this code, please cite the accompanying paper. For the benchmark
and related cryptic-site resources used in the study, see the references in
[`references.bib`](references.bib).
