# Reproducible training

## Environment

Use Python 3.10 or 3.11 and install the minimal trainer dependencies:

```shell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-model.txt
```

For CUDA, install the PyTorch wheel matching the local CUDA driver before
installing the remaining requirements. The pinned `torch==2.1.1` is the
reference version used by the original environment; CPU-only or newer CUDA
wheels may also work but should be recorded with the run.

## Dry run

```shell
python cryptobench/model.py --dry_run --out_dir runs/dry-run
```

The dry run writes `fixed_run_config.json` and does not load data or train.

## Training

```shell
python cryptobench/model.py \
  --train_subset cb-p2rank-apo \
  --test_preset same \
  --val_fold 0 \
  --epochs 20 \
  --seed 42 \
  --out_dir runs/cb-p2rank-apo-seed42
```

Useful alternatives are `cb-full` and `cb-pm` for `--train_subset`. Use
`--device cpu` for a CPU run or `--gpu 0` to select a CUDA device. The default
workflow tunes on three folds, optionally retrains on all four folds, evaluates
validation/test data, and writes:

- `best_model.pt`: model state dictionary;
- `fixed_run_config.json`: resolved paths, runtime settings and fixed hyperparameters;
- `summary.json`: sample counts, skipped-record counts and metrics;
- `history_tune.csv` and `history_final.csv`: epoch histories.

Use `--no_final_train_all_folds` if you need to retain the fold-held-out model
after threshold tuning. Use `--collect_code_stats` to additionally save VQ code
usage/enrichment summaries.

## Repository hygiene

Do not commit downloaded datasets, ESM2 weights, generated caches, or training
checkpoints to GitHub. Store them in external release storage and keep a small
manifest containing source URLs, checksums, model name, package versions,
dataset version, split, seed, and command line. The root `.gitignore` contains
patterns for the generated run artifacts.
