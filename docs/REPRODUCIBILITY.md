# Reproduction guide

Use Python 3.10 or 3.11 and install `requirements-model.txt`. For GPU runs,
install a PyTorch build compatible with the local CUDA driver.

```bash
python model.py --dry_run --out_dir runs/dry-run
python model.py --train_subset apo --test_preset same \\
  --val_fold 0 --epochs 20 --seed 42 --out_dir runs/apo-seed42
```

The run directory records the resolved configuration, weights, epoch histories,
and validation/test metrics. For fair comparison, report the representation
checkpoint, source-data release, split, seed, device, package versions, and
exact command line.
