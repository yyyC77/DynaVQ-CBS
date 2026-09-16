#!/usr/bin/env python3
"""Create train-script-compatible annotation folders for evaluation subsets.

``train_plmnn2_gvp_flex_vq.py`` reads semicolon-separated annotation CSV files
from an ``--ann_dir``. The released CryptoBench subset JSON files are useful
for provenance, but this training code needs the translated annotation format.

This script copies train folds unchanged and filters ``test.final.csv`` /
``test.csv`` to the apo IDs in each reproduced evaluation subset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUBSETS = ["cb-full", "cb-pm", "cb-p2rank-apo"]


def filter_annotation_file(src: Path, dst: Path, apo_ids: set[str]) -> int:
    kept = 0
    with src.open("r", encoding="utf-8-sig") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            apo = stripped.split(";", 1)[0].lower()
            if apo in apo_ids:
                fout.write(stripped + "\n")
                kept += 1
    return kept


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    default_source = Path("cryptobench-translated-annotations-aligned")
    if not default_source.exists():
        default_source = Path("cryptobench-translated-annotations")
    default_out = Path("cryptobench/evaluation-annotations-aligned")
    if default_source.name != "cryptobench-translated-annotations-aligned":
        default_out = Path("cryptobench/evaluation-annotations")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-ann-dir",
        type=Path,
        default=default_source,
        help="Existing annotation directory used by the training script.",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("cryptobench/cryptobench-dataset/evaluation-subsets"),
        help="Directory created by reproduce_eval_subsets.py.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=default_out,
        help="Output root for per-subset annotation directories.",
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    report = []

    for subset in SUBSETS:
        ids_path = args.subset_dir / f"{subset}-ids.txt"
        if not ids_path.exists():
            raise FileNotFoundError(ids_path)
        apo_ids = {line.strip().lower() for line in ids_path.read_text().splitlines() if line.strip()}

        out_dir = args.out_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            for suffix in [".csv", ".final.csv", ".summary.json", ".failures.json"]:
                copy_if_exists(
                    args.source_ann_dir / f"train-fold-{i}{suffix}",
                    out_dir / f"train-fold-{i}{suffix}",
                )

        test_counts = {}
        for name in ["test.csv", "test.final.csv"]:
            src = args.source_ann_dir / name
            if src.exists():
                test_counts[name] = filter_annotation_file(src, out_dir / name, apo_ids)

        readme = out_dir / "README.md"
        readme.write_text(
            f"""# {subset} Training Annotations

This directory is compatible with `cryptobench/train_plmnn2_gvp_flex_vq.py`.

- Train folds are copied unchanged from `{args.source_ann_dir}`.
- Test files are filtered to apo IDs from `{ids_path}`.
- Use this directory with `--ann_dir {out_dir}`.

Filtered rows:
{test_counts}
""",
            encoding="utf-8",
        )
        report.append({"subset": subset, "apo_ids": len(apo_ids), **test_counts, "out_dir": str(out_dir)})

    for row in report:
        print(row)


if __name__ == "__main__":
    main()
