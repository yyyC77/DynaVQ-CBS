#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_pos_idx(ann: str) -> List[int]:
    out = []
    for tok in str(ann).split():
        i = len(tok) - 1
        while i >= 0 and tok[i].isdigit():
            i -= 1
        if i < len(tok) - 1:
            out.append(int(tok[i + 1 :]))
    return out


def load_source_chain_maps(src_pipeline: Path):
    chains = pd.read_csv(src_pipeline / "entities" / "chains.csv")
    chain_npz_by_uid = dict(zip(chains["chain_uid"].astype(str), chains["chain_npz"].astype(str)))
    nres_by_uid = dict(zip(chains["chain_uid"].astype(str), chains["n_residues"].astype(int)))
    return chains, chain_npz_by_uid, nres_by_uid


def iter_rows(csv_path: Path, split_name: str):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split(";")
            if len(parts) < 6:
                continue
            yield {
                "line_no": ln,
                "split": split_name,
                "apo_pdb_id": parts[0].replace("\ufeff", "").strip().lower(),
                "apo_chain": parts[1].replace("\ufeff", "").strip(),
                "uniprot_id": parts[2].strip(),
                "annotation": parts[3].strip(),
                "chain_uid": parts[5].strip(),
            }


def main():
    ap = argparse.ArgumentParser(description="Rebuild entities from aligned translated CSV (with chain_uid column).")
    ap.add_argument("--aligned_ann_dir", type=Path, default=Path("cryptobench-translated-annotations-aligned"))
    ap.add_argument("--src_pipeline_dir", type=Path, default=Path("cryptobench/pipeline_v2"))
    ap.add_argument("--out_pipeline_dir", type=Path, default=Path("cryptobench/pipeline_v2_cut_aligned"))
    ap.add_argument("--use_final_csv", action="store_true")
    args = ap.parse_args()

    suffix = ".final.csv" if args.use_final_csv else ".csv"
    folds = [f"train-fold-{i}" for i in range(4)] + ["test"]
    split_map = {f"train-fold-{i}": f"train_fold_{i}" for i in range(4)}
    split_map["test"] = "test"

    src_chains, chain_npz_by_uid, nres_by_uid = load_source_chain_maps(args.src_pipeline_dir)
    src_npz_dir = args.src_pipeline_dir / "entities" / "chain_npz"

    out_entities = args.out_pipeline_dir / "entities"
    out_npz_dir = out_entities / "chain_npz"
    out_entities.mkdir(parents=True, exist_ok=True)
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    rows_by_uid = defaultdict(list)
    total_rows = 0
    for f in folds:
        p = args.aligned_ann_dir / f"{f}{suffix}"
        if not p.exists():
            continue
        for r in iter_rows(p, split_map[f]):
            rows_by_uid[r["chain_uid"]].append(r)
            total_rows += 1

    chain_rows = []
    model_input_rows = []
    n_kept_uid = 0
    n_skipped_missing_npz = 0
    n_skipped_bad_len = 0
    for uid, rows in rows_by_uid.items():
        npz_name = chain_npz_by_uid.get(uid)
        if npz_name is None:
            n_skipped_missing_npz += 1
            continue
        src_npz = src_npz_dir / npz_name
        if not src_npz.exists():
            n_skipped_missing_npz += 1
            continue
        d = np.load(src_npz)
        n_res = int(d["ca"].shape[0])
        y_union = np.zeros((n_res,), dtype=np.int8)
        ok = True
        for r in rows:
            for idx in parse_pos_idx(r["annotation"]):
                if 0 <= idx < n_res:
                    y_union[idx] = 1
                else:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            n_skipped_bad_len += 1
            continue

        # rebuild npz with new y_union, keep geometry fields as-is
        out_npz = out_npz_dir / f"{uid}.npz"
        np.savez_compressed(
            out_npz,
            y_union=y_union,
            y_regions=np.expand_dims(y_union, 0),
            region_ids=np.array([0], dtype=np.int32),
            ca=d["ca"],
            n=d["n"],
            c=d["c"],
            o=d["o"],
            cb=d["cb"],
            auth_seq_id=d["auth_seq_id"],
            auth_chain=d["auth_chain"] if "auth_chain" in d else np.array(["A"] * n_res),
            label_chain=d["label_chain"] if "label_chain" in d else np.array(["A"] * n_res),
            ins_code=d["ins_code"] if "ins_code" in d else np.array([""] * n_res),
            resname=d["resname"] if "resname" in d else np.array(["UNK"] * n_res),
            aa1=d["aa1"] if "aa1" in d else np.array(["X"] * n_res),
            has_backbone=d["has_backbone"],
            has_cb=d["has_cb"],
        )

        ch = src_chains[src_chains["chain_uid"] == uid].iloc[0].to_dict()
        split = "test" if any(r["split"] == "test" for r in rows) else rows[0]["split"]
        chain_rows.append(
            {
                "chain_uid": uid,
                "apo_pdb_id": ch["apo_pdb_id"],
                "apo_chain": ch["apo_chain"],
                "n_residues": n_res,
                "n_regions": len(rows),
                "n_positive_union": int(y_union.sum()),
                "chain_npz": out_npz.name,
            }
        )
        model_input_rows.append(
            {
                "chain_uid": uid,
                "apo_pdb_id": ch["apo_pdb_id"],
                "apo_chain": ch["apo_chain"],
                "n_residues": n_res,
                "n_regions": len(rows),
                "n_positive_union": int(y_union.sum()),
                "chain_npz": out_npz.name,
                "split": split,
                "is_test": int(split == "test"),
            }
        )
        n_kept_uid += 1

    chains_df = pd.DataFrame(chain_rows)
    model_inputs_df = pd.DataFrame(model_input_rows)
    chains_df.to_csv(out_entities / "chains.csv", index=False)
    model_inputs_df.to_csv(out_entities / "model_inputs.csv", index=False)

    # also copy chain_splits for compatibility
    chain_splits = model_inputs_df[["chain_uid", "split", "is_test"]].copy()
    chain_splits.to_csv(out_entities / "chain_splits.csv", index=False)

    summary = {
        "aligned_ann_dir": str(args.aligned_ann_dir),
        "src_pipeline_dir": str(args.src_pipeline_dir),
        "out_pipeline_dir": str(args.out_pipeline_dir),
        "total_rows": total_rows,
        "kept_chain_uids": n_kept_uid,
        "skipped_missing_npz": n_skipped_missing_npz,
        "skipped_bad_len": n_skipped_bad_len,
        "n_chains": int(len(chains_df)),
        "n_test_chains": int((model_inputs_df["is_test"] == 1).sum()) if not model_inputs_df.empty else 0,
    }
    (out_entities / "summary_entities_from_cut.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

