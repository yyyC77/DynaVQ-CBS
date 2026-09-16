#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def split_chain_token(s: str) -> List[str]:
    return [x.strip() for x in str(s).split("-") if x.strip()]


def load_chains(chains_csv: Path):
    rows = []
    with open(chains_csv, "r", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                {
                    "chain_uid": str(r["chain_uid"]),
                    "apo_pdb_id": str(r["apo_pdb_id"]).strip().lower(),
                    "apo_chain": str(r["apo_chain"]).strip(),
                    "n_residues": int(r["n_residues"]),
                }
            )
    by_pdb: Dict[str, List[dict]] = {}
    for r in rows:
        by_pdb.setdefault(r["apo_pdb_id"], []).append(r)
    return rows, by_pdb


def pick_chain_uid(cands: List[dict], chain_id: str, emb_len: int):
    # 1) keep candidates containing this chain letter in apo_chain token
    c1 = [c for c in cands if chain_id in split_chain_token(c["apo_chain"])]
    if not c1:
        return None, "no_chain_candidate"
    # 2) exact length match
    c2 = [c for c in c1 if int(c["n_residues"]) == int(emb_len)]
    if len(c2) == 1:
        return c2[0]["chain_uid"], "ok_len_exact"
    if len(c2) > 1:
        # prefer exact chain token match first
        exact = [c for c in c2 if c["apo_chain"] == chain_id]
        if len(exact) == 1:
            return exact[0]["chain_uid"], "ok_len_exact_chain_exact"
        # then prefer fewer merged chains
        c2 = sorted(c2, key=lambda x: (len(split_chain_token(x["apo_chain"])), x["chain_uid"]))
        return c2[0]["chain_uid"], "ok_len_exact_tie_break"
    # no exact len match
    return None, "len_mismatch"


def align_one_csv(in_csv: Path, out_csv: Path, emb_dir: Path, by_pdb: Dict[str, List[dict]]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "total_rows": 0,
        "kept_rows": 0,
        "missing_emb": 0,
        "no_pdb_candidate": 0,
        "no_chain_candidate": 0,
        "len_mismatch": 0,
        "tie_break": 0,
    }
    failures = []
    lines_out = []

    with open(in_csv, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            parts = s.split(";")
            if len(parts) < 4:
                failures.append({"line_no": line_no, "error": "bad_row", "line": s})
                continue
            summary["total_rows"] += 1
            pdb_id = parts[0].replace("\ufeff", "").strip().lower()
            chain_id = parts[1].replace("\ufeff", "").strip()
            emb_path = emb_dir / f"{pdb_id}{chain_id}.npy"
            if not emb_path.exists():
                summary["missing_emb"] += 1
                failures.append({"line_no": line_no, "pdb_id": pdb_id, "chain": chain_id, "error": "missing_emb"})
                continue
            emb_len = int(np.load(emb_path, mmap_mode="r").shape[0])

            cands = by_pdb.get(pdb_id, [])
            if not cands:
                summary["no_pdb_candidate"] += 1
                failures.append({"line_no": line_no, "pdb_id": pdb_id, "chain": chain_id, "error": "no_pdb_candidate"})
                continue
            chain_uid, reason = pick_chain_uid(cands, chain_id, emb_len)
            if chain_uid is None:
                summary[reason] += 1
                failures.append(
                    {
                        "line_no": line_no,
                        "pdb_id": pdb_id,
                        "chain": chain_id,
                        "emb_len": emb_len,
                        "error": reason,
                        "candidates": [
                            {"chain_uid": c["chain_uid"], "apo_chain": c["apo_chain"], "n_residues": c["n_residues"]}
                            for c in cands
                        ],
                    }
                )
                continue
            if "tie_break" in reason:
                summary["tie_break"] += 1

            # append chain_uid as 6th column
            line_new = s + f";{chain_uid}"
            lines_out.append(line_new)
            summary["kept_rows"] += 1

    out_csv.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    rep = {
        "summary": summary,
        "failures": failures[:5000],  # cap huge logs
    }
    rep_path = out_csv.with_suffix(".align_report.json")
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return summary, rep_path


def main():
    ap = argparse.ArgumentParser(description="Align translated CSV rows to chain_uid using pdb+chain+embedding length.")
    ap.add_argument("--ann_dir", type=Path, default=Path("cryptobench-translated-annotations"))
    ap.add_argument("--emb_dir", type=Path, default=Path("cryptobench-ahojv2-cut"))
    ap.add_argument("--pipeline_dir", type=Path, default=Path("cryptobench/pipeline_v2"))
    ap.add_argument("--use_final_csv", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path("cryptobench-translated-annotations-aligned"))
    args = ap.parse_args()

    _, by_pdb = load_chains(args.pipeline_dir / "entities" / "chains.csv")
    suffix = ".final.csv" if args.use_final_csv else ".csv"
    folds = [f"train-fold-{i}" for i in range(4)] + ["test"]
    all_sum = []
    for f in folds:
        in_csv = args.ann_dir / f"{f}{suffix}"
        out_csv = args.out_dir / f"{f}{suffix}"
        if not in_csv.exists():
            print(f"[skip] missing {in_csv}")
            continue
        s, rep = align_one_csv(in_csv, out_csv, args.emb_dir, by_pdb)
        all_sum.append(s)
        print(json.dumps({"fold": f, **s, "report": str(rep)}, ensure_ascii=False))

    (args.out_dir / "align_summary.json").write_text(json.dumps(all_sum, indent=2), encoding="utf-8")
    print(f"saved: {args.out_dir / 'align_summary.json'}")


if __name__ == "__main__":
    main()

