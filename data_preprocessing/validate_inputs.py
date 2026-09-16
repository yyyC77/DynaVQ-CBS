import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(base_dir: Path):
    entities = base_dir / "entities"
    graph_dir = base_dir / "graph_cache"
    out_report = base_dir / "validation_report.json"
    out_issues = base_dir / "validation_issues.csv"

    model_inputs = pd.read_csv(entities / "model_inputs.csv")
    chains = pd.read_csv(entities / "chains.csv")
    splits = pd.read_csv(entities / "chain_splits.csv")
    regions = pd.read_csv(entities / "regions.csv")
    graph_index = pd.read_csv(graph_dir / "graph_cache_index.csv")

    issues = []
    report = {}

    # 1) model_inputs.csv 中 chain_npz 是否都存在
    missing_npz = []
    for _, r in model_inputs.iterrows():
        p = entities / "chain_npz" / r["chain_npz"]
        if not p.exists():
            missing_npz.append(r["chain_uid"])
    report["check_1_model_inputs_npz_exists"] = {
        "ok": len(missing_npz) == 0,
        "missing_count": len(missing_npz),
    }
    for uid in missing_npz[:100]:
        issues.append({"check": 1, "chain_uid": uid, "detail": "missing_chain_npz"})

    # 2-4) npz consistency
    mismatch_res = []
    mismatch_union_sum = []
    mismatch_region_union = []

    chain_map = chains.set_index("chain_uid")
    for uid, row in chain_map.iterrows():
        npz_path = entities / "chain_npz" / row["chain_npz"]
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        y_union = d["y_union"]
        y_regions = d["y_regions"]
        n_res = int(row["n_residues"])
        n_pos = int(row["n_positive_union"])

        if y_union.shape[0] != n_res:
            mismatch_res.append((uid, y_union.shape[0], n_res))
        if int(y_union.sum()) != n_pos:
            mismatch_union_sum.append((uid, int(y_union.sum()), n_pos))
        if y_regions.shape[0] > 0:
            y_merge = (y_regions.max(axis=0) > 0).astype(np.int8)
        else:
            y_merge = np.zeros_like(y_union)
        if not np.array_equal(y_merge, y_union.astype(np.int8)):
            mismatch_region_union.append(uid)

    report["check_2_npz_residue_count_match"] = {"ok": len(mismatch_res) == 0, "mismatch_count": len(mismatch_res)}
    report["check_3_union_sum_match"] = {"ok": len(mismatch_union_sum) == 0, "mismatch_count": len(mismatch_union_sum)}
    report["check_4_region_union_match"] = {"ok": len(mismatch_region_union) == 0, "mismatch_count": len(mismatch_region_union)}
    for uid, got, exp in mismatch_res[:100]:
        issues.append({"check": 2, "chain_uid": uid, "detail": f"res_count got={got} expected={exp}"})
    for uid, got, exp in mismatch_union_sum[:100]:
        issues.append({"check": 3, "chain_uid": uid, "detail": f"union_sum got={got} expected={exp}"})
    for uid in mismatch_region_union[:100]:
        issues.append({"check": 4, "chain_uid": uid, "detail": "region_union_not_equal_union_label"})

    # 5) 每个 split 的 chain_uid 无重叠
    # Since chain_splits has one row per chain_uid by construction, check duplicates.
    dup_chain_split = splits[splits.duplicated(subset=["chain_uid"], keep=False)]
    report["check_5_split_no_chain_overlap"] = {
        "ok": len(dup_chain_split) == 0,
        "duplicate_rows": int(len(dup_chain_split)),
    }
    for uid in dup_chain_split["chain_uid"].head(100).tolist():
        issues.append({"check": 5, "chain_uid": uid, "detail": "duplicate_chain_in_chain_splits"})

    # 6) graph_cache 是否每个 chain 都存在
    graph_uid_set = set(graph_index["chain_uid"].tolist())
    chain_uid_set = set(chains["chain_uid"].tolist())
    missing_graph = sorted(chain_uid_set - graph_uid_set)
    extra_graph = sorted(graph_uid_set - chain_uid_set)
    report["check_6_graph_cache_exists_per_chain"] = {
        "ok": len(missing_graph) == 0,
        "missing_graph_count": len(missing_graph),
        "extra_graph_count": len(extra_graph),
    }
    for uid in missing_graph[:100]:
        issues.append({"check": 6, "chain_uid": uid, "detail": "missing_graph_cache"})
    for uid in extra_graph[:100]:
        issues.append({"check": 6, "chain_uid": uid, "detail": "extra_graph_cache"})

    # 7) graph_cache 中每个 center residue 是否有 local graph
    bad_graph_center = []
    for _, r in graph_index.iterrows():
        gp = graph_dir / r["graph_file"]
        if not gp.exists():
            bad_graph_center.append((r["chain_uid"], "graph_file_missing"))
            continue
        g = np.load(gp)
        neigh = g["neighbor_idx"]
        ptr = g["edge_ptr"]
        n = int(r["n_residues"])
        # center existence: each row should have at least self neighbor index
        row_has_center = np.any(np.arange(n)[:, None] == neigh, axis=1) if neigh.shape[0] == n else np.array([], dtype=bool)
        if neigh.shape[0] != n or ptr.shape[0] != n + 1 or (len(row_has_center) > 0 and not row_has_center.all()):
            bad_graph_center.append((r["chain_uid"], f"shape_or_center_issue neigh={neigh.shape} ptr={ptr.shape}"))
    report["check_7_each_center_has_local_graph"] = {"ok": len(bad_graph_center) == 0, "bad_count": len(bad_graph_center)}
    for uid, detail in bad_graph_center[:100]:
        issues.append({"check": 7, "chain_uid": uid, "detail": detail})

    # 8) node/edge 特征中不含 holo/pRMSD/ligand 信息
    # Structural check by schema: model_inputs should not carry those fields.
    forbidden_cols = {"holo_pdb_id", "holo_chain", "ligand", "ligand_chain", "ligand_index", "pRMSD"}
    present_forbidden_model_inputs = sorted(set(model_inputs.columns) & forbidden_cols)
    present_forbidden_graph_index = sorted(set(graph_index.columns) & forbidden_cols)
    report["check_8_no_holo_ligand_prmsd_in_model_inputs_graph_cache"] = {
        "ok": len(present_forbidden_model_inputs) == 0 and len(present_forbidden_graph_index) == 0,
        "model_inputs_forbidden_cols": present_forbidden_model_inputs,
        "graph_index_forbidden_cols": present_forbidden_graph_index,
    }
    if present_forbidden_model_inputs:
        issues.append({"check": 8, "chain_uid": "", "detail": f"model_inputs has forbidden cols {present_forbidden_model_inputs}"})
    if present_forbidden_graph_index:
        issues.append({"check": 8, "chain_uid": "", "detail": f"graph_index has forbidden cols {present_forbidden_graph_index}"})

    # 9) CA/N/C/O 缺失比例
    total_res = 0
    miss_n = miss_ca = miss_c = miss_o = 0
    for _, row in chains.iterrows():
        d = np.load(entities / "chain_npz" / row["chain_npz"])
        total_res += d["ca"].shape[0]
        miss_n += int(np.isnan(d["n"]).any(axis=1).sum())
        miss_ca += int(np.isnan(d["ca"]).any(axis=1).sum())
        miss_c += int(np.isnan(d["c"]).any(axis=1).sum())
        miss_o += int(np.isnan(d["o"]).any(axis=1).sum())
    report["check_9_missing_ratio_CA_N_C_O"] = {
        "total_residues": int(total_res),
        "missing_ratio_CA": float(miss_ca / total_res) if total_res else 0.0,
        "missing_ratio_N": float(miss_n / total_res) if total_res else 0.0,
        "missing_ratio_C": float(miss_c / total_res) if total_res else 0.0,
        "missing_ratio_O": float(miss_o / total_res) if total_res else 0.0,
    }

    # 10) context_group 是否存在，分布是否合理
    context_exists = "context_group" in model_inputs.columns
    report["check_10_context_group"] = {
        "exists": context_exists,
        "distribution": model_inputs["context_group"].value_counts().to_dict() if context_exists else {},
        "note": "context_group not found in current schema" if not context_exists else "present",
    }
    if not context_exists:
        issues.append({"check": 10, "chain_uid": "", "detail": "context_group column missing"})

    # auxiliary sanity
    report["aux_counts"] = {
        "n_model_inputs": int(len(model_inputs)),
        "n_chains": int(len(chains)),
        "n_regions": int(len(regions)),
        "n_graph_cache": int(len(graph_index)),
    }

    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(issues).to_csv(out_issues, index=False)
    print(json.dumps(report, indent=2))
    print(f"issues: {len(issues)} -> {out_issues}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=Path, default=Path("data/structure_pipeline"))
    args = ap.parse_args()
    validate(args.base_dir)


if __name__ == "__main__":
    main()
