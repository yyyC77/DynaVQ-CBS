#!/usr/bin/env python3
import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_float(x: str) -> float:
    return math.nan if x in {".", "?"} else float(x)


def _to_int(x: str) -> Optional[int]:
    if x in {".", "?"}:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    m = np.isfinite(x)
    if not np.any(m):
        return out
    xm = x[m]
    mu = float(np.mean(xm))
    sd = float(np.std(xm))
    if sd < 1e-8:
        out[m] = 0.0
    else:
        out[m] = (xm - mu) / sd
    return out


def _parse_ca_bfactor_map(cif_path: Path) -> Dict[Tuple[str, int, str], float]:
    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    i = 0
    out: Dict[Tuple[str, int, str], Tuple[float, float, str]] = {}

    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        headers: List[str] = []
        while j < n and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if not headers or not headers[0].startswith("_atom_site."):
            i = j
            continue

        h = {name: idx for idx, name in enumerate(headers)}
        required = [
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
        ]
        if any(x not in h for x in required):
            i = j
            continue

        k = j
        while k < n:
            raw = lines[k].strip()
            if not raw or raw.startswith("#") or raw.startswith("_") or raw == "loop_":
                break
            try:
                parts = shlex.split(raw, posix=True)
            except ValueError:
                parts = raw.split()
            if len(parts) < len(headers):
                k += 1
                continue
            if parts[h["_atom_site.group_PDB"]] != "ATOM":
                k += 1
                continue
            if parts[h["_atom_site.label_atom_id"]] != "CA":
                k += 1
                continue

            auth_chain = parts[h["_atom_site.auth_asym_id"]]
            auth_seq = _to_int(parts[h["_atom_site.auth_seq_id"]])
            ins = parts[h["_atom_site.pdbx_PDB_ins_code"]]
            if ins in {".", "?"}:
                ins = ""
            if auth_seq is None:
                k += 1
                continue

            occ = _to_float(parts[h["_atom_site.occupancy"]])
            b = _to_float(parts[h["_atom_site.B_iso_or_equiv"]])
            alt = parts[h["_atom_site.label_alt_id"]]
            key = (auth_chain, int(auth_seq), ins)

            prev = out.get(key)
            cur_occ = occ if np.isfinite(occ) else -1.0
            cur_pref = 0 if alt in {".", "?", "A"} else 1
            if prev is None:
                out[key] = (b, cur_occ, alt)
            else:
                prev_b, prev_occ, prev_alt = prev
                prev_pref = 0 if prev_alt in {".", "?", "A"} else 1
                if (cur_pref, -cur_occ) < (prev_pref, -prev_occ):
                    out[key] = (b, cur_occ, alt)
            k += 1
        i = k + 1

    return {k: float(v[0]) for k, v in out.items()}


def _distance_to_breaks(auth_chain: np.ndarray, auth_seq: np.ndarray) -> np.ndarray:
    # Breaks are sequence jumps >1 inside same chain or chain transitions.
    n = len(auth_seq)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    breaks = np.zeros((n,), dtype=bool)
    breaks[0] = True
    for i in range(1, n):
        if auth_chain[i] != auth_chain[i - 1]:
            breaks[i] = True
            continue
        d = int(auth_seq[i]) - int(auth_seq[i - 1])
        if d > 1 or d < 0:
            breaks[i] = True
    br_idx = np.where(breaks)[0]
    dist = np.full((n,), n, dtype=np.int32)
    for i in range(n):
        dist[i] = int(np.min(np.abs(br_idx - i))) if len(br_idx) > 0 else n
    return dist.astype(np.float32)


def _build_graph_density(graph_npz: Path, n_res: int) -> np.ndarray:
    d = np.load(graph_npz)
    neigh = d["neighbor_idx"]
    # Exclude invalid neighbor and self-neighbor.
    idx = np.arange(neigh.shape[0])[:, None]
    valid = (neigh >= 0) & (neigh != idx)
    cnt = valid.sum(axis=1).astype(np.float32)
    denom = max(neigh.shape[1] - 1, 1)
    dens = cnt / float(denom)
    if int(dens.shape[0]) != int(n_res):
        raise ValueError(f"graph length mismatch: {graph_npz.name}, graph={dens.shape[0]}, chain={n_res}")
    return dens


def main():
    ap = argparse.ArgumentParser(description="Build flexibility feature cache aligned to chain_npz residues.")
    ap.add_argument("--pipeline_dir", type=Path, required=True, help="pipeline dir with entities/ and graph_cache/")
    ap.add_argument("--cif_dir", type=Path, required=True, help="directory with APO cif files (lowercase stem by pdb id)")
    ap.add_argument("--out_dir", type=Path, default=None, help="output flex cache dir (default: <pipeline_dir>/flex_cache)")
    ap.add_argument("--graph_cache_dir", type=Path, default=None, help="default: <pipeline_dir>/graph_cache")
    ap.add_argument("--limit_chains", type=int, default=0, help="for smoke test; 0 means all")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else (args.pipeline_dir / "flex_cache")
    graph_cache_dir = args.graph_cache_dir if args.graph_cache_dir is not None else (args.pipeline_dir / "graph_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    chains_csv = args.pipeline_dir / "entities" / "chains.csv"
    chain_npz_dir = args.pipeline_dir / "entities" / "chain_npz"
    graph_idx_csv = graph_cache_dir / "graph_cache_index.csv"

    chains_df = pd.read_csv(chains_csv)
    graph_df = pd.read_csv(graph_idx_csv)
    graph_map = {str(r["chain_uid"]): str(r["graph_file"]) for _, r in graph_df.iterrows()}

    cif_cache: Dict[str, Dict[Tuple[str, int, str], float]] = {}
    stats = {
        "n_total": 0,
        "n_done": 0,
        "n_missing_cif": 0,
        "n_missing_graph": 0,
        "n_shape_mismatch": 0,
        "n_bfactor_missing_res": 0,
        "n_res_total": 0,
    }

    feature_names = np.array(
        [
            "bfactor_z",
            "bfactor_missing",
            "contact_density",
            "gap_proximity",
        ],
        dtype="<U32",
    )

    for i, row in enumerate(chains_df.itertuples(index=False), start=1):
        if args.limit_chains > 0 and i > args.limit_chains:
            break
        stats["n_total"] += 1

        uid = str(row.chain_uid)
        pdb_id = str(row.apo_pdb_id).lower()
        npz_path = chain_npz_dir / str(row.chain_npz)
        if not npz_path.exists():
            stats["n_shape_mismatch"] += 1
            continue
        gname = graph_map.get(uid)
        if gname is None:
            stats["n_missing_graph"] += 1
            continue
        gpath = graph_cache_dir / gname
        if not gpath.exists():
            stats["n_missing_graph"] += 1
            continue

        cif_path = args.cif_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            stats["n_missing_cif"] += 1
            continue

        if pdb_id not in cif_cache:
            cif_cache[pdb_id] = _parse_ca_bfactor_map(cif_path)
        bmap = cif_cache[pdb_id]

        d = np.load(npz_path)
        auth_chain = np.array([str(x) for x in d["auth_chain"]], dtype=object)
        auth_seq = np.array(d["auth_seq_id"], dtype=np.int32)
        ins_code = np.array([str(x) for x in d["ins_code"]], dtype=object)
        n_res = int(auth_seq.shape[0])

        b_raw = np.full((n_res,), np.nan, dtype=np.float32)
        for j in range(n_res):
            key = (auth_chain[j], int(auth_seq[j]), "" if ins_code[j] in {".", "?"} else str(ins_code[j]))
            if key in bmap:
                b_raw[j] = float(bmap[key])

        b_missing = (~np.isfinite(b_raw)).astype(np.float32)
        stats["n_bfactor_missing_res"] += int(b_missing.sum())
        b_z = _safe_zscore(b_raw)
        b_z = np.where(np.isfinite(b_raw), b_z, 0.0).astype(np.float32)

        try:
            c_dens = _build_graph_density(gpath, n_res).astype(np.float32)
        except Exception:
            stats["n_shape_mismatch"] += 1
            continue

        d2b = _distance_to_breaks(auth_chain.astype(str), auth_seq.astype(np.int32))
        # Higher value means closer to potential unresolved segment boundary.
        gap_prox = np.exp(-d2b / 3.0).astype(np.float32)

        flex_scalar = np.stack([b_z, b_missing, c_dens, gap_prox], axis=1).astype(np.float32)
        flex_mask = np.isfinite(flex_scalar).astype(np.int8)
        flex_scalar = np.nan_to_num(flex_scalar, nan=0.0, posinf=0.0, neginf=0.0)

        np.savez_compressed(
            out_dir / f"{uid}.npz",
            flex_scalar=flex_scalar,
            flex_mask=flex_mask,
            feature_names=feature_names,
            chain_uid=np.array(uid),
        )
        stats["n_done"] += 1
        stats["n_res_total"] += n_res

        if i % 200 == 0:
            print(
                f"[progress] chains={i} done={stats['n_done']} missing_cif={stats['n_missing_cif']} "
                f"missing_graph={stats['n_missing_graph']}"
            )

    summary = {
        **stats,
        "pipeline_dir": str(args.pipeline_dir),
        "cif_dir": str(args.cif_dir),
        "graph_cache_dir": str(graph_cache_dir),
        "out_dir": str(out_dir),
        "feature_names": feature_names.tolist(),
        "bfactor_missing_rate": float(stats["n_bfactor_missing_res"] / max(stats["n_res_total"], 1)),
    }
    (out_dir / "summary_flex_cache.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

