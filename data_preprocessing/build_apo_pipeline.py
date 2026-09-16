import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J",
    "UNK": "X",
}


def _to_float(x: str) -> float:
    return math.nan if x in {".", "?"} else float(x)


def _to_int(x: str) -> Optional[int]:
    if x in {".", "?"}:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _parse_selection_token(token: str) -> Tuple[str, int, str]:
    chain, pos = token.split("_", 1)
    if pos[-1].isalpha():
        return chain, int(pos[:-1]), pos[-1].upper()
    return chain, int(pos), ""


def _parse_mmcif_atom_site(
    cif_path: Path,
) -> Dict[Tuple[str, str, int, str, str], Dict[str, Tuple[float, float, float, float]]]:
    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    i = 0
    residues: Dict[Tuple[str, str, int, str, str], Dict[str, Tuple[float, float, float, float]]] = {}

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
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.auth_asym_id",
            "_atom_site.label_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.label_comp_id",
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

            atom = parts[h["_atom_site.label_atom_id"]]
            alt = parts[h["_atom_site.label_alt_id"]]
            x = _to_float(parts[h["_atom_site.Cartn_x"]])
            y = _to_float(parts[h["_atom_site.Cartn_y"]])
            z = _to_float(parts[h["_atom_site.Cartn_z"]])
            occ = _to_float(parts[h["_atom_site.occupancy"]])
            auth_chain = parts[h["_atom_site.auth_asym_id"]]
            label_chain = parts[h["_atom_site.label_asym_id"]]
            auth_seq = _to_int(parts[h["_atom_site.auth_seq_id"]])
            ins = parts[h["_atom_site.pdbx_PDB_ins_code"]]
            if ins in {".", "?"}:
                ins = ""
            resname = parts[h["_atom_site.label_comp_id"]]
            if auth_seq is None:
                k += 1
                continue
            key = (auth_chain, label_chain, auth_seq, ins, resname)
            if key not in residues:
                residues[key] = {}
            prev = residues[key].get(atom)
            if prev is None:
                residues[key][atom] = (x, y, z, occ, alt)
            else:
                prev_occ = prev[3] if not math.isnan(prev[3]) else -1.0
                cur_occ = occ if not math.isnan(occ) else -1.0
                prev_alt = prev[4]
                prev_pref = 0 if prev_alt in {".", "?", "A"} else 1
                cur_pref = 0 if alt in {".", "?", "A"} else 1
                if (cur_pref, -cur_occ) < (prev_pref, -prev_occ):
                    residues[key][atom] = (x, y, z, occ, alt)
            k += 1
        i = k + 1
    return residues


def _build_cif_index(cif_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in cif_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".cif":
            out[p.stem.lower()] = p
    return out


def _resolve_cif_path(cif_index: Dict[str, Path], apo_pdb_id: str) -> Optional[Path]:
    return cif_index.get(str(apo_pdb_id).lower())


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _pseudo_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    # Same geometric construction style as GVP sidechain proxy.
    c_vec = _normalize(c - ca)
    n_vec = _normalize(n - ca)
    bisector = _normalize(c_vec + n_vec)
    perp = _normalize(np.cross(c_vec, n_vec))
    vec = -bisector * math.sqrt(1.0 / 3.0) - perp * math.sqrt(2.0 / 3.0)
    return ca + vec


def _build_chain_residues(
    parsed: Dict[Tuple[str, str, int, str, str], Dict[str, Tuple[float, float, float, float]]],
    chain_token: str,
) -> List[Dict]:
    rows = []
    chain_ids = [c.strip() for c in str(chain_token).split("-") if c.strip()]
    for chain_id in chain_ids:
        chain_res = [(k, v) for k, v in parsed.items() if k[0] == chain_id or k[1] == chain_id]
        chain_res.sort(key=lambda kv: (kv[0][2], kv[0][3]))
        for k, atom_map in chain_res:
            auth_chain, label_chain, auth_seq, ins, resname = k
            if "CA" not in atom_map:
                continue
            row = {
                "chain_token": chain_token,
                "auth_chain": auth_chain,
                "label_chain": label_chain,
                "auth_seq_id": auth_seq,
                "ins_code": ins,
                "resname": resname,
                "x_ca": atom_map["CA"][0],
                "y_ca": atom_map["CA"][1],
                "z_ca": atom_map["CA"][2],
                "x_n": atom_map["N"][0] if "N" in atom_map else np.nan,
                "y_n": atom_map["N"][1] if "N" in atom_map else np.nan,
                "z_n": atom_map["N"][2] if "N" in atom_map else np.nan,
                "x_c": atom_map["C"][0] if "C" in atom_map else np.nan,
                "y_c": atom_map["C"][1] if "C" in atom_map else np.nan,
                "z_c": atom_map["C"][2] if "C" in atom_map else np.nan,
                "x_o": atom_map["O"][0] if "O" in atom_map else np.nan,
                "y_o": atom_map["O"][1] if "O" in atom_map else np.nan,
                "z_o": atom_map["O"][2] if "O" in atom_map else np.nan,
                "x_cb_raw": atom_map["CB"][0] if "CB" in atom_map else np.nan,
                "y_cb_raw": atom_map["CB"][1] if "CB" in atom_map else np.nan,
                "z_cb_raw": atom_map["CB"][2] if "CB" in atom_map else np.nan,
                "has_N": int("N" in atom_map),
                "has_CA": int("CA" in atom_map),
                "has_C": int("C" in atom_map),
                "has_O": int("O" in atom_map),
                "has_CB_raw": int("CB" in atom_map),
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["auth_chain"], r["auth_seq_id"], r["ins_code"]))
    for i, r in enumerate(rows):
        r["res_idx0"] = i

    if rows:
        n = np.array([[r["x_n"], r["y_n"], r["z_n"]] for r in rows], dtype=np.float32)
        ca = np.array([[r["x_ca"], r["y_ca"], r["z_ca"]] for r in rows], dtype=np.float32)
        c = np.array([[r["x_c"], r["y_c"], r["z_c"]] for r in rows], dtype=np.float32)
        has_backbone = np.array([r["has_N"] and r["has_CA"] and r["has_C"] for r in rows], dtype=bool)
        pseudo = np.full((len(rows), 3), np.nan, dtype=np.float32)
        valid_idx = np.where(has_backbone)[0]
        if len(valid_idx) > 0:
            pseudo[valid_idx] = _pseudo_cb(n[valid_idx], ca[valid_idx], c[valid_idx]).astype(np.float32)
        cb_raw = np.array([[r["x_cb_raw"], r["y_cb_raw"], r["z_cb_raw"]] for r in rows], dtype=np.float32)
        cb = np.where(np.isnan(cb_raw), pseudo, cb_raw)
        for i, r in enumerate(rows):
            r["x_cb"] = float(cb[i, 0]) if not np.isnan(cb[i, 0]) else np.nan
            r["y_cb"] = float(cb[i, 1]) if not np.isnan(cb[i, 1]) else np.nan
            r["z_cb"] = float(cb[i, 2]) if not np.isnan(cb[i, 2]) else np.nan
            r["has_cb"] = int(not np.isnan(cb[i]).any())
    return rows


def _load_split_index(folds_dir: Path) -> Dict[Tuple[str, str, str, str, str, str], str]:
    idx: Dict[Tuple[str, str, str, str, str, str], str] = {}

    def ingest(path: Path, split: str):
        d = json.loads(path.read_text(encoding="utf-8"))
        for apo_pdb_id, records in d.items():
            for r in records:
                key = (
                    apo_pdb_id,
                    str(r["apo_chain"]),
                    str(r.get("holo_pdb_id")),
                    str(r.get("holo_chain")),
                    str(r.get("ligand_chain")),
                    str(r.get("ligand_index")),
                )
                idx[key] = split

    ingest(folds_dir / "test.json", "test")
    for i in range(4):
        ingest(folds_dir / f"train-fold-{i}.json", f"train_fold_{i}")
    return idx


def build_entities(dataset_json: Path, cif_dir: Path, folds_dir: Path, out_dir: Path) -> None:
    entities_dir = out_dir / "entities"
    chain_npz_dir = entities_dir / "chain_npz"
    entities_dir.mkdir(parents=True, exist_ok=True)
    chain_npz_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    split_index = _load_split_index(folds_dir)

    chain_cache: Dict[Tuple[str, str], Dict] = {}
    parsed_cache: Dict[str, Dict] = {}
    cif_index = _build_cif_index(cif_dir)
    regions = []
    sample_regions = []

    region_id = 0
    for apo_pdb_id, records in data.items():
        cif_path = _resolve_cif_path(cif_index, apo_pdb_id)
        if cif_path is None:
            continue
        cif_key = str(cif_path)
        if cif_key not in parsed_cache:
            parsed_cache[cif_key] = _parse_mmcif_atom_site(cif_path)
        parsed = parsed_cache[cif_key]

        for r in records:
            chain_token = str(r["apo_chain"])
            chain_key = (apo_pdb_id, chain_token)
            if chain_key not in chain_cache:
                residues = _build_chain_residues(parsed, chain_token)
                if not residues:
                    continue
                chain_uid = f"{apo_pdb_id}__{chain_token}".replace("/", "_")
                chain_cache[chain_key] = {
                    "chain_uid": chain_uid,
                    "residues": residues,
                    "labels_union": np.zeros(len(residues), dtype=np.int8),
                    "region_labels": [],
                    "region_ids": [],
                }

            chain_obj = chain_cache[chain_key]
            residues = chain_obj["residues"]
            lookup = {}
            for rr in residues:
                i = rr["res_idx0"]
                lookup[(rr["chain_token"], rr["auth_seq_id"], rr["ins_code"])] = i
                lookup[(rr["auth_chain"], rr["auth_seq_id"], rr["ins_code"])] = i
                lookup[(rr["label_chain"], rr["auth_seq_id"], rr["ins_code"])] = i

            y = np.zeros(len(residues), dtype=np.int8)
            mapped = 0
            unresolved = 0
            for tok in r["apo_pocket_selection"]:
                c, seq, ins = _parse_selection_token(tok)
                idx = lookup.get((c, seq, ins))
                if idx is None:
                    unresolved += 1
                else:
                    y[idx] = 1
                    mapped += 1

            chain_obj["labels_union"] = np.maximum(chain_obj["labels_union"], y)
            chain_obj["region_labels"].append(y)
            chain_obj["region_ids"].append(region_id)

            split_key = (
                apo_pdb_id,
                chain_token,
                str(r.get("holo_pdb_id")),
                str(r.get("holo_chain")),
                str(r.get("ligand_chain")),
                str(r.get("ligand_index")),
            )
            split = split_index.get(split_key, "unknown")

            region = {
                "region_id": region_id,
                "chain_uid": chain_obj["chain_uid"],
                "apo_pdb_id": apo_pdb_id,
                "apo_chain": chain_token,
                "holo_pdb_id": r.get("holo_pdb_id"),
                "holo_chain": r.get("holo_chain"),
                "ligand": r.get("ligand"),
                "ligand_chain": r.get("ligand_chain"),
                "ligand_index": r.get("ligand_index"),
                "uniprot_id": r.get("uniprot_id"),
                "pRMSD": r.get("pRMSD"),
                "is_main_holo_structure": r.get("is_main_holo_structure"),
                "selection_size": len(r["apo_pocket_selection"]),
                "mapped_selection": mapped,
                "unresolved_selection": unresolved,
                "mapping_coverage": mapped / len(r["apo_pocket_selection"]) if r["apo_pocket_selection"] else 0.0,
                "split": split,
            }
            regions.append(region)
            sample_regions.append(
                {
                    "region_id": region_id,
                    "chain_uid": chain_obj["chain_uid"],
                    "apo_pocket_selection": json.dumps(r["apo_pocket_selection"]),
                    "holo_pocket_selection": json.dumps(r.get("holo_pocket_selection", [])),
                    "apo_pymol_selection": r.get("apo_pymol_selection"),
                    "holo_pymol_selection": r.get("holo_pymol_selection"),
                }
            )
            region_id += 1

    chain_rows = []
    for (apo_pdb_id, chain_token), obj in chain_cache.items():
        residues = obj["residues"]
        y_union = obj["labels_union"]
        y_regions = np.stack(obj["region_labels"], axis=0) if obj["region_labels"] else np.zeros((0, len(residues)), dtype=np.int8)
        region_ids = np.array(obj["region_ids"], dtype=np.int32)
        chain_uid = obj["chain_uid"]

        ca = np.array([[r["x_ca"], r["y_ca"], r["z_ca"]] for r in residues], dtype=np.float32)
        n = np.array([[r["x_n"], r["y_n"], r["z_n"]] for r in residues], dtype=np.float32)
        c = np.array([[r["x_c"], r["y_c"], r["z_c"]] for r in residues], dtype=np.float32)
        o = np.array([[r["x_o"], r["y_o"], r["z_o"]] for r in residues], dtype=np.float32)
        cb = np.array([[r["x_cb"], r["y_cb"], r["z_cb"]] for r in residues], dtype=np.float32)
        auth_seq = np.array([r["auth_seq_id"] for r in residues], dtype=np.int32)
        has_backbone = np.array([int(r["has_N"] and r["has_CA"] and r["has_C"] and r["has_O"]) for r in residues], dtype=np.int8)
        has_cb = np.array([r["has_cb"] for r in residues], dtype=np.int8)
        resname = np.array([str(r["resname"]) for r in residues], dtype="<U8")
        aa1 = np.array([AA3_TO_1.get(str(r["resname"]).upper(), "X") for r in residues], dtype="<U1")
        auth_chain = np.array([str(r["auth_chain"]) for r in residues], dtype="<U8")
        label_chain = np.array([str(r["label_chain"]) for r in residues], dtype="<U8")
        ins_code = np.array([str(r["ins_code"]) for r in residues], dtype="<U4")

        npz_path = chain_npz_dir / f"{chain_uid}.npz"
        np.savez_compressed(
            npz_path,
            y_union=y_union,
            y_regions=y_regions,
            region_ids=region_ids,
            ca=ca,
            n=n,
            c=c,
            o=o,
            cb=cb,
            auth_seq_id=auth_seq,
            auth_chain=auth_chain,
            label_chain=label_chain,
            ins_code=ins_code,
            resname=resname,
            aa1=aa1,
            has_backbone=has_backbone,
            has_cb=has_cb,
        )

        chain_rows.append(
            {
                "chain_uid": chain_uid,
                "apo_pdb_id": apo_pdb_id,
                "apo_chain": chain_token,
                "n_residues": len(residues),
                "n_regions": len(obj["region_ids"]),
                "n_positive_union": int(y_union.sum()),
                "chain_npz": npz_path.name,
            }
        )

    chains_df = pd.DataFrame(chain_rows)
    regions_df = pd.DataFrame(regions)
    selections_df = pd.DataFrame(sample_regions)

    # split at chain-level by majority of regions, with test priority.
    split_rank = {"test": 3, "train_fold_0": 2, "train_fold_1": 2, "train_fold_2": 2, "train_fold_3": 2, "unknown": 1}
    chain_split_rows = []
    if not regions_df.empty:
        for chain_uid, g in regions_df.groupby("chain_uid"):
            splits = g["split"].tolist()
            best = sorted(splits, key=lambda s: split_rank.get(s, 0), reverse=True)[0]
            chain_split_rows.append({"chain_uid": chain_uid, "split": best, "is_test": int(best == "test")})
    chain_split_df = pd.DataFrame(chain_split_rows)

    # explicit model input columns vs metadata columns
    model_input_df = chains_df.merge(chain_split_df, on="chain_uid", how="left")
    metadata_df = regions_df.merge(selections_df, on=["region_id", "chain_uid"], how="left")

    chains_df.to_csv(entities_dir / "chains.csv", index=False)
    regions_df.to_csv(entities_dir / "regions.csv", index=False)
    chain_split_df.to_csv(entities_dir / "chain_splits.csv", index=False)
    model_input_df.to_csv(entities_dir / "model_inputs.csv", index=False)
    metadata_df.to_csv(entities_dir / "metadata_regions.csv", index=False)

    summary = {
        "n_chains": int(len(chains_df)),
        "n_regions": int(len(regions_df)),
        "mean_regions_per_chain": float(chains_df["n_regions"].mean()) if not chains_df.empty else 0.0,
        "mean_mapping_coverage": float(regions_df["mapping_coverage"].mean()) if not regions_df.empty else 0.0,
        "n_test_chains": int((chain_split_df["is_test"] == 1).sum()) if not chain_split_df.empty else 0,
    }
    (entities_dir / "summary_entities.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_graph_cache(out_dir: Path, dist_cutoff: float, max_neighbors: int) -> None:
    entities_dir = out_dir / "entities"
    chain_npz_dir = entities_dir / "chain_npz"
    graph_dir = out_dir / "graph_cache"
    graph_dir.mkdir(parents=True, exist_ok=True)

    params = {"dist_cutoff": dist_cutoff, "max_neighbors": max_neighbors, "coord": "CB_fallback_pseudoCB"}
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    (graph_dir / "graph_params.json").write_text(json.dumps({"hash": params_hash, **params}, indent=2), encoding="utf-8")

    rows = []
    for npz_path in chain_npz_dir.glob("*.npz"):
        data = np.load(npz_path)
        cb = data["cb"].astype(np.float32)
        ca = data["ca"].astype(np.float32)
        coords = np.where(np.isnan(cb), ca, cb)
        n = coords.shape[0]
        if n == 0:
            continue
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        nearest = np.argsort(d, axis=1)[:, :max_neighbors]
        mask = d[np.arange(n)[:, None], nearest] < dist_cutoff
        neigh = np.where(mask, nearest, -1).astype(np.int32)

        # build per-anchor subgraph edge compression
        edge_src = []
        edge_dst = []
        edge_ptr = [0]
        for i in range(n):
            idx = neigh[i][neigh[i] >= 0]
            sub = d[np.ix_(idx, idx)]
            ii, jj = np.where((sub < dist_cutoff) & (sub > 0))
            edge_src.extend(ii.tolist())
            edge_dst.extend(jj.tolist())
            edge_ptr.append(len(edge_src))

        out = graph_dir / f"{npz_path.stem}__{params_hash}.npz"
        np.savez_compressed(
            out,
            neighbor_idx=neigh,
            edge_src=np.array(edge_src, dtype=np.int32),
            edge_dst=np.array(edge_dst, dtype=np.int32),
            edge_ptr=np.array(edge_ptr, dtype=np.int64),
        )
        rows.append(
            {
                "chain_uid": npz_path.stem,
                "graph_file": out.name,
                "n_residues": int(n),
                "params_hash": params_hash,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(graph_dir / "graph_cache_index.csv", index=False)
    print(
        json.dumps(
            {"n_cached_chains": int(len(df)), "params_hash": params_hash, "graph_cache_index": str(graph_dir / "graph_cache_index.csv")},
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser(description="Two-stage CryptoBench APO pipeline: build_entities + build_graph_cache")
    ap.add_argument("--stage", choices=["build_entities", "build_graph_cache", "all"], default="all")
    ap.add_argument("--dataset_json", type=Path, default=Path("data/raw/dataset.json"))
    ap.add_argument("--cif_dir", type=Path, default=Path("data/raw/cif-files"))
    ap.add_argument("--folds_dir", type=Path, default=Path("data/raw/folds"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/structure_pipeline"))
    ap.add_argument("--dist_cutoff", type=float, default=10.0)
    ap.add_argument("--max_neighbors", type=int, default=40)
    args = ap.parse_args()

    if args.stage in {"build_entities", "all"}:
        build_entities(args.dataset_json, args.cif_dir, args.folds_dir, args.out_dir)
    if args.stage in {"build_graph_cache", "all"}:
        build_graph_cache(args.out_dir, args.dist_cutoff, args.max_neighbors)


if __name__ == "__main__":
    main()
