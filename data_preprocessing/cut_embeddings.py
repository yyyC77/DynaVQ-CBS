#!/usr/bin/env python3
"""
Cut whole-UniProt ESM embeddings into PDB-observed chain embeddings and
write translated annotations for APoLo/CryptoBench style training.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

HEADER = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
USE_ENV_PROXY = False


def _retry_json(url: str, timeout: int, retries: int, sleep_s: float) -> dict:
    last_err = None
    for _ in range(max(retries, 1)):
        try:
            kwargs = {"headers": HEADER, "timeout": timeout}
            if not USE_ENV_PROXY:
                kwargs["proxies"] = {"http": None, "https": None}
            r = requests.get(url, **kwargs)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_s)
    raise RuntimeError(f"request failed: {url}; err={last_err}")


def get_entity_id(pdb_id: str, chain_id: str, timeout: int, retries: int, sleep_s: float) -> str:
    data = _retry_json(
        f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}",
        timeout=timeout,
        retries=retries,
        sleep_s=sleep_s,
    )
    for entity in data[pdb_id]:
        if chain_id in entity.get("in_chains", []):
            return str(entity["entity_id"])
    raise RuntimeError(f"entity_id not found for {pdb_id}/{chain_id}")


def get_pdb_mapping_range(
    pdb_id: str, entity_id: str, chain_id: str, uniprot_id: str, timeout: int, retries: int, sleep_s: float
) -> Dict[str, int]:
    # Preferred endpoint (works in current PDBe API):
    # /pdbe/api/mappings/uniprot/{pdb_id}
    data = _retry_json(
        f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}",
        timeout=timeout,
        retries=retries,
        sleep_s=sleep_s,
    )
    payload = data.get(pdb_id, {})
    uni = payload.get("UniProt", {})
    requested_uid = uniprot_id
    if uniprot_id not in uni:
        # fallback: use first key matching case-insensitive
        k2 = [k for k in uni.keys() if str(k).upper() == str(uniprot_id).upper()]
        if not k2:
            # Degrade gracefully: try chain/entity-based mapping regardless of accession.
            k2 = []
            for _uid, _obj in uni.items():
                for m in _obj.get("mappings", []):
                    if (
                        str(m.get("entity_id")) == str(entity_id)
                        and (str(m.get("chain_id", "")) == chain_id or str(m.get("struct_asym_id", "")) == chain_id)
                    ):
                        k2 = [_uid]
                        break
                if k2:
                    break
            if not k2:
                raise RuntimeError(f"uniprot accession not found in mapping: {pdb_id}/{entity_id}/{requested_uid}")
        uniprot_id = k2[0]
    mappings = uni[uniprot_id].get("mappings", [])
    cands = [
        m
        for m in mappings
        if str(m.get("entity_id")) == str(entity_id)
        and (str(m.get("chain_id", "")) == chain_id or str(m.get("struct_asym_id", "")) == chain_id)
    ]
    if not cands:
        cands = [m for m in mappings if str(m.get("entity_id")) == str(entity_id)]
    if not cands:
        raise RuntimeError(f"no mapping rows for entity: {pdb_id}/{entity_id}/{uniprot_id}")

    # PDB author residue interval used by residue_mapping endpoint.
    best = None
    best_cov = -1
    for m in cands:
        st = m.get("start") or {}
        ed = m.get("end") or {}
        # New API may provide null author_residue_number; fallback to residue_number.
        s = st.get("author_residue_number")
        e = ed.get("author_residue_number")
        if s is None:
            s = st.get("residue_number")
        if e is None:
            e = ed.get("residue_number")
        if s is None or e is None:
            continue
        try:
            s_i = int(s)
            e_i = int(e)
        except Exception:
            continue
        if e_i < s_i:
            continue
        cov = e_i - s_i + 1
        unp_s = m.get("unp_start")
        unp_e = m.get("unp_end")
        if unp_s is None or unp_e is None:
            continue
        if cov > best_cov:
            best_cov = cov
            best = {
                "pdb_start": int(s_i),
                "pdb_end": int(e_i),
                "unp_start": int(unp_s),
                "unp_end": int(unp_e),
                "resolved_uniprot_id": str(uniprot_id),
            }
    if best is None:
        # Fallback: use any mapping row for this UniProt with valid numeric range,
        # even if chain-specific author numbering is absent (common in some entries).
        for m in mappings:
            st = m.get("start") or {}
            ed = m.get("end") or {}
            s = st.get("author_residue_number")
            e = ed.get("author_residue_number")
            if s is None:
                s = st.get("residue_number")
            if e is None:
                e = ed.get("residue_number")
            if s is None or e is None:
                continue
            try:
                s_i = int(s)
                e_i = int(e)
            except Exception:
                continue
            if e_i >= s_i:
                unp_s = m.get("unp_start")
                unp_e = m.get("unp_end")
                if unp_s is None or unp_e is None:
                    continue
                best = {
                    "pdb_start": int(s_i),
                    "pdb_end": int(e_i),
                    "unp_start": int(unp_s),
                    "unp_end": int(unp_e),
                    "resolved_uniprot_id": str(uniprot_id),
                }
                break
    if best is None:
        raise RuntimeError(f"no valid author residue range: {pdb_id}/{entity_id}/{uniprot_id}")
    return best


def get_residues_mapping(
    pdb_id: str, entity_id: str, chain_id: str, start: int, end: int, timeout: int, retries: int, sleep_s: float
) -> List[dict]:
    data = _retry_json(
        f"https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/{pdb_id}/{entity_id}/{start}/{end}",
        timeout=timeout,
        retries=retries,
        sleep_s=sleep_s,
    )
    rows = [i for i in data[pdb_id] if str(i.get("entity_id")) == str(entity_id)]
    if not rows:
        raise RuntimeError(f"entity row missing in residue_mapping: {pdb_id}/{entity_id}")
    chains = [c for c in rows[0].get("chains", []) if str(c.get("auth_asym_id", "")) == chain_id]
    if not chains:
        raise RuntimeError(f"chain row missing in residue_mapping: {pdb_id}/{entity_id}/{chain_id}")
    return chains[0].get("residues", [])


def build_residues_fallback(mapping_row: Dict[str, int], uniprot_id: str) -> List[dict]:
    # Approximate fallback when residue_mapping endpoint is unavailable.
    # Build a synthetic contiguous mapping segment from SIFTS starts/ends.
    pdb_s = int(mapping_row["pdb_start"])
    pdb_e = int(mapping_row["pdb_end"])
    unp_s = int(mapping_row["unp_start"])
    unp_e = int(mapping_row["unp_end"])
    n = min(pdb_e - pdb_s + 1, unp_e - unp_s + 1)
    out = []
    for k in range(max(n, 0)):
        author_num = pdb_s + k
        unp_num = unp_s + k
        out.append(
            {
                "observed": "Y",
                "author_residue_number": author_num,
                "features": {"UniProt": {uniprot_id: {"unp_residue_number": unp_num, "unp_one_letter_code": "X"}}},
            }
        )
    return out


def _split_or_broadcast_uids(chain_ids: List[str], raw_uids: List[str]) -> List[str]:
    if len(raw_uids) == len(chain_ids):
        return raw_uids
    if len(raw_uids) == 1 and len(chain_ids) > 1:
        return [raw_uids[0] for _ in chain_ids]
    raise ValueError(f"uniprot/chain count mismatch: chains={chain_ids}, uniprots={raw_uids}")


def main():
    ap = argparse.ArgumentParser(description="Cut full UniProt embeddings to PDB-observed residues.")
    ap.add_argument("--input_csv", type=Path, required=True, help="APoLo-style input annotation CSV.")
    ap.add_argument("--output_csv", type=Path, required=True, help="Translated annotation CSV output path.")
    ap.add_argument("--embeddings_input_dir", type=Path, required=True, help="Dir containing {UniProtID}.npy")
    ap.add_argument("--embeddings_output_dir", type=Path, required=True, help="Dir to write {pdb}{chain}.npy")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sleep_ms", type=int, default=120)
    ap.add_argument("--limit_rows", type=int, default=0, help="Smoke-test limit on input rows. 0 means all.")
    ap.add_argument("--progress_every", type=int, default=20, help="Print progress every N chains.")
    ap.add_argument("--max_consecutive_fail", type=int, default=0, help="Abort early if consecutive failures exceed this value. 0 disables.")
    ap.add_argument("--use_env_proxy", action="store_true", help="Use system proxy env (default off).")
    ap.add_argument("--append_output", action="store_true", help="Append to output CSV instead of truncating it.")
    ap.add_argument("--out_mapping_meta", type=Path, default=None, help="Optional JSONL meta output for rebuilding entities.")
    args = ap.parse_args()

    # Avoid broken proxy env by default; user can re-enable with --use_env_proxy.
    global USE_ENV_PROXY
    USE_ENV_PROXY = bool(args.use_env_proxy)
    if not USE_ENV_PROXY:
        requests.sessions.Session.trust_env = False

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.embeddings_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.append_output:
        args.output_csv.write_text("", encoding="utf-8")
        if args.out_mapping_meta is not None:
            args.out_mapping_meta.parent.mkdir(parents=True, exist_ok=True)
            args.out_mapping_meta.write_text("", encoding="utf-8")
    sleep_s = max(args.sleep_ms, 0) / 1000.0

    n_rows = 0
    n_chain_ok = 0
    n_chain_fail = 0
    n_chain_total = 0
    consecutive_fail = 0
    fails: List[Dict[str, str]] = []
    t0 = time.time()

    # Caches to avoid repeated PDBe calls for repeated (pdb, chain, uniprot).
    entity_cache: Dict[Tuple[str, str], str] = {}
    range_cache: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    residues_cache: Dict[Tuple[str, str, str, int, int], List[dict]] = {}

    with args.input_csv.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if len(row) < 4:
                continue
            if args.limit_rows > 0 and n_rows >= args.limit_rows:
                break
            n_rows += 1

            pdb_id = row[0].strip().lower()
            chain_ids = [x.strip() for x in row[1].split("-") if x.strip()]
            raw_uids = [x.strip() for x in row[2].split("-") if x.strip()]
            annotations = [x.strip() for x in row[3].split(" ") if x.strip()]

            try:
                uniprot_ids = _split_or_broadcast_uids(chain_ids, raw_uids)
            except Exception as e:
                n_chain_fail += len(chain_ids)
                fails.append({"pdb_id": pdb_id, "chain": row[1], "uniprot": row[2], "error": str(e)})
                continue

            for uniprot_id, chain_id in zip(uniprot_ids, chain_ids):
                n_chain_total += 1
                try:
                    build_annotations: List[str] = []
                    indices_of_embedding: List[int] = []
                    chain_annotations = [i.split("_")[1] for i in annotations if i.split("_")[0] == chain_id]

                    ekey = (pdb_id, chain_id)
                    if ekey not in entity_cache:
                        entity_cache[ekey] = get_entity_id(pdb_id, chain_id, args.timeout, args.retries, sleep_s)
                    entity_id = entity_cache[ekey]

                    rkey = (pdb_id, entity_id, chain_id, uniprot_id)
                    if rkey not in range_cache:
                        range_cache[rkey] = get_pdb_mapping_range(
                            pdb_id, entity_id, chain_id, uniprot_id, args.timeout, args.retries, sleep_s
                        )
                    mrow = range_cache[rkey]
                    mapped_uid = str(mrow.get("resolved_uniprot_id", uniprot_id))
                    start, end = int(mrow["pdb_start"]), int(mrow["pdb_end"])

                    mkey = (pdb_id, entity_id, chain_id, start, end)
                    if mkey not in residues_cache:
                        try:
                            residues_cache[mkey] = get_residues_mapping(
                                pdb_id, entity_id, chain_id, start, end, args.timeout, args.retries, sleep_s
                            )
                        except Exception:
                            residues_cache[mkey] = build_residues_fallback(mrow, mapped_uid)
                    residues_mapping = residues_cache[mkey]

                    old_build_annotations_len = len(build_annotations)
                    for residue in residues_mapping:
                        if residue.get("observed") != "Y":
                            continue
                        uni_feat = (residue.get("features") or {}).get("UniProt") or {}
                        uid_for_feat = mapped_uid if mapped_uid in uni_feat else uniprot_id
                        if uid_for_feat not in uni_feat:
                            continue
                        one = uni_feat[uid_for_feat].get("unp_one_letter_code", "X")
                        author_num = str(residue.get("author_residue_number"))
                        if author_num in chain_annotations:
                            build_annotations.append(f"{chain_id}_{one}")
                        else:
                            build_annotations.append("")
                        indices_of_embedding.append(int(uni_feat[uid_for_feat]["unp_residue_number"]) - 1)

                    emb_path = args.embeddings_input_dir / f"{mapped_uid}.npy"
                    if not emb_path.exists():
                        emb_path = args.embeddings_input_dir / f"{uniprot_id}.npy"
                    if not emb_path.exists():
                        raise FileNotFoundError(f"embedding not found: {emb_path}")
                    embedding = np.load(emb_path)
                    new_embedding = np.take(embedding, indices_of_embedding, axis=0)

                    assert len(build_annotations) - old_build_annotations_len == new_embedding.shape[0], (
                        f"{len(build_annotations)} vs {new_embedding.shape[0]}"
                    )

                    out_npy = args.embeddings_output_dir / f"{pdb_id}{chain_id}.npy"
                    np.save(out_npy, new_embedding)

                    new_annotations = []
                    for idx, value in enumerate(build_annotations):
                        if value:
                            new_annotations.append(f"{value}{idx}")
                    concat = " ".join(new_annotations)
                    out_line = f"{pdb_id};{chain_id};{mapped_uid};{concat};UNKNOWN\n"
                    with args.output_csv.open("a", encoding="utf-8") as f:
                        f.write(out_line)
                    if args.out_mapping_meta is not None:
                        meta = {
                            "pdb_id": pdb_id,
                            "chain_id": chain_id,
                            "uniprot_id_input": uniprot_id,
                            "uniprot_id_mapped": mapped_uid,
                            "embedding_file": out_npy.name,
                            "embedding_len": int(new_embedding.shape[0]),
                            "indices_of_embedding": [int(x) for x in indices_of_embedding],
                            "author_residue_numbers_observed": [
                                int(r.get("author_residue_number"))
                                for r in residues_mapping
                                if r.get("observed") == "Y" and r.get("author_residue_number") is not None
                            ],
                            "annotation_tokens": new_annotations,
                        }
                        with args.out_mapping_meta.open("a", encoding="utf-8") as mf:
                            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    n_chain_ok += 1
                    consecutive_fail = 0
                except Exception as e:
                    n_chain_fail += 1
                    consecutive_fail += 1
                    fails.append(
                        {"pdb_id": pdb_id, "chain": chain_id, "uniprot": uniprot_id, "error": str(e)}
                    )
                    if args.max_consecutive_fail > 0 and consecutive_fail >= args.max_consecutive_fail:
                        print(
                            f"[ABORT] consecutive_fail={consecutive_fail} reached max_consecutive_fail={args.max_consecutive_fail}"
                        )
                        break

                if args.progress_every > 0 and (n_chain_total % args.progress_every == 0):
                    elapsed = time.time() - t0
                    print(
                        f"[progress] chains={n_chain_total} ok={n_chain_ok} fail={n_chain_fail} "
                        f"elapsed={elapsed:.1f}s out_csv={args.output_csv}"
                    )
            if args.max_consecutive_fail > 0 and consecutive_fail >= args.max_consecutive_fail:
                break

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "embeddings_input_dir": str(args.embeddings_input_dir),
        "embeddings_output_dir": str(args.embeddings_output_dir),
        "rows_read": n_rows,
        "chain_total": n_chain_total,
        "chain_ok": n_chain_ok,
        "chain_fail": n_chain_fail,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (args.output_csv.parent / f"{args.output_csv.stem}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_csv.parent / f"{args.output_csv.stem}.failures.json").write_text(
        json.dumps(fails, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
