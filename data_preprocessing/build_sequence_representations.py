import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


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


def _to_int(x: str):
    if x in {".", "?"}:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _parse_mmcif_atom_site(cif_path: Path):
    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i, n = 0, len(lines)
    residues = {}
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        headers = []
        while j < n and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if not headers or not headers[0].startswith("_atom_site."):
            i = j
            continue
        h = {k: idx for idx, k in enumerate(headers)}
        req = [
            "_atom_site.group_PDB", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
            "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
            "_atom_site.occupancy", "_atom_site.auth_asym_id", "_atom_site.label_asym_id",
            "_atom_site.auth_seq_id", "_atom_site.pdbx_PDB_ins_code", "_atom_site.label_comp_id",
        ]
        if any(k not in h for k in req):
            i = j
            continue
        k = j
        while k < n:
            raw = lines[k].strip()
            if not raw or raw == "loop_" or raw.startswith("#") or raw.startswith("_"):
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
            res3 = parts[h["_atom_site.label_comp_id"]]
            if auth_seq is None:
                k += 1
                continue
            key = (auth_chain, label_chain, auth_seq, ins, res3)
            if key not in residues:
                residues[key] = {}
            prev = residues[key].get(atom)
            if prev is None:
                residues[key][atom] = (x, y, z, occ, alt)
            else:
                prev_occ = prev[3] if not math.isnan(prev[3]) else -1.0
                cur_occ = occ if not math.isnan(occ) else -1.0
                prev_alt_pref = 0 if prev[4] in {".", "?", "A"} else 1
                cur_alt_pref = 0 if alt in {".", "?", "A"} else 1
                if (cur_alt_pref, -cur_occ) < (prev_alt_pref, -prev_occ):
                    residues[key][atom] = (x, y, z, occ, alt)
            k += 1
        i = k + 1
    return residues


def _build_chain_records(parsed, chain_token: str):
    chain_ids = [c.strip() for c in chain_token.split("-") if c.strip()]
    rows = []
    for cid in chain_ids:
        chain_res = [(k, v) for k, v in parsed.items() if k[0] == cid or k[1] == cid]
        chain_res.sort(key=lambda kv: (kv[0][2], kv[0][3]))
        for k, atom_map in chain_res:
            auth_chain, label_chain, auth_seq, ins, res3 = k
            if "CA" not in atom_map:
                continue
            rows.append(
                {
                    "auth_chain": auth_chain,
                    "label_chain": label_chain,
                    "auth_seq": auth_seq,
                    "ins": ins,
                    "res3": res3,
                    "aa1": AA3_TO_1.get(res3.upper(), "X"),
                }
            )
    rows.sort(key=lambda r: (r["auth_chain"], r["auth_seq"], r["ins"]))
    return rows


def _build_cif_index(cif_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in cif_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".cif":
            out[p.stem.lower()] = p
    return out


def _resolve_cif_path(cif_index: Dict[str, Path], apo_pdb: str) -> Path:
    p = cif_index.get(str(apo_pdb).lower())
    if p is None:
        raise FileNotFoundError(f"Cannot find CIF for apo_pdb_id={apo_pdb}")
    return p


def _stable_seed_from_uid(chain_uid: str) -> int:
    h = hashlib.sha256(chain_uid.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False) % (2**31)


def _decode_str_array(arr: np.ndarray) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def _build_seq_from_chain_npz(npz_obj, length: int) -> str:
    if "aa1" in npz_obj:
        aa1 = _decode_str_array(np.asarray(npz_obj["aa1"]))
        if len(aa1) != length:
            raise ValueError(f"npz aa1 length {len(aa1)} != {length}")
        seq = "".join([(s[:1] if s else "X").upper() for s in aa1])
        if len(seq) != length:
            raise ValueError(f"npz aa1->seq length {len(seq)} != {length}")
        return seq
    if "resname" in npz_obj:
        res3 = _decode_str_array(np.asarray(npz_obj["resname"]))
        if len(res3) != length:
            raise ValueError(f"npz resname length {len(res3)} != {length}")
        seq = "".join([AA3_TO_1.get(r.upper(), "X") for r in res3])
        if len(seq) != length:
            raise ValueError(f"npz resname->seq length {len(seq)} != {length}")
        return seq
    return ""


def _load_esm(model_name: str, device: str, cache_dir: Path = None, local_files_only: bool = False):
    from transformers import EsmModel, EsmTokenizer
    kwargs = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = EsmTokenizer.from_pretrained(model_name, **kwargs)
    model = EsmModel.from_pretrained(model_name, **kwargs)
    model.eval().to(device)
    return tokenizer, model


def _resolve_device(device_arg: str, gpu: int):
    if gpu is not None:
        if gpu < 0:
            return "cpu"
        if not torch.cuda.is_available():
            raise ValueError(f"--gpu {gpu} requested but CUDA is not available.")
        n_gpu = torch.cuda.device_count()
        if gpu >= n_gpu:
            raise ValueError(f"--gpu {gpu} out of range. Available GPU count: {n_gpu}.")
        return f"cuda:{gpu}"
    return device_arg


@torch.no_grad()
def _embed_chunk(tokenizer, model, seq: str, device: str):
    toks = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks).last_hidden_state  # [1, L+2, D]
    emb = out[0, 1:-1, :].detach().cpu()  # [L, D]
    return emb


@torch.no_grad()
def _embed_sequence(tokenizer, model, seq: str, device: str, max_residues: int, chunk_overlap: int):
    L = len(seq)
    if L <= max_residues:
        return _embed_chunk(tokenizer, model, seq, device)

    # Overlap-chunk average for long sequences.
    first = _embed_chunk(tokenizer, model, seq[:max_residues], device)
    D = int(first.shape[1])
    acc = torch.zeros((L, D), dtype=torch.float32)
    cnt = torch.zeros((L, 1), dtype=torch.float32)
    acc[:max_residues] += first.to(torch.float32)
    cnt[:max_residues] += 1.0

    start = max_residues
    while start < L:
        start = max(0, start - chunk_overlap)
        end = min(start + max_residues, L)
        chunk = seq[start:end]
        emb = _embed_chunk(tokenizer, model, chunk, device).to(torch.float32)
        acc[start:end] += emb
        cnt[start:end] += 1.0
        if end == L:
            break
        start = end

    cnt[cnt == 0] = 1.0
    return acc / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_dir", type=Path, default=Path("data/structure_pipeline"))
    ap.add_argument("--cif_dir", type=Path, default=Path("data/raw/cif-files"))
    ap.add_argument("--esm_model", type=str, default="facebook/esm2_t36_3B_UR50D")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index. Overrides --device when set; use -1 for CPU.")
    ap.add_argument("--hf_cache_dir", type=Path, default=None, help="HuggingFace cache directory.")
    ap.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer only from local files.")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--mock", action="store_true", help="Use random embeddings for offline smoke test.")
    ap.add_argument("--mock_dim", type=int, default=2560)
    ap.add_argument("--max_chains", type=int, default=0)
    ap.add_argument("--esm_max_len", type=int, default=0, help="Max residues per ESM forward pass; 0 means auto from model.")
    ap.add_argument("--chunk_overlap", type=int, default=128)
    args = ap.parse_args()
    args.device = _resolve_device(args.device, args.gpu)

    entities_dir = args.pipeline_dir / "entities"
    out_dir = args.out_dir if args.out_dir is not None else (args.pipeline_dir / "esm_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_inputs = pd.read_csv(entities_dir / "model_inputs.csv")
    chains = pd.read_csv(entities_dir / "chains.csv").set_index("chain_uid")

    if args.max_chains > 0:
        model_inputs = model_inputs.head(args.max_chains)

    tokenizer = model = None
    emb_dim = args.mock_dim
    max_residues = 0
    if not args.mock:
        try:
            tokenizer, model = _load_esm(
                args.esm_model,
                args.device,
                cache_dir=args.hf_cache_dir,
                local_files_only=args.local_files_only,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load ESM model. If your network cannot reach huggingface.co, "
                "download model files first and run with --esm_model <local_dir> --local_files_only."
            ) from e
        emb_dim = int(model.config.hidden_size)
        auto_max = int(getattr(model.config, "max_position_embeddings", 1026)) - 2
        max_residues = int(args.esm_max_len) if args.esm_max_len > 0 else auto_max
        if max_residues <= 0:
            raise ValueError(f"Invalid max residues: {max_residues}")
        if args.chunk_overlap < 0 or args.chunk_overlap >= max_residues:
            raise ValueError(f"chunk_overlap must be in [0, {max_residues - 1}]")

    cif_cache: Dict[str, dict] = {}
    cif_index = _build_cif_index(args.cif_dir)
    done = 0
    failed = 0
    for _, row in model_inputs.iterrows():
        chain_uid = row["chain_uid"]
        apo_pdb = row["apo_pdb_id"]
        apo_chain = row["apo_chain"]
        split = row["split"]
        npz_path = entities_dir / "chain_npz" / chains.loc[chain_uid, "chain_npz"]
        d = np.load(npz_path)
        y = torch.tensor(d["y_union"], dtype=torch.float32)
        L = int(y.shape[0])
        mask = torch.ones(L, dtype=torch.bool)

        try:
            if args.mock:
                g = torch.Generator().manual_seed(_stable_seed_from_uid(str(chain_uid)))
                emb = torch.randn(L, emb_dim, generator=g, dtype=torch.float32)
                seq = None
            else:
                seq = _build_seq_from_chain_npz(d, L)
                if seq:
                    emb = _embed_sequence(tokenizer, model, seq, args.device, max_residues=max_residues, chunk_overlap=args.chunk_overlap).to(torch.float32)
                else:
                    cif_path = _resolve_cif_path(cif_index, str(apo_pdb))
                    cif_key = str(cif_path)
                    if cif_key not in cif_cache:
                        cif_cache[cif_key] = _parse_mmcif_atom_site(cif_path)
                    recs = _build_chain_records(cif_cache[cif_key], str(apo_chain))
                    seq = "".join([r["aa1"] for r in recs])
                    if len(seq) != L:
                        # strict alignment required by this baseline
                        raise ValueError(f"sequence length {len(seq)} != npz length {L}")
                    emb = _embed_sequence(tokenizer, model, seq, args.device, max_residues=max_residues, chunk_overlap=args.chunk_overlap).to(torch.float32)
                if emb.shape[0] != L:
                    raise ValueError(f"embedding length {emb.shape[0]} != {L}")

            obj = {
                "chain_uid": chain_uid,
                "embedding": emb,
                "label": y,
                "mask": mask,
                "split": split,
                "embedding_dim": emb_dim,
                "esm_model": "mock" if args.mock else args.esm_model,
            }
            if seq is not None:
                obj["sequence"] = seq
            torch.save(obj, out_dir / f"{chain_uid}.pt")
            done += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {chain_uid}: {e}")

    summary = {
        "done": done,
        "failed": failed,
        "out_dir": str(out_dir),
        "mock": bool(args.mock),
        "embedding_dim": emb_dim,
        "device": args.device,
        "gpu": args.gpu,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
