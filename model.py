#!/usr/bin/env python3
"""Reference implementation of dynamics-informed selective structural routing.

This file intentionally collects the model, data-loading helpers, training loop,
and metric utilities needed for the fixed model into one program.  It does not
import or dispatch to the older generic training programs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Required by torch deterministic CUDA matmul/einsum paths when CUDA >= 10.2.
# This must be set before CUDA work starts; setting it before importing torch is
# the safest default for reproducible runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import FIXED, TEST_PRESETS, TRAIN_PRESETS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_subset", choices=sorted(TRAIN_PRESETS), default="apo")
    ap.add_argument("--test_preset", choices=sorted(TEST_PRESETS), default="same")
    ap.add_argument("--ann_dir", type=Path, default=None)
    ap.add_argument("--emb_dir", type=Path, default=None)
    ap.add_argument("--pipeline_dir", type=Path, default=None)
    ap.add_argument("--graph_cache_dir", type=Path, default=None)
    ap.add_argument("--flex_cache_dir", type=Path, default=None)
    ap.add_argument("--test_ann_dir", type=Path, default=None)
    ap.add_argument("--test_emb_dir", type=Path, default=None)
    ap.add_argument("--test_pipeline_dir", type=Path, default=None)
    ap.add_argument("--test_graph_cache_dir", type=Path, default=None)
    ap.add_argument("--test_flex_cache_dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val_fold", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold_mode", choices=["fixed", "best_mcc", "target_tpr_fpr"], default="target_tpr_fpr")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--target_tpr", type=float, default=0.48)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    ap.add_argument("--final_train_all_folds", action="store_true", default=True)
    ap.add_argument("--no_final_train_all_folds", action="store_false", dest="final_train_all_folds")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    ap.add_argument("--collect_code_stats", action="store_true")
    ap.add_argument("--fail_if_out_dir_exists", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_dir", type=Path, required=True)
    return ap


def _resolve_paths(args) -> tuple[dict, dict]:
    train = dict(TRAIN_PRESETS[args.train_subset])
    test = dict(TEST_PRESETS[args.test_preset])
    for key in ["ann_dir", "emb_dir", "pipeline_dir", "graph_cache_dir", "flex_cache_dir"]:
        value = getattr(args, key)
        if value is not None:
            train[key] = str(value)
    for key in ["test_ann_dir", "test_emb_dir", "test_pipeline_dir", "test_graph_cache_dir", "test_flex_cache_dir"]:
        value = getattr(args, key)
        if value is not None:
            test[key] = str(value)
    return train, test


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


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

def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def _average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1, dtype=np.float64)
    fp = np.cumsum(y_sorted == 0, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)

def _confusion_from_threshold(prob: np.ndarray, y_true: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    pred = (prob >= threshold).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn

def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float):
    prob = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    y_np = y.cpu().numpy().astype(np.int64)
    tp, tn, fp, fn = _confusion_from_threshold(prob, y_np, threshold)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    auc = _roc_auc_score(y_np, prob)
    auprc = _average_precision_score(y_np, prob)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(auc),
        "auprc": float(auprc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

class VectorLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * (1.0 / math.sqrt(max(in_channels, 1))))

    def forward(self, x):
        return torch.einsum("...ic,oi->...oc", x, self.weight)


class DenseGVPBlock(nn.Module):
    def __init__(self, node_s_dim: int, node_v_dim: int, edge_s_dim: int, edge_v_dim: int, hidden_s: int, dropout: float):
        super().__init__()
        msg_s_in = 2 * node_s_dim + 2 * node_v_dim + edge_s_dim + edge_v_dim
        msg_v_in = 2 * node_v_dim + edge_v_dim
        upd_s_in = 2 * node_s_dim + 2 * node_v_dim
        upd_v_in = 2 * node_v_dim
        self.msg_s = nn.Sequential(nn.Linear(msg_s_in, hidden_s), nn.ReLU(), nn.Linear(hidden_s, node_s_dim))
        self.msg_v = VectorLinear(msg_v_in, node_v_dim)
        self.upd_s = nn.Sequential(nn.Linear(upd_s_in, hidden_s), nn.ReLU(), nn.Linear(hidden_s, node_s_dim))
        self.upd_v = VectorLinear(upd_v_in, node_v_dim)
        self.norm_s = nn.LayerNorm(node_s_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_s, node_v, edge_s, edge_v, edge_mask):
        k = node_s.shape[1]
        src_s = node_s[:, :, None, :].expand(-1, -1, k, -1)
        dst_s = node_s[:, None, :, :].expand(-1, k, -1, -1)
        src_v = node_v[:, :, None, :, :].expand(-1, -1, k, -1, -1)
        dst_v = node_v[:, None, :, :, :].expand(-1, k, -1, -1, -1)
        src_vn = torch.linalg.norm(src_v, dim=-1)
        dst_vn = torch.linalg.norm(dst_v, dim=-1)
        edge_vn = torch.linalg.norm(edge_v, dim=-1)
        pair_s = torch.cat([src_s, dst_s, edge_s, src_vn, dst_vn, edge_vn], dim=-1)
        pair_v = torch.cat([src_v, dst_v, edge_v], dim=-2)
        msg_s = self.msg_s(pair_s)
        msg_v = self.msg_v(pair_v)
        edge_f = edge_mask.float()
        msg_s = msg_s * edge_f[..., None]
        msg_v = msg_v * edge_f[..., None, None]
        deg = edge_f.sum(dim=1).clamp(min=1.0)
        agg_s = msg_s.sum(dim=1) / deg[..., None]
        agg_v = msg_v.sum(dim=1) / deg[..., None, None]
        upd_s_in = torch.cat([node_s, agg_s, torch.linalg.norm(node_v, dim=-1), torch.linalg.norm(agg_v, dim=-1)], dim=-1)
        upd_v_in = torch.cat([node_v, agg_v], dim=-2)
        ds = self.upd_s(upd_s_in)
        dv = self.upd_v(upd_v_in)
        node_s = self.norm_s(node_s + self.drop(ds))
        node_v = node_v + self.drop(dv)
        return node_s, node_v

def _collate_keep_list(batch):
    return batch

def _pos_weight_from_samples(samples: List[Dict]):
    pos, neg = 0.0, 0.0
    for s in samples:
        yy = s["label"][s["mask"]]
        pos += float((yy == 1).sum().item())
        neg += float((yy == 0).sum().item())
    return neg / max(pos, 1.0)

def parse_positive_indices(annotation_string: str) -> List[int]:
    if annotation_string is None:
        return []
    s = annotation_string.strip()
    if s == "" or s.upper() == "UNKNOWN":
        return []
    out = []
    for token in s.split():
        m = re.search(r"(\d+)$", token)
        if m is not None:
            out.append(int(m.group(1)))
    return out


def _decode_str_array(arr: np.ndarray) -> List[str]:
    out = []
    for x in arr:
        out.append(x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x))
    return out


def _chain_ids_from_auth_chain(arr: np.ndarray):
    s = _decode_str_array(np.asarray(arr))
    uniq = {}
    ids = []
    for x in s:
        if x not in uniq:
            uniq[x] = len(uniq)
        ids.append(uniq[x])
    return np.asarray(ids, dtype=np.int64)


def load_chain_maps(pipeline_dir: Path, graph_cache_dir: Path):
    chain_uid_by_key: Dict[str, str] = {}
    chain_npz_by_uid: Dict[str, str] = {}
    with open(pipeline_dir / "entities" / "chains.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = str(r["chain_uid"])
            pdb = str(r["apo_pdb_id"]).strip().lower()
            ch = str(r["apo_chain"]).strip()
            chain_uid_by_key[f"{pdb}|{ch}"] = uid
            chain_npz_by_uid[uid] = str(r["chain_npz"])

    graph_file_by_uid: Dict[str, str] = {}
    with open(graph_cache_dir / "graph_cache_index.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = str(r["chain_uid"])
            if uid not in graph_file_by_uid:
                graph_file_by_uid[uid] = str(r["graph_file"])
    return chain_uid_by_key, chain_npz_by_uid, graph_file_by_uid


def _parse_feature_list(v: str) -> List[str]:
    if v is None:
        return []
    s = v.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _set_reproducibility(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _feature_index_map(names: Sequence[str]) -> Dict[str, int]:
    return {str(x): i for i, x in enumerate(names)}


def _metric_value(metrics: Optional[Dict], key: str) -> float:
    if metrics is None:
        return -float("inf")
    v = metrics.get(key, float("nan"))
    try:
        vf = float(v)
    except Exception:
        return -float("inf")
    return vf if np.isfinite(vf) else -float("inf")


def _is_better(metrics: Optional[Dict], best_metrics: Optional[Dict], primary: str, secondary: str, eps: float = 1e-10) -> bool:
    if best_metrics is None:
        return True
    p = _metric_value(metrics, primary)
    bp = _metric_value(best_metrics, primary)
    if p > bp + eps:
        return True
    if abs(p - bp) <= eps:
        s = _metric_value(metrics, secondary)
        bs = _metric_value(best_metrics, secondary)
        return s > bs + eps
    return False


def _select_flex_columns(flex_scalar: np.ndarray, feature_names: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    if not wanted:
        return flex_scalar
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    missing = [w for w in wanted if w not in name_to_idx]
    if missing:
        raise KeyError(f"missing flex feature(s): {missing}; available={list(feature_names)}")
    idx = [name_to_idx[w] for w in wanted]
    return flex_scalar[:, idx]


def load_samples_from_translated_csv(
    csv_paths: List[Path],
    emb_dir: Path,
    pipeline_dir: Path,
    graph_cache_dir: Path,
    require_graph: bool,
    flex_cache_dir: Optional[Path],
    flex_features: Sequence[str],
    require_flex_for_b1: bool,
    use_region_condition: bool,
    region_bfactor_z_min: float,
    region_contact_density_max: float,
):
    chain_uid_by_key, chain_npz_by_uid, graph_file_by_uid = load_chain_maps(pipeline_dir, graph_cache_dir)
    samples = []
    skipped = {
        "missing_emb": 0,
        "missing_chain_uid": 0,
        "missing_chain_npz": 0,
        "missing_graph": 0,
        "missing_flex": 0,
        "shape_mismatch": 0,
        "bad_row": 0,
    }
    entities_npz_dir = pipeline_dir / "entities" / "chain_npz"
    observed_flex_dim: Optional[int] = None

    for p in csv_paths:
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 4:
                    skipped["bad_row"] += 1
                    continue
                pdb_id = parts[0].replace("\ufeff", "").strip().lower()
                chain_id = parts[1].replace("\ufeff", "").strip()
                ann = parts[3]

                emb_path = emb_dir / f"{pdb_id}{chain_id}.npy"
                if not emb_path.exists():
                    skipped["missing_emb"] += 1
                    continue
                emb = np.load(emb_path).astype(np.float32)
                if emb.ndim != 2:
                    skipped["shape_mismatch"] += 1
                    continue
                L = int(emb.shape[0])
                y = np.zeros((L,), dtype=np.float32)
                for idx in parse_positive_indices(ann):
                    if 0 <= idx < L:
                        y[idx] = 1.0

                uid = None
                if len(parts) >= 6:
                    uid = parts[5].strip()
                if not uid:
                    uid = chain_uid_by_key.get(f"{pdb_id}|{chain_id}")
                if uid is None:
                    skipped["missing_chain_uid"] += 1
                    continue

                npz_name = chain_npz_by_uid.get(uid)
                if npz_name is None:
                    skipped["missing_chain_npz"] += 1
                    continue
                npz_path = entities_npz_dir / npz_name
                if not npz_path.exists():
                    skipped["missing_chain_npz"] += 1
                    continue
                d_npz = np.load(npz_path)
                if int(d_npz["ca"].shape[0]) != L:
                    skipped["shape_mismatch"] += 1
                    continue

                sample = {
                    "chain_uid": uid,
                    "embedding": torch.tensor(emb, dtype=torch.float32),
                    "label": torch.tensor(y, dtype=torch.float32),
                    "mask": torch.ones((L,), dtype=torch.bool),
                }

                if require_graph:
                    gname = graph_file_by_uid.get(uid)
                    if gname is None:
                        skipped["missing_graph"] += 1
                        continue
                    gpath = graph_cache_dir / gname
                    if not gpath.exists():
                        skipped["missing_graph"] += 1
                        continue
                    d_graph = np.load(gpath)
                    nidx = d_graph["neighbor_idx"]
                    if int(nidx.shape[0]) != L:
                        skipped["shape_mismatch"] += 1
                        continue
                    sample.update(
                        {
                            "neighbor_idx": torch.tensor(nidx, dtype=torch.long),
                            "ca": torch.tensor(d_npz["ca"], dtype=torch.float32),
                            "n": torch.tensor(d_npz["n"], dtype=torch.float32),
                            "c": torch.tensor(d_npz["c"], dtype=torch.float32),
                            "cb": torch.tensor(d_npz["cb"], dtype=torch.float32),
                            "auth_seq_id": torch.tensor(d_npz["auth_seq_id"], dtype=torch.float32),
                            "has_backbone": torch.tensor(d_npz["has_backbone"], dtype=torch.float32),
                            "has_cb": torch.tensor(d_npz["has_cb"], dtype=torch.float32),
                            "chain_id": torch.tensor(_chain_ids_from_auth_chain(d_npz["auth_chain"]), dtype=torch.long)
                            if "auth_chain" in d_npz
                            else torch.zeros((L,), dtype=torch.long),
                        }
                    )

                    if flex_cache_dir is not None:
                        p_flex = flex_cache_dir / f"{uid}.npz"
                        if not p_flex.exists():
                            skipped["missing_flex"] += 1
                            if require_flex_for_b1:
                                continue
                        else:
                            d_flex = np.load(p_flex, allow_pickle=False)
                            flex_scalar = np.asarray(d_flex["flex_scalar"], dtype=np.float32)
                            if int(flex_scalar.shape[0]) != L:
                                skipped["shape_mismatch"] += 1
                                continue
                            f_names = [str(x) for x in np.asarray(d_flex["feature_names"]).tolist()]
                            f_map = _feature_index_map(f_names)
                            try:
                                flex_sel = _select_flex_columns(flex_scalar, f_names, flex_features)
                            except Exception:
                                skipped["shape_mismatch"] += 1
                                continue
                            if observed_flex_dim is None:
                                observed_flex_dim = int(flex_sel.shape[1])
                            if int(flex_sel.shape[1]) != int(observed_flex_dim):
                                skipped["shape_mismatch"] += 1
                                continue
                            sample["flex_scalar"] = torch.tensor(flex_sel, dtype=torch.float32)

                            # Optional auxiliary target: predict standardized bfactor.
                            if "bfactor_z" in f_map:
                                bz = flex_scalar[:, f_map["bfactor_z"]].astype(np.float32)
                                if "bfactor_missing" in f_map:
                                    miss = flex_scalar[:, f_map["bfactor_missing"]]
                                    tmask = (miss < 0.5).astype(np.float32)
                                else:
                                    tmask = np.ones((L,), dtype=np.float32)
                                sample["flex_target"] = torch.tensor(bz, dtype=torch.float32)
                                sample["flex_target_mask"] = torch.tensor(tmask, dtype=torch.float32)

                            # Region-conditional prior mask: high-flex OR low-contact residues.
                            if use_region_condition:
                                region = np.zeros((L,), dtype=np.float32)
                                has_any = False
                                if "bfactor_z" in f_map:
                                    region = np.maximum(region, (flex_scalar[:, f_map["bfactor_z"]] >= float(region_bfactor_z_min)).astype(np.float32))
                                    has_any = True
                                if "contact_density" in f_map:
                                    region = np.maximum(
                                        region,
                                        (flex_scalar[:, f_map["contact_density"]] <= float(region_contact_density_max)).astype(np.float32),
                                    )
                                    has_any = True
                                if not has_any:
                                    region[:] = 1.0
                                sample["flex_region_mask"] = torch.tensor(region, dtype=torch.float32)
                    elif require_flex_for_b1:
                        skipped["missing_flex"] += 1
                        continue

                samples.append(sample)
    return samples, skipped


def build_local_graph_features_flex(
    sample: Dict,
    dist_cutoff: float,
    rbf_centers: torch.Tensor,
    rbf_sigma: float,
    device: str,
    flex_dim: int,
):
    neighbor = sample["neighbor_idx"].to(device)
    ca = sample["ca"].to(device)
    n = sample["n"].to(device)
    c = sample["c"].to(device)
    cb = sample["cb"].to(device)
    auth_seq = sample["auth_seq_id"].to(device)
    has_backbone = sample["has_backbone"].to(device)
    has_cb = sample["has_cb"].to(device)
    chain_id = sample["chain_id"].to(device)
    n_res, k = neighbor.shape
    valid = neighbor >= 0
    idx = neighbor.clamp(min=0)

    ca_local = ca[idx]
    n_local = n[idx]
    c_local = c[idx]
    cb_local = cb[idx]
    center_ids = torch.arange(n_res, device=device).unsqueeze(1)
    center_ca = ca.unsqueeze(1)
    seq_local = auth_seq[idx]
    seq_center = auth_seq.unsqueeze(1)
    hb_local = has_backbone[idx]
    cb_flag_local = has_cb[idx]
    density_global = valid.sum(dim=1).float() / float(max(k, 1))
    density_local = density_global[idx]
    center_mask = (idx == center_ids) & valid

    base_node_s = torch.stack(
        [center_mask.float(), (seq_local - seq_center) / 50.0, hb_local.float(), cb_flag_local.float(), density_local.float()],
        dim=-1,
    )
    base_node_s = base_node_s * valid.unsqueeze(-1).float()

    if flex_dim > 0 and ("flex_scalar" in sample):
        flex = sample["flex_scalar"].to(device)
        flex_local = flex[idx] * valid.unsqueeze(-1).float()
        node_s = torch.cat([base_node_s, flex_local], dim=-1)
    elif flex_dim > 0:
        zeros = torch.zeros((n_res, k, flex_dim), dtype=base_node_s.dtype, device=device)
        node_s = torch.cat([base_node_s, zeros], dim=-1)
    else:
        node_s = base_node_s

    v_n_ca = ca_local - n_local
    v_ca_c = c_local - ca_local
    v_ca_cb = cb_local - ca_local
    v_center_node = ca_local - center_ca
    node_v = torch.stack([v_n_ca, v_ca_c, v_ca_cb, v_center_node], dim=-2) / 10.0
    node_v = torch.nan_to_num(node_v, nan=0.0, posinf=0.0, neginf=0.0)
    node_v = node_v * valid.unsqueeze(-1).unsqueeze(-1).float()

    ca_u = ca_local.unsqueeze(2)
    ca_v = ca_local.unsqueeze(1)
    vec_uv = (ca_v - ca_u) / 10.0
    dist = torch.linalg.norm(ca_v - ca_u, dim=-1)
    edge_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (dist < dist_cutoff) & (dist > 1e-6)
    rbf = torch.exp(-((dist.unsqueeze(-1) - rbf_centers.view(1, 1, 1, -1)) ** 2) / (2.0 * (rbf_sigma**2)))
    seq_sep = (seq_local.unsqueeze(2) - seq_local.unsqueeze(1)).abs() / 50.0
    chain_local = chain_id[idx]
    same_chain = (chain_local.unsqueeze(2) == chain_local.unsqueeze(1)).float()
    edge_s = torch.cat([rbf, seq_sep.unsqueeze(-1), same_chain.unsqueeze(-1)], dim=-1)
    edge_s = edge_s * edge_mask.unsqueeze(-1).float()
    edge_v = vec_uv.unsqueeze(-2) * edge_mask.unsqueeze(-1).unsqueeze(-1).float()
    edge_v = torch.nan_to_num(edge_v, nan=0.0, posinf=0.0, neginf=0.0)
    return node_s, node_v, edge_s, edge_v, edge_mask, center_mask, valid


class GVPContinuousEncoderFlex(nn.Module):
    def __init__(self, node_scalar_in_dim: int, rbf_bins=16, node_scalar_dim=32, node_vector_dim=4, layers=2, gvp_out_dim=128, dropout=0.3):
        super().__init__()
        self.node_scalar_proj = nn.Linear(node_scalar_in_dim, node_scalar_dim)
        self.node_vector_proj = VectorLinear(4, node_vector_dim)
        edge_s_dim = rbf_bins + 2
        edge_v_dim = 1
        self.blocks = nn.ModuleList(
            [
                DenseGVPBlock(
                    node_s_dim=node_scalar_dim,
                    node_v_dim=node_vector_dim,
                    edge_s_dim=edge_s_dim,
                    edge_v_dim=edge_v_dim,
                    hidden_s=max(64, node_scalar_dim * 2),
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.center_proj = nn.Sequential(
            nn.Linear(node_scalar_dim + node_vector_dim, gvp_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gvp_out_dim, gvp_out_dim),
        )

    def forward(self, node_s, node_v, edge_s, edge_v, edge_mask, center_mask, node_valid):
        node_s = self.node_scalar_proj(node_s)
        node_v = self.node_vector_proj(node_v)
        for blk in self.blocks:
            node_s, node_v = blk(node_s, node_v, edge_s, edge_v, edge_mask)
        n_graph = node_s.shape[0]
        center_idx = center_mask.float().argmax(dim=1)
        has_center = center_mask.any(dim=1)
        fallback = node_valid.float().argmax(dim=1)
        center_idx = torch.where(has_center, center_idx, fallback)
        idx = torch.arange(n_graph, device=node_s.device)
        s_center = node_s[idx, center_idx]
        v_center = node_v[idx, center_idx]
        z = self.center_proj(torch.cat([s_center, torch.linalg.norm(v_center, dim=-1)], dim=-1))
        return z

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        commitment_beta: float = 0.25,
        update_mode: str = "grad",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_reset_every: int = 0,
        dead_code_reset_threshold: int = 0,
        dead_code_reset_max: int = 8,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.commitment_beta = float(commitment_beta)
        self.update_mode = str(update_mode)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.dead_code_reset_every = int(dead_code_reset_every)
        self.dead_code_reset_threshold = int(dead_code_reset_threshold)
        self.dead_code_reset_max = int(dead_code_reset_max)
        self._step = 0
        self.codebook = nn.Embedding(self.codebook_size, self.code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_w", self.codebook.weight.detach().clone())
        self.register_buffer("inactive_counter", torch.zeros(self.codebook_size, dtype=torch.long))

        if self.update_mode == "ema":
            self.codebook.weight.requires_grad_(False)

    def _maybe_dead_code_reset(self, x: torch.Tensor):
        if self.dead_code_reset_every <= 0 or self.dead_code_reset_threshold <= 0:
            return
        if self._step % self.dead_code_reset_every != 0:
            return
        dead_mask = self.inactive_counter >= self.dead_code_reset_threshold
        if not torch.any(dead_mask):
            return
        dead_idx = torch.where(dead_mask)[0]
        if self.dead_code_reset_max > 0 and dead_idx.numel() > self.dead_code_reset_max:
            dead_idx = dead_idx[: self.dead_code_reset_max]
        if x.size(0) == 0:
            return
        pick = torch.randint(0, x.size(0), (dead_idx.numel(),), device=x.device)
        repl = x[pick].detach()
        self.codebook.weight.data[dead_idx] = repl
        self.ema_w[dead_idx] = repl
        self.ema_cluster_size[dead_idx] = 1.0
        self.inactive_counter[dead_idx] = 0

    def forward(self, x: torch.Tensor):
        # x: [N, D]
        x2 = (x**2).sum(dim=1, keepdim=True)  # [N,1]
        e = self.codebook.weight  # [K,D]
        e2 = (e**2).sum(dim=1).unsqueeze(0)  # [1,K]
        dist = x2 + e2 - 2.0 * (x @ e.t())  # [N,K]
        idx = torch.argmin(dist, dim=1)  # [N]
        q = self.codebook(idx)  # [N,D]

        # straight-through
        q_st = x + (q - x).detach()
        loss_commit_per = ((x - q.detach()) ** 2).mean(dim=1)
        loss_commit = loss_commit_per.mean()

        if self.update_mode == "ema" and self.training:
            self._step += 1
            with torch.no_grad():
                one_hot = F.one_hot(idx, num_classes=self.codebook_size).to(x.dtype)  # [N,K]
                cluster_size = one_hot.sum(dim=0)  # [K]
                dw = one_hot.t() @ x  # [K,D]

                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=(1.0 - self.ema_decay))
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=(1.0 - self.ema_decay))

                n = self.ema_cluster_size.sum()
                cluster_size_norm = (self.ema_cluster_size + self.ema_eps) / (
                    n + self.codebook_size * self.ema_eps
                ) * n
                new_w = self.ema_w / cluster_size_norm.unsqueeze(1).clamp(min=self.ema_eps)
                self.codebook.weight.data.copy_(new_w)

                used = cluster_size > 0
                self.inactive_counter[used] = 0
                self.inactive_counter[~used] += 1
                self._maybe_dead_code_reset(x)

            # EMA mode does not optimize codebook by gradient.
            vq_loss_per = self.commitment_beta * loss_commit_per
            vq_loss = vq_loss_per.mean()
        else:
            loss_codebook_per = ((q - x.detach()) ** 2).mean(dim=1)
            loss_codebook = loss_codebook_per.mean()
            vq_loss_per = loss_codebook_per + self.commitment_beta * loss_commit_per
            vq_loss = loss_codebook + self.commitment_beta * loss_commit
        return q_st, idx, vq_loss, vq_loss_per


def _usage_loss_from_indices(idx: torch.Tensor, codebook_size: int):
    if idx.numel() == 0:
        return torch.tensor(0.0, device=idx.device)
    counts = torch.bincount(idx, minlength=codebook_size).float()
    p = counts / counts.sum().clamp(min=1.0)
    entropy = -(p * (p + 1e-12).log()).sum()
    h_norm = entropy / math.log(max(codebook_size, 2))
    return 1.0 - h_norm


class ESMGVPB1FlexVQ(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        flex_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        gvp_layers: int = 2,
        gvp_out_dim: int = 128,
        node_scalar_dim: int = 32,
        node_vector_dim: int = 4,
        gvp_dropout: float = 0.3,
        dist_cutoff: float = 10.0,
        rbf_bins: int = 16,
        flex_mode: str = "guided_center",
        flex_target: str = "gate",
        flex_scale: float = 0.2,
        flex_dropout: float = 0.1,
        gate_temp_base: float = 1.0,
        use_vq: bool = True,
        vq_dim: int = 32,
        codebook_size: int = 64,
        commitment_beta: float = 0.22,
        vq_fuse_scale: float = 1.0,
        vq_bottleneck_mode: str = "hard",
        vq_struct_path_mode: str = "guidance_only",
        vq_guidance_target: str = "gate_weight",
        vq_gate_scale: float = 0.2,
        use_region_conditioned_vq: bool = False,
        loop_score_bfactor_weight: float = 1.0,
        loop_score_contact_weight: float = 1.0,
        loop_score_gap_weight: float = 0.5,
        loop_score_bias: float = 0.0,
        loop_score_temp: float = 1.0,
        loop_score_threshold: float = 0.5,
        vq_bg_residual_scale: float = 0.2,
        bfactor_feature_idx: int = -1,
        contact_feature_idx: int = -1,
        gap_feature_idx: int = -1,
        vq_update_mode: str = "ema",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_reset_every: int = 200,
        dead_code_reset_threshold: int = 400,
        dead_code_reset_max: int = 8,
    ):
        super().__init__()
        self.flex_dim = int(flex_dim)
        self.flex_mode = str(flex_mode)
        self.flex_target = str(flex_target)
        self.flex_scale = float(flex_scale)
        self.gate_temp_base = float(gate_temp_base)

        concat_flex_dim = self.flex_dim if self.flex_mode == "concat" else 0
        self.encoder = GVPContinuousEncoderFlex(
            node_scalar_in_dim=5 + concat_flex_dim,
            rbf_bins=rbf_bins,
            node_scalar_dim=node_scalar_dim,
            node_vector_dim=node_vector_dim,
            layers=gvp_layers,
            gvp_out_dim=gvp_out_dim,
            dropout=gvp_dropout,
        )

        if self.flex_mode == "guided_center" and self.flex_dim > 0:
            self.flex_proj = nn.Sequential(
                nn.Linear(self.flex_dim, gvp_out_dim),
                nn.LayerNorm(gvp_out_dim),
                nn.GELU(),
                nn.Dropout(flex_dropout),
                nn.Linear(gvp_out_dim, gvp_out_dim),
            )
            self.flex_gate = nn.Sequential(
                nn.Linear(gvp_out_dim * 2, gvp_out_dim),
                nn.ReLU(),
                nn.Linear(gvp_out_dim, gvp_out_dim),
            )
            self.flex_norm = nn.LayerNorm(gvp_out_dim)
        else:
            self.flex_proj = None
            self.flex_gate = None
            self.flex_norm = None

        if self.flex_mode == "guided_center" and self.flex_dim > 0 and self.flex_target == "gate":
            self.flex_gate_temp = nn.Sequential(
                nn.Linear(self.flex_dim, esm_dim),
                nn.ReLU(),
                nn.Dropout(flex_dropout),
                nn.Linear(esm_dim, esm_dim),
            )
            self.flex_gate_temp_norm = nn.LayerNorm(esm_dim)
        else:
            self.flex_gate_temp = None
            self.flex_gate_temp_norm = None

        self.use_vq = bool(use_vq)
        self.vq_fuse_scale = float(vq_fuse_scale)
        self.vq_bottleneck_mode = str(vq_bottleneck_mode)
        self.vq_struct_path_mode = str(vq_struct_path_mode)
        self.vq_guidance_target = str(vq_guidance_target)
        self.vq_gate_scale = float(vq_gate_scale)
        self.use_region_conditioned_vq = bool(use_region_conditioned_vq)
        self.loop_score_bfactor_weight = float(loop_score_bfactor_weight)
        self.loop_score_contact_weight = float(loop_score_contact_weight)
        self.loop_score_gap_weight = float(loop_score_gap_weight)
        self.loop_score_bias = float(loop_score_bias)
        self.loop_score_temp = float(loop_score_temp)
        self.loop_score_threshold = float(loop_score_threshold)
        self.vq_bg_residual_scale = float(vq_bg_residual_scale)
        self.bfactor_feature_idx = int(bfactor_feature_idx)
        self.contact_feature_idx = int(contact_feature_idx)
        self.gap_feature_idx = int(gap_feature_idx)
        self.vq_dim = int(vq_dim)
        self.codebook_size = int(codebook_size)
        if self.use_vq:
            self.v_proj = nn.Linear(gvp_out_dim, self.vq_dim)
            self.vq = VectorQuantizer(
                self.codebook_size,
                self.vq_dim,
                commitment_beta=commitment_beta,
                update_mode=vq_update_mode,
                ema_decay=ema_decay,
                ema_eps=ema_eps,
                dead_code_reset_every=dead_code_reset_every,
                dead_code_reset_threshold=dead_code_reset_threshold,
                dead_code_reset_max=dead_code_reset_max,
            )
            self.vq_to_struct = nn.Linear(self.vq_dim, gvp_out_dim)
            self.vq_norm = nn.LayerNorm(gvp_out_dim)
            self.vq_gate_head = nn.Linear(self.vq_dim, esm_dim)
            self.vq_prior_head = nn.Linear(self.vq_dim, 1)
            nn.init.zeros_(self.vq_gate_head.weight)
            nn.init.zeros_(self.vq_gate_head.bias)
            nn.init.zeros_(self.vq_prior_head.weight)
            nn.init.zeros_(self.vq_prior_head.bias)
        else:
            self.v_proj = None
            self.vq = None
            self.vq_to_struct = None
            self.vq_norm = None
            self.vq_gate_head = None
            self.vq_prior_head = None

        self.struct_to_h = nn.Linear(gvp_out_dim, esm_dim)
        self.gate = nn.Sequential(nn.Linear(esm_dim * 2, esm_dim), nn.ReLU(), nn.Linear(esm_dim, esm_dim))
        self.fuse_norm = nn.LayerNorm(esm_dim)
        self.head = nn.Sequential(nn.Linear(esm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.flex_head = nn.Linear(gvp_out_dim, 1) if self.flex_dim > 0 else None

        self.dist_cutoff = dist_cutoff
        self.rbf_bins = rbf_bins
        self.register_buffer("rbf_centers", torch.linspace(0.0, dist_cutoff, rbf_bins))
        self.rbf_sigma = dist_cutoff / max(rbf_bins, 1)

    def _forward_impl(self, sample: Dict, device: str):
        h = sample["embedding"].to(device)
        node_s, node_v, edge_s, edge_v, edge_mask, center_mask, node_valid = build_local_graph_features_flex(
            sample, self.dist_cutoff, self.rbf_centers, self.rbf_sigma, device, self.flex_dim if self.flex_mode == "concat" else 0
        )
        z_struct = self.encoder(node_s, node_v, edge_s, edge_v, edge_mask, center_mask, node_valid)

        flex = sample["flex_scalar"].to(device) if ("flex_scalar" in sample and self.flex_dim > 0) else None
        region = sample["flex_region_mask"].to(device).unsqueeze(-1) if "flex_region_mask" in sample else None

        aux = {}
        loop_score = None
        loop_mask = None
        if self.flex_dim > 0 and (flex is not None):
            score = torch.zeros(flex.size(0), device=flex.device)
            if self.bfactor_feature_idx >= 0 and self.bfactor_feature_idx < flex.size(1):
                score = score + self.loop_score_bfactor_weight * flex[:, self.bfactor_feature_idx]
            if self.contact_feature_idx >= 0 and self.contact_feature_idx < flex.size(1):
                score = score - self.loop_score_contact_weight * flex[:, self.contact_feature_idx]
            if self.gap_feature_idx >= 0 and self.gap_feature_idx < flex.size(1):
                score = score + self.loop_score_gap_weight * flex[:, self.gap_feature_idx]
            score = (score + self.loop_score_bias) / max(self.loop_score_temp, 1e-6)
            loop_score = torch.sigmoid(score)
            loop_mask = (loop_score >= self.loop_score_threshold)
            aux["vq_loop_score"] = loop_score
            aux["vq_loop_mask"] = loop_mask

        if self.use_vq:
            # 1) Pure geometric structure goes through VQ for guidance/statistics.
            v = self.v_proj(z_struct)
            q_st, q_idx, vq_loss, vq_loss_per = self.vq(v)
            r = self.vq_to_struct(q_st)
            aux["vq_pos_prior"] = torch.sigmoid(self.vq_prior_head(q_st)).squeeze(-1)
            aux["vq_gate_bias"] = self.vq_gate_head(q_st)
            aux["vq_loss_per"] = vq_loss_per

            if self.use_region_conditioned_vq and (loop_mask is not None):
                if self.vq_bottleneck_mode == "hard":
                    z_loop = self.vq_norm(r)
                else:
                    z_loop = self.vq_norm(z_struct + self.vq_fuse_scale * r)
                z_bg = self.vq_norm(z_struct + self.vq_bg_residual_scale * r)
                z_quant = torch.where(loop_mask.unsqueeze(-1), z_loop, z_bg)
            else:
                if self.vq_struct_path_mode == "guidance_only":
                    # Keep continuous structure branch intact; VQ acts as a guidance signal.
                    z_quant = z_struct
                else:
                    if self.vq_bottleneck_mode == "hard":
                        # Hard bottleneck: cut residual bypass from continuous z_struct.
                        z_quant = self.vq_norm(r)
                    else:
                        # Residual mode kept for ablations.
                        z_quant = self.vq_norm(z_struct + self.vq_fuse_scale * r)
            usage_loss = _usage_loss_from_indices(q_idx, self.codebook_size)
            aux["vq_loss"] = vq_loss
            aux["usage_loss"] = usage_loss
            aux["code_idx"] = q_idx
        else:
            z_quant = z_struct
            aux["vq_loss"] = torch.tensor(0.0, device=z_struct.device)
            aux["usage_loss"] = torch.tensor(0.0, device=z_struct.device)

        # 2) Inject dynamics prior after structural discretization (de-entangled flow).
        if self.flex_mode == "guided_center" and self.flex_dim > 0 and (flex is not None) and self.flex_target == "z":
            guide = self.flex_proj(flex * self.flex_scale)
            if region is not None:
                guide = guide * region
            g = torch.sigmoid(self.flex_gate(torch.cat([z_quant, guide], dim=-1)))
            z_final = self.flex_norm(z_quant + g * guide)
        else:
            z_final = z_quant

        s = self.struct_to_h(z_final)
        gate_logits = self.gate(torch.cat([h, s], dim=-1))
        if self.use_vq and ("gate" in self.vq_guidance_target):
            gate_bias = torch.tanh(aux["vq_gate_bias"]) * self.vq_gate_scale
            if loop_score is not None:
                gate_bias = gate_bias * loop_score.unsqueeze(-1)
            gate_logits = gate_logits + gate_bias
        if self.flex_mode == "guided_center" and self.flex_dim > 0 and (flex is not None) and self.flex_target == "gate":
            t = self.flex_gate_temp(flex * self.flex_scale)
            if region is not None:
                t = t * region
            t = self.flex_gate_temp_norm(t)
            temp = torch.clamp(self.gate_temp_base + t, min=0.25, max=4.0)
            gate = torch.sigmoid(gate_logits / temp)
        else:
            gate = torch.sigmoid(gate_logits)
        h_fused = self.fuse_norm(h + gate * s)
        logits = self.head(h_fused).squeeze(-1)
        if self.flex_head is not None:
            aux["flex_pred"] = self.flex_head(z_final).squeeze(-1)
        return logits, aux

    def forward(self, sample: Dict, device: str):
        logits, _ = self._forward_impl(sample, device)
        return logits

    def forward_with_aux(self, sample: Dict, device: str):
        return self._forward_impl(sample, device)


def _forward_logits_aux(model, sample: Dict, device: str):
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(sample, device=device)
    return model(sample, device=device), {}


def _build_instance_weights(
    sample: Dict,
    y_m: torch.Tensor,
    mask_m: torch.Tensor,
    flex_feature_idx: Dict[str, int],
    use_flex_weighting: bool,
    use_region_condition: bool,
    flex_weight_center: float,
    flex_weight_temp: float,
    flex_pos_alpha: float,
    flex_neg_beta: float,
    flex_neg_min: float,
    flex_weight_min: float,
    flex_weight_max: float,
    vq_pos_prior_m: Optional[torch.Tensor] = None,
    use_vq_guidance_weighting: bool = False,
    vq_weight_pos_alpha: float = 0.0,
    vq_weight_neg_beta: float = 0.0,
    vq_weight_min: float = 0.8,
    vq_weight_max: float = 1.2,
    detach_vq_prior_weight: bool = True,
):
    w = torch.ones_like(y_m)
    if use_flex_weighting and ("flex_scalar" in sample) and ("bfactor_z" in flex_feature_idx):
        mask_cpu = mask_m.detach().cpu()
        f = sample["flex_scalar"][mask_cpu].to(y_m.device)
        z = f[:, flex_feature_idx["bfactor_z"]]
        score = torch.sigmoid((z - flex_weight_center) / max(flex_weight_temp, 1e-6))
        if use_region_condition and ("flex_region_mask" in sample):
            region = sample["flex_region_mask"][mask_cpu].to(y_m.device)
            score = score * region
        pos_mul = 1.0 + flex_pos_alpha * score
        neg_mul = 1.0 - flex_neg_beta * score
        neg_mul = torch.clamp(neg_mul, min=flex_neg_min)
        w = w * torch.where(y_m > 0.5, pos_mul, neg_mul)

    if use_vq_guidance_weighting and (vq_pos_prior_m is not None):
        p = torch.clamp(vq_pos_prior_m, min=0.0, max=1.0)
        if detach_vq_prior_weight:
            p = p.detach()
        pos_mul = 1.0 + vq_weight_pos_alpha * p
        neg_mul = 1.0 - vq_weight_neg_beta * p
        w = w * torch.where(y_m > 0.5, pos_mul, neg_mul)

    w = torch.clamp(
        w,
        min=min(float(flex_weight_min), float(vq_weight_min)),
        max=max(float(flex_weight_max), float(vq_weight_max)),
    )
    return w


def _iter_epoch_vq(
    samples: List[Dict],
    model,
    device: str,
    threshold: float,
    pos_weight_tensor: Optional[torch.Tensor],
    flex_feature_idx: Dict[str, int],
    use_flex_weighting: bool,
    use_region_condition: bool,
    flex_weight_center: float,
    flex_weight_temp: float,
    flex_pos_alpha: float,
    flex_neg_beta: float,
    flex_neg_min: float,
    flex_weight_min: float,
    flex_weight_max: float,
    use_vq_guidance_weighting: bool,
    vq_weight_pos_alpha: float,
    vq_weight_neg_beta: float,
    vq_weight_min: float,
    vq_weight_max: float,
    detach_vq_prior_weight: bool,
    lambda_vq_prior: float,
    vq_loss_loop_weight: float,
    vq_loss_bg_weight: float,
    usage_on_loop_only: bool,
    lambda_flex_aux: float,
    aux_target: str,
    lambda_vq: float,
    lambda_usage: float,
    optimizer=None,
    chain_batch_size: int = 1,
    grad_clip_norm: float = 0.0,
):
    training = optimizer is not None
    model.train(training)
    dl = DataLoader(samples, batch_size=chain_batch_size, shuffle=training, collate_fn=_collate_keep_list, drop_last=False)
    all_logits, all_labels = [], []
    total_loss_num, total_loss_den = 0.0, 0
    total_aux_num, total_aux_den = 0.0, 0
    total_vq_num, total_vq_den = 0.0, 0
    total_usage_num, total_usage_den = 0.0, 0
    total_prior_num, total_prior_den = 0.0, 0
    code_counts = None
    codebook_size = int(getattr(model, "codebook_size", 0))
    if codebook_size > 0:
        code_counts = torch.zeros(codebook_size, dtype=torch.long)
    for batch in dl:
        if training:
            optimizer.zero_grad()
        batch_loss_num = 0.0
        batch_loss_den = 0
        batch_aux_num, batch_aux_den = 0.0, 0
        batch_vq_num, batch_vq_den = 0.0, 0
        batch_usage_num, batch_usage_den = 0.0, 0
        for sample in batch:
            logits, aux = _forward_logits_aux(model, sample, device=device)
            y = sample["label"].to(device)
            m = sample["mask"].to(device)
            logits_m = logits[m]
            y_m = y[m]
            if logits_m.numel() == 0:
                continue
            bce = F.binary_cross_entropy_with_logits(logits_m, y_m, reduction="none", pos_weight=pos_weight_tensor)
            vq_prior_m = None
            if "vq_pos_prior" in aux:
                vq_prior_m = aux["vq_pos_prior"][m].to(y_m.device)
            w = _build_instance_weights(
                sample, y_m, m, flex_feature_idx, use_flex_weighting, use_region_condition, flex_weight_center, flex_weight_temp,
                flex_pos_alpha, flex_neg_beta, flex_neg_min, flex_weight_min, flex_weight_max,
                vq_pos_prior_m=vq_prior_m,
                use_vq_guidance_weighting=use_vq_guidance_weighting,
                vq_weight_pos_alpha=vq_weight_pos_alpha,
                vq_weight_neg_beta=vq_weight_neg_beta,
                vq_weight_min=vq_weight_min,
                vq_weight_max=vq_weight_max,
                detach_vq_prior_weight=detach_vq_prior_weight,
            )
            pred_loss = (bce * w).mean()
            loss = pred_loss

            prior_loss = None
            if lambda_vq_prior > 0.0 and (vq_prior_m is not None):
                prior_loss = F.binary_cross_entropy(vq_prior_m, y_m.float(), reduction="mean")
                loss = loss + lambda_vq_prior * prior_loss

            aux_loss = None
            if lambda_flex_aux > 0.0 and ("flex_pred" in aux):
                flex_pred = aux["flex_pred"]
                if aux_target == "bfactor_z" and ("flex_target" in sample):
                    t = sample["flex_target"].to(device)
                    if "flex_target_mask" in sample:
                        tmask = sample["flex_target_mask"].to(device) > 0.5
                    else:
                        tmask = torch.ones_like(t, dtype=torch.bool)
                    if use_region_condition and ("flex_region_mask" in sample):
                        tmask = tmask & (sample["flex_region_mask"].to(device) > 0.5)
                    if tmask.any():
                        aux_loss = F.mse_loss(flex_pred[tmask], t[tmask], reduction="mean")
                        loss = loss + lambda_flex_aux * aux_loss

            vq_loss = aux.get("vq_loss", torch.tensor(0.0, device=device))
            if "vq_loss_per" in aux:
                vq_loss_per_m = aux["vq_loss_per"][m].to(device)
                if usage_on_loop_only and ("vq_loop_mask" in aux):
                    lm = aux["vq_loop_mask"][m].to(device)
                    if lm.any():
                        l_loop = vq_loss_per_m[lm].mean()
                    else:
                        l_loop = torch.tensor(0.0, device=device)
                    if (~lm).any():
                        l_bg = vq_loss_per_m[~lm].mean()
                    else:
                        l_bg = torch.tensor(0.0, device=device)
                    denom = max(vq_loss_loop_weight + vq_loss_bg_weight, 1e-6)
                    vq_loss = (vq_loss_loop_weight * l_loop + vq_loss_bg_weight * l_bg) / denom
                else:
                    vq_loss = vq_loss_per_m.mean()

            usage_loss = aux.get("usage_loss", torch.tensor(0.0, device=device))
            if "code_idx" in aux:
                idx_use = aux["code_idx"][m]
                if usage_on_loop_only and ("vq_loop_mask" in aux):
                    lm = aux["vq_loop_mask"][m]
                    idx_use = idx_use[lm]
                usage_loss = _usage_loss_from_indices(idx_use.to(device), codebook_size)
            loss = loss + lambda_vq * vq_loss + lambda_usage * usage_loss
            if code_counts is not None and ("code_idx" in aux):
                idx_cnt = aux["code_idx"][m]
                if usage_on_loop_only and ("vq_loop_mask" in aux):
                    idx_cnt = idx_cnt[aux["vq_loop_mask"][m]]
                idx_cpu = idx_cnt.detach().cpu()
                code_counts += torch.bincount(idx_cpu, minlength=codebook_size)

            n = int(y_m.numel())
            batch_loss_num = batch_loss_num + loss * n
            batch_loss_den += n
            if aux_loss is not None:
                batch_aux_num = batch_aux_num + aux_loss.detach() * n
                batch_aux_den += n
            if prior_loss is not None:
                total_prior_num += float(prior_loss.detach().item()) * n
                total_prior_den += n
            batch_vq_num = batch_vq_num + vq_loss.detach() * n
            batch_vq_den += n
            batch_usage_num = batch_usage_num + usage_loss.detach() * n
            batch_usage_den += n

            all_logits.append(logits_m.detach().cpu())
            all_labels.append(y_m.detach().cpu())

        if batch_loss_den > 0:
            batch_loss = batch_loss_num / batch_loss_den
            if training:
                batch_loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            total_loss_num += float(batch_loss.item()) * batch_loss_den
            total_loss_den += batch_loss_den
            if batch_aux_den > 0:
                total_aux_num += float((batch_aux_num / batch_aux_den).item()) * batch_aux_den
                total_aux_den += batch_aux_den
            if batch_vq_den > 0:
                total_vq_num += float((batch_vq_num / batch_vq_den).item()) * batch_vq_den
                total_vq_den += batch_vq_den
            if batch_usage_den > 0:
                total_usage_num += float((batch_usage_num / batch_usage_den).item()) * batch_usage_den
                total_usage_den += batch_usage_den

    if not all_logits:
        return {
            "loss": float("nan"),
            "aux_loss": float("nan"),
            "vq_loss": float("nan"),
            "usage_loss": float("nan"),
            "prior_loss": float("nan"),
            "active_code_ratio": float("nan"),
            "metrics": None,
        }
    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    metrics = metrics_from_logits(logits_all, labels_all, threshold=threshold)
    avg_loss = total_loss_num / max(total_loss_den, 1)
    avg_aux = total_aux_num / max(total_aux_den, 1) if total_aux_den > 0 else 0.0
    avg_vq = total_vq_num / max(total_vq_den, 1) if total_vq_den > 0 else 0.0
    avg_usage = total_usage_num / max(total_usage_den, 1) if total_usage_den > 0 else 0.0
    avg_prior = total_prior_num / max(total_prior_den, 1) if total_prior_den > 0 else 0.0
    if code_counts is not None and int(code_counts.sum().item()) > 0:
        active_ratio = float((code_counts > 0).float().mean().item())
    else:
        active_ratio = float("nan")
    return {
        "loss": avg_loss,
        "aux_loss": float(avg_aux),
        "vq_loss": float(avg_vq),
        "usage_loss": float(avg_usage),
        "prior_loss": float(avg_prior),
        "active_code_ratio": active_ratio,
        "metrics": metrics,
    }


def infer_prob_y(samples, model, device, batch_size):
    model.eval()
    all_prob, all_y = [], []
    dl = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=_collate_keep_list)
    with torch.no_grad():
        for batch in dl:
            for s in batch:
                lg = model(s, device=device)
                m = s["mask"].to(device)
                y = s["label"].to(device)
                lg = lg[m]
                yy = y[m]
                if lg.numel() == 0:
                    continue
                all_prob.append(torch.sigmoid(lg).cpu().numpy())
                all_y.append(yy.cpu().numpy().astype(np.int64))
    if not all_prob:
        return np.array([]), np.array([])
    return np.concatenate(all_prob), np.concatenate(all_y)


def collect_code_enrichment(
    samples: List[Dict],
    model,
    device: str,
    batch_size: int,
    codebook_size: int,
    bfactor_feature_idx: Optional[int],
):
    if codebook_size <= 0:
        return None
    code_pos = np.zeros((codebook_size,), dtype=np.int64)
    code_neg = np.zeros((codebook_size,), dtype=np.int64)
    b_sum = np.zeros((codebook_size,), dtype=np.float64)
    b_cnt = np.zeros((codebook_size,), dtype=np.int64)

    model.eval()
    dl = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=_collate_keep_list)
    with torch.no_grad():
        for batch in dl:
            for s in batch:
                _logits, aux = _forward_logits_aux(model, s, device=device)
                if "code_idx" not in aux:
                    continue
                idx = aux["code_idx"].detach().cpu().numpy().astype(np.int64)
                m = s["mask"].numpy().astype(bool)
                y = s["label"].numpy().astype(np.int64)
                if len(idx) != len(y):
                    continue
                idx_m = idx[m]
                y_m = y[m]
                if idx_m.size == 0:
                    continue
                pos_idx = idx_m[y_m == 1]
                neg_idx = idx_m[y_m == 0]
                if pos_idx.size > 0:
                    code_pos += np.bincount(pos_idx, minlength=codebook_size)
                if neg_idx.size > 0:
                    code_neg += np.bincount(neg_idx, minlength=codebook_size)

                if bfactor_feature_idx is not None and ("flex_scalar" in s):
                    f = s["flex_scalar"].numpy()
                    if f.shape[1] > bfactor_feature_idx:
                        bz = f[:, bfactor_feature_idx][m]
                        np.add.at(b_sum, idx_m, bz.astype(np.float64))
                        np.add.at(b_cnt, idx_m, 1)

    total = code_pos + code_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_rate = np.where(total > 0, code_pos / np.maximum(total, 1), 0.0)
        mean_b = np.where(b_cnt > 0, b_sum / np.maximum(b_cnt, 1), np.nan)
    active = np.where(total > 0)[0]
    top_pos = sorted(active.tolist(), key=lambda k: (pos_rate[k], code_pos[k]), reverse=True)[:20]
    top_b = sorted(active.tolist(), key=lambda k: (np.nan_to_num(mean_b[k], nan=-1e9), total[k]), reverse=True)[:20]
    return {
        "code_pos": code_pos.tolist(),
        "code_neg": code_neg.tolist(),
        "code_total": total.tolist(),
        "code_pos_rate": pos_rate.tolist(),
        "code_mean_bfactor_z": [None if np.isnan(x) else float(x) for x in mean_b.tolist()],
        "top_pos_rate_codes": top_pos,
        "top_high_bfactor_codes": top_b,
    }


def pick_threshold(mode: str, fixed_th: float, val_prob: np.ndarray, val_y: np.ndarray, target_tpr: float, target_fpr: float):
    if mode == "fixed":
        return float(fixed_th)
    grid = np.linspace(0.01, 0.99, 99)
    if mode == "best_mcc":
        best = (-1e18, fixed_th)
        for t in grid:
            tp, tn, fp, fn = _confusion_from_threshold(val_prob, val_y, float(t))
            denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
            mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
            if mcc > best[0]:
                best = (mcc, float(t))
        return float(best[1])
    best = (1e18, fixed_th)
    for t in grid:
        tp, tn, fp, fn = _confusion_from_threshold(val_prob, val_y, float(t))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        score = abs(tpr - target_tpr) + abs(fpr - target_fpr)
        if score < best[0]:
            best = (score, float(t))
    return float(best[1])

def train_loop(
    train_samples,
    val_samples,
    model,
    optimizer,
    device,
    epochs,
    threshold,
    batch_size,
    grad_clip,
    pos_weight_tensor,
    flex_feature_idx,
    use_flex_weighting,
    use_region_condition,
    flex_weight_center,
    flex_weight_temp,
    flex_pos_alpha,
    flex_neg_beta,
    flex_neg_min,
    flex_weight_min,
    flex_weight_max,
    use_vq_guidance_weighting,
    vq_weight_pos_alpha,
    vq_weight_neg_beta,
    vq_weight_min,
    vq_weight_max,
    detach_vq_prior_weight,
    lambda_vq_prior,
    vq_loss_loop_weight,
    vq_loss_bg_weight,
    usage_on_loop_only,
    lambda_flex_aux,
    aux_target,
    lambda_vq,
    lambda_usage,
    vq_warmup_epochs,
    vq_ramp_epochs,
    hard_bottleneck_from_epoch,
    select_primary,
    select_secondary,
):
    hist = []
    best_metrics = None
    best_state = None
    for ep in range(1, epochs + 1):
        # VQ scheduling: warmup -> ramp -> full
        if ep <= vq_warmup_epochs:
            lam_vq_eff = 0.0
            lam_usage_eff = 0.0
        else:
            if vq_ramp_epochs > 0:
                t = min(max(ep - vq_warmup_epochs, 0), vq_ramp_epochs) / float(vq_ramp_epochs)
            else:
                t = 1.0
            lam_vq_eff = float(lambda_vq) * float(t)
            lam_usage_eff = float(lambda_usage) * float(t)

        if hasattr(model, "vq_bottleneck_mode"):
            if hard_bottleneck_from_epoch > 0 and ep >= hard_bottleneck_from_epoch:
                model.vq_bottleneck_mode = "hard"
            else:
                model.vq_bottleneck_mode = "residual"

        tr = _iter_epoch_vq(
            train_samples, model, device, threshold, pos_weight_tensor, flex_feature_idx,
            use_flex_weighting, use_region_condition, flex_weight_center, flex_weight_temp,
            flex_pos_alpha, flex_neg_beta, flex_neg_min, flex_weight_min, flex_weight_max,
            use_vq_guidance_weighting, vq_weight_pos_alpha, vq_weight_neg_beta, vq_weight_min, vq_weight_max,
            detach_vq_prior_weight, lambda_vq_prior,
            vq_loss_loop_weight, vq_loss_bg_weight, usage_on_loop_only,
            lambda_flex_aux, aux_target, lam_vq_eff, lam_usage_eff,
            optimizer=optimizer, chain_batch_size=batch_size, grad_clip_norm=grad_clip
        )
        va = _iter_epoch_vq(
            val_samples, model, device, threshold, pos_weight_tensor, flex_feature_idx,
            use_flex_weighting, use_region_condition, flex_weight_center, flex_weight_temp,
            flex_pos_alpha, flex_neg_beta, flex_neg_min, flex_weight_min, flex_weight_max,
            use_vq_guidance_weighting, vq_weight_pos_alpha, vq_weight_neg_beta, vq_weight_min, vq_weight_max,
            detach_vq_prior_weight, lambda_vq_prior,
            vq_loss_loop_weight, vq_loss_bg_weight, usage_on_loop_only,
            lambda_flex_aux, aux_target, lam_vq_eff, lam_usage_eff,
            optimizer=None, chain_batch_size=batch_size, grad_clip_norm=0.0
        )
        rec = {
            "epoch": ep,
            "lambda_vq_eff": lam_vq_eff,
            "lambda_usage_eff": lam_usage_eff,
            "vq_bottleneck_mode_eff": getattr(model, "vq_bottleneck_mode", "na"),
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_aux_loss": tr.get("aux_loss", 0.0),
            "val_aux_loss": va.get("aux_loss", 0.0),
            "train_vq_loss": tr.get("vq_loss", 0.0),
            "val_vq_loss": va.get("vq_loss", 0.0),
            "train_usage_loss": tr.get("usage_loss", 0.0),
            "val_usage_loss": va.get("usage_loss", 0.0),
            "train_prior_loss": tr.get("prior_loss", 0.0),
            "val_prior_loss": va.get("prior_loss", 0.0),
            "train_active_code_ratio": tr.get("active_code_ratio", float("nan")),
            "val_active_code_ratio": va.get("active_code_ratio", float("nan")),
            **{f"train_{k}": v for k, v in tr["metrics"].items()},
            **{f"val_{k}": v for k, v in va["metrics"].items()},
        }
        hist.append(rec)
        val_metrics = va["metrics"]
        if _is_better(val_metrics, best_metrics, primary=select_primary, secondary=select_secondary):
            best_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(rec))
    model.load_state_dict(best_state)
    return hist


def _build_fixed_model(esm_dim: int, flex_dim: int, device: str, flex_feature_idx: dict[str, int]):
    model = ESMGVPB1FlexVQ(
        esm_dim=esm_dim,
        flex_dim=flex_dim,
        hidden_dim=FIXED["hidden_dim"],
        dropout=FIXED["dropout"],
        gvp_layers=FIXED["gvp_layers"],
        gvp_out_dim=FIXED["gvp_out_dim"],
        node_scalar_dim=FIXED["node_scalar_dim"],
        node_vector_dim=FIXED["node_vector_dim"],
        gvp_dropout=FIXED["gvp_dropout"],
        dist_cutoff=FIXED["graph_cutoff"],
        rbf_bins=FIXED["rbf_bins"],
        flex_mode=FIXED["flex_mode"],
        flex_target=FIXED["flex_target"],
        flex_scale=FIXED["flex_scale"],
        flex_dropout=FIXED["flex_dropout"],
        gate_temp_base=FIXED["gate_temp_base"],
        use_vq=FIXED["use_vq"],
        vq_dim=FIXED["vq_dim"],
        codebook_size=FIXED["codebook_size"],
        commitment_beta=FIXED["commitment_beta"],
        vq_fuse_scale=FIXED["vq_fuse_scale"],
        vq_bottleneck_mode=FIXED["vq_bottleneck_mode"],
        vq_struct_path_mode=FIXED["vq_struct_path_mode"],
        vq_guidance_target=FIXED["vq_guidance_target"],
        vq_gate_scale=FIXED["vq_gate_scale"],
        use_region_conditioned_vq=FIXED["use_region_conditioned_vq"],
        loop_score_bfactor_weight=FIXED["loop_score_bfactor_weight"],
        loop_score_contact_weight=FIXED["loop_score_contact_weight"],
        loop_score_gap_weight=FIXED["loop_score_gap_weight"],
        loop_score_bias=FIXED["loop_score_bias"],
        loop_score_temp=FIXED["loop_score_temp"],
        loop_score_threshold=FIXED["loop_score_threshold"],
        vq_bg_residual_scale=FIXED["vq_bg_residual_scale"],
        bfactor_feature_idx=flex_feature_idx.get("bfactor_z", -1),
        contact_feature_idx=flex_feature_idx.get("contact_density", -1),
        gap_feature_idx=flex_feature_idx.get("gap_proximity", -1),
        vq_update_mode=FIXED["vq_update_mode"],
        ema_decay=FIXED["ema_decay"],
        ema_eps=FIXED["ema_eps"],
        dead_code_reset_every=FIXED["dead_code_reset_every"],
        dead_code_reset_threshold=FIXED["dead_code_reset_threshold"],
        dead_code_reset_max=FIXED["dead_code_reset_max"],
    ).to(device)

    gvp_params = [p for p in model.encoder.parameters() if p.requires_grad]
    vq_params = []
    vq_params += [p for p in model.v_proj.parameters() if p.requires_grad]
    vq_params += [p for p in model.vq.parameters() if p.requires_grad]
    vq_params += [p for p in model.vq_to_struct.parameters() if p.requires_grad]
    vq_params += [p for p in model.vq_norm.parameters() if p.requires_grad]
    gvp_ids = {id(p) for p in gvp_params}
    vq_ids = {id(p) for p in vq_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in gvp_ids and id(p) not in vq_ids]

    groups = [
        {"params": gvp_params, "lr": FIXED["lr_gvp"]},
        {"params": vq_params, "lr": FIXED["lr_vq"]},
        {"params": head_params, "lr": FIXED["lr_head"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=FIXED["weight_decay"])
    return model, optimizer


def _load_samples(csv_paths, paths, flex_features):
    return load_samples_from_translated_csv(
        csv_paths,
        Path(paths["emb_dir"]),
        Path(paths["pipeline_dir"]),
        Path(paths["graph_cache_dir"]),
        True,
        Path(paths["flex_cache_dir"]),
        flex_features,
        True,
        FIXED["use_region_condition"],
        FIXED["region_bfactor_z_min"],
        FIXED["region_contact_density_max"],
    )


def _write_empty_train_folds(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (out_dir / f"train-fold-{i}.final.csv").write_text("", encoding="utf-8")


def _maybe_build_indexed_holo_annotations(test_paths: dict, out_dir: Path) -> bool:
    """Build test.final.csv from a holo pipeline when a preset annotation is absent."""
    ann_dir = Path(test_paths["ann_dir"])
    test_csv = ann_dir / "test.final.csv"
    if test_csv.exists():
        return False

    pipeline_dir = Path(test_paths["pipeline_dir"])
    entities_dir = pipeline_dir / "entities"
    regions_csv = entities_dir / "metadata_regions.csv"
    chains_csv = entities_dir / "chains.csv"
    chain_npz_dir = entities_dir / "chain_npz"
    if not (regions_csv.exists() and chains_csv.exists() and chain_npz_dir.exists()):
        msg = (
            f"Missing CSV: {test_csv}\n\n"
            "This preset needs indexed holo annotations. Build the holo pipeline and indexed annotations first:\n\n"
            "python -m cryptobench.build_holo_model_inputs prepare-dataset \\\n"
            "  --source_json cryptobench/cryptobench-dataset/evaluation-subsets/cb-p2rank-apo.json \\\n"
            "  --only_main \\\n"
            "  --deduplicate \\\n"
            "  --out_dir cryptobench/holo_model_inputs/cb-p2rank-holo-main\n\n"
            "python -m cryptobench.build_apo_pipeline \\\n"
            "  --stage all \\\n"
            "  --dataset_json cryptobench/holo_model_inputs/cb-p2rank-holo-main/dataset.json \\\n"
            "  --folds_dir cryptobench/holo_model_inputs/cb-p2rank-holo-main/folds \\\n"
            "  --cif_dir cryptobench/cryptobench-dataset/auxiliary-data/cif-files \\\n"
            "  --out_dir cryptobench/pipeline_holo_cb_p2rank_main \\\n"
            "  --dist_cutoff 10.0 \\\n"
            "  --max_neighbors 40\n\n"
            "python -m cryptobench.build_holo_model_inputs build-annotations \\\n"
            "  --pipeline_dir cryptobench/pipeline_holo_cb_p2rank_main \\\n"
            "  --out_dir cryptobench/evaluation-annotations-aligned/cb-p2rank-holo-main-indexed\n"
        )
        raise FileNotFoundError(msg)

    chain_npz_by_uid = {}
    with chains_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            chain_npz_by_uid[str(row["chain_uid"])] = str(row["chain_npz"])

    labels_by_uid = {}
    for uid, npz_name in chain_npz_by_uid.items():
        d = np.load(chain_npz_dir / npz_name)
        region_ids = [int(x) for x in np.asarray(d["region_ids"]).tolist()]
        y_regions = np.asarray(d["y_regions"])
        labels_by_uid[uid] = {rid: y_regions[i].astype(bool) for i, rid in enumerate(region_ids)}

    ann_dir.mkdir(parents=True, exist_ok=True)
    _write_empty_train_folds(ann_dir)
    rows_out = []
    skipped = 0
    with regions_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            uid = str(row["chain_uid"])
            rid = int(row["region_id"])
            y = labels_by_uid.get(uid, {}).get(rid)
            if y is None:
                skipped += 1
                continue
            pos = np.where(y)[0].tolist()
            chain = str(row["apo_chain"])
            ann = " ".join(f"{chain}_{i}" for i in pos) if pos else "UNKNOWN"
            rows_out.append(
                [
                    str(row["apo_pdb_id"]).lower(),
                    chain,
                    str(row.get("uniprot_id") or "UNKNOWN"),
                    ann,
                    "UNKNOWN",
                    uid,
                ]
            )

    with test_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows(rows_out)

    summary = {
        "auto_built_by": "model.py",
        "pipeline_dir": str(pipeline_dir),
        "out_dir": str(ann_dir),
        "records": len(rows_out),
        "skipped": skipped,
        "format": "pdb_id;chain;uniprot_id;0_based_positive_residues;reserved;chain_uid",
    }
    (ann_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ann_dir / "README.md").write_text(
        "# Holo model annotations\n\n"
        "Auto-generated by model.py from a prepared structure pipeline. "
        "Positive residue tokens use 0-based model positions.\n",
        encoding="utf-8",
    )
    print(f"[auto] built indexed holo annotations: {test_csv} records={len(rows_out)}")
    return True


def _train_once(train_samples, val_samples, model, optimizer, device, epochs, threshold, pos_weight_tensor, flex_feature_idx):
    return train_loop(
        train_samples=train_samples,
        val_samples=val_samples,
        model=model,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        threshold=threshold,
        batch_size=FIXED["chain_batch_size"],
        grad_clip=FIXED["grad_clip_norm"],
        pos_weight_tensor=pos_weight_tensor,
        flex_feature_idx=flex_feature_idx,
        use_flex_weighting=FIXED["use_flex_weighting"],
        use_region_condition=FIXED["use_region_condition"],
        flex_weight_center=FIXED["flex_weight_center"],
        flex_weight_temp=FIXED["flex_weight_temp"],
        flex_pos_alpha=FIXED["flex_pos_alpha"],
        flex_neg_beta=FIXED["flex_neg_beta"],
        flex_neg_min=FIXED["flex_neg_min"],
        flex_weight_min=FIXED["flex_weight_min"],
        flex_weight_max=FIXED["flex_weight_max"],
        use_vq_guidance_weighting=FIXED["use_vq_guidance_weighting"],
        vq_weight_pos_alpha=FIXED["vq_weight_pos_alpha"],
        vq_weight_neg_beta=FIXED["vq_weight_neg_beta"],
        vq_weight_min=FIXED["vq_weight_min"],
        vq_weight_max=FIXED["vq_weight_max"],
        detach_vq_prior_weight=FIXED["detach_vq_prior_weight"],
        lambda_vq_prior=FIXED["lambda_vq_prior"],
        vq_loss_loop_weight=FIXED["vq_loss_loop_weight"],
        vq_loss_bg_weight=FIXED["vq_loss_bg_weight"],
        usage_on_loop_only=FIXED["usage_on_loop_only"],
        lambda_flex_aux=FIXED["lambda_flex_aux"],
        aux_target=FIXED["aux_target"],
        lambda_vq=FIXED["lambda_vq"],
        lambda_usage=FIXED["lambda_usage"],
        vq_warmup_epochs=FIXED["vq_warmup_epochs"],
        vq_ramp_epochs=FIXED["vq_ramp_epochs"],
        hard_bottleneck_from_epoch=FIXED["hard_bottleneck_from_epoch"],
        select_primary=FIXED["select_primary"],
        select_secondary=FIXED["select_secondary"],
    )


def _eval_epoch(samples, model, device, threshold, pos_weight_tensor, flex_feature_idx):
    return _iter_epoch_vq(
        samples,
        model,
        device,
        threshold,
        pos_weight_tensor,
        flex_feature_idx,
        FIXED["use_flex_weighting"],
        FIXED["use_region_condition"],
        FIXED["flex_weight_center"],
        FIXED["flex_weight_temp"],
        FIXED["flex_pos_alpha"],
        FIXED["flex_neg_beta"],
        FIXED["flex_neg_min"],
        FIXED["flex_weight_min"],
        FIXED["flex_weight_max"],
        FIXED["use_vq_guidance_weighting"],
        FIXED["vq_weight_pos_alpha"],
        FIXED["vq_weight_neg_beta"],
        FIXED["vq_weight_min"],
        FIXED["vq_weight_max"],
        FIXED["detach_vq_prior_weight"],
        FIXED["lambda_vq_prior"],
        FIXED["vq_loss_loop_weight"],
        FIXED["vq_loss_bg_weight"],
        FIXED["usage_on_loop_only"],
        FIXED["lambda_flex_aux"],
        FIXED["aux_target"],
        FIXED["lambda_vq"],
        FIXED["lambda_usage"],
        optimizer=None,
        chain_batch_size=FIXED["chain_batch_size"],
        grad_clip_norm=0.0,
    )

def main() -> None:
    args = build_parser().parse_args()
    train_paths, test_paths_raw = _resolve_paths(args)
    test_paths = {
        "ann_dir": test_paths_raw.get("test_ann_dir", train_paths["ann_dir"]),
        "emb_dir": test_paths_raw.get("test_emb_dir", train_paths["emb_dir"]),
        "pipeline_dir": test_paths_raw.get("test_pipeline_dir", train_paths["pipeline_dir"]),
        "graph_cache_dir": test_paths_raw.get("test_graph_cache_dir", train_paths["graph_cache_dir"]),
        "flex_cache_dir": test_paths_raw.get("test_flex_cache_dir", train_paths["flex_cache_dir"]),
    }

    _set_reproducibility(args.seed, args.deterministic)
    device = _resolve_device(args.device, args.gpu)
    if args.fail_if_out_dir_exists and args.out_dir.exists():
        raise FileExistsError(f"out_dir exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flex_features = _parse_feature_list(FIXED["flex_features"])
    suffix = ".final.csv"
    fold_csvs = [Path(train_paths["ann_dir"]) / f"train-fold-{i}{suffix}" for i in range(4)]
    val_csv = fold_csvs[args.val_fold]
    train_csvs = [p for i, p in enumerate(fold_csvs) if i != args.val_fold]
    test_csv = Path(test_paths["ann_dir"]) / f"test{suffix}"
    auto_built_test_annotations = False
    if not test_csv.exists() and args.test_preset.startswith("holo_cb_p2rank"):
        auto_built_test_annotations = _maybe_build_indexed_holo_annotations(test_paths, args.out_dir)
    for path in train_csvs + [val_csv, test_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")

    config = {
        "runner": "model.py",
        "model_name": "Dynamics-informed selective structural routing",
        "train_subset": args.train_subset,
        "test_preset": args.test_preset,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "runtime": {
            "epochs": args.epochs,
            "val_fold": args.val_fold,
            "seed": args.seed,
            "gpu": args.gpu,
            "device_requested": args.device,
            "device_resolved": device,
            "threshold_mode": args.threshold_mode,
            "threshold": args.threshold,
            "target_tpr": args.target_tpr,
            "target_fpr": args.target_fpr,
            "final_train_all_folds": args.final_train_all_folds,
            "deterministic": args.deterministic,
            "auto_built_test_annotations": auto_built_test_annotations,
        },
        "fixed": FIXED,
    }
    fingerprint = hashlib.md5(json.dumps(_jsonable(config), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    config["fingerprint"] = fingerprint
    (args.out_dir / "fixed_run_config.json").write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(_jsonable(config), indent=2))
        return

    train_samples, train_sk = _load_samples(train_csvs, train_paths, flex_features)
    val_samples, val_sk = _load_samples([val_csv], train_paths, flex_features)
    test_samples, test_sk = _load_samples([test_csv], test_paths, flex_features)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("No train/val/test samples loaded from annotations + embedding + graph + flex caches.")

    flex_dim = int(train_samples[0]["flex_scalar"].shape[1])
    flex_feature_idx = {name: i for i, name in enumerate(flex_features)}
    esm_dim = int(train_samples[0]["embedding"].shape[1])

    model, optimizer = _build_fixed_model(esm_dim, flex_dim, device, flex_feature_idx)
    pos_w = _pos_weight_from_samples(train_samples)
    pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
    hist_tune = _train_once(
        train_samples, val_samples, model, optimizer, device, args.epochs, args.threshold, pos_weight_tensor, flex_feature_idx
    )
    val_prob, val_y = infer_prob_y(val_samples, model, device, FIXED["chain_batch_size"])
    threshold_used = pick_threshold(args.threshold_mode, args.threshold, val_prob, val_y, args.target_tpr, args.target_fpr)

    hist_final = []
    train_mode = "3fold_train_plus_val_tune"
    if args.final_train_all_folds:
        full_train_samples, full_train_sk = _load_samples(fold_csvs, train_paths, flex_features)
        model, optimizer = _build_fixed_model(esm_dim, flex_dim, device, flex_feature_idx)
        pos_w = _pos_weight_from_samples(full_train_samples)
        pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
        hist_final = _train_once(
            full_train_samples,
            val_samples,
            model,
            optimizer,
            device,
            args.epochs,
            threshold_used,
            pos_weight_tensor,
            flex_feature_idx,
        )
        train_mode = "4fold_full_train_after_tune"
    else:
        full_train_sk = None

    val_out = _eval_epoch(val_samples, model, device, threshold_used, pos_weight_tensor, flex_feature_idx)
    test_out = _eval_epoch(test_samples, model, device, threshold_used, pos_weight_tensor, flex_feature_idx)

    summary = {
        "runner": "model.py",
        "model_name": "Dynamics-informed selective structural routing",
        "run_fingerprint": fingerprint,
        "train_mode": train_mode,
        "threshold_mode": args.threshold_mode,
        "threshold_used": float(threshold_used),
        "target_tpr": float(args.target_tpr),
        "target_fpr": float(args.target_fpr),
        "seed": int(args.seed),
        "gpu": args.gpu,
        "device": device,
        "train_subset": args.train_subset,
        "test_preset": args.test_preset,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "auto_built_test_annotations": auto_built_test_annotations,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_skipped": train_sk,
        "full_train_skipped": full_train_sk,
        "val_skipped": val_sk,
        "test_skipped": test_sk,
        "val_metrics": val_out["metrics"],
        "test_metrics": test_out["metrics"],
        "val_active_code_ratio": val_out.get("active_code_ratio", float("nan")),
        "test_active_code_ratio": test_out.get("active_code_ratio", float("nan")),
        "history_tune_epochs": len(hist_tune),
        "history_final_epochs": len(hist_final),
        "flex_dim": flex_dim,
        "esm_dim": esm_dim,
        "fixed": FIXED,
    }

    if args.collect_code_stats:
        b_idx = flex_feature_idx.get("bfactor_z")
        code_stats_val = collect_code_enrichment(
            val_samples, model, device, FIXED["chain_batch_size"], FIXED["codebook_size"], b_idx
        )
        code_stats_test = collect_code_enrichment(
            test_samples, model, device, FIXED["chain_batch_size"], FIXED["codebook_size"], b_idx
        )
        (args.out_dir / "code_stats_val.json").write_text(json.dumps(code_stats_val, indent=2), encoding="utf-8")
        (args.out_dir / "code_stats_test.json").write_text(json.dumps(code_stats_test, indent=2), encoding="utf-8")
        summary["code_stats_files"] = {
            "val": str(args.out_dir / "code_stats_val.json"),
            "test": str(args.out_dir / "code_stats_test.json"),
        }

    if hist_tune:
        import pandas as pd

        pd.DataFrame(hist_tune).to_csv(args.out_dir / "history_tune.csv", index=False)
        if hist_final:
            pd.DataFrame(hist_final).to_csv(args.out_dir / "history_final.csv", index=False)

    torch.save(model.state_dict(), args.out_dir / "best_model.pt")
    (args.out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
