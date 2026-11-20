#!/usr/bin/env python3
"""
Build a 12-modality feature cache from PTB-XL v1.0.3: each ECG lead is one
modality (I, II, III, aVR, aVL, aVF, V1-V6).

Each lead is a hand-crafted vector from ``extract_group_features`` in
``prepare_mhealth_mm.py``, called on that lead alone: 1 lead x 14 + magnitude 14
= 28-D. There are no correlation features, because they need two or more leads
in the same modality.

Records, labels and the train/test split are the same as ``prepare_ptbxl_mm.py``
(``--match_dir``), and so is the client split: samples go to the same clients as
that cache, using the seed saved in its ``stats.json``. The script stops if any
of these differ. Nothing in ``--match_dir`` is written.

Keys are ``m{k}_train``/``m{k}_test``; the per-client files are read by
:class:`data_utils.fed_multimodal_n.FedMultiModalN`.

Usage
-----
  python CommEfficient/CommEfficient/prepare_ptbxl_12mod_mm.py \\
    --ptbxl_root datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
    --match_dir  datasets/ptbxl_mm \\
    --out_dir    datasets/ptbxl_12mod_mm

Requirements: wfdb, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import warnings

import numpy as np
import pandas as pd

from prepare_mhealth_mm import extract_group_features
from prepare_ptbxl import (LIMB_LEADS, PRECORDIAL_LEADS, SUPERCLASSES,
                           _build_labels, _load_signals)
from prepare_ptbxl_clients import _partition
from prepare_ptbxl_mm import _clean

LEADS = LIMB_LEADS + PRECORDIAL_LEADS

_LEAD_IDX: list[int] = []


def _init_worker(lead_idx: list[int]) -> None:
    global _LEAD_IDX
    _LEAD_IDX = lead_idx
    # Flat leads make skew/kurtosis warn; the NaNs are handled in _clean.
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _record_features(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float64)
    return np.stack([extract_group_features(signal[:, [i]]) for i in _LEAD_IDX])


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL -> 12-modality feature cache")
    p.add_argument("--ptbxl_root", type=str, required=True,
                   help="Path to ptb-xl-...-1.0.3 directory")
    p.add_argument("--match_dir", type=str, default="datasets/ptbxl_mm",
                   help="2-modality cache whose records and client split are reused")
    p.add_argument("--out_dir", type=str, default="datasets/ptbxl_12mod_mm")
    p.add_argument("--min_likelihood", type=float, default=50.0,
                   help="Minimum likelihood for SCP code to count")
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8,
                   help="Processes for feature extraction")
    p.add_argument("--force", action="store_true", help="replace an existing cache")
    args = p.parse_args()

    stats_path = os.path.join(args.out_dir, "stats.json")
    if os.path.exists(stats_path) and not args.force:
        raise SystemExit(f"{stats_path} already exists; pass --force to replace this cache")

    import wfdb

    root = os.path.abspath(args.ptbxl_root)
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)

    # ---- Single-label records ----
    labels = _build_labels(df, scp_df, args.min_likelihood)
    single = labels.sum(axis=1) == 1
    cls = labels.argmax(axis=1).astype(np.int64)
    splits = {
        "train": np.flatnonzero(single & (df["strat_fold"] <= 8).values),
        "test": np.flatnonzero(single & (df["strat_fold"] >= 9).values),
    }

    # ---- Same records, order and labels as the 2-modality cache ----
    match = np.load(os.path.join(args.match_dir, "data.npz"))
    with open(os.path.join(args.match_dir, "prepare_stats.json"), encoding="utf-8") as f:
        match_prep = json.load(f)
    with open(os.path.join(args.match_dir, "stats.json"), encoding="utf-8") as f:
        match_stats = json.load(f)
    for split, idx in splits.items():
        if df.loc[idx, "ecg_id"].astype(int).tolist() != match_prep[f"ecg_ids_{split}"]:
            raise SystemExit(f"{split} records differ from {args.match_dir}")
        if not np.array_equal(cls[idx], match[f"labels_{split}"]):
            raise SystemExit(f"{split} labels differ from {args.match_dir}")

    # ---- Lead positions ----
    rec0 = wfdb.rdrecord(os.path.join(root, df.loc[splits["train"][0], "filename_lr"]))
    lead_order = [str(s).upper() for s in rec0.sig_name]
    lead_idx = [lead_order.index(l) for l in LEADS]

    # ---- Features: (N, 12, 28) ----
    feats = {}
    for split, idx in splits.items():
        print(f"\nLoading {split} signals ({len(idx)}) ...", flush=True)
        x = _load_signals(root, df.loc[idx, "filename_lr"].tolist(), 100)
        print(f"Extracting {split} features with {args.workers} workers ...", flush=True)
        with mp.Pool(args.workers, _init_worker, (lead_idx,)) as pool:
            out = np.stack(pool.map(_record_features, list(x), chunksize=64))
        feats[split] = _clean(out.reshape(len(out), -1), split).reshape(out.shape)

    # ---- Train-statistics z-score ----
    mean = feats["train"].mean(axis=0, keepdims=True)
    std = feats["train"].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    feats = {s: ((v - mean) / std).astype(np.float32) for s, v in feats.items()}
    lbl_tr, lbl_te = cls[splits["train"]], cls[splits["test"]]
    num_mod, mod_dim = feats["train"].shape[1:]

    os.makedirs(args.out_dir, exist_ok=True)
    savez_kwargs = {"labels_train": lbl_tr, "labels_test": lbl_te}
    for k in range(num_mod):
        savez_kwargs[f"m{k}_train"] = feats["train"][:, k]
        savez_kwargs[f"m{k}_test"] = feats["test"][:, k]
    np.savez(os.path.join(args.out_dir, "data.npz"), **savez_kwargs)

    stats = {
        "dataset": "PTB-XL",
        "source": root,
        "modalities": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "mod_dims": [int(mod_dim)] * int(num_mod),
        "num_classes": len(SUPERCLASSES),
        "num_modalities": int(num_mod),
        "class_names": SUPERCLASSES,
        "sampling_rate": 100,
        "min_likelihood": args.min_likelihood,
        "split_rule": "single superclass; strat_fold 1-8 train, 9-10 test; no subsampling",
        "n_train": int(len(lbl_tr)),
        "n_test": int(len(lbl_te)),
    }
    with open(os.path.join(args.out_dir, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)

    # ---- Client split: same seed, so same clients as the 2-modality cache ----
    seed = match_stats["partition_seed"]
    n_clients = len(match_stats["images_per_client"])
    client_indices = _partition(lbl_tr, len(SUPERCLASSES), n_clients, args.dirichlet_alpha, seed)
    sizes = [len(idx) for idx in client_indices]
    if sizes != match_stats["images_per_client"]:
        raise SystemExit(f"Client sizes {sizes} differ from {match_stats['images_per_client']}")

    for cid, idx in enumerate(client_indices):
        idx = np.array(idx)
        payload = {f"m{k}_feats": feats["train"][idx, k] for k in range(num_mod)}
        np.savez(os.path.join(args.out_dir, f"client{cid}.npz"), labels=lbl_tr[idx], **payload)
        print(f"  client {cid}: {len(idx)} {np.bincount(lbl_tr[idx], minlength=len(SUPERCLASSES)).tolist()}")

    payload = {f"m{k}_feats": feats["test"][:, k] for k in range(num_mod)}
    np.savez(os.path.join(args.out_dir, "test.npz"), labels=lbl_te, **payload)

    # stats.json last: its presence is what marks the cache as complete.
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "images_per_client": sizes,
            "num_val_images": int(len(lbl_te)),
            "multi_label": False,
            "num_classes": len(SUPERCLASSES),
            "num_modalities": int(num_mod),
            "partition_seed": seed,
        }, f)

    print(f"\nWrote {os.path.abspath(args.out_dir)}")
    print(f"  train {stats['n_train']}  test {stats['n_test']}  mod_dims={stats['mod_dims']}")
    print(f"  partition seed {seed}, samples/client {sizes}")


if __name__ == "__main__":
    main()
