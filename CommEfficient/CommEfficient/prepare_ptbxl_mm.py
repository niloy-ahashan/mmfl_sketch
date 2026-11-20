#!/usr/bin/env python3
"""
Build a 2-modality feature cache from PTB-XL v1.0.3, for the shared protocol
that UCI HAR and mHealth use (B1-B3, PMFH, FLASH, FL-FD, FedMobile).

Modalities (the limb / precordial split ``prepare_ptbxl.py`` uses):

* ``img_*`` (Limb)  — leads I, II, III, aVR, aVL, aVF
* ``txt_*`` (Chest) — leads V1, V2, V3, V4, V5, V6

Each modality is a hand-crafted vector from ``extract_group_features`` in
``prepare_mhealth_mm.py``: 6 leads x 14 + magnitude 14 + C(6,2) correlations =
113-D. Keys follow ``prepare_uci_har_mm.py`` so 2-modality loaders can read it.

Records and labels:

* exactly one diagnostic superclass (likelihood >= 50%), so single-label:
  0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP
* official patient-wise folds: 1-8 train, 9-10 test; no subsampling

Features are z-scored with train statistics. Unlike ``prepare_mhealth_mm.py``,
NaN/inf are replaced *before* the statistics are computed: a flat lead gives a
NaN skew, which would otherwise turn a whole column's mean into NaN.

Usage
-----
  python CommEfficient/CommEfficient/prepare_ptbxl_mm.py \\
    --ptbxl_root datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
    --out_dir    datasets/ptbxl_mm

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

_LEAD_IDX: tuple[list[int], list[int]] = ([], [])


def _init_worker(limb_idx: list[int], chest_idx: list[int]) -> None:
    global _LEAD_IDX
    _LEAD_IDX = (limb_idx, chest_idx)
    # Flat leads make skew/kurtosis/corrcoef warn; the NaNs are handled in _clean.
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _record_features(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    limb_idx, chest_idx = _LEAD_IDX
    signal = signal.astype(np.float64)
    return (extract_group_features(signal[:, limb_idx]),
            extract_group_features(signal[:, chest_idx]))


def _clean(feats: np.ndarray, name: str) -> np.ndarray:
    bad = ~np.isfinite(feats)
    print(f"  {name}: {int(bad.any(axis=1).sum())} records with NaN/inf features "
          f"({int(bad.sum())} values) set to 0")
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL -> 2-modality feature cache")
    p.add_argument("--ptbxl_root", type=str, required=True,
                   help="Path to ptb-xl-...-1.0.3 directory")
    p.add_argument("--out_dir", type=str, default="datasets/ptbxl_mm")
    p.add_argument("--min_likelihood", type=float, default=50.0,
                   help="Minimum likelihood for SCP code to count")
    p.add_argument("--workers", type=int, default=8,
                   help="Processes for feature extraction")
    args = p.parse_args()

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
    print(f"Single-label records: {len(splits['train'])} train (folds 1-8), "
          f"{len(splits['test'])} test (folds 9-10), "
          f"of {len(df)} total")
    overlap = set(df.loc[splits["train"], "patient_id"]) & set(df.loc[splits["test"], "patient_id"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} patients appear in both splits")

    # ---- Lead positions ----
    rec0 = wfdb.rdrecord(os.path.join(root, df.loc[splits["train"][0], "filename_lr"]))
    lead_order = [str(s).upper() for s in rec0.sig_name]
    limb_idx = [lead_order.index(l) for l in LIMB_LEADS]
    chest_idx = [lead_order.index(l) for l in PRECORDIAL_LEADS]

    # ---- Features ----
    feats = {}
    for split, idx in splits.items():
        print(f"\nLoading {split} signals ({len(idx)}) ...", flush=True)
        x = _load_signals(root, df.loc[idx, "filename_lr"].tolist(), 100)
        print(f"Extracting {split} features with {args.workers} workers ...", flush=True)
        with mp.Pool(args.workers, _init_worker, (limb_idx, chest_idx)) as pool:
            out = pool.map(_record_features, list(x), chunksize=64)
        feats[split] = (_clean(np.stack([o[0] for o in out]), f"{split} limb"),
                        _clean(np.stack([o[1] for o in out]), f"{split} chest"))

    # ---- Train-statistics z-score, save ----
    savez_kwargs = {"labels_train": cls[splits["train"]], "labels_test": cls[splits["test"]]}
    mod_dims = []
    for k, key in enumerate(["img", "txt"]):
        train, test = feats["train"][k], feats["test"][k]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        savez_kwargs[f"{key}_train"] = ((train - mean) / std).astype(np.float32)
        savez_kwargs[f"{key}_test"] = ((test - mean) / std).astype(np.float32)
        mod_dims.append(int(train.shape[1]))

    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "data.npz")
    np.savez(out_npz, **savez_kwargs)

    stats = {
        "dataset": "PTB-XL",
        "source": root,
        "modalities": ["Limb", "Chest"],
        "limb_leads": LIMB_LEADS,
        "chest_leads": PRECORDIAL_LEADS,
        "mod_dims": mod_dims,
        "num_classes": len(SUPERCLASSES),
        "num_modalities": 2,
        "class_names": SUPERCLASSES,
        "sampling_rate": 100,
        "min_likelihood": args.min_likelihood,
        "split_rule": "single superclass; strat_fold 1-8 train, 9-10 test; no subsampling",
        "n_train": int(len(splits["train"])),
        "n_test": int(len(splits["test"])),
    }
    for split, idx in splits.items():
        counts = np.bincount(cls[idx], minlength=len(SUPERCLASSES))
        stats[f"class_counts_{split}"] = dict(zip(SUPERCLASSES, counts.tolist()))
        stats[f"ecg_ids_{split}"] = df.loc[idx, "ecg_id"].astype(int).tolist()
    stats_path = os.path.join(args.out_dir, "prepare_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)

    print(f"\nWrote {os.path.abspath(out_npz)}")
    print(f"  train {stats['n_train']}  test {stats['n_test']}  mod_dims={mod_dims}")
    print(f"  train classes {stats['class_counts_train']}")
    print(f"  test classes  {stats['class_counts_test']}")
    print(f"  stats -> {stats_path}")


if __name__ == "__main__":
    main()
