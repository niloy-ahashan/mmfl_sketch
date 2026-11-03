#!/usr/bin/env python3
"""
Build a PTBXL_REDUCE folder for PEPSY [Nguyen et al., NeurIPS 2025].

PEPSY's loader (``PTBXLReduceDataset`` in
``src/benchmark/ptbxl_classification_lm/dataset.py``) reads a preprocessed
folder the authors did not release. This script rebuilds one in the same
format from raw PTB-XL v1.0.3. How the authors picked their records is
undocumented, so this is a reconstruction at the same scale, not a copy:

* records with exactly one diagnostic superclass (likelihood >= 50%),
  NORM / MI / STTC / CD / HYP
* official patient-wise folds: 1-8 train, 9-10 test
* stratified subsample to 3,170 train (the size in a PEPSY loader comment)
  and 800 test, keeping the class mix of the full single-label set
* raw 100Hz waveforms in mV, shape (N, 1000, 12), wfdb lead order

Output files (what the loader expects):

* ``x_train.npy`` / ``x_test.npy``  float32 (N, 1000, 12)
* ``y_train.npy`` / ``y_test.npy``  int64 PEPSY label codes; the loader
  remaps 1, 5, 6, 7, 8 -> 0..4 in place, so codes are stored, not 0..4:
  NORM=1, MI=5, STTC=6, CD=7, HYP=8
* ``standard_scaler.pkl``  sklearn StandardScaler fit on x_train.reshape(-1, 1)
  (the loader calls ``ss.transform(x.flatten()[:, None])``)
* ``meta.json``  seed, class counts, ecg_ids, scaler mean/std

Usage
-----
  python CommEfficient/CommEfficient/prepare_ptbxl_reduce.py \\
    --ptbxl_root datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
    --out_dir    datasets/ptbxl_reduce

Requirements: wfdb, pandas, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from prepare_ptbxl import SUPERCLASSES, _build_labels, _load_signals

# PEPSY label codes, in SUPERCLASSES order (NORM, MI, STTC, CD, HYP)
PEPSY_CODES = [1, 5, 6, 7, 8]


def _class_quota(counts: np.ndarray, n: int) -> np.ndarray:
    """Per-class sample counts proportional to ``counts``, summing to ``n``
    (largest-remainder rounding)."""
    exact = counts / counts.sum() * n
    quota = np.floor(exact).astype(int)
    order = np.argsort(-(exact - quota), kind="stable")
    quota[order[: n - quota.sum()]] += 1
    return quota


def _stratified_pick(idx: np.ndarray, cls: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` of ``idx`` without replacement, stratified by ``cls``."""
    counts = np.bincount(cls, minlength=len(SUPERCLASSES))
    if n > counts.sum():
        raise ValueError(f"asked for {n} records, only {counts.sum()} available")
    quota = _class_quota(counts, n)
    picked = [rng.choice(idx[cls == c], size=q, replace=False)
              for c, q in enumerate(quota)]
    return np.sort(np.concatenate(picked))


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL -> PEPSY PTBXL_REDUCE folder")
    p.add_argument("--ptbxl_root", type=str, required=True,
                   help="Path to ptb-xl-...-1.0.3 directory")
    p.add_argument("--out_dir", type=str, default="datasets/ptbxl_reduce")
    p.add_argument("--n_train", type=int, default=3170)
    p.add_argument("--n_test", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500],
                   help="100 Hz gives the 1000-sample records PEPSY expects")
    p.add_argument("--min_likelihood", type=float, default=50.0,
                   help="Minimum likelihood for SCP code to count")
    args = p.parse_args()

    from sklearn.preprocessing import StandardScaler

    root = os.path.abspath(args.ptbxl_root)
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    scp_df = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    fn_col = "filename_lr" if args.sampling_rate == 100 else "filename_hr"

    # ---- Single-label records ----
    labels = _build_labels(df, scp_df, args.min_likelihood)
    single = labels.sum(axis=1) == 1
    cls = labels.argmax(axis=1)
    train_pool = np.flatnonzero(single & (df["strat_fold"] <= 8).values)
    test_pool = np.flatnonzero(single & (df["strat_fold"] >= 9).values)
    print(f"Single-label records: {len(train_pool)} train pool (folds 1-8), "
          f"{len(test_pool)} test pool (folds 9-10)")

    # ---- Stratified subsample ----
    rng = np.random.default_rng(args.seed)
    splits = {
        "train": _stratified_pick(train_pool, cls[train_pool], args.n_train, rng),
        "test": _stratified_pick(test_pool, cls[test_pool], args.n_test, rng),
    }
    overlap = set(df.loc[splits["train"], "patient_id"]) & set(df.loc[splits["test"], "patient_id"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} patients appear in both splits")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "note": "Reconstruction of PEPSY's unreleased PTBXL_REDUCE from PTB-XL v1.0.3; "
                "the authors' record selection is not documented.",
        "source": root,
        "sampling_rate": args.sampling_rate,
        "seed": args.seed,
        "min_likelihood": args.min_likelihood,
        "label_codes": dict(zip(SUPERCLASSES, PEPSY_CODES)),
        "loader_index": {sc: i for i, sc in enumerate(SUPERCLASSES)},
        "split_rule": "single superclass; strat_fold 1-8 train, 9-10 test; "
                      "stratified subsample without replacement",
        "pool_sizes": {"train": int(len(train_pool)), "test": int(len(test_pool))},
    }

    # ---- Load signals, encode labels, save ----
    x_train = None
    for split, idx in splits.items():
        print(f"\nLoading {split} signals ({len(idx)}) ...", flush=True)
        x = _load_signals(root, df.loc[idx, fn_col].tolist(), args.sampling_rate)
        y = np.asarray(PEPSY_CODES, dtype=np.int64)[cls[idx]]
        np.save(os.path.join(args.out_dir, f"x_{split}.npy"), x)
        np.save(os.path.join(args.out_dir, f"y_{split}.npy"), y)
        counts = np.bincount(cls[idx], minlength=len(SUPERCLASSES))
        meta[f"n_{split}"] = int(len(idx))
        meta[f"class_counts_{split}"] = dict(zip(SUPERCLASSES, counts.tolist()))
        meta[f"ecg_ids_{split}"] = df.loc[idx, "ecg_id"].astype(int).tolist()
        print(f"  x_{split} {x.shape}  classes {meta[f'class_counts_{split}']}")
        if split == "train":
            x_train = x

    import wfdb
    rec0 = wfdb.rdrecord(os.path.join(root, df.loc[splits["train"][0], fn_col]))
    meta["lead_order"] = list(rec0.sig_name)

    # ---- Scalar StandardScaler, as the loader uses it ----
    ss = StandardScaler().fit(x_train.reshape(-1, 1))
    with open(os.path.join(args.out_dir, "standard_scaler.pkl"), "wb") as f:
        pickle.dump(ss, f)
    meta["scaler"] = {"mean": float(ss.mean_[0]), "std": float(ss.scale_[0])}

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"\nWrote {os.path.abspath(args.out_dir)}")
    print(f"  lead order: {meta['lead_order']}")
    print(f"  scaler mean={meta['scaler']['mean']:.4f} std={meta['scaler']['std']:.4f}")


if __name__ == "__main__":
    main()
