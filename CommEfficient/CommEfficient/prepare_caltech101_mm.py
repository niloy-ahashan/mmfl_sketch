#!/usr/bin/env python3
"""
Build a 3-modality feature cache from the published multi-view Caltech101 features, as
used by MFedPBA (arXiv:2609.38925).

Source: Caltech101-all.mat from the mvdata Google Drive folder linked by
github.com/yeqinglee/mvdata (9,144 samples, 102 classes, 6 views). Its README asks users
to cite Li, Nie, Huang, Huang, "Large-Scale Multi-View Spectral Clustering via Bipartite
Graph", AAAI 2015, and states no licence. The features are the published ones, not
recomputed here. The images behind them are in datasets/caltech101/caltech-101.zip.

Modalities
----------
  m0 HOG   1984-D
  m1 GIST   512-D
  m2 LBP    928-D
These are the three views MFedPBA uses. They are chosen by dimension, not by position in
the file. All three describe the same image, so this is multi-view data rather than
several sensors. The file's other views (Gabor 48, WM 40, CENTRIST 254) are not used.

Labels
------
The file's 1..102 become 0..101, in the file's order. BACKGROUND_Google stays as a class,
as in the paper. Class names are in ``prepare_stats.json``.

Train/test split
----------------
Seeded 75/25 within each class, the paper's ratio. There is no official split, and the
paper does not publish its own, so sample membership differs from the paper's.

As in prepare_flash_mm.py, features get a per-feature z-score fit on train only, the
client split is Dirichlet over the classes, and the per-client files are read by
:class:`data_utils.fed_multimodal_n.FedMultiModalN`.

Usage
-----
  venv/bin/python CommEfficient/CommEfficient/prepare_caltech101_mm.py \\
    --out_dir datasets/caltech101_mm

Requirements: numpy, scipy (the root venv)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.io as sio

from prepare_ptbxl_clients import _partition

# name -> dimension, in the paper's order
MODALITIES = {"HOG": 1984, "GIST": 512, "LBP": 928}
NUM_CLASSES = 102
EXPECTED_SAMPLES = 9144
TRAIN_FRACTION = 0.75
SOURCE = "Caltech101-all.mat, https://github.com/yeqinglee/mvdata"


def _cell_strings(cell: np.ndarray) -> list[str]:
    """MATLAB cell array of strings -> list of str."""
    return [str(np.asarray(c).ravel()[0]) for c in cell.ravel()]


def load_mat(path: str) -> tuple[list[np.ndarray], np.ndarray, list[str], list[str]]:
    """Return ([HOG, GIST, LBP], labels 0..101, class names, the file's view names)."""
    d = sio.loadmat(path)
    views = [np.asarray(v) for v in d["X"].ravel()]
    by_dim = {v.shape[1]: v for v in views}
    if len(by_dim) != len(views):
        raise SystemExit(f"View dimensions are not unique: {[v.shape for v in views]}")
    missing = [n for n, dim in MODALITIES.items() if dim not in by_dim]
    if missing:
        raise SystemExit(f"No view of the expected dimension for {missing}: "
                         f"{[v.shape for v in views]}")
    mods = [by_dim[dim].astype(np.float32) for dim in MODALITIES.values()]

    labels = d["Y"].ravel().astype(np.int64) - 1
    if len(labels) != EXPECTED_SAMPLES or np.unique(labels).tolist() != list(range(NUM_CLASSES)):
        raise SystemExit(f"Expected {EXPECTED_SAMPLES} samples with labels 1..{NUM_CLASSES}, got "
                         f"{len(labels)} samples and {len(np.unique(labels))} classes")
    for name, m in zip(MODALITIES, mods):
        if len(m) != len(labels) or not np.isfinite(m).all():
            raise SystemExit(f"{name}: shape {m.shape} or non-finite values")
    return mods, labels, _cell_strings(d["categories"]), _cell_strings(d["feanames"])


def build_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted (train_idx, test_idx), split within each class."""
    rng = np.random.RandomState(seed)
    train, test = [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cut = int(TRAIN_FRACTION * len(idx))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def main() -> None:
    p = argparse.ArgumentParser(description="Caltech101 multi-view features -> 3-modality cache")
    p.add_argument("--mat_path", type=str, default="datasets/caltech101/Caltech101-all.mat")
    p.add_argument("--out_dir", type=str, default="datasets/caltech101_mm")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed of the train/test split and of the client split")
    p.add_argument("--force", action="store_true", help="replace an existing cache")
    args = p.parse_args()

    stats_path = os.path.join(args.out_dir, "stats.json")
    if os.path.exists(stats_path) and not args.force:
        raise SystemExit(f"{stats_path} already exists; pass --force to replace this cache")

    mods, labels, categories, file_views = load_mat(args.mat_path)
    names = list(MODALITIES)

    train_idx, test_idx = build_split(labels, args.seed)
    lbl_tr, lbl_te = labels[train_idx], labels[test_idx]

    # ---- Client split first: nothing is written if a client comes out empty ----
    client_indices = _partition(lbl_tr, NUM_CLASSES, args.num_clients,
                                args.dirichlet_alpha, args.seed)
    sizes = [len(idx) for idx in client_indices]
    if min(sizes) == 0:
        raise SystemExit(f"Empty client with seed {args.seed}: {sizes}")

    # ---- Train-statistics z-score, as in prepare_flash_mm.py ----
    mods_tr, mods_te = [], []
    for m in mods:
        train, test = m[train_idx], m[test_idx]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        mods_tr.append(((train - mean) / std).astype(np.float32))
        mods_te.append(((test - mean) / std).astype(np.float32))
    mod_dims = [int(m.shape[1]) for m in mods_tr]

    os.makedirs(args.out_dir, exist_ok=True)
    # The cache is incomplete until stats.json is rewritten below.
    if os.path.exists(stats_path):
        os.remove(stats_path)

    savez_kwargs = {
        "labels_train": lbl_tr,
        "labels_test": lbl_te,
        "index_train": train_idx,
        "index_test": test_idx,
        "mod_names": np.array(names),
    }
    for k in range(len(names)):
        savez_kwargs[f"m{k}_train"] = mods_tr[k]
        savez_kwargs[f"m{k}_test"] = mods_te[k]
    np.savez(os.path.join(args.out_dir, "data.npz"), **savez_kwargs)

    missing_train = np.flatnonzero(np.bincount(lbl_tr, minlength=NUM_CLASSES) == 0).tolist()
    missing_test = np.flatnonzero(np.bincount(lbl_te, minlength=NUM_CLASSES) == 0).tolist()
    stats = {
        "dataset": "Caltech101",
        "source": SOURCE,
        "modalities": names,
        "mod_dims": mod_dims,
        "num_classes": NUM_CLASSES,
        "num_modalities": len(names),
        "class_names": categories,
        "views_in_file": file_views,
        "test_split": "stratified",
        "train_fraction": TRAIN_FRACTION,
        "seed": args.seed,
        "n_train": int(len(lbl_tr)),
        "n_test": int(len(lbl_te)),
        "classes_missing_from_train": missing_train,
        "classes_missing_from_test": missing_test,
    }
    with open(os.path.join(args.out_dir, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    small = 0
    for cid, idx in enumerate(client_indices):
        idx = np.array(idx)
        payload = {f"m{k}_feats": mods_tr[k][idx] for k in range(len(names))}
        np.savez(os.path.join(args.out_dir, f"client{cid}.npz"), labels=lbl_tr[idx], **payload)
        counts = np.bincount(lbl_tr[idx], minlength=NUM_CLASSES)
        small += int(len(idx) < NUM_CLASSES)
        print(f"  client {cid}: {len(idx)} samples, {int((counts > 0).sum())} classes, "
              f"largest class {int(counts.max())} ({counts.max() / len(idx):.0%})")

    payload = {f"m{k}_feats": mods_te[k] for k in range(len(names))}
    np.savez(os.path.join(args.out_dir, "test.npz"), labels=lbl_te, **payload)

    # stats.json last: its presence is what marks the cache as complete.
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "images_per_client": sizes,
            "num_val_images": int(len(lbl_te)),
            "multi_label": False,
            "num_classes": NUM_CLASSES,
            "num_modalities": len(names),
            "partition_seed": args.seed,
        }, f)

    print(f"\nWrote {os.path.abspath(args.out_dir)}")
    print(f"  train {stats['n_train']}  test {stats['n_test']}  "
          f"num_classes={NUM_CLASSES}  mod_dims={mod_dims}")
    print(f"  samples/client {sizes}")
    print(f"  clients with fewer than {NUM_CLASSES} samples: {small}")
    print(f"  classes missing from train {missing_train}, from test {missing_test}")


if __name__ == "__main__":
    main()
