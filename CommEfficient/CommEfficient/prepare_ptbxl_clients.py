#!/usr/bin/env python3
"""
Build the per-client cache of ``datasets/ptbxl_mm`` with no empty client.

``FedMultiModal.prepare_datasets`` always partitions with ``RandomState(42)``,
which on PTB-XL gives client 2 zero samples. ``mm_train.run_batches`` skips any
round with fewer than ``num_workers`` clients, so a 10-worker run would then
never train. This script uses the same Dirichlet algorithm, but tries seeds from
``--start_seed`` upward and keeps the first split where every client has at
least ``--min_samples`` samples.

Writes ``client{i}.npz``, ``test.npz`` and ``stats.json`` in the format
``FedMultiModal`` loads. ``data.npz`` and ``prepare_stats.json`` are only read,
so the train/test samples, features and labels are unchanged.

Usage
-----
  python CommEfficient/CommEfficient/prepare_ptbxl_clients.py --data_dir datasets/ptbxl_mm
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _partition(labels, num_classes, n_clients, alpha, seed):
    """Same steps as FedMultiModal.prepare_datasets, with a chosen seed."""
    rng = np.random.RandomState(seed)
    label_dist = rng.dirichlet([alpha] * n_clients, num_classes)
    client_indices = [[] for _ in range(n_clients)]
    for i in range(len(labels)):
        cid = rng.choice(n_clients, p=label_dist[int(labels[i])])
        client_indices[cid].append(i)
    return client_indices


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL client cache with no empty client")
    p.add_argument("--data_dir", type=str, default="datasets/ptbxl_mm")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--start_seed", type=int, default=42)
    p.add_argument("--max_tries", type=int, default=100)
    p.add_argument("--min_samples", type=int, default=1)
    p.add_argument("--force", action="store_true", help="replace an existing client split")
    args = p.parse_args()

    stats_path = os.path.join(args.data_dir, "stats.json")
    if os.path.exists(stats_path) and not args.force:
        raise SystemExit(f"{stats_path} already exists; pass --force to replace this client split")

    raw = np.load(os.path.join(args.data_dir, "data.npz"))
    lbl_tr, lbl_te = raw["labels_train"], raw["labels_test"]
    num_classes = int(lbl_tr.max()) + 1

    for seed in range(args.start_seed, args.start_seed + args.max_tries):
        client_indices = _partition(lbl_tr, num_classes, args.num_clients, args.dirichlet_alpha, seed)
        sizes = [len(idx) for idx in client_indices]
        print(f"seed {seed}: samples/client {sizes}")
        if min(sizes) >= args.min_samples:
            break
    else:
        raise SystemExit(f"No split with min_samples={args.min_samples} in {args.max_tries} seeds")

    for cid, idx in enumerate(client_indices):
        idx = np.array(idx)
        np.savez(
            os.path.join(args.data_dir, f"client{cid}.npz"),
            img_feats=raw["img_train"][idx].astype(np.float32),
            txt_feats=raw["txt_train"][idx].astype(np.float32),
            labels=lbl_tr[idx].astype(np.int64),
        )
        print(f"  client {cid}: {len(idx)} {np.bincount(lbl_tr[idx], minlength=num_classes).tolist()}")

    np.savez(
        os.path.join(args.data_dir, "test.npz"),
        img_feats=raw["img_test"].astype(np.float32),
        txt_feats=raw["txt_test"].astype(np.float32),
        labels=lbl_te.astype(np.int64),
    )
    # stats.json last: its presence is what marks the cache as complete.
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "images_per_client": sizes,
            "num_val_images": int(len(lbl_te)),
            "multi_label": False,
            "num_classes": num_classes,
            "partition_seed": seed,
        }, f)
    print(f"Partition seed {seed}; client cache written to {args.data_dir}")


if __name__ == "__main__":
    main()
