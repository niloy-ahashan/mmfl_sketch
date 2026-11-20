#!/usr/bin/env python3
"""
Build a copy of the 12-modality PTB-XL cache with a different client split.

Only the partition seed changes. Features, labels, the train/test split and the
z-scoring come from ``--source_dir``'s ``data.npz``, which is only read, so no
ECG is reprocessed and the test set stays identical.

Results are named after the dataset folder and not the partition seed, so each
split needs its own ``--out_dir``. Scripts that hardcode ``ptbxl_12mod_mm`` won't
find the new cache.

Usage
-----
  python CommEfficient/CommEfficient/prepare_ptbxl_12mod_repartition.py \\
    --source_dir datasets/ptbxl_12mod_mm \\
    --out_dir    datasets/ptbxl_12mod_mm_p44 \\
    --partition_seed 44
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from prepare_ptbxl_clients import _partition


def main() -> None:
    p = argparse.ArgumentParser(description="PTB-XL 12-modality cache with a new client split")
    p.add_argument("--source_dir", type=str, default="datasets/ptbxl_12mod_mm")
    p.add_argument("--out_dir", type=str, default="datasets/ptbxl_12mod_mm_p44")
    p.add_argument("--partition_seed", type=int, default=44)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    args = p.parse_args()

    if os.path.abspath(args.source_dir) == os.path.abspath(args.out_dir):
        raise SystemExit("--out_dir must differ from --source_dir")
    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} already exists; choose another --out_dir")

    raw = np.load(os.path.join(args.source_dir, "data.npz"))
    with open(os.path.join(args.source_dir, "prepare_stats.json"), encoding="utf-8") as f:
        prep = json.load(f)
    lbl_tr, lbl_te = raw["labels_train"], raw["labels_test"]
    num_mod, num_classes = prep["num_modalities"], prep["num_classes"]

    client_indices = _partition(lbl_tr, num_classes, args.num_clients,
                                args.dirichlet_alpha, args.partition_seed)
    sizes = [len(idx) for idx in client_indices]
    if min(sizes) == 0:
        raise SystemExit(f"Partition seed {args.partition_seed} leaves an empty client: {sizes}")

    os.makedirs(args.out_dir)
    for name in ("data.npz", "prepare_stats.json"):
        shutil.copy2(os.path.join(args.source_dir, name), os.path.join(args.out_dir, name))

    for cid, idx in enumerate(client_indices):
        idx = np.array(idx)
        payload = {f"m{k}_feats": raw[f"m{k}_train"][idx] for k in range(num_mod)}
        np.savez(os.path.join(args.out_dir, f"client{cid}.npz"), labels=lbl_tr[idx], **payload)
        print(f"  client {cid}: {len(idx)} {np.bincount(lbl_tr[idx], minlength=num_classes).tolist()}")

    payload = {f"m{k}_feats": raw[f"m{k}_test"] for k in range(num_mod)}
    np.savez(os.path.join(args.out_dir, "test.npz"), labels=lbl_te, **payload)

    # stats.json last: its presence is what marks the cache as complete.
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "images_per_client": sizes,
            "num_val_images": int(len(lbl_te)),
            "multi_label": False,
            "num_classes": num_classes,
            "num_modalities": num_mod,
            "partition_seed": args.partition_seed,
        }, f)

    print(f"\nWrote {os.path.abspath(args.out_dir)}")
    print(f"  partition seed {args.partition_seed}, samples/client {sizes}")


if __name__ == "__main__":
    main()
