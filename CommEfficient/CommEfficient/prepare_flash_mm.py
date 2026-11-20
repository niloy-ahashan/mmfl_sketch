#!/usr/bin/env python3
"""
Build a 3-modality feature cache from the FLASH processed dataset (Salehi et al.,
INFOCOM'22), reading directly from its zip. The zip is never extracted.

Source: http://hdl.handle.net/2047/D20423471 (210 episodes, 31,923 samples), one
``<...>/episode_<n>/npz/{rf,gps,image,lidar}.npz`` per episode.

Labels
------
The best sector is ``argmax`` over the 64 ``rf`` columns (ties resolve to the first
index; upstream's argsort picks another one for 10 rows). The 411 all-zero rows have no
best sector and are dropped. The 49 sectors that occur are remapped to 0..48, with the
mapping in ``prepare_stats.json``. Upstream instead keeps all rows and a 64-way output.

Modalities
----------
  m0 GPS    2-D     the stored values
  m1 Image  512-D   frozen ImageNet ResNet18 average-pool output. Stored frames are BGR
                    (upstream's cv2.imread without conversion), flipped to RGB here.
  m2 LiDAR  1006-D  obstacle density, 20^3 grid pooled 2x2x2 (1000), plus the Tx and Rx
                    voxel coordinates scaled to 0..1 (6; -1 if the marker is absent).
                    Stored values are categories (obstacle 1, Tx -1, Rx -2), so they are
                    separated first: averaging the raw values would cancel them out.

Train/test split (``--test_split``; use a separate ``--out_dir`` for each)
--------------------------------------------------------------------------
  random   seeded 90/10 within each episode: the paper's 80+10 train and validation
           portions merged, since all client data is training data here. CAUTION:
           neighbouring frames are close in time and share labels, so near-duplicates
           can land on opposite sides of this split.
  episode  all 21 episodes numbered 9 are the test set, so no episode is on both sides.

As in prepare_mhealth_mm.py, features get a per-feature z-score fit on train only. The
client split is Dirichlet over the 49 classes. Keys are ``m{k}_train``/``m{k}_test``; the
per-client files are read by :class:`data_utils.fed_multimodal_n.FedMultiModalN`.

Features before splitting are kept in ``--features_cache``, so a second split reuses them.

Usage
-----
  venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py \\
    --out_dir datasets/flash_mm
  venv/bin/python CommEfficient/CommEfficient/prepare_flash_mm.py \\
    --test_split episode --out_dir datasets/flash_mm_episode

Requirements: torch, torchvision (the root venv; MFedMC/.venv has no torchvision)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile

import numpy as np

from prepare_ptbxl_clients import _partition

MODALITIES = ["GPS", "Image", "LiDAR"]
NUM_CLASSES = 49
EXPECTED_RAW = 31923
EXPECTED_KEPT = 31512
TEST_EPISODE_NUMBER = 9
TRAIN_FRACTION = 0.9
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _read_npz(z: zipfile.ZipFile, name: str, key: str) -> np.ndarray:
    with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as d:
        return d[key]


def _lidar_features(lidar: np.ndarray) -> tuple[np.ndarray, int]:
    """(N, 20, 20, 20) categories -> (N, 1006) features, and the count of missing markers."""
    n = len(lidar)
    obstacles = (lidar == 1).astype(np.float32)
    pooled = obstacles.reshape(n, 10, 2, 10, 2, 10, 2).mean(axis=(2, 4, 6)).reshape(n, -1)
    coords = np.full((n, 6), -1.0, dtype=np.float32)
    missing = 0
    for slot, value in enumerate((-1, -2)):  # Tx, Rx
        mask = lidar == value
        present = mask.any(axis=(1, 2, 3))
        missing += int((~present).sum())
        flat = mask.reshape(n, -1).argmax(axis=1)
        idx = np.stack(np.unravel_index(flat, lidar.shape[1:]), axis=1)
        coords[present, 3 * slot:3 * slot + 3] = idx[present] / 19.0
    return np.concatenate([pooled, coords], axis=1), missing


def _image_features(images: np.ndarray, model, device, batch_size: int) -> np.ndarray:
    """(N, 360, 640, 3) BGR 0-255 -> (N, 512) ResNet18 average-pool features."""
    import torch
    import torch.nn.functional as F

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch_size):
            rgb = images[s:s + batch_size, :, :, ::-1].astype(np.float32).copy()
            x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2) / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear",
                              align_corners=False, antialias=True)
            out.append(model((x - mean) / std).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def extract_features(zip_path: str, device: str, batch_size: int) -> dict:
    import torch
    import torchvision

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    z = zipfile.ZipFile(zip_path)
    episodes = sorted(n[: -len("/rf.npz")] for n in z.namelist() if n.endswith("/npz/rf.npz"))
    gps, image, lidar, sector, episode_id, episode_number, frame = [], [], [], [], [], [], []
    n_raw = n_zero = n_missing = 0
    for eid, ep in enumerate(episodes):
        rf = _read_npz(z, f"{ep}/rf.npz", "rf")
        keep = rf.max(axis=1) > 0
        n_raw += len(rf)
        n_zero += int((~keep).sum())
        number = int(re.search(r"episode_(\d+)", ep).group(1))

        gps.append(_read_npz(z, f"{ep}/gps.npz", "gps")[keep].astype(np.float32))
        lid, missing = _lidar_features(_read_npz(z, f"{ep}/lidar.npz", "lidar")[keep])
        lidar.append(lid)
        n_missing += missing
        image.append(_image_features(_read_npz(z, f"{ep}/image.npz", "img")[keep],
                                     model, device, batch_size))
        sector.append(rf[keep].argmax(axis=1).astype(np.int64))
        count = int(keep.sum())
        episode_id.append(np.full(count, eid, dtype=np.int64))
        episode_number.append(np.full(count, number, dtype=np.int64))
        frame.append(np.flatnonzero(keep).astype(np.int64))
        print(f"  [{eid + 1}/{len(episodes)}] {ep.split('/', 1)[1]}: kept {count}/{len(rf)}", flush=True)

    if n_raw != EXPECTED_RAW:
        raise SystemExit(f"Expected {EXPECTED_RAW} raw samples, read {n_raw}")
    print(f"Raw {n_raw}, all-zero rf rows dropped {n_zero}, missing Tx/Rx markers {n_missing}")
    return {
        "gps": np.concatenate(gps),
        "image": np.concatenate(image),
        "lidar": np.concatenate(lidar),
        "sector": np.concatenate(sector),
        "episode_id": np.concatenate(episode_id),
        "episode_number": np.concatenate(episode_number),
        "frame": np.concatenate(frame),
        "episode_names": np.array(episodes),
        "n_raw": np.array(n_raw),
        "n_zero": np.array(n_zero),
        "n_missing_markers": np.array(n_missing),
    }


def build_split(episode_id, episode_number, test_split, seed):
    """Return (train_idx, test_idx) over all kept samples."""
    if test_split == "episode":
        test_mask = episode_number == TEST_EPISODE_NUMBER
        return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    rng = np.random.RandomState(seed)
    train, test = [], []
    for eid in np.unique(episode_id):
        idx = rng.permutation(np.flatnonzero(episode_id == eid))
        cut = int(TRAIN_FRACTION * len(idx))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def main() -> None:
    p = argparse.ArgumentParser(description="FLASH processed zip -> 3-modality feature cache")
    p.add_argument("--zip_path", type=str, default="datasets/flash_processed/neu_bz61jz35p.zip")
    p.add_argument("--features_cache", type=str,
                   default="datasets/flash_processed/features_resnet18.npz",
                   help="Features before splitting; built if missing, otherwise reused")
    p.add_argument("--out_dir", type=str, default="datasets/flash_mm")
    p.add_argument("--test_split", choices=["random", "episode"], default="random")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed of the random split and of the client split")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=128, help="ResNet18 inference batch size")
    p.add_argument("--force", action="store_true", help="replace an existing cache")
    args = p.parse_args()

    stats_path = os.path.join(args.out_dir, "stats.json")
    if os.path.exists(stats_path) and not args.force:
        raise SystemExit(f"{stats_path} already exists; pass --force to replace this cache")

    if os.path.isfile(args.features_cache):
        print(f"Reusing features in {args.features_cache}")
        with np.load(args.features_cache, allow_pickle=False) as d:
            feats = {k: d[k] for k in d.files}
    else:
        feats = extract_features(args.zip_path, args.device, args.batch_size)
        # Written to a temporary name first, so an interrupted save is never reused.
        tmp = args.features_cache + ".tmp.npz"
        np.savez(tmp, **feats)
        os.replace(tmp, args.features_cache)

    # ---- Labels: the sectors that occur, in id order ----
    sector_ids, labels = np.unique(feats["sector"], return_inverse=True)
    labels = labels.astype(np.int64)
    if len(labels) != EXPECTED_KEPT or len(sector_ids) != NUM_CLASSES:
        raise SystemExit(f"Expected {EXPECTED_KEPT} samples and {NUM_CLASSES} classes, "
                         f"got {len(labels)} and {len(sector_ids)}")

    train_idx, test_idx = build_split(feats["episode_id"], feats["episode_number"],
                                      args.test_split, args.seed)
    lbl_tr, lbl_te = labels[train_idx], labels[test_idx]
    groups = feats["episode_id"]

    # ---- Client split first: nothing is written if a client comes out empty ----
    client_indices = _partition(lbl_tr, NUM_CLASSES, args.num_clients,
                                args.dirichlet_alpha, args.seed)
    sizes = [len(idx) for idx in client_indices]
    if min(sizes) == 0:
        raise SystemExit(f"Empty client with seed {args.seed}: {sizes}")

    # ---- Train-statistics z-score, as in prepare_mhealth_mm.py ----
    mods_tr, mods_te = [], []
    for key in ("gps", "image", "lidar"):
        train, test = feats[key][train_idx], feats[key][test_idx]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        train = np.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        test = np.nan_to_num((test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        mods_tr.append(train.astype(np.float32))
        mods_te.append(test.astype(np.float32))
    mod_dims = [int(m.shape[1]) for m in mods_tr]

    os.makedirs(args.out_dir, exist_ok=True)
    savez_kwargs = {
        "labels_train": lbl_tr,
        "labels_test": lbl_te,
        "groups_train": groups[train_idx],
        "groups_test": groups[test_idx],
        "mod_names": np.array(MODALITIES),
    }
    for k in range(len(MODALITIES)):
        savez_kwargs[f"m{k}_train"] = mods_tr[k]
        savez_kwargs[f"m{k}_test"] = mods_te[k]
    np.savez(os.path.join(args.out_dir, "data.npz"), **savez_kwargs)

    missing_train = np.flatnonzero(np.bincount(lbl_tr, minlength=NUM_CLASSES) == 0).tolist()
    missing_test = np.flatnonzero(np.bincount(lbl_te, minlength=NUM_CLASSES) == 0).tolist()
    stats = {
        "dataset": "FLASH",
        "source": "http://hdl.handle.net/2047/D20423471",
        "modalities": MODALITIES,
        "mod_dims": mod_dims,
        "num_classes": NUM_CLASSES,
        "num_modalities": len(MODALITIES),
        "sector_ids": sector_ids.tolist(),
        "n_raw": int(feats["n_raw"]),
        "n_zero_rf_dropped": int(feats["n_zero"]),
        "n_missing_lidar_markers": int(feats["n_missing_markers"]),
        "image_features": "ResNet18 IMAGENET1K_V1 average pool, BGR flipped to RGB, 224x224",
        "lidar_features": "obstacle density pooled 2x2x2 + Tx and Rx voxel coordinates / 19",
        "test_split": args.test_split,
        "test_episode_number": TEST_EPISODE_NUMBER if args.test_split == "episode" else None,
        "train_fraction": TRAIN_FRACTION if args.test_split == "random" else None,
        "seed": args.seed,
        "n_train": int(len(lbl_tr)),
        "n_test": int(len(lbl_te)),
        "classes_missing_from_train": missing_train,
        "classes_missing_from_test": missing_test,
    }
    with open(os.path.join(args.out_dir, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for cid, idx in enumerate(client_indices):
        idx = np.array(idx)
        payload = {f"m{k}_feats": mods_tr[k][idx] for k in range(len(MODALITIES))}
        np.savez(os.path.join(args.out_dir, f"client{cid}.npz"), labels=lbl_tr[idx], **payload)
        counts = np.bincount(lbl_tr[idx], minlength=NUM_CLASSES)
        print(f"  client {cid}: {len(idx)} samples, {int((counts > 0).sum())} classes")

    payload = {f"m{k}_feats": mods_te[k] for k in range(len(MODALITIES))}
    np.savez(os.path.join(args.out_dir, "test.npz"), labels=lbl_te, **payload)

    # stats.json last: its presence is what marks the cache as complete.
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "images_per_client": sizes,
            "num_val_images": int(len(lbl_te)),
            "multi_label": False,
            "num_classes": NUM_CLASSES,
            "num_modalities": len(MODALITIES),
            "partition_seed": args.seed,
        }, f)

    print(f"\nWrote {os.path.abspath(args.out_dir)}")
    print(f"  test_split={args.test_split}  train {stats['n_train']}  test {stats['n_test']}  "
          f"num_classes={NUM_CLASSES}  mod_dims={mod_dims}")
    print(f"  samples/client {sizes}")
    print(f"  classes missing from train {missing_train}, from test {missing_test}")


if __name__ == "__main__":
    main()
