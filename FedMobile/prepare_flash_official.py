#!/usr/bin/env python3
"""Build the directory the released FedMobile FLASH loader reads.

``FedMobile/upstream/FLASH/data_pre_flash`` loads, relative to the process
working directory:

    FLASH-data/FLASH-data/train/node_<id>/{gps,lidar,image,rf}.npz
    FLASH-data/FLASH-data/test_<group>/{gps,lidar,image,rf}.npz

with the array key equal to the modality name. FedMobile did not release that
tree. This script builds it from ``datasets/flash_processed/neu_bz61jz35p.zip``
(the zip is never extracted) and symlinks
``FedMobile/upstream/FLASH/FLASH-data`` at the output directory, so
``python MM_flash`` finds it when started in the FLASH folder.

Split. Episodes whose number is 9 are the shared test set (``test_0``, one
file for nodes 0-9, which is the path the loader computes). Every other
episode is dealt round-robin, in sorted path order, across 10 nodes. That is
the client count hardcoded in ``MM_flash``.

Tensors, matched to ``model_flash``:

    gps    (N, 2, 1)        float32, the stored values
    image  (N, 3, 112, 112) float32 in [0, 1], still BGR, resized from 360x640
    lidar  (N, 20, 20, 20)  float32, the stored categories (-2, -1, 0, 1)
    rf     (N,)             int64, argmax over the 64 rf columns

All rows are kept, including all-zero rf rows (their label is 0), so the
64-way head in ``MM_flash`` matches the file. This is not the 49-class remap
``prepare_flash_mm.py`` writes.

Usage
-----
  venv/bin/python FedMobile/prepare_flash_official.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import zipfile

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
N_NODES = 10
TEST_EPISODE_NUMBER = 9
IMAGE_SIZE = 112


def _read_npz(z: zipfile.ZipFile, name: str, key: str) -> np.ndarray:
    with np.load(io.BytesIO(z.read(name))) as data:
        return data[key]


def _episode_number(path: str) -> int:
    match = re.search(r"episode_(\d+)", path)
    if match is None:
        raise SystemExit(f"No episode number in {path}")
    return int(match.group(1))


def _resize_images(images: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) values in 0..255 -> (N, 3, 112, 112) float32 in [0, 1]."""
    import torch
    import torch.nn.functional as F

    out = np.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    step = 16
    for start in range(0, len(images), step):
        batch = np.ascontiguousarray(images[start:start + step], dtype=np.float32)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).div_(255.0)
        resized = F.interpolate(
            tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
            align_corners=False, antialias=True)
        out[start:start + step] = resized.numpy()
    return out


def _assign(episodes: list[str]) -> tuple[list[tuple[str, str, int]], dict]:
    """Return (episode, split, node) rows. Test episodes use node -1."""
    rows = []
    train_index = 0
    for episode in episodes:
        number = _episode_number(episode)
        if number == TEST_EPISODE_NUMBER:
            rows.append((episode, "test", -1))
        else:
            rows.append((episode, "train", train_index % N_NODES))
            train_index += 1
    counts = {"train_episodes": train_index, "test_episodes": len(rows) - train_index}
    return rows, counts


def _save_split(directory: str, gps, lidar, image, rf) -> None:
    os.makedirs(directory, exist_ok=True)
    np.savez(os.path.join(directory, "gps.npz"), gps=np.ascontiguousarray(gps))
    np.savez(os.path.join(directory, "lidar.npz"), lidar=np.ascontiguousarray(lidar))
    np.savez(os.path.join(directory, "image.npz"), image=np.ascontiguousarray(image))
    np.savez(os.path.join(directory, "rf.npz"), rf=np.ascontiguousarray(rf))


def _symlink(link_path: str, target: str) -> None:
    if os.path.islink(link_path) or os.path.isfile(link_path):
        os.remove(link_path)
    elif os.path.isdir(link_path):
        raise SystemExit(f"{link_path} is a real directory; refusing to replace it")
    os.symlink(target, link_path)


def prepare(zip_path: str, out_dir: str, link_path: str) -> None:
    if not os.path.isfile(zip_path):
        raise SystemExit(f"FLASH zip not found: {zip_path}")

    z = zipfile.ZipFile(zip_path)
    episodes = sorted(name[: -len("/rf.npz")] for name in z.namelist() if name.endswith("/npz/rf.npz"))
    rows, episode_counts = _assign(episodes)
    print(f"{len(episodes)} episodes, {episode_counts['train_episodes']} train, "
          f"{episode_counts['test_episodes']} test", flush=True)

    # Counts come from the small rf arrays, so the big tensors can be filled
    # into preallocated buffers instead of held as one Python list.
    lengths = []
    for episode, _split, _node in rows:
        lengths.append(len(_read_npz(z, f"{episode}/rf.npz", "rf")))
    per_node = [0] * N_NODES
    n_test = 0
    for (_episode, split, node), length in zip(rows, lengths):
        if split == "test":
            n_test += length
        else:
            per_node[node] += length

    scratch = os.path.join(out_dir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    def _buffers(name: str, n: int):
        # Memmaps keep the ~6 GB of frames on disk. The machine this runs on
        # does not have that much free RAM.
        path = os.path.join(scratch, name)
        return {
            "gps": np.memmap(path + "_gps.dat", dtype=np.float32, mode="w+", shape=(n, 2, 1)),
            "lidar": np.memmap(path + "_lidar.dat", dtype=np.float32, mode="w+", shape=(n, 20, 20, 20)),
            "image": np.memmap(path + "_image.dat", dtype=np.float32, mode="w+", shape=(n, 3, IMAGE_SIZE, IMAGE_SIZE)),
            "rf": np.memmap(path + "_rf.dat", dtype=np.int64, mode="w+", shape=(n,)),
        }

    nodes = [_buffers(f"node{i}", n) for i, n in enumerate(per_node)]
    test = _buffers("test", n_test)
    cursor = [0] * N_NODES
    test_cursor = 0
    n_raw = 0
    n_zero = 0

    for index, ((episode, split, node), length) in enumerate(zip(rows, lengths)):
        rf = _read_npz(z, f"{episode}/rf.npz", "rf")
        gps = _read_npz(z, f"{episode}/gps.npz", "gps")
        lidar = _read_npz(z, f"{episode}/lidar.npz", "lidar")
        image = _read_npz(z, f"{episode}/image.npz", "img")
        if not (len(rf) == len(gps) == len(lidar) == len(image) == length):
            raise SystemExit(f"Length mismatch in {episode}")
        n_raw += length
        n_zero += int((rf.max(axis=1) == 0).sum())

        dest = test if split == "test" else nodes[node]
        start = test_cursor if split == "test" else cursor[node]
        stop = start + length
        dest["gps"][start:stop, :, 0] = gps.astype(np.float32, copy=False)
        dest["lidar"][start:stop] = lidar.astype(np.float32, copy=False)
        dest["image"][start:stop] = _resize_images(image)
        dest["rf"][start:stop] = rf.argmax(axis=1).astype(np.int64, copy=False)
        if split == "test":
            test_cursor = stop
        else:
            cursor[node] = stop
        print(f"  [{index + 1}/{len(rows)}] node {node if node >= 0 else 'test'} "
              f"{length} frames  {episode.split('/', 1)[1]}", flush=True)
        del image, lidar, gps, rf

    for buffers in nodes + [test]:
        for array in buffers.values():
            array.flush()

    data_root = os.path.join(out_dir, "FLASH-data")
    for node, buffers in enumerate(nodes):
        directory = os.path.join(data_root, "train", f"node_{node}")
        _save_split(directory, buffers["gps"], buffers["lidar"], buffers["image"], buffers["rf"])
        print(f"wrote {directory} ({len(buffers['rf'])} frames)", flush=True)
    test_dir = os.path.join(data_root, "test_0")
    _save_split(test_dir, test["gps"], test["lidar"], test["image"], test["rf"])
    print(f"wrote {test_dir} ({len(test['rf'])} frames)", flush=True)

    # Drop the memmap objects before unlinking. On NFS an open file is
    # renamed to .nfs* instead of disappearing, and rmdir then fails.
    nodes.clear()
    del test
    shutil.rmtree(scratch)

    stats = {
        "source": zip_path,
        "n_episodes": len(episodes),
        "n_raw": n_raw,
        "n_zero_rf": n_zero,
        "n_nodes": N_NODES,
        "test_episode_number": TEST_EPISODE_NUMBER,
        "train_episodes": episode_counts["train_episodes"],
        "test_episodes": episode_counts["test_episodes"],
        "train_frames": per_node,
        "test_frames": n_test,
        "num_classes": 64,
        "image": "BGR, bilinear resize to 112x112, float32 / 255",
        "gps": "(N, 2, 1) raw float32",
        "lidar": "(N, 20, 20, 20) category float32",
        "label": "argmax over 64 rf columns, all-zero rows kept as class 0",
    }
    with open(os.path.join(out_dir, "prepare_stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")

    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    _symlink(link_path, os.path.abspath(out_dir))
    print(f"symlink {link_path} -> {os.path.abspath(out_dir)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--zip", default=os.path.join(ROOT, "datasets", "flash_processed", "neu_bz61jz35p.zip"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "datasets", "fedmobile_flash_official"))
    parser.add_argument("--link", default=os.path.join(HERE, "upstream", "FLASH", "FLASH-data"))
    args = parser.parse_args()
    prepare(args.zip, args.out_dir, args.link)


if __name__ == "__main__":
    main()
