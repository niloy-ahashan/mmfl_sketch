#!/usr/bin/env bash
# Run the released FedMobile FLASH entry point (FedMobile/upstream/FLASH/MM_flash)
# without modifying it.
#
# The README's `python MM_flash.py` does not match the tree: the file has no
# .py suffix, its sibling modules have none either, and sampling_noniid is
# imported but not shipped in this folder. This wrapper loads those modules,
# points the process at the node files from prepare_flash_official.py, and
# tees stdout to FedMobile/result_updated/.
#
# Two crash guards, both outside the released file. kmeans-pytorch squeezes a
# one-point cluster into a 0-d index and then fails index_select, and an empty
# cluster makes gen_agg fail; the wrapper keeps the index 1-d and, only when a
# cluster is empty, moves one point into it from the largest cluster.
# data_pre_flash builds every tensor with .tolist(), which would hold the
# 112x112 frames as nested Python lists once per client (the test set ten
# times). The wrapper returns the same arrays via torch.tensor.
#
# Defaults stay the released ones (100 rounds, Adam 1e-3, batch 16, distill
# on, GPS missing on 80% of each batch). --cuda_device 0 is passed because the
# script's own default is device 1. Do not pass --distill: argparse declares
# it as type=bool, so any string on the command line is True.
#
# Extra args are forwarded, for example:
#   bash FedMobile/run_flash_official.sh --epochs 1
#
# The 100-round run is long. Start it in tmux.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${ROOT}/venv/bin/python"
FLASH="${HERE}/upstream/FLASH"
DATA_MARK="${ROOT}/datasets/fedmobile_flash_official/FLASH-data/test_0/rf.npz"
LOG_DIR="${HERE}/result_updated"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "root venv python not found at ${PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON}" -c "import kmeans_pytorch" >/dev/null 2>&1; then
  "${PYTHON}" -m pip install kmeans-pytorch
fi

if [[ ! -f "${DATA_MARK}" ]]; then
  "${PYTHON}" "${HERE}/prepare_flash_official.py"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/FedMobileOfficial_FLASH_${STAMP}.txt"
echo "Logging to ${LOG}"

cd "${FLASH}"
"${PYTHON}" -u - --cuda_device 0 "$@" << 'PY' 2>&1 | tee "${LOG}"
import importlib.abc
import importlib.util
import os
import runpy
import sys
import types

flash_dir = os.getcwd()


class _ExtensionlessLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.path) as handle:
            code = compile(handle.read(), self.path, "exec")
        exec(code, module.__dict__)


class _ExtensionlessFinder:
    """Import util, model_flash and data_pre_flash, which have no .py suffix."""

    def find_spec(self, fullname, path, target=None):
        if path is not None or "." in fullname:
            return None
        candidate = os.path.join(flash_dir, fullname)
        if os.path.isfile(candidate) and os.path.splitext(candidate)[1] == "":
            return importlib.util.spec_from_loader(fullname, _ExtensionlessLoader(candidate))
        return None


sys.meta_path.insert(0, _ExtensionlessFinder())
# Imported by MM_flash and never called. The CMAHD copy imports a private numpy module.
sys.modules["sampling_noniid"] = types.ModuleType("sampling_noniid")

import numpy as np
import torch

_data = types.ModuleType("data_pre_flash")


def _as_tensor(array, dtype):
    # Share the cached array's storage. torch.tensor would copy the test set
    # once per client, and ten copies of the 112x112 frames do not fit in RAM.
    np_dtype = np.float32 if dtype is torch.float32 else np.int64
    if array.dtype != np_dtype:
        array = np.ascontiguousarray(array, dtype=np_dtype)
    return torch.from_numpy(array)


class Multimodal_dataset:
    def __init__(self, x1, x2, x3, y):
        self.data1 = _as_tensor(x1, torch.float32)
        self.data2 = _as_tensor(x2, torch.float32)
        self.data3 = _as_tensor(x3, torch.float32)
        self.labels = _as_tensor(y, torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx], self.labels[idx]


class Unimodal_dataset:
    def __init__(self, x, y):
        self.data = _as_tensor(x, torch.float32)
        self.labels = _as_tensor(y, torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


_loaded = {}


def load_single_modality_from_one_node(modality, node_id, test_flag):
    node_path = "FLASH-data/FLASH-data/"
    if test_flag == 0:
        temp_path = node_path + "train/node_" + str(node_id) + "/" + modality + ".npz"
    else:
        group_id = int(int(node_id / 10) * 10)
        temp_path = node_path + "test_{}/".format(group_id) + modality + ".npz"
    cached = _loaded.get(temp_path)
    if cached is None:
        cached = np.ascontiguousarray(np.load(temp_path)[modality])
        _loaded[temp_path] = cached
    return cached


def load_data(modality, node_id, test_flag):
    x = load_single_modality_from_one_node(modality, node_id, test_flag)
    y = load_single_modality_from_one_node("rf", node_id, test_flag)
    return x, y


_data.Multimodal_dataset = Multimodal_dataset
_data.Unimodal_dataset = Unimodal_dataset
_data.load_data = load_data
_data.load_single_modality_from_one_node = load_single_modality_from_one_node
sys.modules["data_pre_flash"] = _data

import kmeans_pytorch
from kmeans_pytorch import pairwise_cosine, pairwise_distance


def _kmeans_no_empty(X, num_clusters, distance="euclidean", tol=1e-4,
                     device=torch.device("cpu"), **_ignored):
    """k-means that survives a one-point cluster and fills an empty one.

    The installed kmeans-pytorch squeezes a single index down to 0-d, which
    index_select rejects, and an empty cluster makes MM_flash's gen_agg fail.
    A point moves only when a cluster would otherwise be empty.
    """
    if distance == "euclidean":
        dist_fn = pairwise_distance
    elif distance == "cosine":
        dist_fn = pairwise_cosine
    else:
        raise NotImplementedError(distance)
    points = X.float().to(device)
    n_points = points.shape[0]
    choice = torch.randperm(n_points, device=device)[:num_clusters]
    centers = points[choice].clone()
    ids = torch.zeros(n_points, dtype=torch.long, device=device)
    for _ in range(100):
        ids = torch.argmin(dist_fn(points, centers, device=device), dim=1)
        shift = 0.0
        for index in range(num_clusters):
            selected = torch.nonzero(ids == index, as_tuple=False).view(-1)
            if selected.numel() == 0:
                continue
            mean = points[selected].mean(dim=0)
            shift += torch.sum((centers[index] - mean) ** 2).item()
            centers[index] = mean
        if shift < tol:
            break
    ids = ids.cpu()
    centers = centers.cpu()
    counts = {i: int((ids == i).sum()) for i in range(num_clusters)}
    for cluster in [i for i, count in counts.items() if count == 0]:
        donors = [i for i, count in counts.items() if count > 1]
        if not donors:
            break
        donor = max(donors, key=counts.get)
        moved = torch.nonzero(ids == donor, as_tuple=False).view(-1)[0]
        ids[moved] = cluster
        counts[donor] -= 1
        counts[cluster] = 1
        centers[cluster] = points[moved].detach().cpu()
    return ids, centers


kmeans_pytorch.kmeans = _kmeans_no_empty

# python - puts "-" in argv[0]. The script's parser should see the flags only.
sys.argv = ["MM_flash"] + sys.argv[1:]
runpy.run_path(os.path.join(flash_dir, "MM_flash"), run_name="__main__")
PY
