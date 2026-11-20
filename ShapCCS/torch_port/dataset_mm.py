"""Shared UCI HAR / mHealth split wrapped in upstream ShapCCS's data interface.

Upstream (`ShapCCS/upstream/data/data_loder.py`) gives every client a ``DataLoader`` over a
``PrunableDataset`` that indexes one pooled torchvision training set and yields
``(x, y, global_idx, local_idx)``. The selection code relies on that interface:
``len(client_loaders[0].dataset.data)`` is the pooled training-set size, and pruning edits each
client's ``active_global_indices``.

Here the pooled set is the concatenation of the clients from HybridSketchMFedMC_v2's loaders (the
same 10-client Dirichlet(0.1) split, cached ``client*.npz``, that the other baselines use), and ``x``
is a tuple with one tensor per modality. ``PrunableDataset`` is upstream's class, copied unchanged.
There is no validation hold-out: every client sample is training data (shared protocol; upstream keeps
5% aside, which the Moderate path never reads).
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "HybridSketchMFedMC_v2") not in sys.path:
    sys.path.append(os.path.join(ROOT, "HybridSketchMFedMC_v2"))

from dataset import load_mhealth_mm, load_uci_har_mm  # HybridSketchMFedMC_v2


class MMTensorData(Dataset):
    """Pooled multimodal training set: item i -> (tuple of per-modality tensors, label)."""

    def __init__(self, xs, y):
        self.xs = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in xs]
        self.targets = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return tuple(x[i] for x in self.xs), int(self.targets[i])


# --- copied unchanged from ShapCCS/upstream/data/data_loder.py ---
class PrunableDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.data = original_dataset
        self.active_global_indices = list(indices)
        self.original_global_indices = list(indices)

    def __len__(self):
        return len(self.active_global_indices)

    def __getitem__(self, i):
        global_idx = self.active_global_indices[i]
        x, y = self.data[global_idx]
        return x, y, global_idx, i

    def prune_by_global_indices(self, global_indices_to_remove):
        self.active_global_indices = [
            idx for idx in self.active_global_indices
            if idx not in global_indices_to_remove
        ]

    def remain_by_global_indices(self, global_indices_to_remain):
        self.active_global_indices = [
            idx for idx in self.active_global_indices
            if idx in global_indices_to_remain
        ]

    def prune_by_local_indices(self, local_indices_to_remove):
        self.active_global_indices = [
            idx for i, idx in enumerate(self.active_global_indices)
            if i not in local_indices_to_remove
        ]

    def remain_by_local_indices(self, local_indices_to_remain):
        self.active_global_indices = [
            idx for i, idx in enumerate(self.active_global_indices)
            if i in local_indices_to_remain
        ]

    def get_current_length(self):
        return len(self.active_global_indices)

    def get_original_length(self):
        return len(self.original_global_indices)

    def recover_to_origin(self):
        self.active_global_indices = list(self.original_global_indices)
# --- end of upstream copy ---


def load_shared_split(dataset_dir, dataset="auto", num_clients=10, dirichlet_alpha=0.1, seed=42):
    """Hybrid v2's loader for the dataset in ``dataset_dir``: (clients, test, meta)."""
    kind = dataset
    if kind == "auto":
        kind = "mhealth" if "mhealth" in os.path.basename(os.path.normpath(dataset_dir)) else "uci_har"
    loader = load_mhealth_mm if kind == "mhealth" else load_uci_har_mm
    return loader(dataset_dir, num_clients=num_clients, dirichlet_alpha=dirichlet_alpha, seed=seed)


def build_client_loaders(clients, batch_size, rng=np.random):
    """Pool the clients' data and give each client a shuffled DataLoader over its slice.

    As upstream's partitioner, a client with no data gets one random sample of the pooled set.
    Returns (client_loaders, pooled_dataset).
    """
    xs = [np.concatenate([c["xs"][m] for c in clients], axis=0) for m in range(len(clients[0]["xs"]))]
    y = np.concatenate([c["y"] for c in clients], axis=0)
    pooled = MMTensorData(xs, y)
    loaders, start = [], 0
    for cid, c in enumerate(clients):
        n = len(c["y"])
        idx = list(range(start, start + n))
        start += n
        if not idx:
            idx.append(int(rng.choice(len(pooled), 1)[0]))
            print(f"client {cid} does not have data, assigned 1 random sample.")
        loaders.append(DataLoader(PrunableDataset(pooled, idx), batch_size=batch_size, shuffle=True))
    return loaders, pooled
