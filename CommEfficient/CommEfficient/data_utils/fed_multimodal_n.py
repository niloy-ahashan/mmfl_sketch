"""
Federated dataset with any number of modality vectors per sample.

Reads the cache a ``prepare_*`` script already wrote: ``client{i}.npz`` and
``test.npz`` with ``m0_feats`` … ``m{N-1}_feats`` + ``labels``, plus ``stats.json``
with ``num_modalities``. Unlike :class:`FedMultiModal` it never builds the client
split itself, so a missing cache is an error instead of a new split.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from data_utils.fed_dataset import FedDataset

__all__ = ["FedMultiModalN"]


class FedMultiModalN(FedDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.type == "train":
            self.client_m = [[] for _ in range(self.num_modalities)]
            self.client_labels = []
            for cid in range(len(self.images_per_client)):
                data = np.load(self.client_fn(cid))
                for k in range(self.num_modalities):
                    self.client_m[k].append(data[f"m{k}_feats"])
                self.client_labels.append(data["labels"])
        else:
            data = np.load(self.test_fn())
            self.test_m = [data[f"m{k}_feats"] for k in range(self.num_modalities)]
            self.test_labels = data["labels"]

    def _load_meta(self, train):
        super()._load_meta(train)
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
        self.multi_label = stats.get("multi_label", False)
        self.mm_num_classes = stats.get("num_classes", None)
        self.num_modalities = stats["num_modalities"]

    def prepare_datasets(self, download=False):
        raise FileNotFoundError(
            f"No client cache in {self.dataset_dir}: run its prepare script first."
        )

    def __getitem__(self, idx):
        if self.type == "train":
            orig_idx = idx
            if self.do_iid:
                idx = self.iid_shuffle[idx]

            cumsum = np.cumsum(self.images_per_client)
            cid_store = int(np.searchsorted(cumsum, idx, side="right"))
            offsets = np.hstack([[0], cumsum[:-1]])
            idx_within_client = idx - offsets[cid_store]

            mods = tuple(
                torch.as_tensor(self.client_m[k][cid_store][idx_within_client],
                                dtype=torch.float32)
                for k in range(self.num_modalities)
            )
            target = int(self.client_labels[cid_store][idx_within_client])

            cumsum_dpc = np.cumsum(self.data_per_client)
            client_id = int(np.searchsorted(cumsum_dpc, orig_idx,
                                            side="right"))
        else:
            mods = tuple(
                torch.as_tensor(self.test_m[k][idx], dtype=torch.float32)
                for k in range(self.num_modalities)
            )
            target = int(self.test_labels[idx])
            client_id = -1

        return (client_id, *mods, target)

    def client_fn(self, client_id):
        return os.path.join(self.dataset_dir, f"client{client_id}.npz")

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")
