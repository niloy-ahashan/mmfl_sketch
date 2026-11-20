import math
import os
from decimal import Decimal, getcontext

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from args import args


def write_iterable(file_path, iters, model='a', print_file_path=True):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, model) as f:
        for i, elem in enumerate(iters):
            f.write(f"{elem}, ")
            if (i + 1) % 5 == 0: f.write('\n')
        f.write('\n')
    if print_file_path:
        print(f'written to {file_path}')


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


def add_client_noise(data, indices):
    targets = data.targets
    noisy_indices = np.random.choice(indices, int(len(indices) * args.noise_ratio), replace=False)
    # noisy_indices = indices[-math.floor(len(indices) * args.noise_ratio):]
    for idx in noisy_indices:
        new_label = np.random.choice([x for x in range(args.num_classes) if x != targets[idx]])
        targets[idx] = new_label


def iid_partition_loader(data, batch_size):
    """
    partition the dataset into a dataloader for each client, iid style
    """
    client_num_total = int(len(data) * args.train_ratio)

    raws = np.random.lognormal(mean=0, sigma=0.75, size=args.num_clients)
    getcontext().prec = 50
    decimal_raws = [Decimal(str(raw)) for raw in raws]
    total_raw = sum(decimal_raws)
    client_nums = [math.floor(raw / total_raw * client_num_total) for raw in decimal_raws]
    client_nums[-1] += client_num_total - sum(client_nums)
    all_indices = torch.randperm(len(data))
    client_indices = all_indices[:client_num_total]
    splits = torch.split(client_indices, client_nums)
    clients_indices = [split.tolist() for split in splits]

    num_noisy_clients = math.floor(args.num_clients * args.noisy_client_ratio)
    for i in range(num_noisy_clients):
        add_client_noise(data, clients_indices[i])

    clients_data = [PrunableDataset(data, indices) for indices in clients_indices]

    client_loaders = [
        DataLoader(x, batch_size=batch_size, shuffle=True)
        for x in clients_data
    ]

    val_indices = all_indices[client_num_total:]
    val_data = PrunableDataset(data, val_indices.tolist())
    val_loader = DataLoader(
        val_data,
        batch_size=args.test_batchsize,
        shuffle=False
    )

    return client_loaders, val_loader


def get_primary_classes(client):
    cluster = client % args.num_clusters
    avg = args.num_classes // args.num_clusters
    extra = args.num_classes % args.num_clusters

    start = cluster * avg + min(cluster, extra)
    end = start + avg + (1 if cluster < extra else 0)
    return list(range(start, end))


def niid_partition_loader_dirichlet(data, batch_size):
    class_indices = {i: [] for i in range(args.num_classes)}
    for idx, label in enumerate(data.targets):
        class_indices[int(label)].append(idx)

    alpha = args.dirichlet_alpha

    class_proportions = np.random.dirichlet([alpha] * args.num_clients, size=args.num_classes)

    clients_indices = [[] for _ in range(args.num_clients)]

    for cls in range(args.num_classes):
        indices = class_indices[cls]
        num_cls_samples = int(len(indices) * args.train_ratio)

        cls_clients_samples = (class_proportions[cls] * num_cls_samples).astype(int)

        cls_clients_samples[-1] += num_cls_samples - np.sum(cls_clients_samples)

        np.random.shuffle(indices)

        current_idx = 0
        for client in range(args.num_clients):
            num_samples = cls_clients_samples[client]

            if num_samples > 0:
                selected_indices = indices[current_idx: current_idx + num_samples]
                clients_indices[client].extend(selected_indices)
                current_idx += num_samples

        class_indices[cls] = indices[current_idx:]  # 仅保留未分配的索引

    client_loaders = []

    num_noisy_clients = math.floor(args.num_clients * args.noisy_client_ratio)

    for i, client in enumerate(range(args.num_clients)):
        selected_indices = clients_indices[client]

        if len(selected_indices) == 0:
            selected_indices.append(np.random.choice(len(data), 1)[0])
            print(f'client {client} does not have data, assigned 1 random sample.')

        if i < num_noisy_clients:
            add_client_noise(data, selected_indices)

        client_data = PrunableDataset(data, selected_indices)
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)

        if args.record_data:
            file_path = f'results_seed_{args.seed}/data/{args.dataset}_{args.dirichlet_alpha}_dis.txt'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w' if client == 0 else 'a') as f:
                labels = [data.targets[i] for i in selected_indices]
                f.write(f"client:{client} alpha:{alpha}: \n")
                labels_count = [labels.count(i) for i in range(args.num_classes)]
                f.write(f"Total samples: {len(selected_indices)}\n")
                f.write("Class distribution: ")
                f.write(f"{labels_count}\n")

    total = sum(len(loader.dataset) for loader in client_loaders)
    args.ratios = [len(loader.dataset) / total for loader in client_loaders]

    val_indices = []
    for cls in range(args.num_classes):
        val_indices.extend(class_indices[cls])

    val_data = PrunableDataset(data, val_indices)
    val_loader = DataLoader(val_data, batch_size=args.test_batchsize, shuffle=False)

    return client_loaders, val_loader


def partition_client_data(data, batch_size):
    if math.isinf(args.dirichlet_alpha):
        return iid_partition_loader(data, batch_size)
    else:
        return niid_partition_loader_dirichlet(data, batch_size)
