import math

import numpy as np
import torch

from args import args


def prune_fedcs(client_loaders, global_model, clients=None, budgets=None):
    if clients is None:
        clients = np.arange(len(client_loaders))

    num_arr = np.array([client_loaders[client].dataset.get_original_length() for client in clients])

    if budgets is None:
        budgets = np.ceil(num_arr * (1 - args.prune_frac)).astype(int)

    # line 3-9, cal class center for each client
    global_model.eval()
    cs_clients_center = {cls: [] for cls in range(args.num_classes)}
    clients_cs_reps = {}
    clients_cs_rep_indices = {}
    for client in clients:
        cs_client_rep_list = {cls: [] for cls in range(args.num_classes)}
        cs_client_rep_tensor = {cls: torch.empty(0) for cls in range(args.num_classes)}
        cs_client_rep_indices = {cls: [] for cls in range(args.num_classes)}
        with torch.no_grad():
            for batch_x, batch_y, batch_index, _ in client_loaders[client]:
                batch_x = batch_x.to(args.device)
                backbone = global_model.get_backbone(batch_x).detach().cpu()
                batch_y = batch_y.cpu()
                batch_index = batch_index.cpu()
                for i in range(len(batch_y)):
                    label = batch_y[i].item()
                    cs_client_rep_list[label].append(backbone[i].detach().cpu())
                    cs_client_rep_indices[label].append(batch_index[i].item())
        for cls, client_reps in cs_client_rep_list.items():
            if len(client_reps) > 0:
                stacked_reps = torch.stack(client_reps)
                cs_clients_center[cls].append(stacked_reps.mean(dim=0))
                # Update dict with tensor instead of list of tensors
                cs_client_rep_tensor[cls] = stacked_reps
            else:
                cs_client_rep_tensor[cls] = torch.empty(0)
        clients_cs_reps[client] = cs_client_rep_tensor
        clients_cs_rep_indices[client] = cs_client_rep_indices
    # line 10-12, cal global class center
    # tensor (k, dim)
    cs_global_center = torch.stack([torch.stack(clients_center).median(dim=0).values
                                    for clients_center in cs_clients_center.values()])
    for client in clients:
        all_prune_data_indices = []
        # indentify larger class
        cls_counts = np.array([len(clients_cs_reps[client][cls]) for cls in range(args.num_classes)])
        max_counts = np.max(cls_counts)
        # larger_classes = set(np.where(cls_counts > max_counts * 0.65)[0])
        larger_classes = set(np.where(cls_counts > max_counts * 0.65)[0])
        all_cls_masks = {}
        all_cls_scores = {}
        for label in range(args.num_classes):
            reps = clients_cs_reps[client][label]  # tensor (n, dim)
            if len(reps) == 0:
                continue
            # (n, k), distance to k class centers of n samples
            dists = torch.cdist(reps, cs_global_center, p=2)
            # d_y: Distance to correct class (column = label)
            dist_y = dists[:, label]
            mask = torch.ones_like(dists)
            mask[:, label] = float('inf')
            dist_min, _ = torch.min(dists * mask, dim=1)
            scores = torch.abs(dist_y - dist_min).numpy()

            indices = np.array(clients_cs_rep_indices[client][label])  # array (n)
            # employ a boolean mask to track which is kept, false means pruned
            keep_mask = np.ones(len(indices), dtype=bool)

            # prune for larger classes
            current_indices = np.where(keep_mask)[0]
            if label in larger_classes:
                current_scores = scores[current_indices]
                num_pruned = math.floor(len(current_scores) * 0.1)
                if num_pruned > 0:
                    prune_indices = current_indices[np.argsort(current_scores)[-num_pruned:]]
                    keep_mask[prune_indices] = False
                    all_prune_data_indices.extend(indices[prune_indices])

            all_cls_masks[label] = keep_mask
            all_cls_scores[label] = scores

        labels = all_cls_masks.keys()
        sorted_labels = sorted(labels, key=lambda x: len(clients_cs_reps[client][x]))
        all_prune_num = (client_loaders[client].dataset.get_original_length()
                         - len(all_prune_data_indices) - budgets[client])
        # prune for all classes
        for rank, label in enumerate(sorted_labels):
            keep_mask = all_cls_masks[label]
            scores = all_cls_scores[label]
            current_indices = np.where(keep_mask)[0]
            current_scores = scores[current_indices]
            # num_pruned = math.floor(len(current_scores) * all_prune_frac)
            num_pruned = min(len(current_scores) - 1, all_prune_num // (len(labels) - rank))
            if num_pruned > 0:
                all_prune_num -= num_pruned
                indices = np.array(clients_cs_rep_indices[client][label])
                prune_indices = current_indices[np.argsort(current_scores)[-num_pruned:]]
                keep_mask[prune_indices] = False
                all_prune_data_indices.extend(indices[prune_indices])

        client_loaders[client].dataset.prune_by_global_indices(all_prune_data_indices)
