import numpy as np
import torch

from args import args


def get_median(features, targets):
    num_classes = args.num_classes
    feature_dim = features.shape[-1]
    prot = np.zeros((num_classes, feature_dim), dtype=features.dtype)

    for i in range(num_classes):
        mask = (targets == i)
        if np.any(mask):
            prot[i] = np.median(features[mask], axis=0)
        else:
            prot[i] = np.zeros(feature_dim, dtype=features.dtype)
    return prot


def get_distance(features, labels):
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)

    return distance


def get_features(client_loader, global_model):
    targets, features = [], []

    global_model.eval()
    with torch.no_grad():
        for x, y, global_idx, _ in client_loader:
            x = x.to(args.device)
            feature = global_model.get_backbone(x).detach().cpu().numpy()
            features.extend([feature[i] for i in range(feature.shape[0])])
            targets.extend(y.numpy().tolist())

    features = np.array(features)
    targets = np.array(targets)

    return features, targets


def prune_moderate(client_loader, global_model, budget=None):
    features, targets = get_features(client_loader, global_model)
    distance = get_distance(features, targets)

    if budget is None:
        rate = 1 - args.prune_frac
    else:
        rate = budget / client_loader.dataset.get_original_length()

    low = 0.5 - rate / 2
    high = 0.5 + rate / 2

    sorted_idx = distance.argsort()
    low_idx = round(distance.shape[0] * low)
    high_idx = round(distance.shape[0] * high)

    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    ids = ids[:client_loader.dataset.get_original_length() - 1]

    return ids
