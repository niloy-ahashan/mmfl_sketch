import numpy as np
import torch
from torch.nn import functional as F

from args import args


def cal_el2n(client_loader, global_model):
    scores = []
    global_idxes = []
    global_model.eval()
    with torch.no_grad():
        for x, y, global_idx, _ in client_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = global_model(x)
            probs = F.softmax(logits, dim=-1)
            y_hot = F.one_hot(y, num_classes=args.num_classes).float()
            error = probs - y_hot
            scores.append(error.cpu().numpy())
            global_idxes.append(global_idx)

    scores = np.concatenate(scores, axis=0)
    scores = np.linalg.norm(scores, axis=1)

    global_idxes = np.concatenate(global_idxes, axis=0)

    return scores, global_idxes
