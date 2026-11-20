import math

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from args import args
from train.client import train_client_gcfl


def OrthogonalMP_REG_Parallel_V1_ED(A, b, budget=None, positive=False, lam=1.0, ):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      budget = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    tol = 1E-4
    device = args.device
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape

    if budget is None:
        budget = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []

    chosen = torch.zeros(AT.shape[0], device=device).bool()
    combine = 5
    if budget < 15:
        combine = 1

    for i in range(0, budget, combine):
        if resid.norm().item() / normb < tol:
            break

        # print(i,torch.isnan(AT).sum(),torch.isnan(resid).sum())
        projections = (torch.pow(AT - resid, 2).sum(1))

        if (~chosen.cpu()).sum() > combine:
            if np.random.rand() < 0.5:  # *n/budget :
                p = (projections[~chosen].max() - projections[~chosen] + 1e-14) / (
                    (projections[~chosen].max() + 1e-14 - projections[~chosen]).sum())
                tm_index = np.random.choice((~chosen.cpu()).sum(), combine, replace=False, p=p.detach().cpu().numpy())
            else:
                tm_index = torch.topk(projections[~chosen], combine, largest=False)[1]
        else:
            tm_index = torch.arange((~chosen.cpu()).sum())

        index = torch.arange(AT.shape[0], device=device)[~chosen][tm_index]
        chosen[index] = True
        indices.extend(index)

        current_k = len(index)

        if len(indices) == combine:
            A_i = A[:, index]
            # A_i = A[:, index].view(combine, -1)
            A_i = A[:, index].view(current_k, -1)
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i = torch.linalg.lstsq(torch.matmul(A_i, b).view(-1, 1), temp).solution.view(-1)
        else:
            # A_i = torch.cat((A_i, A[:, index].view(combine, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            A_i = torch.cat((A_i, A[:, index].view(current_k, -1)), dim=0)
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i = torch.linalg.lstsq(torch.matmul(A_i, b).view(-1, 1), temp).solution.view(-1)
            if positive:
                while min(x_i) < 0.0:

                    if len(x_i) == 1:
                        indices = []
                        break

                    argmin = torch.argmin(x_i)
                    chosen[indices[argmin]] = False
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                    x_i = x_i.view(-1) / (A_i.norm(dim=1))
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i.view(-1, 1)).view(-1)

    diff = budget - len(indices)

    # print("Points selected", len(indices),len(projections[chosen]))
    if diff > 0:
        # sorted_proj = (torch.abs(projections[~chosen])).sort(descending=True)
        sorted_proj = (projections[~chosen]).sort(descending=False)  # True)
        ind = (sorted_proj.indices[:diff]).tolist()
        indices.extend((torch.arange(AT.shape[0])[~chosen][ind]).tolist())
        # x_i = torch.cat((x_i, torch.mean(torch.abs(x_i)) * torch.ones(diff, device=device)))
        x_i = torch.cat((x_i.view(-1), torch.min(x_i) * torch.ones(diff, device=device)))

    for i, sel_idx in enumerate(indices):
        x[sel_idx] = x_i[i]

    return indices, x


def prune_gcfl(client_loader, global_model, validate_loader, budget):
    client_loader.dataset.recover_to_origin()
    trn_lbls = torch.zeros(len(client_loader.sampler)).to(dtype=int)
    for batch_idx, (inputs, targets, _, data_ids) in enumerate(client_loader):
        trn_lbls[data_ids] = targets
    val_lbls = torch.zeros(len(validate_loader.sampler)).to(dtype=int)
    for batch_idx, (inputs, targets, _, data_ids) in enumerate(validate_loader):
        val_lbls[data_ids] = targets
    idxes = []
    embDim = global_model.get_backbone_dim()
    for cls in range(args.num_classes):
        trn_subset_idx = torch.where(trn_lbls == cls)[0].tolist()

        if len(trn_subset_idx) < 3:
            continue

        trn_data_sub = Subset(client_loader.dataset, trn_subset_idx)
        cs_client_loader = DataLoader(trn_data_sub, batch_size=args.batch_size,
                                      shuffle=False, pin_memory=True)
        val_subset_idx = torch.where(val_lbls == cls)[0].tolist()
        val_data_sub = Subset(validate_loader.dataset, val_subset_idx)
        cs_validate_loader = DataLoader(val_data_sub, batch_size=args.test_batchsize,
                                        shuffle=False, pin_memory=True)

        l0_grads, l1_grads = train_client_gcfl(cs_client_loader, global_model, embDim)
        grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        l0_grads, l1_grads = train_client_gcfl(cs_validate_loader, global_model, embDim)
        val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        torch.cuda.empty_cache()

        trn_gradients = grads_per_elem
        tmp_gradients = trn_gradients[:, cls].view(-1, 1)
        tmp1_gradients = trn_gradients[:,
        args.num_classes + (embDim * cls): args.num_classes + (embDim * (cls + 1))]
        trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
        val_gradients = val_grads_per_elem
        tmp_gradients = val_gradients[:, cls].view(-1, 1)
        tmp1_gradients = val_gradients[:,
        args.num_classes + (embDim * cls): args.num_classes + (embDim * (cls + 1))]
        val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)

        sum_val_grad = torch.mean(val_gradients, dim=0)

        selected_idxs, omp_weights = OrthogonalMP_REG_Parallel_V1_ED(
            torch.transpose(trn_gradients, 0, 1),
            sum_val_grad, math.ceil(budget * len(trn_subset_idx) / len(client_loader.sampler)),
            positive=False, lam=0.5,
        )
        idxs_temp = torch.LongTensor(selected_idxs).tolist()
        idxes.extend(list(np.array(trn_subset_idx)[idxs_temp]))

    diff = budget - len(idxes)
    if diff > 0:
        remainList = set(np.arange(len(client_loader.sampler))).difference(set(idxes))
        new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
        idxes.extend(new_idxs)
        idxes = np.array(idxes)
    rand_indices = np.random.permutation(len(idxes))
    idxes = list(np.array(idxes)[rand_indices])
    if len(idxes) == 0:
        idxes = range(len(client_loader.sampler))

    return idxes
