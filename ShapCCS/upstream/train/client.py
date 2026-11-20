import math

import numpy as np
import torch
from torch import nn

from args import args

criterion = nn.CrossEntropyLoss()


def copy_global_model(global_model):
    global_model_copy = global_model.__class__()
    global_model_copy.load_state_dict(global_model.state_dict())
    global_model_copy.to(args.device)
    return global_model_copy


def get_lr(current_round):
    if args.dataset in ['mnist', 'cifar10', 'fmnist', 'cinic10']:
        return pow(0.99, current_round - 1) * args.lr
    elif args.dataset in ['cifar100']:
        cos_inner = math.pi * current_round / args.rounds
        return 0.5 * args.lr * (1 + math.cos(cos_inner))
    else:
        raise 'lr decay strategy is not set'


def train_client(client_loader, global_model, cur_round):
    # global model is on gpu
    client_model = copy_global_model(global_model)
    client_model.train()
    lr = get_lr(cur_round)
    optimizer = torch.optim.SGD(client_model.parameters(), lr)
    total = 0
    for epoch in range(args.num_epochs):
        for x, y, _, _ in client_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += x.shape[0]
    local_state = client_model.state_dict()

    # free cache
    del client_model
    del optimizer

    return local_state


def train_client_weight(client_loader, global_model, cur_round, cs_weight):
    client_model = copy_global_model(global_model)
    client_model.train()
    lr = get_lr(cur_round)
    optimizer = torch.optim.SGD(client_model.parameters(), lr)
    total = 0
    for epoch in range(args.num_epochs):
        for x, y, _, idx in client_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            idx = idx.to(args.device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss = (loss * cs_weight[idx.long()]).mean()
            loss.backward()
            optimizer.step()
            total += x.shape[0]
    local_state = client_model.state_dict()

    # free cache
    del client_model
    del optimizer

    return local_state


def train_client_gcfl(cs_client_loader, global_model, embDim):
    client_model = copy_global_model(global_model)
    for batch_idx, (inputs, targets, _, _) in enumerate(cs_client_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device, non_blocking=True)
        if batch_idx == 0:
            out, l1 = client_model.backbone_frozen_forward(inputs)
            loss = criterion(out, targets).sum()
            l0_grads = torch.autograd.grad(loss, out)[0]
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, args.num_classes)
        else:
            out, l1 = client_model.backbone_frozen_forward(inputs)
            loss = criterion(out, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, out)[0]
            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
            batch_l1_grads = batch_l0_expand * l1.repeat(1, args.num_classes)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

    del client_model

    return l0_grads, l1_grads
