import math

import numpy as np
import torch
from tqdm import tqdm

from args import args
from coreset_selection.utils.ccs_utils import mislabel_mask, stratified_sampling
from coreset_selection.utils.craig_utils import prune_craig
from coreset_selection.utils.el2n_utils import cal_el2n
from coreset_selection.utils.moderate_utils import prune_moderate
from train.client import train_client_weight, train_client
from train.server import get_weights, weight_aggregate, validate_model


def fed_craig(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    clients_cs_weight = {client: torch.ones(len(client_loaders[client].dataset), device=args.device)
                         for client in clients}
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round > 0 and cur_round % args.pre_train_round == 0:
            for client in clients:
                client_loader = client_loaders[client]
                client_dataset = client_loader.dataset
                client_dataset.recover_to_origin()

                B = max(math.floor((1 - args.prune_frac) * client_dataset.get_original_length()), 1)
                subset_indices, cs_weight = prune_craig(global_model, client_loader, B)
                subset_indices = subset_indices[:B]
                client_dataset.remain_by_local_indices(subset_indices)
                clients_cs_weight[client] = cs_weight

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for cls, client in enumerate(selected_clients):
            client_state = train_client_weight(client_loaders[client], global_model,
                                               cur_round, clients_cs_weight[client])
            global_state = weight_aggregate(global_state, client_state, weights[cls])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)
    return round_accuracy


def fed_el2n(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            for client in clients:
                client_loader = client_loaders[client]

                scores, global_idxes = cal_el2n(client_loader, global_model)
                local_prune_indies = np.argsort(scores)[:math.floor(args.prune_frac * len(scores))]
                client_loader.dataset.prune_by_local_indices(local_prune_indies)

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy


def fed_el2n_execute(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    remained_clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            total_original = 0
            clients_num_remained = np.zeros(len(clients))
            for client in clients:
                client_loader = client_loaders[client]
                total_original += len(client_loader.dataset)
                scores, global_idxes = cal_el2n(client_loader, global_model)
                local_remain_indies = np.argsort(scores)[math.floor(args.prune_frac * len(scores)):]
                client_loader.dataset.remain_by_local_indices(local_remain_indies)
                clients_num_remained[client] = len(local_remain_indies)

            sorted_clients = np.argsort(-clients_num_remained)
            remained_clients = sorted_clients[:args.num_clients_remain]
            num_clients_per_round = math.ceil(len(remained_clients) * args.client_selected_ratio)

        selected_clients = np.random.choice(remained_clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy, remained_clients


def fed_ccs(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            for client in clients:
                client_loader = client_loaders[client]
                scores, global_idxes = cal_el2n(client_loader, global_model)

                mis_num = int(0.3 * len(scores))

                data_score, score_index = mislabel_mask(scores, mis_num, True)
                coreset_num = math.floor((1 - args.prune_frac) * len(scores))
                coreset_index, _ = stratified_sampling(data_score=data_score, coreset_num=coreset_num)
                coreset_index = score_index[coreset_index]
                client_loader.dataset.remain_by_local_indices(coreset_index)

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            if client_loaders[client].dataset.get_current_length() == 0:
                client_state = global_model.state_dict()
            else:
                client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy


def fed_moderate(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            for client in clients:
                client_loader = client_loaders[client]

                local_indices = prune_moderate(client_loader, global_model)
                client_loader.dataset.prune_by_local_indices(local_indices)

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            if client_loaders[client].dataset.get_current_length() == 0:
                client_state = global_model.state_dict()
            else:
                client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy
