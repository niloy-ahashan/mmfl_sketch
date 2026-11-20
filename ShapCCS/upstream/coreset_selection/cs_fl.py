import math

import numpy as np
from tqdm import tqdm

from args import args
from coreset_selection.utils.fedcs_utils import prune_fedcs
from coreset_selection.utils.gcfl_utils import prune_gcfl
from train.client import train_client
from train.server import get_weights, weight_aggregate, validate_model


def fed_cs(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            prune_fedcs(client_loaders, global_model)

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


def fed_gc(global_model, client_loaders, test_loader, validate_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)

    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round > 0 and cur_round % args.pre_train_round == 0:
            for client in clients:
                client_loader = client_loaders[client]
                budget = max(math.floor((1 - args.prune_frac) * client_loader.dataset.get_original_length()), 1)
                idxes = prune_gcfl(client_loader, global_model, validate_loader, budget)
                idxes = idxes[:budget]
                client_loader.dataset.remain_by_local_indices(idxes)
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


def fed_gc_execute(global_model, client_loaders, test_loader, validate_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    remained_clients = []
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round > 0 and cur_round % args.pre_train_round == 0:
            for client in clients:
                client_loader = client_loaders[client]
                budget = max(math.floor((1 - args.prune_frac) * len(client_loader.sampler)), 1)
                idxes = prune_gcfl(client_loader, global_model, validate_loader, budget)
                idxes = idxes[:budget]
                client_loader.dataset.remain_by_local_indices(idxes)

            sorted_clients = sorted(clients, key=lambda x: client_loaders[x].dataset.get_current_length())
            remained_clients = sorted_clients[-args.num_clients_remain:]
            num_clients_per_round = math.ceil(len(remained_clients) * args.client_selected_ratio)

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)

        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)
    return round_accuracy, remained_clients
