import math

import numpy as np
from tqdm import tqdm

from args import args
from coreset_selection.utils.sv_utils import prune_by_max_sv_clients
from train.client import train_client
from train.server import get_weights, weight_aggregate, validate_model


def fed_random_data_cs(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            for client in clients:
                client_dataset = client_loaders[client].dataset
                client_dataset.recover_to_origin()
                indies = client_dataset.active_global_indices
                prune_indies = indies[:math.floor(len(indies) * args.prune_frac)]
                client_dataset.prune_by_global_indices(prune_indies)

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


def fed_random_data_cs_execute(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    remained_clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            clients_num_remained = np.zeros(len(clients))
            for client in clients:
                client_dataset = client_loaders[client].dataset
                client_dataset.recover_to_origin()
                indies = client_dataset.active_global_indices
                remain_indies = indies[math.floor(len(indies) * args.prune_frac):]
                client_dataset.remain_by_global_indices(remain_indies)
                clients_num_remained[client] = len(remain_indies)

            sorted_clients = np.argsort(-clients_num_remained)
            remained_clients = sorted_clients[:args.num_clients_remain]
            print('extra prune', clients_num_remained[sorted_clients[args.num_clients_remain:]].sum()
                  / clients_num_remained.sum())
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


def fed_random_client_cs(global_model, client_loaders, test_loader):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            sv = {client: 0 for client in clients}
            clients = prune_by_max_sv_clients(sv, client_loaders)
            num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)

        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])

        global_model.load_state_dict(global_state)
        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy, clients
