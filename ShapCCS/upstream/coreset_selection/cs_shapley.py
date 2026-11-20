import math

import numpy as np
from tqdm import tqdm

from args import args
from coreset_selection.utils.sv_utils import prune_by_max_sv_clients, read_sv
from train.client import train_client
from train.server import get_weights, weight_aggregate, validate_model


def fed_other_sv_cs(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if cur_round == args.pre_train_round:
            sv = read_sv()
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
