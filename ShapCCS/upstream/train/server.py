import math

import numpy as np
import torch
from tqdm import tqdm

from args import args
from train.client import train_client


def fed_avg(global_model, client_loaders, test_loader):
    clients = np.arange(len(client_loaders))
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        # choose clients
        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            # train client
            client_state = train_client(client_loaders[client], global_model, cur_round)
            # aggregate model
            global_state = weight_aggregate(global_state, client_state, weights[i])

        # update the global model
        global_model.load_state_dict(global_state)

        # validate
        accuracy = validate_model(global_model, test_loader)
        # print(f'\n{accuracy}')
        round_accuracy.append(accuracy)

    return round_accuracy


def get_weights(selected_clients, client_loaders):
    temp = np.array([len(client_loaders[client]) for client in selected_clients])
    return temp / temp.sum()


def weight_aggregate(global_state, local_state, weight):
    if global_state is None:
        global_state = {k: local_state[k] * weight for k in local_state}
    else:
        for k in global_state:
            global_state[k] += local_state[k] * weight
    return global_state


def weights_aggregate(client_states, clients, weights):
    global_state = None
    for i, client in enumerate(clients):
        global_state = weight_aggregate(global_state, client_states[client], weights[i])
    return global_state


def validate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    # print(f'accuracy {correct / total:.2g}')
    return correct / total
