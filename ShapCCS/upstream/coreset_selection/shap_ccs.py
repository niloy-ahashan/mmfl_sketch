import itertools
import math
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from args import args
from coreset_selection.utils.sv_utils import get_last_layer_state, get_weight_delta, get_grad_cos_sim, \
    prune_by_sv, select_clients, prune_by_max_sv_clients
from train.client import train_client_weight
from train.server import get_weights, weight_aggregate, validate_model


def shap_ccs_max(global_model, client_loaders, test_loader):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv: List[List[float]] = [[] for _ in range(num_clients)]

    client_selected_counts = {client: 0 for client in clients}
    clients_cs_weight = {client: torch.ones(len(client_loaders[client].dataset), device=args.device)
                         for client in clients}
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):

        if cur_round == args.pre_train_round:
            sv = {k: float(np.mean(v)) if len(v) > 0 else 0 for k, v in enumerate(round_sv)}
            if args.cs_reverse:
                sv = {k: -v for k, v in sv.items()}
            clients = prune_by_max_sv_clients(sv, client_loaders)
            num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)

        selected_clients = select_clients(client_selected_counts, clients, num_clients_per_round, cur_round)

        weights = get_weights(selected_clients, client_loaders)
        global_state = None

        last_client_states = {}
        for i, client in enumerate(selected_clients):
            client_state = train_client_weight(client_loaders[client], global_model, cur_round,
                                               clients_cs_weight[client])
            global_state = weight_aggregate(global_state, client_state, weights[i])
            if cur_round < args.pre_train_round:
                last_client_states[client] = get_last_layer_state(client_state)

        if cur_round < args.pre_train_round:
            cal_gpsv(global_model, global_state, selected_clients, weights, last_client_states, round_sv)

        # update global model
        global_model.load_state_dict(global_state)
        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)
    return round_accuracy, clients


def shap_ccs(global_model, client_loaders, test_loader, validate_loader):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv: List[List[float]] = [[] for _ in range(num_clients)]

    remained_clients = np.arange(num_clients)
    clients_cs_weight = {client: torch.ones(len(client_loaders[client].dataset), device=args.device)
                         for client in remained_clients}
    client_selected_counts = {client: 0 for client in clients}

    is_interval = args.ml_method in ['craig', 'gcfl']
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True):
        if is_interval:
            is_prune = cur_round % args.pre_train_round == 0
        else:
            is_prune = cur_round == args.pre_train_round

        if cur_round > 0 and is_prune:
            print('start prune')
            remained_clients, sv = prune_by_sv(round_sv,
                                               client_loaders, global_model,
                                               validate_loader, clients_cs_weight)

            num_clients_per_round = math.ceil(len(remained_clients) * args.client_selected_ratio)
        selected_clients = select_clients(client_selected_counts, remained_clients, num_clients_per_round, cur_round)

        weights = get_weights(selected_clients, client_loaders)
        global_state = None

        last_client_states = {}
        for i, client in enumerate(selected_clients):
            try:
                client_state = train_client_weight(client_loaders[client], global_model,
                                                   cur_round, clients_cs_weight[client])
            except Exception:
                print(f'{args.noisy_client_ratio} client {client}')
                raise 1
            global_state = weight_aggregate(global_state, client_state, weights[i])
            if cur_round < args.pre_train_round:
                last_client_states[client] = get_last_layer_state(client_state)

        if cur_round < args.pre_train_round:
            cal_gpsv(global_model, global_state, selected_clients, weights, last_client_states, round_sv)

        # update global model
        global_model.load_state_dict(global_state)
        accuracy = validate_model(global_model, test_loader)
        round_accuracy.append(accuracy)

    return round_accuracy, remained_clients


def cal_gpsv(global_model, global_state, selected_clients, weights,
             last_client_states, round_sv):
    def psi(subset_utl):
        if 1.25 * last_global_norm < subset_utl <= 1.75 * last_global_norm:
            subset_utl = 1.25 * last_global_norm
        elif subset_utl > 1.75 * last_global_norm:
            subset_utl = 3 * last_global_norm - subset_utl
        return subset_utl

    proj = {}
    old_state = global_model.state_dict()
    last_old_state = get_last_layer_state(old_state)
    last_global_state = get_last_layer_state(global_state)
    last_global_norm = get_weight_delta(last_global_state, last_old_state).norm(2).cpu().item()
    for i, client in enumerate(selected_clients):
        last_client_state = last_client_states[client]
        last_grad_sim = get_grad_cos_sim(last_client_state, last_global_state, last_old_state)
        last_grad_norm = get_weight_delta(last_client_state, last_old_state).norm(2).cpu().item()
        proj[client] = last_grad_sim * last_grad_norm

    num_clients_per_round = len(selected_clients)
    subset_utilities = {frozenset(): 0.0}

    if num_clients_per_round <= 15:
        last_subset = set()
        A, B = 0.0, 0.0  # proj, weight
        weights_dict = {c: weights[i] for i, c in enumerate(selected_clients)}

        for g in range(1 << num_clients_per_round):
            gray = g ^ (g >> 1)
            subset = [selected_clients[i] for i in range(num_clients_per_round) if (gray >> i) & 1]
            if last_subset:
                curr_subset = set(subset)
                diff = list(curr_subset.symmetric_difference(last_subset))
                if diff:
                    c = diff[0]
                    if c in curr_subset:
                        # add
                        A += proj[c] * weights_dict[c]
                        B += weights_dict[c]
                    else:
                        # delete
                        A -= proj[c] * weights_dict[c]
                        B -= weights_dict[c]
            else:
                A = sum(proj[c] * weights_dict[c] for c in subset)
                B = sum(weights_dict[c] for c in subset)

            subset_utl = A / B if B > 0 else 0.0
            subset_utl = psi(subset_utl)
            subset_utilities[frozenset(subset)] = subset_utl
            last_subset = subset

        shapley = [0.0] * num_clients_per_round
        fact_n = math.factorial(num_clients_per_round)
        for i, client in enumerate(selected_clients):
            for subset, s_utility in subset_utilities.items():
                if client not in subset:
                    s_u_i = subset | {client}
                    s_u_i_utility = subset_utilities[frozenset(s_u_i)]
                    utility_diff = s_u_i_utility - s_utility
                    fact_s = math.factorial(len(subset))
                    fact_n_s = math.factorial(num_clients_per_round - len(subset) - 1)
                    shapley[i] += fact_s * fact_n_s / fact_n * utility_diff
    else:
        # GTG approximation
        utl_diffs = {client: [] for client in selected_clients}

        num_permutations = 2 ** 15
        subset_prefix_len = 2
        within_round_threshold = 0.001

        # partial permutations
        for permutation_prefix in itertools.permutations(selected_clients, r=subset_prefix_len):
            each_prefix_num = num_permutations // math.perm(num_clients_per_round, subset_prefix_len)
            for _ in range(each_prefix_num):
                remaining_clients = [client for client in selected_clients if client not in permutation_prefix]

                permutation = np.concatenate((np.array(permutation_prefix),
                                              np.random.permutation(remaining_clients)))
                last_subset_utl = 0
                for i, new_client in enumerate(permutation):
                    subset = permutation[:i + 1]
                    if abs(last_global_norm - last_subset_utl) > within_round_threshold:
                        subset_utl = 0
                        subset_indices = np.where(np.isin(selected_clients, subset))[0]
                        subset_weights = weights[subset_indices]
                        subset_weights /= subset_weights.sum()
                        for j, client in enumerate(subset):
                            subset_utl += proj[client] * subset_weights[j]
                        subset_utl = psi(subset_utl)
                    else:
                        subset_utl = last_subset_utl

                    utl_diffs[new_client].append(subset_utl - last_subset_utl)
                    last_subset_utl = subset_utl

        shapley = [np.mean(utl_diffs[client]) for client in selected_clients]

    shapley = np.array(shapley)
    shapley /= shapley.sum()

    for i, client in enumerate(selected_clients):
        round_sv[client].append(shapley[i])
