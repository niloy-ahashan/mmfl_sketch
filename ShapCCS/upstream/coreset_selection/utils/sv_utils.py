import copy
import itertools
import json
import math
import os
import time
from typing import List

import numpy as np
import torch
from torch import cosine_similarity
from torch.nn import functional as F
from tqdm import tqdm

from args import args
from coreset_selection.utils.ccs_utils import mislabel_mask, stratified_sampling
from coreset_selection.utils.craig_utils import prune_craig
from coreset_selection.utils.el2n_utils import cal_el2n
from coreset_selection.utils.fedcs_utils import prune_fedcs
from coreset_selection.utils.gcfl_utils import prune_gcfl
from coreset_selection.utils.moderate_utils import prune_moderate
from train.client import train_client
from train.server import get_weights, weight_aggregate, validate_model, weights_aggregate

output_weight = 'output.weight'
output_bias = 'output.bias'
output_params = [output_weight, output_bias]


def get_front_layers_state(state):
    return {k: state[k] for k in set(state) - {output_weight, output_bias}}


def get_last_layer_state(state):
    return {k: state[k] for k in [output_weight, output_bias]}


def get_weight_delta(new_state, old_state):
    state_delta = {k: new_state[k] - old_state[k] for k in old_state}
    delta = torch.cat([v.view(-1) for v in state_delta.values()])
    return delta


def get_logits_delta(new_state, old_state):
    return new_state[output_bias] - old_state[output_bias]


def get_grad_cos_sim(state_a, state_b, old_state):
    delta_a = get_weight_delta(state_a, old_state)
    delta_b = get_weight_delta(state_b, old_state)
    return cosine_similarity(delta_a, delta_b, dim=0).mean().cpu().item()


def get_sv_file_path():
    file_path = f'results_seed_{args.seed}'
    if args.noisy_client_ratio > 0:
        file_path += f'_{args.noisy_client_ratio}_{args.noise_ratio}'
    file_path += f'/{args.dirichlet_alpha}/{args.dataset}/sv'
    file_path += f'/{args.option.removesuffix("_cs")}.txt'
    return file_path


def read_sv():
    with open(get_sv_file_path(), 'r') as f:
        sv = json.load(f)
    sv = {int(k): v for k, v in sv.items()}
    return sv


def write_sv(round_sv):
    sv = {i: np.mean(elem).astype(float) if len(elem) > 0 else 0.0 for i, elem in enumerate(round_sv)}
    file_path = get_sv_file_path()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(sv, f, indent=4)
        print(f'written to {file_path}')


def prune_by_max_sv_clients(sv, client_loaders):
    total_remain = 0
    # total_original = len(client_loaders[0].dataset.data)
    total_original = len(client_loaders[0].dataset.data) * args.train_ratio

    sorted_clients = sorted(sv, key=sv.get, reverse=True)
    remained_clients = []
    for i, client in enumerate(sorted_clients):
        total_remain += len(client_loaders[client].dataset)
        remained_clients.append(client)
        if total_remain / total_original >= (1 - args.prune_frac):
            break

    client_remain_frac = total_remain / total_original
    data_prune_frac = 1 - (1 - args.prune_frac) / client_remain_frac
    data_prune_frac = max(0, data_prune_frac)

    for client in remained_clients:
        indies = client_loaders[client].dataset.active_global_indices
        prune_indies = np.random.choice(indies, math.floor(len(indies) * data_prune_frac), replace=False)
        client_loaders[client].dataset.prune_by_global_indices(prune_indies)

    return remained_clients


def select_clients(client_selected_counts, clients, num_clients_per_round, cur_round):
    if args.pre_train_round < 50 and cur_round < args.pre_train_round:
        selected_clients = []
        min_count = min(client_selected_counts.values())
        least_selected_clients = [client_id for client_id, count in client_selected_counts.items() if
                                  count == min_count]

        if len(least_selected_clients) >= num_clients_per_round:
            selected_clients = np.random.choice(least_selected_clients, num_clients_per_round,
                                                replace=False).tolist()
        else:
            selected_clients.extend(least_selected_clients)
            num_remaining = num_clients_per_round - len(least_selected_clients)
            remaining_clients = [client for client in clients if client not in selected_clients]
            selected_clients.extend(np.random.choice(remaining_clients, num_remaining, replace=False).tolist())

        for client in selected_clients:
            client_selected_counts[client] += 1

        return np.array(selected_clients)

    else:
        return np.random.choice(clients, num_clients_per_round, replace=False)


def prune_by_sv(round_sv, client_loaders, global_model=None,
                validate_loader=None, clients_cs_weight=None):
    def check_remain_enough(num_prune):
        nonlocal threshold, abundant
        cur_budget_remain = budgets[right_clients].sum()
        cur_data_remain = num_arr[right_clients].sum()
        if abundant and cur_budget_remain >= cur_data_remain - num_prune:
            # 剩余预算 > 剩余数据量, 说明threshold过大, 接下来要尽量将预算花完
            # threshold设置为-1,不斩杀任何客户端
            # 更新剩余客户端的预算
            print(f'no prune after client {client} rank {idx} '
                  f'dirichlet {args.dirichlet_alpha} prune {args.prune_frac} '
                  f'budget_remain {cur_budget_remain} data_remain {cur_data_remain - num_prune} '
                  f'threshold {threshold} ')
            num_prune = cur_data_remain - cur_budget_remain
            abundant = False
            threshold = -1
            budgets[right_clients[1:]] = num_arr[right_clients[1:]]
            budgets[client] = num_data - num_prune
            return False
        else:
            return True

    def adjust_budgets(num_adj, adj_clients):
        while num_adj > 0:
            remain_capacity = num_arr[adj_clients] - budgets[adj_clients]
            remain_idxes = np.where(remain_capacity > 0)[0]
            quotient = num_adj // len(remain_idxes)
            remainder = num_adj % len(remain_idxes)
            adj = np.full(len(remain_idxes), quotient)
            adj[:remainder] += 1

            adj = np.minimum(adj, remain_capacity[remain_idxes])
            budgets[adj_clients[remain_idxes]] += adj

            if adj.sum() > 0:
                num_adj -= adj.sum()
            else:
                break

    sv = {k: np.mean(v).astype(float) if len(v) > 0 else 0 for k, v in enumerate(round_sv)}

    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    num_arr = np.array([client_loaders[client].dataset.get_original_length() for client in clients])
    sorted_clients = np.array(sorted(clients, key=lambda x: sv[x]))

    # 根据 sv 分配预算
    sv_normalize = np.array(list(sv.values()))
    sv_normalize -= sv_normalize.min()
    sv_normalize /= sv_normalize.sum()
    total_original = len(client_loaders[0].dataset.data) * args.train_ratio
    total_budget = total_original * (1 - args.prune_frac)
    budgets = np.ceil(sv_normalize * total_budget).astype(int)
    # 调整 budget > 自身数据量的 budget
    total_excess_budget = 0
    for client in clients:
        if budgets[client] > num_arr[client]:
            total_excess_budget += budgets[client] - num_arr[client]
            budgets[client] = num_arr[client]
    adjust_budgets(total_excess_budget, sorted_clients)
    threshold = total_budget / args.num_clients_remain
    # threshold = args.threshold

    remained_clients = []
    abundant = True
    for idx, client in enumerate(sorted_clients):
        client_loader = client_loaders[client]
        client_dataset = client_loader.dataset
        client_dataset.recover_to_origin()
        num_data = client_dataset.get_original_length()
        right_clients = sorted_clients[idx:]
        budget = budgets[client]
        if budget < threshold:
            if check_remain_enough(num_data):
                # 剩余总预算 < 剩余总数据量, 斩杀,不需要考虑了
                budgets[client] = 0
                adjust_budgets(budget, right_clients[1:])
                continue
            else:
                # 剩余总预算 > 剩余总数据量, 当前客户端不被展示,且预算被更新
                budget = budgets[client]

        if not check_remain_enough(num_data - budget):
            # 剩余总预算 > 剩余总数据量, 当前客户端的预算被更新
            budget = budgets[client]

        if args.ml_method == "craig":
            subset_indices, cs_weight = prune_craig(global_model, client_loader, budget)
            client_dataset.remain_by_local_indices(subset_indices)
            clients_cs_weight[client] = cs_weight
        elif args.ml_method == "el2n":
            scores, global_idxes = cal_el2n(client_loader, global_model)
            local_remain_indies = np.argsort(scores)[-budget:]
            client_dataset.remain_by_local_indices(local_remain_indies)
        elif args.ml_method in ["random", "random_size_max"]:
            indies = client_dataset.active_global_indices
            remain_indices = np.random.choice(indies, budget, replace=False)
            client_dataset.remain_by_global_indices(remain_indices)
        elif args.ml_method == "gcfl":
            idxes = prune_gcfl(client_loader, global_model, validate_loader, budget)
            client_dataset.remain_by_local_indices(idxes)
        elif args.ml_method == 'ccs':
            scores, global_idxes = cal_el2n(client_loader, global_model)
            mis_num = min(int(0.3 * len(scores)), len(scores) - budget)
            data_score, score_index = mislabel_mask(scores, mis_num, True)
            coreset_index, _ = stratified_sampling(data_score=data_score, coreset_num=budget)
            coreset_index = score_index[coreset_index]
            client_loader.dataset.remain_by_local_indices(coreset_index)
        elif args.ml_method == 'moderate':
            local_indices = prune_moderate(client_loader, global_model, budget)
            client_loader.dataset.prune_by_local_indices(local_indices)
        elif args.ml_method == 'fed_cs':
            pass
        else:
            raise "ml_method is invalid"

        remained_clients.append(client)

    if args.ml_method == 'fed_cs':
        prune_fedcs(client_loaders, global_model, remained_clients,
                    {client: budgets[client] for client in remained_clients}
                    )
    return remained_clients, sv


def fed_gtg(global_model, client_loaders, validate_loader):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv: List[List[np.floating]] = [[] for _ in range(num_clients)]

    num_permutations = 50
    subset_prefix_len = math.floor(num_clients_per_round * 0.1)
    print('subset_prefix_len:', subset_prefix_len)

    last_round_full_utl = 0
    within_round_threshold = 0.001
    between_round_threshold = 0.001

    client_selected_counts = {client: 0 for client in clients}

    times = []

    for cur_round in tqdm(range(args.pre_train_round), dynamic_ncols=True):

        client_states = {}
        selected_clients = select_clients(client_selected_counts, clients, num_clients_per_round, cur_round)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            client_states[client] = client_state

        start = time.time()

        global_model.load_state_dict(global_state)
        full_utl = validate_model(global_model, validate_loader)
        # between-round truncation
        if abs(last_round_full_utl - full_utl) < between_round_threshold:
            continue

        last_round_full_utl = full_utl

        cur_sv = {client: [] for client in selected_clients}
        # partial permutations
        for permutation_prefix in itertools.permutations(selected_clients, r=subset_prefix_len):
            # each partial permutation has same num of remaining permutations
            each_prefix_num = num_permutations // math.perm(num_clients_per_round, subset_prefix_len)
            for _ in range(each_prefix_num):
                print(f'round {cur_round} prefix {permutation_prefix} {_} of {each_prefix_num}')
                remaining_clients = [client for client in selected_clients if client not in permutation_prefix]

                permutation = np.concatenate((np.array(permutation_prefix),
                                              np.random.permutation(remaining_clients)))

                last_subset_utl = 0
                for i, new_client in enumerate(permutation):
                    # cal subset utl
                    subset = permutation[:i + 1]
                    # within-round truncation
                    if abs(full_utl - last_subset_utl) > within_round_threshold:
                        weights = get_weights(selected_clients, client_loaders)
                        sub_state = weights_aggregate(client_states, subset, weights)
                        global_model.load_state_dict(sub_state)
                        subset_utl = validate_model(global_model, validate_loader)
                    else:
                        subset_utl = last_subset_utl
                    cur_sv[new_client].append(subset_utl - last_subset_utl)
                    last_subset_utl = subset_utl

        for client, sv in cur_sv.items():
            round_sv[client].append(np.mean(sv))
        end = time.time()
        times.append(end - start)
        # reset global model
        global_model.load_state_dict(global_state)
    print(f'gtg with {sum(times)}')
    write_sv(round_sv)


def fed_cssv(global_model, client_loaders):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv = [[] for _ in range(num_clients)]
    times = []
    for cur_round in tqdm(range(args.pre_train_round), dynamic_ncols=True):
        similarity_matrix = torch.zeros((num_clients_per_round, args.num_classes))
        client_states = {}
        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            client_states[client] = client_state
        start = time.time()
        for cls in range(args.num_classes):
            w1 = torch.cat([
                global_state[output_weight][cls].view(-1),
                global_state[output_bias][cls].view(-1),
            ]).view(1, -1)
            w1 = F.normalize(w1, p=2)

            for i, client in enumerate(selected_clients):
                client_state = client_states[client]
                w2 = torch.cat([
                    client_state[output_weight][cls].view(-1),
                    client_state[output_bias][cls].view(-1),
                ]).view(1, -1)
                w2 = F.normalize(w2, p=2)

                sim = F.cosine_similarity(w1, w2).item()
                similarity_matrix[i][cls] = sim

        svs = torch.mean(similarity_matrix, dim=1).numpy()
        for i, client in enumerate(selected_clients):
            round_sv[client].append(svs[i])
        end = time.time()
        times.append(end - start)
        global_model.load_state_dict(global_state)
    print(f'cssv with {sum(times)}')
    write_sv(round_sv)


def fed_cgsv_sample(global_model, client_loaders):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv: List[List[np.floating]] = [[] for _ in range(num_clients)]
    times = []
    num_permutations = 50
    for cur_round in tqdm(range(args.pre_train_round), dynamic_ncols=True):
        client_states = {}
        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            client_states[client] = client_state

        start = time.time()

        cur_sv = {client: [] for client in selected_clients}
        old_state = global_model.state_dict()
        for _ in range(num_permutations):
            permutation = np.random.permutation(selected_clients)
            last_subset_utl = 0
            for i, new_client in enumerate(permutation):
                subset = permutation[:i + 1]
                weights = get_weights(selected_clients, client_loaders)
                sub_state = weights_aggregate(client_states, subset, weights)
                subset_utl = get_grad_cos_sim(sub_state, global_state, old_state)
                cur_sv[new_client].append(subset_utl - last_subset_utl)
                last_subset_utl = subset_utl
        for client, sv in cur_sv.items():
            round_sv[client].append(np.mean(sv))
        end = time.time()
        times.append(end - start)
        global_model.load_state_dict(global_state)
    print(f'cgsv with {sum(times)}')
    write_sv(round_sv)


def fed_nsv(global_model, client_loaders, validate_loader):
    def average_weights(w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] = w_avg[key] + w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv = [[] for _ in range(num_clients)]

    num_permutations = 50

    accuracy = validate_model(global_model, validate_loader)
    round_accuracy.append(accuracy)

    times = []

    for cur_round in tqdm(range(args.pre_train_round), dynamic_ncols=True):
        client_states = {}
        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            client_states[i] = client_state

        shapley = np.zeros(num_clients_per_round)
        shapley_estimator = [[[] for _ in range(num_clients_per_round)] for _ in range(num_clients_per_round)]
        start = time.time()

        for _ in range(num_permutations):
            print(_)
            for j in range(num_clients_per_round):
                index = np.random.permutation(num_clients_per_round)
                left_length = j + 1

                left_weights = []
                for i in range(left_length):
                    current_weight = client_states[index[i]]
                    left_weights.append(current_weight)

                left_weight = average_weights(left_weights)
                global_model.load_state_dict(left_weight)
                left_acc = validate_model(global_model, validate_loader)

                right_weights = []
                for i in range(left_length, num_clients_per_round):
                    current_weight = client_states[index[i]]
                    right_weights.append(current_weight)

                if len(right_weights) > 0:
                    right_weight = average_weights(right_weights)
                    global_model.load_state_dict(right_weight)
                    right_acc = validate_model(global_model, validate_loader)
                else:
                    right_acc = round_accuracy[-1]

                for k in range(len(index)):
                    if k <= j:
                        shapley_estimator[index[k]][j].append(left_acc - right_acc)
                    else:
                        shapley_estimator[index[k]][num_clients_per_round - j - 2].append(right_acc - left_acc)

        for i in range(num_clients_per_round):
            for j in range(num_clients_per_round):
                if len(shapley_estimator[i][j]) > 0:
                    shapley[i] += np.mean(shapley_estimator[i][j]) / num_clients_per_round

        for i, client in enumerate(selected_clients):
            round_sv[client].append(shapley[i])

        end = time.time()
        times.append(end - start)

        global_model.load_state_dict(global_state)
        accuracy = validate_model(global_model, validate_loader)
        round_accuracy.append(accuracy)
    print(f'nsv with {sum(times)}')
    write_sv(round_sv)


def fed_cgsv(global_model, client_loaders):
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv = [[] for _ in range(num_clients)]

    for cur_round in tqdm(range(args.pre_train_round), dynamic_ncols=True):
        client_states = {}
        selected_clients = np.random.choice(clients, num_clients_per_round, replace=False)
        weights = get_weights(selected_clients, client_loaders)
        global_state = None
        for i, client in enumerate(selected_clients):
            client_state = train_client(client_loaders[client], global_model, cur_round)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            client_states[client] = client_state

        old_state = global_model.state_dict()
        for client in selected_clients:
            client_state = client_states[client]
            round_sv[client].append(get_grad_cos_sim(client_state, global_state, old_state))

        global_model.load_state_dict(global_state)
    write_sv(round_sv)
