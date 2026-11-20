"""ShapCCS selection and training logic, copied from the released code (ShapCCS/upstream @ 8a45bbb).

Function bodies are upstream's text. Every change is on a line marked ``# PORT:`` and is one of:
  * device moves: upstream calls ``x.to(device)`` on one tensor; here ``x`` is a tuple with one tensor per
    modality, so ``_to_dev(x)`` moves each (a plain tensor passes through, so single-modality data works too);
  * ``copy_global_model``: upstream rebuilds the model with ``global_model.__class__()`` (no-argument
    constructors only); here ``copy.deepcopy``, which gives the same weights;
  * ``get_lr``: our datasets take the FMNIST / CIFAR-10 schedule (0.99 decay per round);
  * ``prune_by_sv``: only the data-level methods the port runs ('moderate', the released default, and
    'random') are kept; the others raise;
  * the ``shap_ccs`` loop: test accuracy and logging go through an ``on_round`` callback, and the loop also
    returns the per-round GPSV, the selected clients and the clients kept at round ``pre_train_round``.
``args`` is upstream's global namespace; ``configure(namespace)`` sets it.
``check_port.py`` imports the upstream modules and checks these functions give the same results.

Kept as released (behaviour worth knowing when reading results):
  * ``get_weights`` weights clients by ``len(client_loader)``, the number of batches, not samples.
  * ``prune_moderate`` reads features through the client's shuffled DataLoader, so the positions it marks
    for pruning are positions in that shuffled order, while ``prune_by_local_indices`` applies them to the
    unshuffled order. The number of samples kept matches the budget; which samples are kept is close to random.
  * ``select_clients`` is round-robin (least-selected first) during pretraining; the paper's Alg. 1 samples at random.
"""

import copy
import itertools
import math
from typing import List

import numpy as np
import torch
from torch import cosine_similarity, nn
from tqdm import tqdm

args = None  # PORT: set by configure(); upstream does ``from args import args``

output_weight = 'output.weight'
output_bias = 'output.bias'
output_params = [output_weight, output_bias]

criterion = nn.CrossEntropyLoss()


def configure(namespace):  # PORT: replaces upstream's args.py module
    global args
    args = namespace


def _to_dev(x):  # PORT: tuple of per-modality tensors, or one tensor
    if isinstance(x, (list, tuple)):
        return [t.to(args.device) for t in x]
    return x.to(args.device)


# ==== train/server.py ====
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


# ==== train/client.py ====
def copy_global_model(global_model):
    global_model_copy = copy.deepcopy(global_model)  # PORT: upstream global_model.__class__() + load_state_dict
    global_model_copy.load_state_dict(global_model.state_dict())
    global_model_copy.to(args.device)
    return global_model_copy


def get_lr(current_round):
    if args.dataset in ['mnist', 'cifar10', 'fmnist', 'cinic10', 'uci_har', 'mhealth']:  # PORT: + our datasets
        return pow(0.99, current_round - 1) * args.lr
    elif args.dataset in ['cifar100']:
        cos_inner = math.pi * current_round / args.rounds
        return 0.5 * args.lr * (1 + math.cos(cos_inner))
    else:
        raise 'lr decay strategy is not set'


def train_client_weight(client_loader, global_model, cur_round, cs_weight):
    client_model = copy_global_model(global_model)
    client_model.train()
    lr = get_lr(cur_round)
    optimizer = torch.optim.SGD(client_model.parameters(), lr)
    total = 0
    for epoch in range(args.num_epochs):
        for x, y, _, idx in client_loader:
            x = _to_dev(x)  # PORT: x.to(args.device)
            y = y.to(args.device)
            idx = idx.to(args.device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss = (loss * cs_weight[idx.long()]).mean()
            loss.backward()
            optimizer.step()
            total += len(y)  # PORT: x.shape[0]
    local_state = client_model.state_dict()

    # free cache
    del client_model
    del optimizer

    return local_state


# ==== coreset_selection/utils/sv_utils.py ====
def get_last_layer_state(state):
    return {k: state[k] for k in [output_weight, output_bias]}


def get_weight_delta(new_state, old_state):
    state_delta = {k: new_state[k] - old_state[k] for k in old_state}
    delta = torch.cat([v.view(-1) for v in state_delta.values()])
    return delta


def get_grad_cos_sim(state_a, state_b, old_state):
    delta_a = get_weight_delta(state_a, old_state)
    delta_b = get_weight_delta(state_b, old_state)
    return cosine_similarity(delta_a, delta_b, dim=0).mean().cpu().item()


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

        if args.ml_method in ["random", "random_size_max"]:  # PORT: elif in upstream (branches before it removed)
            indies = client_dataset.active_global_indices
            remain_indices = np.random.choice(indies, budget, replace=False)
            client_dataset.remain_by_global_indices(remain_indices)
        elif args.ml_method == 'moderate':
            local_indices = prune_moderate(client_loader, global_model, budget)
            client_loader.dataset.prune_by_local_indices(local_indices)
        else:  # PORT: craig / el2n / gcfl / ccs / fed_cs branches not ported
            raise NotImplementedError(f"ml_method {args.ml_method!r} is not ported")  # PORT

        remained_clients.append(client)

    return remained_clients, sv


# ==== coreset_selection/utils/moderate_utils.py ====
def get_median(features, targets):
    num_classes = args.num_classes
    feature_dim = features.shape[-1]
    prot = np.zeros((num_classes, feature_dim), dtype=features.dtype)

    for i in range(num_classes):
        mask = (targets == i)
        if np.any(mask):
            prot[i] = np.median(features[mask], axis=0)
        else:
            prot[i] = np.zeros(feature_dim, dtype=features.dtype)
    return prot


def get_distance(features, labels):
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)

    return distance


def get_features(client_loader, global_model):
    targets, features = [], []

    global_model.eval()
    with torch.no_grad():
        for x, y, global_idx, _ in client_loader:
            x = _to_dev(x)  # PORT: x.to(args.device)
            feature = global_model.get_backbone(x).detach().cpu().numpy()
            features.extend([feature[i] for i in range(feature.shape[0])])
            targets.extend(y.numpy().tolist())

    features = np.array(features)
    targets = np.array(targets)

    return features, targets


def prune_moderate(client_loader, global_model, budget=None):
    features, targets = get_features(client_loader, global_model)
    distance = get_distance(features, targets)

    if budget is None:
        rate = 1 - args.prune_frac
    else:
        rate = budget / client_loader.dataset.get_original_length()

    low = 0.5 - rate / 2
    high = 0.5 + rate / 2

    sorted_idx = distance.argsort()
    low_idx = round(distance.shape[0] * low)
    high_idx = round(distance.shape[0] * high)

    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    ids = ids[:client_loader.dataset.get_original_length() - 1]

    return ids


# ==== coreset_selection/shap_ccs.py ====
def shap_ccs(global_model, client_loaders, on_round, validate_loader=None):  # PORT: test_loader -> on_round
    num_clients = len(client_loaders)
    clients = np.arange(num_clients)
    round_accuracy = []
    num_clients_per_round = math.ceil(len(clients) * args.client_selected_ratio)
    round_sv: List[List[float]] = [[] for _ in range(num_clients)]
    selected_per_round = []  # PORT: bookkeeping for the npz
    sv = None  # PORT

    remained_clients = np.arange(num_clients)
    clients_cs_weight = {client: torch.ones(len(client_loaders[client].dataset), device=args.device)
                         for client in remained_clients}
    client_selected_counts = {client: 0 for client in clients}

    is_interval = args.ml_method in ['craig', 'gcfl']
    for cur_round in tqdm(range(args.rounds), dynamic_ncols=True, disable=True):  # PORT: no bar (log table)
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
                raise  # PORT: upstream `raise 1` (a TypeError that hides the real error)
            global_state = weight_aggregate(global_state, client_state, weights[i])
            if cur_round < args.pre_train_round:
                last_client_states[client] = get_last_layer_state(client_state)

        if cur_round < args.pre_train_round:
            cal_gpsv(global_model, global_state, selected_clients, weights, last_client_states, round_sv)

        # update global model
        global_model.load_state_dict(global_state)
        selected_per_round.append([int(c) for c in selected_clients])  # PORT
        accuracy = on_round(cur_round, selected_clients, remained_clients)  # PORT: validate_model(...)
        round_accuracy.append(accuracy)

    return round_accuracy, remained_clients, round_sv, sv, selected_per_round  # PORT: + bookkeeping


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
