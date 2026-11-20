import gc
import heapq
import math
import time

import numpy as np
import sklearn
import torch
from torch import nn

from args import args
from train.client import copy_global_model


class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None):
    '''
    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    # print('Computing facility location submodular order...')
    N = S.shape[0]
    no = smtk if no == 0 else no

    V = list(range(N))
    start = time.time()
    F = FacilityLocation(S, V)
    order, _ = lazy_greedy_heap(F, V, B)
    greedy_time = time.time() - start
    F_val = 0

    order = np.asarray(order, dtype=np.int64)

    if order.size == 0:
        return order, np.asarray([], dtype=np.float64), greedy_time, F_val

    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    # print('time (sec) for computing facility location:', greedy_time, flush=True)
    collected = gc.collect()
    return order, sz, greedy_time, F_val


def similarity(X, metric):
    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    '''
    # print(f'Computing similarity for {metric}...', flush=True)
    start = time.time()
    dists = sklearn.metrics.pairwise_distances(X, metric=metric, n_jobs=1)
    # dists = gdist(X, X, optimize_level=0, output='cpu')
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')

    return S, elapsed


def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None):
    class_indices = np.where(y == c)[0]
    # print(f'num_per_class: {num_per_class}')
    # print(f'c: {c}')
    # print(f'len(class_indices): {len(class_indices)}')

    if len(class_indices) == 0:
        return class_indices, np.array([0]), 0.0, 0.0

    if len(class_indices) == 1:
        return class_indices, np.array([1.0]), 0.0, 0.0

    S, S_time = similarity(X[class_indices], metric=metric)
    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S, num_per_class, c, smtk, no, stoch_greedy, weights)
    return class_indices[order], cluster_sz, greedy_time, S_time


def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, outdir='.'):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    # classes = np.array([_ for _ in range(len(classes_pos))])
    classes = np.array(range(args.num_classes))
    C = len(classes)  # number of classes
    # assert np.array_equal(classes, np.arange(C))
    # assert B % C == 0
    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        # temp = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        # num_per_class = np.array([0 for _ in range(args.num_classes)])
        # for i, cls in enumerate(classes):
        #     num_per_class[cls] = temp[i]
        # print('not equal_num')
    # print(f'classes: {classes} num_per_class: {num_per_class} len(x): {len(y)}')
    # print(f'Greedy: selecting {num_per_class} elements')

    # order_mg_all = np.zeros([C, num_per_class], dtype=np.int64)
    # cluster_sizes_all = np.zeros([C, num_per_class], dtype=np.float32)
    # greedy_time_all = np.zeros([C, num_per_class], dtype=np.int64)
    # similarity_time_all = np.zeros([C, num_per_class], dtype=np.int64)

    # pool = ThreadPool(C)
    # order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*pool.map(
    #     lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, stoch_greedy, weights), classes))
    # pool.terminate()
    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios[class_ratios > 0]))
        # print(f'Selecting with ratios {np.array(class_ratios)}')
        # print(f'Class proportions {np.array(props)}')

    # order_mg_all = np.array(order_mg_all)
    # cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes if props[c] > 0])))):
        for c in classes:
            ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    # class_ratios = np.divide([np.sum(y == i) for i in classes], N)
    # weights_mg[y[order_mg] == np.argmax(class_ratios)] /= (np.max(class_ratios) / np.min(class_ratios))

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    # for c in classes:
    #     class_indices = np.where(y == c)[0]
    #     S, similarity_time_all[c] = similarity(X[class_indices], metric=metric)
    #     order, cluster_sz, greedy_time_all[c], F_val = get_facility_location_submodular_order(S, num_per_class, c, smtk)
    #     order_mg_all[c] = class_indices[order]
    #     cluster_sizes_all[c] = cluster_sz
    #     save_cluster_sizes(cluster_sizes_all[c], metric=f'{metric}_class{c}', outdir=outdir)
    # cluster_sizes_all /= N

    # choose 1st from each class, then 2nd from each class, etc.
    # i.e. column-major order
    # order_mg_all = np.array(order_mg_all)
    # cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32) / N
    # order_mg = order_mg_all.flatten(order='F')
    # weights_mg = cluster_sizes_all.flatten(order='F')

    # sort by descending cluster size within each class
    # cluster_order = np.argsort(-cluster_sizes_all, axis=1)
    # rows_selector = np.arange(C)[:, np.newaxis]
    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = []  # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv >= 0:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals


def prune_craig(global_model, client_loader, budget):
    B = budget

    client_model = copy_global_model(global_model)
    client_model.eval()
    preds = torch.zeros(len(client_loader.dataset), args.num_classes).cuda()
    labels = torch.zeros(len(client_loader.dataset), dtype=torch.int)
    with torch.no_grad():
        for i, (inputs, targets, _, idx) in enumerate(client_loader):
            input_var = inputs.cuda()
            preds[idx, :] = nn.Softmax(dim=1)(client_model(input_var))
            labels[idx] = targets.int()
    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    preds -= np.eye(args.num_classes)[labels]

    fl_labels = labels

    subset_indices, cs_subset_weight, _, _, ordering_time, similarity_time = get_orders_and_weights(
        B, preds, 'euclidean', smtk=0, no=0, y=fl_labels, stoch_greedy=0,
        equal_num=False)

    cs_weights = np.zeros(len(client_loader.dataset))
    cs_subset_weight = cs_subset_weight / np.sum(cs_subset_weight) * len(cs_subset_weight)

    cs_weights[subset_indices] = cs_subset_weight
    cs_weight = torch.from_numpy(cs_weights).float().cuda()
    return subset_indices, cs_weight
