import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    '''specify'''
    parser.add_argument("--result_fp", type=str, default="results_111")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--option", type=str, default="")
    parser.add_argument("--record_data", type=bool, default=False)
    parser.add_argument("--prune_frac", type=float, default=0.0)
    parser.add_argument("--ml_method", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    """test"""
    parser.add_argument("--num_clients_remain", type=int, default=0)
    parser.add_argument("--cs_reverse", type=bool, default=False)

    """dataset"""
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="cifar10")

    """noisy"""
    parser.add_argument("--noisy_client_ratio", type=float, default=0.0)
    parser.add_argument("--noise_ratio", type=float, default=0.0)

    if "ipykernel" in sys.modules:
        res = parser.parse_args([])
    else:
        res = parser.parse_args()

    res.batch_size = 32
    res.test_batchsize = 32
    res.train_ratio = 0.95  # for methods requiring validation
    res.client_selected_ratio = 0.15

    res.num_classes = 10
    res.rounds = 100
    res.pre_train_round = 10

    if res.dataset in ['mnist', 'fmnist']:
        res.num_epochs = 1
        res.lr = 0.05
    elif res.dataset == ['cifar10', 'cinic10']:
        res.num_epochs = 3
        res.lr = 0.05
    elif res.dataset == 'cifar100':
        res.num_epochs = 3
        res.lr = 0.01
        res.rounds = 500
        res.num_classes = 100
        res.pre_train_round = 50

    return res


args = parse_args()
