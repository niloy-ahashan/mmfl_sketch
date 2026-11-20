import random
from args import args
from torch.utils.data import DataLoader

from train.model import CNN_MNIST, CNN_CIFAR_10, CNN_FMNIST, ResNet_CIFAR100, ResNet_TinyImageNet, ResNet_CINIC10

"""
0
1
10
100
1000
"""


random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch.utils.data

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from data.data_loder import partition_client_data, PrunableDataset
from data.fetch import fetch_dataset_mnist, fetch_dataset_fmnist, fetch_dataset_cifar10, fetch_dataset_cifar100, \
    fetch_dataset_tiny_imagenet, fetch_dataset_cinic10
from args import args
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


def init_training():
    """
    init for model training
    """
    if args.dataset == 'mnist':
        train_data, test_data = fetch_dataset_mnist()
        global_model = CNN_MNIST()
    elif args.dataset == 'cifar10':
        train_data, test_data = fetch_dataset_cifar10()
        global_model = CNN_CIFAR_10()
    elif args.dataset == 'cinic10':
        train_data, test_data = fetch_dataset_cinic10()
        global_model = CNN_CIFAR_10()
        # global_model = ResNet_CINIC10(num_classes=10)
    elif args.dataset == 'cifar100':
        train_data, test_data = fetch_dataset_cifar100()
        global_model = ResNet_CIFAR100(num_classes=100)
    elif args.dataset == 'fmnist':
        train_data, test_data = fetch_dataset_fmnist()
        global_model = CNN_FMNIST()
    elif args.dataset == 'tiny_imagenet':
        train_data, test_data = fetch_dataset_tiny_imagenet()
        global_model = ResNet_TinyImageNet()
    else:
        raise 'dataset is not correct'

    client_loaders, validate_loader = partition_client_data(train_data, batch_size=args.batch_size)

    test_loader = DataLoader(
        PrunableDataset(test_data, range(len(test_data))),
        batch_size=args.test_batchsize,
        shuffle=False
    )

    global_model.to(args.device)

    return test_loader, client_loaders, validate_loader, global_model


test_loader, client_loaders, validate_loader, global_model = init_training()
