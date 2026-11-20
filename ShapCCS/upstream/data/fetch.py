import os

import torchvision
from torchvision import transforms as transforms


def fetch_dataset_mnist():
    """ Collect MNIST """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        r'/data/mnist',
        train=True,
        download=True,
        transform=transform,
    )

    test_data = torchvision.datasets.MNIST(
        r'/data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    return train_data, test_data


def fetch_dataset_fmnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = torchvision.datasets.FashionMNIST(
        r'/data/fmnist',
        train=True,
        download=True,
        transform=transform,
    )

    test_data = torchvision.datasets.FashionMNIST(
        r'/data/fmnist',
        train=False,
        download=True,
        transform=transform
    )

    return train_data, test_data


def fetch_dataset_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        r'/data/cifar10',
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = torchvision.datasets.CIFAR10(
        r'/data/cifar10',
        train=False,
        download=True,
        transform=transform_test
    )

    return train_data, test_data


def fetch_dataset_cifar100():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = torchvision.datasets.CIFAR100(
        r'/data/cifar100',
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = torchvision.datasets.CIFAR100(
        r'/data/cifar100',
        train=False,
        download=True,
        transform=transform_test
    )

    return train_data, test_data


def fetch_dataset_tiny_imagenet():
    data_dir = r'/data/tiny_imagenet'

    IMAGE_SIZE = 64

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, 'train')

    val_dir = os.path.join(data_dir, 'val')

    train_data = torchvision.datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )

    test_data = torchvision.datasets.ImageFolder(
        val_dir,
        transform=transform_val
    )

    return train_data, test_data

def fetch_dataset_cinic10():
    data_dir = r'/data/cinic10'

    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_data = torchvision.datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )

    test_data = torchvision.datasets.ImageFolder(
        test_dir,
        transform=transform_test
    )

    return train_data, test_data