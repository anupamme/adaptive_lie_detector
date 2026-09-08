import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices


def get_federated_dataset(dataset_name: str, num_clients: int, alpha: float, seed: int = 42):
    if dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
        labels = np.array(train_dataset.targets)
        num_classes = 10

    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        labels = np.array(train_dataset.targets)
        num_classes = 100

    elif dataset_name == "femnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        labels = np.array(train_dataset.targets)
        num_classes = 62
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]

    return client_datasets, test_dataset, num_classes
