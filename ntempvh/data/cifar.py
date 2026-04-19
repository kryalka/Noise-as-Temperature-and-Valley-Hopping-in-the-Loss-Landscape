from __future__ import annotations

"""Compatibility layer for older imports.

The core pipeline uses `ntempvh.data.image_classification`. This module keeps the
older helper names for built-in CIFAR and SVHN presets.
"""

from torch.utils.data import DataLoader

from .image_classification import (
    build_test_loader,
    build_train_loaders,
    ensure_dataset_registered,
    get_num_classes_for_dataset,
    get_supported_image_datasets,
)


def get_cifar_loaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    **kwargs,
):
    ensure_dataset_registered(dataset_name)
    return build_train_loaders(dataset_name, root, batch_size, **kwargs)


def get_cifar10_loaders(root: str, batch_size: int, **kwargs):
    return build_train_loaders("cifar10", root, batch_size, **kwargs)


def get_cifar100_loaders(root: str, batch_size: int, **kwargs):
    return build_train_loaders("cifar100", root, batch_size, **kwargs)


def get_svhn_loaders(root: str, batch_size: int, **kwargs):
    return build_train_loaders("svhn", root, batch_size, **kwargs)


def get_cifar_test_loader(
    dataset_name: str,
    root: str,
    **kwargs,
) -> DataLoader:
    ensure_dataset_registered(dataset_name)
    return build_test_loader(dataset_name, root, **kwargs)


def get_cifar10_test_loader(root: str, **kwargs) -> DataLoader:
    return build_test_loader("cifar10", root, **kwargs)


def get_cifar100_test_loader(root: str, **kwargs) -> DataLoader:
    return build_test_loader("cifar100", root, **kwargs)


def get_svhn_test_loader(root: str, **kwargs) -> DataLoader:
    return build_test_loader("svhn", root, **kwargs)
