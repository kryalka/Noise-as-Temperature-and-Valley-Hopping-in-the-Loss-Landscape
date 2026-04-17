from __future__ import annotations

from torch.utils.data import DataLoader

from ntempvh.data._builtin_specs import register_builtin_dataset_specs
from ntempvh.data._image_loaders import DataLoaders, get_dataset_loaders, get_dataset_test_loader
from ntempvh.data.registry import get_dataset_spec, get_supported_dataset_names

SUPPORTED_CIFAR_DATASETS = ("cifar10", "cifar100")


def _normalize_dataset_name(dataset_name: str) -> str:
    return get_dataset_spec(dataset_name).name


def get_num_classes_for_dataset(dataset_name: str) -> int:
    return int(get_dataset_spec(dataset_name).num_classes)


def get_supported_image_datasets() -> tuple[str, ...]:
    return get_supported_dataset_names()


register_builtin_dataset_specs()


def get_cifar_loaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    *,
    val_size: int = 5000,
    split_seed: int = 0,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_batch_size: int = 256,
    bn_batch_size: int | None = None,
) -> DataLoaders:
    return get_dataset_loaders(
        dataset_name,
        root,
        batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_batch_size=val_batch_size,
        bn_batch_size=bn_batch_size,
    )


def get_cifar10_loaders(
    root: str,
    batch_size: int,
    *,
    val_size: int = 5000,
    split_seed: int = 0,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_batch_size: int = 256,
    bn_batch_size: int | None = None,
) -> DataLoaders:
    return get_dataset_loaders(
        "cifar10",
        root,
        batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_batch_size=val_batch_size,
        bn_batch_size=bn_batch_size,
    )


def get_cifar100_loaders(
    root: str,
    batch_size: int,
    *,
    val_size: int = 5000,
    split_seed: int = 0,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_batch_size: int = 256,
    bn_batch_size: int | None = None,
) -> DataLoaders:
    return get_dataset_loaders(
        "cifar100",
        root,
        batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_batch_size=val_batch_size,
        bn_batch_size=bn_batch_size,
    )


def get_svhn_loaders(
    root: str,
    batch_size: int,
    *,
    val_size: int = 5000,
    split_seed: int = 0,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_batch_size: int = 256,
    bn_batch_size: int | None = None,
) -> DataLoaders:
    return get_dataset_loaders(
        "svhn",
        root,
        batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_batch_size=val_batch_size,
        bn_batch_size=bn_batch_size,
    )


def get_cifar_test_loader(
    dataset_name: str,
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return get_dataset_test_loader(
        dataset_name,
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_cifar10_test_loader(
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return get_dataset_test_loader(
        "cifar10",
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_cifar100_test_loader(
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return get_dataset_test_loader(
        "cifar100",
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_svhn_test_loader(
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return get_dataset_test_loader(
        "svhn",
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
