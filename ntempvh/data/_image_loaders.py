from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ntempvh.data.registry import get_dataset_spec


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader


def build_eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    *,
    horizontal_flip: bool,
) -> transforms.Compose:
    ops: list[object] = [transforms.RandomCrop(32, padding=4)]
    if horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader_runtime(
    *,
    seed: int,
    num_workers: int,
) -> tuple[torch.Generator, bool, object | None]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    use_workers = int(num_workers) > 0
    worker_init = seed_worker if use_workers else None
    return generator, use_workers, worker_init


def get_dataset_loaders(
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
    spec = get_dataset_spec(dataset_name)
    bn_bs = int(bn_batch_size) if bn_batch_size is not None else int(val_batch_size)

    train_full_aug = spec.build_train_dataset(root, spec.build_train_transform())
    train_full_noaug = spec.build_train_dataset(root, spec.build_eval_transform())
    num_samples = len(train_full_aug)

    split_generator = torch.Generator()
    split_generator.manual_seed(int(split_seed))
    permutation = torch.randperm(num_samples, generator=split_generator).tolist()

    if val_size == 0:
        train_idx = permutation
        val_idx: list[int] = []
    else:
        val_idx = permutation[:val_size]
        train_idx = permutation[val_size:]

    train_ds = Subset(train_full_aug, train_idx)
    val_ds = Subset(train_full_noaug, val_idx)
    bn_ds = Subset(train_full_noaug, train_idx)

    loader_seed = int(split_seed) if shuffle_seed is None else int(shuffle_seed)
    generator, use_workers, worker_init = build_loader_runtime(
        seed=loader_seed,
        num_workers=int(num_workers),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=use_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=use_workers,
    )
    bn_loader = DataLoader(
        bn_ds,
        batch_size=bn_bs,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=use_workers,
    )
    return DataLoaders(train=train_loader, val=val_loader, bn=bn_loader)


def get_dataset_test_loader(
    dataset_name: str,
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    spec = get_dataset_spec(dataset_name)
    test_ds = spec.build_test_dataset(root, spec.build_eval_transform())
    generator, use_workers, worker_init = build_loader_runtime(
        seed=0,
        num_workers=int(num_workers),
    )
    return DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=use_workers,
    )
