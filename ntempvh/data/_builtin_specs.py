from __future__ import annotations

from torchvision import datasets

from ntempvh.data._image_loaders import build_eval_transform, build_train_transform
from ntempvh.data.registry import DatasetSpec, register_dataset_spec

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

def register_builtin_dataset_specs() -> None:
    builtins = [
        DatasetSpec(
            name="cifar10",
            num_classes=10,
            build_train_transform=lambda: build_train_transform(CIFAR10_MEAN, CIFAR10_STD, horizontal_flip=True),
            build_eval_transform=lambda: build_eval_transform(CIFAR10_MEAN, CIFAR10_STD),
            build_train_dataset=lambda root, transform: datasets.CIFAR10(
                root=root,
                train=True,
                download=True,
                transform=transform,
            ),
            build_test_dataset=lambda root, transform: datasets.CIFAR10(
                root=root,
                train=False,
                download=True,
                transform=transform,
            ),
        ),
        DatasetSpec(
            name="cifar100",
            num_classes=100,
            build_train_transform=lambda: build_train_transform(CIFAR100_MEAN, CIFAR100_STD, horizontal_flip=True),
            build_eval_transform=lambda: build_eval_transform(CIFAR100_MEAN, CIFAR100_STD),
            build_train_dataset=lambda root, transform: datasets.CIFAR100(
                root=root,
                train=True,
                download=True,
                transform=transform,
            ),
            build_test_dataset=lambda root, transform: datasets.CIFAR100(
                root=root,
                train=False,
                download=True,
                transform=transform,
            ),
        ),
        DatasetSpec(
            name="svhn",
            num_classes=10,
            build_train_transform=lambda: build_train_transform(SVHN_MEAN, SVHN_STD, horizontal_flip=False),
            build_eval_transform=lambda: build_eval_transform(SVHN_MEAN, SVHN_STD),
            build_train_dataset=lambda root, transform: datasets.SVHN(
                root=root,
                split="train",
                download=True,
                transform=transform,
            ),
            build_test_dataset=lambda root, transform: datasets.SVHN(
                root=root,
                split="test",
                download=True,
                transform=transform,
            ),
        ),
    ]

    for spec in builtins:
        try:
            register_dataset_spec(spec)
        except ValueError:
            pass
