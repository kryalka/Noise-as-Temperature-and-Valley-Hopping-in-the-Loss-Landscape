from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    build_train_transform: Callable[[], Any]
    build_eval_transform: Callable[[], Any]
    build_train_dataset: Callable[[str, Any], Any]
    build_test_dataset: Callable[[str, Any], Any]


_DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset_spec(spec: DatasetSpec) -> None:
    name = str(spec.name).strip().lower()
    if not name:
        raise ValueError("Dataset spec name must be a non-empty string")
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is already registered")
    _DATASET_REGISTRY[name] = DatasetSpec(
        name=name,
        num_classes=int(spec.num_classes),
        build_train_transform=spec.build_train_transform,
        build_eval_transform=spec.build_eval_transform,
        build_train_dataset=spec.build_train_dataset,
        build_test_dataset=spec.build_test_dataset,
    )


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    name = str(dataset_name).strip().lower()
    if name not in _DATASET_REGISTRY:
        allowed = tuple(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset: {dataset_name}. Expected one of {allowed}")
    return _DATASET_REGISTRY[name]


def get_supported_dataset_names() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_REGISTRY))
