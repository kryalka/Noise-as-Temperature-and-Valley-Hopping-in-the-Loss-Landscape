from __future__ import annotations

from torch.utils.data import DataLoader

from ntempvh.data._builtin_specs import register_builtin_dataset_specs
from ntempvh.data._image_loaders import DataLoaders, get_dataset_loaders, get_dataset_test_loader
from ntempvh.data.registry import DatasetSpec, get_dataset_spec, get_supported_dataset_names, register_dataset_spec
from ntempvh.utils.runtime import call_with_supported_kwargs, import_from_string


register_builtin_dataset_specs()


def ensure_dataset_registered(
    dataset_name: str,
    *,
    builder_path: str | None = None,
) -> str:
    normalized_name = str(dataset_name).strip().lower()
    if not normalized_name:
        raise ValueError("Dataset name must be a non-empty string")

    try:
        return get_dataset_spec(normalized_name).name
    except ValueError:
        if not builder_path:
            raise

    builder = import_from_string(builder_path)
    if not callable(builder):
        raise ValueError(f"Dataset builder must be callable: {builder_path}")

    spec = call_with_supported_kwargs(builder, name=normalized_name)
    if not isinstance(spec, DatasetSpec):
        raise ValueError(
            f"Dataset builder must return DatasetSpec, got {type(spec).__name__}"
        )
    if str(spec.name).strip().lower() != normalized_name:
        raise ValueError(
            "Dataset builder returned a spec with a different name: "
            f"expected '{normalized_name}', got '{spec.name}'"
        )

    try:
        register_dataset_spec(spec)
    except ValueError:
        pass
    return get_dataset_spec(normalized_name).name


def get_num_classes_for_dataset(dataset_name: str) -> int:
    return int(get_dataset_spec(dataset_name).num_classes)


def get_supported_image_datasets() -> tuple[str, ...]:
    return get_supported_dataset_names()


def build_train_loaders(
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


def build_test_loader(
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
