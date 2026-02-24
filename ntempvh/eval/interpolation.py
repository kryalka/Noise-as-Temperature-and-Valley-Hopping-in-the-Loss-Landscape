from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.data.cifar import (
    get_cifar10_loaders,
    get_cifar10_test_loader,
    get_cifar100_loaders,
    get_cifar100_test_loader,
    get_dataset_loaders,
    get_dataset_test_loader,
    get_num_classes_for_dataset,
)
from ntempvh.eval._interpolation_runtime import run_interpolation_compute
from ntempvh.models.resnet_cifar import make_model
from ntempvh.utils.artifacts import build_interpolation_artifact_context
from ntempvh.utils.checkpoints import validate_checkpoint_pair
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, load_yaml, save_json
from ntempvh.utils.seed import set_seed



def _select_train_loader_fn(dataset_name: str):
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        return get_cifar10_loaders
    if dataset_name == "cifar100":
        return get_cifar100_loaders

    def loader_fn(*, root, batch_size, val_size=5000, split_seed=0, shuffle_seed=None, num_workers=0, pin_memory=True, val_batch_size=256, bn_batch_size=None):
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

    return loader_fn



def _select_test_loader_fn(dataset_name: str):
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        return get_cifar10_test_loader
    if dataset_name == "cifar100":
        return get_cifar100_test_loader

    def loader_fn(*, root, batch_size=256, num_workers=0, pin_memory=True):
        return get_dataset_test_loader(
            dataset_name,
            root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loader_fn



def run_interpolation_from_config(
    ckpt_a: str,
    ckpt_b: str,
    cfg: dict[str, Any],
    out_dir: str,
) -> Path:
    return run_interpolation_compute(
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        cfg=cfg,
        out_dir=out_dir,
        build_artifact_context_fn=build_interpolation_artifact_context,
        get_device_fn=get_device,
        set_seed_fn=set_seed,
        validate_checkpoint_pair_fn=validate_checkpoint_pair,
        get_num_classes_fn=get_num_classes_for_dataset,
        select_train_loader_fn=_select_train_loader_fn,
        select_test_loader_fn=_select_test_loader_fn,
        make_model_fn=make_model,
        ensure_dir_fn=ensure_dir,
        save_json_fn=save_json,
    )



def run_interpolation(ckpt_a: str, ckpt_b: str, config_path: str, out_dir: str) -> Path:
    cfg = load_yaml(config_path)
    return run_interpolation_from_config(ckpt_a, ckpt_b, cfg, out_dir)
