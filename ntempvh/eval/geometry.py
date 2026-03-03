from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ntempvh.data.cifar import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_dataset_loaders,
    get_num_classes_for_dataset,
)
from ntempvh.eval._geometry_runtime import run_geometry_compute
from ntempvh.eval.bn import recalibrate_bn
from ntempvh.eval.metrics import eval_classification, params_to_vector, vector_to_params
from ntempvh.models.resnet_cifar import make_model
from ntempvh.utils.artifacts import build_geometry_artifact_context
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, load_yaml, save_json
from ntempvh.utils.runtime import call_with_supported_kwargs
from ntempvh.utils.seed import set_seed


def _select_train_loader_fn(dataset_name: str):
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        return get_cifar10_loaders
    if dataset_name == "cifar100":
        return get_cifar100_loaders

    def loader_fn(*, root, batch_size, val_size=5000, split_seed=0, shuffle_seed=None, num_workers=0, pin_memory=True, val_batch_size=256, bn_batch_size=None):
        return get_dataset_loaders(dataset_name, root, batch_size, val_size=val_size, split_seed=split_seed, shuffle_seed=shuffle_seed, num_workers=num_workers, pin_memory=pin_memory, val_batch_size=val_batch_size, bn_batch_size=bn_batch_size)

    return loader_fn


def _sample_unit_directions(d: int, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    z = torch.randn((m, d), device=device, dtype=dtype)
    z_norm = torch.norm(z, dim=1, keepdim=True).clamp_min(1e-12)
    return z / z_norm



def _save_failure_json(
    *,
    out_dir: str | Path,
    artifact: dict[str, Any],
    ckpt_path: str,
    model_name: str,
    dataset_name: str,
    device: torch.device,
    alpha: float,
    m: int,
    eval_batch_size: int,
    num_eval_batches: int | None,
    bn_batches: int,
    raw_base: dict | None,
    bn_base: dict | None,
    reason: str,
    extra: dict | None = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    fail_path = Path(out_dir) / f"{artifact['stem']}.json"
    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "ckpt": str(ckpt_path),
        "dataset": dataset_name,
        "model": model_name,
        "device": str(device),
        "alpha": alpha,
        "num_directions": m,
        "eval_batch_size": eval_batch_size,
        "num_eval_batches": num_eval_batches,
        "bn_recalib_batches": bn_batches,
        "raw_base": raw_base,
        "bn_base": bn_base,
        "artifact": {
            "kind": "geometry",
            "checkpoint_tag": artifact["checkpoint_tag"],
            "config_signature": artifact["config_signature"],
            "stem": artifact["stem"],
        },
    }
    if extra:
        payload["extra"] = extra
    save_json(fail_path, payload)
    return fail_path



@torch.no_grad()
def compute_geometry(ckpt_path: str, geometry_cfg_path: str, out_path: str) -> Path:
    return run_geometry_compute(
        ckpt_path,
        geometry_cfg_path,
        out_path,
        load_yaml_fn=load_yaml,
        build_geometry_artifact_context_fn=build_geometry_artifact_context,
        get_device_fn=get_device,
        set_seed_fn=set_seed,
        get_num_classes_fn=get_num_classes_for_dataset,
        select_train_loader_fn=_select_train_loader_fn,
        call_with_supported_kwargs_fn=call_with_supported_kwargs,
        make_model_fn=make_model,
        eval_classification_fn=eval_classification,
        recalibrate_bn_fn=recalibrate_bn,
        params_to_vector_fn=params_to_vector,
        vector_to_params_fn=vector_to_params,
        sample_unit_directions_fn=_sample_unit_directions,
        save_failure_json_fn=_save_failure_json,
        ensure_dir_fn=ensure_dir,
        save_json_fn=save_json,
    )
