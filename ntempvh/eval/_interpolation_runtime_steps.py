from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ntempvh.eval.bn import recalibrate_bn
from ntempvh.utils.runtime import call_with_supported_kwargs



@dataclass(frozen=True)
class InterpolationRunSettings:
    artifact: dict[str, Any]
    interp_cfg: dict[str, Any]
    path_cfg: dict[str, Any]
    data_cfg: dict[str, Any]
    num_points: int
    bn_batches: int
    path_type: str
    data_root: str
    eval_batch_size: int
    eval_split: str
    device: torch.device



def build_runtime_settings(
    *,
    cfg: dict[str, Any],
    artifact: dict[str, Any],
    get_device_fn,
) -> InterpolationRunSettings:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    if data_cfg is None:
        data_cfg = {}

    interp_cfg = artifact["config"]
    path_cfg = dict(interp_cfg["path"])

    return InterpolationRunSettings(
        artifact=artifact,
        interp_cfg=interp_cfg,
        path_cfg=path_cfg,
        data_cfg=data_cfg,
        num_points=int(path_cfg["num_points"]),
        bn_batches=int(path_cfg["bn_recalib_batches"]),
        path_type=str(path_cfg["type"]),
        data_root=str(interp_cfg["data_root"]),
        eval_batch_size=int(interp_cfg["evaluation"]["batch_size"]),
        eval_split=str(interp_cfg["evaluation"]["split"]),
        device=get_device_fn(),
    )



def build_checkpoint_bundle(
    *,
    ckpt_a: str,
    ckpt_b: str,
    settings: InterpolationRunSettings,
    set_seed_fn,
    validate_checkpoint_pair_fn,
    get_num_classes_fn,
    resolve_path_sequence_fn,
) -> dict[str, Any]:
    checkpoint_a = torch.load(ckpt_a, map_location="cpu")
    checkpoint_b = torch.load(ckpt_b, map_location="cpu")
    set_seed_fn(int(checkpoint_a.get("seed", 0)))
    validate_checkpoint_pair_fn(checkpoint_a, checkpoint_b)

    model_name = str(checkpoint_a["model"]).lower()
    dataset_name = str(checkpoint_a.get("dataset", "")).lower()
    num_classes = int(get_num_classes_fn(dataset_name))
    reference_state_dict = checkpoint_a["state_dict"]

    checkpoint_cache: dict[str, dict[str, Any]] = {
        str(Path(ckpt_a).resolve()): checkpoint_a,
        str(Path(ckpt_b).resolve()): checkpoint_b,
    }
    sequence_ckpts, path_meta, segment_breakpoints = resolve_path_sequence_fn(
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        path_cfg=settings.path_cfg,
        cache=checkpoint_cache,
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        expected_model=model_name,
        expected_dataset=dataset_name,
        reference_state_dict=reference_state_dict,
    )

    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "path_meta": path_meta,
        "segment_breakpoints": segment_breakpoints,
        "state_dicts": [checkpoint["state_dict"] for checkpoint in sequence_ckpts],
    }



def build_loader_bundle(
    *,
    settings: InterpolationRunSettings,
    dataset_name: str,
    select_train_loader_fn,
    select_test_loader_fn,
) -> dict[str, Any]:
    train_loader_fn = select_train_loader_fn(dataset_name)
    loaders = call_with_supported_kwargs(
        train_loader_fn,
        root=settings.data_root,
        batch_size=128,
        val_batch_size=settings.eval_batch_size,
        val_size=int(settings.interp_cfg["evaluation"]["val_size"]),
        bn_batch_size=int(settings.interp_cfg["evaluation"]["bn_batch_size"]),
        split_seed=int(settings.interp_cfg["evaluation"]["split_seed"]),
        num_workers=int(settings.data_cfg.get("num_workers", 0)),
        pin_memory=bool(settings.data_cfg.get("pin_memory", True)),
    )

    bn_loader = loaders.bn
    if bn_loader is None:
        bn_loader = getattr(loaders, "train", None) or getattr(loaders, "val", None)

    if settings.eval_split == "val":
        eval_loader = loaders.val
    elif settings.eval_split == "test":
        test_loader_fn = select_test_loader_fn(dataset_name)
        eval_loader = test_loader_fn(
            root=settings.data_root,
            batch_size=settings.eval_batch_size,
            num_workers=int(settings.data_cfg.get("num_workers", 0)),
            pin_memory=bool(settings.data_cfg.get("pin_memory", True)),
        )
    else:
        raise ValueError(
            f"Unknown evaluation split: {settings.eval_split}. Expected 'val' or 'test'."
        )

    return {
        "bn_loader": bn_loader,
        "eval_loader": eval_loader,
    }



def collect_interpolation_rows(
    *,
    model,
    state_dicts: list[dict[str, Any]],
    path_type: str,
    num_points: int,
    segment_breakpoints,
    interpolate_state_dict_fn,
    bn_loader,
    eval_loader,
    device: torch.device,
    bn_batches: int,
    eval_model_fn,
) -> np.ndarray:
    rows: list[list[float]] = []
    progress = tqdm(np.linspace(0.0, 1.0, num_points), desc="interpolation points", total=num_points)

    for t_value in progress:
        t = float(t_value)
        interpolated_state = interpolate_state_dict_fn(
            path_type=path_type,
            state_dicts=state_dicts,
            t=t,
            segment_breakpoints=segment_breakpoints,
        )
        model.load_state_dict(interpolated_state, strict=True)
        recalibrate_bn(model, bn_loader, device, num_batches=bn_batches, reset_stats=True)
        val_loss, val_acc = eval_model_fn(model, eval_loader, device)
        rows.append([t, float(val_loss), float(val_acc)])
        progress.set_postfix(
            t=f"{t:.2f}",
            val_loss=f"{float(val_loss):.4f}",
            val_acc=f"{float(val_acc):.4f}",
        )

    return np.array(rows, dtype=np.float64)



def build_interpolation_meta(
    *,
    settings: InterpolationRunSettings,
    ckpt_a: str,
    ckpt_b: str,
    model_name: str,
    dataset_name: str,
    path_meta: dict[str, Any],
    endpoint_a_loss: float,
    endpoint_a_acc: float,
    endpoint_b_loss: float,
    endpoint_b_acc: float,
    checkpoint_a: dict[str, Any],
    checkpoint_b: dict[str, Any],
) -> dict[str, Any]:
    meta = {
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "model": model_name,
        "dataset": dataset_name,
        "data_root": settings.data_root,
        "path": path_meta,
        "evaluation": dict(settings.interp_cfg["evaluation"]),
        "artifact": {
            "kind": "interpolation",
            "pair_tag": settings.artifact["pair_tag"],
            "config_signature": settings.artifact["config_signature"],
            "stem": settings.artifact["stem"],
        },
        "endpoint_eval": {
            "A": {"loss": float(endpoint_a_loss), "acc": float(endpoint_a_acc)},
            "B": {"loss": float(endpoint_b_loss), "acc": float(endpoint_b_acc)},
        },
    }

    for tag, checkpoint in (("A", checkpoint_a), ("B", checkpoint_b)):
        if "epoch" in checkpoint:
            meta[f"epoch_{tag}"] = int(checkpoint["epoch"])
        if "seed" in checkpoint:
            meta[f"seed_{tag}"] = int(checkpoint["seed"])

    return meta
