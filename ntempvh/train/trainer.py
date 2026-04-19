from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ntempvh.data.image_classification import (
    build_test_loader,
    build_train_loaders,
    ensure_dataset_registered,
    get_num_classes_for_dataset,
)
from ntempvh.eval.metrics import eval_loss_acc
from ntempvh.models.factory import ensure_model_registered, make_model
from ntempvh.train._trainer_runtime import run_train_one_run
from ntempvh.train.optim import make_optimizer
from ntempvh.train.schedules import make_scheduler, step_scheduler
from ntempvh.utils.config_validation import validate_train_config
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, save_json
from ntempvh.utils.logging import RunLogger
from ntempvh.utils.runtime import call_with_supported_kwargs
from ntempvh.utils.seed import set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    val_loss, val_acc = eval_loss_acc(model, loader, device)
    return {"val_loss": val_loss, "val_acc": val_acc}


def _resolve_intervention_config(
    config: dict[str, Any],
    *,
    base_batch_size: int,
) -> dict[str, Any]:
    raw = dict(config.get("intervention", {}) or {})
    enabled = bool(raw.get("enabled", False))
    start_epoch = int(raw["start_epoch"]) if raw.get("start_epoch", None) is not None else None
    end_epoch = int(raw["end_epoch"]) if raw.get("end_epoch", None) is not None else None
    batch_size = int(raw["batch_size"]) if raw.get("batch_size", None) is not None else None
    return {
        "enabled": enabled,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "lr_multiplier": float(raw.get("lr_multiplier", 1.0)),
        "batch_size": batch_size,
        "effective_batch_size": int(base_batch_size if batch_size is None else batch_size),
        "num_intervention_epochs": (
            int(end_epoch - start_epoch + 1)
            if enabled and start_epoch is not None and end_epoch is not None
            else 0
        ),
    }


def _is_intervention_epoch(intervention_cfg: dict[str, Any], epoch: int) -> bool:
    return bool(
        intervention_cfg["enabled"]
        and intervention_cfg["start_epoch"] is not None
        and intervention_cfg["end_epoch"] is not None
        and intervention_cfg["start_epoch"] <= int(epoch) <= intervention_cfg["end_epoch"]
    )


def _build_train_loaders(
    *,
    dataset_name: str,
    batch_size: int,
    data_root: str,
    val_size: int,
    split_seed: int,
    shuffle_seed: int,
    num_workers: int,
    pin_memory: bool,
):
    return call_with_supported_kwargs(
        build_train_loaders,
        dataset_name=dataset_name,
        root=data_root,
        batch_size=batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _build_test_loader(
    *,
    dataset_name: str,
    batch_size: int,
    data_root: str,
    num_workers: int,
    pin_memory: bool,
):
    return call_with_supported_kwargs(
        build_test_loader,
        dataset_name=dataset_name,
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _save_checkpoint(
    ckpt_dir: Path,
    tag: str,
    *,
    model_name: str,
    dataset_name: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    ckpt = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": int(seed),
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra:
        ckpt.update(extra)
    path = ckpt_dir / f"{tag}.pt"
    torch.save(ckpt, path)
    return path


def train_one_run(config: dict[str, Any], seed: int, out_dir: str) -> Path:
    ensure_dataset_registered(
        str(config.get("dataset", "")),
        builder_path=(
            None
            if config.get("dataset_builder") in (None, "")
            else str(config.get("dataset_builder"))
        ),
    )
    ensure_model_registered(
        str(config.get("model", "")),
        builder_path=(
            None
            if config.get("model_builder") in (None, "")
            else str(config.get("model_builder"))
        ),
    )
    return run_train_one_run(
        config,
        seed,
        out_dir,
        validate_train_config_fn=validate_train_config,
        get_device_fn=get_device,
        set_seed_fn=set_seed,
        ensure_dir_fn=ensure_dir,
        save_json_fn=save_json,
        run_logger_cls=RunLogger,
        get_num_classes_fn=get_num_classes_for_dataset,
        build_train_loaders_fn=_build_train_loaders,
        resolve_intervention_config_fn=_resolve_intervention_config,
        build_test_loader_fn=_build_test_loader,
        make_model_fn=make_model,
        make_optimizer_fn=make_optimizer,
        make_scheduler_fn=make_scheduler,
        step_scheduler_fn=step_scheduler,
        evaluate_fn=evaluate,
        save_checkpoint_fn=_save_checkpoint,
    )
