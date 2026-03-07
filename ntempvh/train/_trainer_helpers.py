from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from ntempvh.data.cifar import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_dataset_loaders,
)
from ntempvh.utils.io import save_json
from ntempvh.utils.runtime import call_with_supported_kwargs




def resolve_intervention_config(
    config: dict[str, Any],
    *,
    base_batch_size: int,
) -> dict[str, Any]:
    raw = dict(config.get("intervention", {}) or {})
    enabled = bool(raw.get("enabled", False))
    start_epoch = raw.get("start_epoch", None)
    end_epoch = raw.get("end_epoch", None)
    lr_multiplier = float(raw.get("lr_multiplier", 1.0))
    batch_size = raw.get("batch_size", None)

    if start_epoch is not None:
        start_epoch = int(start_epoch)
    if end_epoch is not None:
        end_epoch = int(end_epoch)
    if batch_size is not None:
        batch_size = int(batch_size)

    return {
        "enabled": enabled,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "lr_multiplier": lr_multiplier,
        "batch_size": batch_size,
        "effective_batch_size": int(base_batch_size if batch_size is None else batch_size),
        "num_intervention_epochs": (
            int(end_epoch - start_epoch + 1)
            if enabled and start_epoch is not None and end_epoch is not None
            else 0
        ),
    }



def is_intervention_epoch(intervention_cfg: dict[str, Any], epoch: int) -> bool:
    return bool(
        intervention_cfg["enabled"] and
        intervention_cfg["start_epoch"] is not None and
        intervention_cfg["end_epoch"] is not None and
        intervention_cfg["start_epoch"] <= int(epoch) <= intervention_cfg["end_epoch"]
    )



def build_train_loaders(
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
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        loader_fn = get_cifar10_loaders
    elif dataset_name == "cifar100":
        loader_fn = get_cifar100_loaders
    else:
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

    return call_with_supported_kwargs(
        loader_fn,
        root=data_root,
        batch_size=batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



def save_checkpoint(
    ckpt_dir: Path,
    tag: str,
    *,
    model_name: str,
    dataset_name: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
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



def run_training_epoch(
    *,
    epoch: int,
    epochs_total: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs_total}", leave=False)

    running_loss = 0.0
    seen = 0
    correct_train = 0

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        correct_train += int((pred == y).sum().item())
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        seen += x.size(0)

        pbar.set_postfix(
            train_loss=running_loss / max(1, seen),
            lr=float(optimizer.param_groups[0]["lr"]),
        )

    train_loss = running_loss / max(1, seen)
    train_acc = correct_train / max(1, seen)
    return float(train_loss), float(train_acc)



def set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    intervention_epoch: bool,
    lr_multiplier: float,
) -> tuple[list[float], list[float]]:
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    effective_lrs = [
        float(base_lr * lr_multiplier) if intervention_epoch else float(base_lr)
        for base_lr in base_lrs
    ]

    if intervention_epoch and lr_multiplier != 1.0:
        for group, lr_value in zip(optimizer.param_groups, effective_lrs):
            group["lr"] = lr_value

    return base_lrs, effective_lrs



def restore_base_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    intervention_epoch: bool,
    lr_multiplier: float,
    base_lrs: list[float],
) -> None:
    if intervention_epoch and lr_multiplier != 1.0:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr



def save_train_summary(
    out_path: Path,
    *,
    seed: int,
    epochs: int,
    started_at: float,
    final_ckpt_path: Path,
    best_ckpt_path: Path | None,
    best_val_loss: float,
    best_epoch: int | None,
    last_val: dict[str, float] | None,
    save_every_epochs: int,
    intervention_cfg: dict[str, Any],
    data_root: str,
    val_size: int,
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
    provenance: dict[str, Any],
) -> None:
    save_json(out_path / "summary.json", {
        "seed": int(seed),
        "epochs": int(epochs),
        "seconds_total": float(time.time() - started_at),
        "provenance": provenance,
        "final_checkpoint": str(final_ckpt_path),
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "final_val_loss": float(last_val["val_loss"]) if last_val is not None else None,
        "final_val_acc": float(last_val["val_acc"]) if last_val is not None else None,
        "save_every_epochs": int(save_every_epochs),
        "intervention": {
            "enabled": bool(intervention_cfg["enabled"]),
            "start_epoch": intervention_cfg["start_epoch"],
            "end_epoch": intervention_cfg["end_epoch"],
            "lr_multiplier": float(intervention_cfg["lr_multiplier"]),
            "batch_size": intervention_cfg["batch_size"],
            "effective_batch_size": int(intervention_cfg["effective_batch_size"]),
            "num_intervention_epochs": int(intervention_cfg["num_intervention_epochs"]),
        },
        "data": {
            "data_root": data_root,
            "val_size": val_size,
            "split_seed": split_seed,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    })
