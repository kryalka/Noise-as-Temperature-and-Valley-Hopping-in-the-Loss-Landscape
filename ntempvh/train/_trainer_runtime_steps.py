from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch



@dataclass(frozen=True)
class TrainRunSettings:
    dataset_name: str
    model_name: str
    train_cfg: dict[str, Any]
    epochs: int
    batch_size: int
    save_every_epochs: int
    save_final: bool
    save_best: bool
    data_root: str
    val_size: int
    split_seed: int
    num_workers: int
    pin_memory: bool
    num_classes: int



def build_run_settings(
    config: dict[str, Any],
    *,
    device: torch.device,
    get_num_classes_fn,
) -> TrainRunSettings:
    dataset_name = str(config["dataset"]).lower()
    model_name = str(config["model"]).lower()
    train_cfg = dict(config["training"])
    log_cfg = dict(config.get("logging", {}))
    data_cfg = dict(config.get("data", {}))

    return TrainRunSettings(
        dataset_name=dataset_name,
        model_name=model_name,
        train_cfg=train_cfg,
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        save_every_epochs=int(log_cfg.get("save_every_epochs", 0) or 0),
        save_final=bool(log_cfg.get("save_final", True)),
        save_best=bool(log_cfg.get("save_best", True)),
        data_root=str(config.get("data_root", "./data")),
        val_size=int(data_cfg.get("val_size", 5000)),
        split_seed=int(data_cfg.get("split_seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type in ("cuda",))),
        num_classes=int(get_num_classes_fn(dataset_name)),
    )



def build_loader_bundle(
    settings: TrainRunSettings,
    *,
    seed: int,
    config: dict[str, Any],
    build_train_loaders_fn,
    resolve_intervention_config_fn,
    build_test_loader_fn,
) -> dict[str, Any]:
    loaders = build_train_loaders_fn(
        dataset_name=settings.dataset_name,
        batch_size=settings.batch_size,
        data_root=settings.data_root,
        val_size=settings.val_size,
        split_seed=settings.split_seed,
        shuffle_seed=seed,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )

    intervention_cfg = resolve_intervention_config_fn(config, base_batch_size=settings.batch_size)
    intervention_loaders = None
    if intervention_cfg["enabled"] and intervention_cfg["effective_batch_size"] != settings.batch_size:
        intervention_loaders = build_train_loaders_fn(
            dataset_name=settings.dataset_name,
            batch_size=int(intervention_cfg["effective_batch_size"]),
            data_root=settings.data_root,
            val_size=settings.val_size,
            split_seed=settings.split_seed,
            shuffle_seed=seed,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory,
        )

    test_loader = build_test_loader_fn(
        dataset_name=settings.dataset_name,
        batch_size=256,
        data_root=settings.data_root,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )

    return {
        "loaders": loaders,
        "val_loader": loaders.val,
        "bn_loader": loaders.bn,
        "test_loader": test_loader,
        "intervention_cfg": intervention_cfg,
        "intervention_loaders": intervention_loaders,
    }



def build_epoch_state(
    *,
    epoch: int,
    batch_size: int,
    loaders,
    intervention_loaders,
    intervention_cfg: dict[str, Any],
    is_intervention_epoch_fn,
    set_epoch_learning_rates_fn,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    intervention_epoch = is_intervention_epoch_fn(intervention_cfg, epoch)
    base_lrs, effective_lrs = set_epoch_learning_rates_fn(
        optimizer,
        intervention_epoch=intervention_epoch,
        lr_multiplier=float(intervention_cfg["lr_multiplier"]),
    )

    effective_batch_size = (
        int(intervention_cfg["effective_batch_size"])
        if intervention_epoch
        else int(batch_size)
    )
    train_loader = (
        intervention_loaders.train
        if intervention_epoch and intervention_loaders is not None
        else loaders.train
    )

    return {
        "intervention_epoch": intervention_epoch,
        "base_lrs": base_lrs,
        "effective_lrs": effective_lrs,
        "effective_batch_size": effective_batch_size,
        "train_loader": train_loader,
    }



def save_final_checkpoint(
    *,
    save_final: bool,
    ckpt_dir: Path,
    model_name: str,
    dataset_name: str,
    seed: int,
    epochs: int,
    model,
    optimizer,
    scheduler,
    save_checkpoint_fn,
) -> Path:
    if save_final:
        return save_checkpoint_fn(
            ckpt_dir,
            "final",
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            epoch=epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    final_ckpt_path = ckpt_dir / "final.pt"
    torch.save(
        {
            "model": model_name,
            "dataset": dataset_name,
            "seed": int(seed),
            "epoch": int(epochs),
            "state_dict": model.state_dict(),
        },
        final_ckpt_path,
    )
    return final_ckpt_path



def save_run_summary(
    *,
    out_path: Path,
    save_json_fn,
    seed: int,
    settings: TrainRunSettings,
    started_at: float,
    final_ckpt_path: Path,
    best_ckpt_path: Path | None,
    best_val_loss: float,
    best_epoch: int | None,
    last_val: dict[str, float] | None,
    intervention_cfg: dict[str, Any],
    final_train: dict[str, float],
    final_test: dict[str, float],
) -> None:
    final_train_loss = float(final_train["val_loss"])
    final_train_acc = float(final_train["val_acc"])
    final_test_loss = float(final_test["val_loss"])
    final_test_acc = float(final_test["val_acc"])

    save_json_fn(
        out_path / "summary.json",
        {
            "seed": int(seed),
            "epochs": int(settings.epochs),
            "seconds_total": float(time.time() - started_at),
            "final_checkpoint": str(final_ckpt_path),
            "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else None,
            "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": float(last_val["val_loss"]) if last_val is not None else None,
            "final_val_acc": float(last_val["val_acc"]) if last_val is not None else None,
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
            "train_test_gap": float(final_train_acc - final_test_acc),
            "save_every_epochs": int(settings.save_every_epochs),
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
                "data_root": settings.data_root,
                "val_size": settings.val_size,
                "split_seed": settings.split_seed,
                "num_workers": settings.num_workers,
                "pin_memory": settings.pin_memory,
            },
        },
    )
