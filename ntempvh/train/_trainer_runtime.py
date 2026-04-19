from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch.nn as nn

from ntempvh.train._trainer_helpers import (
    is_intervention_epoch,
    restore_base_learning_rates,
    run_training_epoch,
    set_epoch_learning_rates,
)
from ntempvh.train._trainer_runtime_steps import (
    build_epoch_state,
    build_loader_bundle,
    build_run_settings,
    save_final_checkpoint,
    save_run_summary,
)


def run_train_one_run(
    config: dict[str, Any],
    seed: int,
    out_dir: str,
    *,
    validate_train_config_fn,
    get_device_fn,
    set_seed_fn,
    ensure_dir_fn,
    save_json_fn,
    run_logger_cls,
    get_num_classes_fn,
    build_train_loaders_fn,
    resolve_intervention_config_fn,
    build_test_loader_fn,
    make_model_fn,
    make_optimizer_fn,
    make_scheduler_fn,
    step_scheduler_fn,
    evaluate_fn,
    save_checkpoint_fn,
) -> Path:
    validate_train_config_fn(config)
    device = get_device_fn()
    set_seed_fn(seed)

    out_path = ensure_dir_fn(out_dir)
    ckpt_dir = ensure_dir_fn(out_path / "checkpoints")
    logger = run_logger_cls(out_path)
    settings = build_run_settings(config, device=device, get_num_classes_fn=get_num_classes_fn)
    loader_bundle = build_loader_bundle(
        settings,
        seed=seed,
        config=config,
        build_train_loaders_fn=build_train_loaders_fn,
        resolve_intervention_config_fn=resolve_intervention_config_fn,
        build_test_loader_fn=build_test_loader_fn,
    )

    model = make_model_fn(settings.model_name, num_classes=settings.num_classes).to(device)
    optimizer = make_optimizer_fn(settings.train_cfg, model)
    scheduler = make_scheduler_fn(settings.train_cfg, optimizer)
    criterion = nn.CrossEntropyLoss()
    save_json_fn(out_path / "run_config.json", {"seed": int(seed), "device": str(device), **config})

    best_val_loss = float("inf")
    best_ckpt_path: Path | None = None
    best_epoch: int | None = None
    last_val: dict[str, float] | None = None
    started_at = time.time()

    for epoch in range(1, settings.epochs + 1):
        epoch_state = build_epoch_state(
            epoch=epoch,
            batch_size=settings.batch_size,
            loaders=loader_bundle["loaders"],
            intervention_loaders=loader_bundle["intervention_loaders"],
            intervention_cfg=loader_bundle["intervention_cfg"],
            is_intervention_epoch_fn=is_intervention_epoch,
            set_epoch_learning_rates_fn=set_epoch_learning_rates,
            optimizer=optimizer,
        )
        train_loss_ep, train_acc_ep = run_training_epoch(
            epoch=epoch,
            epochs_total=settings.epochs,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=epoch_state["train_loader"],
            device=device,
        )
        restore_base_learning_rates(
            optimizer,
            intervention_epoch=epoch_state["intervention_epoch"],
            lr_multiplier=float(loader_bundle["intervention_cfg"]["lr_multiplier"]),
            base_lrs=epoch_state["base_lrs"],
        )

        step_scheduler_fn(scheduler)
        val = evaluate_fn(model, loader_bundle["val_loader"], device)
        last_val = val

        logger.log({
            "epoch": int(epoch),
            "train_loss": float(train_loss_ep),
            "train_acc": float(train_acc_ep),
            "val_loss": float(val["val_loss"]),
            "val_acc": float(val["val_acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "is_intervention_epoch": bool(epoch_state["intervention_epoch"]),
            "effective_learning_rate": float(epoch_state["effective_lrs"][0]),
            "effective_batch_size": int(epoch_state["effective_batch_size"]),
            "seconds_elapsed": float(time.time() - started_at),
        })
        print(
            f"epoch {epoch:03d} | "
            f"val_loss={val['val_loss']:.4f} "
            f"val_acc={val['val_acc']:.4f} "
            f"train_acc={train_acc_ep:.4f}"
        )

        if settings.save_best and val["val_loss"] < best_val_loss:
            best_val_loss = float(val["val_loss"])
            best_epoch = epoch
            best_ckpt_path = save_checkpoint_fn(
                ckpt_dir,
                "best",
                model_name=settings.model_name,
                dataset_name=settings.dataset_name,
                seed=seed,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={"best_val_loss": best_val_loss},
            )

        if settings.save_every_epochs > 0 and epoch % settings.save_every_epochs == 0:
            save_checkpoint_fn(
                ckpt_dir,
                f"epoch_{epoch:03d}",
                model_name=settings.model_name,
                dataset_name=settings.dataset_name,
                seed=seed,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    final_ckpt_path = save_final_checkpoint(
        save_final=settings.save_final,
        ckpt_dir=ckpt_dir,
        model_name=settings.model_name,
        dataset_name=settings.dataset_name,
        seed=seed,
        epochs=settings.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    final_train = evaluate_fn(model, loader_bundle["bn_loader"], device)
    final_test = evaluate_fn(model, loader_bundle["test_loader"], device)
    save_run_summary(
        out_path=out_path,
        save_json_fn=save_json_fn,
        seed=seed,
        settings=settings,
        started_at=started_at,
        final_ckpt_path=final_ckpt_path,
        best_ckpt_path=best_ckpt_path,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        last_val=last_val,
        intervention_cfg=loader_bundle["intervention_cfg"],
        final_train=final_train,
        final_test=final_test,
    )
    return final_ckpt_path
