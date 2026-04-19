from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


def is_intervention_epoch(intervention_cfg: dict[str, Any], epoch: int) -> bool:
    return (
        intervention_cfg["enabled"]
        and intervention_cfg["start_epoch"] is not None
        and intervention_cfg["end_epoch"] is not None
        and intervention_cfg["start_epoch"] <= int(epoch) <= intervention_cfg["end_epoch"]
    )


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
