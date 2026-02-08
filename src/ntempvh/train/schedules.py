from __future__ import annotations
import torch
from torch.optim import Optimizer


def make_scheduler(cfg: dict, optimizer: Optimizer):
    name = cfg.get("scheduler", "cosine")
    epochs = int(cfg["epochs"])

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none" or name is None:
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def step_scheduler(scheduler) -> None:
    if scheduler is not None:
        scheduler.step()
