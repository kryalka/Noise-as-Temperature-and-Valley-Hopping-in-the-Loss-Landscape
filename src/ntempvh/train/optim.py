from __future__ import annotations
import torch
from torch import nn
from torch.optim import Optimizer


def make_optimizer(cfg: dict, model: nn.Module) -> Optimizer:
    opt_name = cfg.get("optimizer", "sgd").lower()
    lr = float(cfg["learning_rate"])
    wd = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.0))

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")
