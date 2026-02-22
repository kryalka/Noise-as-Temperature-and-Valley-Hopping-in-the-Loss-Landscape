from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from ntempvh.models.registry import (
    get_model_builder,
    get_supported_model_names as _get_supported_model_names,
    register_model_builder,
)


def _adapt_resnet_for_cifar(m: nn.Module) -> nn.Module:
    
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m



def _register_builtin_models() -> None:
    builtins = {
        "resnet18": lambda num_classes: _adapt_resnet_for_cifar(resnet18(num_classes=num_classes)),
        "resnet34": lambda num_classes: _adapt_resnet_for_cifar(resnet34(num_classes=num_classes)),
        "resnet50": lambda num_classes: _adapt_resnet_for_cifar(resnet50(num_classes=num_classes)),
        "resnet101": lambda num_classes: _adapt_resnet_for_cifar(resnet101(num_classes=num_classes)),
    }
    for name, builder in builtins.items():
        try:
            register_model_builder(name, builder)
        except ValueError:
            
            pass


_register_builtin_models()


def make_model(name: str, num_classes: int) -> nn.Module:
    builder = get_model_builder(name)
    return builder(int(num_classes))


def get_supported_model_names() -> tuple[str, ...]:
    return _get_supported_model_names()
