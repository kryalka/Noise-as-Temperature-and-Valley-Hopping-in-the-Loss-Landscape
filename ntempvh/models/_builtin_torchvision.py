from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from ntempvh.models.registry import register_model_builder


def adapt_resnet_for_small_images(model: nn.Module) -> nn.Module:
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def register_builtin_model_builders() -> None:
    resnet_builders = {
        "torchvision_resnet18": resnet18,
        "torchvision_resnet34": resnet34,
        "torchvision_resnet50": resnet50,
        "torchvision_resnet101": resnet101,
    }

    small_input_aliases = {
        "small_input_resnet18": resnet18,
        "small_input_resnet34": resnet34,
        "small_input_resnet50": resnet50,
        "small_input_resnet101": resnet101,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
    }

    for name, builder in resnet_builders.items():
        try:
            register_model_builder(
                name,
                lambda num_classes, builder=builder: builder(num_classes=num_classes),
            )
        except ValueError:
            pass

    for name, builder in small_input_aliases.items():
        try:
            register_model_builder(
                name,
                lambda num_classes, builder=builder: adapt_resnet_for_small_images(
                    builder(num_classes=num_classes)
                ),
            )
        except ValueError:
            pass
