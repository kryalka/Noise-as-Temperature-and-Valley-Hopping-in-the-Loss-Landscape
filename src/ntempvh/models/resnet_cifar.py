from __future__ import annotations
import torch.nn as nn
from torchvision.models import resnet18, resnet34


def _adapt_resnet_for_cifar(m: nn.Module) -> nn.Module:
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = resnet18(num_classes=num_classes)
        return _adapt_resnet_for_cifar(m)
    if name == "resnet34":
        m = resnet34(num_classes=num_classes)
        return _adapt_resnet_for_cifar(m)
    raise ValueError(f"Unknown model: {name}")
