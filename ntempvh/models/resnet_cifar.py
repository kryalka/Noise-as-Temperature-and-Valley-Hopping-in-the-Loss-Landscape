from __future__ import annotations

"""Compatibility layer for older model imports.

The reusable path uses `ntempvh.models.factory`. This module preserves the
previous helper names for small-input ResNet presets.
"""

from ._builtin_torchvision import adapt_resnet_for_small_images
from .factory import get_supported_model_names, make_model

_adapt_resnet_for_cifar = adapt_resnet_for_small_images
