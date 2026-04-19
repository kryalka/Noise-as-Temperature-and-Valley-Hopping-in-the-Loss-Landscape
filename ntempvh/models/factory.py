from __future__ import annotations

from typing import Callable

import torch.nn as nn

from ntempvh.models._builtin_torchvision import register_builtin_model_builders
from ntempvh.models.registry import (
    get_model_builder,
    get_supported_model_names as _get_supported_model_names,
    register_model_builder,
)
from ntempvh.utils.runtime import import_from_string


register_builtin_model_builders()


def ensure_model_registered(
    model_name: str,
    *,
    builder_path: str | None = None,
) -> str:
    normalized_name = str(model_name).strip().lower()
    if not normalized_name:
        raise ValueError("Model name must be a non-empty string")

    try:
        get_model_builder(normalized_name)
        return normalized_name
    except ValueError:
        if not builder_path:
            raise

    builder = import_from_string(builder_path)
    if not callable(builder):
        raise ValueError(f"Model builder must be callable: {builder_path}")

    try:
        register_model_builder(normalized_name, builder)
    except ValueError:
        pass
    get_model_builder(normalized_name)
    return normalized_name


def make_model(name: str, num_classes: int) -> nn.Module:
    builder: Callable[[int], nn.Module] = get_model_builder(name)
    return builder(int(num_classes))


def get_supported_model_names() -> tuple[str, ...]:
    return _get_supported_model_names()
