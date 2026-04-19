from __future__ import annotations

from typing import Callable

import torch.nn as nn

_MODEL_REGISTRY: dict[str, Callable[[int], nn.Module]] = {}

def register_model_builder(name: str, builder: Callable[[int], nn.Module]) -> None:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Model name must be a non-empty string")
    if normalized_name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{normalized_name}' is already registered")
    _MODEL_REGISTRY[normalized_name] = builder

def get_model_builder(name: str) -> Callable[[int], nn.Module]:
    normalized_name = str(name).strip().lower()
    if normalized_name not in _MODEL_REGISTRY:
        allowed = tuple(sorted(_MODEL_REGISTRY))
        raise ValueError(f"unknown model: {name}. expected one of {allowed}")
    return _MODEL_REGISTRY[normalized_name]


def get_supported_model_names() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))
