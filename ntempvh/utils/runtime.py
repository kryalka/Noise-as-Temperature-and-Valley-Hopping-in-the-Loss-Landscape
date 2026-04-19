from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def import_from_string(target: str) -> Any:
    text = str(target).strip()
    if not text:
        raise ValueError("Import target must be a non-empty string")

    if ":" in text:
        module_name, attr_name = text.split(":", 1)
    else:
        module_name, _sep, attr_name = text.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(
            "Import target must look like 'package.module:object' or "
            "'package.module.object'"
        )

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Could not resolve '{attr_name}' from module '{module_name}'"
        ) from exc
