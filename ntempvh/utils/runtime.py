from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)
