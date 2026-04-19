from __future__ import annotations

import math
from typing import Any, Iterable




def ensure_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def reject_unknown_keys(name: str, mapping: dict[str, Any], allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    unknown = sorted(str(key) for key in mapping.keys() if str(key) not in allowed_set)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {', '.join(unknown)}")



def reject_alias_conflict(
    *,
    cfg: dict[str, Any],
    section: dict[str, Any],
    legacy_key: str,
    nested_key: str,
    section_name: str,
) -> None:
    if legacy_key in cfg and nested_key in section:
        raise ValueError(
            f"Ambiguous config: both top-level '{legacy_key}' and "
            f"'{section_name}.{nested_key}' are set"
        )



def coerce_int(name: str, value: Any, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if min_value is not None and ivalue < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {ivalue}")
    return ivalue



def coerce_float(
    name: str,
    value: Any,
    *,
    min_value: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float, got bool")
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if not math.isfinite(fvalue):
        raise ValueError(f"{name} must be finite, got {fvalue}")
    if positive and fvalue <= 0.0:
        raise ValueError(f"{name} must be > 0, got {fvalue}")
    if min_value is not None and fvalue < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {fvalue}")
    return fvalue


def require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}")
    return value



def require_choice(name: str, value: Any, allowed: Iterable[str]) -> str:
    text = str(value).strip().lower()
    allowed_list = tuple(str(item).strip().lower() for item in allowed)
    if text not in allowed_list:
        raise ValueError(f"{name} must be one of {allowed_list}, got {value!r}")
    return text
