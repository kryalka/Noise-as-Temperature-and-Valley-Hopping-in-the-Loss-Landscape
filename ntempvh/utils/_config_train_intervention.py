from __future__ import annotations

from typing import Any

from ._config_core import coerce_float, coerce_int, ensure_mapping, reject_unknown_keys, require_bool


def validate_train_intervention_config(
    intervention_cfg: dict[str, Any],
    *,
    epochs_total: int,
) -> None:
    reject_unknown_keys("train.intervention", intervention_cfg, {"enabled", "start_epoch", "end_epoch", "lr_multiplier", "batch_size"})
    enabled = require_bool("train.intervention.enabled", intervention_cfg.get("enabled", False))
    start_epoch = intervention_cfg.get("start_epoch", None)
    end_epoch = intervention_cfg.get("end_epoch", None)
    lr_multiplier = intervention_cfg.get("lr_multiplier", 1.0)
    batch_size = intervention_cfg.get("batch_size", None)

    start_value = coerce_int("train.intervention.start_epoch", start_epoch, min_value=1) if start_epoch is not None else None
    end_value = coerce_int("train.intervention.end_epoch", end_epoch, min_value=1) if end_epoch is not None else None
    if start_value is not None and start_value > epochs_total:
        raise ValueError(f"train.intervention.start_epoch must be <= train.training.epochs ({epochs_total}), got {start_value}")
    if end_value is not None and end_value > epochs_total:
        raise ValueError(f"train.intervention.end_epoch must be <= train.training.epochs ({epochs_total}), got {end_value}")
    if start_value is not None and end_value is not None and start_value > end_value:
        raise ValueError(f"train.intervention.start_epoch must be <= end_epoch, got {start_value} > {end_value}")
    if "lr_multiplier" in intervention_cfg or enabled:
        coerce_float("train.intervention.lr_multiplier", lr_multiplier, positive=True)
    if batch_size is not None:
        coerce_int("train.intervention.batch_size", batch_size, min_value=1)
    if enabled and (start_value is None or end_value is None):
        raise ValueError("train.intervention.start_epoch and end_epoch are required when intervention.enabled=true")


def normalize_variant_name(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", "."}:
            out.append("_")
    return "".join(out).strip("_")


def parse_window_spec(name: str, value: Any) -> tuple[int, int, str | None, int | None]:
    label: str | None = None
    batch_size: int | None = None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(":")]
        if len(parts) != 2:
            raise ValueError(f"{name} entries must look like '12:19', got {value!r}")
        start_value = coerce_int(f"{name}.start_epoch", parts[0], min_value=1)
        end_value = coerce_int(f"{name}.end_epoch", parts[1], min_value=1)
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} list entries must contain exactly two integers, got {value!r}")
        start_value = coerce_int(f"{name}.start_epoch", value[0], min_value=1)
        end_value = coerce_int(f"{name}.end_epoch", value[1], min_value=1)
    else:
        window_cfg = ensure_mapping(name, value)
        reject_unknown_keys(name, window_cfg, {"name", "start_epoch", "end_epoch", "batch_size"})
        label_raw = window_cfg.get("name", None)
        if label_raw not in (None, ""):
            label = str(label_raw).strip()
        start_value = coerce_int(f"{name}.start_epoch", window_cfg.get("start_epoch"), min_value=1)
        end_value = coerce_int(f"{name}.end_epoch", window_cfg.get("end_epoch"), min_value=1)
        if window_cfg.get("batch_size", None) is not None:
            batch_size = coerce_int(f"{name}.batch_size", window_cfg["batch_size"], min_value=1)
    if start_value > end_value:
        raise ValueError(f"{name} must satisfy start_epoch <= end_epoch, got {start_value} > {end_value}")
    return int(start_value), int(end_value), label, batch_size
