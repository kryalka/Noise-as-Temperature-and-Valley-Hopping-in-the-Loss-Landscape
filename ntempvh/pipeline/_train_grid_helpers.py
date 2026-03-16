from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ntempvh.utils.config_validation import validate_train_grid_config
from ntempvh.utils.io import load_yaml


def slugify(value: str) -> str:
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", "."}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "variant"


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def fmt_lr_for_filename(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def fmt_variant_number(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def variant_tag(variant: dict[str, Any] | None, *, idx: int) -> str:
    if not variant:
        return ""
    name = variant.get("name", None)
    if name not in (None, ""):
        return slugify(str(name))
    return f"variant{idx + 1}"


def parse_window_spec(value: Any) -> tuple[int, int, str | None, int | None]:
    label: str | None = None
    batch_size: int | None = None

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(":")]
        if len(parts) != 2:
            raise ValueError(f"intervention_windows entries must look like '12:19', got {value!r}")
        start_epoch = int(parts[0])
        end_epoch = int(parts[1])
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"intervention_windows list entries must contain two integers, got {value!r}")
        start_epoch = int(value[0])
        end_epoch = int(value[1])
    else:
        window_cfg = dict(value)
        label_raw = window_cfg.get("name", None)
        if label_raw not in (None, ""):
            label = str(label_raw).strip()
        start_epoch = int(window_cfg["start_epoch"])
        end_epoch = int(window_cfg["end_epoch"])
        if window_cfg.get("batch_size", None) is not None:
            batch_size = int(window_cfg["batch_size"])

    return start_epoch, end_epoch, label, batch_size


def expand_intervention_variants(grid_cfg: dict[str, Any]) -> list[dict[str, Any] | None]:
    explicit_variants = list(grid_cfg.get("intervention_variants", []) or [])
    window_specs = list(grid_cfg.get("intervention_windows", []) or [])
    lr_multipliers = [float(value) for value in grid_cfg.get("intervention_lr_multipliers", []) or []]
    batch_sizes = list(grid_cfg.get("intervention_batch_sizes", []) or [])

    variants: list[dict[str, Any] | None] = list(explicit_variants)
    if not window_specs:
        return variants or [None]

    batch_options = batch_sizes if batch_sizes else [None]
    for window_value in window_specs:
        start_epoch, end_epoch, label, window_batch_size = parse_window_spec(window_value)
        window_name = label or f"w{start_epoch}_{end_epoch}"
        for lr_multiplier in lr_multipliers:
            effective_batch_options = [window_batch_size] if window_batch_size is not None else batch_options
            for batch_size in effective_batch_options:
                variant_name = f"{window_name}_x{fmt_variant_number(lr_multiplier)}"
                if batch_size is not None and window_batch_size is None:
                    variant_name = f"{variant_name}_b{int(batch_size)}"

                variant = {
                    "name": variant_name,
                    "enabled": True,
                    "start_epoch": int(start_epoch),
                    "end_epoch": int(end_epoch),
                    "lr_multiplier": float(lr_multiplier),
                }
                if batch_size is not None:
                    variant["batch_size"] = int(batch_size)
                variants.append(variant)

    return variants or [None]


def path_for_cli(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_train_grid_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    validate_train_grid_config(cfg)
    return cfg


def build_train_variant_cfg(
    base_cfg: dict[str, Any],
    *,
    learning_rate: float,
    batch_size: int,
    config_overrides: dict[str, Any],
    intervention_variant: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = deep_merge(cfg, config_overrides)
    cfg.setdefault("training", {})
    cfg["training"]["learning_rate"] = float(learning_rate)
    cfg["training"]["batch_size"] = int(batch_size)

    if intervention_variant:
        cfg.setdefault("intervention", {})
        variant_override = {key: copy.deepcopy(value) for key, value in intervention_variant.items() if key != "name"}
        cfg["intervention"] = deep_merge(dict(cfg.get("intervention", {})), variant_override)

    return cfg
