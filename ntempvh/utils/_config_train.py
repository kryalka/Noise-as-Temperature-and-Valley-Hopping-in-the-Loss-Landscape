from __future__ import annotations

from typing import Any

from ntempvh.data.cifar import get_supported_image_datasets
from ntempvh.models.resnet_cifar import get_supported_model_names

from ._config_core import coerce_float, coerce_int, ensure_mapping, reject_unknown_keys, require_bool, require_choice
from ._config_train_intervention import (
    normalize_variant_name,
    parse_window_spec,
    validate_train_intervention_config,
)

_normalize_variant_name = normalize_variant_name
_parse_window_spec = parse_window_spec


def validate_train_grid_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("train-grid config", cfg)
    reject_unknown_keys("train-grid config", cfg, {
        "base_config", "out_root", "seeds", "learning_rates", "batch_sizes", "config_overrides",
        "intervention_variants", "intervention_windows", "intervention_lr_multipliers", "intervention_batch_sizes",
    })

    if not str(cfg.get("base_config", "")).strip():
        raise ValueError("train-grid.base_config is required")
    if "out_root" in cfg and not str(cfg.get("out_root", "")).strip():
        raise ValueError("train-grid.out_root must be a non-empty string")

    seeds = cfg.get("seeds", None)
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("train-grid.seeds must be a non-empty list")
    for idx, value in enumerate(seeds):
        coerce_int(f"train-grid.seeds[{idx}]", value, min_value=0)

    learning_rates = cfg.get("learning_rates", None)
    if not isinstance(learning_rates, (list, tuple)) or not learning_rates:
        raise ValueError("train-grid.learning_rates must be a non-empty list")
    for idx, value in enumerate(learning_rates):
        coerce_float(f"train-grid.learning_rates[{idx}]", value, positive=True)

    batch_sizes = cfg.get("batch_sizes", None)
    if not isinstance(batch_sizes, (list, tuple)) or not batch_sizes:
        raise ValueError("train-grid.batch_sizes must be a non-empty list")
    for idx, value in enumerate(batch_sizes):
        coerce_int(f"train-grid.batch_sizes[{idx}]", value, min_value=1)

    config_overrides = ensure_mapping("train-grid.config_overrides", cfg.get("config_overrides"))
    reject_unknown_keys("train-grid.config_overrides", config_overrides, {"dataset", "model", "data_root", "data", "training", "logging", "intervention"})
    training_overrides = ensure_mapping("train-grid.config_overrides.training", config_overrides.get("training"))
    epochs_total = coerce_int("train-grid.config_overrides.training.epochs", training_overrides.get("epochs", 10**9), min_value=1)

    intervention_override = ensure_mapping("train-grid.config_overrides.intervention", config_overrides.get("intervention"))
    if intervention_override:
        validate_train_intervention_config(intervention_override, epochs_total=int(epochs_total))

    intervention_variants = list(cfg.get("intervention_variants", []) or [])
    intervention_windows = list(cfg.get("intervention_windows", []) or [])
    intervention_lr_multipliers = list(cfg.get("intervention_lr_multipliers", []) or [])
    intervention_batch_sizes = list(cfg.get("intervention_batch_sizes", []) or [])
    if not isinstance(intervention_variants, list):
        raise ValueError("train-grid.intervention_variants must be a list")
    if not isinstance(intervention_windows, list):
        raise ValueError("train-grid.intervention_windows must be a list")
    if not isinstance(intervention_lr_multipliers, list):
        raise ValueError("train-grid.intervention_lr_multipliers must be a list")
    if not isinstance(intervention_batch_sizes, list):
        raise ValueError("train-grid.intervention_batch_sizes must be a list")

    for idx, value in enumerate(intervention_lr_multipliers):
        coerce_float(f"train-grid.intervention_lr_multipliers[{idx}]", value, positive=True)
    for idx, value in enumerate(intervention_batch_sizes):
        if value is not None:
            coerce_int(f"train-grid.intervention_batch_sizes[{idx}]", value, min_value=1)
    if intervention_windows and not intervention_lr_multipliers:
        raise ValueError("train-grid.intervention_lr_multipliers must be a non-empty list when intervention_windows are set")

    for idx, window in enumerate(intervention_windows):
        start_epoch, end_epoch, _label, _batch_size = parse_window_spec(f"train-grid.intervention_windows[{idx}]", window)
        if start_epoch > int(epochs_total):
            raise ValueError(f"train-grid.intervention_windows[{idx}].start_epoch must be <= training.epochs ({epochs_total})")
        if end_epoch > int(epochs_total):
            raise ValueError(f"train-grid.intervention_windows[{idx}].end_epoch must be <= training.epochs ({epochs_total})")

    seen_names: set[str] = set()
    for idx, variant in enumerate(intervention_variants):
        variant_cfg = ensure_mapping(f"train-grid.intervention_variants[{idx}]", variant)
        reject_unknown_keys(f"train-grid.intervention_variants[{idx}]", variant_cfg, {"name", "enabled", "start_epoch", "end_epoch", "lr_multiplier", "batch_size"})
        if "name" in variant_cfg:
            name = str(variant_cfg.get("name", "")).strip()
            if not name:
                raise ValueError(f"train-grid.intervention_variants[{idx}].name must be a non-empty string")
            normalized_name = normalize_variant_name(name)
            if not normalized_name:
                raise ValueError(f"train-grid.intervention_variants[{idx}].name must contain at least one alphanumeric character")
            if normalized_name in seen_names:
                raise ValueError(f"Duplicate intervention variant name: {name}")
            seen_names.add(normalized_name)
        variant_without_name = {key: value for key, value in variant_cfg.items() if key != "name"}
        if variant_without_name:
            validate_train_intervention_config(variant_without_name, epochs_total=int(epochs_total))


def validate_train_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("train config", cfg)
    reject_unknown_keys("train config", cfg, {"dataset", "model", "data_root", "data", "training", "logging", "intervention"})
    require_choice("train.dataset", cfg.get("dataset", "cifar10"), get_supported_image_datasets())
    require_choice("train.model", cfg.get("model", "resnet18"), get_supported_model_names())

    data_cfg = ensure_mapping("train.data", cfg.get("data"))
    train_cfg = ensure_mapping("train.training", cfg.get("training"))
    log_cfg = ensure_mapping("train.logging", cfg.get("logging"))
    intervention_cfg = ensure_mapping("train.intervention", cfg.get("intervention"))

    reject_unknown_keys("train.data", data_cfg, {"val_size", "split_seed", "num_workers", "pin_memory"})
    reject_unknown_keys("train.training", train_cfg, {"optimizer", "epochs", "batch_size", "learning_rate", "momentum", "weight_decay", "nesterov", "scheduler"})
    reject_unknown_keys("train.logging", log_cfg, {"save_every_epochs", "save_final", "save_best"})

    coerce_int("train.data.val_size", data_cfg.get("val_size", 5000), min_value=0)
    coerce_int("train.data.split_seed", data_cfg.get("split_seed", 0))
    coerce_int("train.data.num_workers", data_cfg.get("num_workers", 0), min_value=0)
    require_bool("train.data.pin_memory", data_cfg.get("pin_memory", False))
    require_choice("train.training.optimizer", train_cfg.get("optimizer", "sgd"), {"sgd"})
    coerce_int("train.training.epochs", train_cfg.get("epochs"), min_value=1)
    coerce_int("train.training.batch_size", train_cfg.get("batch_size"), min_value=1)
    coerce_float("train.training.learning_rate", train_cfg.get("learning_rate"), positive=True)
    coerce_float("train.training.momentum", train_cfg.get("momentum", 0.0), min_value=0.0)
    coerce_float("train.training.weight_decay", train_cfg.get("weight_decay", 0.0), min_value=0.0)
    require_bool("train.training.nesterov", train_cfg.get("nesterov", True))
    require_choice("train.training.scheduler", train_cfg.get("scheduler", "none"), {"none", "cosine"})
    coerce_int("train.logging.save_every_epochs", log_cfg.get("save_every_epochs", 0), min_value=0)
    require_bool("train.logging.save_final", log_cfg.get("save_final", True))
    require_bool("train.logging.save_best", log_cfg.get("save_best", True))
    validate_train_intervention_config(intervention_cfg, epochs_total=int(train_cfg.get("epochs")))
