from __future__ import annotations

from typing import Any

from ._config_core import (
    coerce_float,
    coerce_int,
    ensure_mapping,
    reject_alias_conflict,
    reject_unknown_keys,
    require_bool,
    require_choice,
)




def validate_interpolation_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("interpolation config", cfg)
    reject_unknown_keys(
        "interpolation config",
        cfg,
        {
            "data_root",
            "path",
            "evaluation",
            "data",
            "metrics",
            "num_points",
            "bn_recalib_batches",
            "batch_size",
            "split",
        },
    )

    path_cfg = ensure_mapping("interpolation.path", cfg.get("path"))
    eval_cfg = ensure_mapping("interpolation.evaluation", cfg.get("evaluation"))
    data_cfg = ensure_mapping("interpolation.data", cfg.get("data"))

    reject_unknown_keys("interpolation.path", path_cfg, {"type", "num_points", "bn_recalib_batches", "pivots", "observed"})
    reject_unknown_keys("interpolation.evaluation", eval_cfg, {"model_mode", "batch_size", "bn_batch_size", "split", "val_size", "split_seed"})
    reject_unknown_keys("interpolation.data", data_cfg, {"num_workers", "pin_memory"})

    observed_cfg = ensure_mapping("interpolation.path.observed", path_cfg.get("observed"))
    reject_unknown_keys("interpolation.path.observed", observed_cfg, {"selection", "milestone_epochs", "epochs"})

    reject_alias_conflict(cfg=cfg, section=path_cfg, legacy_key="num_points", nested_key="num_points", section_name="path")
    reject_alias_conflict(cfg=cfg, section=path_cfg, legacy_key="bn_recalib_batches", nested_key="bn_recalib_batches", section_name="path")
    reject_alias_conflict(cfg=cfg, section=eval_cfg, legacy_key="batch_size", nested_key="batch_size", section_name="evaluation")
    reject_alias_conflict(cfg=cfg, section=eval_cfg, legacy_key="split", nested_key="split", section_name="evaluation")

    path_type = require_choice(
        "interpolation.path.type",
        path_cfg.get("type", "linear"),
        {"linear", "polyline", "piecewise", "piecewise_linear", "observed"},
    )
    coerce_int("interpolation.path.num_points", path_cfg.get("num_points", cfg.get("num_points", 41)), min_value=2)
    coerce_int("interpolation.path.bn_recalib_batches", path_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0)), min_value=0)

    pivots = path_cfg.get("pivots", [])
    if pivots is None:
        pivots = []
    if not isinstance(pivots, (list, tuple)):
        raise ValueError("interpolation.path.pivots must be a list")

    if path_type == "observed" or observed_cfg:
        selection = require_choice(
            "interpolation.path.observed.selection",
            observed_cfg.get("selection", "all"),
            {"all", "milestones", "explicit"},
        )

        milestone_epochs = observed_cfg.get("milestone_epochs", [])
        if milestone_epochs is None:
            milestone_epochs = []
        if not isinstance(milestone_epochs, (list, tuple)):
            raise ValueError("interpolation.path.observed.milestone_epochs must be a list")
        milestone_epochs = [
            coerce_int(f"interpolation.path.observed.milestone_epochs[{idx}]", value, min_value=1)
            for idx, value in enumerate(milestone_epochs)
        ]
        if sorted(set(milestone_epochs)) != milestone_epochs:
            raise ValueError("interpolation.path.observed.milestone_epochs must be strictly increasing and unique")

        epochs = observed_cfg.get("epochs", [])
        if epochs is None:
            epochs = []
        if not isinstance(epochs, (list, tuple)):
            raise ValueError("interpolation.path.observed.epochs must be a list")
        epochs = [
            coerce_int(f"interpolation.path.observed.epochs[{idx}]", value, min_value=1)
            for idx, value in enumerate(epochs)
        ]
        if sorted(set(epochs)) != epochs:
            raise ValueError("interpolation.path.observed.epochs must be strictly increasing and unique")
        if selection == "explicit" and len(epochs) == 0:
            raise ValueError("interpolation.path.observed.selection='explicit' requires a non-empty epochs list")

    model_mode = str(eval_cfg.get("model_mode", "eval")).strip().lower()
    if model_mode != "eval":
        raise ValueError("interpolation.evaluation.model_mode is a legacy no-op field and only 'eval' is accepted")

    metrics = cfg.get("metrics", None)
    if metrics is not None:
        if not isinstance(metrics, (list, tuple)):
            raise ValueError("interpolation.metrics must be a list")
        allowed_metrics = {"val_loss", "val_acc", "val_accuracy"}
        bad_metrics = [str(metric) for metric in metrics if str(metric) not in allowed_metrics]
        if bad_metrics:
            raise ValueError(
                "interpolation.metrics is a legacy no-op field; allowed legacy values are "
                f"{sorted(allowed_metrics)}, got invalid entries: {bad_metrics}"
            )

    require_choice("interpolation.evaluation.split", eval_cfg.get("split", cfg.get("split", "val")), {"val", "test"})
    coerce_int("interpolation.evaluation.batch_size", eval_cfg.get("batch_size", cfg.get("batch_size", 256)), min_value=1)
    coerce_int("interpolation.evaluation.bn_batch_size", eval_cfg.get("bn_batch_size", eval_cfg.get("batch_size", cfg.get("batch_size", 256))), min_value=1)
    coerce_int("interpolation.evaluation.val_size", eval_cfg.get("val_size", 5000), min_value=0)
    coerce_int("interpolation.evaluation.split_seed", eval_cfg.get("split_seed", 0))
    coerce_int("interpolation.data.num_workers", data_cfg.get("num_workers", 0), min_value=0)
    require_bool("interpolation.data.pin_memory", data_cfg.get("pin_memory", True))



def validate_geometry_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("geometry config", cfg)
    reject_unknown_keys(
        "geometry config",
        cfg,
        {
            "data_root",
            "geometry",
            "evaluation",
            "data",
            "alpha",
            "num_directions",
            "eval_batch_size",
            "num_eval_batches",
            "bn_recalib_batches",
        },
    )

    geometry_cfg = ensure_mapping("geometry.geometry", cfg.get("geometry"))
    eval_cfg = ensure_mapping("geometry.evaluation", cfg.get("evaluation"))
    data_cfg = ensure_mapping("geometry.data", cfg.get("data"))

    reject_unknown_keys("geometry.geometry", geometry_cfg, {"alpha", "num_directions", "eval_batch_size", "num_eval_batches", "bn_recalib_batches"})
    reject_unknown_keys("geometry.evaluation", eval_cfg, {"val_size", "split_seed"})
    reject_unknown_keys("geometry.data", data_cfg, {"num_workers", "pin_memory"})

    for key in ("alpha", "num_directions", "eval_batch_size", "num_eval_batches", "bn_recalib_batches"):
        reject_alias_conflict(cfg=cfg, section=geometry_cfg, legacy_key=key, nested_key=key, section_name="geometry")

    coerce_float("geometry.alpha", geometry_cfg.get("alpha", cfg.get("alpha", 1e-3)), positive=True)
    coerce_int("geometry.num_directions", geometry_cfg.get("num_directions", cfg.get("num_directions", 10)), min_value=1)
    coerce_int("geometry.eval_batch_size", geometry_cfg.get("eval_batch_size", cfg.get("eval_batch_size", 256)), min_value=1)

    num_eval_batches = geometry_cfg.get("num_eval_batches", cfg.get("num_eval_batches", None))
    if num_eval_batches is not None:
        coerce_int("geometry.num_eval_batches", num_eval_batches, min_value=1)

    coerce_int("geometry.bn_recalib_batches", geometry_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0)), min_value=0)
    coerce_int("geometry.evaluation.val_size", eval_cfg.get("val_size", 5000), min_value=0)
    coerce_int("geometry.evaluation.split_seed", eval_cfg.get("split_seed", 0))
    coerce_int("geometry.data.num_workers", data_cfg.get("num_workers", 0), min_value=0)
    require_bool("geometry.data.pin_memory", data_cfg.get("pin_memory", True))



def validate_barrier_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("barrier config", cfg)
    reject_unknown_keys("barrier config", cfg, {"barrier"})

    barrier_cfg = ensure_mapping("barrier.barrier", cfg.get("barrier"))
    reject_unknown_keys("barrier.barrier", barrier_cfg, {"definition", "thresholds"})

    require_choice(
        "barrier.definition",
        barrier_cfg.get("definition", "max_minus_endpoints"),
        {
            "max_loss_minus_endpoints",
            "max_minus_endpoints",
            "endpoints",
            "max_minus_linear_baseline",
            "max_loss_minus_linear_baseline",
            "max_minus_linear",
            "linear",
        },
    )

    thresholds = barrier_cfg.get("thresholds", [0.005, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5])
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) == 0:
        raise ValueError("barrier.thresholds must be a non-empty list")
    for idx, value in enumerate(thresholds):
        coerce_float(f"barrier.thresholds[{idx}]", value, min_value=0.0)



def validate_path_compare_config(cfg: dict[str, Any] | None) -> None:
    cfg = ensure_mapping("path-compare config", cfg)
    reject_unknown_keys(
        "path-compare config",
        cfg,
        {
            "data_root",
            "path",
            "evaluation",
            "data",
            "metrics",
            "num_points",
            "bn_recalib_batches",
            "batch_size",
            "split",
            "barrier",
        },
    )

    interp_cfg = {key: value for key, value in cfg.items() if key != "barrier"}
    validate_interpolation_config(interp_cfg)
    validate_barrier_config({"barrier": cfg.get("barrier", {})})
