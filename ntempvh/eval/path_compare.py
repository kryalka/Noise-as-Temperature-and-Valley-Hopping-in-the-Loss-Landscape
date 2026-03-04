from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ntempvh.eval._path_compare_helpers import (
    append_summary_row,
    build_interpolation_cfg,
    profile_deviation_metrics,
)
from ntempvh.eval.barrier import compute_barrier_from_config
from ntempvh.eval.interpolation import run_interpolation_from_config
from ntempvh.eval.metrics import state_dict_l2_distance
from ntempvh.utils.path_metrics import (
    compute_linear_baseline_shape_metrics,
)
from ntempvh.utils.artifacts import (
    build_path_compare_artifact_context,
    load_interpolation_metadata,
    load_json_artifact,
)

from ntempvh.utils.config_validation import validate_path_compare_config
from ntempvh.utils.io import ensure_dir, load_yaml, save_json


SUMMARY_COLUMNS = [
    "ckptA",
    "ckptB",
    "pair_tag",
    "observed_selection",
    "num_points",
    "eval_split",
    "chord_interp_csv",
    "observed_interp_csv",
    "chord_barrier_json",
    "observed_barrier_json",
    "chord_DeltaL",
    "observed_DeltaL",
    "barrier_gap",
    "chord_length",
    "observed_length",
    "length_ratio",
    "length_excess",
    "loss_profile_l1_mean",
    "loss_profile_linf",
    "Peakobs",
    "Pitchord",
    "Pitobs",
    "BarrierGap",
    "LengthRatio",
    "LengthExcess",
    "devL1",
    "comparison_json",
]



def compare_paths_from_config(
    ckpt_a: str,
    ckpt_b: str,
    cfg: dict[str, Any],
    out_dir: str,
) -> Path:
    validate_path_compare_config(cfg)
    artifact = build_path_compare_artifact_context(ckpt_a, ckpt_b, cfg)

    out_root = ensure_dir(out_dir)
    interp_dir = ensure_dir(out_root / "interpolation")
    barrier_dir = ensure_dir(out_root / "barrier")
    compare_dir = ensure_dir(out_root / "comparisons")

    raw_data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    raw_data_cfg = dict(raw_data_cfg or {})

    chord_interp_cfg = build_interpolation_cfg(
        artifact["config"],
        raw_data_cfg,
        path_type="linear",
    )
    observed_interp_cfg = build_interpolation_cfg(
        artifact["config"],
        raw_data_cfg,
        path_type="observed",
    )
    barrier_cfg = {"barrier": dict(artifact["config"]["barrier"])}

    chord_interp_csv = run_interpolation_from_config(
        ckpt_a,
        ckpt_b,
        chord_interp_cfg,
        str(interp_dir),
    )
    observed_interp_csv = run_interpolation_from_config(
        ckpt_a,
        ckpt_b,
        observed_interp_cfg,
        str(interp_dir),
    )

    chord_barrier_json = compute_barrier_from_config(
        str(chord_interp_csv),
        barrier_cfg,
        str(barrier_dir),
    )
    observed_barrier_json = compute_barrier_from_config(
        str(observed_interp_csv),
        barrier_cfg,
        str(barrier_dir),
    )

    _, observed_meta = load_interpolation_metadata(observed_interp_csv)
    observed_info = ((observed_meta.get("path") or {}).get("observed") or {})

    chord_barrier = load_json_artifact(chord_barrier_json)
    observed_barrier = load_json_artifact(observed_barrier_json)

    ckpt_a_obj = torch.load(ckpt_a, map_location="cpu")
    ckpt_b_obj = torch.load(ckpt_b, map_location="cpu")
    chord_length = state_dict_l2_distance(
        ckpt_a_obj["state_dict"],
        ckpt_b_obj["state_dict"],
    )
    observed_length = float(observed_info.get("total_path_length", chord_length))

    dev_metrics = profile_deviation_metrics(chord_interp_csv, observed_interp_csv)
    chord_shape_metrics = compute_linear_baseline_shape_metrics(chord_interp_csv)
    observed_shape_metrics = compute_linear_baseline_shape_metrics(observed_interp_csv)

    eps = 1e-12
    metrics = {
        "chord_DeltaL": float(chord_barrier["DeltaL"]),
        "observed_DeltaL": float(observed_barrier["DeltaL"]),
        "barrier_gap": float(observed_barrier["DeltaL"] - chord_barrier["DeltaL"]),
        "chord_length": float(chord_length),
        "observed_length": float(observed_length),
        "length_ratio": float(observed_length / max(chord_length, eps)),
        "length_excess": float(observed_length - chord_length),
        **dev_metrics,
    }
    report_metrics = {
        "Peakobs": float(observed_shape_metrics["peak"]),
        "Pitchord": float(chord_shape_metrics["pit"]),
        "Pitobs": float(observed_shape_metrics["pit"]),
        "BarrierGap": float(metrics["barrier_gap"]),
        "LengthRatio": float(metrics["length_ratio"]),
        "LengthExcess": float(metrics["length_excess"]),
        "devL1": float(metrics["loss_profile_l1_mean"]),
    }

    payload: dict[str, Any] = {
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "pair_tag": artifact["pair_tag"],
        "config": artifact["config"],
        "artifacts": {
            "chord_interp_csv": str(chord_interp_csv),
            "chord_meta_json": str(chord_interp_csv.with_suffix(".meta.json")),
            "chord_barrier_json": str(chord_barrier_json),
            "observed_interp_csv": str(observed_interp_csv),
            "observed_meta_json": str(observed_interp_csv.with_suffix(".meta.json")),
            "observed_barrier_json": str(observed_barrier_json),
            "comparison_json": str(compare_dir / f"{artifact['stem']}.json"),
            "stem": artifact["stem"],
        },
        "metrics": metrics,
        "report_metrics": report_metrics,
        "observed_path": {
            "selection": observed_info.get("selection"),
            "resolved_checkpoints": observed_info.get("resolved_checkpoints"),
            "resolved_epochs": observed_info.get("resolved_epochs"),
            "num_checkpoints": observed_info.get("num_checkpoints"),
            "num_segments": observed_info.get("num_segments"),
            "parameterization": observed_info.get("parameterization"),
            "segment_lengths": observed_info.get("segment_lengths"),
            "segment_endpoints_t": observed_info.get("segment_endpoints_t"),
            "operational_peak_definition": (
                "Observed-path DeltaL is computed by applying the configured barrier/profile "
                "statistic to the normalized observed piecewise path through saved checkpoints."
            ),
        },
        "definitions": {
            "barrier_gap": "observed_DeltaL - chord_DeltaL",
            "length_ratio": "observed_length / chord_length",
            "length_excess": "observed_length - chord_length",
            "loss_profile_l1_mean": (
                "Mean absolute loss-profile deviation after resampling chord and observed "
                "profiles to a shared normalized t grid"
            ),
            "loss_profile_linf": (
                "Maximum absolute loss-profile deviation after resampling chord and observed "
                "profiles to a shared normalized t grid"
            ),
            "Peakobs": (
                "Observed-path peak height above the linear endpoint baseline on the normalized observed path"
            ),
            "Pitchord": (
                "Chord-path pit depth below the linear endpoint baseline on the normalized chord path"
            ),
            "Pitobs": (
                "Observed-path pit depth below the linear endpoint baseline on the normalized observed path"
            ),
            "BarrierGap": "Alias for barrier_gap = observed_DeltaL - chord_DeltaL",
            "LengthRatio": "Alias for length_ratio = observed_length / chord_length",
            "LengthExcess": "Alias for length_excess = observed_length - chord_length",
            "devL1": "Alias for loss_profile_l1_mean",
        },
    }

    compare_json = compare_dir / f"{artifact['stem']}.json"
    save_json(compare_json, payload)

    summary_row = {
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "pair_tag": artifact["pair_tag"],
        "observed_selection": str(observed_info.get("selection", "")),
        "num_points": int(artifact["config"]["path"]["num_points"]),
        "eval_split": str(artifact["config"]["evaluation"]["split"]),
        "chord_interp_csv": str(chord_interp_csv),
        "observed_interp_csv": str(observed_interp_csv),
        "chord_barrier_json": str(chord_barrier_json),
        "observed_barrier_json": str(observed_barrier_json),
        "comparison_json": str(compare_json),
        **metrics,
        **report_metrics,
    }
    append_summary_row(compare_dir / "path_comparisons.csv", SUMMARY_COLUMNS, summary_row)

    return compare_json



def compare_paths(
    ckpt_a: str,
    ckpt_b: str,
    config_path: str,
    out_dir: str,
) -> Path:
    cfg = load_yaml(config_path)
    return compare_paths_from_config(ckpt_a, ckpt_b, cfg, out_dir)
