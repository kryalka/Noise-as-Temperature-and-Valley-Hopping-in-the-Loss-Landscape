from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._diagnostic_sections import (
    build_regime_maps,
    build_regime_rows,
    build_report_markdown,
    write_csv,
)


PAIR_COLUMNS = [
    "comparison_json",
    "status",
    "reason",
    "run_name",
    "pair_tag",
    "dataset",
    "model",
    "seed",
    "learning_rate",
    "batch_size",
    "epoch_A",
    "epoch_B",
    "ckptA",
    "ckptB",
    "observed_selection",
    "eval_split",
    "chord_DeltaL",
    "observed_DeltaL",
    "Peakobs",
    "Pitchord",
    "Pitobs",
    "BarrierGap",
    "devL1",
    "LengthRatio",
    "LengthExcess",
    "chord_length",
    "observed_length",
    "geometry_A_json",
    "geometry_B_json",
    "curvature_proxy_A",
    "curvature_proxy_B",
    "curvature_proxy_mean",
    "sigma_kappa_A",
    "sigma_kappa_B",
    "anisotropy_A",
    "anisotropy_B",
]

REGIME_COLUMNS = [
    "dataset",
    "model",
    "learning_rate",
    "batch_size",
    "num_pairs",
    "num_ok_pairs",
    "num_geometry_partial_pairs",
    "num_compare_failed_pairs",
    "observed_selection_modes",
    "mean_chord_DeltaL",
    "mean_observed_DeltaL",
    "mean_Peakobs",
    "mean_Pitchord",
    "mean_Pitobs",
    "mean_BarrierGap",
    "mean_devL1",
    "mean_LengthRatio",
    "mean_LengthExcess",
    "mean_curvature_proxy_A",
    "mean_curvature_proxy_B",
    "mean_curvature_proxy_mean",
    "mean_sigma_kappa_A",
    "mean_sigma_kappa_B",
    "mean_anisotropy_A",
    "mean_anisotropy_B",
]

MAP_METRICS = [
    "mean_Peakobs",
    "mean_Pitchord",
    "mean_Pitobs",
    "mean_BarrierGap",
    "mean_devL1",
    "mean_curvature_proxy_mean",
]



def write_diagnostic_outputs(
    *,
    rows: list[dict[str, Any]],
    out_root: str | Path,
    config_path: str | Path,
    input_meta: dict[str, Any],
    issues: list[str],
    unique_checkpoint_count: int,
) -> dict[str, Any]:
    out_root = ensure_dir(out_root)

    pair_csv = out_root / "diagnostic_pairs.csv"
    regime_csv = out_root / "diagnostic_regime_table.csv"
    maps_json = out_root / "diagnostic_regime_maps.json"
    report_json = out_root / "diagnostic_report.json"
    report_md = out_root / "diagnostic_report.md"
    manifest_json = out_root / "diagnostic_manifest.json"

    write_csv(pair_csv, PAIR_COLUMNS, rows)
    regime_rows = build_regime_rows(rows)
    write_csv(regime_csv, REGIME_COLUMNS, regime_rows)

    regime_maps = {
        "metric_maps": build_regime_maps(regime_rows, MAP_METRICS),
        "num_regimes": int(len(regime_rows)),
        "notes": [
            "regime maps are aggregated directly from diagnostic pair rows",
            "curvature_proxy is an alias for geometry kappa_tr in this diagnostic layer",
        ],
    }
    save_json(maps_json, regime_maps)

    report = {
        "kind": "diagnostic_toolkit_report",
        "config_path": str(config_path),
        "inputs": dict(input_meta),
        "tool_scope": {
            "purpose": "checkpoint trajectory diagnostics from pair inputs to report-ready summaries",
            "reference_presets": "cifar and resnet configs are reference scenarios, not the only intended use",
            "minimum_input_contract": [
                "ckptA and ckptB paths for each pair",
                "compatible compare_config",
                "compatible geometry_config",
            ],
        },
        "counts": {
            "num_pairs_input": int(len(rows)),
            "num_pair_rows": int(len(rows)),
            "num_ok_pairs": int(sum(1 for row in rows if str(row.get("status", "")) == "ok")),
            "num_geometry_partial_pairs": int(sum(1 for row in rows if str(row.get("status", "")) == "geometry_partial")),
            "num_compare_failed_pairs": int(sum(1 for row in rows if str(row.get("status", "")) == "compare_failed")),
            "num_unique_checkpoints": int(unique_checkpoint_count),
            "num_regimes": int(len(regime_rows)),
        },
        "issues": list(issues),
        "outputs": {
            "diagnostic_pairs_csv": str(pair_csv),
            "diagnostic_regime_table_csv": str(regime_csv),
            "diagnostic_regime_maps_json": str(maps_json),
            "diagnostic_report_md": str(report_md),
        },
    }
    save_json(report_json, report)
    report_md.write_text(
        build_report_markdown(
            report=report,
            pair_csv=pair_csv,
            regime_csv=regime_csv,
            maps_json=maps_json,
        ),
        encoding="utf-8",
    )

    manifest = {
        "kind": "diagnostic_pipeline",
        "config_path": str(config_path),
        "tool_scope": report["tool_scope"],
        "outputs": report["outputs"],
        "report_json": str(report_json),
        "issues": list(issues),
    }
    save_json(manifest_json, manifest)
    return {
        "pair_csv": pair_csv,
        "regime_csv": regime_csv,
        "maps_json": maps_json,
        "report_json": report_json,
        "report_md": report_md,
        "manifest_json": manifest_json,
        "report": report,
    }
