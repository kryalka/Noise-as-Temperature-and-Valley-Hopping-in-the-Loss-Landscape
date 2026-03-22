from __future__ import annotations

from pathlib import Path
from typing import Any

from ._report_flow_paths import (
    default_grid_out_root,
    ensure_mapping,
    resolve_grid_out_root,
    resolve_pairs_outputs,
    resolve_path_outputs,
    resolve_results_outputs,
    resolve_runs_root,
)
from ._report_flow_stages import build_base_stages


def build_stage_plan(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    python_bin: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    pipeline_cfg = ensure_mapping("pipeline", cfg.get("pipeline"))
    baseline_cfg = ensure_mapping("baseline", cfg.get("baseline"))
    intervention_cfg = ensure_mapping("intervention", cfg.get("intervention"))
    results_cfg = ensure_mapping("results", cfg.get("results"))
    has_final_outputs_section = "final_outputs" in cfg
    final_outputs_cfg = ensure_mapping("final_outputs", cfg.get("final_outputs"))
    has_figure_outputs_section = "figure_outputs" in cfg
    figure_outputs_cfg = ensure_mapping("figure_outputs", cfg.get("figure_outputs"))

    baseline_enabled = bool(baseline_cfg.get("enabled", True))
    intervention_enabled = bool(intervention_cfg.get("enabled", True))
    results_enabled = bool(results_cfg.get("enabled", True))
    final_outputs_enabled = bool(final_outputs_cfg.get("enabled", False))
    figure_outputs_enabled = bool(figure_outputs_cfg.get("enabled", False))
    reuse_existing_outputs = bool(pipeline_cfg.get("reuse_existing_outputs", False))

    baseline_grid = str(baseline_cfg.get("train_grid_config", "configs/train/lr_bs_grid.yaml"))
    intervention_grid = str(intervention_cfg.get("train_grid_config", "configs/train/intervention_lr_bs_grid.yaml"))

    
    baseline_runs_root = resolve_runs_root(baseline_cfg, grid_config=baseline_grid, project_root=project_root)

    
    if intervention_enabled or results_enabled:
        intervention_runs_root = resolve_runs_root(
            intervention_cfg,
            grid_config=intervention_grid,
            project_root=project_root,
        )
    else:
        override = intervention_cfg.get("runs_root")
        if override not in (None, ""):
            intervention_runs_root = Path(str(override))
            if not intervention_runs_root.is_absolute():
                intervention_runs_root = (project_root / intervention_runs_root).resolve()
        else:
            intervention_runs_root = (project_root / default_grid_out_root(intervention_grid)).resolve()

    pairs_csv, pairs_json, pair_mode, milestone_epochs, explicit_pairs = resolve_pairs_outputs(
        baseline_cfg,
        project_root=project_root,
    )
    interpolation_cfg, barrier_cfg, path_compare_cfg, interpolation_out, barrier_out, path_compare_out = resolve_path_outputs(
        baseline_cfg,
        project_root=project_root,
    )
    geometry_cfg, geometry_out, results_out, final_outputs_out, figure_outputs_out = resolve_results_outputs(
        intervention_cfg,
        results_cfg,
        final_outputs_cfg,
        figure_outputs_cfg,
        project_root=project_root,
    )

    geometry_summary_csv = geometry_out / "intervention_geometry_summary.csv"
    results_manifest = results_out / "results_manifest.json"
    final_outputs_manifest = final_outputs_out / "final_outputs_manifest.json"
    figure_outputs_manifest = figure_outputs_out / "figure_outputs_manifest.json"

    stages = build_base_stages(
        project_root=project_root,
        python_bin=python_bin,
        baseline_enabled=baseline_enabled,
        intervention_enabled=intervention_enabled,
        results_enabled=results_enabled,
        baseline_grid=baseline_grid,
        intervention_grid=intervention_grid,
        baseline_runs_root=baseline_runs_root,
        intervention_runs_root=intervention_runs_root,
        pairs_csv=pairs_csv,
        pairs_json=pairs_json,
        pair_mode=pair_mode,
        milestone_epochs=milestone_epochs,
        explicit_pairs=explicit_pairs,
        interpolation_cfg=interpolation_cfg,
        barrier_cfg=barrier_cfg,
        path_compare_cfg=path_compare_cfg,
        interpolation_out=interpolation_out,
        barrier_out=barrier_out,
        path_compare_out=path_compare_out,
        geometry_cfg=geometry_cfg,
        geometry_out=geometry_out,
        geometry_summary_csv=geometry_summary_csv,
        results_out=results_out,
        results_manifest=results_manifest,
    )

    if has_final_outputs_section:
        stages.append({
            "stage": "final_outputs",
            "enabled": final_outputs_enabled,
            "command": [
                "bash",
                str((project_root / "experiments/run_final_outputs.sh").resolve()),
                str(results_out),
                str(final_outputs_out),
            ],
            "primary_output": str(final_outputs_manifest),
            "note": "Build final report-ready tables and compact summary maps",
        })

    if has_figure_outputs_section:
        stages.append({
            "stage": "figure_outputs",
            "enabled": figure_outputs_enabled,
            "command": [
                "bash",
                str((project_root / "experiments/run_figure_outputs.sh").resolve()),
                str(final_outputs_out),
                str(figure_outputs_out),
            ],
            "primary_output": str(figure_outputs_manifest),
            "note": "Build reproducible figure-ready SVG outputs from final tables",
        })

    for stage in stages:
        if stage["stage"] == "trajectory_pairs" and "--milestone_epochs" in stage["command"] and len(milestone_epochs) == 0:
            idx = stage["command"].index("--milestone_epochs")
            stage["command"] = stage["command"][:idx]
        if stage["stage"] == "trajectory_pairs" and "--explicit_pairs" in stage["command"] and len(explicit_pairs) == 0:
            idx = stage["command"].index("--explicit_pairs")
            stage["command"] = stage["command"][:idx]
        stage["skip_if_output_exists"] = reuse_existing_outputs

    outputs = {
        "baseline_runs_root": str(baseline_runs_root),
        "trajectory_pairs_csv": str(pairs_csv),
        "trajectory_pairs_summary_json": str(pairs_json),
        "interpolation_out": str(interpolation_out),
        "barrier_out": str(barrier_out),
        "path_compare_out": str(path_compare_out),
        "intervention_runs_root": str(intervention_runs_root),
        "intervention_geometry_summary_csv": str(geometry_summary_csv),
        "results_out": str(results_out),
        "results_manifest": str(results_manifest),
        "final_outputs_out": str(final_outputs_out),
        "final_outputs_manifest": str(final_outputs_manifest),
        "figure_outputs_out": str(figure_outputs_out),
        "figure_outputs_manifest": str(figure_outputs_manifest),
        "reuse_existing_outputs": str(reuse_existing_outputs),
    }
    return stages, outputs
