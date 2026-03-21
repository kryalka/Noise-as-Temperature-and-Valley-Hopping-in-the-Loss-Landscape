from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import load_yaml


def ensure_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def default_grid_out_root(grid_config: str | Path) -> str:
    name = Path(grid_config).name
    if name == "lr_bs_grid.yaml":
        return "outputs/runs_lr_bs_grid"
    if name == "lr_bs_grid_cifar100.yaml":
        return "outputs/runs_lr_bs_grid_cifar100"
    if name == "intervention_lr_bs_grid.yaml":
        return "outputs/runs_lr_bs_grid_intervention"
    if name == "intervention_lr_bs_grid_cifar100.yaml":
        return "outputs/runs_lr_bs_grid_intervention_cifar100"
    return "outputs/runs_lr_bs_grid"


def resolve_grid_out_root(grid_config: str | Path, project_root: Path) -> Path:
    grid_config_path = Path(grid_config)
    if not grid_config_path.is_absolute():
        grid_config_path = (project_root / grid_config_path).resolve()
    grid_cfg = load_yaml(grid_config_path)
    out_root = str(grid_cfg.get("out_root", default_grid_out_root(grid_config_path)))
    out_root_path = Path(out_root)
    if not out_root_path.is_absolute():
        out_root_path = (project_root / out_root_path).resolve()
    return out_root_path


def resolve_runs_root(
    section_cfg: dict[str, Any],
    *,
    grid_config: str | Path,
    project_root: Path,
) -> Path:
    override = section_cfg.get("runs_root")
    if override not in (None, ""):
        runs_root = Path(str(override))
        if not runs_root.is_absolute():
            runs_root = (project_root / runs_root).resolve()
        return runs_root
    return resolve_grid_out_root(grid_config, project_root)


def resolve_pairs_outputs(
    baseline_cfg: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[Path, Path, str, list[Any], list[Any]]:
    pairs_cfg = ensure_mapping("baseline.trajectory_pairs", baseline_cfg.get("trajectory_pairs"))
    pairs_csv = Path(str(pairs_cfg.get("out_csv", "outputs/summaries/trajectory_pairs.csv")))
    pairs_json = Path(str(pairs_cfg.get("out_json", "outputs/summaries/trajectory_pairs_summary.json")))
    if not pairs_csv.is_absolute():
        pairs_csv = (project_root / pairs_csv).resolve()
    if not pairs_json.is_absolute():
        pairs_json = (project_root / pairs_json).resolve()
    return (
        pairs_csv,
        pairs_json,
        str(pairs_cfg.get("pair_mode", "milestones")),
        list(pairs_cfg.get("milestone_epochs", [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])),
        list(pairs_cfg.get("explicit_pairs", [])),
    )


def resolve_path_outputs(
    baseline_cfg: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[str, str, str, Path, Path, Path]:
    interpolation_cfg = str(baseline_cfg.get("interpolation_config", "configs/eval/interpolation.yaml"))
    barrier_cfg = str(baseline_cfg.get("barrier_config", "configs/eval/barrier.yaml"))
    path_compare_cfg = str(baseline_cfg.get("path_compare_config", "configs/eval/path_compare.yaml"))
    interpolation_out = Path(str(baseline_cfg.get("interpolation_out", "outputs/artifacts/interpolation_trajectory")))
    barrier_out = Path(str(baseline_cfg.get("barrier_out", "outputs/artifacts/barrier_trajectory")))
    path_compare_out = Path(str(baseline_cfg.get("path_compare_out", "outputs/artifacts/path_compare")))
    if not interpolation_out.is_absolute():
        interpolation_out = (project_root / interpolation_out).resolve()
    if not barrier_out.is_absolute():
        barrier_out = (project_root / barrier_out).resolve()
    if not path_compare_out.is_absolute():
        path_compare_out = (project_root / path_compare_out).resolve()
    return interpolation_cfg, barrier_cfg, path_compare_cfg, interpolation_out, barrier_out, path_compare_out


def resolve_results_outputs(
    intervention_cfg: dict[str, Any],
    results_cfg: dict[str, Any],
    final_outputs_cfg: dict[str, Any],
    figure_outputs_cfg: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[str, Path, Path, Path, Path]:
    geometry_cfg = str(intervention_cfg.get("geometry_config", "configs/eval/geometry.yaml"))
    geometry_out = Path(str(intervention_cfg.get("geometry_out", "outputs/artifacts/geometry_intervention")))
    results_out = Path(str(results_cfg.get("out_root", "outputs/summaries/results_pipeline")))
    final_outputs_out = Path(str(final_outputs_cfg.get("out_root", "outputs/summaries/final_outputs")))
    figure_outputs_out = Path(str(figure_outputs_cfg.get("out_root", "outputs/summaries/figure_outputs")))
    if not geometry_out.is_absolute():
        geometry_out = (project_root / geometry_out).resolve()
    if not results_out.is_absolute():
        results_out = (project_root / results_out).resolve()
    if not final_outputs_out.is_absolute():
        final_outputs_out = (project_root / final_outputs_out).resolve()
    if not figure_outputs_out.is_absolute():
        figure_outputs_out = (project_root / figure_outputs_out).resolve()
    return geometry_cfg, geometry_out, results_out, final_outputs_out, figure_outputs_out
