from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from pathlib import Path





def run_name(*, dataset: str = "cifar10", seed: int = 1, lr: float = 0.1, bs: int = 128) -> str:
    return (
        f"{dataset}_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0.0005_mom0.9_schnone__deadbeef"
    )



@dataclass
class ReportFlowPaths:
    pipeline_out: Path
    pairs_csv: Path
    pairs_json: Path
    interpolation_out: Path
    barrier_out: Path
    path_compare_out: Path
    geometry_out: Path
    results_out: Path
    final_outputs_out: Path
    figure_outputs_out: Path


def write_grid_stub(path: Path, *, out_root: Path) -> None:
    path.write_text(
        json.dumps({"base_config": "ignored", "out_root": str(out_root)}),
        encoding="utf-8",
    )



def write_run_stub(
    *,
    runs_root: Path,
    run_name: str,
    dataset: str = "cifar10",
    epochs: list[int],
) -> Path:
    run_dir = runs_root / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epochs:
        (ckpt_dir / f"epoch_{int(epoch):03d}.pt").write_text("stub", encoding="utf-8")

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "dataset": dataset,
                "model": "resnet18",
                "training": {
                    "optimizer": "sgd",
                    "epochs": max(epochs),
                    "batch_size": 128,
                    "learning_rate": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                    "nesterov": True,
                    "scheduler": "none",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"seed": 1, "epochs": max(epochs)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir



def make_report_flow_paths(tmp_path: Path) -> ReportFlowPaths:
    
    return ReportFlowPaths(
        pipeline_out=tmp_path / "report_flow",
        pairs_csv=tmp_path / "pairs.csv",
        pairs_json=tmp_path / "pairs_summary.json",
        interpolation_out=tmp_path / "interpolation",
        barrier_out=tmp_path / "barrier",
        path_compare_out=tmp_path / "path_compare",
        geometry_out=tmp_path / "geometry",
        results_out=tmp_path / "results_pipeline",
        final_outputs_out=tmp_path / "final_outputs",
        figure_outputs_out=tmp_path / "figure_outputs",
    )



def write_report_flow_config(
    path: Path,
    *,
    paths: ReportFlowPaths,
    baseline_grid: Path | None,
    intervention_grid: Path | None = None,
    baseline_enabled: bool = True,
    include_results: bool = True,
    include_final_outputs: bool = False,
    include_figure_outputs: bool = False,
    intervention_enabled: bool = True,
    baseline_runs_root: Path | None = None,
    intervention_runs_root: Path | None = None,
    reuse_existing_outputs: bool = False,
) -> None:
    lines = [
        "pipeline:",
        f"  out_root: {paths.pipeline_out}",
    ]
    if reuse_existing_outputs:
        lines.append("  reuse_existing_outputs: true")

    lines.extend([
        "",
        "baseline:",
        f"  enabled: {'true' if baseline_enabled else 'false'}",
    ])
    if baseline_grid is not None:
        lines.append(f"  train_grid_config: {baseline_grid}")
    if baseline_runs_root is not None:
        lines.append(f"  runs_root: {baseline_runs_root}")
    lines.extend([
        "  trajectory_pairs:",
        f"    out_csv: {paths.pairs_csv}",
        f"    out_json: {paths.pairs_json}",
        "    pair_mode: milestones",
        "    milestone_epochs: [1, 10]",
        "  interpolation_config: configs/eval/interpolation.yaml",
        f"  interpolation_out: {paths.interpolation_out}",
        "  barrier_config: configs/eval/barrier.yaml",
        f"  barrier_out: {paths.barrier_out}",
        "  path_compare_config: configs/eval/path_compare.yaml",
        f"  path_compare_out: {paths.path_compare_out}",
        "",
        "intervention:",
        f"  enabled: {'true' if intervention_enabled else 'false'}",
    ])

    if intervention_grid is not None:
        lines.append(f"  train_grid_config: {intervention_grid}")
    if intervention_runs_root is not None:
        lines.append(f"  runs_root: {intervention_runs_root}")
    if intervention_enabled or intervention_grid is not None or intervention_runs_root is not None:
        lines.extend(
            [
                "  geometry_config: configs/eval/geometry.yaml",
                f"  geometry_out: {paths.geometry_out}",
            ]
        )

    lines.extend(["", "results:", f"  enabled: {'true' if include_results else 'false'}"])
    if include_results:
        lines.append(f"  out_root: {paths.results_out}")

    if include_final_outputs:
        lines.extend(
            [
                "",
                "final_outputs:",
                "  enabled: true",
                f"  out_root: {paths.final_outputs_out}",
            ]
        )

    if include_figure_outputs:
        lines.extend(
            [
                "",
                "figure_outputs:",
                "  enabled: true",
                f"  out_root: {paths.figure_outputs_out}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def make_fake_stage_runner(
    *,
    baseline_grid: Path,
    baseline_runs: Path,
    intervention_grid: Path | None,
    intervention_runs: Path | None,
    paths: ReportFlowPaths,
):
    
    def fake_run_stage(command: list[str], *, cwd: Path, python_bin: str):
        del cwd, python_bin
        cmd_text = " ".join(command)

        if "run_lr_bs_grid.sh" in cmd_text and str(baseline_grid) in cmd_text:
            baseline_runs.mkdir(parents=True, exist_ok=True)
        elif "ntempvh.pipeline.trajectory_pairs" in cmd_text:
            paths.pairs_csv.write_text("run_dir,run_name\n", encoding="utf-8")
            paths.pairs_json.write_text("{}", encoding="utf-8")
        elif "run_interpolation_grid.sh" in cmd_text:
            paths.interpolation_out.mkdir(parents=True, exist_ok=True)
        elif "run_barrier_grid.sh" in cmd_text:
            paths.barrier_out.mkdir(parents=True, exist_ok=True)
        elif "run_path_compare_grid.sh" in cmd_text:
            (paths.path_compare_out / "comparisons").mkdir(parents=True, exist_ok=True)
        elif (
            intervention_grid is not None
            and intervention_runs is not None
            and "run_lr_bs_grid.sh" in cmd_text
            and str(intervention_grid) in cmd_text
        ):
            intervention_runs.mkdir(parents=True, exist_ok=True)
        elif "run_intervention_geometry_batch.sh" in cmd_text:
            paths.geometry_out.mkdir(parents=True, exist_ok=True)
            (paths.geometry_out / "intervention_geometry_summary.csv").write_text(
                "run_dir,run_name\n",
                encoding="utf-8",
            )
        elif "run_results_pipeline.sh" in cmd_text:
            paths.results_out.mkdir(parents=True, exist_ok=True)
            (paths.results_out / "results_manifest.json").write_text("{}", encoding="utf-8")
        elif "run_final_outputs.sh" in cmd_text:
            paths.final_outputs_out.mkdir(parents=True, exist_ok=True)
            (paths.final_outputs_out / "final_outputs_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
        elif "run_figure_outputs.sh" in cmd_text:
            paths.figure_outputs_out.mkdir(parents=True, exist_ok=True)
            (paths.figure_outputs_out / "figure_outputs_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
        else:
            raise AssertionError(f"Unexpected command: {command}")

        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    return fake_run_stage
