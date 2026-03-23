from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ntempvh.pipeline.report_flow import run_report_flow

from ._report_flow_support import (
    make_fake_stage_runner,
    make_report_flow_paths,
    write_grid_stub,
    write_report_flow_config,
)



def test_run_report_flow_writes_stage_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.pipeline.report_flow as report_mod

    baseline_grid = tmp_path / "baseline_grid.yaml"
    baseline_runs = tmp_path / "baseline_runs"
    intervention_grid = tmp_path / "intervention_grid.yaml"
    intervention_runs = tmp_path / "intervention_runs"
    paths = make_report_flow_paths(tmp_path)
    pipeline_cfg = tmp_path / "report_flow.yaml"

    write_grid_stub(baseline_grid, out_root=baseline_runs)
    write_grid_stub(intervention_grid, out_root=intervention_runs)
    write_report_flow_config(
        pipeline_cfg,
        paths=paths,
        baseline_grid=baseline_grid,
        intervention_grid=intervention_grid,
        include_results=True,
    )
    monkeypatch.setattr(
        report_mod,
        "_run_stage_command",
        make_fake_stage_runner(
            baseline_grid=baseline_grid,
            baseline_runs=baseline_runs,
            intervention_grid=intervention_grid,
            intervention_runs=intervention_runs,
            paths=paths,
        ),
    )

    manifest_path = run_report_flow(str(pipeline_cfg), python_bin="python3.11")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = (paths.pipeline_out / "report_flow_steps.csv").read_text(encoding="utf-8").strip().splitlines()

    assert manifest["status"] == "ok"
    assert manifest["failed_stage"] is None
    assert len(manifest["stage_rows"]) == 8
    assert all(row["status"] == "ok" for row in manifest["stage_rows"])
    assert len(lines) == 9



@pytest.mark.parametrize(
    ("include_final_outputs", "include_figure_outputs", "expected_stage", "expected_count"),
    [
        (True, False, "final_outputs", 9),
        (True, True, "figure_outputs", 10),
    ],
)
def test_run_report_flow_supports_optional_output_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_final_outputs: bool,
    include_figure_outputs: bool,
    expected_stage: str,
    expected_count: int,
) -> None:
    import ntempvh.pipeline.report_flow as report_mod

    baseline_grid = tmp_path / "baseline_grid.yaml"
    baseline_runs = tmp_path / "baseline_runs"
    intervention_grid = tmp_path / "intervention_grid.yaml"
    intervention_runs = tmp_path / "intervention_runs"
    paths = make_report_flow_paths(tmp_path)
    pipeline_cfg = tmp_path / "report_flow_optional.yaml"

    write_grid_stub(baseline_grid, out_root=baseline_runs)
    write_grid_stub(intervention_grid, out_root=intervention_runs)
    write_report_flow_config(
        pipeline_cfg,
        paths=paths,
        baseline_grid=baseline_grid,
        intervention_grid=intervention_grid,
        include_results=True,
        include_final_outputs=include_final_outputs,
        include_figure_outputs=include_figure_outputs,
    )
    monkeypatch.setattr(
        report_mod,
        "_run_stage_command",
        make_fake_stage_runner(
            baseline_grid=baseline_grid,
            baseline_runs=baseline_runs,
            intervention_grid=intervention_grid,
            intervention_runs=intervention_runs,
            paths=paths,
        ),
    )

    manifest_path = run_report_flow(str(pipeline_cfg), python_bin="python3.11")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert len(manifest["stage_rows"]) == expected_count
    assert manifest["stage_rows"][-1]["stage"] == expected_stage
    assert manifest["stage_rows"][-1]["status"] == "ok"



def test_run_report_flow_writes_partial_manifest_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.pipeline.report_flow as report_mod

    baseline_grid = tmp_path / "baseline_grid.yaml"
    baseline_runs = tmp_path / "baseline_runs"
    paths = make_report_flow_paths(tmp_path)
    pipeline_cfg = tmp_path / "report_flow_fail.yaml"

    write_grid_stub(baseline_grid, out_root=baseline_runs)
    write_report_flow_config(
        pipeline_cfg,
        paths=paths,
        baseline_grid=baseline_grid,
        intervention_grid=None,
        include_results=False,
        intervention_enabled=False,
    )

    def fake_run_stage(command: list[str], *, cwd: Path, python_bin: str):
        del cwd, python_bin
        cmd_text = " ".join(command)
        if "run_lr_bs_grid.sh" in cmd_text:
            baseline_runs.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        if "ntempvh.pipeline.trajectory_pairs" in cmd_text:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(report_mod, "_run_stage_command", fake_run_stage)

    with pytest.raises(RuntimeError, match="trajectory_pairs"):
        run_report_flow(str(pipeline_cfg), python_bin="python3.11")

    manifest = json.loads((paths.pipeline_out / "report_flow_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_stage"] == "trajectory_pairs"
    assert manifest["stage_rows"][0]["status"] == "ok"
    assert manifest["stage_rows"][1]["status"] == "failed"
