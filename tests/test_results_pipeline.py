from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.results.pipeline import (
    COMPARE_RESULTS_COLUMNS,
    INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
    INTERVENTION_RUN_RESULTS_COLUMNS,
    PATH_QUALITY_LINK_COLUMNS,
    run_results_pipeline,
)

from ._results_support import (
    run_name,
    write_compare_artifact,
    write_intervention_geometry_summary,
    write_intervention_run,
)



def test_run_results_pipeline_empty_inputs_write_schema_stable_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    out_dir = tmp_path / "results"

    manifest_path = run_results_pipeline(
        runs_root=str(runs_root),
        path_compare_root=str(tmp_path / "missing_path_compare"),
        intervention_geometry_summary_csv=str(tmp_path / "missing_geometry.csv"),
        out_dir=str(out_dir),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["compare_rows"] == 0
    assert manifest["counts"]["intervention_run_rows"] == 0
    assert manifest["counts"]["intervention_geometry_run_rows"] == 0
    assert manifest["counts"]["path_quality_link_rows"] == 0

    assert (out_dir / "compare_paths_results.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_RESULTS_COLUMNS)
    assert (out_dir / "intervention_runs_results.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_RUN_RESULTS_COLUMNS)
    assert (out_dir / "intervention_geometry_runs_results.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS)
    assert (out_dir / "path_quality_links.csv").read_text(encoding="utf-8").strip() == ",".join(PATH_QUALITY_LINK_COLUMNS)



def test_run_results_pipeline_smoke(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    name = run_name(seed=1, lr=0.2, bs=8)
    run_dir = write_intervention_run(root=runs_root, run_name=name)

    path_compare_root = tmp_path / "path_compare"
    write_compare_artifact(root=path_compare_root, run_dir=run_dir, with_endpoint_meta=True)

    geometry_summary_csv = tmp_path / "geometry_intervention" / "intervention_geometry_summary.csv"
    write_intervention_geometry_summary(geometry_summary_csv, run_dir=run_dir, run_name=name)

    out_dir = tmp_path / "results"
    manifest_path = run_results_pipeline(
        runs_root=str(runs_root),
        path_compare_root=str(path_compare_root),
        intervention_geometry_summary_csv=str(geometry_summary_csv),
        out_dir=str(out_dir),
    )

    assert manifest_path.exists()

    compare_rows = list(csv.DictReader((out_dir / "compare_paths_results.csv").open(encoding="utf-8")))
    assert len(compare_rows) == 1
    assert compare_rows[0]["run_name"] == name
    assert compare_rows[0]["quality_signal_scope"] == "test_and_validation"
    assert compare_rows[0]["final_test_acc"] == "0.83"
    assert compare_rows[0]["train_test_gap"] == "0.07"
    assert compare_rows[0]["Peakobs"] == "0.0"
    assert compare_rows[0]["BarrierGap"] == "-0.05"
    assert compare_rows[0]["devL1"] == "0.02"

    intervention_rows = list(csv.DictReader((out_dir / "intervention_runs_results.csv").open(encoding="utf-8")))
    assert len(intervention_rows) == 1
    assert intervention_rows[0]["status"] == "ok"
    assert intervention_rows[0]["num_intervention_metric_rows"] == "2"

    geometry_rows = list(csv.DictReader((out_dir / "intervention_geometry_runs_results.csv").open(encoding="utf-8")))
    assert len(geometry_rows) == 1
    assert geometry_rows[0]["status"] == "ok"
    assert geometry_rows[0]["delta_kappa_post_minus_pre"] == "0.5"
    assert geometry_rows[0]["delta_kappa_final_minus_pre"] == "1.0"
    assert geometry_rows[0]["pre_sigma_kappa"] == "0.1"
    assert geometry_rows[0]["final_anisotropy"] == "0.1"

    link_rows = list(csv.DictReader((out_dir / "path_quality_links.csv").open(encoding="utf-8")))
    assert len(link_rows) == 1
    assert link_rows[0]["final_val_acc"] == "0.85"
    assert link_rows[0]["final_test_acc"] == "0.83"
    assert link_rows[0]["train_test_gap"] == "0.07"
    assert link_rows[0]["geometry_pre_kappa_tr"] == "1.0"
    assert link_rows[0]["geometry_pre_sigma_kappa"] == "0.1"
    assert link_rows[0]["quality_signal_note"] == "test_metrics_available"
    assert link_rows[0]["LengthRatio"] == "1.75"
    assert link_rows[0]["Pitchord"] != ""



def test_run_results_pipeline_handles_partial_and_missing_artifacts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    name = run_name(seed=1, lr=0.2, bs=8)
    run_dir = write_intervention_run(
        root=runs_root,
        run_name=name,
        with_metrics=False,
        with_pre_checkpoint=False,
        with_post_checkpoint=True,
    )

    path_compare_root = tmp_path / "path_compare"
    write_compare_artifact(root=path_compare_root, run_dir=run_dir, with_endpoint_meta=False)

    geometry_summary_csv = tmp_path / "geometry_intervention" / "intervention_geometry_summary.csv"
    geometry_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    geometry_summary_csv.write_text(
        "run_dir,run_name,status\n"
        f"{run_dir},{name},ok\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    run_results_pipeline(
        runs_root=str(runs_root),
        path_compare_root=str(path_compare_root),
        intervention_geometry_summary_csv=str(geometry_summary_csv),
        out_dir=str(out_dir),
    )

    compare_rows = list(csv.DictReader((out_dir / "compare_paths_results.csv").open(encoding="utf-8")))
    assert len(compare_rows) == 1
    assert compare_rows[0]["endpoint_A_loss"] == ""
    assert compare_rows[0]["quality_signal_scope"] == "test_and_validation"

    compare_summary = json.loads((out_dir / "compare_paths_results_summary.json").read_text(encoding="utf-8"))
    assert compare_summary["num_missing_endpoint_eval"] == 1

    intervention_rows = list(csv.DictReader((out_dir / "intervention_runs_results.csv").open(encoding="utf-8")))
    assert len(intervention_rows) == 1
    assert intervention_rows[0]["status"] == "partial"
    assert intervention_rows[0]["has_pre_checkpoint"] == "False"

    geometry_roles_csv = (out_dir / "intervention_geometry_roles_results.csv").read_text(encoding="utf-8").strip().splitlines()
    geometry_runs_csv = (out_dir / "intervention_geometry_runs_results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(geometry_roles_csv) == 1
    assert len(geometry_runs_csv) == 1

    geometry_summary = json.loads((out_dir / "intervention_geometry_roles_results_summary.json").read_text(encoding="utf-8"))
    assert "missing required columns" in geometry_summary["input_issues"][0].lower()
