from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.results.final_outputs import (
    BASELINE_REGIME_TABLE_COLUMNS,
    COMPARE_SECTION_SUMMARY_COLUMNS,
    GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
    INTERVENTION_WINDOW_SUMMARY_COLUMNS,
    run_final_outputs,
)

from ntempvh.results.pipeline import (
    COMPARE_RESULTS_COLUMNS,
    INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
    INTERVENTION_RUN_RESULTS_COLUMNS,
    PATH_QUALITY_LINK_COLUMNS,
)


def _run_name(*, dataset: str = "cifar10", seed: int = 1, lr: float = 0.2, bs: int = 8) -> str:
    return (
        f"{dataset}_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0_mom0_schnone__deadbeef"
    )



def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def test_run_final_outputs_empty_inputs_write_schema_stable_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results_pipeline"
    results_root.mkdir()
    out_dir = tmp_path / "final_outputs"

    manifest_path = run_final_outputs(results_root=str(results_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["baseline_regime_rows"] == 0
    assert manifest["counts"]["compare_summary_rows"] == 0
    assert manifest["counts"]["intervention_window_rows"] == 0
    assert manifest["counts"]["geometry_transition_rows"] == 0
    assert manifest["status"] == "partial"

    assert (out_dir / "baseline_regime_table.csv").read_text(encoding="utf-8").strip() == ",".join(BASELINE_REGIME_TABLE_COLUMNS)
    assert (out_dir / "compare_paths_final_summary.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_SECTION_SUMMARY_COLUMNS)
    assert (out_dir / "intervention_window_summary.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_WINDOW_SUMMARY_COLUMNS)
    assert (out_dir / "geometry_transition_summary.csv").read_text(encoding="utf-8").strip() == ",".join(GEOMETRY_TRANSITION_SUMMARY_COLUMNS)



def test_run_final_outputs_smoke(tmp_path: Path) -> None:
    results_root = tmp_path / "results_pipeline"
    out_dir = tmp_path / "final_outputs"
    run_name = _run_name()

    _write_csv(
        results_root / "compare_paths_results.csv",
        COMPARE_RESULTS_COLUMNS,
        [
            {
                "comparison_json": "cmp.json",
                "run_dir": "/tmp/run",
                "run_name": run_name,
                "seed": 1,
                "learning_rate": 0.2,
                "batch_size": 8,
                "epoch_A": 1,
                "epoch_B": 3,
                "ckptA": "a.pt",
                "ckptB": "b.pt",
                "pair_tag": "pair",
                "observed_selection": "all",
                "num_points": 5,
                "eval_split": "val",
                "chord_interp_csv": "chord.csv",
                "observed_interp_csv": "obs.csv",
                "chord_barrier_json": "chord.json",
                "observed_barrier_json": "obs.json",
                "chord_DeltaL": 0.10,
                "observed_DeltaL": 0.05,
                "barrier_gap": -0.05,
                "chord_length": 2.0,
                "observed_length": 3.5,
                "length_ratio": 1.75,
                "length_excess": 1.5,
                "loss_profile_l1_mean": 0.02,
                "loss_profile_linf": 0.08,
                "endpoint_A_loss": 0.9,
                "endpoint_A_acc": 0.55,
                "endpoint_B_loss": 0.6,
                "endpoint_B_acc": 0.75,
                "run_summary_json": "summary.json",
                "final_train_acc": 0.9,
                "final_val_loss": 0.4,
                "final_val_acc": 0.85,
                "final_test_acc": 0.83,
                "train_test_gap": 0.07,
                "best_val_loss": 0.35,
                "best_epoch": 3,
                "quality_signal_scope": "test_and_validation",
            }
        ],
    )

    _write_csv(
        results_root / "intervention_runs_results.csv",
        INTERVENTION_RUN_RESULTS_COLUMNS,
        [
            {
                "run_dir": "/tmp/run",
                "run_name": run_name,
                "seed": 1,
                "learning_rate": 0.2,
                "batch_size": 8,
                "epochs_total": 4,
                "run_config_json": "run_config.json",
                "summary_json": "summary.json",
                "metrics_jsonl": "metrics.jsonl",
                "intervention_enabled": True,
                "intervention_start_epoch": 2,
                "intervention_end_epoch": 3,
                "intervention_lr_multiplier": 2.0,
                "intervention_batch_size": 4,
                "intervention_effective_batch_size": 4,
                "num_intervention_epochs": 2,
                "num_metrics_rows": 4,
                "num_intervention_metric_rows": 2,
                "expected_pre_epoch": 1,
                "expected_post_epoch": 3,
                "has_pre_checkpoint": True,
                "has_post_checkpoint": True,
                "has_final_checkpoint": True,
                "final_checkpoint": "final.pt",
                "final_train_acc": 0.9,
                "final_val_loss": 0.4,
                "final_val_acc": 0.85,
                "final_test_acc": 0.83,
                "train_test_gap": 0.07,
                "best_val_loss": 0.35,
                "best_epoch": 3,
                "status": "ok",
                "reason": "",
            }
        ],
    )

    _write_csv(
        results_root / "intervention_geometry_runs_results.csv",
        INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
        [
            {
                "run_dir": "/tmp/run",
                "run_name": run_name,
                "seed": 1,
                "learning_rate": 0.2,
                "batch_size": 8,
                "intervention_start_epoch": 2,
                "intervention_end_epoch": 3,
                "intervention_lr_multiplier": 2.0,
                "intervention_batch_size": 4,
                "num_roles_present": 3,
                "status": "ok",
                "reason": "",
                "pre_status": "ok",
                "pre_reason": "",
                "pre_checkpoint_path": "pre.pt",
                "pre_checkpoint_epoch": 1,
                "pre_geometry_json": "pre.json",
                "pre_kappa_tr": 1.0,
                "pre_kappa_tr_std": 0.1,
                "pre_sigma_kappa": 0.1,
                "pre_anisotropy": 0.1,
                "pre_base_loss": 0.9,
                "pre_base_acc": 0.55,
                "post_status": "ok",
                "post_reason": "",
                "post_checkpoint_path": "post.pt",
                "post_checkpoint_epoch": 3,
                "post_geometry_json": "post.json",
                "post_kappa_tr": 1.5,
                "post_kappa_tr_std": 0.1,
                "post_sigma_kappa": 0.1,
                "post_anisotropy": 0.1,
                "post_base_loss": 0.7,
                "post_base_acc": 0.7,
                "final_status": "ok",
                "final_reason": "",
                "final_checkpoint_path": "final.pt",
                "final_checkpoint_epoch": 4,
                "final_geometry_json": "final.json",
                "final_kappa_tr": 2.0,
                "final_kappa_tr_std": 0.1,
                "final_sigma_kappa": 0.1,
                "final_anisotropy": 0.1,
                "final_base_loss": 0.4,
                "final_base_acc": 0.85,
                "delta_kappa_post_minus_pre": 0.5,
                "delta_kappa_final_minus_pre": 1.0,
            }
        ],
    )

    _write_csv(
        results_root / "path_quality_links.csv",
        PATH_QUALITY_LINK_COLUMNS,
        [
            {
                "comparison_json": "cmp.json",
                "run_dir": "/tmp/run",
                "run_name": run_name,
                "pair_tag": "pair",
                "seed": 1,
                "learning_rate": 0.2,
                "batch_size": 8,
                "epoch_A": 1,
                "epoch_B": 3,
                "observed_selection": "all",
                "num_points": 5,
                "eval_split": "val",
                "chord_DeltaL": 0.10,
                "observed_DeltaL": 0.05,
                "barrier_gap": -0.05,
                "chord_length": 2.0,
                "observed_length": 3.5,
                "length_ratio": 1.75,
                "length_excess": 1.5,
                "loss_profile_l1_mean": 0.02,
                "loss_profile_linf": 0.08,
                "endpoint_A_loss": 0.9,
                "endpoint_A_acc": 0.55,
                "endpoint_B_loss": 0.6,
                "endpoint_B_acc": 0.75,
                "final_train_acc": 0.9,
                "final_val_loss": 0.4,
                "final_val_acc": 0.85,
                "final_test_acc": 0.83,
                "train_test_gap": 0.07,
                "best_val_loss": 0.35,
                "best_epoch": 3,
                "intervention_start_epoch": 2,
                "intervention_end_epoch": 3,
                "intervention_lr_multiplier": 2.0,
                "intervention_batch_size": 4,
                "intervention_effective_batch_size": 4,
                "num_intervention_epochs": 2,
                "geometry_pre_kappa_tr": 1.0,
                "geometry_post_kappa_tr": 1.5,
                "geometry_final_kappa_tr": 2.0,
                "geometry_delta_post_minus_pre": 0.5,
                "geometry_delta_final_minus_pre": 1.0,
                "quality_signal_scope": "test_and_validation",
                "quality_signal_note": "test_metrics_available",
            }
        ],
    )

    manifest_path = run_final_outputs(results_root=str(results_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["counts"]["baseline_regime_rows"] == 1
    assert manifest["counts"]["compare_summary_rows"] == 2
    assert manifest["counts"]["intervention_window_rows"] == 1
    assert manifest["counts"]["geometry_transition_rows"] == 1

    baseline_rows = list(csv.DictReader((out_dir / "baseline_regime_table.csv").open(encoding="utf-8")))
    assert len(baseline_rows) == 1
    assert baseline_rows[0]["dataset"] == "cifar10"
    assert baseline_rows[0]["mean_final_val_acc"] == "0.85"
    assert baseline_rows[0]["mean_final_test_acc"] == "0.83"
    assert baseline_rows[0]["mean_train_test_gap"] == "0.07"
    assert baseline_rows[0]["mean_quality_acc"] == "0.83"
    assert baseline_rows[0]["mean_barrier_gap"] == "-0.05"
    assert baseline_rows[0]["mean_Peakobs"] == "0.05"
    assert baseline_rows[0]["mean_BarrierGap"] == "-0.05"
    assert baseline_rows[0]["mean_devL1"] == "0.02"

    compare_rows = list(csv.DictReader((out_dir / "compare_paths_final_summary.csv").open(encoding="utf-8")))
    assert len(compare_rows) == 2
    assert compare_rows[0]["dataset"] == "__all__"
    assert compare_rows[1]["observed_selection"] == "all"
    assert compare_rows[1]["frac_barrier_gap_negative"] == "1.0"
    assert compare_rows[1]["mean_final_test_acc"] == "0.83"
    assert compare_rows[1]["mean_Peakobs"] == "0.05"
    assert compare_rows[1]["mean_devL1"] == "0.02"

    intervention_rows = list(csv.DictReader((out_dir / "intervention_window_summary.csv").open(encoding="utf-8")))
    assert len(intervention_rows) == 1
    assert intervention_rows[0]["num_runs"] == "1"
    assert intervention_rows[0]["frac_with_pre_checkpoint"] == "1.0"
    assert intervention_rows[0]["mean_final_test_acc"] == "0.83"

    geometry_rows = list(csv.DictReader((out_dir / "geometry_transition_summary.csv").open(encoding="utf-8")))
    assert len(geometry_rows) == 1
    assert geometry_rows[0]["mean_delta_kappa_post_minus_pre"] == "0.5"
    assert geometry_rows[0]["mean_final_base_acc"] == "0.85"
    assert geometry_rows[0]["mean_final_sigma_kappa"] == "0.1"
    assert geometry_rows[0]["mean_final_anisotropy"] == "0.1"



def test_run_final_outputs_handles_partial_and_missing_inputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results_pipeline"
    out_dir = tmp_path / "final_outputs"

    _write_csv(
        results_root / "compare_paths_results.csv",
        COMPARE_RESULTS_COLUMNS[:-1],
        [],
    )
    _write_csv(
        results_root / "intervention_runs_results.csv",
        INTERVENTION_RUN_RESULTS_COLUMNS,
        [],
    )
    _write_csv(
        results_root / "intervention_geometry_runs_results.csv",
        INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
        [],
    )

    manifest_path = run_final_outputs(results_root=str(results_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    compare_summary = json.loads((out_dir / "compare_paths_final_summary.json").read_text(encoding="utf-8"))
    assert "missing required columns" in compare_summary["input_issues"][0].lower()

    baseline_summary = json.loads((out_dir / "baseline_regime_maps.json").read_text(encoding="utf-8"))
    assert "missing required input csv" in baseline_summary["input_issues"][0].lower()

    assert (out_dir / "compare_paths_final_summary.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_SECTION_SUMMARY_COLUMNS)
