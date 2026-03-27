from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.results.figure_outputs import (
    COMPARE_PATH_FIGURE_DATA_COLUMNS,
    GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS,
    INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS,
    REGIME_HEATMAP_CELL_COLUMNS,
    run_figure_outputs,
)

from ntempvh.results.final_outputs import (
    BASELINE_REGIME_TABLE_COLUMNS,
    COMPARE_SECTION_SUMMARY_COLUMNS,
    GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
    INTERVENTION_WINDOW_SUMMARY_COLUMNS,
)



def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def test_run_figure_outputs_empty_inputs_write_schema_stable_outputs(tmp_path: Path) -> None:
    final_outputs_root = tmp_path / "final_outputs"
    final_outputs_root.mkdir()
    out_dir = tmp_path / "figure_outputs"

    manifest_path = run_figure_outputs(final_outputs_root=str(final_outputs_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["regime_heatmap_cells"] == 0
    assert manifest["counts"]["compare_plot_rows"] == 0
    assert manifest["counts"]["intervention_plot_rows"] == 0
    assert manifest["counts"]["geometry_plot_rows"] == 0
    assert manifest["status"] == "partial"

    assert (out_dir / "regime_heatmap_cells.csv").read_text(encoding="utf-8").strip() == ",".join(REGIME_HEATMAP_CELL_COLUMNS)
    assert (out_dir / "compare_paths_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_PATH_FIGURE_DATA_COLUMNS)
    assert (out_dir / "intervention_window_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS)
    assert (out_dir / "geometry_transition_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS)

    assert (out_dir / "regime_heatmap__empty__BarrierGap.svg").exists()
    assert (out_dir / "compare_paths_summary__empty.svg").exists()
    assert (out_dir / "intervention_window_summary__empty.svg").exists()
    assert (out_dir / "geometry_transition_summary__empty.svg").exists()



def test_run_figure_outputs_smoke(tmp_path: Path) -> None:
    final_outputs_root = tmp_path / "final_outputs"
    out_dir = tmp_path / "figure_outputs"

    _write_csv(
        final_outputs_root / "baseline_regime_table.csv",
        BASELINE_REGIME_TABLE_COLUMNS,
        [
            {
                "dataset": "cifar10",
                "learning_rate": 0.1,
                "batch_size": 64,
                "num_rows": 2,
                "num_runs": 2,
                "observed_selection_modes": "all",
                "num_rows_with_final_metrics": 2,
                "mean_chord_DeltaL": 0.2,
                "mean_observed_DeltaL": 0.15,
                "mean_barrier_gap": -0.05,
                "mean_length_ratio": 1.8,
                "mean_length_excess": 1.2,
                "mean_loss_profile_l1_mean": 0.03,
                "mean_loss_profile_linf": 0.08,
                "mean_Peakobs": 0.15,
                "mean_Pitchord": 0.01,
                "mean_Pitobs": 0.02,
                "mean_BarrierGap": -0.05,
                "mean_LengthRatio": 1.8,
                "mean_LengthExcess": 1.2,
                "mean_devL1": 0.03,
                "mean_final_val_loss": 0.4,
                "mean_final_val_acc": 0.82,
                "mean_best_val_loss": 0.35,
            },
            {
                "dataset": "cifar10",
                "learning_rate": 0.2,
                "batch_size": 128,
                "num_rows": 2,
                "num_runs": 2,
                "observed_selection_modes": "all",
                "num_rows_with_final_metrics": 2,
                "mean_chord_DeltaL": 0.25,
                "mean_observed_DeltaL": 0.18,
                "mean_barrier_gap": -0.07,
                "mean_length_ratio": 2.0,
                "mean_length_excess": 1.4,
                "mean_loss_profile_l1_mean": 0.04,
                "mean_loss_profile_linf": 0.09,
                "mean_Peakobs": 0.18,
                "mean_Pitchord": 0.015,
                "mean_Pitobs": 0.025,
                "mean_BarrierGap": -0.07,
                "mean_LengthRatio": 2.0,
                "mean_LengthExcess": 1.4,
                "mean_devL1": 0.04,
                "mean_final_val_loss": 0.37,
                "mean_final_val_acc": 0.84,
                "mean_best_val_loss": 0.33,
            },
        ],
    )

    _write_csv(
        final_outputs_root / "compare_paths_final_summary.csv",
        COMPARE_SECTION_SUMMARY_COLUMNS,
        [
            {
                "dataset": "__all__",
                "observed_selection": "__all__",
                "num_rows": 4,
                "num_runs": 2,
                "num_rows_with_endpoint_eval": 4,
                "num_rows_with_final_metrics": 4,
                "num_barrier_gap_negative": 4,
                "frac_barrier_gap_negative": 1.0,
                "mean_chord_DeltaL": 0.2,
                "mean_observed_DeltaL": 0.15,
                "mean_barrier_gap": -0.05,
                "mean_chord_length": 2.0,
                "mean_observed_length": 3.6,
                "mean_length_ratio": 1.8,
                "mean_length_excess": 1.6,
                "mean_loss_profile_l1_mean": 0.03,
                "mean_loss_profile_linf": 0.08,
                "mean_Peakobs": 0.15,
                "mean_Pitchord": 0.01,
                "mean_Pitobs": 0.02,
                "mean_BarrierGap": -0.05,
                "mean_LengthRatio": 1.8,
                "mean_LengthExcess": 1.6,
                "mean_devL1": 0.03,
                "mean_final_val_loss": 0.4,
                "mean_final_val_acc": 0.83,
                "quality_signal_scope_modes": "validation_only",
            },
            {
                "dataset": "cifar10",
                "observed_selection": "all",
                "num_rows": 4,
                "num_runs": 2,
                "num_rows_with_endpoint_eval": 4,
                "num_rows_with_final_metrics": 4,
                "num_barrier_gap_negative": 4,
                "frac_barrier_gap_negative": 1.0,
                "mean_chord_DeltaL": 0.2,
                "mean_observed_DeltaL": 0.15,
                "mean_barrier_gap": -0.05,
                "mean_chord_length": 2.0,
                "mean_observed_length": 3.6,
                "mean_length_ratio": 1.8,
                "mean_length_excess": 1.6,
                "mean_loss_profile_l1_mean": 0.03,
                "mean_loss_profile_linf": 0.08,
                "mean_Peakobs": 0.15,
                "mean_Pitchord": 0.01,
                "mean_Pitobs": 0.02,
                "mean_BarrierGap": -0.05,
                "mean_LengthRatio": 1.8,
                "mean_LengthExcess": 1.6,
                "mean_devL1": 0.03,
                "mean_final_val_loss": 0.4,
                "mean_final_val_acc": 0.83,
                "quality_signal_scope_modes": "validation_only",
            },
        ],
    )

    _write_csv(
        final_outputs_root / "intervention_window_summary.csv",
        INTERVENTION_WINDOW_SUMMARY_COLUMNS,
        [
            {
                "dataset": "cifar10",
                "learning_rate": 0.2,
                "batch_size": 8,
                "intervention_start_epoch": 2,
                "intervention_end_epoch": 3,
                "intervention_lr_multiplier": 2.0,
                "intervention_batch_size": 4,
                "intervention_effective_batch_size": 4,
                "num_runs": 2,
                "num_ok_runs": 2,
                "num_partial_runs": 0,
                "mean_num_intervention_epochs": 2.0,
                "frac_with_pre_checkpoint": 1.0,
                "frac_with_post_checkpoint": 1.0,
                "frac_with_final_checkpoint": 1.0,
                "mean_final_val_loss": 0.41,
                "mean_final_val_acc": 0.8,
                "mean_best_val_loss": 0.35,
                "mean_best_epoch": 3.0,
            }
        ],
    )

    _write_csv(
        final_outputs_root / "geometry_transition_summary.csv",
        GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
        [
            {
                "dataset": "cifar10",
                "learning_rate": 0.2,
                "batch_size": 8,
                "intervention_start_epoch": 2,
                "intervention_end_epoch": 3,
                "intervention_lr_multiplier": 2.0,
                "intervention_batch_size": 4,
                "num_runs": 2,
                "num_ok_runs": 2,
                "num_partial_runs": 0,
                "mean_pre_kappa_tr": 1.0,
                "mean_post_kappa_tr": 1.3,
                "mean_final_kappa_tr": 1.7,
                "mean_delta_kappa_post_minus_pre": 0.3,
                "mean_delta_kappa_final_minus_pre": 0.7,
                "mean_pre_base_loss": 0.8,
                "mean_post_base_loss": 0.6,
                "mean_final_base_loss": 0.4,
                "mean_pre_base_acc": 0.6,
                "mean_post_base_acc": 0.72,
                "mean_final_base_acc": 0.84,
            }
        ],
    )

    manifest_path = run_figure_outputs(final_outputs_root=str(final_outputs_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["counts"]["regime_heatmap_cells"] == 4
    assert manifest["counts"]["compare_plot_rows"] == 1
    assert manifest["counts"]["intervention_plot_rows"] == 1
    assert manifest["counts"]["geometry_plot_rows"] == 1
    assert manifest["counts"]["num_figures"] >= 5

    heatmap_rows = list(csv.DictReader((out_dir / "regime_heatmap_cells.csv").open(encoding="utf-8")))
    assert len(heatmap_rows) == 4
    quality_rows = [row for row in heatmap_rows if row["metric_name"] == "Quality"]
    assert {row["value"] for row in quality_rows} == {"0.82", "0.84"}
    assert (out_dir / "regime_heatmap__cifar10__BarrierGap.svg").exists()
    assert (out_dir / "regime_heatmap__cifar10__Quality.svg").exists()
    assert (out_dir / "compare_paths_summary__cifar10.svg").exists()
    assert (out_dir / "intervention_window_summary__cifar10.svg").exists()
    assert (out_dir / "geometry_transition_summary__cifar10.svg").exists()



def test_run_figure_outputs_handles_partial_inputs(tmp_path: Path) -> None:
    final_outputs_root = tmp_path / "final_outputs"
    out_dir = tmp_path / "figure_outputs"

    _write_csv(
        final_outputs_root / "baseline_regime_table.csv",
        [name for name in BASELINE_REGIME_TABLE_COLUMNS if name != "mean_final_val_acc"],
        [],
    )

    manifest_path = run_figure_outputs(final_outputs_root=str(final_outputs_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    regime_summary = json.loads((out_dir / "regime_heatmaps_summary.json").read_text(encoding="utf-8"))
    assert "missing required columns" in regime_summary["input_issues"][0].lower()
    assert (out_dir / "regime_heatmap__empty__BarrierGap.svg").exists()
