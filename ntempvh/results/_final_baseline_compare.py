from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._common import mean, preferred_value, read_csv_rows, safe_float, write_csv
from ._final_common import build_metric_map, dataset_from_row, preferred_quality_acc, selection_modes


BASELINE_REGIME_TABLE_COLUMNS = [
    "dataset", "learning_rate", "batch_size", "num_rows", "num_runs", "observed_selection_modes",
    "num_rows_with_final_metrics", "num_rows_with_test_metrics", "mean_chord_DeltaL", "mean_observed_DeltaL",
    "mean_barrier_gap", "mean_length_ratio", "mean_length_excess", "mean_loss_profile_l1_mean",
    "mean_loss_profile_linf", "mean_Peakobs", "mean_Pitchord", "mean_Pitobs", "mean_BarrierGap",
    "mean_LengthRatio", "mean_LengthExcess", "mean_devL1", "mean_final_train_acc", "mean_final_test_acc",
    "mean_train_test_gap", "mean_quality_acc", "mean_final_val_loss", "mean_final_val_acc", "mean_best_val_loss",
]

COMPARE_SECTION_SUMMARY_COLUMNS = [
    "dataset", "observed_selection", "num_rows", "num_runs", "num_rows_with_endpoint_eval",
    "num_rows_with_final_metrics", "num_rows_with_test_metrics", "num_barrier_gap_negative",
    "frac_barrier_gap_negative", "mean_chord_DeltaL", "mean_observed_DeltaL", "mean_barrier_gap",
    "mean_chord_length", "mean_observed_length", "mean_length_ratio", "mean_length_excess",
    "mean_loss_profile_l1_mean", "mean_loss_profile_linf", "mean_Peakobs", "mean_Pitchord", "mean_Pitobs",
    "mean_BarrierGap", "mean_LengthRatio", "mean_LengthExcess", "mean_devL1", "mean_final_train_acc",
    "mean_final_test_acc", "mean_train_test_gap", "mean_quality_acc", "mean_final_val_loss",
    "mean_final_val_acc", "quality_signal_scope_modes",
]


def _mean_thesis_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_Peakobs": mean([preferred_value(row, "Peakobs", "observed_DeltaL") for row in rows]),
        "mean_Pitchord": mean([row.get("Pitchord") for row in rows]),
        "mean_Pitobs": mean([row.get("Pitobs") for row in rows]),
        "mean_BarrierGap": mean([preferred_value(row, "BarrierGap", "barrier_gap") for row in rows]),
        "mean_LengthRatio": mean([preferred_value(row, "LengthRatio", "length_ratio") for row in rows]),
        "mean_LengthExcess": mean([preferred_value(row, "LengthExcess", "length_excess") for row in rows]),
        "mean_devL1": mean([preferred_value(row, "devL1", "loss_profile_l1_mean") for row in rows]),
    }



def build_baseline_regime_outputs(results_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = ensure_dir(out_dir)
    input_csv = results_root / "path_quality_links.csv"
    out_csv = out_dir / "baseline_regime_table.csv"
    out_json = out_dir / "baseline_regime_maps.json"
    required_columns = [
        "run_name", "learning_rate", "batch_size", "observed_selection", "chord_DeltaL", "observed_DeltaL",
        "barrier_gap", "length_ratio", "length_excess", "loss_profile_l1_mean", "loss_profile_linf",
        "final_val_loss", "final_val_acc", "best_val_loss",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    grouped: dict[tuple[str, float | None, int | None], list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault((dataset_from_row(row), safe_float(row.get("learning_rate")), int(row.get("batch_size")) if str(row.get("batch_size", "")).strip() else None), []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, learning_rate, batch_size), regime_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1.0, item[0][2] or -1)):
        rows.append({
            "dataset": dataset,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_rows": int(len(regime_rows)),
            "num_runs": int(len({str(row.get("run_name", "")) for row in regime_rows})),
            "observed_selection_modes": selection_modes(regime_rows),
            "num_rows_with_final_metrics": int(sum(1 for row in regime_rows if safe_float(row.get("final_val_loss")) is not None and safe_float(row.get("final_val_acc")) is not None)),
            "num_rows_with_test_metrics": int(sum(1 for row in regime_rows if safe_float(row.get("final_test_acc")) is not None)),
            "mean_chord_DeltaL": mean([row.get("chord_DeltaL") for row in regime_rows]),
            "mean_observed_DeltaL": mean([row.get("observed_DeltaL") for row in regime_rows]),
            "mean_barrier_gap": mean([row.get("barrier_gap") for row in regime_rows]),
            "mean_length_ratio": mean([row.get("length_ratio") for row in regime_rows]),
            "mean_length_excess": mean([row.get("length_excess") for row in regime_rows]),
            "mean_loss_profile_l1_mean": mean([row.get("loss_profile_l1_mean") for row in regime_rows]),
            "mean_loss_profile_linf": mean([row.get("loss_profile_linf") for row in regime_rows]),
            **_mean_thesis_metrics(regime_rows),
            "mean_final_train_acc": mean([row.get("final_train_acc") for row in regime_rows]),
            "mean_final_test_acc": mean([row.get("final_test_acc") for row in regime_rows]),
            "mean_train_test_gap": mean([row.get("train_test_gap") for row in regime_rows]),
            "mean_quality_acc": mean([preferred_quality_acc(row) for row in regime_rows]),
            "mean_final_val_loss": mean([row.get("final_val_loss") for row in regime_rows]),
            "mean_final_val_acc": mean([row.get("final_val_acc") for row in regime_rows]),
            "mean_best_val_loss": mean([row.get("best_val_loss") for row in regime_rows]),
        })

    write_csv(out_csv, BASELINE_REGIME_TABLE_COLUMNS, rows)
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "num_source_rows": int(len(source_rows)),
        "num_regimes": int(len(rows)),
        "metric_maps": build_metric_map(rows, metric_names=[
            "num_rows", "num_runs", "mean_chord_DeltaL", "mean_observed_DeltaL", "mean_barrier_gap",
            "mean_length_ratio", "mean_length_excess", "mean_Peakobs", "mean_Pitchord", "mean_Pitobs",
            "mean_BarrierGap", "mean_LengthRatio", "mean_LengthExcess", "mean_devL1", "mean_final_train_acc",
            "mean_final_test_acc", "mean_train_test_gap", "mean_quality_acc", "mean_final_val_loss",
            "mean_final_val_acc", "mean_best_val_loss",
        ]),
        "input_issues": input_issues,
        "limitations": [
            "baseline regime summaries are aggregated from already-produced path-quality link artifacts",
            "only metrics already available in results_pipeline outputs are surfaced here",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}


def _compare_group_summary(*, dataset: str, observed_selection: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    negative_count = sum(1 for row in rows if (safe_float(row.get("barrier_gap")) or 0.0) < 0.0)
    return {
        "dataset": dataset,
        "observed_selection": observed_selection,
        "num_rows": int(len(rows)),
        "num_runs": int(len({str(row.get("run_name", "")) for row in rows})),
        "num_rows_with_endpoint_eval": int(sum(1 for row in rows if safe_float(row.get("endpoint_A_loss")) is not None and safe_float(row.get("endpoint_B_loss")) is not None)),
        "num_rows_with_final_metrics": int(sum(1 for row in rows if safe_float(row.get("final_val_loss")) is not None and safe_float(row.get("final_val_acc")) is not None)),
        "num_rows_with_test_metrics": int(sum(1 for row in rows if safe_float(row.get("final_test_acc")) is not None)),
        "num_barrier_gap_negative": int(negative_count),
        "frac_barrier_gap_negative": float(negative_count / len(rows)) if rows else None,
        "mean_chord_DeltaL": mean([row.get("chord_DeltaL") for row in rows]),
        "mean_observed_DeltaL": mean([row.get("observed_DeltaL") for row in rows]),
        "mean_barrier_gap": mean([row.get("barrier_gap") for row in rows]),
        "mean_chord_length": mean([row.get("chord_length") for row in rows]),
        "mean_observed_length": mean([row.get("observed_length") for row in rows]),
        "mean_length_ratio": mean([row.get("length_ratio") for row in rows]),
        "mean_length_excess": mean([row.get("length_excess") for row in rows]),
        "mean_loss_profile_l1_mean": mean([row.get("loss_profile_l1_mean") for row in rows]),
        "mean_loss_profile_linf": mean([row.get("loss_profile_linf") for row in rows]),
        **_mean_thesis_metrics(rows),
        "mean_final_train_acc": mean([row.get("final_train_acc") for row in rows]),
        "mean_final_test_acc": mean([row.get("final_test_acc") for row in rows]),
        "mean_train_test_gap": mean([row.get("train_test_gap") for row in rows]),
        "mean_quality_acc": mean([preferred_quality_acc(row) for row in rows]),
        "mean_final_val_loss": mean([row.get("final_val_loss") for row in rows]),
        "mean_final_val_acc": mean([row.get("final_val_acc") for row in rows]),
        "quality_signal_scope_modes": ",".join(sorted({str(row.get("quality_signal_scope", "")) for row in rows if str(row.get("quality_signal_scope", ""))})),
    }



def build_compare_paths_section_outputs(results_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = ensure_dir(out_dir)
    input_csv = results_root / "compare_paths_results.csv"
    out_csv = out_dir / "compare_paths_final_summary.csv"
    out_json = out_dir / "compare_paths_final_summary.json"
    required_columns = [
        "run_name", "observed_selection", "chord_DeltaL", "observed_DeltaL", "barrier_gap", "chord_length",
        "observed_length", "length_ratio", "length_excess", "loss_profile_l1_mean", "loss_profile_linf",
        "endpoint_A_loss", "endpoint_B_loss", "final_val_loss", "final_val_acc", "quality_signal_scope",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    rows = [_compare_group_summary(dataset="__all__", observed_selection="__all__", rows=source_rows)] if source_rows else []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault((dataset_from_row(row), str(row.get("observed_selection", ""))), []).append(row)
    for (dataset, observed_selection), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append(_compare_group_summary(dataset=dataset, observed_selection=observed_selection, rows=group_rows))

    write_csv(out_csv, COMPARE_SECTION_SUMMARY_COLUMNS, rows)
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "num_source_rows": int(len(source_rows)),
        "num_summary_rows": int(len(rows)),
        "num_rows_with_negative_barrier_gap": int(sum(1 for row in source_rows if (safe_float(row.get("barrier_gap")) or 0.0) < 0.0)),
        "input_issues": input_issues,
        "limitations": [
            "this section is a pure aggregation of compare-path artifacts and does not recompute path metrics",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}
