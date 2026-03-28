from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._common import fraction_true, mean, preferred_value, read_csv_rows, safe_float, safe_int, write_csv
from ._final_common import dataset_from_row, preferred_quality_acc


INTERVENTION_WINDOW_SUMMARY_COLUMNS = [
    "dataset", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
    "intervention_lr_multiplier", "intervention_batch_size", "intervention_effective_batch_size", "num_runs",
    "num_ok_runs", "num_partial_runs", "mean_num_intervention_epochs", "frac_with_pre_checkpoint",
    "frac_with_post_checkpoint", "frac_with_final_checkpoint", "mean_final_train_acc", "mean_final_test_acc",
    "mean_train_test_gap", "mean_quality_acc", "mean_final_val_loss", "mean_final_val_acc",
    "mean_best_val_loss", "mean_best_epoch",
]

GEOMETRY_TRANSITION_SUMMARY_COLUMNS = [
    "dataset", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
    "intervention_lr_multiplier", "intervention_batch_size", "num_runs", "num_ok_runs", "num_partial_runs",
    "mean_pre_kappa_tr", "mean_post_kappa_tr", "mean_final_kappa_tr", "mean_delta_kappa_post_minus_pre",
    "mean_delta_kappa_final_minus_pre", "mean_pre_sigma_kappa", "mean_post_sigma_kappa", "mean_final_sigma_kappa",
    "mean_pre_anisotropy", "mean_post_anisotropy", "mean_final_anisotropy", "mean_pre_base_loss",
    "mean_post_base_loss", "mean_final_base_loss", "mean_pre_base_acc", "mean_post_base_acc", "mean_final_base_acc",
]



def build_intervention_window_outputs(results_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = ensure_dir(out_dir)
    input_csv = results_root / "intervention_runs_results.csv"
    out_csv = out_dir / "intervention_window_summary.csv"
    out_json = out_dir / "intervention_window_summary.json"
    required_columns = [
        "run_name", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
        "intervention_lr_multiplier", "intervention_batch_size", "intervention_effective_batch_size",
        "num_intervention_epochs", "has_pre_checkpoint", "has_post_checkpoint", "has_final_checkpoint",
        "final_val_loss", "final_val_acc", "best_val_loss", "best_epoch", "status",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (
            dataset_from_row(row), safe_float(row.get("learning_rate")), safe_int(row.get("batch_size")),
            safe_int(row.get("intervention_start_epoch")), safe_int(row.get("intervention_end_epoch")),
            safe_float(row.get("intervention_lr_multiplier")), safe_int(row.get("intervention_batch_size")),
            safe_int(row.get("intervention_effective_batch_size")),
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        dataset, learning_rate, batch_size, start_epoch, end_epoch, lr_multiplier, intervention_batch_size, effective_batch_size = key
        rows.append({
            "dataset": dataset,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "intervention_start_epoch": start_epoch,
            "intervention_end_epoch": end_epoch,
            "intervention_lr_multiplier": lr_multiplier,
            "intervention_batch_size": intervention_batch_size,
            "intervention_effective_batch_size": effective_batch_size,
            "num_runs": int(len(group_rows)),
            "num_ok_runs": int(sum(1 for row in group_rows if str(row.get("status", "")) == "ok")),
            "num_partial_runs": int(sum(1 for row in group_rows if str(row.get("status", "")) != "ok")),
            "mean_num_intervention_epochs": mean([row.get("num_intervention_epochs") for row in group_rows]),
            "frac_with_pre_checkpoint": fraction_true([row.get("has_pre_checkpoint") for row in group_rows]),
            "frac_with_post_checkpoint": fraction_true([row.get("has_post_checkpoint") for row in group_rows]),
            "frac_with_final_checkpoint": fraction_true([row.get("has_final_checkpoint") for row in group_rows]),
            "mean_final_train_acc": mean([row.get("final_train_acc") for row in group_rows]),
            "mean_final_test_acc": mean([row.get("final_test_acc") for row in group_rows]),
            "mean_train_test_gap": mean([row.get("train_test_gap") for row in group_rows]),
            "mean_quality_acc": mean([preferred_quality_acc(row) for row in group_rows]),
            "mean_final_val_loss": mean([row.get("final_val_loss") for row in group_rows]),
            "mean_final_val_acc": mean([row.get("final_val_acc") for row in group_rows]),
            "mean_best_val_loss": mean([row.get("best_val_loss") for row in group_rows]),
            "mean_best_epoch": mean([row.get("best_epoch") for row in group_rows]),
        })

    write_csv(out_csv, INTERVENTION_WINDOW_SUMMARY_COLUMNS, rows)
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "num_source_rows": int(len(source_rows)),
        "num_summary_rows": int(len(rows)),
        "num_partial_source_rows": int(sum(1 for row in source_rows if str(row.get("status", "")) != "ok")),
        "input_issues": input_issues,
        "limitations": [
            "intervention window summaries reflect only already-logged run artifacts and do not infer missing checkpoints",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}



def build_geometry_transition_outputs(results_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = ensure_dir(out_dir)
    input_csv = results_root / "intervention_geometry_runs_results.csv"
    out_csv = out_dir / "geometry_transition_summary.csv"
    out_json = out_dir / "geometry_transition_summary.json"
    required_columns = [
        "run_name", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
        "intervention_lr_multiplier", "intervention_batch_size", "status", "pre_kappa_tr", "post_kappa_tr",
        "final_kappa_tr", "delta_kappa_post_minus_pre", "delta_kappa_final_minus_pre", "pre_base_loss",
        "post_base_loss", "final_base_loss", "pre_base_acc", "post_base_acc", "final_base_acc",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (
            dataset_from_row(row), safe_float(row.get("learning_rate")), safe_int(row.get("batch_size")),
            safe_int(row.get("intervention_start_epoch")), safe_int(row.get("intervention_end_epoch")),
            safe_float(row.get("intervention_lr_multiplier")), safe_int(row.get("intervention_batch_size")),
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        dataset, learning_rate, batch_size, start_epoch, end_epoch, lr_multiplier, intervention_batch_size = key
        rows.append({
            "dataset": dataset,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "intervention_start_epoch": start_epoch,
            "intervention_end_epoch": end_epoch,
            "intervention_lr_multiplier": lr_multiplier,
            "intervention_batch_size": intervention_batch_size,
            "num_runs": int(len(group_rows)),
            "num_ok_runs": int(sum(1 for row in group_rows if str(row.get("status", "")) == "ok")),
            "num_partial_runs": int(sum(1 for row in group_rows if str(row.get("status", "")) != "ok")),
            "mean_pre_kappa_tr": mean([row.get("pre_kappa_tr") for row in group_rows]),
            "mean_post_kappa_tr": mean([row.get("post_kappa_tr") for row in group_rows]),
            "mean_final_kappa_tr": mean([row.get("final_kappa_tr") for row in group_rows]),
            "mean_delta_kappa_post_minus_pre": mean([row.get("delta_kappa_post_minus_pre") for row in group_rows]),
            "mean_delta_kappa_final_minus_pre": mean([row.get("delta_kappa_final_minus_pre") for row in group_rows]),
            "mean_pre_sigma_kappa": mean([preferred_value(row, "pre_sigma_kappa", "pre_kappa_tr_std") for row in group_rows]),
            "mean_post_sigma_kappa": mean([preferred_value(row, "post_sigma_kappa", "post_kappa_tr_std") for row in group_rows]),
            "mean_final_sigma_kappa": mean([preferred_value(row, "final_sigma_kappa", "final_kappa_tr_std") for row in group_rows]),
            "mean_pre_anisotropy": mean([preferred_value(row, "pre_anisotropy", "pre_sigma_kappa") for row in group_rows]),
            "mean_post_anisotropy": mean([preferred_value(row, "post_anisotropy", "post_sigma_kappa") for row in group_rows]),
            "mean_final_anisotropy": mean([preferred_value(row, "final_anisotropy", "final_sigma_kappa") for row in group_rows]),
            "mean_pre_base_loss": mean([row.get("pre_base_loss") for row in group_rows]),
            "mean_post_base_loss": mean([row.get("post_base_loss") for row in group_rows]),
            "mean_final_base_loss": mean([row.get("final_base_loss") for row in group_rows]),
            "mean_pre_base_acc": mean([row.get("pre_base_acc") for row in group_rows]),
            "mean_post_base_acc": mean([row.get("post_base_acc") for row in group_rows]),
            "mean_final_base_acc": mean([row.get("final_base_acc") for row in group_rows]),
        })

    write_csv(out_csv, GEOMETRY_TRANSITION_SUMMARY_COLUMNS, rows)
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "num_source_rows": int(len(source_rows)),
        "num_summary_rows": int(len(rows)),
        "num_partial_source_rows": int(sum(1 for row in source_rows if str(row.get("status", "")) != "ok")),
        "input_issues": input_issues,
        "limitations": [
            "geometry transition summaries are aggregated from existing pre, post and final geometry artifacts only",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}
