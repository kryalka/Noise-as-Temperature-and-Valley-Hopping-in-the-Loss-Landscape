from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._common import safe_float, write_csv
from ._pipeline_schema import PATH_QUALITY_LINK_COLUMNS



def aggregate_path_quality_links(
    compare_rows: list[dict[str, Any]],
    intervention_run_rows: list[dict[str, Any]],
    intervention_geometry_run_rows: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "path_quality_links.csv"
    out_json = out_dir / "path_quality_links_summary.json"

    intervention_map = {str(row.get("run_name", "")): row for row in intervention_run_rows}
    geometry_map = {str(row.get("run_name", "")): row for row in intervention_geometry_run_rows}
    rows: list[dict[str, Any]] = []

    for compare_row in compare_rows:
        run_name = str(compare_row.get("run_name", ""))
        intervention_row = intervention_map.get(run_name, {})
        geometry_row = geometry_map.get(run_name, {})
        rows.append({
            "comparison_json": compare_row.get("comparison_json"),
            "run_dir": compare_row.get("run_dir"),
            "run_name": run_name,
            "pair_tag": compare_row.get("pair_tag"),
            "seed": compare_row.get("seed"),
            "learning_rate": compare_row.get("learning_rate"),
            "batch_size": compare_row.get("batch_size"),
            "epoch_A": compare_row.get("epoch_A"),
            "epoch_B": compare_row.get("epoch_B"),
            "observed_selection": compare_row.get("observed_selection"),
            "num_points": compare_row.get("num_points"),
            "eval_split": compare_row.get("eval_split"),
            "chord_DeltaL": compare_row.get("chord_DeltaL"),
            "observed_DeltaL": compare_row.get("observed_DeltaL"),
            "barrier_gap": compare_row.get("barrier_gap"),
            "chord_length": compare_row.get("chord_length"),
            "observed_length": compare_row.get("observed_length"),
            "length_ratio": compare_row.get("length_ratio"),
            "length_excess": compare_row.get("length_excess"),
            "loss_profile_l1_mean": compare_row.get("loss_profile_l1_mean"),
            "loss_profile_linf": compare_row.get("loss_profile_linf"),
            "Peakobs": compare_row.get("Peakobs"),
            "Pitchord": compare_row.get("Pitchord"),
            "Pitobs": compare_row.get("Pitobs"),
            "BarrierGap": compare_row.get("BarrierGap"),
            "LengthRatio": compare_row.get("LengthRatio"),
            "LengthExcess": compare_row.get("LengthExcess"),
            "devL1": compare_row.get("devL1"),
            "endpoint_A_loss": compare_row.get("endpoint_A_loss"),
            "endpoint_A_acc": compare_row.get("endpoint_A_acc"),
            "endpoint_B_loss": compare_row.get("endpoint_B_loss"),
            "endpoint_B_acc": compare_row.get("endpoint_B_acc"),
            "final_train_loss": intervention_row.get("final_train_loss", compare_row.get("final_train_loss")),
            "final_train_acc": intervention_row.get("final_train_acc", compare_row.get("final_train_acc")),
            "final_val_loss": intervention_row.get("final_val_loss", compare_row.get("final_val_loss")),
            "final_val_acc": intervention_row.get("final_val_acc", compare_row.get("final_val_acc")),
            "final_test_loss": intervention_row.get("final_test_loss", compare_row.get("final_test_loss")),
            "final_test_acc": intervention_row.get("final_test_acc", compare_row.get("final_test_acc")),
            "train_test_gap": intervention_row.get("train_test_gap", compare_row.get("train_test_gap")),
            "best_val_loss": intervention_row.get("best_val_loss", compare_row.get("best_val_loss")),
            "best_epoch": intervention_row.get("best_epoch", compare_row.get("best_epoch")),
            "intervention_start_epoch": intervention_row.get("intervention_start_epoch"),
            "intervention_end_epoch": intervention_row.get("intervention_end_epoch"),
            "intervention_lr_multiplier": intervention_row.get("intervention_lr_multiplier"),
            "intervention_batch_size": intervention_row.get("intervention_batch_size"),
            "intervention_effective_batch_size": intervention_row.get("intervention_effective_batch_size"),
            "num_intervention_epochs": intervention_row.get("num_intervention_epochs"),
            "geometry_pre_kappa_tr": geometry_row.get("pre_kappa_tr"),
            "geometry_post_kappa_tr": geometry_row.get("post_kappa_tr"),
            "geometry_final_kappa_tr": geometry_row.get("final_kappa_tr"),
            "geometry_pre_sigma_kappa": geometry_row.get("pre_sigma_kappa"),
            "geometry_post_sigma_kappa": geometry_row.get("post_sigma_kappa"),
            "geometry_final_sigma_kappa": geometry_row.get("final_sigma_kappa"),
            "geometry_delta_post_minus_pre": geometry_row.get("delta_kappa_post_minus_pre"),
            "geometry_delta_final_minus_pre": geometry_row.get("delta_kappa_final_minus_pre"),
            "quality_signal_scope": compare_row.get("quality_signal_scope"),
            "quality_signal_note": (
                "test_metrics_available"
                if safe_float(intervention_row.get("final_test_acc", compare_row.get("final_test_acc"))) is not None
                else "validation_metrics_only"
            ),
        })

    write_csv(out_csv, PATH_QUALITY_LINK_COLUMNS, rows)
    summary = {
        "out_csv": str(out_csv),
        "num_rows": int(len(rows)),
        "num_rows_with_final_val_metrics": int(sum(1 for row in rows if safe_float(row.get("final_val_loss")) is not None)),
        "num_rows_with_endpoint_eval": int(sum(1 for row in rows if safe_float(row.get("endpoint_A_loss")) is not None and safe_float(row.get("endpoint_B_loss")) is not None)),
        "num_rows_with_geometry": int(sum(1 for row in rows if safe_float(row.get("geometry_pre_kappa_tr")) is not None)),
        "limitations": [
            "quality-related signals prefer first-class test metrics when they are present upstream",
            "rows without test metrics fall back to validation-derived endpoint and run metrics",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}
