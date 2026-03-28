from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._common import preferred_value, read_csv_rows, safe_float, safe_int, write_csv
from ._figure_svg import render_multi_metric_bars_svg, render_placeholder_svg


INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS = [
    "dataset", "regime_label", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
    "intervention_lr_multiplier", "intervention_batch_size", "mean_quality_acc", "mean_final_val_acc",
    "mean_final_val_loss", "num_runs", "num_partial_runs",
]

GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS = [
    "dataset", "regime_label", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
    "intervention_lr_multiplier", "intervention_batch_size", "mean_pre_kappa_tr", "mean_post_kappa_tr",
    "mean_final_kappa_tr", "mean_delta_kappa_post_minus_pre", "mean_delta_kappa_final_minus_pre",
    "num_runs", "num_partial_runs",
]


def _regime_label(row: dict[str, Any]) -> str:
    return (
        f"lr{row.get('learning_rate', '')}_bs{row.get('batch_size', '')}"
        f"_w{row.get('intervention_start_epoch', '')}-{row.get('intervention_end_epoch', '')}"
        f"_x{row.get('intervention_lr_multiplier', '')}_ibs{row.get('intervention_batch_size', '')}"
    )



def build_intervention_window_figure_outputs(final_outputs_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    final_outputs_root = Path(final_outputs_root)
    out_dir = ensure_dir(out_dir)
    input_csv = final_outputs_root / "intervention_window_summary.csv"
    out_csv = out_dir / "intervention_window_figure_data.csv"
    out_json = out_dir / "intervention_window_figure_summary.json"
    required_columns = [
        "dataset", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
        "intervention_lr_multiplier", "intervention_batch_size", "mean_final_val_acc", "mean_final_val_loss",
        "num_runs", "num_partial_runs",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    rows = [{
        "dataset": row.get("dataset", ""),
        "regime_label": _regime_label(row),
        "learning_rate": row.get("learning_rate", ""),
        "batch_size": row.get("batch_size", ""),
        "intervention_start_epoch": row.get("intervention_start_epoch", ""),
        "intervention_end_epoch": row.get("intervention_end_epoch", ""),
        "intervention_lr_multiplier": row.get("intervention_lr_multiplier", ""),
        "intervention_batch_size": row.get("intervention_batch_size", ""),
        "mean_quality_acc": safe_float(preferred_value(row, "mean_quality_acc", "mean_final_val_acc")),
        "mean_final_val_acc": safe_float(row.get("mean_final_val_acc")),
        "mean_final_val_loss": safe_float(row.get("mean_final_val_loss")),
        "num_runs": safe_int(row.get("num_runs")),
        "num_partial_runs": safe_int(row.get("num_partial_runs")),
    } for row in source_rows]
    write_csv(out_csv, INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS, rows)

    figures: list[str] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})
    if not datasets:
        figure_path = out_dir / "intervention_window_summary__empty.svg"
        render_placeholder_svg(figure_path, title="Intervention Window Summary", message="No intervention summary rows available")
        figures.append(str(figure_path))
    else:
        for dataset in datasets:
            dataset_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
            labels = [str(row.get("regime_label", "")) for row in dataset_rows]
            figure_path = out_dir / f"intervention_window_summary__{dataset}.svg"
            render_multi_metric_bars_svg(figure_path, title=f"Intervention Window Summary [{dataset}]", labels=labels, metric_specs=[("mean_quality_acc", "Quality Acc", "#059669"), ("mean_final_val_loss", "Final Val Loss", "#ea580c")], rows=dataset_rows)
            figures.append(str(figure_path))

    summary = {"input_csv": str(input_csv), "out_csv": str(out_csv), "num_source_rows": int(len(source_rows)), "num_plot_rows": int(len(rows)), "figures": figures, "input_issues": input_issues}
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary, "figures": figures}



def build_geometry_transition_figure_outputs(final_outputs_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    final_outputs_root = Path(final_outputs_root)
    out_dir = ensure_dir(out_dir)
    input_csv = final_outputs_root / "geometry_transition_summary.csv"
    out_csv = out_dir / "geometry_transition_figure_data.csv"
    out_json = out_dir / "geometry_transition_figure_summary.json"
    required_columns = [
        "dataset", "learning_rate", "batch_size", "intervention_start_epoch", "intervention_end_epoch",
        "intervention_lr_multiplier", "intervention_batch_size", "mean_pre_kappa_tr", "mean_post_kappa_tr",
        "mean_final_kappa_tr", "mean_delta_kappa_post_minus_pre", "mean_delta_kappa_final_minus_pre",
        "num_runs", "num_partial_runs",
    ]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    rows = [{
        "dataset": row.get("dataset", ""),
        "regime_label": _regime_label(row),
        "learning_rate": row.get("learning_rate", ""),
        "batch_size": row.get("batch_size", ""),
        "intervention_start_epoch": row.get("intervention_start_epoch", ""),
        "intervention_end_epoch": row.get("intervention_end_epoch", ""),
        "intervention_lr_multiplier": row.get("intervention_lr_multiplier", ""),
        "intervention_batch_size": row.get("intervention_batch_size", ""),
        "mean_pre_kappa_tr": safe_float(row.get("mean_pre_kappa_tr")),
        "mean_post_kappa_tr": safe_float(row.get("mean_post_kappa_tr")),
        "mean_final_kappa_tr": safe_float(row.get("mean_final_kappa_tr")),
        "mean_delta_kappa_post_minus_pre": safe_float(row.get("mean_delta_kappa_post_minus_pre")),
        "mean_delta_kappa_final_minus_pre": safe_float(row.get("mean_delta_kappa_final_minus_pre")),
        "num_runs": safe_int(row.get("num_runs")),
        "num_partial_runs": safe_int(row.get("num_partial_runs")),
    } for row in source_rows]
    write_csv(out_csv, GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS, rows)

    figures: list[str] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})
    if not datasets:
        figure_path = out_dir / "geometry_transition_summary__empty.svg"
        render_placeholder_svg(figure_path, title="Geometry Transition Summary", message="No geometry transition rows available")
        figures.append(str(figure_path))
    else:
        for dataset in datasets:
            dataset_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
            labels = [str(row.get("regime_label", "")) for row in dataset_rows]
            figure_path = out_dir / f"geometry_transition_summary__{dataset}.svg"
            render_multi_metric_bars_svg(figure_path, title=f"Geometry Transition Summary [{dataset}]", labels=labels, metric_specs=[("mean_pre_kappa_tr", "Pre kappa_tr", "#2563eb"), ("mean_post_kappa_tr", "Post kappa_tr", "#7c3aed"), ("mean_final_kappa_tr", "Final kappa_tr", "#dc2626")], rows=dataset_rows)
            figures.append(str(figure_path))

    summary = {"input_csv": str(input_csv), "out_csv": str(out_csv), "num_source_rows": int(len(source_rows)), "num_plot_rows": int(len(rows)), "figures": figures, "input_issues": input_issues}
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary, "figures": figures}
