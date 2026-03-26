from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.io import ensure_dir, save_json

from ._common import preferred_value, read_csv_rows, safe_float, safe_int, write_csv
from ._figure_svg import render_heatmap_svg, render_multi_metric_bars_svg, render_placeholder_svg


REGIME_HEATMAP_CELL_COLUMNS = ["dataset", "metric_name", "learning_rate", "batch_size", "value", "num_rows", "num_runs"]
COMPARE_PATH_FIGURE_DATA_COLUMNS = [
    "dataset", "observed_selection", "num_rows", "num_runs", "mean_BarrierGap", "mean_LengthRatio",
    "mean_devL1", "mean_Peakobs", "mean_quality_acc", "mean_final_val_acc",
]


def _sorted_numeric_strings(values: set[str], *, as_int: bool = False) -> list[str]:
    if as_int:
        return [str(value) for value in sorted((safe_int(item) for item in values if safe_int(item) is not None))]
    return [str(value) for value in sorted((safe_float(item) for item in values if safe_float(item) is not None))]



def build_regime_heatmap_outputs(final_outputs_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    final_outputs_root = Path(final_outputs_root)
    out_dir = ensure_dir(out_dir)
    input_csv = final_outputs_root / "baseline_regime_table.csv"
    out_csv = out_dir / "regime_heatmap_cells.csv"
    out_json = out_dir / "regime_heatmaps_summary.json"
    required_columns = ["dataset", "learning_rate", "batch_size", "num_rows", "num_runs", "mean_barrier_gap", "mean_final_val_acc"]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    metric_specs = [("BarrierGap", "mean_BarrierGap", "mean_barrier_gap", True), ("Quality", "mean_quality_acc", "mean_final_val_acc", False)]
    rows = [{
        "dataset": row.get("dataset", ""),
        "metric_name": metric_name,
        "learning_rate": row.get("learning_rate", ""),
        "batch_size": row.get("batch_size", ""),
        "value": safe_float(preferred_value(row, primary_key, fallback_key)),
        "num_rows": safe_int(row.get("num_rows")),
        "num_runs": safe_int(row.get("num_runs")),
    } for row in source_rows for metric_name, primary_key, fallback_key, _ in metric_specs]
    write_csv(out_csv, REGIME_HEATMAP_CELL_COLUMNS, rows)

    figures: list[str] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})
    if not datasets:
        for metric_name, _, _, _ in metric_specs:
            figure_path = out_dir / f"regime_heatmap__empty__{metric_name}.svg"
            render_placeholder_svg(figure_path, title=f"Regime Heatmap: {metric_name}", message="No baseline regime rows available")
            figures.append(str(figure_path))
    else:
        for dataset in datasets:
            dataset_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
            lr_labels = _sorted_numeric_strings({str(row.get("learning_rate", "")) for row in dataset_rows})
            bs_labels = _sorted_numeric_strings({str(row.get("batch_size", "")) for row in dataset_rows}, as_int=True)
            for metric_name, _, _, diverging in metric_specs:
                figure_path = out_dir / f"regime_heatmap__{dataset}__{metric_name}.svg"
                value_map = {(str(row.get("learning_rate", "")), str(row.get("batch_size", ""))): safe_float(row.get("value")) for row in dataset_rows if str(row.get("metric_name", "")) == metric_name}
                render_heatmap_svg(figure_path, title=f"Regime Map [{dataset}]", metric_name=metric_name, lr_labels=lr_labels, bs_labels=bs_labels, value_map=value_map, diverging=diverging)
                figures.append(str(figure_path))

    summary = {"input_csv": str(input_csv), "out_csv": str(out_csv), "num_source_rows": int(len(source_rows)), "num_plot_rows": int(len(rows)), "figures": figures, "input_issues": input_issues}
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary, "figures": figures}



def build_compare_paths_figure_outputs(final_outputs_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    final_outputs_root = Path(final_outputs_root)
    out_dir = ensure_dir(out_dir)
    input_csv = final_outputs_root / "compare_paths_final_summary.csv"
    out_csv = out_dir / "compare_paths_figure_data.csv"
    out_json = out_dir / "compare_paths_figure_summary.json"
    required_columns = ["dataset", "observed_selection", "num_rows", "num_runs", "mean_barrier_gap", "mean_length_ratio", "mean_loss_profile_l1_mean", "mean_final_val_acc"]
    source_rows, input_issues = read_csv_rows(input_csv, required_columns=required_columns)

    rows = [{
        "dataset": row.get("dataset", ""),
        "observed_selection": row.get("observed_selection", ""),
        "num_rows": safe_int(row.get("num_rows")),
        "num_runs": safe_int(row.get("num_runs")),
        "mean_BarrierGap": safe_float(preferred_value(row, "mean_BarrierGap", "mean_barrier_gap")),
        "mean_LengthRatio": safe_float(preferred_value(row, "mean_LengthRatio", "mean_length_ratio")),
        "mean_devL1": safe_float(preferred_value(row, "mean_devL1", "mean_loss_profile_l1_mean")),
        "mean_Peakobs": safe_float(preferred_value(row, "mean_Peakobs", "mean_observed_DeltaL")),
        "mean_quality_acc": safe_float(preferred_value(row, "mean_quality_acc", "mean_final_val_acc")),
        "mean_final_val_acc": safe_float(row.get("mean_final_val_acc")),
    } for row in source_rows if str(row.get("observed_selection", "")) != "__all__"]
    write_csv(out_csv, COMPARE_PATH_FIGURE_DATA_COLUMNS, rows)

    figures: list[str] = []
    datasets = sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})
    if not datasets:
        figure_path = out_dir / "compare_paths_summary__empty.svg"
        render_placeholder_svg(figure_path, title="Compare-Path Summary", message="No compare-path summary rows available")
        figures.append(str(figure_path))
    else:
        for dataset in datasets:
            dataset_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
            labels = [str(row.get("observed_selection", "")) for row in dataset_rows]
            for row in dataset_rows:
                row["regime_label"] = row.get("observed_selection", "")
            figure_path = out_dir / f"compare_paths_summary__{dataset}.svg"
            render_multi_metric_bars_svg(
                figure_path,
                title=f"Compare-Path Summary [{dataset}]",
                labels=labels,
                metric_specs=[("mean_BarrierGap", "BarrierGap", "#dc2626"), ("mean_LengthRatio", "LengthRatio", "#2563eb"), ("mean_devL1", "devL1", "#7c3aed"), ("mean_Peakobs", "Peakobs", "#059669")],
                rows=dataset_rows,
            )
            figures.append(str(figure_path))

    summary = {"input_csv": str(input_csv), "out_csv": str(out_csv), "num_source_rows": int(len(source_rows)), "num_plot_rows": int(len(rows)), "figures": figures, "input_issues": input_issues}
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary, "figures": figures}
