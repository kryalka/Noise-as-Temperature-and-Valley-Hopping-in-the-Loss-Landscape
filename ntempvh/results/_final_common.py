from __future__ import annotations

from typing import Any

from ntempvh.utils.checkpoints import parse_run_name

from ._common import safe_float, safe_int


def dataset_from_row(row: dict[str, Any]) -> str:
    run_name = str(row.get("run_name", ""))
    parsed = parse_run_name(run_name)
    return str(parsed.get("dataset", "")) if parsed else ""


def selection_modes(rows: list[dict[str, Any]]) -> str:
    modes = sorted({str(row.get("observed_selection", "")) for row in rows if str(row.get("observed_selection", ""))})
    return ",".join(modes)


def build_metric_map(
    rows: list[dict[str, Any]],
    *,
    metric_names: list[str],
) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    for metric_name in metric_names:
        metric_map: dict[str, Any] = {}
        for row in rows:
            dataset = str(row.get("dataset", "") or "unknown")
            lr = str(row.get("learning_rate", ""))
            bs = str(row.get("batch_size", ""))
            value = safe_float(row.get(metric_name))
            if value is None and safe_int(row.get(metric_name)) is not None:
                value = float(safe_int(row.get(metric_name)) or 0)
            metric_map.setdefault(dataset, {}).setdefault(lr, {})[bs] = value
        maps[metric_name] = metric_map
    return maps


def preferred_quality_acc(row: dict[str, Any]) -> Any:
    value = row.get("final_test_acc", None)
    if value not in (None, ""):
        return value
    return row.get("final_val_acc", None)
