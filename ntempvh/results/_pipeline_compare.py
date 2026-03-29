from __future__ import annotations

from pathlib import Path
from typing import Any

from ntempvh.utils.checkpoints import parse_checkpoint_path
from ntempvh.utils.io import ensure_dir, save_json
from ntempvh.utils.path_metrics import compute_linear_baseline_shape_metrics

from ._common import load_json_object, safe_float, safe_int, write_csv
from ._pipeline_schema import COMPARE_RESULTS_COLUMNS


def _comparison_json_paths(path_compare_root: Path) -> tuple[list[Path], list[str]]:
    issues: list[str] = []
    search_dir = path_compare_root
    if path_compare_root.name != "comparisons" and (path_compare_root / "comparisons").is_dir():
        search_dir = path_compare_root / "comparisons"
    if not search_dir.exists():
        issues.append(f"Path-compare root not found: {search_dir}")
        return [], issues
    return sorted(search_dir.glob("pathcompare__*.json")), issues


def _load_endpoint_eval(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        return {}
    meta, error = load_json_object(meta_path)
    if error is not None or meta is None:
        return {}
    endpoint_eval = meta.get("endpoint_eval", {}) or {}
    return endpoint_eval if isinstance(endpoint_eval, dict) else {}


def _run_summary_metrics(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}, f"Missing run summary: {summary_path}"
    summary, error = load_json_object(summary_path)
    if error is not None or summary is None:
        return {}, error
    return summary, None


def _coalesce_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _quality_signal_scope(
    *,
    endpoint_a_loss: float | None,
    endpoint_b_loss: float | None,
    final_val_loss: float | None,
    final_test_loss: float | None,
) -> str:
    has_endpoint = endpoint_a_loss is not None and endpoint_b_loss is not None
    has_final_val = final_val_loss is not None
    has_final_test = final_test_loss is not None
    if has_final_test and has_final_val:
        return "test_and_validation"
    if has_final_test:
        return "test_only"
    if has_endpoint and has_final_val:
        return "validation_only"
    if has_endpoint or has_final_val:
        return "validation_only_partial"
    return "unavailable"
