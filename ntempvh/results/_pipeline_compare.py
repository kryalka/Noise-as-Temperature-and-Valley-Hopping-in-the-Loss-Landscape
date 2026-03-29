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



def aggregate_compare_paths(
    path_compare_root: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    compare_root = Path(path_compare_root)
    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "compare_paths_results.csv"
    out_json = out_dir / "compare_paths_results_summary.json"

    rows: list[dict[str, Any]] = []
    invalid_examples: list[dict[str, str]] = []
    json_paths, input_issues = _comparison_json_paths(compare_root)
    num_missing_endpoint_eval = 0
    num_missing_run_summary = 0
    num_run_summary_errors = 0
    required_metrics = [
        "chord_DeltaL",
        "observed_DeltaL",
        "barrier_gap",
        "chord_length",
        "observed_length",
        "length_ratio",
        "length_excess",
        "loss_profile_l1_mean",
        "loss_profile_linf",
    ]

    for comparison_json in json_paths:
        payload, error = load_json_object(comparison_json)
        if error is not None or payload is None:
            invalid_examples.append({"comparison_json": str(comparison_json), "error": error or "invalid_json"})
            continue

        missing_top = [key for key in ("ckptA", "ckptB", "metrics", "artifacts") if key not in payload]
        if missing_top:
            invalid_examples.append({"comparison_json": str(comparison_json), "error": f"Missing required keys: {missing_top}"})
            continue

        metrics = payload.get("metrics", {}) or {}
        report_metrics = payload.get("report_metrics", {}) or {}
        artifacts = payload.get("artifacts", {}) or {}
        if not isinstance(metrics, dict) or not isinstance(report_metrics, dict) or not isinstance(artifacts, dict):
            invalid_examples.append({"comparison_json": str(comparison_json), "error": "metrics/report_metrics/artifacts must be JSON objects"})
            continue

        missing_metrics = [key for key in required_metrics if key not in metrics]
        if missing_metrics:
            invalid_examples.append({"comparison_json": str(comparison_json), "error": f"Missing required metrics: {missing_metrics}"})
            continue

        ckpt_a = str(payload.get("ckptA", ""))
        ckpt_b = str(payload.get("ckptB", ""))
        try:
            info_a = parse_checkpoint_path(ckpt_a)
            info_b = parse_checkpoint_path(ckpt_b)
        except Exception as exc:
            invalid_examples.append({"comparison_json": str(comparison_json), "error": f"Could not parse checkpoint paths: {exc}"})
            continue

        run_dir = Path(ckpt_a).parent.parent
        run_summary, run_summary_error = _run_summary_metrics(run_dir)
        if run_summary_error is not None:
            num_missing_run_summary += 1
            num_run_summary_errors += 1

        chord_interp_csv = str(artifacts.get("chord_interp_csv", ""))
        observed_interp_csv = str(artifacts.get("observed_interp_csv", ""))
        chord_meta_json = artifacts.get("chord_meta_json")
        endpoint_eval = _load_endpoint_eval(
            Path(str(chord_meta_json)) if chord_meta_json else Path(str(chord_interp_csv)).with_suffix(".meta.json")
        )

        endpoint_a = endpoint_eval.get("A", {}) or {}
        endpoint_b = endpoint_eval.get("B", {}) or {}
        endpoint_a_loss = safe_float(endpoint_a.get("loss"))
        endpoint_a_acc = safe_float(endpoint_a.get("acc"))
        endpoint_b_loss = safe_float(endpoint_b.get("loss"))
        endpoint_b_acc = safe_float(endpoint_b.get("acc"))
        if endpoint_a_loss is None or endpoint_b_loss is None:
            num_missing_endpoint_eval += 1

        final_train_loss = safe_float(run_summary.get("final_train_loss"))
        final_train_acc = safe_float(run_summary.get("final_train_acc"))
        final_val_loss = safe_float(run_summary.get("final_val_loss"))
        final_val_acc = safe_float(run_summary.get("final_val_acc"))
        final_test_loss = safe_float(run_summary.get("final_test_loss"))
        final_test_acc = safe_float(run_summary.get("final_test_acc"))
        train_test_gap = _coalesce_numeric(
            run_summary.get("train_test_gap"),
            None if final_train_acc is None or final_test_acc is None else float(final_train_acc - final_test_acc),
        )

        chord_shape_metrics: dict[str, float] = {}
        observed_shape_metrics: dict[str, float] = {}
        if chord_interp_csv:
            try:
                chord_shape_metrics = compute_linear_baseline_shape_metrics(chord_interp_csv)
            except Exception:
                chord_shape_metrics = {}
        if observed_interp_csv:
            try:
                observed_shape_metrics = compute_linear_baseline_shape_metrics(observed_interp_csv)
            except Exception:
                observed_shape_metrics = {}

        rows.append({
            "comparison_json": str(comparison_json),
            "run_dir": str(run_dir),
            "run_name": str(info_a["run_name"]),
            "seed": int(info_a["seed"]),
            "learning_rate": float(info_a["learning_rate"]),
            "batch_size": int(info_a["batch_size"]),
            "epoch_A": int(info_a["epoch"]),
            "epoch_B": int(info_b["epoch"]),
            "ckptA": ckpt_a,
            "ckptB": ckpt_b,
            "pair_tag": str(payload.get("pair_tag", "")),
            "observed_selection": str(((payload.get("observed_path") or {}).get("selection", ""))),
            "num_points": safe_int(((payload.get("config") or {}).get("path") or {}).get("num_points")),
            "eval_split": str(((payload.get("config") or {}).get("evaluation") or {}).get("split", "")),
            "chord_interp_csv": chord_interp_csv,
            "observed_interp_csv": observed_interp_csv,
            "chord_barrier_json": str(artifacts.get("chord_barrier_json", "")),
            "observed_barrier_json": str(artifacts.get("observed_barrier_json", "")),
            "chord_DeltaL": safe_float(metrics.get("chord_DeltaL")),
            "observed_DeltaL": safe_float(metrics.get("observed_DeltaL")),
            "barrier_gap": safe_float(metrics.get("barrier_gap")),
            "chord_length": safe_float(metrics.get("chord_length")),
            "observed_length": safe_float(metrics.get("observed_length")),
            "length_ratio": safe_float(metrics.get("length_ratio")),
            "length_excess": safe_float(metrics.get("length_excess")),
            "loss_profile_l1_mean": safe_float(metrics.get("loss_profile_l1_mean")),
            "loss_profile_linf": safe_float(metrics.get("loss_profile_linf")),
            "Peakobs": _coalesce_numeric(report_metrics.get("Peakobs"), metrics.get("Peakobs"), observed_shape_metrics.get("peak"), metrics.get("observed_DeltaL")),
            "Pitchord": _coalesce_numeric(report_metrics.get("Pitchord"), metrics.get("Pitchord"), chord_shape_metrics.get("pit")),
            "Pitobs": _coalesce_numeric(report_metrics.get("Pitobs"), metrics.get("Pitobs"), observed_shape_metrics.get("pit")),
            "BarrierGap": _coalesce_numeric(report_metrics.get("BarrierGap"), metrics.get("BarrierGap"), metrics.get("barrier_gap")),
            "LengthRatio": _coalesce_numeric(report_metrics.get("LengthRatio"), metrics.get("LengthRatio"), metrics.get("length_ratio")),
            "LengthExcess": _coalesce_numeric(report_metrics.get("LengthExcess"), metrics.get("LengthExcess"), metrics.get("length_excess")),
            "devL1": _coalesce_numeric(report_metrics.get("devL1"), metrics.get("devL1"), metrics.get("loss_profile_l1_mean")),
            "endpoint_A_loss": endpoint_a_loss,
            "endpoint_A_acc": endpoint_a_acc,
            "endpoint_B_loss": endpoint_b_loss,
            "endpoint_B_acc": endpoint_b_acc,
            "run_summary_json": str(run_dir / "summary.json"),
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
            "train_test_gap": train_test_gap,
            "best_val_loss": safe_float(run_summary.get("best_val_loss")),
            "best_epoch": safe_int(run_summary.get("best_epoch")),
            "quality_signal_scope": _quality_signal_scope(
                endpoint_a_loss=endpoint_a_loss,
                endpoint_b_loss=endpoint_b_loss,
                final_val_loss=final_val_loss,
                final_test_loss=final_test_loss,
            ),
        })

    write_csv(out_csv, COMPARE_RESULTS_COLUMNS, rows)
    summary = {
        "input_root": str(compare_root),
        "out_csv": str(out_csv),
        "num_comparison_json_total": int(len(json_paths)),
        "num_rows": int(len(rows)),
        "num_invalid": int(len(invalid_examples)),
        "num_missing_endpoint_eval": int(num_missing_endpoint_eval),
        "num_missing_run_summary": int(num_missing_run_summary),
        "num_run_summary_errors": int(num_run_summary_errors),
        "invalid_examples": invalid_examples[:20],
        "input_issues": input_issues,
        "limitations": [
            "quality signals prefer first-class test metrics when upstream train summaries provide them",
            "when test metrics are absent upstream, this summary falls back to validation-derived signals only",
        ],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}
