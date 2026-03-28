from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ntempvh.eval.intervention_geometry import SUMMARY_COLUMNS as INTERVENTION_GEOMETRY_ROLE_INPUT_COLUMNS
from ntempvh.utils.io import ensure_dir, save_json

from ._common import load_json_object, load_jsonl_rows, safe_float, safe_int, write_csv
from ._pipeline_schema import INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS, INTERVENTION_RUN_RESULTS_COLUMNS


def _coalesce_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return None



def aggregate_intervention_runs(
    runs_root: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    out_dir = ensure_dir(out_dir)
    out_csv = out_dir / "intervention_runs_results.csv"
    out_json = out_dir / "intervention_runs_results_summary.json"

    rows: list[dict[str, Any]] = []
    invalid_examples: list[dict[str, str]] = []
    num_runs_scanned = 0
    num_non_intervention_runs = 0
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir()) if runs_root.exists() else []

    for run_dir in run_dirs:
        num_runs_scanned += 1
        run_config_path = run_dir / "run_config.json"
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.jsonl"
        checkpoints_dir = run_dir / "checkpoints"

        if not run_config_path.exists() or not summary_path.exists():
            invalid_examples.append({"run_dir": str(run_dir), "error": "Missing run_config.json or summary.json"})
            continue

        run_config, run_config_error = load_json_object(run_config_path)
        summary, summary_error = load_json_object(summary_path)
        if run_config_error is not None or summary_error is not None or run_config is None or summary is None:
            invalid_examples.append({"run_dir": str(run_dir), "error": run_config_error or summary_error or "invalid_json"})
            continue

        intervention_cfg = dict(run_config.get("intervention", {}) or {})
        if not bool(intervention_cfg.get("enabled", False)):
            num_non_intervention_runs += 1
            continue

        training_cfg = dict(run_config.get("training", {}) or {})
        start_epoch = safe_int(intervention_cfg.get("start_epoch"))
        end_epoch = safe_int(intervention_cfg.get("end_epoch"))
        effective_batch_size = intervention_cfg.get("batch_size", training_cfg.get("batch_size"))

        metrics_rows, metrics_error = ([], None)
        if metrics_path.exists():
            parsed_rows, metrics_error = load_jsonl_rows(metrics_path)
            metrics_rows = parsed_rows or []
        else:
            metrics_error = f"Missing metrics.jsonl: {metrics_path}"

        pre_epoch = int(start_epoch - 1) if start_epoch is not None else None
        post_epoch = end_epoch
        pre_ckpt_exists = pre_epoch is not None and pre_epoch >= 1 and (checkpoints_dir / f"epoch_{pre_epoch:03d}.pt").exists()
        post_ckpt_exists = post_epoch is not None and (checkpoints_dir / f"epoch_{int(post_epoch):03d}.pt").exists()
        final_checkpoint = summary.get("final_checkpoint", None)
        final_checkpoint_path = Path(str(final_checkpoint)) if final_checkpoint else checkpoints_dir / "final.pt"
        if not final_checkpoint_path.is_absolute():
            final_checkpoint_path = run_dir / final_checkpoint_path
        final_ckpt_exists = final_checkpoint_path.exists()

        reasons: list[str] = []
        if metrics_error is not None:
            reasons.append(metrics_error)
        if pre_epoch is not None and not pre_ckpt_exists:
            reasons.append(f"Missing expected pre checkpoint epoch_{pre_epoch:03d}.pt")
        if post_epoch is not None and not post_ckpt_exists:
            reasons.append(f"Missing expected post checkpoint epoch_{int(post_epoch):03d}.pt")
        if not final_ckpt_exists:
            reasons.append(f"Missing final checkpoint: {final_checkpoint_path}")

        final_train_acc = safe_float(summary.get("final_train_acc"))
        final_test_acc = safe_float(summary.get("final_test_acc"))
        rows.append({
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "seed": safe_int(run_config.get("seed")),
            "learning_rate": safe_float(training_cfg.get("learning_rate")),
            "batch_size": safe_int(training_cfg.get("batch_size")),
            "epochs_total": safe_int(summary.get("epochs", training_cfg.get("epochs"))),
            "run_config_json": str(run_config_path),
            "summary_json": str(summary_path),
            "metrics_jsonl": str(metrics_path),
            "intervention_enabled": True,
            "intervention_start_epoch": start_epoch,
            "intervention_end_epoch": end_epoch,
            "intervention_lr_multiplier": safe_float(intervention_cfg.get("lr_multiplier")),
            "intervention_batch_size": safe_int(intervention_cfg.get("batch_size")),
            "intervention_effective_batch_size": safe_int(effective_batch_size),
            "num_intervention_epochs": safe_int((summary.get("intervention", {}) or {}).get("num_intervention_epochs")),
            "num_metrics_rows": int(len(metrics_rows)),
            "num_intervention_metric_rows": int(sum(1 for row in metrics_rows if bool(row.get("is_intervention_epoch", False)))),
            "expected_pre_epoch": pre_epoch,
            "expected_post_epoch": post_epoch,
            "has_pre_checkpoint": bool(pre_ckpt_exists),
            "has_post_checkpoint": bool(post_ckpt_exists),
            "has_final_checkpoint": bool(final_ckpt_exists),
            "final_checkpoint": str(final_checkpoint_path),
            "final_train_loss": safe_float(summary.get("final_train_loss")),
            "final_train_acc": final_train_acc,
            "final_val_loss": safe_float(summary.get("final_val_loss")),
            "final_val_acc": safe_float(summary.get("final_val_acc")),
            "final_test_loss": safe_float(summary.get("final_test_loss")),
            "final_test_acc": final_test_acc,
            "train_test_gap": _coalesce_numeric(summary.get("train_test_gap"), None if final_train_acc is None or final_test_acc is None else float(final_train_acc - final_test_acc)),
            "best_val_loss": safe_float(summary.get("best_val_loss")),
            "best_epoch": safe_int(summary.get("best_epoch")),
            "status": "ok" if not reasons else "partial",
            "reason": " | ".join(reasons),
        })

    write_csv(out_csv, INTERVENTION_RUN_RESULTS_COLUMNS, rows)
    summary = {
        "runs_root": str(runs_root),
        "out_csv": str(out_csv),
        "num_runs_scanned": int(num_runs_scanned),
        "num_non_intervention_runs": int(num_non_intervention_runs),
        "num_rows": int(len(rows)),
        "num_invalid_runs": int(len(invalid_examples)),
        "num_partial_rows": int(sum(1 for row in rows if row["status"] == "partial")),
        "invalid_examples": invalid_examples[:20],
        "input_issues": [] if runs_root.exists() else [f"Runs root not found: {runs_root}"],
    }
    save_json(out_json, summary)
    return {"rows": rows, "csv": out_csv, "summary_json": out_json, "summary": summary}



def aggregate_intervention_geometry(
    intervention_geometry_summary_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    source_csv = Path(intervention_geometry_summary_csv)
    out_dir = ensure_dir(out_dir)
    role_csv = out_dir / "intervention_geometry_roles_results.csv"
    role_json = out_dir / "intervention_geometry_roles_results_summary.json"
    run_csv = out_dir / "intervention_geometry_runs_results.csv"
    run_json = out_dir / "intervention_geometry_runs_results_summary.json"

    role_rows: list[dict[str, Any]] = []
    input_issues: list[str] = []
    duplicate_role_examples: list[dict[str, str]] = []
    if not source_csv.exists():
        input_issues.append(f"Intervention geometry summary not found: {source_csv}")
    else:
        with open(source_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing_cols = [col for col in INTERVENTION_GEOMETRY_ROLE_INPUT_COLUMNS if col not in fieldnames]
            if missing_cols:
                input_issues.append(f"Intervention geometry summary is missing required columns: {missing_cols}")
            else:
                for row in reader:
                    role_rows.append({col: row.get(col, "") for col in INTERVENTION_GEOMETRY_ROLE_INPUT_COLUMNS})

    write_csv(role_csv, list(INTERVENTION_GEOMETRY_ROLE_INPUT_COLUMNS), role_rows)
    save_json(role_json, {
        "input_csv": str(source_csv),
        "out_csv": str(role_csv),
        "num_rows": int(len(role_rows)),
        "num_ok": int(sum(1 for row in role_rows if row.get("status") == "ok")),
        "num_failed": int(sum(1 for row in role_rows if row.get("status") == "failed")),
        "num_error": int(sum(1 for row in role_rows if row.get("status") == "error")),
        "invalid_examples": [],
        "input_issues": input_issues,
    })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in role_rows:
        grouped.setdefault((str(row.get("run_dir", "")), str(row.get("run_name", ""))), []).append(row)

    run_rows: list[dict[str, Any]] = []
    for rows in (grouped[key] for key in sorted(grouped)):
        role_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            role = str(row.get("checkpoint_role", ""))
            if role in role_map:
                duplicate_role_examples.append({"run_dir": str(row.get("run_dir", "")), "run_name": str(row.get("run_name", "")), "checkpoint_role": role})
                continue
            role_map[role] = row

        sample = rows[0]
        missing_roles = [role for role in ("pre", "post", "final") if role not in role_map]
        reasons = [f"Missing roles: {missing_roles}"] if missing_roles else []
        for role_name in ("pre", "post", "final"):
            role_row = role_map.get(role_name)
            if role_row is not None and str(role_row.get("status", "")) not in {"", "ok"}:
                reasons.append(f"{role_name}:{role_row.get('status', '')}:{role_row.get('reason', '')}")

        role_value = lambda role_name, field: role_map.get(role_name, {}).get(field)  
        pre_kappa = safe_float(role_value("pre", "kappa_tr"))
        post_kappa = safe_float(role_value("post", "kappa_tr"))
        final_kappa = safe_float(role_value("final", "kappa_tr"))
        run_rows.append({
            "run_dir": str(sample.get("run_dir", "")),
            "run_name": str(sample.get("run_name", "")),
            "seed": safe_int(sample.get("seed")),
            "learning_rate": safe_float(sample.get("learning_rate")),
            "batch_size": safe_int(sample.get("batch_size")),
            "intervention_start_epoch": safe_int(sample.get("intervention_start_epoch")),
            "intervention_end_epoch": safe_int(sample.get("intervention_end_epoch")),
            "intervention_lr_multiplier": safe_float(sample.get("intervention_lr_multiplier")),
            "intervention_batch_size": safe_int(sample.get("intervention_batch_size")),
            "num_roles_present": int(len(role_map)),
            "status": "ok" if not reasons and len(role_map) == 3 else "partial",
            "reason": " | ".join(reasons),
            "pre_status": str(role_value("pre", "status") or ""),
            "pre_reason": str(role_value("pre", "reason") or ""),
            "pre_checkpoint_path": str(role_value("pre", "checkpoint_path") or ""),
            "pre_checkpoint_epoch": safe_int(role_value("pre", "checkpoint_epoch")),
            "pre_geometry_json": str(role_value("pre", "geometry_json") or ""),
            "pre_kappa_tr": pre_kappa,
            "pre_kappa_tr_std": safe_float(role_value("pre", "kappa_tr_std")),
            "pre_sigma_kappa": _coalesce_numeric(role_value("pre", "sigma_kappa"), role_value("pre", "kappa_tr_std")),
            "pre_anisotropy": _coalesce_numeric(role_value("pre", "anisotropy"), role_value("pre", "sigma_kappa"), role_value("pre", "kappa_tr_std")),
            "pre_base_loss": safe_float(role_value("pre", "base_loss")),
            "pre_base_acc": safe_float(role_value("pre", "base_acc")),
            "post_status": str(role_value("post", "status") or ""),
            "post_reason": str(role_value("post", "reason") or ""),
            "post_checkpoint_path": str(role_value("post", "checkpoint_path") or ""),
            "post_checkpoint_epoch": safe_int(role_value("post", "checkpoint_epoch")),
            "post_geometry_json": str(role_value("post", "geometry_json") or ""),
            "post_kappa_tr": post_kappa,
            "post_kappa_tr_std": safe_float(role_value("post", "kappa_tr_std")),
            "post_sigma_kappa": _coalesce_numeric(role_value("post", "sigma_kappa"), role_value("post", "kappa_tr_std")),
            "post_anisotropy": _coalesce_numeric(role_value("post", "anisotropy"), role_value("post", "sigma_kappa"), role_value("post", "kappa_tr_std")),
            "post_base_loss": safe_float(role_value("post", "base_loss")),
            "post_base_acc": safe_float(role_value("post", "base_acc")),
            "final_status": str(role_value("final", "status") or ""),
            "final_reason": str(role_value("final", "reason") or ""),
            "final_checkpoint_path": str(role_value("final", "checkpoint_path") or ""),
            "final_checkpoint_epoch": safe_int(role_value("final", "checkpoint_epoch")),
            "final_geometry_json": str(role_value("final", "geometry_json") or ""),
            "final_kappa_tr": final_kappa,
            "final_kappa_tr_std": safe_float(role_value("final", "kappa_tr_std")),
            "final_sigma_kappa": _coalesce_numeric(role_value("final", "sigma_kappa"), role_value("final", "kappa_tr_std")),
            "final_anisotropy": _coalesce_numeric(role_value("final", "anisotropy"), role_value("final", "sigma_kappa"), role_value("final", "kappa_tr_std")),
            "final_base_loss": safe_float(role_value("final", "base_loss")),
            "final_base_acc": safe_float(role_value("final", "base_acc")),
            "delta_kappa_post_minus_pre": None if pre_kappa is None or post_kappa is None else float(post_kappa - pre_kappa),
            "delta_kappa_final_minus_pre": None if pre_kappa is None or final_kappa is None else float(final_kappa - pre_kappa),
        })

    write_csv(run_csv, INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS, run_rows)
    run_summary = {
        "input_csv": str(source_csv),
        "out_csv": str(run_csv),
        "num_rows": int(len(run_rows)),
        "num_partial_rows": int(sum(1 for row in run_rows if row["status"] != "ok")),
        "num_duplicate_roles": int(len(duplicate_role_examples)),
        "duplicate_role_examples": duplicate_role_examples[:20],
        "input_issues": input_issues,
    }
    save_json(run_json, run_summary)
    return {
        "role_rows": role_rows,
        "run_rows": run_rows,
        "role_csv": role_csv,
        "role_summary_json": role_json,
        "run_csv": run_csv,
        "run_summary_json": run_json,
        "role_summary": {"input_issues": input_issues},
        "run_summary": run_summary,
    }
