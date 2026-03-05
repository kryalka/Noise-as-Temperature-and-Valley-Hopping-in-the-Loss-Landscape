from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ntempvh.eval.geometry import compute_geometry
from ntempvh.utils.artifacts import build_geometry_artifact_context, load_json_artifact
from ntempvh.utils.intervention_runs import (
    InterventionGeometryCheckpoint,
    InterventionGeometrySelection,
    NonInterventionRunError,
    resolve_intervention_geometry_checkpoints,
)
from ntempvh.utils.io import ensure_dir, load_yaml

SUMMARY_COLUMNS = [
    "run_dir",
    "run_name",
    "seed",
    "learning_rate",
    "batch_size",
    "intervention_start_epoch",
    "intervention_end_epoch",
    "intervention_lr_multiplier",
    "intervention_batch_size",
    "checkpoint_role",
    "checkpoint_path",
    "checkpoint_epoch",
    "geometry_json",
    "status",
    "reason",
    "kappa_tr",
    "kappa_tr_std",
    "sigma_kappa",
    "anisotropy",
    "base_loss",
    "base_acc",
]



def _existing_geometry_artifact(
    ckpt_path: str,
    geometry_cfg: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    success_artifact = build_geometry_artifact_context(ckpt_path, geometry_cfg, failed=False)
    failure_artifact = build_geometry_artifact_context(ckpt_path, geometry_cfg, failed=True)

    success_path = out_dir / f"{success_artifact['stem']}.json"
    failure_path = out_dir / f"{failure_artifact['stem']}.json"

    if success_path.exists():
        return success_path
    if failure_path.exists():
        return failure_path
    return None



def _run_or_reuse_geometry(
    *,
    ckpt_path: str,
    geometry_cfg_path: str,
    geometry_cfg: dict[str, Any],
    out_dir: Path,
) -> Path:
    existing = _existing_geometry_artifact(ckpt_path, geometry_cfg, out_dir)
    if existing is not None:
        return existing
    return compute_geometry(ckpt_path, geometry_cfg_path, str(out_dir))



def _row_from_geometry_payload(
    selection: InterventionGeometrySelection,
    checkpoint: InterventionGeometryCheckpoint,
    geometry_json: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = "ok"
    reason = ""
    kappa_tr = payload.get("kappa_tr")
    kappa_tr_std = payload.get("kappa_tr_std")
    sigma_kappa = payload.get("sigma_kappa", kappa_tr_std)
    anisotropy = payload.get("anisotropy", sigma_kappa)
    base = payload.get("base", {}) or {}
    base_loss = base.get("loss")
    base_acc = base.get("acc")

    if str(payload.get("status", "")).lower() == "failed":
        status = "failed"
        reason = str(payload.get("reason", ""))
        kappa_tr = None
        kappa_tr_std = None
        sigma_kappa = None
        anisotropy = None
        bn_base = payload.get("bn_base", {}) or {}
        raw_base = payload.get("raw_base", {}) or {}
        base_loss = bn_base.get("loss", raw_base.get("loss"))
        base_acc = bn_base.get("acc", raw_base.get("acc"))

    return {
        "run_dir": selection.run_dir,
        "run_name": selection.run_name,
        "seed": selection.seed,
        "learning_rate": selection.learning_rate,
        "batch_size": selection.batch_size,
        "intervention_start_epoch": selection.intervention_start_epoch,
        "intervention_end_epoch": selection.intervention_end_epoch,
        "intervention_lr_multiplier": selection.intervention_lr_multiplier,
        "intervention_batch_size": selection.intervention_batch_size,
        "checkpoint_role": checkpoint.checkpoint_role,
        "checkpoint_path": checkpoint.checkpoint_path,
        "checkpoint_epoch": checkpoint.checkpoint_epoch,
        "geometry_json": geometry_json,
        "status": status,
        "reason": reason,
        "kappa_tr": kappa_tr,
        "kappa_tr_std": kappa_tr_std,
        "sigma_kappa": sigma_kappa,
        "anisotropy": anisotropy,
        "base_loss": base_loss,
        "base_acc": base_acc,
    }



def _error_row(
    selection: InterventionGeometrySelection,
    checkpoint: InterventionGeometryCheckpoint,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_dir": selection.run_dir,
        "run_name": selection.run_name,
        "seed": selection.seed,
        "learning_rate": selection.learning_rate,
        "batch_size": selection.batch_size,
        "intervention_start_epoch": selection.intervention_start_epoch,
        "intervention_end_epoch": selection.intervention_end_epoch,
        "intervention_lr_multiplier": selection.intervention_lr_multiplier,
        "intervention_batch_size": selection.intervention_batch_size,
        "checkpoint_role": checkpoint.checkpoint_role,
        "checkpoint_path": checkpoint.checkpoint_path,
        "checkpoint_epoch": checkpoint.checkpoint_epoch,
        "geometry_json": "",
        "status": "error",
        "reason": reason,
        "kappa_tr": None,
        "kappa_tr_std": None,
        "sigma_kappa": None,
        "anisotropy": None,
        "base_loss": None,
        "base_acc": None,
    }



def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def run_intervention_geometry_batch(
    runs_root: str,
    geometry_cfg_path: str,
    out_dir: str,
) -> Path:
    runs_root_path = Path(runs_root)
    if not runs_root_path.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root_path}")

    geometry_cfg = load_yaml(geometry_cfg_path)
    out_dir_path = ensure_dir(out_dir)
    summary_csv = out_dir_path / "intervention_geometry_summary.csv"
    summary_json = out_dir_path / "intervention_geometry_summary.json"

    rows: list[dict[str, Any]] = []
    run_errors: list[dict[str, str]] = []
    num_runs_scanned = 0
    num_intervention_runs = 0
    num_non_intervention_runs = 0

    run_dirs = sorted(path for path in runs_root_path.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        num_runs_scanned += 1
        try:
            selection = resolve_intervention_geometry_checkpoints(run_dir)
        except NonInterventionRunError:
            num_non_intervention_runs += 1
            continue
        except Exception as exc:
            run_errors.append({
                "run_dir": str(run_dir),
                "error": str(exc),
            })
            continue

        num_intervention_runs += 1
        for checkpoint in selection.checkpoints:
            try:
                geometry_json = _run_or_reuse_geometry(
                    ckpt_path=checkpoint.checkpoint_path,
                    geometry_cfg_path=geometry_cfg_path,
                    geometry_cfg=geometry_cfg,
                    out_dir=out_dir_path,
                )
                payload = load_json_artifact(geometry_json)
                rows.append(
                    _row_from_geometry_payload(
                        selection,
                        checkpoint,
                        str(geometry_json),
                        payload,
                    )
                )
            except Exception as exc:
                rows.append(
                    _error_row(
                        selection,
                        checkpoint,
                        reason=str(exc),
                    )
                )

    _write_summary_csv(summary_csv, rows)

    summary: dict[str, Any] = {
        "runs_root": str(runs_root_path),
        "geometry_config": str(Path(geometry_cfg_path)),
        "out_dir": str(out_dir_path),
        "summary_csv": str(summary_csv),
        "num_runs_scanned": int(num_runs_scanned),
        "num_intervention_runs": int(num_intervention_runs),
        "num_non_intervention_runs": int(num_non_intervention_runs),
        "num_rows": int(len(rows)),
        "num_success": int(sum(1 for row in rows if row["status"] == "ok")),
        "num_failed": int(sum(1 for row in rows if row["status"] == "failed")),
        "num_errors": int(sum(1 for row in rows if row["status"] == "error")),
        "num_run_errors": int(len(run_errors)),
        "run_error_examples": run_errors[:20],
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved intervention-geometry summary csv: {summary_csv}")
    print(f"Saved intervention-geometry summary json: {summary_json}")
    print(f"Runs scanned                    : {num_runs_scanned}")
    print(f"Intervention runs processed     : {num_intervention_runs}")
    print(f"Non-intervention runs skipped   : {num_non_intervention_runs}")
    print(f"Summary rows                    : {len(rows)}")

    return summary_json



def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.eval.intervention_geometry",
        description="Run geometry on pre/post/final checkpoints for intervention runs",
    )
    ap.add_argument("--runs_root", default="outputs/runs_lr_bs_grid")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/artifacts/geometry_intervention")
    args = ap.parse_args()

    run_intervention_geometry_batch(args.runs_root, args.config, args.out)


if __name__ == "__main__":
    main()
