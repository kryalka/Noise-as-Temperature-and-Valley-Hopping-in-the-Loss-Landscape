from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ntempvh.utils.io import ensure_dir, load_yaml, save_json

from ._report_flow_helpers import STAGE_COLUMNS, resolve_project_root as _resolve_project_root, run_stage_command as _run_stage_command, write_stage_csv as _write_stage_csv
from ._report_flow_paths import ensure_mapping as _ensure_mapping
from ._report_flow_plan import build_stage_plan as _build_stage_plan


def run_report_flow(
    config_path: str | Path,
    *,
    python_bin: str | None = None,
) -> Path:
    project_root = _resolve_project_root()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    cfg = load_yaml(config_path)
    pipeline_cfg = _ensure_mapping("pipeline", cfg.get("pipeline"))
    out_root = Path(str(pipeline_cfg.get("out_root", "outputs/summaries/report_flow")))
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    ensure_dir(out_root)

    python_bin = str(python_bin or sys.executable)
    stages, outputs = _build_stage_plan(cfg, project_root=project_root, python_bin=python_bin)

    rows: list[dict[str, object]] = []
    stage_details: list[dict[str, object]] = []
    failed_stage: str | None = None

    for stage in stages:
        stage_name = str(stage["stage"])
        enabled = bool(stage["enabled"])
        primary_output = str(stage["primary_output"])
        row: dict[str, object] = {
            "stage": stage_name,
            "enabled": enabled,
            "status": "disabled",
            "primary_output": primary_output,
            "duration_seconds": 0.0,
            "returncode": "",
            "command": " ".join(stage["command"]),
            "note": str(stage.get("note", "")),
        }
        detail: dict[str, object] = {**row, "stdout": "", "stderr": ""}

        if not enabled:
            rows.append(row)
            stage_details.append(detail)
            continue

        start = time.time()
        result = _run_stage_command(stage["command"], cwd=project_root, python_bin=python_bin)
        duration = time.time() - start

        output_exists = Path(primary_output).exists()
        if result.returncode == 0 and output_exists:
            status = "ok"
            note = row["note"]
        elif result.returncode == 0 and not output_exists:
            status = "missing_output"
            note = f"{row['note']} | expected output not found"
        else:
            status = "failed"
            note = row["note"]

        row.update({
            "status": status,
            "duration_seconds": float(duration),
            "returncode": int(result.returncode),
            "note": note,
        })
        detail.update({
            **row,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        })
        rows.append(row)
        stage_details.append(detail)

        if status != "ok":
            failed_stage = stage_name
            break

    steps_csv = out_root / "report_flow_steps.csv"
    manifest_json = out_root / "report_flow_manifest.json"
    _write_stage_csv(steps_csv, rows)

    manifest = {
        "config_path": str(config_path),
        "out_root": str(out_root),
        "python_bin": python_bin,
        "steps_csv": str(steps_csv),
        "stage_rows": rows,
        "stage_details": stage_details,
        "outputs": outputs,
        "status": "ok" if failed_stage is None else "failed",
        "failed_stage": failed_stage,
        "limitations": [
            "report flow orchestration reuses existing compute wrappers and preserves upstream artifact contracts",
            "if a stage fails, the flow stops and writes a partial machine-readable manifest for debugging and restart",
        ],
    }
    save_json(manifest_json, manifest)

    print(f"Saved report-flow steps   : {steps_csv}")
    print(f"Saved report-flow manifest: {manifest_json}")
    if failed_stage is not None:
        raise RuntimeError(f"Report flow failed at stage: {failed_stage}")
    return manifest_json


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.pipeline.report_flow",
        description="Run the practical report-oriented orchestration flow on top of existing pipeline blocks",
    )
    ap.add_argument("--config", default="configs/pipeline/report_cifar10.yaml")
    ap.add_argument("--python_bin", default=sys.executable)
    args = ap.parse_args()
    run_report_flow(args.config, python_bin=args.python_bin)


if __name__ == "__main__":
    main()
