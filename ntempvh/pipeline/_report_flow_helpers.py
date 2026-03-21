from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from ._report_flow_plan import (
    build_stage_plan,
    default_grid_out_root,
    resolve_grid_out_root,
)




STAGE_COLUMNS = [
    "stage",
    "enabled",
    "status",
    "primary_output",
    "duration_seconds",
    "returncode",
    "command",
    "note",
]



def ensure_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[3]



def write_stage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STAGE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in STAGE_COLUMNS})



def run_stage_command(
    command: list[str],
    *,
    cwd: Path,
    python_bin: str,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHON_BIN"] = python_bin
    src_root = resolve_project_root() / "src"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else str(src_root)
    )
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


