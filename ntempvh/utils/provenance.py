from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable




def _normalize_paths(paths: Iterable[str | Path] | None) -> list[str]:
    if paths is None:
        return []
    normalized: list[str] = []
    for value in paths:
        normalized.append(str(Path(value).expanduser().resolve()))
    return normalized



def _find_repo_root(start_path: str | Path | None) -> Path | None:
    if start_path is None:
        candidate = Path.cwd().resolve()
    else:
        candidate = Path(start_path).expanduser().resolve()
        if candidate.is_file():
            candidate = candidate.parent

    for path in (candidate, *candidate.parents):
        if (path / ".git").exists():
            return path

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=candidate,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    repo_root = result.stdout.strip()
    return Path(repo_root).resolve() if repo_root else None



def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None



def _collect_git_info(repo_root: Path | None) -> dict[str, object] | None:
    if repo_root is None:
        return None

    commit = _run_git(repo_root, "rev-parse", "HEAD")
    branch = _run_git(repo_root, "branch", "--show-current")
    status = _run_git(repo_root, "status", "--short")

    return {
        "repo_root": str(repo_root),
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
    }



def build_provenance(
    *,
    project_root: str | Path | None = None,
    config_paths: Iterable[str | Path] | None = None,
    input_paths: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    repo_root = _find_repo_root(project_root)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "config_paths": _normalize_paths(config_paths),
        "input_paths": _normalize_paths(input_paths),
        "git": _collect_git_info(repo_root),
    }
