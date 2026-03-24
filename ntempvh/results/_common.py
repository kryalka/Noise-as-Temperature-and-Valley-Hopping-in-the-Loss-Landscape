from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ntempvh.utils.artifacts import load_json_artifact




def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None



def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if safe_float(value) is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def fraction_true(values: list[Any]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if normalize_bool(value)) / len(values))


def preferred_value(row: dict[str, Any], primary: str, fallback: str) -> Any:
    value = row.get(primary, None)
    if value not in (None, ""):
        return value
    return row.get(fallback, None)



def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})



def read_csv_rows(
    path: Path,
    *,
    required_columns: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    if not path.exists():
        issues.append(f"Missing required input csv: {path}")
        return [], issues

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_columns = [name for name in required_columns if name not in fieldnames]
        if missing_columns:
            issues.append(f"{path.name} is missing required columns: {missing_columns}")
            return [], issues
        return list(reader), issues



def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = load_json_artifact(path)
    except Exception as exc:
        return None, str(exc)

    if not isinstance(obj, dict):
        return None, f"Expected JSON object in {path}, got {type(obj).__name__}"
    return obj, None



def load_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return None, f"Line {lineno} in {path} is not a JSON object"
                rows.append(obj)
    except Exception as exc:
        return None, str(exc)
    return rows, None
