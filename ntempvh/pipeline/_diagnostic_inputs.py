from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from ntempvh.pipeline.trajectory_pairs import build_trajectory_pairs_batch
from ntempvh.utils.checkpoints import build_pair_tag, parse_checkpoint_path
from ntempvh.utils.io import load_yaml


RESOLVED_PAIR_COLUMNS = [
    "run_name",
    "dataset",
    "model",
    "seed",
    "learning_rate",
    "batch_size",
    "epoch_A",
    "epoch_B",
    "ckptA",
    "ckptB",
    "pair_tag",
]


def ensure_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


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



def validate_diagnostic_config(cfg: dict[str, Any]) -> None:
    inputs_cfg = ensure_mapping("inputs", cfg.get("inputs"))
    diagnostic_cfg = ensure_mapping("diagnostics", cfg.get("diagnostics"))
    outputs_cfg = ensure_mapping("outputs", cfg.get("outputs"))

    pairs_csv = str(inputs_cfg.get("pairs_csv", "")).strip()
    runs_root = str(inputs_cfg.get("runs_root", "")).strip()
    if not pairs_csv and not runs_root:
        raise ValueError("diagnostic config must define either inputs.pairs_csv or inputs.runs_root")

    compare_config = str(diagnostic_cfg.get("compare_config", "")).strip()
    geometry_config = str(diagnostic_cfg.get("geometry_config", "")).strip()
    if not compare_config:
        raise ValueError("diagnostic config must define diagnostics.compare_config")
    if not geometry_config:
        raise ValueError("diagnostic config must define diagnostics.geometry_config")

    out_root = str(outputs_cfg.get("out_root", "")).strip()
    if not out_root:
        raise ValueError("diagnostic config must define outputs.out_root")



def load_pairs_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "ckptA" not in fieldnames or "ckptB" not in fieldnames:
            raise ValueError(f"{path} must contain columns ckptA and ckptB")
        return list(reader)


def pair_from_checkpoint_paths(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
) -> dict[str, Any]:
    meta_a = parse_checkpoint_path(ckpt_a)
    ckpt_obj = torch.load(ckpt_a, map_location="cpu")
    return {
        "run_name": str(meta_a.get("run_name", "")),
        "dataset": str(meta_a.get("dataset", ckpt_obj.get("dataset", ""))),
        "model": str(meta_a.get("model", ckpt_obj.get("model", ""))),
        "seed": safe_int(meta_a.get("seed")),
        "learning_rate": safe_float(meta_a.get("learning_rate")),
        "batch_size": safe_int(meta_a.get("batch_size")),
        "epoch_A": safe_int(meta_a.get("epoch")),
        "epoch_B": safe_int(parse_checkpoint_path(ckpt_b).get("epoch")),
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "pair_tag": build_pair_tag(ckpt_a, ckpt_b),
    }


def normalize_pair_row(row: dict[str, Any]) -> dict[str, Any]:
    ckpt_a = str(row.get("ckptA", ""))
    ckpt_b = str(row.get("ckptB", ""))
    meta = pair_from_checkpoint_paths(ckpt_a, ckpt_b)
    return {
        "run_name": str(row.get("run_name", meta["run_name"])),
        "dataset": str(row.get("dataset", meta["dataset"])),
        "model": str(row.get("model", meta["model"])),
        "seed": safe_int(row.get("seed", meta["seed"])),
        "learning_rate": safe_float(row.get("learning_rate", meta["learning_rate"])),
        "batch_size": safe_int(row.get("batch_size", meta["batch_size"])),
        "epoch_A": safe_int(row.get("epoch_A", meta["epoch_A"])),
        "epoch_B": safe_int(row.get("epoch_B", meta["epoch_B"])),
        "ckptA": ckpt_a,
        "ckptB": ckpt_b,
        "pair_tag": str(row.get("pair_tag", meta["pair_tag"])),
    }


def write_resolved_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESOLVED_PAIR_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RESOLVED_PAIR_COLUMNS})



def resolve_pair_input(
    cfg: dict[str, Any],
    *,
    config_path: Path,
    out_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_dir = config_path.parent
    inputs_cfg = ensure_mapping("inputs", cfg.get("inputs"))

    resolved_pairs_csv = out_root / "resolved_pairs.csv"
    resolved_pairs_json = out_root / "resolved_pairs_summary.json"

    pairs_csv_value = str(inputs_cfg.get("pairs_csv", "")).strip()
    if pairs_csv_value:
        source_pairs_csv = resolve_path(pairs_csv_value, base_dir=base_dir)
        raw_pairs = load_pairs_csv(source_pairs_csv)
        rows = [normalize_pair_row(row) for row in raw_pairs]
        write_resolved_pairs_csv(resolved_pairs_csv, rows)
        summary = {
            "input_mode": "pairs_csv",
            "source_pairs_csv": str(source_pairs_csv),
            "resolved_pairs_csv": str(resolved_pairs_csv),
            "num_pairs": int(len(rows)),
        }
        resolved_pairs_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows, summary

    runs_root = resolve_path(str(inputs_cfg.get("runs_root", "")), base_dir=base_dir)
    pair_mode = str(inputs_cfg.get("pair_mode", "milestones"))
    milestone_epochs = list(inputs_cfg.get("milestone_epochs", [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
    explicit_pairs = list(inputs_cfg.get("explicit_pairs", []))

    build_trajectory_pairs_batch(
        runs_root,
        out_csv=resolved_pairs_csv,
        out_json=resolved_pairs_json,
        pair_mode=pair_mode,
        milestone_epochs=milestone_epochs,
        explicit_pairs=explicit_pairs,
    )
    rows = [normalize_pair_row(row) for row in load_pairs_csv(resolved_pairs_csv)]
    write_resolved_pairs_csv(resolved_pairs_csv, rows)
    summary = load_yaml(resolved_pairs_json)
    summary["input_mode"] = "runs_root"
    summary["resolved_pairs_csv"] = str(resolved_pairs_csv)
    return rows, summary
