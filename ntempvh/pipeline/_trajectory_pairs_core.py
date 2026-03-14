from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ntempvh.utils.checkpoints import collect_epoch_checkpoints, parse_run_name





@dataclass
class TrajectoryPair:
    run_dir: str
    run_name: str
    dataset: str
    model: str
    seed: int
    learning_rate: float
    batch_size: int
    optimizer: str
    weight_decay: float
    momentum: float
    scheduler: str
    epochs_total: int
    epoch_A: int
    epoch_B: int
    ckptA: str
    ckptB: str
    pair_index: int



def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj



def _try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None



def _parse_explicit_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        epoch_a = _safe_int(value[0])
        epoch_b = _safe_int(value[1])
    else:
        text = str(value).strip()
        if ":" not in text:
            raise ValueError(
                f"explicit_pairs entries must look like '50:100' or [50, 100], got {value!r}"
            )
        left, right = text.split(":", maxsplit=1)
        epoch_a = _safe_int(left)
        epoch_b = _safe_int(right)

    if epoch_a is None or epoch_b is None:
        raise ValueError(f"explicit_pairs entries must contain integers, got {value!r}")
    if epoch_a < 1 or epoch_b < 1 or epoch_a >= epoch_b:
        raise ValueError(f"explicit_pairs must satisfy 1 <= epoch_A < epoch_B, got {value!r}")
    return int(epoch_a), int(epoch_b)



def extract_run_meta(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_name = run_dir.name

    meta: dict[str, Any] = {
        "run_name": run_name,
        "dataset": None,
        "model": None,
        "seed": None,
        "learning_rate": None,
        "batch_size": None,
        "optimizer": None,
        "weight_decay": None,
        "momentum": None,
        "scheduler": None,
        "epochs_total": None,
    }

    parsed = parse_run_name(run_name)
    meta.update({k: v for k, v in parsed.items() if k in meta and v is not None})

    run_cfg = _try_load_json(run_dir / "run_config.json")
    if run_cfg is not None:
        meta["dataset"] = str(run_cfg.get("dataset", meta["dataset"] or ""))
        meta["model"] = str(run_cfg.get("model", meta["model"] or ""))

        training = run_cfg.get("training", {}) or {}
        meta["learning_rate"] = _safe_float(training.get("learning_rate", meta["learning_rate"]))
        meta["batch_size"] = _safe_int(training.get("batch_size", meta["batch_size"]))
        meta["optimizer"] = str(training.get("optimizer", meta["optimizer"] or ""))
        meta["weight_decay"] = _safe_float(training.get("weight_decay", meta["weight_decay"]))
        meta["momentum"] = _safe_float(training.get("momentum", meta["momentum"]))
        meta["scheduler"] = str(training.get("scheduler", meta["scheduler"] or ""))
        meta["epochs_total"] = _safe_int(training.get("epochs", meta["epochs_total"]))

        seed_val = run_cfg.get("seed")
        if seed_val is not None:
            meta["seed"] = _safe_int(seed_val)

    manifest = _try_load_json(run_dir / "cli_manifest.json")
    if manifest is not None and meta["seed"] is None:
        meta["seed"] = _safe_int(manifest.get("seed"))

    summary = _try_load_json(run_dir / "summary.json")
    if summary is not None and meta["epochs_total"] is None:
        meta["epochs_total"] = _safe_int(summary.get("epochs"))

    missing = [k for k, v in meta.items() if k not in {"run_name"} and v in (None, "")]
    if missing:
        raise ValueError(
            f"Could not fully resolve metadata for run '{run_name}'. Missing: {missing}"
        )

    return meta



def select_epochs_for_pairing(
    *,
    run_name: str,
    epoch_ckpts: list[tuple[int, Path]],
    pair_mode: str,
    milestone_epochs: list[int],
    explicit_pairs: list[tuple[int, int]] | None = None,
) -> list[int] | list[tuple[int, int]]:
    if len(epoch_ckpts) < 2:
        raise ValueError(
            f"Run '{run_name}' has fewer than 2 epoch checkpoints; found {len(epoch_ckpts)}"
        )

    epoch_to_path = {epoch: path for epoch, path in epoch_ckpts}
    available_epochs = sorted(epoch_to_path.keys())

    if pair_mode == "adjacent":
        selected_epochs = available_epochs
    elif pair_mode == "milestones":
        selected_epochs = [epoch for epoch in milestone_epochs if epoch in epoch_to_path]
    elif pair_mode == "explicit_pairs":
        explicit_pairs = list(explicit_pairs or [])
        if not explicit_pairs:
            raise ValueError("pair_mode=explicit_pairs requires a non-empty explicit_pairs list")
        missing = [
            f"{epoch_a}:{epoch_b}"
            for epoch_a, epoch_b in explicit_pairs
            if epoch_a not in epoch_to_path or epoch_b not in epoch_to_path
        ]
        if missing:
            raise ValueError(
                f"Run '{run_name}' is missing checkpoint epochs required by explicit_pairs: {missing}"
            )
        return explicit_pairs
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    if len(selected_epochs) < 2:
        raise ValueError(
            f"Run '{run_name}' has fewer than 2 selected epochs under mode={pair_mode}. "
            f"Selected: {selected_epochs}"
        )
    return selected_epochs



def build_trajectory_pairs_for_run(
    run_dir: str | Path,
    *,
    pair_mode: str,
    milestone_epochs: list[int],
    explicit_pairs: list[Any] | None = None,
) -> list[TrajectoryPair]:
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")

    meta = extract_run_meta(run_dir)
    epoch_ckpts = collect_epoch_checkpoints(checkpoints_dir)
    normalized_explicit_pairs = [_parse_explicit_pair(value) for value in (explicit_pairs or [])]
    selected = select_epochs_for_pairing(
        run_name=run_dir.name,
        epoch_ckpts=epoch_ckpts,
        pair_mode=pair_mode,
        milestone_epochs=milestone_epochs,
        explicit_pairs=normalized_explicit_pairs,
    )
    epoch_to_path = {epoch: path for epoch, path in epoch_ckpts}

    pairs: list[TrajectoryPair] = []
    if pair_mode == "explicit_pairs":
        selected_pairs = list(selected)
    else:
        selected_epochs = list(selected)
        selected_pairs = list(zip(selected_epochs[:-1], selected_epochs[1:]))

    for pair_index, (epoch_a, epoch_b) in enumerate(selected_pairs):
        pairs.append(
            TrajectoryPair(
                run_dir=str(run_dir),
                run_name=str(meta["run_name"]),
                dataset=str(meta["dataset"]),
                model=str(meta["model"]),
                seed=int(meta["seed"]),
                learning_rate=float(meta["learning_rate"]),
                batch_size=int(meta["batch_size"]),
                optimizer=str(meta["optimizer"]),
                weight_decay=float(meta["weight_decay"]),
                momentum=float(meta["momentum"]),
                scheduler=str(meta["scheduler"]),
                epochs_total=int(meta["epochs_total"]),
                epoch_A=int(epoch_a),
                epoch_B=int(epoch_b),
                ckptA=str(epoch_to_path[epoch_a]),
                ckptB=str(epoch_to_path[epoch_b]),
                pair_index=int(pair_index),
            )
        )

    return pairs



def write_pairs_csv(path: str | Path, rows: Iterable[TrajectoryPair]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(TrajectoryPair.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

