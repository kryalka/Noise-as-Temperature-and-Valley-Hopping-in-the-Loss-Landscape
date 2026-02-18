from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ntempvh.utils.checkpoints import collect_epoch_checkpoints, parse_run_name



class InterventionRunError(RuntimeError):
    pass



class NonInterventionRunError(InterventionRunError):
    pass



@dataclass(frozen=True)
class InterventionGeometryCheckpoint:
    checkpoint_role: str
    checkpoint_path: str
    checkpoint_epoch: int



@dataclass(frozen=True)
class InterventionGeometrySelection:
    run_dir: str
    run_name: str
    seed: int
    learning_rate: float
    batch_size: int
    intervention_start_epoch: int
    intervention_end_epoch: int
    intervention_lr_multiplier: float
    intervention_batch_size: int | None
    checkpoints: tuple[InterventionGeometryCheckpoint, ...]



def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise InterventionRunError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None



def resolve_intervention_geometry_checkpoints(
    run_dir: str | Path,
) -> InterventionGeometrySelection:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise InterventionRunError(f"Run path is not a directory: {run_dir}")

    run_name = run_dir.name
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"

    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json for run '{run_name}': {run_config_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for run '{run_name}': {summary_path}")
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory for run '{run_name}': {checkpoints_dir}")

    run_config = _load_json(run_config_path)
    summary = _load_json(summary_path)

    intervention_cfg = dict(run_config.get("intervention", {}) or {})
    if not bool(intervention_cfg.get("enabled", False)):
        raise NonInterventionRunError(f"Run '{run_name}' does not have intervention enabled")

    training_cfg = dict(run_config.get("training", {}) or {})
    parsed = parse_run_name(run_name)

    seed = _safe_int(run_config.get("seed", parsed.get("seed")))
    learning_rate = _safe_float(training_cfg.get("learning_rate", parsed.get("learning_rate")))
    batch_size = _safe_int(training_cfg.get("batch_size", parsed.get("batch_size")))
    start_epoch = _safe_int(intervention_cfg.get("start_epoch"))
    end_epoch = _safe_int(intervention_cfg.get("end_epoch"))
    lr_multiplier = _safe_float(intervention_cfg.get("lr_multiplier"))
    intervention_batch_size = intervention_cfg.get("batch_size", None)
    if intervention_batch_size is not None:
        intervention_batch_size = _safe_int(intervention_batch_size)

    missing = []
    if seed is None:
        missing.append("seed")
    if learning_rate is None:
        missing.append("learning_rate")
    if batch_size is None:
        missing.append("batch_size")
    if start_epoch is None:
        missing.append("intervention.start_epoch")
    if end_epoch is None:
        missing.append("intervention.end_epoch")
    if lr_multiplier is None:
        missing.append("intervention.lr_multiplier")
    if missing:
        raise InterventionRunError(
            f"Could not resolve required intervention metadata for run '{run_name}': {missing}"
        )

    pre_epoch = int(start_epoch) - 1
    post_epoch = int(end_epoch)
    if pre_epoch < 1:
        raise InterventionRunError(
            f"Cannot resolve theta_pre for run '{run_name}': intervention starts at epoch {start_epoch}"
        )

    epoch_to_path = {epoch: path for epoch, path in collect_epoch_checkpoints(checkpoints_dir)}
    if pre_epoch not in epoch_to_path:
        raise FileNotFoundError(
            f"Missing theta_pre checkpoint for run '{run_name}': expected epoch_{pre_epoch:03d}.pt"
        )
    if post_epoch not in epoch_to_path:
        raise FileNotFoundError(
            f"Missing theta_post checkpoint for run '{run_name}': expected epoch_{post_epoch:03d}.pt"
        )

    final_checkpoint = summary.get("final_checkpoint", None)
    final_path = Path(final_checkpoint) if final_checkpoint else checkpoints_dir / "final.pt"
    if not final_path.is_absolute():
        final_path = (run_dir / final_path).resolve()
    if not final_path.exists():
        raise FileNotFoundError(
            f"Missing theta_final checkpoint for run '{run_name}': expected {final_path}"
        )

    final_epoch = _safe_int(summary.get("epochs", None))
    if final_epoch is None:
        raise InterventionRunError(
            f"Could not resolve final checkpoint epoch for run '{run_name}' from {summary_path}"
        )

    checkpoints = (
        InterventionGeometryCheckpoint(
            checkpoint_role="pre",
            checkpoint_path=str(epoch_to_path[pre_epoch]),
            checkpoint_epoch=int(pre_epoch),
        ),
        InterventionGeometryCheckpoint(
            checkpoint_role="post",
            checkpoint_path=str(epoch_to_path[post_epoch]),
            checkpoint_epoch=int(post_epoch),
        ),
        InterventionGeometryCheckpoint(
            checkpoint_role="final",
            checkpoint_path=str(final_path),
            checkpoint_epoch=int(final_epoch),
        ),
    )

    return InterventionGeometrySelection(
        run_dir=str(run_dir),
        run_name=run_name,
        seed=int(seed),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        intervention_start_epoch=int(start_epoch),
        intervention_end_epoch=int(end_epoch),
        intervention_lr_multiplier=float(lr_multiplier),
        intervention_batch_size=(
            None if intervention_batch_size is None else int(intervention_batch_size)
        ),
        checkpoints=checkpoints,
    )
