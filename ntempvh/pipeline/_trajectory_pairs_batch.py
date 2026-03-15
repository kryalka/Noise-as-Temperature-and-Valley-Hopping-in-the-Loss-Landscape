from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ._trajectory_pairs_core import (
    TrajectoryPair,
    _parse_explicit_pair,
    _safe_int,
    build_trajectory_pairs_for_run,
    write_pairs_csv,
)

DEFAULT_PAIR_MODE = "milestones"
DEFAULT_MILESTONE_EPOCHS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def normalize_pair_mode(pair_mode: str) -> str:
    pair_mode = str(pair_mode).strip().lower()
    if pair_mode not in {"adjacent", "milestones", "explicit_pairs"}:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")
    return pair_mode



def normalize_milestone_epochs(milestone_epochs: Iterable[int] | None) -> list[int]:
    epochs = list(DEFAULT_MILESTONE_EPOCHS if milestone_epochs is None else milestone_epochs)
    normalized = [_safe_int(value) for value in epochs]
    if any(value is None for value in normalized):
        raise ValueError(f"milestone_epochs must be integers, got {epochs}")
    result = [int(value) for value in normalized if value is not None]
    if sorted(set(result)) != result:
        raise ValueError("milestone_epochs must be strictly increasing and unique")
    return result


def normalize_explicit_pairs(explicit_pairs: Iterable[Any] | None) -> list[tuple[int, int]]:
    if explicit_pairs is None:
        return []
    normalized = [_parse_explicit_pair(value) for value in explicit_pairs]
    if len(set(normalized)) != len(normalized):
        raise ValueError("explicit_pairs must be unique")
    return normalized



def build_pairs_summary(
    *,
    runs_root: str | Path,
    out_csv: str | Path,
    rows: list[TrajectoryPair],
    pair_mode: str,
    milestone_epochs: list[int],
    explicit_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    runs = sorted({row.run_name for row in rows})
    by_lr: dict[str, int] = {}
    by_lr_bs: dict[str, int] = {}
    by_run: dict[str, int] = {}
    by_dataset: dict[str, int] = {}

    for row in rows:
        lr_key = f"{row.learning_rate:.10g}"
        lr_bs_key = f"lr={row.learning_rate:.10g},bs={row.batch_size}"
        by_lr[lr_key] = by_lr.get(lr_key, 0) + 1
        by_lr_bs[lr_bs_key] = by_lr_bs.get(lr_bs_key, 0) + 1
        by_run[row.run_name] = by_run.get(row.run_name, 0) + 1
        by_dataset[row.dataset] = by_dataset.get(row.dataset, 0) + 1

    return {
        "runs_root": str(runs_root),
        "out_csv": str(out_csv),
        "num_runs": len(runs),
        "num_pairs": len(rows),
        "pairs_per_run": by_run,
        "pairs_by_dataset": by_dataset,
        "pairs_by_learning_rate": by_lr,
        "pairs_by_learning_rate_batch_size": by_lr_bs,
        "pair_definition": (
            "explicit within-run checkpoint pairs"
            if pair_mode == "explicit_pairs"
            else "consecutive pairs on a selected epoch grid within one training trajectory"
        ),
        "notes": [
            "pairs are constructed only within a single run",
            f"pair_mode={pair_mode}",
            f"milestone_epochs={milestone_epochs}",
            (
                f"explicit_pairs={explicit_pairs}"
                if pair_mode == "explicit_pairs"
                else "pairs connect consecutive checkpoints from the selected epoch grid"
            ),
            "best.pt and final.pt are intentionally ignored here",
        ],
        "pair_mode": pair_mode,
        "milestone_epochs": milestone_epochs if pair_mode == "milestones" else None,
        "explicit_pairs": explicit_pairs if pair_mode == "explicit_pairs" else None,
    }


def build_trajectory_pairs_batch(
    runs_root: str | Path,
    *,
    out_csv: str | Path,
    out_json: str | Path,
    pair_mode: str = DEFAULT_PAIR_MODE,
    milestone_epochs: Iterable[int] | None = None,
    explicit_pairs: Iterable[Any] | None = None,
) -> Path:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    pair_mode = normalize_pair_mode(pair_mode)
    milestone_epochs = normalize_milestone_epochs(milestone_epochs)
    normalized_explicit_pairs = normalize_explicit_pairs(explicit_pairs)

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No run directories found in: {runs_root}")

    all_rows: list[TrajectoryPair] = []
    errors: list[str] = []
    for run_dir in run_dirs:
        try:
            rows = build_trajectory_pairs_for_run(
                run_dir,
                pair_mode=pair_mode,
                milestone_epochs=milestone_epochs,
                explicit_pairs=normalized_explicit_pairs,
            )
            all_rows.extend(rows)
        except Exception as exc:
            errors.append(f"{run_dir.name}: {exc}")

    if errors:
        raise RuntimeError("Failed to build trajectory pairs for all runs.\nDetails:\n" + "\n".join(errors))
    if not all_rows:
        raise RuntimeError("No trajectory pairs were built.")

    write_pairs_csv(out_csv, all_rows)
    summary = build_pairs_summary(
        runs_root=runs_root,
        out_csv=out_csv,
        rows=all_rows,
        pair_mode=pair_mode,
        milestone_epochs=milestone_epochs,
        explicit_pairs=normalized_explicit_pairs,
    )

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved trajectory pairs: {out_csv}")
    print(f"Saved summary        : {out_json}")
    print(f"Runs processed       : {summary['num_runs']}")
    print(f"Pairs collected      : {summary['num_pairs']}")
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.pipeline.trajectory_pairs",
        description="Build within-run trajectory checkpoint pairs on a selected epoch grid",
    )
    ap.add_argument("--runs_root", default="outputs/runs_lr_bs_grid")
    ap.add_argument("--out_csv", default="outputs/summaries/trajectory_pairs.csv")
    ap.add_argument("--out_json", default="outputs/summaries/trajectory_pairs_summary.json")
    ap.add_argument("--pair_mode", default=DEFAULT_PAIR_MODE, choices=["adjacent", "milestones", "explicit_pairs"])
    ap.add_argument("--milestone_epochs", nargs="*", type=int, default=DEFAULT_MILESTONE_EPOCHS)
    ap.add_argument("--explicit_pairs", nargs="*", default=[], help="Pairs like 50:100 70:100 when pair_mode=explicit_pairs")
    args = ap.parse_args()

    build_trajectory_pairs_batch(
        args.runs_root,
        out_csv=args.out_csv,
        out_json=args.out_json,
        pair_mode=args.pair_mode,
        milestone_epochs=args.milestone_epochs,
        explicit_pairs=args.explicit_pairs,
    )
