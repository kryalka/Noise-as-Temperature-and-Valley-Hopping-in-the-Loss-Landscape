from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ntempvh.utils.path_metrics import load_interpolation_profile


def build_interpolation_cfg(
    compare_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    *,
    path_type: str,
) -> dict[str, Any]:
    path_cfg: dict[str, Any] = {
        "type": str(path_type),
        "num_points": int(compare_cfg["path"]["num_points"]),
        "bn_recalib_batches": int(compare_cfg["path"]["bn_recalib_batches"]),
        "pivots": [],
    }
    if path_type == "observed":
        path_cfg["observed"] = dict(compare_cfg["path"]["observed"])
    return {
        "data_root": str(compare_cfg["data_root"]),
        "path": path_cfg,
        "evaluation": dict(compare_cfg["evaluation"]),
        "data": dict(data_cfg),
    }


def profile_deviation_metrics(chord_csv: str | Path, observed_csv: str | Path) -> dict[str, float]:
    t_chord, loss_chord, _ = load_interpolation_profile(chord_csv)
    t_observed, loss_observed, _ = load_interpolation_profile(observed_csv)
    shared_points = max(len(t_chord), len(t_observed), 2)
    shared_t = np.linspace(0.0, 1.0, shared_points)
    chord_interp = np.interp(shared_t, t_chord, loss_chord)
    observed_interp = np.interp(shared_t, t_observed, loss_observed)
    abs_diff = np.abs(observed_interp - chord_interp)
    return {
        "loss_profile_l1_mean": float(np.mean(abs_diff)),
        "loss_profile_linf": float(np.max(abs_diff)),
    }


def append_summary_row(csv_path: Path, columns: list[str], row: dict[str, Any]) -> None:
    if not csv_path.exists():
        csv_path.write_text(",".join(columns) + "\n", encoding="utf-8")
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
