from __future__ import annotations

from pathlib import Path

import numpy as np



def load_interpolation_profile(
    interp_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(interp_csv, delimiter=",", skiprows=1)
    arr = np.atleast_2d(arr)

    t = arr[:, 0].astype(np.float64)
    loss = arr[:, 1].astype(np.float64)
    if arr.shape[1] >= 3:
        acc = arr[:, 2].astype(np.float64)
    else:
        acc = np.full_like(loss, np.nan)

    order = np.argsort(t)
    return t[order], loss[order], acc[order]



def compute_linear_baseline_shape_metrics(
    interp_csv: str | Path,
) -> dict[str, float]:
    t, loss, _ = load_interpolation_profile(interp_csv)
    if len(t) == 0:
        return {
            "peak": 0.0,
            "peak_t": 0.0,
            "pit": 0.0,
            "pit_t": 0.0,
        }

    baseline = (1.0 - t) * float(loss[0]) + t * float(loss[-1])
    diff = loss - baseline

    peak_idx = int(np.argmax(diff))
    pit_idx = int(np.argmin(diff))
    return {
        "peak": float(max(0.0, diff[peak_idx])),
        "peak_t": float(t[peak_idx]),
        "pit": float(max(0.0, -diff[pit_idx])),
        "pit_t": float(t[pit_idx]),
    }
