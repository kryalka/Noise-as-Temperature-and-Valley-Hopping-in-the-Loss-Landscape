from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ntempvh.utils.path_metrics import (
    compute_linear_baseline_shape_metrics,
    load_interpolation_profile,
)




def _write_interp_csv(path: Path, rows: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ["t,val_loss,val_acc"]
    text.extend(f"{t},{loss},{acc}" for t, loss, acc in rows)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")



def test_load_interpolation_profile_sorts_points_by_t(tmp_path: Path) -> None:
    csv_path = tmp_path / "interp.csv"
    _write_interp_csv(
        csv_path,
        [
            (1.0, 0.8, 0.7),
            (0.0, 1.0, 0.5),
            (0.5, 0.9, 0.6),
        ],
    )

    t, loss, acc = load_interpolation_profile(csv_path)

    assert np.allclose(t, [0.0, 0.5, 1.0])
    assert np.allclose(loss, [1.0, 0.9, 0.8])
    assert np.allclose(acc, [0.5, 0.6, 0.7])



def test_compute_linear_baseline_shape_metrics_returns_peak_and_pit(tmp_path: Path) -> None:
    csv_path = tmp_path / "interp.csv"
    _write_interp_csv(
        csv_path,
        [
            (0.0, 1.0, 0.5),
            (0.25, 1.4, 0.52),
            (0.5, 0.7, 0.6),
            (1.0, 1.0, 0.7),
        ],
    )

    metrics = compute_linear_baseline_shape_metrics(csv_path)

    assert metrics["peak"] == pytest.approx(0.4)
    assert metrics["peak_t"] == pytest.approx(0.25)
    assert metrics["pit"] == pytest.approx(0.3)
    assert metrics["pit_t"] == pytest.approx(0.5)



def test_compute_linear_baseline_shape_metrics_clamps_negative_peak_and_pit(tmp_path: Path) -> None:
    csv_path = tmp_path / "interp.csv"
    _write_interp_csv(
        csv_path,
        [
            (0.0, 1.0, 0.5),
            (0.5, 1.0, 0.6),
            (1.0, 1.0, 0.7),
        ],
    )

    metrics = compute_linear_baseline_shape_metrics(csv_path)

    assert metrics["peak"] == 0.0
    assert metrics["pit"] == 0.0
