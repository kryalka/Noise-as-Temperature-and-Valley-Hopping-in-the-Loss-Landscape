from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.results.figure_outputs import (
    COMPARE_PATH_FIGURE_DATA_COLUMNS,
    GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS,
    INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS,
    REGIME_HEATMAP_CELL_COLUMNS,
    run_figure_outputs,
)

from ntempvh.results.final_outputs import (
    BASELINE_REGIME_TABLE_COLUMNS,
    COMPARE_SECTION_SUMMARY_COLUMNS,
    GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
    INTERVENTION_WINDOW_SUMMARY_COLUMNS,
)



def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def test_run_figure_outputs_empty_inputs_write_schema_stable_outputs(tmp_path: Path) -> None:
    final_outputs_root = tmp_path / "final_outputs"
    final_outputs_root.mkdir()
    out_dir = tmp_path / "figure_outputs"

    manifest_path = run_figure_outputs(final_outputs_root=str(final_outputs_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["regime_heatmap_cells"] == 0
    assert manifest["counts"]["compare_plot_rows"] == 0
    assert manifest["counts"]["intervention_plot_rows"] == 0
    assert manifest["counts"]["geometry_plot_rows"] == 0
    assert manifest["status"] == "partial"

    assert (out_dir / "regime_heatmap_cells.csv").read_text(encoding="utf-8").strip() == ",".join(REGIME_HEATMAP_CELL_COLUMNS)
    assert (out_dir / "compare_paths_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_PATH_FIGURE_DATA_COLUMNS)
    assert (out_dir / "intervention_window_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS)
    assert (out_dir / "geometry_transition_figure_data.csv").read_text(encoding="utf-8").strip() == ",".join(GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS)

    assert (out_dir / "regime_heatmap__empty__BarrierGap.svg").exists()
    assert (out_dir / "compare_paths_summary__empty.svg").exists()
    assert (out_dir / "intervention_window_summary__empty.svg").exists()
    assert (out_dir / "geometry_transition_summary__empty.svg").exists()
