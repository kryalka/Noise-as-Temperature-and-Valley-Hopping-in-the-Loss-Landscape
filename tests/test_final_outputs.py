from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.results.final_outputs import (
    BASELINE_REGIME_TABLE_COLUMNS,
    COMPARE_SECTION_SUMMARY_COLUMNS,
    GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
    INTERVENTION_WINDOW_SUMMARY_COLUMNS,
    run_final_outputs,
)

from ntempvh.results.pipeline import (
    COMPARE_RESULTS_COLUMNS,
    INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
    INTERVENTION_RUN_RESULTS_COLUMNS,
    PATH_QUALITY_LINK_COLUMNS,
)


def _run_name(*, dataset: str = "cifar10", seed: int = 1, lr: float = 0.2, bs: int = 8) -> str:
    return (
        f"{dataset}_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0_mom0_schnone__deadbeef"
    )



def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def test_run_final_outputs_empty_inputs_write_schema_stable_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results_pipeline"
    results_root.mkdir()
    out_dir = tmp_path / "final_outputs"

    manifest_path = run_final_outputs(results_root=str(results_root), out_dir=str(out_dir))

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["baseline_regime_rows"] == 0
    assert manifest["counts"]["compare_summary_rows"] == 0
    assert manifest["counts"]["intervention_window_rows"] == 0
    assert manifest["counts"]["geometry_transition_rows"] == 0
    assert manifest["status"] == "partial"

    assert (out_dir / "baseline_regime_table.csv").read_text(encoding="utf-8").strip() == ",".join(BASELINE_REGIME_TABLE_COLUMNS)
    assert (out_dir / "compare_paths_final_summary.csv").read_text(encoding="utf-8").strip() == ",".join(COMPARE_SECTION_SUMMARY_COLUMNS)
    assert (out_dir / "intervention_window_summary.csv").read_text(encoding="utf-8").strip() == ",".join(INTERVENTION_WINDOW_SUMMARY_COLUMNS)
    assert (out_dir / "geometry_transition_summary.csv").read_text(encoding="utf-8").strip() == ",".join(GEOMETRY_TRANSITION_SUMMARY_COLUMNS)
