from __future__ import annotations

import argparse
from pathlib import Path

from ntempvh.utils.io import ensure_dir, save_json

from ._figure_baseline_compare import (
    COMPARE_PATH_FIGURE_DATA_COLUMNS,
    REGIME_HEATMAP_CELL_COLUMNS,
    build_compare_paths_figure_outputs,
    build_regime_heatmap_outputs,
)
from ._figure_intervention_geometry import (
    GEOMETRY_TRANSITION_FIGURE_DATA_COLUMNS,
    INTERVENTION_WINDOW_FIGURE_DATA_COLUMNS,
    build_geometry_transition_figure_outputs,
    build_intervention_window_figure_outputs,
)



def run_figure_outputs(
    *,
    final_outputs_root: str,
    out_dir: str,
) -> Path:
    final_outputs_root_path = Path(final_outputs_root)
    out_dir_path = ensure_dir(out_dir)

    regime_bundle = build_regime_heatmap_outputs(final_outputs_root_path, out_dir_path)
    compare_bundle = build_compare_paths_figure_outputs(final_outputs_root_path, out_dir_path)
    intervention_bundle = build_intervention_window_figure_outputs(final_outputs_root_path, out_dir_path)
    geometry_bundle = build_geometry_transition_figure_outputs(final_outputs_root_path, out_dir_path)

    manifest_path = out_dir_path / "figure_outputs_manifest.json"
    all_figures = [*regime_bundle["figures"], *compare_bundle["figures"], *intervention_bundle["figures"], *geometry_bundle["figures"]]
    manifest = {
        "inputs": {
            "final_outputs_root": str(final_outputs_root_path),
            "final_outputs_manifest_json": str(final_outputs_root_path / "final_outputs_manifest.json"),
        },
        "outputs": {
            "regime_heatmap_cells_csv": str(regime_bundle["csv"]),
            "regime_heatmaps_summary_json": str(regime_bundle["summary_json"]),
            "compare_paths_figure_data_csv": str(compare_bundle["csv"]),
            "compare_paths_figure_summary_json": str(compare_bundle["summary_json"]),
            "intervention_window_figure_data_csv": str(intervention_bundle["csv"]),
            "intervention_window_figure_summary_json": str(intervention_bundle["summary_json"]),
            "geometry_transition_figure_data_csv": str(geometry_bundle["csv"]),
            "geometry_transition_figure_summary_json": str(geometry_bundle["summary_json"]),
        },
        "figures": all_figures,
        "counts": {
            "regime_heatmap_cells": int(len(regime_bundle["rows"])),
            "compare_plot_rows": int(len(compare_bundle["rows"])),
            "intervention_plot_rows": int(len(intervention_bundle["rows"])),
            "geometry_plot_rows": int(len(geometry_bundle["rows"])),
            "num_figures": int(len(all_figures)),
        },
        "status": (
            "partial"
            if any(bundle["summary"].get("input_issues") for bundle in (regime_bundle, compare_bundle, intervention_bundle, geometry_bundle))
            else "ok"
        ),
        "limitations": [
            "figure outputs are read-only svg renderings built from existing final_outputs tables",
            "no heavy recomputation or plotting dependency is introduced, figures are deterministic svg files",
        ],
    }
    save_json(manifest_path, manifest)

    print(f"Сохранён manifest figure outputs: {manifest_path}")
    print(f"Ячеек heatmap           : {len(regime_bundle['rows'])}")
    print(f"Строк compare figure    : {len(compare_bundle['rows'])}")
    print(f"Строк intervention plot : {len(intervention_bundle['rows'])}")
    print(f"Строк geometry plot     : {len(geometry_bundle['rows'])}")
    print(f"Сгенерировано svg       : {len(all_figures)}")
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.results.figure_outputs",
        description="build reproducible figure-ready svg outputs from final_outputs artifacts",
    )
    ap.add_argument("--final_outputs_root", default="outputs/summaries/final_outputs")
    ap.add_argument("--out", default="outputs/summaries/figure_outputs")
    args = ap.parse_args()
    run_figure_outputs(final_outputs_root=args.final_outputs_root, out_dir=args.out)


if __name__ == "__main__":
    main()
