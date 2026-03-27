from __future__ import annotations

import argparse
from pathlib import Path

from ntempvh.utils.io import ensure_dir, save_json

from ._final_baseline_compare import (
    BASELINE_REGIME_TABLE_COLUMNS,
    COMPARE_SECTION_SUMMARY_COLUMNS,
    build_baseline_regime_outputs,
    build_compare_paths_section_outputs,
)
from ._final_intervention_geometry import (
    GEOMETRY_TRANSITION_SUMMARY_COLUMNS,
    INTERVENTION_WINDOW_SUMMARY_COLUMNS,
    build_geometry_transition_outputs,
    build_intervention_window_outputs,
)
from .pipeline import (
    COMPARE_RESULTS_COLUMNS,
    INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
    INTERVENTION_RUN_RESULTS_COLUMNS,
    PATH_QUALITY_LINK_COLUMNS,
)



def run_final_outputs(
    *,
    results_root: str,
    out_dir: str,
) -> Path:
    results_root_path = Path(results_root)
    out_dir_path = ensure_dir(out_dir)

    baseline_bundle = build_baseline_regime_outputs(results_root_path, out_dir_path)
    compare_bundle = build_compare_paths_section_outputs(results_root_path, out_dir_path)
    intervention_bundle = build_intervention_window_outputs(results_root_path, out_dir_path)
    geometry_bundle = build_geometry_transition_outputs(results_root_path, out_dir_path)

    manifest_path = out_dir_path / "final_outputs_manifest.json"
    manifest = {
        "inputs": {
            "results_root": str(results_root_path),
            "results_manifest_json": str(results_root_path / "results_manifest.json"),
        },
        "outputs": {
            "baseline_regime_table_csv": str(baseline_bundle["csv"]),
            "baseline_regime_maps_json": str(baseline_bundle["summary_json"]),
            "compare_paths_final_summary_csv": str(compare_bundle["csv"]),
            "compare_paths_final_summary_json": str(compare_bundle["summary_json"]),
            "intervention_window_summary_csv": str(intervention_bundle["csv"]),
            "intervention_window_summary_json": str(intervention_bundle["summary_json"]),
            "geometry_transition_summary_csv": str(geometry_bundle["csv"]),
            "geometry_transition_summary_json": str(geometry_bundle["summary_json"]),
        },
        "counts": {
            "baseline_regime_rows": int(len(baseline_bundle["rows"])),
            "compare_summary_rows": int(len(compare_bundle["rows"])),
            "intervention_window_rows": int(len(intervention_bundle["rows"])),
            "geometry_transition_rows": int(len(geometry_bundle["rows"])),
        },
        "status": (
            "partial"
            if any(bundle["summary"].get("input_issues") for bundle in (baseline_bundle, compare_bundle, intervention_bundle, geometry_bundle))
            else "ok"
        ),
        "limitations": [
            "final outputs are read-only summaries built strictly from results_pipeline outputs",
            "no new scientific metrics are introduced here, only report-ready tables and maps are assembled",
        ],
    }
    save_json(manifest_path, manifest)

    print(f"Сохранён manifest final outputs: {manifest_path}")
    print(f"Строк baseline regime      : {len(baseline_bundle['rows'])}")
    print(f"Строк compare summary      : {len(compare_bundle['rows'])}")
    print(f"Строк intervention summary : {len(intervention_bundle['rows'])}")
    print(f"Строк geometry summary     : {len(geometry_bundle['rows'])}")
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.results.final_outputs",
        description="build final report-ready csv and json outputs from results_pipeline artifacts",
    )
    ap.add_argument("--results_root", default="outputs/summaries/results_pipeline")
    ap.add_argument("--out", default="outputs/summaries/final_outputs")
    args = ap.parse_args()
    run_final_outputs(results_root=args.results_root, out_dir=args.out)


if __name__ == "__main__":
    main()
