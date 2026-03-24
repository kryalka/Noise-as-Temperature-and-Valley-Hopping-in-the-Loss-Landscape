from __future__ import annotations

import argparse
from pathlib import Path

from ntempvh.utils.io import ensure_dir, save_json

from ._pipeline_compare import aggregate_compare_paths
from ._pipeline_intervention import (
    aggregate_intervention_geometry,
    aggregate_intervention_runs,
)
from ._pipeline_links import aggregate_path_quality_links
from ._pipeline_schema import (
    COMPARE_RESULTS_COLUMNS,
    INTERVENTION_GEOMETRY_RUN_RESULTS_COLUMNS,
    INTERVENTION_RUN_RESULTS_COLUMNS,
    PATH_QUALITY_LINK_COLUMNS,
)



def run_results_pipeline(
    *,
    runs_root: str,
    path_compare_root: str,
    intervention_geometry_summary_csv: str,
    out_dir: str,
) -> Path:
    out_dir_path = ensure_dir(out_dir)

    compare_bundle = aggregate_compare_paths(path_compare_root, out_dir_path)
    intervention_bundle = aggregate_intervention_runs(runs_root, out_dir_path)
    geometry_bundle = aggregate_intervention_geometry(intervention_geometry_summary_csv, out_dir_path)
    links_bundle = aggregate_path_quality_links(
        compare_bundle["rows"],
        intervention_bundle["rows"],
        geometry_bundle["run_rows"],
        out_dir_path,
    )

    manifest_path = out_dir_path / "results_manifest.json"
    manifest = {
        "inputs": {
            "runs_root": str(runs_root),
            "path_compare_root": str(path_compare_root),
            "intervention_geometry_summary_csv": str(intervention_geometry_summary_csv),
        },
        "outputs": {
            "compare_paths_results_csv": str(compare_bundle["csv"]),
            "compare_paths_results_summary_json": str(compare_bundle["summary_json"]),
            "intervention_runs_results_csv": str(intervention_bundle["csv"]),
            "intervention_runs_results_summary_json": str(intervention_bundle["summary_json"]),
            "intervention_geometry_roles_results_csv": str(geometry_bundle["role_csv"]),
            "intervention_geometry_roles_results_summary_json": str(geometry_bundle["role_summary_json"]),
            "intervention_geometry_runs_results_csv": str(geometry_bundle["run_csv"]),
            "intervention_geometry_runs_results_summary_json": str(geometry_bundle["run_summary_json"]),
            "path_quality_links_csv": str(links_bundle["csv"]),
            "path_quality_links_summary_json": str(links_bundle["summary_json"]),
        },
        "counts": {
            "compare_rows": int(len(compare_bundle["rows"])),
            "intervention_run_rows": int(len(intervention_bundle["rows"])),
            "intervention_geometry_role_rows": int(len(geometry_bundle["role_rows"])),
            "intervention_geometry_run_rows": int(len(geometry_bundle["run_rows"])),
            "path_quality_link_rows": int(len(links_bundle["rows"])),
        },
        "limitations": [
            "results aggregation is read-only and built on top of existing upstream artifacts",
            "when upstream artifacts are missing or partial, stable csv and json outputs are still written with empty rows and machine-readable issue summaries",
        ],
    }
    save_json(manifest_path, manifest)

    print(f"Сохранён manifest results: {manifest_path}")
    print(f"Строк compare           : {len(compare_bundle['rows'])}")
    print(f"Строк intervention run  : {len(intervention_bundle['rows'])}")
    print(f"Строк geometry roles    : {len(geometry_bundle['role_rows'])}")
    print(f"Строк geometry runs     : {len(geometry_bundle['run_rows'])}")
    print(f"Строк path-quality link : {len(links_bundle['rows'])}")
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ntempvh.results.pipeline")
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--path_compare_root", required=True)
    ap.add_argument("--intervention_geometry_summary_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_results_pipeline(
        runs_root=args.runs_root,
        path_compare_root=args.path_compare_root,
        intervention_geometry_summary_csv=args.intervention_geometry_summary_csv,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
