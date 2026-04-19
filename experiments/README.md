# Supplementary Wrappers

The core package no longer depends on the files in this directory. The canonical public entry points are the Python modules documented in [README.md](../README.md).

This directory is kept for two narrower purposes:

- optional shell wrappers around package-level commands
- local helper scripts that are useful for inspection or ad hoc validation

## When to use this directory

Use these files only if a shell wrapper is more convenient than calling the Python module directly, or if you need a local helper that is not part of the main artifact workflow.

## Main wrappers

- `run_report_flow.sh`
- `run_diagnostic_pipeline.sh`
- `run_lr_bs_grid.sh`
- `run_trajectory_pairs.sh`
- `run_interpolation_grid.sh`
- `run_barrier_grid.sh`
- `run_path_compare_grid.sh`
- `run_intervention_geometry_batch.sh`
- `run_results_pipeline.sh`
- `run_final_outputs.sh`
- `run_figure_outputs.sh`

Each wrapper delegates to a package module with repository-relative defaults. They are optional convenience layers, not required parts of the pipeline.

## Notes

- Run wrappers from the repository root.
- Override the interpreter with `PYTHON_BIN` when needed.
- Dataset-specific example material has been moved under [`examples/`](../examples/).
