# Reproducibility Notes

This note describes what can be reproduced from the repository, what the environment assumptions are, and how the generated artifacts are laid out on disk. Public entry points are listed in [README.md](README.md).

## Practical scope

The repository supports two realistic reproduction levels:

- `make smoke`: verifies package wiring and a synthetic-data pipeline without downloading a dataset
- `make report-flow REPORT_CONFIG=configs/pipeline/report_example.yaml`: runs the full staged workflow for the shipped example config, provided the required dataset is available under the configured `data_root`

The repository does not bundle datasets or large result artifacts. Full reruns therefore depend on local data access, available disk space, and enough compute for the selected grid sizes.

## Environment assumptions

- Python 3.10 or newer
- editable installation from the repository root

```bash
make install
```

- dependencies from `pyproject.toml` or `requirements.txt`
- local execution from the repository root

## Config-driven reproducibility

The pipeline is intended to run from config files rather than from source edits.

- Training configs declare `dataset`, `model`, optimization settings, and logging settings.
- Grid configs declare the learning-rate and batch-size search space.
- Pipeline configs declare checkpoint pairing, interpolation, barrier, geometry, and aggregation outputs.
- When the dataset or model is not built in, `dataset_builder` and `model_builder` can be set explicitly in the training config.

Each run writes `run_config.json` next to its checkpoints. Downstream stages read that stored configuration to recover dataset and model builders when needed.

## Determinism

- Seeds are set through `ntempvh.utils.seed.set_seed` for Python, NumPy, and PyTorch.
- Split seeds and training seeds are stored in the corresponding configs.
- CuDNN deterministic mode is enabled when CUDA is available.
- Exact numerical agreement across machines is not guaranteed. Hardware, drivers, CUDA, and PyTorch versions still matter.

## Output layout

The output layout is defined by the selected pipeline config.

For the shipped example workflow in `configs/pipeline/report_example.yaml`, the main roots are:

- `outputs/example/report_flow`: stage log and `report_flow_manifest.json`
- `outputs/example/trajectory_pairs.csv` and `trajectory_pairs_summary.json`
- `outputs/example/interpolation`: interpolation CSV files and metadata
- `outputs/example/barrier`: barrier summaries
- `outputs/example/path_compare`: path-comparison artifacts
- `outputs/example/geometry`: intervention-geometry summaries
- `outputs/example/results`: aggregate result tables and `results_manifest.json`
- `outputs/example/final_outputs`: report-level tables and `final_outputs_manifest.json`
- `outputs/example/figure_outputs`: SVG figures and `figure_outputs_manifest.json`

Training outputs for the same example workflow are written under:

- `outputs/runs_example/baseline`
- `outputs/runs_example/intervention`

## Reviewer workflow

Suggested local review path:

```bash
make smoke
make report-flow REPORT_CONFIG=configs/pipeline/report_example.yaml
```

Then inspect:

- `outputs/example/report_flow/report_flow_manifest.json`
- `outputs/example/results/results_manifest.json`
- `outputs/example/final_outputs/final_outputs_manifest.json`
- `outputs/example/figure_outputs/figure_outputs_manifest.json`

If you only need the diagnostic path:

```bash
make diagnostic DIAGNOSTIC_CONFIG=configs/pipeline/diagnostic_pairs_example.yaml
```

## Limits

- The repository is designed for local execution.
- The smoke tests do not exercise dataset downloads or long training grids.
- Full report-flow reproduction depends on the chosen dataset, model, and grid sizes.
