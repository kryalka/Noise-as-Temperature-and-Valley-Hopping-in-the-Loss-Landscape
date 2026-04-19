# Noise as Temperature and Valley Hopping in the Loss Landscape

This repository contains a configurable research pipeline for training runs, checkpoint-trajectory analysis, path-based loss diagnostics, local geometry estimation, and intervention studies. It is organized as a reusable package with YAML-driven workflows, plus example presets for small-image benchmarks and a small set of optional example-specific helpers.

## Main capabilities

- train single runs or learning-rate and batch-size grids from YAML configs
- build checkpoint pairs and evaluate interpolation, barrier, and path-comparison artifacts
- estimate local geometry around checkpoints before and after interventions
- aggregate stage outputs into stable CSV, JSON, Markdown, and SVG artifacts
- support custom datasets and models through explicit config fields and builder paths

## Repository structure

- `ntempvh/`: reusable pipeline code
- `configs/`: runnable example configs and editable templates
- `examples/`: optional dataset-specific examples outside the core workflow
- `experiments/`: optional shell wrappers and local helper scripts
- `tests/`: smoke and regression tests

## Installation

From the repository root:

```bash
make install
```

This installs the package in editable mode with the development dependencies used by the local test suite.

## Quickstart

Minimal local verification:

```bash
make smoke
```

This checks the CLI surface and runs a synthetic-data smoke pipeline. It does not download a dataset.

### Common entry points

- `python -m ntempvh train --config configs/train/train_example.yaml --seed 1 --out outputs/runs/manual_example`
- `python -m ntempvh.pipeline.report_flow --config configs/pipeline/report_example.yaml`
- `python -m ntempvh.pipeline.diagnostic_pipeline --config configs/pipeline/diagnostic_pairs_example.yaml`

## Reproducible workflows

Canonical full example:

```bash
make report-flow REPORT_CONFIG=configs/pipeline/report_example.yaml
```

The shipped `report_example.yaml` preset is a runnable example built around the built-in `cifar10` and `resnet18` components. The same workflow can target a different dataset or model by editing the config family under `configs/train/` and, when needed, setting `dataset_builder` or `model_builder`.

The example workflow writes its main manifests to:

- `outputs/example/report_flow/report_flow_manifest.json`
- `outputs/example/results/results_manifest.json`
- `outputs/example/final_outputs/final_outputs_manifest.json`
- `outputs/example/figure_outputs/figure_outputs_manifest.json`

## Main configs and modifying experiments

Use these files as starting points:

- `configs/pipeline/report_example.yaml`: runnable end-to-end example
- `configs/train/train_example.yaml`: single-run training example
- `configs/train/lr_bs_grid_example.yaml`: baseline grid example
- `configs/train/intervention_lr_bs_grid_example.yaml`: intervention grid example
- `configs/train/train_template.yaml`: editable training template for a new dataset or model
- `configs/pipeline/report_template.yaml`: editable end-to-end workflow template

Built-in datasets and models are selected through `dataset` and `model`. For components that are not built in, set:

- `dataset_builder: package.module:build_dataset_spec`
- `model_builder: package.module:build_model`

Those builder paths are stored in `run_config.json` and reused by downstream evaluation stages, so new datasets and models do not require source edits once the config is in place.

## Output artifacts and where results appear

Training runs write:

- `run_config.json`
- `summary.json`
- `metrics.jsonl`
- `checkpoints/*.pt`

Analysis stages write under their configured output roots:

- interpolation CSV files with matching `.meta.json`
- barrier JSON summaries and `barriers.csv`
- path-comparison JSON summaries plus aggregate CSV tables
- geometry JSON summaries and `geometries.csv`
- report-flow, results, final-output, and figure-output manifests

The exact roots are defined in the pipeline config. The shipped example uses `outputs/example/`.

## Testing

Full test suite:

```bash
make test
```

Fast local check:

```bash
make smoke
```

The local CI workflow in `.github/workflows/ci.yml` runs the same test suite.

## Reproducibility notes

- Training, evaluation, and pipeline parameters should be declared explicitly in YAML configs.
- Checkpoints are accompanied by `run_config.json`, which records dataset, model, and optional builder paths.
- Evaluation stages resolve dataset and model builders from that stored run configuration when present.
- Seeds are set for Python, NumPy, and PyTorch, but exact numeric agreement can still depend on hardware and library versions.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment assumptions, reviewer scope, and output layout.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
