# Configuration Guide

This directory contains runnable example configs, dataset-specific reference presets, and editable templates. The public entry points are described in [README.md](../README.md).

## Layout

- `train/`: single-run configs, grid configs, and intervention-grid configs
- `eval/`: interpolation, barrier, path-comparison, and geometry settings
- `pipeline/`: staged report-flow and diagnostic-pipeline configs

## Config families

| Group | Purpose | Typical outputs |
| --- | --- | --- |
| `train/*.yaml` | single runs, baseline grids, intervention grids | run directories with checkpoints and summaries |
| `eval/*.yaml` | per-stage analysis settings | interpolation, barrier, path-comparison, and geometry artifacts |
| `pipeline/*.yaml` | end-to-end orchestration | report-flow manifests, aggregate tables, final outputs, figures |

## Recommended starting points

Runnable example configs:

- `train/train_example.yaml`
- `train/lr_bs_grid_example.yaml`
- `train/intervention_lr_bs_grid_example.yaml`
- `eval/path_compare_example.yaml`
- `pipeline/report_example.yaml`

Editable templates:

- `train/train_template.yaml`
- `train/train_grid_template.yaml`
- `train/intervention_train_grid_template.yaml`
- `pipeline/report_template.yaml`

Older dataset- or model-specific presets such as `sgd_svhn.yaml`, `report_cifar100.yaml`, or `report_cifar10_resnet34.yaml` are kept as reference examples. They are not required by the core package.

## Dataset and model selection

The training surface is driven by explicit config keys:

- `dataset`
- `model`
- `training.*`
- `data.*`
- `intervention.*`

For built-in components, set `dataset` and `model` to one of the registered names.

For custom components, set the optional builder fields:

- `dataset_builder: package.module:build_dataset_spec`
- `model_builder: package.module:build_model`

Those builder paths are stored in `run_config.json` and reused by evaluation stages, so interpolation, geometry, and path comparison can resolve the same dataset or model without source edits.

## Safe edits

These changes are usually low risk:

- output roots such as `out_root`, `interpolation_out`, or `geometry_out`
- seeds, learning-rate grids, and batch-size grids
- training epoch counts and scheduler choices
- interpolation point counts and observed-path selection
- barrier thresholds
- geometry sampling settings
- enabled or disabled pipeline stages

## Changes that need care

- changing `dataset` or `model` without updating the corresponding builder fields for a non-built-in component
- running evaluation configs against checkpoints produced with incompatible data or model settings
- changing output roots in one stage without updating downstream references in the pipeline config

## Editing pattern

1. Start from `*_example.yaml` if you want a runnable example close to the shipped workflow.
2. Start from `*_template.yaml` if you are adapting the pipeline to a new dataset or model family.
3. Copy the chosen config to a new file under the same subtree.
4. Pass that new path explicitly to the relevant command.

Example:

```bash
python -m ntempvh.pipeline.report_flow --config configs/pipeline/my_report.yaml
```
