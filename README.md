Репозиторий для диагностики траекторий обучения и анализа ландшафта потерь

Можно:
- запускать сетки обучений
- строить interpolation и barrier артефакты между checkpoint-ами
- сравнивать `chord-path` и `observed-path`
- запускать локальные температурные интервенции
- считать proxy geometry
- собирать итоговые таблицы, summary и figure-ready outputs
- запускать отдельный diagnostic pipeline по парам checkpoint-ов

## Что здесь есть

В репозитории есть несколько слоёв:

- `ntempvh/train`  
  обучение одной модели, чекпоинты, метрики, интервенции

- `ntempvh/eval`  
  interpolation, barrier, geometry, compare-paths

- `ntempvh/pipeline`  
  orchestration поверх compute-слоя, в том числе `report_flow` и `diagnostic_pipeline`

- `ntempvh/results`  
  results aggregation, final outputs, figure outputs, diagnostic summaries

- `configs/`  
  готовые пресеты под отчёт и примеры кастомных сценариев

- `experiments/`  
  shell wrappers для типовых запусков

- `tests/`  
  smoke, integration и regression tests

## Что уже доступно для использования

Есть рабочие сценарии для:

- datasets  
  `cifar10`, `cifar100`, `svhn`

- models  
  `resnet18`, `resnet34`, `resnet50`, `resnet100`

## Как запускать

Запускать проект из корня репозитория

### Проверка тестов

```bash
python3.11 -m pytest -q tests
```

### Полноценные запуски

Если нужен готовый сценарий для полноценной работы, самый прямой путь такой:

```bash
bash experiments/run_report_flow.sh configs/pipeline/report_cifar10.yaml
```

или так:

```bash
python3.11 -m ntempvh.pipeline.report_flow --config configs/pipeline/report_cifar10.yaml
```

есть и другие готовые способы:
- `configs/pipeline/report_cifar100.yaml`
- `configs/pipeline/report_cifar10_resnet34.yaml`
- `configs/pipeline/report_cifar100_resnet34.yaml`

Если нужен именно диагностический анализ по парам checkpoint-ов, запуск такой:

```bash
bash experiments/run_diagnostic_pipeline.sh configs/pipeline/diagnostic_pairs_example.yaml
```

или напрямую модулем:

```bash
python3.11 -m ntempvh.pipeline.diagnostic_pipeline --config configs/pipeline/diagnostic_pairs_example.yaml
```

Если хочется запускать стадии по одной, есть центральный CLI:

```bash
python3.11 -m ntempvh.cli train --config configs/train/sgd_base.yaml --seed 1 --out outputs/runs
python3.11 -m ntempvh.cli interpolate --ckptA path/to/a.pt --ckptB path/to/b.pt --config configs/eval/interpolation.yaml --out outputs/artifacts/interp
python3.11 -m ntempvh.cli barrier --interp_csv outputs/artifacts/interp/example.csv --config configs/eval/barrier.yaml --out outputs/artifacts/barrier
python3.11 -m ntempvh.cli compare-paths --ckptA path/to/a.pt --ckptB path/to/b.pt --config configs/eval/path_compare.yaml --out outputs/artifacts/path_compare
python3.11 -m ntempvh.cli geometry --ckpt path/to/a.pt --config configs/eval/geometry.yaml --out outputs/artifacts/geometry
```

## Какие артефакты получаются

### После вычислительных стадий

Появляются такие каталоги:

- `outputs/runs_*`  
  train run директории, `summary.json`, `metrics.jsonl`, `checkpoints/`

- `outputs/artifacts/interpolation*`  
  interpolation csv и meta json

- `outputs/artifacts/barrier*`  
  barrier json

- `outputs/artifacts/path_compare*`  
  compare-path json и summary csv

- `outputs/artifacts/geometry*`  
  geometry json

### После слоев сборки результатов

results слой сводит compute артефакты в таблицы:

- `compare_paths_results.csv`
- `intervention_runs_results.csv`
- `intervention_geometry_runs_results.csv`
- `path_quality_links.csv`
- `results_manifest.json`

### После финальных слоев

- `baseline_regime_table.csv`
- `compare_paths_final_summary.csv`
- `intervention_window_summary.csv`
- `geometry_transition_summary.csv`
- `baseline_regime_maps.json`
- `final_outputs_manifest.json`

### После слоев для построения графиков/карт

- svg heatmaps по режимам
- svg summaries для compare-paths
- svg summaries для interventions
- svg summaries для geometry transitions
- `figure_outputs_manifest.json`

## Диагностический пайплайн

Пайплайн задуман как инструмент для анализа траекторий чекпоинтов

на входе у него может быть:
- готовый `pairs_csv` с колонками `ckptA` и `ckptB`
- или `runs_root`, если пары нужно собрать из run директорий автоматически

на выходе он пишет:
- pair-level таблицу с `Peakobs`, `Pitchord`, `Pitobs`, `BarrierGap`, `devL1`
- `curvature_proxy_A`, `curvature_proxy_B`, `curvature_proxy_mean`
- regime summary table
- regime maps json
- markdown и json report
- machine-readable manifest

## Советы

Если хочется быстро понять проект руками, самые полезные конфиги такие:

- `configs/train/sgd_base.yaml`
- `configs/train/lr_bs_grid.yaml`
- `configs/train/windowed_intervention.yaml`
- `configs/eval/interpolation.yaml`
- `configs/eval/barrier.yaml`
- `configs/eval/path_compare.yaml`
- `configs/eval/geometry.yaml`
- `configs/pipeline/report_cifar10.yaml`
- `configs/pipeline/diagnostic_pairs_example.yaml`

если нужен кастомный запуск, полезно смотреть и example-конфиги:

- `configs/train/lr_bs_grid_custom_example.yaml`
- `configs/train/intervention_lr_bs_grid_custom_example.yaml`
- `configs/eval/path_compare_custom_example.yaml`
- `configs/pipeline/report_custom_example.yaml`

## Что важно для повторного использования

- checkpoint файлы должны быть доступны по путям из `pairs_csv` или `runs_root`
- compare config и geometry config должны подходить к этим checkpoint-ам
- для train-generated runs репозиторий уже умеет извлекать метаданные автоматически
- для внешних траекторий минимальный контракт это валидные пути `ckptA` и `ckptB`

