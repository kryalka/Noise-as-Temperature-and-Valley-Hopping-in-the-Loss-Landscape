#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNS_ROOT="${1:-outputs/runs_lr_bs_grid}"
PATH_COMPARE_ROOT="${2:-outputs/artifacts/path_compare}"
INTERVENTION_GEOMETRY_SUMMARY="${3:-outputs/artifacts/geometry_intervention/intervention_geometry_summary.csv}"
OUT_ROOT="${4:-outputs/summaries/results_pipeline}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m ntempvh.results.pipeline \
  --runs_root "$RUNS_ROOT" \
  --path_compare_root "$PATH_COMPARE_ROOT" \
  --intervention_geometry_summary "$INTERVENTION_GEOMETRY_SUMMARY" \
  --out "$OUT_ROOT"

