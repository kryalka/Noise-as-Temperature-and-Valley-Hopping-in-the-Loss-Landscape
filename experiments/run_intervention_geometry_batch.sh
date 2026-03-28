#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNS_ROOT="${1:-outputs/runs_lr_bs_grid}"
GEOMETRY_CFG="${2:-configs/eval/geometry.yaml}"
OUT_ROOT="${3:-outputs/artifacts/geometry_intervention}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "error: runs root not found: $RUNS_ROOT"
  exit 1
fi

if [[ ! -f "$GEOMETRY_CFG" ]]; then
  echo "error: geometry config not found: $GEOMETRY_CFG"
  exit 1
fi

"$PYTHON_BIN" -m ntempvh.eval.intervention_geometry \
  --runs_root "$RUNS_ROOT" \
  --config "$GEOMETRY_CFG" \
  --out "$OUT_ROOT"
