#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_ROOT="${1:-outputs/summaries/results_pipeline}"
OUT_ROOT="${2:-outputs/summaries/final_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m ntempvh.results.final_outputs \
  --results_root "$RESULTS_ROOT" \
  --out "$OUT_ROOT"

