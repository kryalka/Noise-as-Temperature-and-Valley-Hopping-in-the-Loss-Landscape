#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FINAL_OUTPUTS_ROOT="${1:-outputs/summaries/final_outputs}"
OUT_ROOT="${2:-outputs/summaries/figure_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m ntempvh.results.figure_outputs \
  --final_outputs_root "$FINAL_OUTPUTS_ROOT" \
  --out "$OUT_ROOT"

