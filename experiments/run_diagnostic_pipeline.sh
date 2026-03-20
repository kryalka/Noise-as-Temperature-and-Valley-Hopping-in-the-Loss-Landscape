#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CFG="${1:-configs/pipeline/diagnostic_pairs_example.yaml}"
OUT_ROOT="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "$OUT_ROOT" ]]; then
  "$PYTHON_BIN" -m ntempvh.pipeline.diagnostic_pipeline --config "$CFG" --out "$OUT_ROOT"
else
  "$PYTHON_BIN" -m ntempvh.pipeline.diagnostic_pipeline --config "$CFG"
fi
