#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PIPELINE_CONFIG="${1:-configs/pipeline/report_cifar10.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m ntempvh.pipeline.report_flow \
  --config "$PIPELINE_CONFIG" \
  --python_bin "$PYTHON_BIN"

