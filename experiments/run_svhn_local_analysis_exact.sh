#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-svhn}"
DATA_ROOT="${DATA_ROOT:-data}"
ANALYSIS_SUBDIR="${ANALYSIS_SUBDIR:-chord_vs_observed_exact_local}"
AGGREGATE_DIRNAME="${AGGREGATE_DIRNAME:-aggregate_exact_local}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"$PYTHON_BIN" experiments/fix_svhn_local_analysis.py \
  --runs-root "$RUNS_ROOT" \
  --data-root "$DATA_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --num-points 21 \
  --bn-recalib-batches 20 \
  --eval-batch-size 256 \
  --bn-batch-size 256 \
  --val-size 5000 \
  --analysis-subdir "$ANALYSIS_SUBDIR" \
  --aggregate-dirname "$AGGREGATE_DIRNAME" \
  "$@"
