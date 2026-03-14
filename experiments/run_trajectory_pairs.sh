#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNS_ROOT="${1:-outputs/runs_lr_bs_grid}"
OUT_CSV="${2:-outputs/summaries/trajectory_pairs.csv}"
OUT_JSON="${3:-outputs/summaries/trajectory_pairs_summary.json}"
PAIR_MODE="${4:-milestones}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -ge 4 ]]; then
  shift 4
fi

CMD=(
  "$PYTHON_BIN" -m ntempvh.pipeline.trajectory_pairs
  --runs_root "$RUNS_ROOT"
  --out_csv "$OUT_CSV"
  --out_json "$OUT_JSON"
  --pair_mode "$PAIR_MODE"
)

if [[ $# -gt 0 ]]; then
  CMD+=(--milestone_epochs "$@")
fi

"${CMD[@]}"

