#!/usr/bin/env bash
set -euo pipefail

GRID_CONFIG="${1:-configs/train/lr_bs_grid.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TMP_CFG_DIR=".tmp/lr_bs_grid_configs"
mkdir -p "$TMP_CFG_DIR"

if [[ ! -f "$GRID_CONFIG" ]]; then
  echo "error: grid config not found: $GRID_CONFIG"
  exit 1
fi

echo "project root: $PROJECT_ROOT"
echo "grid config: $GRID_CONFIG"
echo "temp cfg dir: $TMP_CFG_DIR"

RUN_COUNT=0
SKIP_DONE_COUNT=0
SKIP_PARTIAL_COUNT=0
PLANNED_COUNT=0


while IFS=$'\t' read -r seed lr bs cfg_path out_root run_dir; do
  PLANNED_COUNT=$((PLANNED_COUNT + 1))

  final_ckpt="${run_dir}/checkpoints/final.pt"

  if [[ -f "$final_ckpt" ]]; then
    echo "skip ready: seed=${seed} lr=${lr} bs=${bs} -> ${run_dir}"
    SKIP_DONE_COUNT=$((SKIP_DONE_COUNT + 1))
    continue
  fi

  if [[ -d "$run_dir" ]]; then
    echo "skip partial: seed=${seed} lr=${lr} bs=${bs} -> ${run_dir}"
    echo "partial run dir already exists, remove it manually if you want to rerun"
    SKIP_PARTIAL_COUNT=$((SKIP_PARTIAL_COUNT + 1))
    continue
  fi

  echo "run: seed=${seed} lr=${lr} bs=${bs}"
  "$PYTHON_BIN" -m ntempvh.cli train --config "$cfg_path" --seed "$seed" --out "$out_root"
  RUN_COUNT=$((RUN_COUNT + 1))

done < <(
"$PYTHON_BIN" -m ntempvh.pipeline.train_grid \
  --grid_config "$GRID_CONFIG" \
  --tmp_cfg_dir "$TMP_CFG_DIR" \
  --project_root "$PROJECT_ROOT"
)

echo
echo "planned: $PLANNED_COUNT"
echo "launched: $RUN_COUNT"
echo "skipped ready: $SKIP_DONE_COUNT"
echo "skipped partial: $SKIP_PARTIAL_COUNT"
