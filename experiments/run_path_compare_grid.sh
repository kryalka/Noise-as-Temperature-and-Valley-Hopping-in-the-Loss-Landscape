#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PAIRS_CSV="${1:-outputs/summaries/trajectory_pairs.csv}"
COMPARE_CFG="${2:-configs/eval/path_compare.yaml}"
OUT_ROOT="${3:-outputs/artifacts/path_compare}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="outputs/logs"
FAIL_LOG="$LOG_DIR/path_compare_failures.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
: > "$FAIL_LOG"

if [[ ! -f "$PAIRS_CSV" ]]; then
  echo "error: trajectory pairs file not found: $PAIRS_CSV"
  exit 1
fi

if [[ ! -f "$COMPARE_CFG" ]]; then
  echo "error: path-compare config not found: $COMPARE_CFG"
  exit 1
fi

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS=$'\t' read -r run_name epoch_a epoch_b ckpt_a ckpt_b out_json; do
  [[ -n "${run_name:-}" ]] || continue

  if [[ -f "$out_json" ]]; then
    echo "skip: already exists: $run_name epoch_${epoch_a}->epoch_${epoch_b}"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "run: chord-vs-observed compare for $run_name epoch_${epoch_a}->epoch_${epoch_b}"

  if "$PYTHON_BIN" -m ntempvh.cli compare-paths \
      --ckptA "$ckpt_a" \
      --ckptB "$ckpt_b" \
      --config "$COMPARE_CFG" \
      --out "$OUT_ROOT"
  then
    RUN_COUNT=$((RUN_COUNT + 1))
  else
    {
      echo "fail: path compare for $run_name epoch_${epoch_a}->epoch_${epoch_b}"
      echo "ckptA=$ckpt_a"
      echo "ckptB=$ckpt_b"
      echo "out=$out_json"
      echo
    } | tee -a "$FAIL_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done < <(
"$PYTHON_BIN" - "$PAIRS_CSV" "$COMPARE_CFG" "$OUT_ROOT" <<'PY'
import csv
from pathlib import Path
import sys

from ntempvh.utils.artifacts import build_path_compare_artifact_context
from ntempvh.utils.io import load_yaml

pairs_csv = Path(sys.argv[1])
compare_cfg = load_yaml(sys.argv[2])
out_root = Path(sys.argv[3])

with open(pairs_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_name = row["run_name"]
        epoch_a = int(row["epoch_A"])
        epoch_b = int(row["epoch_B"])
        ckpt_a = row["ckptA"]
        ckpt_b = row["ckptB"]

        artifact = build_path_compare_artifact_context(ckpt_a, ckpt_b, compare_cfg)
        out_json = out_root / "comparisons" / f"{artifact['stem']}.json"

        print(
            "\t".join([
                run_name,
                str(epoch_a),
                str(epoch_b),
                ckpt_a,
                ckpt_b,
                str(out_json),
            ])
        )
PY
)

echo
echo "success: $RUN_COUNT"
echo "skipped: $SKIP_COUNT"
echo "failed: $FAIL_COUNT"
echo "fail log: $FAIL_LOG"
