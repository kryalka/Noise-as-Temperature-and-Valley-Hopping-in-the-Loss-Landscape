#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INTERP_ROOT="${1:-outputs/artifacts/interpolation_trajectory}"
BARRIER_CFG="${2:-configs/eval/barrier.yaml}"
OUT_ROOT="${3:-outputs/artifacts/barrier_trajectory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="outputs/logs"
FAIL_LOG="$LOG_DIR/barrier_failures.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
: > "$FAIL_LOG"

if [[ ! -d "$INTERP_ROOT" ]]; then
  echo "error: interpolation root not found: $INTERP_ROOT"
  exit 1
fi

if [[ ! -f "$BARRIER_CFG" ]]; then
  echo "error: barrier config not found: $BARRIER_CFG"
  exit 1
fi

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS=$'\t' read -r interp_csv meta_json out_json; do
  [[ -n "${interp_csv:-}" ]] || continue

  if [[ ! -f "$meta_json" ]]; then
    echo "skip: missing meta for $(basename "$interp_csv")"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  if [[ -f "$out_json" ]]; then
    echo "skip: already exists: $out_json"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "run: barrier for $(basename "$interp_csv")"

  if "$PYTHON_BIN" -m ntempvh.cli barrier \
      --interp_csv "$interp_csv" \
      --config "$BARRIER_CFG" \
      --out "$OUT_ROOT"
  then
    RUN_COUNT=$((RUN_COUNT + 1))
  else
    {
      echo "fail: barrier for $(basename "$interp_csv")"
      echo "interp_csv=$interp_csv"
      echo "meta_json=$meta_json"
      echo "expected_json=$out_json"
      echo
    } | tee -a "$FAIL_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done < <(
"$PYTHON_BIN" - "$INTERP_ROOT" "$BARRIER_CFG" "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys

from ntempvh.utils.artifacts import build_barrier_artifact_context
from ntempvh.utils.io import load_yaml

interp_root = Path(sys.argv[1])
barrier_cfg = load_yaml(sys.argv[2])
out_root = Path(sys.argv[3])

for interp_csv in sorted(interp_root.glob("interp__*.csv")):
    meta_json = interp_csv.with_suffix(".meta.json")
    artifact = build_barrier_artifact_context(interp_csv, barrier_cfg)
    out_json = out_root / f"{artifact['stem']}.json"
    print("\t".join([str(interp_csv), str(meta_json), str(out_json)]))
PY
)

echo
echo "success: $RUN_COUNT"
echo "skipped: $SKIP_COUNT"
echo "failed: $FAIL_COUNT"
echo "fail log: $FAIL_LOG"
