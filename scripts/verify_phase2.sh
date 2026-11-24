#!/usr/bin/env bash
# Phase 2 exit: pretrain to token budget, sane holdout curve, coherent gen, repro meta.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/pretrain.yaml}"
RUN_DIR="${RUN_DIR:-runs/pretrain}"

# Ensure Phase 1 artifacts exist.
if [[ ! -f tokenizer/tokenizer.json || ! -f data/standin/train.jsonl ]]; then
  echo "missing tokenizer/stand-in data; building Phase 1 artifacts"
  ./scripts/build_phase1_data.sh
fi

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

echo "==> [1/3] pretrain MVP (${CONFIG})"
python3 -m src.train --config "$CONFIG"

test -f "${RUN_DIR}/ckpt_latest.pt"
test -f "${RUN_DIR}/metrics.jsonl"
test -f "${RUN_DIR}/val_metrics.jsonl"
test -f "${RUN_DIR}/run_meta.json"
test -f "${RUN_DIR}/config.yaml"

echo "==> [2/3] verify holdout + coherent generation + reproducibility"
python3 -m src.verify_pretrain \
  --run-dir "$RUN_DIR" \
  --ckpt "${RUN_DIR}/ckpt_latest.pt" \
  --prompts data/personal_smoke.jsonl \
  --max-final-val-loss "${MAX_FINAL_VAL_LOSS:-3.0}" \
  --out "${RUN_DIR}/phase2_report.json"

echo "==> [3/3] show train summary"
python3 - <<PY
import json
from pathlib import Path
p = Path("${RUN_DIR}/train_summary.json")
print(p.read_text())
PY

echo "PHASE2 GREEN"
