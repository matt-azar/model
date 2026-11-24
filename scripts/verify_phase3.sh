#!/usr/bin/env bash
# Phase 3 exit: SFT emits scratchpad+final answer; GSM8K-style beats Phase 2 base.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

BASE_CONFIG="${BASE_CONFIG:-configs/phase3_base.yaml}"
SFT_CONFIG="${SFT_CONFIG:-configs/sft.yaml}"
BASE_DIR="${BASE_DIR:-runs/phase3/base}"
SFT_DIR="${SFT_DIR:-runs/phase3/sft}"

if [[ ! -f tokenizer/tokenizer.json ]]; then
  ./scripts/build_phase1_data.sh
fi
./scripts/build_sft_data.sh

rm -rf runs/phase3
mkdir -p "$BASE_DIR" "$SFT_DIR"

echo "==> [1/4] train Phase-2-style base"
python3 -m src.train --config "$BASE_CONFIG"
test -f "${BASE_DIR}/ckpt_latest.pt"

echo "==> [2/4] SFT from base with prompt masking"
python3 -m src.train --config "$SFT_CONFIG"
test -f "${SFT_DIR}/ckpt_latest.pt"

echo "==> [3/4] GSM8K-style scores (base vs SFT)"
python3 -m src.eval_gsm8k_style \
  --ckpt "${BASE_DIR}/ckpt_latest.pt" \
  --eval data/sft/gsm8k_style_eval.jsonl \
  --out "${BASE_DIR}/gsm8k_style_report.json"
python3 -m src.eval_gsm8k_style \
  --ckpt "${SFT_DIR}/ckpt_latest.pt" \
  --eval data/sft/gsm8k_style_eval.jsonl \
  --out "${SFT_DIR}/gsm8k_style_report.json"

echo "==> [4/4] Phase 3 exit verify"
python3 -m src.verify_sft \
  --base-ckpt "${BASE_DIR}/ckpt_latest.pt" \
  --sft-ckpt "${SFT_DIR}/ckpt_latest.pt" \
  --eval data/sft/gsm8k_style_eval.jsonl \
  --smoke data/personal_smoke.jsonl \
  --out "${SFT_DIR}/phase3_report.json"

echo "PHASE3 GREEN"
