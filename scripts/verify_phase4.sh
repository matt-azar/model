#!/usr/bin/env bash
# Phase 4 exit: same commit + same checkpoint => same canonical report.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
OUT_DIR="${OUT_DIR:-runs/eval}"
CKPT="${CKPT:-}"

# Ensure fixtures / tokenizer / SFT eval set exist.
if [[ ! -f tokenizer/tokenizer.json || ! -f data/standin/val.jsonl ]]; then
  ./scripts/build_phase1_data.sh
fi
if [[ ! -f data/sft/gsm8k_style_eval.jsonl ]]; then
  ./scripts/build_sft_data.sh
fi

# Produce a checkpoint if none provided: short base + SFT (Phase 3 path).
if [[ -z "${CKPT}" ]]; then
  if [[ ! -f runs/phase3/sft/ckpt_latest.pt ]]; then
    echo "==> training Phase 3 SFT checkpoint for harness input"
    BASE_CONFIG=configs/phase3_base.yaml \
      SFT_CONFIG=configs/sft.yaml \
      ./scripts/verify_phase3.sh
  fi
  CKPT="runs/phase3/sft/ckpt_latest.pt"
fi
test -f "$CKPT"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/run_a" "$OUT_DIR/run_b"

echo "==> [1/3] eval harness run A"
python3 -m src.eval --ckpt "$CKPT" --config "$EVAL_CONFIG" --out "$OUT_DIR/run_a" --stem report

echo "==> [2/3] eval harness run B (same commit + ckpt)"
python3 -m src.eval --ckpt "$CKPT" --config "$EVAL_CONFIG" --out "$OUT_DIR/run_b" --stem report

echo "==> [3/3] canonical compare"
python3 -m src.eval.compare "$OUT_DIR/run_a/report.json" "$OUT_DIR/run_b/report.json"

# Keep a top-level copy for convenience / Phase 5 gate input.
cp "$OUT_DIR/run_a/report.json" "$OUT_DIR/report.json"
cp "$OUT_DIR/run_a/report.md" "$OUT_DIR/report.md"
cp "$OUT_DIR/run_a/report.canonical.json" "$OUT_DIR/report.canonical.json"

python3 - <<PY
import json
from pathlib import Path
rep = json.loads(Path("${OUT_DIR}/report.json").read_text())
print("benchmarks:", sorted(rep["benchmarks"].keys()))
print("git_sha:", rep["git_sha"])
print("ckpt_sha256:", rep["ckpt_sha256"])
assert "gsm8k_style" in rep["benchmarks"]
assert "personal_smoke" in rep["benchmarks"]
assert "code_smoke" in rep["benchmarks"]
assert "logic_mc" in rep["benchmarks"]
assert "val_lm_loss" in rep["benchmarks"]
print("PHASE4 VERIFY OK")
PY

echo "PHASE4 GREEN"
