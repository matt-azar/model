#!/usr/bin/env bash
# Milestone E: unified eval with official GSM8K adapter; canonical determinism.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${CKPT:-runs/functional/sft/ckpt_latest.pt}"
OUT="${OUT:-runs/functional/eval}"

python3 -m src.download_real_data --sample-docs 50 --gsm8k-train 80 --gsm8k-test 20
if [[ ! -f "$CKPT" ]]; then
  ./scripts/verify_milestone_d.sh
  CKPT="runs/functional/sft/ckpt_latest.pt"
fi

rm -rf "$OUT"
mkdir -p "$OUT/a" "$OUT/b"

python3 -m src.eval --ckpt "$CKPT" --config configs/eval_functional.yaml --out "$OUT/a" --stem report
python3 -m src.eval --ckpt "$CKPT" --config configs/eval_functional.yaml --out "$OUT/b" --stem report
python3 -m src.eval.compare "$OUT/a/report.json" "$OUT/b/report.json"

python3 - <<PY
import json
from pathlib import Path
rep = json.loads(Path("${OUT}/a/report.json").read_text())
assert "gsm8k" in rep["benchmarks"]
assert "personal_smoke" in rep["benchmarks"]
print("benchmarks", sorted(rep["benchmarks"]))
print("MILESTONE E GREEN")
PY
