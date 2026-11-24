#!/usr/bin/env bash
# Convenience wrapper around the unified eval CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${1:?usage: ./scripts/eval.sh <ckpt> [out_dir]}"
OUT="${2:-runs/eval}"
CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"

python3 -m src.eval --ckpt "$CKPT" --config "$CONFIG" --out "$OUT" --stem report
echo "report: ${OUT}/report.md"
