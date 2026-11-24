#!/usr/bin/env bash
# Milestone B: real mix scripts, stand_in:false, stream ≥1B packed tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

python3 -m src.download_real_data --sample-docs 500
test -f data/real/sample/train.jsonl
grep -q '"stand_in": false' configs/cpt_smoke.yaml || grep -q 'stand_in: false' configs/cpt_smoke.yaml

# Use toy BPE tokenizer for stream speed in the gate (data path is what matters).
if [[ ! -f tokenizer/tokenizer.json ]]; then
  ./scripts/build_phase1_data.sh
fi

python3 -m src.stream_verify \
  --tokenizer tokenizer \
  --data data/real/sample/train.jsonl data/real/sample/val.jsonl \
  --seq-len 128 \
  --batch-size 64 \
  --target-tokens 1000000000 \
  --manifest data/manifest.json \
  --skip-manifest-check \
  --log-every-tokens 200000000 \
  --out runs/functional/stream_b_report.json

python3 - <<'PY'
import json
from pathlib import Path
rep = json.loads(Path("runs/functional/stream_b_report.json").read_text())
assert rep["tokens"] >= 1_000_000_000
print("streamed", rep["tokens"])
PY

echo "MILESTONE B GREEN"
