#!/usr/bin/env bash
# Phase 1 exit: frozen tokenizer present, manifest hashes ok, stream >1B tokens.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_DIR="${RUN_DIR:-runs/phase1}"
mkdir -p "$RUN_DIR"

test -f tokenizer/tokenizer.json
test -f tokenizer/tokenizer_config.json
test -f data/standin/train.jsonl
test -f data/standin/val.jsonl
test -f data/standin/STANDIN.md

echo "==> [1/3] verify manifest hashes"
python3 -m src.refresh_manifest --manifest data/manifest.json --root . --verify-only

echo "==> [2/3] round-trip jsonl + parquet text counts"
python3 - <<'PY'
from src.data import load_texts
jsonl_n = len(load_texts("data/standin/train.jsonl"))
pq_n = len(load_texts("data/standin/train.parquet"))
assert jsonl_n == pq_n and jsonl_n > 0, (jsonl_n, pq_n)
print(f"jsonl/parquet train docs match: {jsonl_n}")
val_n = len(load_texts("data/standin/val.jsonl"))
assert val_n > 0
print(f"val docs: {val_n}")
PY

echo "==> [3/3] stream ${TARGET_TOKENS} packed tokens from stand-in mix"
python3 -m src.stream_verify \
  --tokenizer tokenizer \
  --data data/standin/train.jsonl data/standin/val.jsonl \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --target-tokens "$TARGET_TOKENS" \
  --stand-in \
  --manifest data/manifest.json \
  --log-every-tokens "$(( TARGET_TOKENS / 5 ))" \
  --out "${RUN_DIR}/stream_report.json"

echo "PHASE1 GREEN"
