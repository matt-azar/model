#!/usr/bin/env bash
# Build stand-in corpus, frozen tokenizer, and hashed manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

N_DOCS="${N_DOCS:-2000}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"

echo "==> build stand-in corpus (n_docs=${N_DOCS})"
python3 -m src.build_standin_corpus --out-dir data/standin --n-docs "$N_DOCS" --also-parquet

echo "==> train + freeze BPE tokenizer"
python3 -m src.build_tokenizer \
  --inputs data/standin/train.jsonl data/toy/documents.jsonl \
  --out tokenizer \
  --vocab-size "$VOCAB_SIZE"

echo "==> refresh manifest hashes"
python3 -m src.refresh_manifest --manifest data/manifest.json --root .

echo "PHASE1 DATA BUILD OK"
