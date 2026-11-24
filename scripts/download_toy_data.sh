#!/usr/bin/env bash
# Offline Phase 0/1 data is checked in or built locally.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "toy data: ${ROOT}/data/toy/documents.jsonl"
ls -la data/toy/documents.jsonl

if [[ ! -f data/standin/train.jsonl ]]; then
  echo "stand-in corpus missing; building via scripts/build_phase1_data.sh"
  ./scripts/build_phase1_data.sh
else
  echo "stand-in corpus present under data/standin/"
  ls -la data/standin/train.jsonl data/standin/val.jsonl
fi
