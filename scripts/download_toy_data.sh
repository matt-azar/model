#!/usr/bin/env bash
# Toy data is checked into data/toy/ for offline Phase 0 smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "toy data already present at ${ROOT}/data/toy/documents.jsonl"
ls -la "${ROOT}/data/toy/documents.jsonl"
