#!/usr/bin/env bash
# Full Phase 1 gate: rebuild data/tokenizer if needed, then verify exit criteria.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${REBUILD:-1}" == "1" ]]; then
  ./scripts/build_phase1_data.sh
fi
./scripts/verify_phase1.sh
