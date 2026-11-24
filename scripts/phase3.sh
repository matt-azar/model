#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/build_sft_data.sh scripts/verify_phase3.sh
./scripts/verify_phase3.sh
