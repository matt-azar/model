#!/usr/bin/env bash
# Full Phase 2 gate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/verify_phase2.sh
./scripts/verify_phase2.sh
