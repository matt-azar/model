#!/usr/bin/env bash
# Run Milestones A–G smoke gates (CI-scale). Overnight commands are in README.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/verify_milestone_*.sh scripts/download_real_data.sh 2>/dev/null || true

export HF_MODEL="${HF_MODEL:-sshleifer/tiny-gpt2}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

./scripts/verify_milestone_a.sh
./scripts/verify_milestone_b.sh
./scripts/verify_milestone_c.sh
./scripts/verify_milestone_d.sh
./scripts/verify_milestone_e.sh
./scripts/verify_milestone_f.sh
./scripts/verify_milestone_g.sh

echo "FUNCTIONAL PATH A–G GREEN"
