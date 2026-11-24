#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
EXTRA=()
if [[ "${FULL:-0}" == "1" ]]; then
  EXTRA+=(--full)
fi
python3 -m src.download_real_data --sample-docs "${SAMPLE_DOCS:-500}" "${EXTRA[@]}"
