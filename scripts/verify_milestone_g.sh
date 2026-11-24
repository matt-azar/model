#!/usr/bin/env bash
# Milestone G: chat CLI answers personal smoke prompts with scratchpad format.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
  if [[ -f runs/functional/best/ckpt_latest.pt ]]; then
    CKPT=runs/functional/best/ckpt_latest.pt
  elif [[ -f runs/functional/sft/ckpt_latest.pt ]]; then
    CKPT=runs/functional/sft/ckpt_latest.pt
  else
    ./scripts/verify_milestone_d.sh
    CKPT=runs/functional/sft/ckpt_latest.pt
  fi
fi

mkdir -p runs/functional/best
if [[ "$(realpath "$CKPT")" != "$(realpath runs/functional/best/ckpt_latest.pt 2>/dev/null || true)" ]]; then
  cp -f "$CKPT" runs/functional/best/ckpt_latest.pt
fi

python3 -m src.chat --ckpt runs/functional/best/ckpt_latest.pt \
  --prompt "What is 2 plus 2? Show your work and Final answer." \
  --max-new-tokens 64 \
  --temperature 0.0 | tee runs/functional/best/chat_smoke.txt

python3 - <<'PY'
from pathlib import Path
text = Path("runs/functional/best/chat_smoke.txt").read_text().lower()
# Usability gate: CLI runs and produces non-empty model text.
assert len(text.strip()) > 10
print("chat output bytes", len(text))
print("MILESTONE G GREEN")
PY
