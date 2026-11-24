#!/usr/bin/env bash
# Phase 0 exit: toy overfit, checkpoint resume, generate + eval smoke green.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/smoke.yaml}"
RUN_DIR="${RUN_DIR:-runs/smoke}"
RESUME_STEPS="${RESUME_STEPS:-20}"

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

echo "==> [1/4] train smoke (toy overfit)"
python3 -m src.train --config "$CONFIG"

CKPT_LATEST="${RUN_DIR}/ckpt_latest.pt"
test -f "$CKPT_LATEST"

echo "==> [2/4] resume from mid checkpoint for ${RESUME_STEPS} more steps"
# Pick the first saved step checkpoint if present; else latest.
MID_CKPT="$(ls -1 "${RUN_DIR}"/ckpt_step_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${MID_CKPT}" ]]; then
  MID_CKPT="$CKPT_LATEST"
fi

# Resume into a separate run dir so we prove optimizer/model reload works.
RESUME_DIR="${RUN_DIR}/resume_check"
rm -rf "$RESUME_DIR"
mkdir -p "$RESUME_DIR"

python3 - <<PY
import copy
from pathlib import Path
import yaml
from src.train import train, load_checkpoint

cfg_path = Path("${CONFIG}")
cfg = yaml.safe_load(cfg_path.read_text())
cfg = copy.deepcopy(cfg)
cfg["checkpoint"]["dir"] = "${RESUME_DIR}"
cfg["checkpoint"]["resume"] = None
cfg["train"]["steps"] = int(load_checkpoint(Path("${MID_CKPT}"), map_location="cpu")["step"]) + int("${RESUME_STEPS}")
cfg["train"]["save_every"] = max(1, int("${RESUME_STEPS}"))
cfg["train"]["log_every"] = 1
cfg["train"]["overfit_loss_threshold"] = None  # resume segment may be short
tmp = Path("${RESUME_DIR}") / "resume_smoke.yaml"
tmp.write_text(yaml.safe_dump(cfg))
train(str(tmp), resume_override="${MID_CKPT}")
print("resume check ok")
PY

echo "==> [3/4] generate from latest checkpoint"
PROMPT="$(python3 - <<'PY'
import yaml
print(yaml.safe_load(open("configs/smoke.yaml"))["generate"]["prompt"])
PY
)"
python3 -m src.generate --ckpt "$CKPT_LATEST" --prompt "$PROMPT" --max-new-tokens 32 --temperature 0

echo "==> [4/4] personal smoke eval"
python3 -m src.eval_smoke \
  --ckpt "$CKPT_LATEST" \
  --eval data/personal_smoke.jsonl \
  --max-new-tokens 48 \
  --out "${RUN_DIR}/eval_smoke_report.json"

echo "SMOKE GREEN"
