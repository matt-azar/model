#!/usr/bin/env bash
# Phase 5 exit: rejection-sampling RL-lite improves verifier pass rate vs SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

SFT_CKPT="${SFT_CKPT:-runs/phase3/sft/ckpt_latest.pt}"
RLLITE_DIR="${RLLITE_DIR:-runs/phase5/rllite}"
WINNERS="${WINNERS:-data/rllite/winners.jsonl}"
EVAL_PATH="${EVAL_PATH:-data/sft/gsm8k_style_eval.jsonl}"

if [[ ! -f tokenizer/tokenizer.json ]]; then
  ./scripts/build_phase1_data.sh
fi
if [[ ! -f data/sft/train.jsonl || ! -f "$EVAL_PATH" ]]; then
  ./scripts/build_sft_data.sh
fi

# Need an SFT policy checkpoint.
if [[ ! -f "$SFT_CKPT" ]]; then
  echo "==> missing SFT ckpt; running Phase 3 gate to produce one"
  ./scripts/verify_phase3.sh
  SFT_CKPT="runs/phase3/sft/ckpt_latest.pt"
fi
test -f "$SFT_CKPT"

mkdir -p data/rllite "$RLLITE_DIR"

echo "==> [1/4] rejection-sample verifier winners from SFT policy"
python3 -m src.reject_sample \
  --ckpt "$SFT_CKPT" \
  --pool data/sft/train.jsonl \
  --out "$WINNERS" \
  --k "${RS_K:-2}" \
  --temperature "${RS_TEMP:-0.8}" \
  --max-prompts "${RS_MAX_PROMPTS:-160}" \
  --seed 42

echo "==> [2/4] SFT on winners (RL-lite)"
# Point init_from at the live SFT ckpt via a temp config.
python3 - <<PY
import copy
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("configs/rllite.yaml").read_text())
cfg = copy.deepcopy(cfg)
cfg["checkpoint"]["init_from"] = "${SFT_CKPT}"
cfg["checkpoint"]["dir"] = "${RLLITE_DIR}"
cfg["data"]["train_path"] = "${WINNERS}"
tmp = Path("${RLLITE_DIR}") / "rllite_run.yaml"
tmp.parent.mkdir(parents=True, exist_ok=True)
tmp.write_text(yaml.safe_dump(cfg))
print(tmp)
PY
python3 -m src.train --config "${RLLITE_DIR}/rllite_run.yaml"
test -f "${RLLITE_DIR}/ckpt_latest.pt"

echo "==> [3/4] score SFT vs RL-lite with verifier (GSM8K-style)"
python3 -m src.eval_gsm8k_style --ckpt "$SFT_CKPT" --eval "$EVAL_PATH" --out "${RLLITE_DIR}/sft_eval.json"
python3 -m src.eval_gsm8k_style --ckpt "${RLLITE_DIR}/ckpt_latest.pt" --eval "$EVAL_PATH" --out "${RLLITE_DIR}/rllite_eval.json"

echo "==> [4/4] Phase 5 exit verify"
python3 -m src.verify_rllite \
  --sft-ckpt "$SFT_CKPT" \
  --rllite-ckpt "${RLLITE_DIR}/ckpt_latest.pt" \
  --eval "$EVAL_PATH" \
  --out "${RLLITE_DIR}/phase5_report.json"

# Optional: also emit unified harness report for Phase 5 artifact continuity.
if [[ -f configs/eval.yaml ]]; then
  python3 -m src.eval \
    --ckpt "${RLLITE_DIR}/ckpt_latest.pt" \
    --config configs/eval.yaml \
    --out "${RLLITE_DIR}/eval" \
    --stem report || true
fi

echo "PHASE5 GREEN"
