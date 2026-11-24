#!/usr/bin/env bash
# Milestone A: load HF base, one forward+generate, baseline eval report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

HF_MODEL="${HF_MODEL:-sshleifer/tiny-gpt2}"
OUT="${OUT:-runs/functional/base_smoke}"
mkdir -p "$OUT"

echo "==> [A] load HF base ${HF_MODEL}"
python3 - <<PY
from src.hf_adapter import load_hf_pair
import torch
model, tok = load_hf_pair("${HF_MODEL}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
ids = torch.tensor([tok.encode("Hello", add_bos=True, add_eos=False)], device=device)
logits, loss = model(ids, ids)
assert logits.ndim == 3
out = model.generate(ids, max_new_tokens=8, temperature=0.0, eos_id=tok.eos_id)
text = tok.decode(out[0].tolist())
print("forward_ok", float(loss.item()) if loss is not None else None)
print("generate:", repr(text[:200]))
# Save a baseline HF checkpoint via a 1-step smoke train for harness compatibility.
PY

# Persist a tiny CPT-style ckpt so eval/generate share one path.
python3 - <<PY
import copy
from pathlib import Path
import yaml
from src.train import train
cfg = yaml.safe_load(Path("configs/cpt_smoke.yaml").read_text())
# Ensure sample data exists.
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "src.download_real_data", "--sample-docs", "50"])
cfg = copy.deepcopy(cfg)
cfg["tokenizer"]["hf_name"] = "${HF_MODEL}"
cfg["model"]["hf_name"] = "${HF_MODEL}"
cfg["checkpoint"]["init_from"] = "${HF_MODEL}"
cfg["checkpoint"]["dir"] = "${OUT}"
cfg["train"]["token_budget"] = 512
cfg["train"]["eval_every"] = 1000
cfg["data"]["train_path"] = "data/real/sample/train.jsonl"
cfg["data"]["val_path"] = "data/real/sample/val.jsonl"
tmp = Path("${OUT}") / "a_run.yaml"
tmp.write_text(yaml.safe_dump(cfg))
train(str(tmp))
print("saved", "${OUT}/ckpt_latest.pt")
PY

echo "==> [A] baseline eval harness"
python3 -m src.eval \
  --ckpt "${OUT}/ckpt_latest.pt" \
  --config configs/eval_functional.yaml \
  --out "${OUT}/eval" \
  --stem baseline

echo "MILESTONE A GREEN"
