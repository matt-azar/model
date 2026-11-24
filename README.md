# reasoning-llm

Personal small reasoning LLM lab (Phases 0–5) plus the **functional path** (Milestones A–G): continued-pretrain an open base (SmolLM-135M), real data, GSM8K SFT/eval, RL-lite, local chat.

## Linked task

Functional reasoning LLM path — Milestones A–G.

## Quick gates (CI / laptop smoke)

Uses `sshleifer/tiny-gpt2` by default so scripts finish without multi-GB downloads:

```bash
python3 -m pip install -r requirements.txt
./scripts/functional_path.sh
```

Individual milestones:

```bash
./scripts/verify_milestone_a.sh   # HF base load + baseline eval
./scripts/verify_milestone_b.sh   # real mix + stream ≥1B tokens
./scripts/verify_milestone_c.sh   # CPT smoke
./scripts/verify_milestone_d.sh   # GSM8K SFT beats CPT
./scripts/verify_milestone_e.sh   # official GSM8K eval harness
./scripts/verify_milestone_f.sh   # scaled RL-lite
./scripts/verify_milestone_g.sh   # chat CLI
```

Legacy Phase 0–5 gates remain:

```bash
./scripts/smoke.sh && ./scripts/phase1.sh && ./scripts/phase2.sh
./scripts/phase3.sh && ./scripts/phase4.sh && ./scripts/phase5.sh
```

## Overnight / real GPU (SmolLM-135M)

```bash
export PYTHONPATH=.
export HF_MODEL=HuggingFaceTB/SmolLM-135M

# B: sample fixtures always; FULL=1 attempts small FineWeb-Edu/OpenWebMath slices
FULL=1 ./scripts/download_real_data.sh

# C: continued-pretrain (1B+ tokens; raise token_budget toward 5B when stable)
python3 -m src.train --config configs/cpt_overnight.yaml
# resume:
python3 -m src.train --config configs/cpt_overnight.yaml \
  --resume runs/functional/cpt/ckpt_latest.pt

# D: GSM8K SFT from CPT weights
python3 -m src.train --config configs/sft_gsm8k_overnight.yaml

# E: eval report (promotion gate)
./scripts/eval.sh runs/functional/sft/ckpt_latest.pt runs/functional/eval
# or: python3 -m src.eval --ckpt ... --config configs/eval_functional.yaml

# F: scaled rejection-sampling RL-lite
SFT_CKPT=runs/functional/sft/ckpt_latest.pt \
  HF_MODEL=$HF_MODEL RS_K=8 RS_MAX_PROMPTS=500 RL_STEPS=500 \
  ./scripts/verify_milestone_f.sh

# G: chat
python3 -m src.chat --ckpt runs/functional/best/ckpt_latest.pt
```

Best-checkpoint convention: `runs/functional/best/ckpt_latest.pt` (updated by Milestone F gate).

## Layout (functional)

```text
src/hf_adapter.py          SmolLM/Qwen HF wrappers
src/download_real_data.py  data/real sample + GSM8K fixtures
src/chat.py                local CoT chat CLI
configs/base_smollm.yaml
configs/cpt_overnight.yaml
configs/sft_gsm8k_overnight.yaml
configs/eval_functional.yaml
data/real/                 gitignored real mix (sample fixtures for gates)
```
