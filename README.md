# reasoning-llm

Personal small reasoning LLM foundation.

- Phase 0: train/smoke loop
- Phase 1: frozen tokenizer + streaming data pipeline
- Phase 2: pretrain / continued-pretrain MVP
- Phase 3: SFT for reasoning format (CoT + Final answer)

## Layout

```text
configs/          smoke / phase1 / pretrain / phase3_base / sft
src/              model, data, template, sft_data, train, eval
scripts/smoke.sh  Phase 0
scripts/phase1.sh Phase 1
scripts/phase2.sh Phase 2
scripts/phase3.sh Phase 3 (base -> SFT -> format + GSM8K-style gate)
data/sft/         curated CoT SFT + local gsm8k-style eval
tokenizer/        frozen BPE
runs/             gitignored outputs
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Phase gates

```bash
./scripts/smoke.sh
./scripts/phase1.sh          # or REBUILD=0 ./scripts/phase1.sh
./scripts/phase2.sh
./scripts/phase3.sh
```

### Phase 3 exit checks

1. Train a Phase-2-style base, then SFT with **prompt masking** and CoT template
2. SFT checkpoint emits `Show your work:` + `Final answer:` on personal smoke
3. Local GSM8K-style exact-match score **beats** the Phase 2-style base

### Overnight pretrain / continued-pretrain

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/pretrain_overnight.yaml
python3 -m src.train --config configs/pretrain_overnight.yaml \
  --resume runs/pretrain_overnight/ckpt_latest.pt
```

## Manual commands

```bash
export PYTHONPATH=.
python3 -m src.build_sft_corpus --out-dir data/sft
python3 -m src.train --config configs/phase3_base.yaml
python3 -m src.train --config configs/sft.yaml
python3 -m src.eval_gsm8k_style --ckpt runs/phase3/sft/ckpt_latest.pt
python3 -m src.verify_sft \
  --base-ckpt runs/phase3/base/ckpt_latest.pt \
  --sft-ckpt runs/phase3/sft/ckpt_latest.pt
```
