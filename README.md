# reasoning-llm

Personal small reasoning LLM foundation.

- Phase 0: train/smoke loop
- Phase 1: frozen tokenizer + streaming data pipeline
- Phase 2: pretrain / continued-pretrain MVP

## Layout

```text
configs/          smoke / phase1 / pretrain / pretrain_overnight / sft
src/              model, data, tokenizer, train, generate, eval, builders
scripts/smoke.sh  Phase 0 gate
scripts/phase1.sh Phase 1 gate (>1B token stream)
scripts/phase2.sh Phase 2 gate (pretrain + holdout + coherent gen)
tokenizer/        frozen BPE
data/standin/     clearly marked toy stand-in mix
data/manifest.json
runs/             gitignored outputs
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Phase 0 smoke

```bash
./scripts/smoke.sh
```

## Phase 1 (tokenizer + data pipeline)

```bash
./scripts/phase1.sh
# or: REBUILD=0 ./scripts/phase1.sh
```

## Phase 2 (pretrain MVP)

From-scratch pretrain on the license-documented stand-in mix with the frozen tokenizer, periodic holdout loss, and reproducibility metadata (git SHA, manifest/tokenizer hashes).

```bash
./scripts/phase2.sh
```

Exit checks:

1. Train to `train.token_budget` with logged loss / grad_norm / tok/s
2. Holdout `val_loss` improves across eval points
3. Checkpoint generates coherent text on `data/personal_smoke.jsonl`
4. Run dir includes `config.yaml` + `run_meta.json` matching the checkpoint

### Overnight / continued-pretrain

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/pretrain_overnight.yaml
# resume / continued-pretrain from a checkpoint:
python3 -m src.train --config configs/pretrain_overnight.yaml \
  --resume runs/pretrain_overnight/ckpt_latest.pt
```

## Manual commands

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/smoke.yaml
python3 -m src.train --config configs/pretrain.yaml
python3 -m src.generate --ckpt runs/pretrain/ckpt_latest.pt --prompt "The capital of France is"
python3 -m src.verify_pretrain --run-dir runs/pretrain
```
