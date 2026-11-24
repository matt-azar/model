# reasoning-llm

Personal small reasoning LLM foundation.

- Phase 0: train/smoke loop
- Phase 1: frozen tokenizer + streaming data pipeline
- Phase 2: pretrain / continued-pretrain MVP
- Phase 3: SFT for reasoning format (CoT + Final answer)
- Phase 4: unified eval harness v1

## Layout

```text
configs/eval.yaml     harness benchmark list
src/eval/             unified CLI + adapters + report helpers
data/eval/            code + logic fixtures
data/personal_smoke.jsonl
scripts/phase4.sh     determinism gate (run twice, compare)
scripts/eval.sh       convenience wrapper
runs/eval/            report artifacts (gitignored)
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Phase gates

```bash
./scripts/smoke.sh
./scripts/phase1.sh
./scripts/phase2.sh
./scripts/phase3.sh
./scripts/phase4.sh
```

### Phase 4 exit checks

1. Unified CLI: `python -m src.eval --ckpt <pt> --config configs/eval.yaml --out runs/eval`
2. Adapters: gsm8k_style, personal_smoke, code_smoke, logic_mc, val_lm_loss
3. Same commit + same checkpoint → identical **canonical** report (allowlisted nondeterminism: wall time, timestamps, device label)

### Eval a checkpoint

```bash
export PYTHONPATH=.
./scripts/eval.sh runs/phase3/sft/ckpt_latest.pt runs/eval
# or:
python3 -m src.eval --ckpt runs/phase3/sft/ckpt_latest.pt --config configs/eval.yaml --out runs/eval
```

### Overnight pretrain

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/pretrain_overnight.yaml
```
