# reasoning-llm (Phase 0)

Personal small reasoning LLM foundation. Phase 0 proves a debugger-friendly train loop: toy overfit, checkpoint resume, generate, and a tiny personal smoke eval.

## Layout

```text
configs/          smoke / pretrain / sft YAML
src/              model, data, train, generate, eval_smoke
scripts/smoke.sh  end-to-end Phase 0 gate
data/toy/         tiny overfit corpus
data/personal_smoke.jsonl
runs/             gitignored outputs
```

## Setup

Python 3.11+, CPU is enough for smoke.

```bash
python3 -m pip install -r requirements.txt
# or: python3 -m pip install -e .
```

## Run smoke

```bash
chmod +x scripts/smoke.sh
./scripts/smoke.sh
```

What it checks:

1. Trains `configs/smoke.yaml` until loss is under the overfit threshold
2. Resumes from a mid checkpoint and continues a few steps
3. Generates from `ckpt_latest.pt`
4. Runs `data/personal_smoke.jsonl` and requires at least one expected substring match

## Manual commands

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/smoke.yaml
python3 -m src.generate --ckpt runs/smoke/ckpt_latest.pt --prompt "The capital of France is"
python3 -m src.eval_smoke --ckpt runs/smoke/ckpt_latest.pt --eval data/personal_smoke.jsonl
```

## Phase 0 exit criteria

- Toy overfit works (final loss below `train.overfit_loss_threshold`)
- Checkpoint resume works
- `./scripts/smoke.sh` exits 0 and prints `SMOKE GREEN`
