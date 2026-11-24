# reasoning-llm

Personal small reasoning LLM foundation. Phase 0: train/smoke loop. Phase 1: frozen tokenizer + streaming data pipeline.

## Layout

```text
configs/          smoke / phase1 / pretrain / sft YAML
src/              model, data, tokenizer, train, generate, eval, builders
scripts/smoke.sh  Phase 0 gate
scripts/phase1.sh Phase 1 gate (build + verify >1B token stream)
tokenizer/        frozen BPE (tokenizer.json)
data/toy/         Phase 0 overfit corpus
data/standin/     clearly marked toy stand-in mix (train/val jsonl+parquet)
data/manifest.json
runs/             gitignored outputs
```

## Setup

Python 3.11+, CPU is enough for Phase 0/1 gates.

```bash
python3 -m pip install -r requirements.txt
```

## Phase 0 smoke

```bash
chmod +x scripts/smoke.sh
./scripts/smoke.sh
```

## Phase 1 (tokenizer + data pipeline)

```bash
chmod +x scripts/phase1.sh scripts/build_phase1_data.sh scripts/verify_phase1.sh
./scripts/phase1.sh
```

What Phase 1 checks:

1. Builds a clearly marked CC0 stand-in corpus with train/val split (jsonl + parquet)
2. Trains and freezes a BPE tokenizer under `tokenizer/`
3. Writes sha256 hashes into `data/manifest.json` and verifies them
4. Streams **>1B packed tokens** from the license-documented stand-in mix without babysitting

To verify an already-built tree without rebuilding:

```bash
REBUILD=0 ./scripts/phase1.sh
# or: ./scripts/verify_phase1.sh
```

## Manual commands

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/smoke.yaml
python3 -m src.generate --ckpt runs/smoke/ckpt_latest.pt --prompt "The capital of France is"
python3 -m src.eval_smoke --ckpt runs/smoke/ckpt_latest.pt --eval data/personal_smoke.jsonl
python3 -m src.stream_verify --tokenizer tokenizer --data data/standin/train.jsonl --stand-in --target-tokens 1000000000
```
