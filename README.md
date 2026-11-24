# reasoning-llm

Personal small reasoning LLM foundation (Phases 0–5).

## Phase gates

```bash
./scripts/smoke.sh
./scripts/phase1.sh
./scripts/phase2.sh
./scripts/phase3.sh
./scripts/phase4.sh
./scripts/phase5.sh
```

### Phase 5 (RL-lite)

Rejection sampling + SFT on verifier-passed winners (format + exact-match).

Exit checks:

1. Verifier pass rate (GSM8K-style exact match) **improves** vs the SFT checkpoint
2. Format rate does not collapse

```bash
./scripts/phase5.sh
# or reuse an existing SFT ckpt:
SFT_CKPT=runs/phase3/sft/ckpt_latest.pt ./scripts/verify_phase5.sh
```

### Eval harness

```bash
./scripts/eval.sh runs/phase5/rllite/ckpt_latest.pt runs/eval
```

### Overnight pretrain

```bash
export PYTHONPATH=.
python3 -m src.train --config configs/pretrain_overnight.yaml
```
