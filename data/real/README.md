# Real data (gitignored contents)

Gate fixtures and overnight downloads land here via:

```bash
./scripts/download_real_data.sh
FULL=1 ./scripts/download_real_data.sh
```

- `sample/` — small CC0 mix for Milestone B/C gates (`stand_in: false`)
- `train.jsonl` / `val.jsonl` — overnight CPT defaults
- `sft/` — GSM8K train/val
- `eval/gsm8k_test.jsonl` — Milestone E eval
