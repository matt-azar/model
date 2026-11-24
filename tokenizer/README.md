# Frozen tokenizer

Phase 1 ships a frozen byte-level BPE (`tokenizer.json` + `tokenizer_config.json`).

- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Rebuild only via `python3 -m src.build_tokenizer` / `./scripts/build_phase1_data.sh`
- After rebuild, refresh `data/manifest.json` hashes

The rest of the stack should load this directory rather than training a new vocab mid-experiment.
