# RL-lite artifacts (Phase 5)

Method: **rejection sampling + SFT on verifier-passed winners**.

- `winners.jsonl` — gold + self-generated trajectories that pass format + exact-match checks
- `winners_pairs.jsonl` — chosen/rejected pairs retained for inspection (DPO-ready later)

Not full RLHF. No learned reward model.
