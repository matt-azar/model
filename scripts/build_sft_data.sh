#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

python3 -m src.build_sft_corpus --out-dir data/sft --eval-n 40

# Extend manifest with SFT artifacts and refresh hashes.
python3 - <<'PY'
import json
from pathlib import Path
from src.manifest import load_manifest, refresh_hashes, save_manifest

path = Path("data/manifest.json")
manifest = load_manifest(path)
existing = {d["name"] for d in manifest.get("datasets", [])}
additions = [
    {
        "name": "sft-train",
        "path": "data/sft/train.jsonl",
        "stage": "sft",
        "license": "CC0-1.0 (curated/synthetic arithmetic CoT authored for this repo)",
        "url": None,
        "stand_in": False,
        "notes": "Phase 3 curated SFT train split with scratchpad + Final answer.",
        "sha256": None,
    },
    {
        "name": "sft-val",
        "path": "data/sft/val.jsonl",
        "stage": "sft",
        "license": "CC0-1.0 (curated/synthetic arithmetic CoT authored for this repo)",
        "url": None,
        "stand_in": False,
        "notes": "Phase 3 SFT holdout split.",
        "sha256": None,
    },
    {
        "name": "gsm8k-style-eval",
        "path": "data/sft/gsm8k_style_eval.jsonl",
        "stage": "eval",
        "license": "CC0-1.0 (curated/synthetic; not the real GSM8K dataset)",
        "url": None,
        "stand_in": False,
        "notes": "Local GSM8K-style exact-match eval for Phase 3 exit criteria.",
        "sha256": None,
    },
]
for row in additions:
    if row["name"] not in existing:
        manifest.setdefault("datasets", []).append(row)
save_manifest(refresh_hashes(manifest, "."), path)
print(f"updated {path}")
PY

echo "SFT DATA BUILD OK"
