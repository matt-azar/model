"""Verify Phase 1 streaming: pack and stream a token budget without babysitting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data import make_streaming_dataloader, stream_packed_token_count
from src.manifest import load_manifest, verify_manifest_hashes
from src.tokenizer import FrozenTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream packed tokens to a budget")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="jsonl/parquet paths to stream (cycled if needed)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=1_000_000_000,
        help="stop after this many packed tokens (Phase 1 exit: >1e9)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stand-in", action="store_true")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--skip-manifest-check", action="store_true")
    parser.add_argument("--log-every-tokens", type=int, default=50_000_000)
    parser.add_argument(
        "--dataloader-smoke-batches",
        type=int,
        default=8,
        help="also pull N real DataLoader batches to prove tensor packing works",
    )
    parser.add_argument("--out", default=None, help="optional JSON report path")
    args = parser.parse_args()

    if not args.skip_manifest_check:
        manifest = load_manifest(args.manifest)
        errors = verify_manifest_hashes(manifest, ".")
        if errors:
            for err in errors:
                print(err)
            raise SystemExit("manifest hash verification failed")
        stand_in_entries = [
            d for d in manifest.get("datasets", []) if d.get("stand_in")
        ]
        if args.stand_in and not stand_in_entries:
            raise SystemExit("stand_in requested but no stand_in entries in manifest")
        print(
            f"manifest ok; stand_in_entries={len(stand_in_entries)} "
            f"total_entries={len(manifest.get('datasets', []))}"
        )

    tokenizer = FrozenTokenizer.load(args.tokenizer)

    # Prove DataLoader packing emits real tensors before the long count.
    smoke_loader = make_streaming_dataloader(
        paths=args.data,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        cycle=True,
        max_tokens=args.seq_len * args.batch_size * max(args.dataloader_smoke_batches, 1),
        stand_in=args.stand_in,
    )
    smoke_tokens = 0
    smoke_batches = 0
    for batch in smoke_loader:
        assert batch["input_ids"].ndim == 2
        assert batch["labels"].shape == batch["input_ids"].shape
        smoke_tokens += int(batch["input_ids"].numel())
        smoke_batches += 1
        if smoke_batches >= args.dataloader_smoke_batches:
            break
    print(f"dataloader smoke ok: batches={smoke_batches} tokens={smoke_tokens}")

    t0 = time.time()
    count = stream_packed_token_count(
        paths=args.data,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        target_tokens=args.target_tokens,
        seed=args.seed,
        stand_in=args.stand_in,
        log_every_tokens=args.log_every_tokens,
    )
    count["dataloader_smoke_batches"] = smoke_batches
    count["dataloader_smoke_tokens"] = smoke_tokens
    count["target_tokens"] = args.target_tokens
    count["data"] = args.data
    count["tokenizer_vocab_size"] = tokenizer.vocab_size
    count["passed"] = count["tokens"] >= args.target_tokens
    count["wall_seconds"] = time.time() - t0

    print(json.dumps(count, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(count, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    if count["tokens"] < args.target_tokens:
        raise SystemExit(
            f"stream verify failed: got {count['tokens']} tokens "
            f"< target {args.target_tokens}"
        )
    print("PHASE1 STREAM OK")


if __name__ == "__main__":
    main()
