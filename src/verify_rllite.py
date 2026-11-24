"""Phase 5 exit: verifier pass rate improves vs SFT; format does not collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_gsm8k_style import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 5 RL-lite exit criteria")
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--rllite-ckpt", required=True)
    parser.add_argument("--eval", default="data/sft/gsm8k_style_eval.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-format-drop",
        type=float,
        default=0.05,
        help="allowed absolute drop in format_rate vs SFT",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sft = evaluate(args.sft_ckpt, args.eval, max_new_tokens=args.max_new_tokens)
    rl = evaluate(args.rllite_ckpt, args.eval, max_new_tokens=args.max_new_tokens)

    improved = rl["exact_match"] > sft["exact_match"]
    format_ok = rl["format_rate"] + 1e-9 >= (sft["format_rate"] - args.max_format_drop)

    report = {
        "sft": {
            "ckpt": args.sft_ckpt,
            "exact_match": sft["exact_match"],
            "format_rate": sft["format_rate"],
            "n_correct": sft["n_correct"],
            "n": sft["n"],
        },
        "rllite": {
            "ckpt": args.rllite_ckpt,
            "exact_match": rl["exact_match"],
            "format_rate": rl["format_rate"],
            "n_correct": rl["n_correct"],
            "n": rl["n"],
        },
        "verifier_improved": improved,
        "format_ok": format_ok,
        "max_format_drop": args.max_format_drop,
        "passed": improved and format_ok,
        "method": "rejection_sampling_plus_sft_on_winners",
    }
    print(json.dumps(report, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    if not improved:
        raise SystemExit(
            f"verifier pass rate did not improve: "
            f"sft={sft['exact_match']:.4f} rllite={rl['exact_match']:.4f}"
        )
    if not format_ok:
        raise SystemExit(
            f"format collapsed: sft={sft['format_rate']:.4f} "
            f"rllite={rl['format_rate']:.4f}"
        )
    print("PHASE5 VERIFY OK")


if __name__ == "__main__":
    main()
