"""Milestone G: local chat CLI over the best functional checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.generate import generate_text
from src.template import format_prompt


DEFAULT_BEST = "runs/functional/best/ckpt_latest.pt"


def resolve_best_ckpt(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        DEFAULT_BEST,
        "runs/functional/rllite/ckpt_latest.pt",
        "runs/functional/sft/ckpt_latest.pt",
        "runs/functional/cpt_smoke/ckpt_latest.pt",
        "runs/phase5/rllite/ckpt_latest.pt",
        "runs/phase3/sft/ckpt_latest.pt",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise SystemExit(
        "no checkpoint found; pass --ckpt or train via ./scripts/functional_path.sh"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CoT chat CLI")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prompt",
        default=None,
        help="single-shot prompt (non-interactive)",
    )
    args = parser.parse_args()
    ckpt = resolve_best_ckpt(args.ckpt)
    print(f"loading {ckpt}")

    def ask(question: str) -> str:
        prompt = format_prompt(question)
        return generate_text(
            ckpt,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device_name=args.device,
        )

    if args.prompt:
        print(ask(args.prompt))
        return

    print("Chat ready. Empty line or Ctrl-D to exit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except EOFError:
            print()
            break
        if not question:
            break
        print("\nModel>")
        print(ask(question))


if __name__ == "__main__":
    main()
