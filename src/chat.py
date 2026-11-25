"""Local reasoning CLI over a selected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate import Generator
from src.template import format_chat_prompt, format_reasoning_prompt


DEFAULT_BEST = "runs/functional/best/ckpt_latest.pt"
DEFAULT_RUN_SUMMARY = Path("runs/reasoning/summary.json")


def resolve_best_ckpt(explicit: str | None) -> str:
    if explicit:
        return explicit
    if DEFAULT_RUN_SUMMARY.is_file():
        selected = json.loads(DEFAULT_RUN_SUMMARY.read_text(encoding="utf-8")).get("selected")
        if isinstance(selected, str) and Path(selected).exists():
            return selected
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
        "no checkpoint found; pass --ckpt or train via python -m src.reasoning_train"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local chat CLI")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--worked-solution",
        action="store_true",
        help="request the format used by the GSM8K training and evaluation pipeline",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="single-shot prompt (non-interactive)",
    )
    args = parser.parse_args()
    ckpt = resolve_best_ckpt(args.ckpt)
    print(f"loading {ckpt}")
    generator = Generator.from_checkpoint(ckpt, args.device)
    format_question = format_reasoning_prompt if args.worked_solution else format_chat_prompt

    def ask(question: str) -> str:
        prompt = format_question(question, generator.tokenizer)
        return generator.complete(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
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
