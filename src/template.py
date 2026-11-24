"""Chat / chain-of-thought templates for SFT."""

from __future__ import annotations

INSTRUCTION_HEADER = "### Instruction"
RESPONSE_HEADER = "### Response"
SCRATCH_LABEL = "Show your work:"
FINAL_LABEL = "Final answer:"


def format_prompt(question: str) -> str:
    q = question.strip()
    return f"{INSTRUCTION_HEADER}\n{q}\n\n{RESPONSE_HEADER}\n"


def format_response(scratchpad: str, answer: str) -> str:
    scratch = scratchpad.strip()
    ans = str(answer).strip()
    return f"{SCRATCH_LABEL}\n{scratch}\n{FINAL_LABEL} {ans}"


def format_example(question: str, scratchpad: str, answer: str) -> str:
    return format_prompt(question) + format_response(scratchpad, answer)


def extract_final_answer(text: str) -> str | None:
    """Parse the last 'Final answer:' span (case-insensitive)."""
    lower = text.lower()
    key = FINAL_LABEL.lower()
    idx = lower.rfind(key)
    if idx < 0:
        return None
    tail = text[idx + len(FINAL_LABEL) :].strip()
    # Stop at newline if present.
    return tail.splitlines()[0].strip() if tail else None


def has_scratchpad_and_final(text: str) -> bool:
    lower = text.lower()
    return SCRATCH_LABEL.lower() in lower and FINAL_LABEL.lower() in lower
