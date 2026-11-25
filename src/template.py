"""Chat / chain-of-thought templates for SFT."""

from __future__ import annotations

import re

INSTRUCTION_HEADER = "### Instruction"
RESPONSE_HEADER = "### Response"
SCRATCH_LABEL = "Show your work:"
FINAL_LABEL = "Final answer:"
REASONING_INSTRUCTION = (
    f"Begin with '{SCRATCH_LABEL}' and explain the steps. "
    f"Put your result on the last line after '{FINAL_LABEL}'. "
    "If the result is numeric, write only the number after that label."
)
_FINAL_LINE = re.compile(rf"{re.escape(FINAL_LABEL)}[ \t]*(\S.*)", re.IGNORECASE)
_FIELD_PREFIXES = tuple(
    label.casefold() for label in (SCRATCH_LABEL, FINAL_LABEL, INSTRUCTION_HEADER, RESPONSE_HEADER)
)


def format_prompt(question: str) -> str:
    return f"{INSTRUCTION_HEADER}\n{question.strip()}\n\n{RESPONSE_HEADER}\n"


def format_chat_prompt(question: str, tokenizer: object) -> str:
    """Serialize a user question with the tokenizer's chat template when available."""
    hf = getattr(tokenizer, "tok", None)
    if not getattr(hf, "chat_template", None):
        return format_prompt(question)
    return hf.apply_chat_template(
        [{"role": "user", "content": question.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_reasoning_prompt(question: str, tokenizer: object) -> str:
    """Add the worked-solution instructions used in math training and evaluation."""
    if getattr(tokenizer, "tok", None) is None:
        return format_prompt(question)
    instruction = f"{question.strip()}\n\n{REASONING_INSTRUCTION}"
    return format_chat_prompt(instruction, tokenizer)


def format_response(scratchpad: str, answer: str) -> str:
    scratch = scratchpad.strip()
    ans = str(answer).strip()
    return f"{SCRATCH_LABEL}\n{scratch}\n{FINAL_LABEL} {ans}"


def format_example(question: str, scratchpad: str, answer: str) -> str:
    return format_prompt(question) + format_response(scratchpad, answer)


def extract_final_answer(text: str) -> str | None:
    """Read a labeled answer from the final nonblank line of a completion."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = _FINAL_LINE.fullmatch(lines[-1].strip())
    return match[1].strip() if match else None


def parse_response(text: str) -> tuple[str, str] | None:
    """Return the scratchpad and terminal answer from a complete template."""
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0].strip().casefold() != SCRATCH_LABEL.casefold():
        return None
    scratch = "\n".join(lines[1:-1]).strip()
    answer = extract_final_answer(text)
    if not scratch or answer is None:
        return None
    for line in scratch.splitlines():
        if line.lstrip().casefold().startswith(_FIELD_PREFIXES):
            return None
    return scratch, answer


def has_scratchpad_and_final(text: str) -> bool:
    return parse_response(text) is not None
