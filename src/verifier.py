"""Rule + exact-match verifier for RL-lite (Phase 5)."""

from __future__ import annotations

import re

from src.template import extract_final_answer, parse_response


_DECIMAL = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
)
_ANSWER_PREFIX = re.compile(
    r"(?:(?P<bold>\*\*)?(?:the )?(?:final )?answer(?: is)?[: ]+(?(bold)\*\*)|####\s*)",
    re.IGNORECASE,
)
_NUMERIC_WRAPPERS = (
    re.compile(r"\*\*(.+)\*\*\.?"),
    re.compile(r"\$(.+)\$\.?"),
    re.compile(r"\\boxed\{(.+)\}\.?"),
)


def _numeric_key(value: str) -> str | None:
    text = value.strip().removesuffix(".").replace("\N{MINUS SIGN}", "-")
    if _DECIMAL.fullmatch(text) is None:
        return None
    whole, _, decimal = text.lstrip("+-").replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    decimal = decimal.rstrip("0")
    sign = "-" if text.startswith("-") and (whole != "0" or decimal) else ""
    return sign + whole + (f".{decimal}" if decimal else "")


def normalize_answer(value: str) -> str:
    """Compare decimal values exactly, preserving nonnumeric text for legacy data."""
    return _numeric_key(value) or value.strip().casefold()


def _unwrap_number(text: str) -> str | None:
    text = text.removeprefix(r"\[").strip()
    for wrapper in _NUMERIC_WRAPPERS:
        match = wrapper.fullmatch(text)
        if match:
            text = match[1].strip()
    return text if _numeric_key(text) is not None else None


def extract_eval_answer(completion: str) -> str | None:
    """Read a terminal result, including conventional math answer markup."""
    text = completion.strip().removesuffix(r"\]").rstrip()
    answer = extract_final_answer(text)
    if answer is not None:
        return _unwrap_number(answer) or answer
    lines = text.splitlines()
    if not lines:
        return None
    final = lines[-1].strip()
    if final.startswith("**") and final.endswith("**"):
        final = final[2:-2].strip()
    final = _ANSWER_PREFIX.sub("", final, count=1).strip()
    return _unwrap_number(final)


def score_response(
    text: str,
    gold_answer: str | None = None,
    require_format: bool = True,
) -> dict:
    """Return verifier components and a pass/fail bit.

    Pass requires:
    - scratchpad + Final answer format (if require_format)
    - exact-match final answer when gold_answer is provided
    """
    parsed = parse_response(text)
    fmt = parsed is not None
    pred = parsed[1] if parsed else extract_final_answer(text)
    answer_ok = None
    if gold_answer is not None:
        answer_ok = (
            pred is not None
            and normalize_answer(pred) == normalize_answer(str(gold_answer))
        )
    passed = True
    if require_format and not fmt:
        passed = False
    if gold_answer is not None and not answer_ok:
        passed = False
    return {
        "format_ok": fmt,
        "pred": pred,
        "answer_ok": answer_ok,
        "passed": passed,
        "reward": 1.0 if passed else 0.0,
    }
