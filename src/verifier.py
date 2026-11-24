"""Rule + exact-match verifier for RL-lite (Phase 5)."""

from __future__ import annotations

from src.eval_gsm8k_style import normalize_answer
from src.template import extract_final_answer, has_scratchpad_and_final


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
    fmt = has_scratchpad_and_final(text)
    pred = extract_final_answer(text)
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
