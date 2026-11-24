"""Benchmark adapters for the Phase 4 harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from src.data import make_dataloader, load_texts
from src.eval_gsm8k_style import evaluate as eval_gsm8k_style
from src.generate import Generator, generate_text
from src.template import format_prompt, has_scratchpad_and_final
from src.train import resolve_device


AdapterFn = Callable[..., dict[str, Any]]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def adapter_gsm8k_style(
    ckpt: str | Path,
    path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str,
    batch_size: int = 8,
    **_: Any,
) -> dict[str, Any]:
    report = eval_gsm8k_style(
        ckpt,
        path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        batch_size=batch_size,
    )
    return {
        "summary": {
            "exact_match": report["exact_match"],
            "format_rate": report["format_rate"],
            "truncation_rate": report["truncation_rate"],
            "mean_generated_tokens": report["mean_generated_tokens"],
            "n": report["n"],
            "n_correct": report["n_correct"],
        },
        "results": report["results"],
    }


def adapter_personal_smoke(
    ckpt: str | Path,
    path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str,
    **_: Any,
) -> dict[str, Any]:
    rows = _load_jsonl(Path(path))
    results = []
    matched = 0
    format_ok = 0
    format_n = 0
    generator = Generator.from_checkpoint(ckpt, device_name=device)
    for row in rows:
        gen = generator.complete(
            row["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        expected = row.get("expected")
        ok = None
        if expected is not None:
            ok = str(expected).lower() in gen.lower()
            if ok:
                matched += 1
        want_fmt = bool(row.get("expect_scratchpad"))
        fmt = None
        if want_fmt:
            format_n += 1
            fmt = has_scratchpad_and_final(gen)
            if fmt:
                format_ok += 1
        results.append(
            {
                "id": row.get("id"),
                "match": ok,
                "format_ok": fmt,
                "generation": gen,
            }
        )
    scored = sum(1 for r in rows if r.get("expected") is not None)
    return {
        "summary": {
            "n": len(rows),
            "substring_acc": (matched / scored) if scored else None,
            "format_rate": (format_ok / format_n) if format_n else None,
            "n_matched": matched,
            "n_format_ok": format_ok,
        },
        "results": results,
    }


def adapter_code_smoke(
    ckpt: str | Path,
    path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str,
    **_: Any,
) -> dict[str, Any]:
    """Tiny MBPP-style: generate a function body and exec local asserts."""
    rows = _load_jsonl(Path(path))
    results = []
    passed = 0
    for row in rows:
        prompt = row["prompt"]
        gen = generate_text(
            ckpt,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device_name=device,
        )
        # Prefer extracting a def block; fall back to code_prefix + continuation.
        code = _extract_code(gen, row.get("code_prefix", ""))
        ok, detail = _run_code_tests(code, row.get("tests") or [])
        if ok:
            passed += 1
        results.append(
            {
                "id": row.get("id"),
                "passed": ok,
                "detail": detail,
                "code": code,
                "generation": gen,
            }
        )
    n = len(rows)
    return {
        "summary": {
            "n": n,
            "pass_rate": (passed / n) if n else 0.0,
            "n_passed": passed,
        },
        "results": results,
    }


def _extract_code(generation: str, code_prefix: str) -> str:
    text = generation
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            block = part
            if block.startswith("python"):
                block = block[len("python") :]
            block = block.strip()
            if "def " in block or code_prefix:
                text = block
                break
    if code_prefix and code_prefix not in text:
        # Model may continue after the prefix prompt.
        text = code_prefix + text.split(code_prefix)[-1]
    # Truncate at common stop markers.
    for stop in ["\n###", "\nassert", "\n```", "\nFinal answer"]:
        if stop == "\nassert":
            continue
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def _run_code_tests(code: str, tests: list[str]) -> tuple[bool, str]:
    if not tests:
        return False, "no tests"
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "list": list,
        "dict": dict,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    env: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(code, env, env)  # noqa: S102  # intentional tiny local sandbox
        for test in tests:
            exec(test, env, env)  # noqa: S102
        return True, "ok"
    except Exception as exc:  # noqa: BLE001  # surface failure reason in report
        return False, f"{type(exc).__name__}: {exc}"


def adapter_logic_mc(
    ckpt: str | Path,
    path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str,
    **_: Any,
) -> dict[str, Any]:
    rows = _load_jsonl(Path(path))
    results = []
    correct = 0
    generator = Generator.from_checkpoint(ckpt, device_name=device)
    for row in rows:
        choices = row["choices"]
        choice_lines = "\n".join(f"{k}. {v}" for k, v in choices.items())
        prompt = (
            f"{format_prompt(row['question'] + chr(10) + choice_lines)}"
            f"Reply with the letter only.\n"
        )
        gen = generator.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        pred = _parse_choice(gen, list(choices.keys()))
        ok = pred == str(row["answer"]).upper()
        if ok:
            correct += 1
        results.append(
            {
                "id": row.get("id"),
                "gold": row["answer"],
                "pred": pred,
                "match": ok,
                "generation": gen,
            }
        )
    n = len(rows)
    return {
        "summary": {
            "n": n,
            "accuracy": (correct / n) if n else 0.0,
            "n_correct": correct,
        },
        "results": results,
    }


def _parse_choice(text: str, letters: list[str]) -> str | None:
    upper = text.upper()
    for letter in letters:
        token = letter.upper()
        if f"FINAL ANSWER: {token}" in upper:
            return token
    # First standalone letter occurrence after Response.
    for letter in letters:
        token = letter.upper()
        if f" {token}." in upper or f"\n{token}\n" in upper or upper.strip().startswith(token):
            return token
    for letter in letters:
        if letter.upper() in upper:
            return letter.upper()
    return None


@torch.no_grad()
def adapter_val_lm_loss(
    ckpt: str | Path,
    path: str | Path,
    device: str,
    max_batches: int = 20,
    seq_len: int = 128,
    batch_size: int = 8,
    seed: int = 42,
    **_: Any,
) -> dict[str, Any]:
    from src.generate import _load_model_and_tokenizer

    dev = resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(ckpt, dev)
    model.eval()
    texts = load_texts(path)
    loader = make_dataloader(
        texts, tokenizer, seq_len=seq_len, batch_size=batch_size, seed=seed
    )
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(dev)
        labels = batch["labels"].to(dev)
        _, loss = model(input_ids, labels)
        total += float(loss.item())
        n += 1
    val_loss = total / n if n else float("nan")
    return {
        "summary": {
            "val_loss": val_loss,
            "n_batches": n,
            "path": str(path),
        },
        "results": [],
    }


def adapter_gsm8k_official(
    ckpt: str | Path,
    path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Official GSM8K-format JSONL (question / scratchpad / answer)."""
    return adapter_gsm8k_style(
        ckpt=ckpt,
        path=path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        **kwargs,
    )


ADAPTERS: dict[str, AdapterFn] = {
    "gsm8k_style": adapter_gsm8k_style,
    "gsm8k_official": adapter_gsm8k_official,
    "personal_smoke": adapter_personal_smoke,
    "code_smoke": adapter_code_smoke,
    "logic_mc": adapter_logic_mc,
    "val_lm_loss": adapter_val_lm_loss,
}
