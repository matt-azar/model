"""Report writing and canonicalization for deterministic comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# Fields allowed to differ across identical harness runs.
NONDETERMINISM_ALLOWLIST = {
    "wall_seconds",
    "generated_at",
    "hostname",
    "device_name",  # cpu vs cuda label only; scores must still match
    "artifacts",  # output paths depend on --out
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonicalize(report: dict[str, Any]) -> dict[str, Any]:
    """Drop allowlisted nondeterministic fields for equality checks."""

    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: scrub(v)
                for k, v in obj.items()
                if k not in NONDETERMINISM_ALLOWLIST
            }
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        return obj

    return scrub(report)


def reports_equal(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    ca = json.dumps(canonicalize(a), sort_keys=True, ensure_ascii=False)
    cb = json.dumps(canonicalize(b), sort_keys=True, ensure_ascii=False)
    if ca == cb:
        return True, ""
    return False, "canonical report mismatch (see dumped JSON diffs)"


def write_report(report: dict[str, Any], out_dir: str | Path, stem: str = "report") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    canon_path = out_dir / f"{stem}.canonical.json"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    canon = canonicalize(report)
    canon_path.write_text(json.dumps(canon, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": json_path, "md": md_path, "canonical": canon_path}


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval harness report",
        "",
        f"- git_sha: `{report.get('git_sha')}`",
        f"- ckpt: `{report.get('ckpt')}`",
        f"- ckpt_sha256: `{report.get('ckpt_sha256')}`",
        f"- seed: `{report.get('seed')}`",
        f"- temperature: `{report.get('temperature')}`",
        "",
        "## Benchmarks",
        "",
        "| name | metric | value |",
        "|------|--------|-------|",
    ]
    for name, bench in (report.get("benchmarks") or {}).items():
        summary = bench.get("summary") or {}
        for metric, value in summary.items():
            if isinstance(value, float):
                cell = f"{value:.4f}"
            else:
                cell = str(value)
            lines.append(f"| {name} | {metric} | {cell} |")
    lines.append("")
    return "\n".join(lines) + "\n"
