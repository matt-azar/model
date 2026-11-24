"""Phase 2 exit checks: holdout curve, coherent generation, reproducibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate import generate_text
from src.train import load_checkpoint


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_holdout_curve(val_rows: list[dict], max_final: float | None) -> dict:
    if len(val_rows) < 2:
        raise SystemExit(
            f"need >=2 val_loss points for a curve, got {len(val_rows)}"
        )
    first = float(val_rows[0]["val_loss"])
    last = float(val_rows[-1]["val_loss"])
    improved = last < first
    ok = improved
    detail = {
        "first_val_loss": first,
        "final_val_loss": last,
        "improved": improved,
        "n_points": len(val_rows),
    }
    if max_final is not None:
        detail["max_final"] = max_final
        detail["under_max_final"] = last <= max_final
        ok = ok and last <= max_final
    detail["passed"] = ok
    if not ok:
        raise SystemExit(f"holdout curve check failed: {detail}")
    return detail


def check_coherent_generation(
    ckpt: Path,
    prompts: list[dict],
    max_new_tokens: int,
) -> dict:
    results = []
    matched = 0
    for case in prompts:
        text = generate_text(
            ckpt,
            prompt=case["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        expected = str(case["expected"])
        ok = expected.lower() in text.lower()
        if ok:
            matched += 1
        results.append(
            {
                "id": case.get("id"),
                "prompt": case["prompt"],
                "expected": expected,
                "generation": text,
                "match": ok,
            }
        )
    report = {
        "n": len(prompts),
        "matched": matched,
        "passed": matched >= max(1, (len(prompts) + 1) // 2),
        "results": results,
    }
    if not report["passed"]:
        raise SystemExit(f"coherent generation check failed: {report}")
    return report


def check_reproducibility(run_dir: Path, ckpt: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    cfg_path = run_dir / "config.yaml"
    if not meta_path.is_file() or not cfg_path.is_file():
        raise SystemExit("missing run_meta.json or config.yaml in run dir")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    blob = load_checkpoint(ckpt, map_location="cpu")
    ckpt_meta = blob.get("run_meta") or {}
    if not meta.get("git_sha") or meta["git_sha"] == "unknown":
        raise SystemExit(f"run_meta git_sha missing/unknown: {meta}")
    if ckpt_meta.get("git_sha") != meta["git_sha"]:
        raise SystemExit(
            f"checkpoint git_sha mismatch: ckpt={ckpt_meta.get('git_sha')} "
            f"run_meta={meta.get('git_sha')}"
        )
    if not meta.get("manifest_sha256"):
        raise SystemExit("run_meta missing manifest_sha256")
    return {
        "git_sha": meta["git_sha"],
        "manifest_sha256": meta["manifest_sha256"],
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "config_copied": True,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 2 pretrain exit criteria")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-final-val-loss", type=float, default=2.5)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--prompts",
        default="data/personal_smoke.jsonl",
        help="JSONL with prompt/expected for coherence checks",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "ckpt_latest.pt"
    val_rows = load_jsonl(run_dir / "val_metrics.jsonl")
    prompts = load_jsonl(Path(args.prompts))
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")

    holdout = check_holdout_curve(val_rows, args.max_final_val_loss)
    gen = check_coherent_generation(ckpt, prompts, args.max_new_tokens)
    repro = check_reproducibility(run_dir, ckpt)

    report = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt),
        "holdout": holdout,
        "generation": {k: v for k, v in gen.items() if k != "results"},
        "generation_results": gen["results"],
        "reproducibility": repro,
        "passed": True,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "generation_results"}, indent=2))
    for row in gen["results"]:
        flag = "PASS" if row["match"] else "FAIL"
        print(f"[{flag}] {row.get('id')}: {row['generation']!r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print("PHASE2 VERIFY OK")


if __name__ == "__main__":
    main()
