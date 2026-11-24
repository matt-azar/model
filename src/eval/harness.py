"""Unified eval harness runner."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.eval.adapters import ADAPTERS
from src.eval.report import sha256_file, write_report


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_eval_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_harness(
    ckpt: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    stem: str = "report",
    device_override: str | None = None,
) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    temperature = float(cfg.get("temperature", 0.0))
    max_new_tokens = int(cfg.get("max_new_tokens", 96))
    device = device_override or str(cfg.get("device", "auto"))

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    t0 = time.time()
    benchmarks: dict[str, Any] = {}
    for bench in cfg.get("benchmarks", []):
        name = bench["name"]
        adapter_name = bench["adapter"]
        if adapter_name not in ADAPTERS:
            raise KeyError(f"unknown adapter: {adapter_name}")
        adapter = ADAPTERS[adapter_name]
        kwargs = dict(bench.get("args") or {})
        path = bench.get("path")
        result = adapter(
            ckpt=ckpt_path,
            path=path,
            max_new_tokens=int(bench.get("max_new_tokens", max_new_tokens)),
            temperature=float(bench.get("temperature", temperature)),
            device=device,
            seed=seed,
            **kwargs,
        )
        benchmarks[name] = {
            "adapter": adapter_name,
            "path": path,
            "summary": result["summary"],
            "results": result.get("results") or [],
        }
        print(f"[eval] {name}: {json.dumps(result['summary'], sort_keys=True)}")

    report: dict[str, Any] = {
        "harness_version": 1,
        "git_sha": git_sha(),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "config_path": str(config_path),
        "seed": seed,
        "temperature": temperature,
        "device_name": device,
        "wall_seconds": time.time() - t0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nondeterminism_allowlist": sorted(
            {
                "wall_seconds",
                "generated_at",
                "hostname",
                "device_name",
                "artifacts",
            }
        ),
        "benchmarks": benchmarks,
    }

    paths = write_report(report, out_dir, stem=stem)
    report["artifacts"] = {k: str(v) for k, v in paths.items()}
    # Rewrite with artifacts included.
    write_report(report, out_dir, stem=stem)
    print(f"wrote report to {out_dir}/{stem}.{{json,md,canonical.json}}")
    return report
