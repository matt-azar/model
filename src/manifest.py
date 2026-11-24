"""Dataset manifest with licenses and content hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def refresh_hashes(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    """Fill/update sha256 for each dataset entry that has a path."""
    root = Path(root)
    out = dict(manifest)
    datasets = []
    for entry in manifest.get("datasets", []):
        item = dict(entry)
        rel = item.get("path")
        if rel:
            file_path = root / rel
            if file_path.is_file():
                item["sha256"] = sha256_file(file_path)
                item["bytes"] = file_path.stat().st_size
            else:
                item["sha256"] = None
                item["bytes"] = None
        datasets.append(item)
    out["datasets"] = datasets
    return out


def verify_manifest_hashes(manifest: dict[str, Any], root: str | Path) -> list[str]:
    """Return list of error strings; empty means ok."""
    root = Path(root)
    errors: list[str] = []
    for entry in manifest.get("datasets", []):
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            continue
        file_path = root / rel
        if not file_path.is_file():
            errors.append(f"missing file for {entry.get('name')}: {rel}")
            continue
        actual = sha256_file(file_path)
        if actual != expected:
            errors.append(
                f"hash mismatch for {entry.get('name')}: expected {expected}, got {actual}"
            )
    return errors
