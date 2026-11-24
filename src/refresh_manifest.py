"""Refresh sha256 hashes in data/manifest.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.manifest import load_manifest, refresh_hashes, save_manifest, verify_manifest_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh dataset manifest hashes")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    if args.verify_only:
        errors = verify_manifest_hashes(manifest, root)
        if errors:
            for err in errors:
                print(err)
            raise SystemExit(1)
        print(f"manifest hashes ok ({len(manifest.get('datasets', []))} entries)")
        return

    updated = refresh_hashes(manifest, root)
    save_manifest(updated, manifest_path)
    print(f"updated hashes in {manifest_path}")


if __name__ == "__main__":
    main()
