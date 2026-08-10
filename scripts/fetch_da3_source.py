"""Acquire the exact Apache-2.0 DA3 source revision used by the adapter."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SOURCE_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3"
SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def _run(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(
        list(arguments),
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/upstream/Depth-Anything-3"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target: Path = args.target
    print(f"DA3 source: {SOURCE_URL}@{SOURCE_REVISION}")
    print("Source license: Apache-2.0; model checkpoint licenses are separate.")
    print(f"Target (ignored, not redistributed): {target}")
    if args.dry_run:
        return

    if (target / ".git").is_dir():
        dirty = _run("git", "-C", str(target), "status", "--porcelain", capture=True)
        if dirty:
            raise RuntimeError(f"refusing to alter dirty external checkout: {target}")
        current = _run("git", "-C", str(target), "rev-parse", "HEAD", capture=True)
        if current != SOURCE_REVISION:
            _run("git", "-C", str(target), "fetch", "origin", SOURCE_REVISION)
            _run("git", "-C", str(target), "checkout", "--detach", SOURCE_REVISION)
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"target exists and is not a git checkout: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--no-checkout", SOURCE_URL, str(target))
        _run("git", "-C", str(target), "checkout", "--detach", SOURCE_REVISION)

    actual = _run("git", "-C", str(target), "rev-parse", "HEAD", capture=True)
    if actual != SOURCE_REVISION:
        raise RuntimeError(f"source verification failed: expected {SOURCE_REVISION}, got {actual}")
    print(f"verified {target} at {actual}")


if __name__ == "__main__":
    main()
