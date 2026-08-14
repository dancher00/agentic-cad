"""Acquire the exact Apache-2.0 SAM2 source revision used by DA3-CAD."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from da3_cad.segmentation.sam2_box import SAM2_SOURCE_REVISION, SAM2_SOURCE_URL


def _run(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(list(arguments), check=True, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/upstream/SAM2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target: Path = args.target
    print(f"SAM2 source: {SAM2_SOURCE_URL}@{SAM2_SOURCE_REVISION}")
    print("Source and official SAM2.1 checkpoints: Apache-2.0.")
    print(f"Target (ignored, not redistributed): {target}")
    if args.dry_run:
        return
    if (target / ".git").is_dir():
        dirty = _run("git", "-C", str(target), "status", "--porcelain", capture=True)
        if dirty:
            raise RuntimeError(f"refusing to alter dirty external checkout: {target}")
        current = _run("git", "-C", str(target), "rev-parse", "HEAD", capture=True)
        if current != SAM2_SOURCE_REVISION:
            _run("git", "-C", str(target), "fetch", "origin", SAM2_SOURCE_REVISION)
            _run("git", "-C", str(target), "checkout", "--detach", SAM2_SOURCE_REVISION)
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"target exists and is not a git checkout: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--no-checkout", SAM2_SOURCE_URL, str(target))
        _run("git", "-C", str(target), "checkout", "--detach", SAM2_SOURCE_REVISION)
    actual = _run("git", "-C", str(target), "rev-parse", "HEAD", capture=True)
    if actual != SAM2_SOURCE_REVISION:
        raise RuntimeError(
            f"SAM2 source verification failed: expected {SAM2_SOURCE_REVISION}, got {actual}"
        )
    print(f"verified {target} at {actual}")


if __name__ == "__main__":
    main()
