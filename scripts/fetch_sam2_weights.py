"""Acquire and fully verify the official Apache-2.0 SAM2.1 Small checkpoint."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from da3_cad.segmentation.sam2_box import (
    SAM2_CHECKPOINT_SHA256,
    SAM2_CHECKPOINT_URL,
    verify_sam2_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/checkpoints/sam2.1_hiera_small.pt"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target: Path = args.target
    print("SAM2.1 Hiera Small checkpoint (Apache-2.0):")
    print(f"- source: {SAM2_CHECKPOINT_URL}")
    print(f"- expected SHA-256: {SAM2_CHECKPOINT_SHA256}")
    print(f"- target (ignored, not redistributed): {target}")
    if args.dry_run:
        return
    if target.exists():
        report = verify_sam2_checkpoint(target)
        print(f"verified existing checkpoint: {report}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    request = urllib.request.Request(SAM2_CHECKPOINT_URL, headers={"User-Agent": "DA3-CAD/0.4"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_sam2_checkpoint(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded and verified: {verify_sam2_checkpoint(target)}")


if __name__ == "__main__":
    main()
