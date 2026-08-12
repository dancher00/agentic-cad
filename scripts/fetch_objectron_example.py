"""Fetch one pinned Objectron video after explicit dataset-license acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DATASET_PAGE = "https://github.com/google-research-datasets/Objectron"
LICENSE_URL = "https://github.com/google-research-datasets/Objectron/blob/master/LICENSE"
SOURCE_URL = "https://storage.googleapis.com/objectron/videos/camera/batch-1/0/video.MOV"
SOURCE_SHA256 = "c7467e1fb940f748dc4c10c12277bcac8ce700fdd6faa83d7dea7aea86a3c5d1"
SOURCE_BYTES = 16_910_858
ACCEPTANCE = "c-uda-1.0"


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify(path: Path) -> None:
    size, digest = _digest(path)
    if size != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise RuntimeError(
            f"Objectron video verification failed for {path}: "
            f"expected {SOURCE_BYTES} bytes/{SOURCE_SHA256}, got {size}/{digest}"
        )


def _write_metadata(path: Path, video: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "dataset": "Google Objectron",
        "dataset_page": DATASET_PAGE,
        "license": "C-UDA-1.0",
        "license_url": LICENSE_URL,
        "category": "camera",
        "sequence": "batch-1/0",
        "source_url": SOURCE_URL,
        "bytes": SOURCE_BYTES,
        "sha256": SOURCE_SHA256,
        "local_video": str(video),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "redistributed": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = set(payload) - {"retrieved_at"}
        if any(existing.get(key) != payload[key] for key in stable_keys):
            raise RuntimeError(f"refusing to overwrite divergent metadata: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("captures/objectron_camera"))
    parser.add_argument("--accept-license", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root: Path = args.root
    video = root / "raw" / "objectron_camera_batch-1_0.MOV"
    metadata = root / "source.json"
    print("Optional DA3-CAD real-video example:")
    print(f"- dataset: {DATASET_PAGE}")
    print(f"- terms: C-UDA-1.0 — {LICENSE_URL}")
    print(f"- source: {SOURCE_URL}")
    print(f"- expected: {SOURCE_BYTES} bytes; SHA-256 {SOURCE_SHA256}")
    print(f"- target: {video} (ignored; not redistributed)")
    print(f"- required acknowledgement: --accept-license {ACCEPTANCE}")
    if args.dry_run:
        return
    if args.accept_license != ACCEPTANCE:
        raise SystemExit(
            f"review the displayed dataset terms, then pass --accept-license {ACCEPTANCE}"
        )

    video.parent.mkdir(parents=True, exist_ok=True)
    if video.exists():
        _verify(video)
        print(f"verified existing video: {video}")
    else:
        temporary = video.with_name(f".{video.name}.{os.getpid()}.partial")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "DA3-CAD/0.2"})
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("xb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            _verify(temporary)
            os.replace(temporary, video)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"downloaded and verified: {video}")
    _write_metadata(metadata, video)
    print(f"wrote source metadata: {metadata}")


if __name__ == "__main__":
    main()
