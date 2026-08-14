"""Fetch the pinned five-object Objectron release benchmark after license acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DATASET_PAGE = "https://github.com/google-research-datasets/Objectron"
LICENSE_URL = "https://github.com/google-research-datasets/Objectron/blob/master/LICENSE"
ACCEPTANCE = "c-uda-1.0"


@dataclass(frozen=True, slots=True)
class Source:
    object_id: str
    category: str
    sequence: str
    source_url: str
    bytes: int
    sha256: str
    annotation_url: str
    annotation_bytes: int
    annotation_sha256: str
    local_raw_name: str
    local_annotation_name: str


SOURCES = (
    Source(
        object_id="book",
        category="book",
        sequence="batch-47/25",
        source_url="https://storage.googleapis.com/objectron/videos/book/batch-47/25/video.MOV",
        bytes=20_805_866,
        sha256="6c5133d4b06cb63b4ab4fa006f9b25f701dee3e1759e33e143922424ea345df4",
        annotation_url=(
            "https://storage.googleapis.com/objectron/annotations/book/batch-47/25.pbdata"
        ),
        annotation_bytes=178_196,
        annotation_sha256="5680a22e807c20ae2afc6aa4920633e006519c17faa13d38f9c5988a2524feb0",
        local_raw_name="book_annotated",
        local_annotation_name="book",
    ),
    Source(
        object_id="bottle",
        category="bottle",
        sequence="batch-16/11",
        source_url="https://storage.googleapis.com/objectron/videos/bottle/batch-16/11/video.MOV",
        bytes=20_452_273,
        sha256="e2cd7c14edd56227e7df4b2f7216fdafe38c0ef9edd243ae2c1587a8e16cf9ad",
        annotation_url=(
            "https://storage.googleapis.com/objectron/annotations/bottle/batch-16/11.pbdata"
        ),
        annotation_bytes=168_553,
        annotation_sha256="3d9610e786e94f81ec62b69410d4b0b7d1d21525db81621b1756d2517a12d519",
        local_raw_name="bottle_cylindrical",
        local_annotation_name="bottle_cylindrical",
    ),
    Source(
        object_id="camera",
        category="camera",
        sequence="batch-1/0",
        source_url="https://storage.googleapis.com/objectron/videos/camera/batch-1/0/video.MOV",
        bytes=16_910_858,
        sha256="c7467e1fb940f748dc4c10c12277bcac8ce700fdd6faa83d7dea7aea86a3c5d1",
        annotation_url=(
            "https://storage.googleapis.com/objectron/annotations/camera/batch-1/0.pbdata"
        ),
        annotation_bytes=137_046,
        annotation_sha256="e149c6f811cba5d524280948fa85770ea469b02beb031f2891fdd96af2a29cc1",
        local_raw_name="camera",
        local_annotation_name="camera",
    ),
    Source(
        object_id="cup",
        category="cup",
        sequence="batch-1/0",
        source_url="https://storage.googleapis.com/objectron/videos/cup/batch-1/0/video.MOV",
        bytes=14_791_956,
        sha256="5258f48b874a764541bfd1f3a9d9d27eb79daf2721e7a908bd76e7e4b4ba1aca",
        annotation_url=(
            "https://storage.googleapis.com/objectron/annotations/cup/batch-1/0.pbdata"
        ),
        annotation_bytes=125_471,
        annotation_sha256="22a2fc4f100313249de560204a77dc800c85d473bd285dbbfcd89eac2c5ed191",
        local_raw_name="cup",
        local_annotation_name="cup",
    ),
    Source(
        object_id="laptop",
        category="laptop",
        sequence="batch-34/40",
        source_url="https://storage.googleapis.com/objectron/videos/laptop/batch-34/40/video.MOV",
        bytes=24_428_404,
        sha256="a4bc241c8ea07fc7caff36a1bbf2a0a20e6870778fdbf731c8aec1200ba3c7d7",
        annotation_url=(
            "https://storage.googleapis.com/objectron/annotations/laptop/batch-34/40.pbdata"
        ),
        annotation_bytes=196_202,
        annotation_sha256="03f65c0d969586eb351103517b42217e21d6df4fb38a6896a5edeb98a21eb85b",
        local_raw_name="laptop_visible",
        local_annotation_name="laptop_visible",
    ),
)


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    size, digest = _digest(path)
    if size != expected_bytes or digest != expected_sha256:
        raise RuntimeError(
            f"Objectron verification failed for {path}: expected "
            f"{expected_bytes} bytes/{expected_sha256}, got {size}/{digest}"
        )


def _download_verified(
    path: Path,
    *,
    url: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _verify(path, expected_bytes=expected_bytes, expected_sha256=expected_sha256)
        print(f"verified existing artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "DA3-CAD/0.4"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        _verify(
            temporary,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded and verified: {path}")


def _write_metadata(
    path: Path,
    video: Path,
    annotation: Path,
    source: Source,
) -> None:
    payload = {
        "schema_version": "2.0",
        "benchmark": "real-objectron-v1-target-localized",
        "dataset": "Google Objectron",
        "dataset_page": DATASET_PAGE,
        "license": "C-UDA-1.0",
        "license_url": LICENSE_URL,
        "object_id": source.object_id,
        "category": source.category,
        "sequence": source.sequence,
        "video": {
            "source_url": source.source_url,
            "bytes": source.bytes,
            "sha256": source.sha256,
            "local_path": str(video),
        },
        "annotation": {
            "source_url": source.annotation_url,
            "bytes": source.annotation_bytes,
            "sha256": source.annotation_sha256,
            "local_path": str(annotation),
            "role": "projected-3d-box target localization; not pixel-mask or CAD ground truth",
        },
        "retrieved_at": datetime.now(UTC).isoformat(),
        "redistributed": False,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = set(payload) - {"retrieved_at"}
        if any(existing.get(key) != payload[key] for key in stable_keys):
            raise RuntimeError(f"refusing to overwrite divergent metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fetch(root: Path, source: Source) -> None:
    video = root / "raw" / source.local_raw_name / "video.MOV"
    annotation = root / "annotations" / f"{source.local_annotation_name}.pbdata"
    metadata = root / "sources" / "target_localized" / f"{source.local_annotation_name}.json"
    _download_verified(
        video,
        url=source.source_url,
        expected_bytes=source.bytes,
        expected_sha256=source.sha256,
    )
    _download_verified(
        annotation,
        url=source.annotation_url,
        expected_bytes=source.annotation_bytes,
        expected_sha256=source.annotation_sha256,
    )
    _write_metadata(metadata, video, annotation, source)
    print(f"wrote source metadata: {metadata}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("captures/real_objects"))
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=[source.object_id for source in SOURCES],
        default=[source.object_id for source in SOURCES],
    )
    parser.add_argument("--accept-license", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [source for source in SOURCES if source.object_id in set(args.objects)]
    print("DA3-CAD five-object real benchmark:")
    print(f"- dataset: {DATASET_PAGE}")
    print(f"- terms: C-UDA-1.0 — {LICENSE_URL}")
    print("- raw videos and derived frames remain ignored and are not redistributed")
    print(f"- required acknowledgement: --accept-license {ACCEPTANCE}")
    for source in selected:
        target = args.root / "raw" / source.local_raw_name / "video.MOV"
        annotation = args.root / "annotations" / f"{source.local_annotation_name}.pbdata"
        print(
            f"- {source.object_id}: {source.source_url} -> {target}; "
            f"{source.bytes} bytes; SHA-256 {source.sha256}"
        )
        print(
            f"  annotation: {source.annotation_url} -> {annotation}; "
            f"{source.annotation_bytes} bytes; SHA-256 {source.annotation_sha256}"
        )
    if args.dry_run:
        return
    if args.accept_license != ACCEPTANCE:
        raise SystemExit(
            f"review the displayed dataset terms, then pass --accept-license {ACCEPTANCE}"
        )
    for source in selected:
        _fetch(args.root, source)


if __name__ == "__main__":
    main()
