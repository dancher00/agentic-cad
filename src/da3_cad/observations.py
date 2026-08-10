"""Image discovery, decoding and inexpensive pre-flight diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from da3_cad.models import ImageObservation, ObservationSet, UInt8Array

IMAGE_EXTENSIONS: Final = {".jpg", ".jpeg", ".png"}
EXIF_ORIENTATION: Final = 274
CAPTURE_BENCHMARK: Final = {
    "status": "no-numeric-capture-threshold-established",
    "tested_view_counts": [8, 16, 24, 32],
    "numeric_warning_below": None,
    "numeric_minimum_views": None,
    "numeric_recommended_views": None,
    "reason": (
        "the common-object reconstruction curve was non-monotone and neither "
        "GT-blind parameter criterion passed its gate at any tested view count"
    ),
    "capture_design_limit": (
        "the nested schedule changes image count and angular fill together; "
        "count versus separation is not causally identified"
    ),
    "sources": [
        {
            "path": "benchmarks/high_view_sweep/report.json",
            "sha256": "29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072",
        },
        {
            "path": "benchmarks/gt_blind_view_curve/report.json",
            "sha256": "fee8e65a31608fcaf5bd24673b4eb587578c6afb22e18e4edb9325f3c5f4e5e6",
        },
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> UInt8Array:
    with Image.open(path) as image:
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(corrected, dtype=np.uint8)


def _luma(rgb: UInt8Array) -> npt.NDArray[np.float64]:
    values = rgb.astype(np.float64)
    result: npt.NDArray[np.float64] = (
        0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    )
    return result


def _blur_score(gray: npt.NDArray[np.float64]) -> float:
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(np.var(laplacian))


def _average_hash(rgb: UInt8Array) -> str:
    image = Image.fromarray(rgb).convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float64)
    bits = values >= values.mean()
    number = 0
    for bit in bits.flat:
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def discover_images(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")
    paths = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
    )
    if not paths:
        raise ValueError(f"no JPG or PNG images found in: {root}")
    return paths


def load_observations(root: Path) -> ObservationSet:
    root = root.resolve()
    seen_digests: dict[str, str] = {}
    observations: list[ImageObservation] = []
    set_digest = hashlib.sha256()

    for path in discover_images(root):
        digest = _sha256(path)
        with Image.open(path) as image:
            orientation_raw = image.getexif().get(EXIF_ORIENTATION)
            orientation = int(orientation_raw) if orientation_raw is not None else None
        rgb = load_rgb(path)
        gray = _luma(rgb)
        relative = path.relative_to(root).as_posix()
        observations.append(
            ImageObservation(
                path=path,
                relative_path=relative,
                sha256=digest,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                exif_orientation=orientation,
                mean_luma=float(gray.mean()),
                blur_score=_blur_score(gray),
                perceptual_hash=_average_hash(rgb),
                exact_duplicate_of=seen_digests.get(digest),
            )
        )
        seen_digests.setdefault(digest, relative)
        set_digest.update(relative.encode())
        set_digest.update(b"\0")
        set_digest.update(digest.encode())
        set_digest.update(b"\0")

    return ObservationSet(root=root, images=tuple(observations), digest=set_digest.hexdigest())


def doctor_report(observations: ObservationSet) -> dict[str, object]:
    images = observations.images
    warnings: list[str] = []
    exact_duplicates = [item.relative_path for item in images if item.exact_duplicate_of]
    near_pairs: list[tuple[str, str]] = []
    for index, left in enumerate(images):
        for right in images[index + 1 :]:
            if (
                left.sha256 != right.sha256
                and _hamming(left.perceptual_hash, right.perceptual_hash) <= 4
            ):
                near_pairs.append((left.relative_path, right.relative_path))

    if len(images) < 3:
        warnings.append("fewer than 3 views: unseen geometry will be inferred")
    if exact_duplicates:
        warnings.append(f"exact duplicate inputs: {', '.join(exact_duplicates)}")
    if near_pairs:
        warnings.append(f"{len(near_pairs)} suspected near-duplicate view pair(s)")
    orientations = {"portrait" if item.height > item.width else "landscape" for item in images}
    if len(orientations) > 1:
        warnings.append("mixed portrait/landscape inputs; EXIF rotation was applied")
    exposure_values = np.array([item.mean_luma for item in images], dtype=np.float64)
    if float(np.ptp(exposure_values)) > 80.0:
        warnings.append("large exposure spread across views")
    blurry = [item.relative_path for item in images if item.blur_score < 20.0]
    if blurry:
        warnings.append(f"low high-frequency detail in: {', '.join(blurry)}")

    unique_view_signatures = len({item.perceptual_hash for item in images})
    coverage = "insufficient" if unique_view_signatures < 3 else "plausible"
    if coverage == "insufficient":
        warnings.append("view diversity appears insufficient; add views from different sides")

    return {
        "input_digest": observations.digest,
        "count": len(images),
        "resolutions": sorted({f"{item.width}x{item.height}" for item in images}),
        "exif_orientations": [item.exif_orientation for item in images],
        "mean_luma_range": [float(exposure_values.min()), float(exposure_values.max())],
        "blur_scores": {item.relative_path: item.blur_score for item in images},
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": [list(pair) for pair in near_pairs],
        "coverage_heuristic": coverage,
        "capture_benchmark": CAPTURE_BENCHMARK,
        "verdict": "ready-with-warnings" if warnings else "ready",
        "warnings": warnings,
        "note": "coverage is an image-diversity heuristic, not recovered camera-pose proof",
    }
