"""Typed data passed between DA3-CAD stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class ImageObservation:
    """Metadata for one decoded, EXIF-corrected input image."""

    path: Path
    relative_path: str
    sha256: str
    width: int
    height: int
    exif_orientation: int | None
    mean_luma: float
    blur_score: float
    perceptual_hash: str
    exact_duplicate_of: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "exif_orientation": self.exif_orientation,
            "mean_luma": self.mean_luma,
            "blur_score": self.blur_score,
            "perceptual_hash": self.perceptual_hash,
            "exact_duplicate_of": self.exact_duplicate_of,
        }


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """Deterministically ordered input observations."""

    root: Path
    images: tuple[ImageObservation, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "digest": self.digest,
            "count": len(self.images),
            "images": [item.as_dict() for item in self.images],
        }


@dataclass(frozen=True, slots=True)
class DepthPrediction:
    """Backend-neutral depth prediction contract."""

    depth: FloatArray
    confidence: FloatArray | None
    intrinsics: FloatArray
    extrinsics: FloatArray
    processed_images: tuple[UInt8Array, ...]
    backend: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.processed_images)
        if self.depth.ndim != 3 or self.depth.shape[0] != count:
            raise ValueError("depth must have shape (N,H,W) matching processed_images")
        if self.confidence is not None and self.confidence.shape != self.depth.shape:
            raise ValueError("confidence must match depth shape")
        if self.intrinsics.shape != (count, 3, 3):
            raise ValueError("intrinsics must have shape (N,3,3)")
        if self.extrinsics.shape not in {(count, 3, 4), (count, 4, 4)}:
            raise ValueError("extrinsics must have shape (N,3,4) or (N,4,4)")

    def summary(self) -> dict[str, object]:
        finite = np.isfinite(self.depth)
        return {
            "backend": self.backend,
            "shape": list(self.depth.shape),
            "finite_depth_fraction": float(finite.mean()),
            "mean_confidence": (
                float(self.confidence.mean()) if self.confidence is not None else None
            ),
            "intrinsics_shape": list(self.intrinsics.shape),
            "extrinsics_shape": list(self.extrinsics.shape),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class CadProgram:
    """Generated parameterized CadQuery source."""

    source: str
    parameters: dict[str, float]
    backend: str
    program_family: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of executing and validating a generated program."""

    valid: bool
    error: str | None
    volume: float | None
    bbox: tuple[float, float, float, float, float, float] | None
    execution_seconds: float
    step_path: Path | None = None
    stl_path: Path | None = None
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "error": self.error,
            "volume": self.volume,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "execution_seconds": self.execution_seconds,
            "step_path": self.step_path.name if self.step_path else None,
            "stl_path": self.stl_path.name if self.stl_path else None,
            "details": self.details,
        }
