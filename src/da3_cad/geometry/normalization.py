"""Isotropic bounding-box normalization for canonical object coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_cad.models import FloatArray


@dataclass(frozen=True, slots=True)
class BboxNormalization:
    """Isotropic largest-extent transform with centered short axes."""

    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    midpoint: tuple[float, float, float]
    largest_extent: float

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "midpoint": list(self.midpoint),
            "largest_extent": self.largest_extent,
            "unit_formula": "u = (p - midpoint) / largest_extent + 0.5",
            "centered_formula": "c = (u - 0.5) * 2",
        }


def normalize_bbox_isotropic(
    points: FloatArray,
) -> tuple[FloatArray, FloatArray, BboxNormalization]:
    """Map points isotropically to [0,1]^3 and centered [-1,1]^3."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("normalization points must have non-empty shape (N,3)")
    if not np.isfinite(values).all():
        raise ValueError("normalization points must be finite")

    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    extents = maximum - minimum
    largest = float(extents.max())
    if not np.isfinite(largest) or largest <= 1e-12:
        raise ValueError("normalization rejects a degenerate zero-extent cloud")
    midpoint = (minimum + maximum) / 2.0
    unit = (values - midpoint) / largest + 0.5
    centered = (unit - 0.5) * 2.0
    transform = BboxNormalization(
        bbox_min=(float(minimum[0]), float(minimum[1]), float(minimum[2])),
        bbox_max=(float(maximum[0]), float(maximum[1]), float(maximum[2])),
        midpoint=(float(midpoint[0]), float(midpoint[1]), float(midpoint[2])),
        largest_extent=largest,
    )
    return unit.astype(np.float32), centered.astype(np.float32), transform
