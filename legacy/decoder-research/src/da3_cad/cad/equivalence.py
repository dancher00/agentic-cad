"""Geometry-equivalence checks for raw and parameterized CadQuery programs."""

from __future__ import annotations

import math

from da3_cad.models import ValidationResult


def compare_validation_geometry(
    raw: ValidationResult,
    parameterized: ValidationResult,
) -> dict[str, object]:
    """Compare sandbox results without claiming equivalence for invalid solids."""

    common: dict[str, object] = {
        "raw_valid": raw.valid,
        "parameterized_valid": parameterized.valid,
        "relative_tolerance": 1e-9,
        "absolute_tolerance": 1e-9,
    }
    if raw.valid != parameterized.valid:
        return {**common, "status": "validity-mismatch", "equivalent": False}
    if not raw.valid:
        return {
            **common,
            "status": "both-invalid-not-comparable",
            "equivalent": None,
        }
    if (
        raw.volume is None
        or parameterized.volume is None
        or raw.bbox is None
        or parameterized.bbox is None
    ):
        return {**common, "status": "missing-geometry", "equivalent": False}

    volume_difference = abs(raw.volume - parameterized.volume)
    bbox_differences = [
        abs(left - right) for left, right in zip(raw.bbox, parameterized.bbox, strict=True)
    ]
    equivalent = math.isclose(
        raw.volume,
        parameterized.volume,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ) and all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        for left, right in zip(raw.bbox, parameterized.bbox, strict=True)
    )
    return {
        **common,
        "status": "equivalent" if equivalent else "geometry-mismatch",
        "equivalent": equivalent,
        "volume_absolute_difference": volume_difference,
        "bbox_max_absolute_difference": max(bbox_differences),
    }
