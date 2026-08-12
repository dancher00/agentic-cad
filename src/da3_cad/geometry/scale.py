"""Scale evidence kept separate from similarity-normalized geometry."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from da3_cad.models import FloatArray

_SCALE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?P<unit>mm|cm|m|in)$"
)
_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


@dataclass(frozen=True, slots=True)
class KnownDimension:
    parameter: str
    value_mm: float
    original: str

    @classmethod
    def parse(cls, value: str) -> KnownDimension:
        match = _SCALE_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "known dimension must be NAME=VALUEmm (also cm, m or in), "
                "for example hole_1_diameter=8mm"
            )
        numeric = float(match.group("value"))
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("known dimension must be finite and positive")
        unit = match.group("unit")
        return cls(
            parameter=match.group("name"),
            value_mm=numeric * _UNIT_TO_MM[unit],
            original=value.strip(),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "value_mm": self.value_mm,
            "original": self.original,
        }


@dataclass(frozen=True, slots=True)
class ScaleDecision:
    status: Literal["unresolved", "pending", "known"]
    source: Literal["none", "known-dimension", "fiducial", "metric-model", "camera-bundle"]
    units: Literal["normalized", "mm"]
    millimeters_per_unit: float | None
    reference: dict[str, object] | None
    warning: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "units": self.units,
            "millimeters_per_unit": self.millimeters_per_unit,
            "reference": self.reference,
            "warning": self.warning,
        }


NativeSpaceKind = Literal[
    "canonical-model-space",
    "metric-mm-space",
    "stub-test-space",
]


@dataclass(frozen=True, slots=True)
class CadCoordinateContract:
    """Explicit map from generated CAD coordinates to a unit cube and metric space.

    The normalized representation is isotropic: the largest bbox extent spans one
    unit and every bbox axis is centred at 0.5. Short axes are never stretched.
    """

    native_kind: NativeSpaceKind
    native_units: str
    native_bbox: tuple[float, float, float, float, float, float]
    native_center: tuple[float, float, float]
    native_largest_extent: float
    normalized_bbox: tuple[float, float, float, float, float, float]
    millimeters_per_native_unit: float | None
    scale_evidence: dict[str, object]

    @classmethod
    def from_bbox(
        cls,
        bbox: Sequence[float],
        *,
        native_kind: NativeSpaceKind,
        native_units: str,
        millimeters_per_native_unit: float | None,
        scale_evidence: dict[str, object],
    ) -> CadCoordinateContract:
        values = np.asarray(tuple(bbox), dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("CAD bbox must contain six finite coordinates")
        minimum = values[:3]
        maximum = values[3:]
        extents = maximum - minimum
        largest = float(extents.max())
        if np.any(extents <= 0.0) or not np.isfinite(largest):
            raise ValueError("CAD bbox must be nondegenerate on all three axes")
        if millimeters_per_native_unit is not None and (
            not np.isfinite(millimeters_per_native_unit) or millimeters_per_native_unit <= 0.0
        ):
            raise ValueError("millimeters_per_native_unit must be finite and positive")
        center = (minimum + maximum) / 2.0
        normalized_minimum = (minimum - center) / largest + 0.5
        normalized_maximum = (maximum - center) / largest + 0.5
        normalized = np.concatenate((normalized_minimum, normalized_maximum))
        return cls(
            native_kind=native_kind,
            native_units=native_units,
            native_bbox=tuple(float(item) for item in values),  # type: ignore[arg-type]
            native_center=tuple(float(item) for item in center),  # type: ignore[arg-type]
            native_largest_extent=largest,
            normalized_bbox=tuple(float(item) for item in normalized),  # type: ignore[arg-type]
            millimeters_per_native_unit=millimeters_per_native_unit,
            scale_evidence=scale_evidence,
        )

    @property
    def millimeters_per_normalized_unit(self) -> float | None:
        if self.millimeters_per_native_unit is None:
            return None
        return self.native_largest_extent * self.millimeters_per_native_unit

    def native_points_to_normalized(self, points: FloatArray) -> FloatArray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("native CAD points must have finite shape (N,3)")
        center = np.asarray(self.native_center, dtype=np.float64)
        return (values - center) / self.native_largest_extent + 0.5

    def native_length_to_normalized(self, length: float) -> float:
        if not np.isfinite(length):
            raise ValueError("native length must be finite")
        return float(length / self.native_largest_extent)

    def normalized_length_to_millimeters(self, length: float) -> float:
        factor = self.millimeters_per_normalized_unit
        if factor is None:
            raise ValueError("metric scale is unresolved")
        if not np.isfinite(length):
            raise ValueError("normalized length must be finite")
        return float(length * factor)

    def as_dict(self) -> dict[str, object]:
        metric_status = "known" if self.millimeters_per_native_unit is not None else "unresolved"
        return {
            "schema_version": "1.0",
            "native_space": {
                "kind": self.native_kind,
                "units": self.native_units,
                "bbox": list(self.native_bbox),
                "bbox_center": list(self.native_center),
                "largest_bbox_extent": self.native_largest_extent,
            },
            "normalized_cube": {
                "units": "normalized-cube-unit",
                "container": "[0,1]^3",
                "bbox": list(self.normalized_bbox),
                "transform": "u = (x_native - bbox_center) / largest_bbox_extent + 0.5",
                "isotropic": True,
                "short_axes_centered": True,
                "per_axis_scaling": False,
                "native_units_per_normalized_unit": self.native_largest_extent,
            },
            "metric_space": {
                "status": metric_status,
                "units": "mm",
                "millimeters_per_native_unit": self.millimeters_per_native_unit,
                "millimeters_per_normalized_unit": self.millimeters_per_normalized_unit,
                "transform": (
                    "x_mm = x_native * millimeters_per_native_unit"
                    if metric_status == "known"
                    else None
                ),
                "evidence": self.scale_evidence,
                "no_evidence_policy": (
                    None
                    if metric_status == "known"
                    else "do not label canonical or normalized-cube values as millimetres"
                ),
            },
        }


def cad_coordinate_contract(
    bbox: Sequence[float],
    *,
    backend: str,
    scale: ScaleDecision,
) -> CadCoordinateContract:
    """Build a backend-aware output-space contract after solid validation."""

    if backend in {"geometric-fitter-v1", "visual-hull-v1"} and scale.status == "known":
        kind: NativeSpaceKind = "metric-mm-space"
        units = "mm"
        mm_per_native = 1.0
    elif backend in {"geometric-fitter-v1", "visual-hull-v1"}:
        kind = "canonical-model-space"
        units = "canonical-model-unit"
        mm_per_native = None
    elif backend == "stub":
        kind = "stub-test-space"
        units = "stub-test-unit"
        mm_per_native = None
    else:
        raise ValueError(f"cannot assign coordinate semantics to backend {backend!r}")
    return CadCoordinateContract.from_bbox(
        bbox,
        native_kind=kind,
        native_units=units,
        millimeters_per_native_unit=mm_per_native,
        scale_evidence=scale.as_dict(),
    )


def unresolved_scale(known_dimension: KnownDimension | None = None) -> ScaleDecision:
    if known_dimension is None:
        return ScaleDecision(
            status="unresolved",
            source="none",
            units="normalized",
            millimeters_per_unit=None,
            reference=None,
            warning="no scale evidence; output units are explicitly normalized, not millimetres",
        )
    return ScaleDecision(
        status="pending",
        source="known-dimension",
        units="normalized",
        millimeters_per_unit=None,
        reference=known_dimension.as_dict(),
        warning=(
            f"known dimension {known_dimension.original} is pending until the generated "
            "parameter table contains that named feature"
        ),
    )


def resolve_camera_bundle_scale(
    *,
    world_units_to_mm: float,
    world_units_per_normalized_unit: float,
    camera_source: str,
    evidence: Mapping[str, object],
) -> ScaleDecision:
    """Map metric external-camera world scale into normalized object coordinates."""

    if not np.isfinite(world_units_to_mm) or world_units_to_mm <= 0.0:
        raise ValueError("world_units_to_mm must be finite and positive")
    if not np.isfinite(world_units_per_normalized_unit) or world_units_per_normalized_unit <= 0.0:
        raise ValueError("world_units_per_normalized_unit must be finite and positive")
    millimeters_per_unit = float(world_units_to_mm * world_units_per_normalized_unit)
    return ScaleDecision(
        status="known",
        source="camera-bundle",
        units="mm",
        millimeters_per_unit=millimeters_per_unit,
        reference={
            "camera_source": camera_source,
            "world_units_to_mm": float(world_units_to_mm),
            "world_units_per_normalized_unit": float(world_units_per_normalized_unit),
            "evidence": dict(evidence),
        },
        warning=None,
    )


def resolve_known_dimension(
    known_dimension: KnownDimension,
    parameters: Mapping[str, float],
) -> ScaleDecision:
    if known_dimension.parameter not in parameters:
        raise ValueError(
            f"generated parameter table has no {known_dimension.parameter!r}; "
            "scale remains unresolved rather than guessing a correspondence"
        )
    normalized_value = float(parameters[known_dimension.parameter])
    if not np.isfinite(normalized_value) or normalized_value <= 0.0:
        raise ValueError("known-dimension reference parameter must be finite and positive")
    factor = known_dimension.value_mm / normalized_value
    return ScaleDecision(
        status="known",
        source="known-dimension",
        units="mm",
        millimeters_per_unit=factor,
        reference={
            **known_dimension.as_dict(),
            "generated_parameter_value": normalized_value,
        },
        warning=None,
    )
