"""Scale evidence kept separate from similarity-normalized geometry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

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
    source: Literal["none", "known-dimension", "fiducial", "metric-model"]
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
