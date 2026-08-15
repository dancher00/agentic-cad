"""Kernel-level validation for generated CadQuery B-Rep results."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cadquery as cq


@dataclass(frozen=True, slots=True)
class CadKernelValidation:
    solids: int
    faces: int
    edges: int
    volume: float
    valid: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _solids(result: Any) -> list[Any]:
    if isinstance(result, cq.Workplane):
        return list(result.solids().vals())
    if isinstance(result, cq.Shape):
        values = list(result.Solids())
        return values if values else [result]
    raise ValueError(f"CAD result has unsupported type: {type(result).__name__}")


def normalize_step_metadata(path: Path) -> None:
    """Replace process-local OpenCascade metadata without touching B-Rep entities."""

    text = path.read_text(encoding="utf-8")
    normalized, replacements = re.subn(
        r"(FILE_NAME\('[^']*',)'[^']*'",
        r"\1'1970-01-01T00:00:00'",
        text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"CAD kernel STEP header has no normalizable FILE_NAME: {path}")
    normalized = re.sub(
        r"(Open CASCADE STEP translator [^']*?) \d+'",
        r"\1 1'",
        normalized,
    )
    path.write_text(normalized, encoding="utf-8")


def validate_and_export_cadquery(result: Any, output_path: Path) -> CadKernelValidation:
    """Require one positive-volume valid solid before exporting STEP."""

    solids = _solids(result)
    if len(solids) != 1:
        raise ValueError(f"CAD result must contain exactly one solid, found {len(solids)}")
    solid = solids[0]
    valid = bool(solid.isValid())
    volume = float(solid.Volume())
    validation = CadKernelValidation(
        solids=1,
        faces=len(solid.Faces()),
        edges=len(solid.Edges()),
        volume=volume,
        valid=valid,
    )
    if not valid:
        raise ValueError("CAD kernel reports an invalid B-Rep solid")
    if volume <= 0.0:
        raise ValueError("CAD B-Rep solid has non-positive volume")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(result, str(output_path))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"CAD kernel did not write a STEP file: {output_path}")
    normalize_step_metadata(output_path)
    return validation
