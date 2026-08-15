from __future__ import annotations

import hashlib
from pathlib import Path

import cadquery as cq
import pytest

from da3_cad.cad_validation import validate_and_export_cadquery
from da3_cad.evaluation.source_view_verifier import (
    SourceViewScore,
    decide_source_view_score,
)


def test_kernel_validation_exports_one_valid_solid(tmp_path: Path) -> None:
    output = tmp_path / "box.step"
    result = validate_and_export_cadquery(cq.Workplane("XY").box(2, 3, 4), output)

    assert output.is_file()
    assert result.valid
    assert result.solids == 1
    assert result.faces == 6
    assert result.volume == pytest.approx(24.0)


def test_kernel_export_is_byte_deterministic_and_reopens(tmp_path: Path) -> None:
    solid = cq.Workplane("XY").box(2, 3, 4)
    first = tmp_path / "first.step"
    second = tmp_path / "second.step"

    validate_and_export_cadquery(solid, first)
    validate_and_export_cadquery(solid, second)

    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    assert "1970-01-01T00:00:00" in first.read_text(encoding="utf-8")
    reopened = cq.importers.importStep(str(first)).val()
    assert isinstance(reopened, cq.Shape)
    assert reopened.isValid()


def test_kernel_validation_rejects_multiple_solids(tmp_path: Path) -> None:
    separated = cq.Workplane("XY").box(1, 1, 1).transformed(offset=(3, 0, 0)).box(1, 1, 1)

    with pytest.raises(ValueError, match="exactly one solid"):
        validate_and_export_cadquery(separated, tmp_path / "invalid.step")


def test_source_view_decision_requires_mask_and_depth_support() -> None:
    accepted = SourceViewScore(0.9, 0.88, 0.91, 0.95, 0.01, ())
    wrong_shape = SourceViewScore(0.9, 0.86, 0.93, 0.95, 0.01, ())

    assert decide_source_view_score(accepted).accepted
    decision = decide_source_view_score(wrong_shape)
    assert not decision.accepted
    assert "silhouette IoU" in decision.reasons[0]
