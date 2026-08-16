from __future__ import annotations

import hashlib
from pathlib import Path

import cadquery as cq
import pytest

from da3_cad.cad_validation import validate_and_export_cadquery, validate_cadquery
from da3_cad.evaluation.source_view_verifier import (
    SourceViewScore,
    decide_source_view_progress,
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

    with pytest.raises(ValueError, match="exactly one solid"):
        validate_cadquery(separated)


def test_source_view_decision_requires_mask_and_depth_support() -> None:
    accepted = SourceViewScore(0.9, 0.88, 0.91, 0.95, 0.01, ())
    wrong_shape = SourceViewScore(0.9, 0.86, 0.93, 0.95, 0.01, ())

    assert decide_source_view_score(accepted).accepted
    decision = decide_source_view_score(wrong_shape)
    assert not decision.accepted
    assert "silhouette IoU" in decision.reasons[0]


def test_source_view_decision_rejects_unexplained_internal_edges() -> None:
    score = SourceViewScore(
        0.92,
        0.90,
        0.95,
        0.96,
        0.01,
        (),
        appearance_edge_precision=0.28,
        appearance_edge_recall=0.04,
        appearance_edge_pixels=1000,
        rendered_geometry_edge_pixels=200,
    )

    decision = decide_source_view_score(score)

    assert not decision.accepted
    assert any("appearance-edge precision" in reason for reason in decision.reasons)
    assert any("appearance-edge recall" in reason for reason in decision.reasons)


def test_rgb_supported_topology_step_may_trade_bounded_outer_score() -> None:
    previous = SourceViewScore(
        0.882,
        0.860,
        0.923,
        0.924,
        0.002,
        (),
        appearance_edge_precision=0.69,
        appearance_edge_recall=0.23,
        appearance_edge_pixels=1009,
        rendered_geometry_edge_pixels=405,
    )
    cavity = SourceViewScore(
        0.848,
        0.824,
        0.892,
        0.894,
        0.002,
        (),
        appearance_edge_precision=0.66,
        appearance_edge_recall=0.39,
        appearance_edge_pixels=1009,
        rendered_geometry_edge_pixels=839,
    )

    accepted, reason = decide_source_view_progress(cavity, previous)

    assert accepted
    assert reason == "bounded-appearance-topology-improvement"


def test_unexplained_cut_cannot_use_topology_escape_hatch() -> None:
    previous = SourceViewScore(
        0.89,
        0.88,
        0.91,
        0.94,
        0.01,
        (),
        appearance_edge_precision=0.60,
        appearance_edge_recall=0.10,
        appearance_edge_pixels=800,
        rendered_geometry_edge_pixels=200,
    )
    invented = SourceViewScore(
        0.88,
        0.87,
        0.90,
        0.93,
        0.01,
        (),
        appearance_edge_precision=0.20,
        appearance_edge_recall=0.30,
        appearance_edge_pixels=800,
        rendered_geometry_edge_pixels=600,
    )

    accepted, reason = decide_source_view_progress(invented, previous)

    assert not accepted
    assert reason == "no-supported-geometric-or-topology-progress"
