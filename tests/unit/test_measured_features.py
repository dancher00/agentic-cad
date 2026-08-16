from __future__ import annotations

import cadquery as cq
import numpy as np
import trimesh

from da3_cad.cad_validation import validate_cadquery
from da3_cad.integrations.measured_features import (
    axial_revolved_add,
    axial_revolved_cut,
    fit_axial_revolved_add,
    fit_axial_revolved_cut,
)


def _cavity_surface(*, full_rotation: bool = True) -> np.ndarray:
    angles = np.linspace(
        0.0,
        2.0 * np.pi if full_rotation else 0.45 * np.pi,
        32,
        endpoint=False,
    )
    axial = np.linspace(1.0, 1.9, 32)
    points = []
    for position in axial:
        radius = 0.12 + (position - 1.0) * 1.1
        points.extend(
            (radius * np.cos(angle), radius * np.sin(angle), position) for angle in angles
        )
    return np.asarray(points, dtype=np.float64)


def _addition_surface(*, full_rotation: bool = True) -> np.ndarray:
    angles = np.linspace(
        0.0,
        2.0 * np.pi if full_rotation else 0.45 * np.pi,
        32,
        endpoint=False,
    )
    axial = np.linspace(1.05, 1.8, 32)
    points = []
    for position in axial:
        radius = 0.55 + 0.10 * (position - 1.05)
        points.extend(
            (1.0 + radius * np.cos(angle), 2.0 + radius * np.sin(angle), position)
            for angle in angles
        )
    return np.asarray(points, dtype=np.float64)


def test_fit_axial_revolved_add_finds_supported_end_extension() -> None:
    current = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=64)
    current.apply_translation((1.0, 2.0, 0.0))

    feature = fit_axial_revolved_add(
        _addition_surface(),
        current,
        axis=2,
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert feature is not None
    assert feature.axis == 2
    assert feature.side == 1
    assert feature.angular_coverage_fraction == 1.0
    assert feature.axial_span_fraction > 0.25

    base = cq.Workplane("XY", origin=(1.0, 2.0, 0.0)).circle(1.0).extrude(1.0, both=True)
    result = axial_revolved_add(base, feature.center, feature.axis, feature.profile)
    validation = validate_cadquery(result)
    assert validation.solids == 1
    assert result.val().Volume() > base.val().Volume()


def test_fit_axial_revolved_add_rejects_partial_arc() -> None:
    current = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=64)
    current.apply_translation((1.0, 2.0, 0.0))

    feature = fit_axial_revolved_add(
        _addition_surface(full_rotation=False),
        current,
        axis=2,
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert feature is None


def test_fit_axial_revolved_cut_finds_supported_end_cavity() -> None:
    current = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=64)

    feature = fit_axial_revolved_cut(
        _cavity_surface(),
        current,
        axis=2,
        minimum_surface_distance=0.05,
        minimum_support_points=64,
    )

    assert feature is not None
    assert feature.axis == 2
    assert feature.side == 1
    assert feature.angular_coverage_fraction == 1.0
    assert feature.radial_growth_fraction > 0.25
    assert len(feature.profile) >= 2


def test_fit_axial_revolved_cut_rejects_partial_arc() -> None:
    current = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=64)

    feature = fit_axial_revolved_cut(
        _cavity_surface(full_rotation=False),
        current,
        axis=2,
        minimum_surface_distance=0.05,
        minimum_support_points=64,
    )

    assert feature is None


def test_axial_revolved_cut_is_one_valid_translated_solid() -> None:
    base = cq.Workplane("XY", origin=(1.0, 2.0, 0.0)).circle(2.0).extrude(4.0, both=True)

    result = axial_revolved_cut(
        base,
        (1.0, 2.0, 0.0),
        2,
        ((0.2, 0.5), (0.8, 1.5), (1.0, 1.9)),
        2.1,
    )

    validation = validate_cadquery(result)
    assert validation.solids == 1
    assert result.val().Volume() < base.val().Volume()
