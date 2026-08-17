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
    fit_planar_profile_add,
    fit_planar_profile_add_candidates,
    fit_planar_profile_cut,
    planar_profile_add,
    planar_profile_cut,
)


def _mesh(result: cq.Workplane) -> trimesh.Trimesh:
    vertices, faces = result.val().tessellate(0.01, 0.1)
    return trimesh.Trimesh(
        vertices=np.asarray([vertex.toTuple() for vertex in vertices], dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )


def _sample(result: cq.Workplane, count: int = 24000) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(_mesh(result), count, seed=20260816)
    return np.asarray(points, dtype=np.float64)


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


def test_fit_planar_profile_add_recovers_nonconvex_constant_section() -> None:
    base = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    target = planar_profile_add(
        base,
        1,
        ((-0.8, 0.8), (-0.8, 2.2), (-0.2, 2.2), (-0.2, 1.4), (0.8, 1.4), (0.8, 0.8)),
        -1.0,
        1.0,
    )
    current = trimesh.creation.box(extents=(2.0, 2.0, 2.0))

    feature = fit_planar_profile_add(
        _sample(target),
        current,
        axis=1,
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert feature is not None
    assert feature.axis == 1
    assert feature.axial_coverage_fraction >= 0.75
    assert feature.profile_occupancy_iou >= 0.78
    assert feature.convexity_ratio < 0.95
    result = planar_profile_add(base, feature.axis, feature.profile, feature.lower, feature.upper)
    validation = validate_cadquery(result)
    assert validation.solids == 1
    assert result.val().Volume() > base.val().Volume()


def test_fit_planar_profile_cut_recovers_nonconvex_through_slot() -> None:
    base = cq.Workplane("XY").box(4.0, 4.0, 2.0)
    profile = (
        (-1.2, -0.3),
        (-0.3, -0.3),
        (-0.3, -1.4),
        (0.3, -1.4),
        (0.3, -0.3),
        (1.2, -0.3),
        (1.2, 0.3),
        (-1.2, 0.3),
    )
    target = planar_profile_cut(base, 2, profile, -1.1, 1.1)
    current = trimesh.creation.box(extents=(4.0, 4.0, 2.0))

    feature = fit_planar_profile_cut(
        _sample(target),
        current,
        axis=2,
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert feature is not None
    assert feature.axis == 2
    assert len(feature.profile) >= 6
    assert feature.convexity_ratio < 0.90
    result = planar_profile_cut(base, feature.axis, feature.profile, feature.lower, feature.upper)
    validation = validate_cadquery(result)
    assert validation.solids == 1
    assert result.val().Volume() < base.val().Volume()


def test_fit_planar_profile_add_rejects_tapered_residual() -> None:
    current = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    rows: list[np.ndarray] = []
    edge_samples = np.linspace(0.0, 1.0, 24)
    for axial in np.linspace(-0.9, 0.9, 40):
        width = 0.2 + (axial + 0.9) / 1.8
        x_lower, x_upper = 1.1, 1.1 + width
        z_lower, z_upper = -0.5, 0.5
        for value in edge_samples:
            rows.extend(
                (
                    np.asarray((x_lower + value * width, axial, z_lower)),
                    np.asarray((x_lower + value * width, axial, z_upper)),
                    np.asarray((x_lower, axial, z_lower + value)),
                    np.asarray((x_upper, axial, z_lower + value)),
                )
            )
    points = np.asarray(rows, dtype=np.float64)

    feature = fit_planar_profile_add(
        points,
        current,
        axis=1,
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert feature is None


def test_planar_profile_candidates_reject_nonconstant_frustum() -> None:
    root = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    frustum = (
        cq.Workplane("XY", origin=(0.0, 0.0, 0.98))
        .rect(0.7, 0.7)
        .workplane(offset=1.8)
        .rect(2.0, 1.5)
        .loft(combine=True)
    )
    target = root.union(frustum)

    candidates = fit_planar_profile_add_candidates(
        _sample(target),
        _mesh(root),
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )

    assert candidates == ()


def test_planar_profile_operations_support_all_cad_axes() -> None:
    for axis in (0, 1, 2):
        base = cq.Workplane("XY").box(2.0, 2.0, 2.0)
        added = planar_profile_add(
            base,
            axis,
            ((0.8, -0.5), (1.4, -0.5), (1.4, 0.5), (0.8, 0.5)),
            -0.8,
            0.8,
        )
        cut = planar_profile_cut(
            added,
            axis,
            ((-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)),
            -1.1,
            1.1,
        )
        assert validate_cadquery(added).solids == 1
        assert validate_cadquery(cut).solids == 1
        assert cut.val().Volume() < added.val().Volume()
