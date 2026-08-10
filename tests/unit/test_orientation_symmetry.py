from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.orientation import orient_canonical_frame
from da3_cad.geometry.symmetry import (
    SymmetryPlane,
    complete_across_symmetry,
    detect_symmetry_plane,
)


def _rotated_plate() -> np.ndarray:
    x_values, y_values = np.meshgrid(
        np.linspace(-2.0, 2.0, 40),
        np.linspace(-1.0, 1.0, 30),
        indexing="ij",
    )
    points = np.column_stack((x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)))
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (points @ rotation.T + [4.0, -3.0, 9.0]).astype(np.float32)


def test_planar_orientation_uses_explicit_safe_branch() -> None:
    points = _rotated_plate()
    symmetry = detect_symmetry_plane(points, tolerance_fraction=0.04, seed=17)
    first = orient_canonical_frame(
        points,
        symmetry,
        seed=17,
        planar_extent_ratio_threshold=0.20,
        plane_distance_fraction=0.02,
        plane_ransac_iterations=128,
        eigenvalue_tie_tolerance=0.05,
    )
    second = orient_canonical_frame(
        points,
        symmetry,
        seed=17,
        planar_extent_ratio_threshold=0.20,
        plane_distance_fraction=0.02,
        plane_ransac_iterations=128,
        eigenvalue_tie_tolerance=0.05,
    )
    assert first.method == "planar-dominance-symmetry"
    assert first.determinant == pytest.approx(1.0, abs=1e-10)
    assert first.details["provenance_branch_explicit"] is True
    assert np.array_equal(first.points, second.points)
    np.testing.assert_allclose(np.sort(np.ptp(first.points, axis=0)), [0.0, 2.0, 4.0], atol=1e-6)


def test_volumetric_orientation_is_right_handed() -> None:
    x, y, z = np.meshgrid(
        np.linspace(-2.0, 2.0, 14),
        np.linspace(-1.0, 1.0, 10),
        np.linspace(-0.6, 0.6, 8),
        indexing="ij",
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    symmetry = detect_symmetry_plane(points, tolerance_fraction=0.04, seed=3)
    result = orient_canonical_frame(
        points,
        symmetry,
        seed=3,
        planar_extent_ratio_threshold=0.20,
        plane_distance_fraction=0.02,
        plane_ransac_iterations=64,
        eigenvalue_tie_tolerance=0.05,
    )
    assert result.method == "pca-symmetry"
    assert result.determinant == pytest.approx(1.0, abs=1e-10)


def test_rank_one_cloud_is_rejected() -> None:
    line = np.column_stack((np.linspace(-1.0, 1.0, 300), np.zeros(300), np.zeros(300))).astype(
        np.float32
    )
    symmetry = detect_symmetry_plane(line, tolerance_fraction=0.04, seed=1)
    with pytest.raises(ValueError, match="rank-one"):
        orient_canonical_frame(
            line,
            symmetry,
            seed=1,
            planar_extent_ratio_threshold=0.20,
            plane_distance_fraction=0.02,
            plane_ransac_iterations=32,
            eigenvalue_tie_tolerance=0.05,
        )


def test_symmetry_completion_returns_label_sources_only() -> None:
    points = np.asarray(
        [[0.5, -1.0, 0.0], [0.5, 0.0, 0.0], [0.5, 1.0, 0.0]],
        dtype=np.float32,
    )
    plane = SymmetryPlane(
        point=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        score=0.0,
        accepted=True,
        candidate_source="unit-test",
        tolerance_fraction=0.04,
        evaluated_points=3,
    )
    completion = complete_across_symmetry(points, plane, duplicate_radius_fraction=0.01)
    np.testing.assert_allclose(completion.added_points[:, 0], np.full(3, -0.5, dtype=np.float32))
    assert completion.source_indices.tolist() == [0, 1, 2]
    assert completion.report["inferred_geometry"] is True
