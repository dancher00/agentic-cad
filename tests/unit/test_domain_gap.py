from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from da3_cad.benchmark.domain_gap import (
    AxisTransform,
    best_proper_axis_alignment,
    density_metrics,
    measure_domain_gap,
    proper_axis_transforms,
    surface_relation_metrics,
)


def test_proper_axis_transforms_are_the_24_unique_right_handed_rotations() -> None:
    transforms = proper_axis_transforms()
    identities = {(transform.permutation, transform.signs) for transform in transforms}

    assert len(transforms) == 24
    assert len(identities) == 24
    assert all(transform.determinant == 1 for transform in transforms)
    assert ((0, 1, 2), (1, 1, 1)) in identities


def test_gt_axis_oracle_recovers_a_known_proper_axis_rotation() -> None:
    rng = np.random.default_rng(20260810)
    gt = rng.normal(size=(300, 3)) * np.asarray([0.83, 0.41, 0.17])
    rotated = AxisTransform((2, 0, 1), (-1, 1, -1), 1).apply(gt)

    aligned, transform, score = best_proper_axis_alignment(rotated, gt)
    nearest = cKDTree(gt).query(aligned, k=1, workers=1)[0]

    assert transform.determinant == 1
    assert score < 1.0e-20
    assert float(nearest.max()) < 1.0e-12


def test_surface_relation_separates_surface_interior_and_exterior_points() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    surface, _ = trimesh.sample.sample_surface(mesh, 2048, seed=7)
    points = np.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    metrics = surface_relation_metrics(points, surface, mesh)
    signed = metrics["signed_side"]

    assert signed["near_surface_fraction_within_0.02"] == pytest.approx(1.0 / 3.0)
    assert signed["interior_fraction_beyond_0.02"] == pytest.approx(1.0 / 3.0)
    assert signed["exterior_fraction_beyond_0.02"] == pytest.approx(1.0 / 3.0)
    assert metrics["normal_residual_absolute"]["mean"] == pytest.approx(0.2 / 3.0)
    assert metrics["tangential_residual"]["p90"] < 1.0e-12


def test_identical_gt_and_da3_decoder_clouds_have_identical_baselines() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    surface, _ = trimesh.sample.sample_surface(mesh, 8192, seed=11)
    decoder = np.asarray(surface[:256], dtype=np.float32)

    result = measure_domain_gap(decoder, decoder, surface, mesh)

    assert result["density"]["gt_upstream_fps"] == result["density"]["da3_canonical_fps"]
    assert result["gt_upstream_frame"] == result["emitted_frame"]
    assert result["orientation_diagnostic"]["oracle_sample_chamfer_x1000"] == (
        pytest.approx(result["orientation_diagnostic"]["emitted_sample_chamfer_x1000"])
    )


def test_domain_gap_enforces_decoder_shape_and_density_finiteness() -> None:
    mesh = trimesh.creation.box()
    surface, _ = trimesh.sample.sample_surface(mesh, 512, seed=13)
    decoder = np.asarray(surface[:256], dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        measure_domain_gap(decoder[:255], decoder, surface, mesh)

    broken = decoder.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        density_metrics(broken)
