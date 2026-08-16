from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np
import pytest
import trimesh

from da3_cad.evaluation.chamfer import (
    brute_force_directional_squared_means,
    chamfer_metrics,
    directional_squared_means,
)
from da3_cad.evaluation.evaluator import Evaluator
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    normalize_evaluation_mesh,
    normalize_prediction_mesh,
    tessellate_step,
    validate_mesh,
    verify_centered_evaluation_frame,
    verify_official_test_mesh_frame,
)
from da3_cad.evaluation.mesh_iou import mesh_iou
from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted


def _box(
    extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_kdtree_chamfer_matches_brute_force_and_analytic_translation() -> None:
    rng = np.random.default_rng(9)
    prediction = rng.normal(size=(13, 3))
    ground_truth = rng.normal(size=(17, 3))
    np.testing.assert_allclose(
        directional_squared_means(prediction, ground_truth),
        brute_force_directional_squared_means(prediction, ground_truth),
        rtol=0.0,
        atol=1e-14,
    )
    identity = chamfer_metrics(
        prediction,
        prediction.copy(),
        prediction_seed=1,
        ground_truth_seed=2,
    )
    assert identity.prediction_to_ground_truth == pytest.approx(0.0)
    assert identity.ground_truth_to_prediction == pytest.approx(0.0)
    assert identity.scaled_bidirectional == pytest.approx(0.0)

    origin = np.asarray([[0.0, 0.0, 0.0]])
    translated = np.asarray([[0.1, 0.2, 0.3]])
    result = chamfer_metrics(
        translated,
        origin,
        prediction_seed=3,
        ground_truth_seed=4,
    )
    assert result.prediction_to_ground_truth == pytest.approx(0.14)
    assert result.ground_truth_to_prediction == pytest.approx(0.14)
    assert result.scaled_bidirectional == pytest.approx(280.0)


def test_surface_sampler_is_seeded_and_area_weighted() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [10.0, 2.0, 0.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
        process=False,
    )
    first = sample_surface_area_weighted(mesh, 20_000, seed=123)
    repeated = sample_surface_area_weighted(mesh, 20_000, seed=123)
    np.testing.assert_array_equal(first.points, repeated.points)
    np.testing.assert_array_equal(first.face_indices, repeated.face_indices)
    assert np.mean(first.face_indices == 1) == pytest.approx(0.8, abs=0.015)


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (_box(), 100.0),
        (_box(center=(2.0, 0.5, 0.5)), 0.0),
        (_box(center=(1.0, 0.5, 0.5)), 100.0 / 3.0),
        (_box(extents=(0.5, 0.5, 0.5)), 12.5),
    ],
)
def test_complete_mesh_iou_synthetic_cases(
    prediction: trimesh.Trimesh,
    expected: float,
) -> None:
    result = mesh_iou(_box(), prediction)
    assert result.percent == pytest.approx(expected, abs=1e-8)
    assert result.engine == "manifold"
    assert result.engine_version == "3.5.2"


def test_iou_volumes_cross_check_cadquery_occ() -> None:
    unit_occ = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    half_occ = cq.Workplane("XY").box(0.5, 0.5, 0.5).val()
    assert isinstance(unit_occ, cq.Shape)
    assert isinstance(half_occ, cq.Shape)
    result = mesh_iou(_box(), _box(extents=(0.5, 0.5, 0.5)))
    assert result.union_volume == pytest.approx(unit_occ.Volume(), abs=1e-10)
    assert result.intersection_volume == pytest.approx(half_occ.Volume(), abs=1e-10)


def test_prediction_normalization_is_isotropic_without_alignment() -> None:
    prediction = _box(extents=(200.0, 84.0, 25.0), center=(20.0, -3.0, 7.0))
    normalized = normalize_prediction_mesh(prediction)
    np.testing.assert_allclose(
        normalized.bounds,
        [[-0.5, -0.21, -0.0625], [0.5, 0.21, 0.0625]],
        atol=1e-12,
    )
    validation = validate_mesh(normalized)
    assert validation.valid
    assert validation.watertight


def test_step_tessellation_uses_fixed_tolerances_and_matches_occ_volume(
    tmp_path: Path,
) -> None:
    result = cq.Workplane("XY").box(2.0, 1.0, 0.5)
    step_path = tmp_path / "box.step"
    cq.exporters.export(result, str(step_path))
    config = TessellationConfig(linear_tolerance=0.001, angular_tolerance=0.1)
    mesh = tessellate_step(step_path, config)
    validation = validate_mesh(mesh)

    assert validation.valid
    assert validation.volume == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(mesh.extents, [2.0, 1.0, 0.5], atol=1e-9)
    assert config.as_dict() == {
        "source": "CadQuery/OpenCascade Shape.tessellate",
        "linear_tolerance": 0.001,
        "angular_tolerance": 0.1,
        "vertex_welding": {
            "method": "trimesh.merge_vertices",
            "digits_vertex": 12,
            "merge_texture_and_normals": True,
        },
        "analytic_seam_cleanup": (
            "remove exact zero-area triangles emitted by OCC tessellation; "
            "no hole filling, remeshing, or geometry repair"
        ),
    }


def test_step_tessellation_removes_only_occ_axis_seam_triangles(tmp_path: Path) -> None:
    revolved = (
        cq.Workplane("XZ")
        .moveTo(0.0, -1.0)
        .lineTo(1.0, -1.0)
        .lineTo(1.0, 1.0)
        .lineTo(0.0, 1.0)
        .close()
        .revolve(360.0, (0.0, 0.0), (0.0, 1.0))
    )
    step_path = tmp_path / "axis-touching-revolve.step"
    cq.exporters.export(revolved, str(step_path))

    mesh = tessellate_step(step_path)
    validation = validate_mesh(mesh)

    assert validation.valid
    assert validation.watertight
    assert np.all(np.asarray(mesh.area_faces) > 0.0)


def test_file_mesh_loading_welds_vertices_but_does_not_repair(tmp_path: Path) -> None:
    path = tmp_path / "unit.stl"
    _box().export(path)
    raw = trimesh.load(path, process=False)
    assert isinstance(raw, trimesh.Trimesh)
    assert not validate_mesh(raw).valid

    loaded = load_mesh(path, TessellationConfig())
    assert validate_mesh(loaded).valid
    assert loaded.is_watertight


def test_storage_and_evaluation_frames_are_distinct_and_verified() -> None:
    stored = _box()
    verify_official_test_mesh_frame(stored)

    centered = normalize_evaluation_mesh(stored)
    verify_centered_evaluation_frame(centered)
    centered.vertices = np.asarray(centered.vertices, dtype=np.float64) * 0.9999
    verify_centered_evaluation_frame(centered)
    with pytest.raises(ValueError, match="largest bbox extent"):
        verify_centered_evaluation_frame(centered, tolerance=1e-5)

    shifted = centered.copy()
    shifted.apply_translation([0.000525, 0.0, 0.0])
    with pytest.raises(ValueError, match="origin"):
        verify_centered_evaluation_frame(shifted)


def test_evaluator_normalizes_different_input_frames_identically() -> None:
    ground_truth = _box(
        extents=(1.0, 0.5, 0.25),
        center=(0.0, 0.0, 0.0),
    )
    prediction = _box(
        extents=(200.0, 100.0, 50.0),
        center=(20.0, -3.0, 7.0),
    )
    gt_normalized = normalize_evaluation_mesh(ground_truth)
    pred_normalized = normalize_evaluation_mesh(prediction)
    np.testing.assert_allclose(gt_normalized.bounds, pred_normalized.bounds, atol=1e-12)

    result = Evaluator().evaluate(
        "different-native-coordinate-frames",
        prediction,
        ground_truth,
    )
    assert result.valid_prediction
    assert result.iou is not None
    assert result.iou.percent == pytest.approx(100.0, abs=1e-8)
    assert result.chamfer is not None
    assert result.chamfer.scaled_bidirectional < 1.0
