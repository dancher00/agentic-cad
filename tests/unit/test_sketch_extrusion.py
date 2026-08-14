from __future__ import annotations

import cadquery as cq
import cv2
import numpy as np
import pytest

from da3_cad.backends.sketch_extrusion import (
    SketchExtrusionCadBackend,
    UnsupportedProfileError,
    _candidate_with_pose_delta,
    _capture_prism_vertices,
    _mask_iou,
    _rectilinear_raster,
    _render_prism_silhouette,
    _silhouette_length_refinement,
    _silhouette_pose_refinement,
)
from da3_cad.cad.program import extract_parameters
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import CanonicalizerConfig, SandboxConfig, SketchExtrusionConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.geometry.scale import KnownDimension
from da3_cad.models import DepthPrediction


def _plate_surface(*, with_hole: bool) -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 101),
        np.linspace(-0.7, 0.7, 71),
        indexing="ij",
    )
    keep = x * x + y * y >= 0.2**2 if with_hole else np.ones_like(x, dtype=np.bool_)
    parts = [np.column_stack((x[keep], y[keep], np.full(int(keep.sum()), z))) for z in (-0.1, 0.1)]
    z_values = np.linspace(-0.1, 0.1, 11)
    for x_value in (-1.0, 1.0):
        yy, zz = np.meshgrid(np.linspace(-0.7, 0.7, 71), z_values, indexing="ij")
        parts.append(np.column_stack((np.full(yy.size, x_value), yy.ravel(), zz.ravel())))
    for y_value in (-0.7, 0.7):
        xx, zz = np.meshgrid(np.linspace(-1.0, 1.0, 101), z_values, indexing="ij")
        parts.append(np.column_stack((xx.ravel(), np.full(xx.size, y_value), zz.ravel())))
    if with_hole:
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        aa, zz = np.meshgrid(angles, z_values, indexing="ij")
        parts.append(
            np.column_stack(
                (
                    0.2 * np.cos(aa).ravel(),
                    0.2 * np.sin(aa).ravel(),
                    zz.ravel(),
                )
            )
        )
    return np.concatenate(parts).astype(np.float32)


def _l_profile_surface() -> np.ndarray:
    yz = []
    for y in np.linspace(-0.7, 0.7, 71):
        for z in np.linspace(-0.6, -0.35, 14):
            yz.append((y, z))
    for y in np.linspace(-0.7, -0.45, 14):
        for z in np.linspace(-0.6, 0.6, 61):
            yz.append((y, z))
    profile = np.unique(np.asarray(yz, dtype=np.float32), axis=0)
    parts = []
    for x in (-1.0, 1.0):
        parts.append(np.column_stack((np.full(len(profile), x), profile[:, 0], profile[:, 1])))
    for x in np.linspace(-1.0, 1.0, 81):
        boundary = profile[
            (np.isclose(profile[:, 0], -0.7))
            | (np.isclose(profile[:, 0], 0.7))
            | (np.isclose(profile[:, 1], -0.6))
            | (np.isclose(profile[:, 1], -0.35))
            | (np.isclose(profile[:, 0], -0.45))
            | (np.isclose(profile[:, 1], 0.6))
        ]
        parts.append(np.column_stack((np.full(len(boundary), x), boundary[:, 0], boundary[:, 1])))
    return np.concatenate(parts).astype(np.float32)


def _canonical(points: np.ndarray, *, metric: bool = False):
    doubled = np.concatenate((points, points))
    count = len(points)
    views = np.repeat(np.asarray([0, 1], dtype=np.int32), count)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic-surface",
        require_confidence=True,
        views=(
            ViewFusionStats(0, count, count, count, count, count),
            ViewFusionStats(1, count, count, count, count, count),
        ),
    )
    cloud = FusedPointCloud(
        points=doubled,
        colors=np.zeros((len(doubled), 3), dtype=np.uint8),
        confidences=np.ones(len(doubled), dtype=np.float32),
        view_indices=views,
        pixel_xy=np.zeros((len(doubled), 2), dtype=np.int32),
        report=report,
        scale=(
            ScaleChannel(
                status="known",
                units="calibrated-world-unit",
                world_units_to_mm=10.0,
                source="synthetic-calibrated-cameras",
            )
            if metric
            else ScaleChannel()
        ),
    )
    config = CanonicalizerConfig(
        outlier_enabled=False,
        consistency_radius_fraction=0.001,
        plane_ransac_iterations=64,
    )
    return PointCloudCanonicalizer(config).run(cloud, seed=3)


def _prediction_and_masks(*, hole: bool) -> tuple[DepthPrediction, np.ndarray]:
    view_count, height, width = 3, 128, 128
    masks = np.zeros((view_count, height, width), dtype=np.bool_)
    yy, xx = np.ogrid[:height, :width]
    for mask in masks:
        mask[29:99, 14:114] = True
        if hole:
            mask[(xx - 64) ** 2 + (yy - 64) ** 2 <= 10**2] = False
    intrinsics = np.repeat(
        np.asarray([[[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]]]),
        view_count,
        axis=0,
    )
    extrinsics = np.repeat(
        np.asarray([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]]),
        view_count,
        axis=0,
    )
    depth = np.full((view_count, height, width), 2.0, dtype=np.float32)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(
            np.zeros((height, width, 3), dtype=np.uint8) for _ in range(view_count)
        ),
        backend="synthetic-regression",
    )
    return prediction, masks


def _look_at(eye: np.ndarray) -> np.ndarray:
    forward = -np.asarray(eye, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(forward @ up)) > 0.9:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = -(rotation @ eye)
    return result.astype(np.float32)


def test_general_polyline_profile_emits_valid_plate(tmp_path) -> None:
    canonical = _canonical(_plate_surface(with_hole=False))
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    program = backend.generate(canonical, seed=3)

    assert backend.last_report is not None
    assert backend.last_report.program_family == "sketch-extrusion"
    assert backend.last_report.profile_evidence_points >= backend.last_report.input_points
    assert backend.last_report.outer_loop.kind == "polyline"
    assert len(backend.last_report.outer_loop.points) == 4
    assert program.program_family == "sketch-extrusion"
    assert not any(name.startswith("aperture_") for name in program.parameters)
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error
    shape = cq.importers.importStep(str(tmp_path / "model.step")).val()
    assert {face.geomType() for face in shape.Faces()} == {"PLANE"}
    assert {edge.geomType() for edge in shape.Edges()} == {"LINE"}


def test_one_grammar_recovers_an_l_profile(tmp_path) -> None:
    canonical = _canonical(_l_profile_surface())
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    program = backend.generate(canonical, seed=3)

    assert backend.last_report is not None
    assert backend.last_report.selected_axis == 0
    assert backend.last_report.outer_loop.kind == "polyline"
    assert len(backend.last_report.outer_loop.points) >= 6
    selected = next(
        candidate
        for candidate in backend.last_report.axis_candidates
        if candidate.axis == backend.last_report.selected_axis
    )
    assert selected.profile_occupancy_iou >= 0.97
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error


def test_aperture_uses_repeated_masks_and_prefers_3d_profile_measurement() -> None:
    cloud_with_hole = _canonical(_plate_surface(with_hole=True))
    solid_cloud = _canonical(_plate_surface(with_hole=False))
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    solid_prediction, solid_masks = _prediction_and_masks(hole=False)
    solid = backend.generate(
        cloud_with_hole,
        seed=3,
        prediction=solid_prediction,
        masks=solid_masks,
    )
    assert not any(name.startswith("aperture_") for name in solid.parameters)
    assert backend.last_report is not None
    assert backend.last_report.apertures == ()

    hole_prediction, hole_masks = _prediction_and_masks(hole=True)
    mask_only = backend.generate(
        solid_cloud,
        seed=3,
        prediction=hole_prediction,
        masks=hole_masks,
    )
    assert mask_only.parameters["aperture_000_radius"] == pytest.approx(0.2, abs=0.05)
    assert backend.last_report is not None
    assert backend.last_report.apertures[0].measurement_source == "multi-view-mask"

    perforated = backend.generate(
        cloud_with_hole,
        seed=3,
        prediction=hole_prediction,
        masks=hole_masks,
    )
    assert perforated.parameters["aperture_000_radius"] == pytest.approx(0.2, abs=0.04)
    assert backend.last_report is not None
    aperture = backend.last_report.apertures[0]
    assert aperture.measurement_source == "raw-3d+mask"
    assert aperture.supporting_views == (0, 1, 2)
    assert aperture.radius == pytest.approx(0.2, abs=0.04)
    assert aperture.mask_radius != pytest.approx(aperture.radius)
    selected = next(
        candidate
        for candidate in backend.last_report.axis_candidates
        if candidate.axis == backend.last_report.selected_axis
    )
    assert len(selected.profile_aperture_candidates) == 1


def test_metric_scale_and_known_dimension_apply_to_the_whole_sketch() -> None:
    metric = _canonical(_plate_surface(with_hole=False), metric=True)
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    metric_program = backend.generate(metric, seed=3)
    assert backend.last_report is not None
    factor = backend.last_report.scale.millimeters_per_unit
    assert factor is not None
    for name, value in backend.last_report.parameters_normalized.items():
        assert metric_program.parameters[name] == pytest.approx(value * factor)

    unresolved = _canonical(_plate_surface(with_hole=False))
    scaled = backend.generate(
        unresolved,
        seed=3,
        known_dimension=KnownDimension.parse("extrusion_length=6mm"),
    )
    parameters = extract_parameters(scaled.source)
    assert parameters["extrusion_length"] == pytest.approx(6.0)


def test_side_silhouettes_refine_extrusion_length_without_ground_truth() -> None:
    canonical = _canonical(_plate_surface(with_hole=False))
    initial_backend = SketchExtrusionCadBackend(
        SketchExtrusionConfig(silhouette_length_refinement_enabled=False)
    )
    initial_backend.generate(canonical, seed=3)
    assert initial_backend.last_report is not None
    candidate = next(
        item
        for item in initial_backend.last_report.axis_candidates
        if item.axis == initial_backend.last_report.selected_axis
    )
    assert canonical.orientation is not None
    base = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    local = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    axis_world = (base @ local)[:, candidate.axis]
    transverse = np.eye(3)[int(np.argmin(np.abs(axis_world)))]
    first_side = np.cross(axis_world, transverse)
    first_side /= np.linalg.norm(first_side)
    second_side = np.cross(axis_world, first_side)
    second_side /= np.linalg.norm(second_side)
    directions = (
        first_side,
        -first_side,
        second_side,
        -second_side,
        (first_side + second_side) / np.sqrt(2.0),
        (first_side - second_side) / np.sqrt(2.0),
    )
    extrinsics = np.stack([_look_at(4.0 * direction) for direction in directions])
    intrinsics = np.repeat(
        np.asarray([[[90.0, 0.0, 64.0], [0.0, 90.0, 64.0], [0.0, 0.0, 1.0]]]),
        len(directions),
        axis=0,
    ).astype(np.float32)
    target_ends = _capture_prism_vertices(
        canonical,
        candidate,
        length_scale=0.8,
        offset_fraction=0.0,
    )
    masks = np.stack(
        [
            _render_prism_silhouette(
                target_ends,
                intrinsics[index],
                extrinsics[index],
                (128, 128),
            )
            for index in range(len(directions))
        ]
    )
    depth = np.ones(masks.shape, dtype=np.float32)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(np.zeros((128, 128, 3), dtype=np.uint8) for _ in directions),
        backend="synthetic-silhouette-regression",
    )
    config = SketchExtrusionConfig(
        silhouette_length_scale_minimum=0.7,
        silhouette_length_scale_maximum=1.1,
        silhouette_length_scale_steps=17,
        silhouette_length_offset_fraction=0.0,
        silhouette_length_offset_steps=1,
        silhouette_length_regularization_weight=0.02,
        silhouette_length_minimum_score_gain=0.001,
    )
    refined = _silhouette_length_refinement(
        canonical,
        prediction,
        masks,
        candidate,
        (),
        config,
    )

    assert refined.silhouette_length_refinement is not None
    assert refined.silhouette_length_refinement.applied
    assert refined.length / candidate.length == pytest.approx(0.8, abs=0.03)
    assert refined.silhouette_length_refinement.as_dict()["ground_truth_access"] is False


def test_rectilinear_cells_require_source_profile_preservation() -> None:
    config = SketchExtrusionConfig()
    tee = np.zeros((160, 160), dtype=np.bool_)
    tee[20:145, 70:90] = True
    tee[115:145, 25:135] = True
    tee_result = _rectilinear_raster(tee, config)
    assert tee_result is not None
    tee_rectified, _, _ = tee_result
    assert _mask_iou(tee, tee_rectified) > 0.98

    hexagon = np.zeros((160, 160), dtype=np.uint8)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    vertices = np.rint(
        np.column_stack((80.0 + 60.0 * np.cos(angles), 80.0 + 60.0 * np.sin(angles)))
    ).astype(np.int32)
    cv2.fillPoly(hexagon, [vertices], 1)
    hex_result = _rectilinear_raster(hexagon.astype(np.bool_), config)
    if hex_result is not None:
        hex_rectified, _, _ = hex_result
        assert _mask_iou(hexagon.astype(np.bool_), hex_rectified) < (
            config.rectilinear_profile_minimum_source_iou
        )


def test_calibrated_silhouettes_refine_small_rigid_pose_without_ground_truth() -> None:
    canonical = _canonical(_l_profile_surface())
    initial_backend = SketchExtrusionCadBackend(
        SketchExtrusionConfig(silhouette_pose_refinement_enabled=False)
    )
    initial_backend.generate(canonical, seed=3)
    assert initial_backend.last_report is not None
    candidate = next(
        item
        for item in initial_backend.last_report.axis_candidates
        if item.axis == initial_backend.last_report.selected_axis
    )
    directions = tuple(
        np.asarray(value, dtype=np.float64) / np.linalg.norm(value)
        for value in (
            (1.0, 0.2, 0.3),
            (-1.0, 0.3, 0.2),
            (0.2, 1.0, 0.3),
            (0.3, -1.0, 0.2),
            (0.3, 0.2, 1.0),
            (0.2, 0.3, -1.0),
        )
    )
    extrinsics = np.stack([_look_at(4.0 * direction) for direction in directions])
    intrinsics = np.repeat(
        np.asarray([[[90.0, 0.0, 64.0], [0.0, 90.0, 64.0], [0.0, 0.0, 1.0]]]),
        len(directions),
        axis=0,
    ).astype(np.float32)
    expected_rotation = (2.0, -2.0, 2.0)
    target = _candidate_with_pose_delta(candidate, expected_rotation)
    target_ends = _capture_prism_vertices(
        canonical,
        target,
        length_scale=1.0,
        offset_fraction=0.0,
    )
    masks = np.stack(
        [
            _render_prism_silhouette(
                target_ends,
                intrinsics[index],
                extrinsics[index],
                (128, 128),
            )
            for index in range(len(directions))
        ]
    )
    depth = np.ones(masks.shape, dtype=np.float32)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(np.zeros((128, 128, 3), dtype=np.uint8) for _ in directions),
        backend="synthetic-pose-regression",
    )
    config = SketchExtrusionConfig(
        silhouette_pose_maximum_degrees=4.0,
        silhouette_pose_coarse_step_degrees=2.0,
        silhouette_pose_fine_step_degrees=1.0,
        silhouette_pose_regularization_weight=0.0,
        silhouette_pose_minimum_score_gain=0.001,
    )
    refined = _silhouette_pose_refinement(
        canonical,
        prediction,
        masks,
        candidate,
        (),
        config,
    )

    assert refined.silhouette_pose_refinement is not None
    report = refined.silhouette_pose_refinement
    assert report.applied
    assert report.selected_iou is not None
    assert report.baseline_iou is not None
    assert report.selected_iou > report.baseline_iou + 0.001
    assert report.rotation_degrees == pytest.approx(expected_rotation, abs=1.0)
    assert report.as_dict()["ground_truth_access"] is False

    baseline_ends = _capture_prism_vertices(
        canonical,
        candidate,
        length_scale=1.0,
        offset_fraction=0.0,
    )
    unchanged_masks = np.stack(
        [
            _render_prism_silhouette(
                baseline_ends,
                intrinsics[index],
                extrinsics[index],
                (128, 128),
            )
            for index in range(len(directions))
        ]
    )
    unchanged = _silhouette_pose_refinement(
        canonical,
        prediction,
        unchanged_masks,
        candidate,
        (),
        SketchExtrusionConfig(
            silhouette_pose_maximum_degrees=4.0,
            silhouette_pose_coarse_step_degrees=2.0,
            silhouette_pose_fine_step_degrees=1.0,
        ),
    )
    assert unchanged.silhouette_pose_refinement is not None
    assert not unchanged.silhouette_pose_refinement.applied
    assert unchanged.silhouette_pose_refinement.baseline_iou == pytest.approx(1.0)


def test_non_extrudable_surface_is_explicitly_unsupported() -> None:
    polar, azimuth = np.meshgrid(
        np.linspace(0.0, np.pi, 65),
        np.linspace(0.0, 2.0 * np.pi, 129, endpoint=False),
        indexing="ij",
    )
    sphere = np.column_stack(
        (
            np.sin(polar).ravel() * np.cos(azimuth).ravel(),
            np.sin(polar).ravel() * np.sin(azimuth).ravel(),
            np.cos(polar).ravel(),
        )
    ).astype(np.float32)
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    with pytest.raises(UnsupportedProfileError, match="constant-section extrusion"):
        backend.generate(_canonical(sphere), seed=3)
