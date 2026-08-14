from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cadquery as cq
import cv2
import numpy as np
import pytest

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.backends.revolve import (
    AxialProfile,
    RevolveCadBackend,
    _cad_conditioned_revolve_refinement,
    _candidate_with_cad_refinement,
    _capture_revolve_rings,
    _interior_step_hypothesis,
    _render_revolve_silhouette,
    _revolve_plateau_simplification,
    _revolve_step_refinement,
    _step_axial_profile,
)
from da3_cad.backends.sketch_extrusion import UnsupportedProfileError
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import (
    CanonicalizerConfig,
    ConstructionGrammarConfig,
    RevolveConfig,
    SandboxConfig,
    SketchExtrusionConfig,
)
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.geometry.scale import KnownDimension
from da3_cad.models import DepthPrediction


def _revolved_surface(
    radius: Callable[[np.ndarray], np.ndarray],
    *,
    inner_radius: float | None = None,
    cavity_floor: float = -0.45,
) -> np.ndarray:
    axial = np.linspace(-1.0, 1.0, 129)
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    aa, zz = np.meshgrid(angles, axial, indexing="ij")
    rr = radius(zz)
    parts = [
        np.column_stack(
            (
                rr.ravel() * np.cos(aa).ravel(),
                rr.ravel() * np.sin(aa).ravel(),
                zz.ravel(),
            )
        )
    ]

    cap_angles, cap_radii = np.meshgrid(
        angles,
        np.linspace(0.0, float(radius(np.asarray([-1.0]))[0]), 33),
        indexing="ij",
    )
    parts.append(
        np.column_stack(
            (
                cap_radii.ravel() * np.cos(cap_angles).ravel(),
                cap_radii.ravel() * np.sin(cap_angles).ravel(),
                np.full(cap_radii.size, -1.0),
            )
        )
    )
    top_outer = float(radius(np.asarray([1.0]))[0])
    top_inner = 0.0 if inner_radius is None else inner_radius
    top_angles, top_radii = np.meshgrid(
        angles,
        np.linspace(top_inner, top_outer, 33),
        indexing="ij",
    )
    parts.append(
        np.column_stack(
            (
                top_radii.ravel() * np.cos(top_angles).ravel(),
                top_radii.ravel() * np.sin(top_angles).ravel(),
                np.full(top_radii.size, 1.0),
            )
        )
    )
    if inner_radius is not None:
        inner_axial = np.linspace(cavity_floor, 1.0, 97)
        inner_angles, inner_z = np.meshgrid(angles, inner_axial, indexing="ij")
        parts.append(
            np.column_stack(
                (
                    inner_radius * np.cos(inner_angles).ravel(),
                    inner_radius * np.sin(inner_angles).ravel(),
                    inner_z.ravel(),
                )
            )
        )
        floor_angles, floor_radii = np.meshgrid(
            angles,
            np.linspace(0.0, inner_radius, 33),
            indexing="ij",
        )
        parts.append(
            np.column_stack(
                (
                    floor_radii.ravel() * np.cos(floor_angles).ravel(),
                    floor_radii.ravel() * np.sin(floor_angles).ravel(),
                    np.full(floor_radii.size, cavity_floor),
                )
            )
        )
    return np.concatenate(parts).astype(np.float32)


def _box_surface() -> np.ndarray:
    values = np.linspace(-1.0, 1.0, 81)
    short = np.linspace(-0.5, 0.5, 41)
    x, y = np.meshgrid(values, short, indexing="ij")
    parts = [np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))) for z in (-0.35, 0.35)]
    x, z = np.meshgrid(values, np.linspace(-0.35, 0.35, 31), indexing="ij")
    parts.extend(
        np.column_stack((x.ravel(), np.full(x.size, y_value), z.ravel())) for y_value in (-0.5, 0.5)
    )
    y, z = np.meshgrid(short, np.linspace(-0.35, 0.35, 31), indexing="ij")
    parts.extend(
        np.column_stack((np.full(y.size, x_value), y.ravel(), z.ravel())) for x_value in (-1.0, 1.0)
    )
    return np.concatenate(parts).astype(np.float32)


def _canonical(points: np.ndarray, *, metric: bool = False):
    doubled = np.concatenate((points, points))
    count = len(points)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic-revolve",
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
        view_indices=np.repeat(np.asarray([0, 1], dtype=np.int32), count),
        pixel_xy=np.zeros((len(doubled), 2), dtype=np.int32),
        report=report,
        scale=(
            ScaleChannel(
                status="known",
                units="calibrated-world-unit",
                world_units_to_mm=12.0,
                source="synthetic-calibrated-cameras",
            )
            if metric
            else ScaleChannel()
        ),
    )
    return PointCloudCanonicalizer(
        CanonicalizerConfig(
            outlier_enabled=False,
            consistency_radius_fraction=0.001,
            symmetry_detection_enabled=False,
            plane_ransac_iterations=64,
        )
    ).run(cloud, seed=7)


def _bottle_silhouettes(*, varying_width: bool = False) -> np.ndarray:
    height, width = 160, 120
    yy, xx = np.mgrid[:height, :width]
    masks = []
    factors = np.linspace(0.65, 1.35, 8) if varying_width else np.ones(8)
    for factor in factors:
        axial = np.clip((yy - 15.0) / 130.0, 0.0, 1.0)
        radius = np.where(axial < 0.18, 13.0, 22.0)
        radius = np.where(
            (axial >= 0.18) & (axial < 0.28), 13.0 + 9.0 * (axial - 0.18) / 0.1, radius
        )
        radius = radius * factor
        mask = (yy >= 15) & (yy <= 145) & (np.abs(xx - width / 2.0) <= radius)
        masks.append(mask)
    return np.stack(masks).astype(np.bool_)


def _config() -> RevolveConfig:
    return RevolveConfig(
        axial_bins=64,
        minimum_points_per_bin=12,
        minimum_supported_bin_fraction=0.75,
        minimum_angular_coverage_fraction=0.5,
        maximum_surface_residual_fraction=0.075,
        maximum_unsupported_point_fraction=0.1,
        minimum_radial_symmetry_score=0.78,
    )


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


def _concentric_rim_prediction(
    masks: np.ndarray,
    *,
    depth_discontinuity: bool,
    endpoint_signs: tuple[int, ...] | None = None,
    opposite_camera_groups: bool = False,
) -> DepthPrediction:
    view_count, height, width = masks.shape
    if endpoint_signs is not None and len(endpoint_signs) != view_count:
        raise ValueError("endpoint_signs must match the mask view count")
    images: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    if opposite_camera_groups:
        if endpoint_signs is None:
            raise ValueError("opposite camera groups require endpoint signs")
        eyes = [
            np.asarray(
                [
                    4.0 if sign < 0 else -4.0,
                    0.20 * np.cos(angle),
                    0.20 * np.sin(angle),
                ]
            )
            for sign, angle in zip(
                endpoint_signs,
                np.linspace(0.0, 2.0 * np.pi, view_count, endpoint=False),
                strict=True,
            )
        ]
    else:
        eyes = [
            np.asarray([4.0, 0.35 * np.cos(angle), 0.35 * np.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * np.pi, view_count, endpoint=False)
        ]
    for view_index, mask in enumerate(masks):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[mask] = 220
        sign = 0 if endpoint_signs is None else endpoint_signs[view_index]
        center = (width // 2, height // 2 + sign * 40)
        cv2.ellipse(image, center, (20, 14), 0.0, 0.0, 360.0, (35, 35, 35), 2)
        cv2.ellipse(image, center, (8, 5), 0.0, 0.0, 360.0, (20, 20, 20), -1)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        if depth_discontinuity:
            inner = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(inner, center, (7, 4), 0.0, 0.0, 360.0, 1, -1)
            depth[inner.astype(bool)] += 0.08
        images.append(image)
        depths.append(depth)
    intrinsics = np.repeat(
        np.asarray([[[100.0, 0.0, width / 2.0], [0.0, 100.0, height / 2.0], [0.0, 0.0, 1.0]]]),
        view_count,
        axis=0,
    ).astype(np.float32)
    return DepthPrediction(
        depth=np.stack(depths),
        confidence=np.ones((view_count, height, width), dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=np.stack([_look_at(eye) for eye in eyes]),
        processed_images=tuple(images),
        backend="synthetic-concentric-rim",
    )


def test_recovers_axis_and_valid_outer_profile(tmp_path: Path) -> None:
    points = _revolved_surface(lambda axial: 0.52 + 0.08 * (axial + 1.0) / 2.0)
    backend = RevolveCadBackend(_config())
    program = backend.generate(_canonical(points), seed=7)

    assert backend.last_report is not None
    assert backend.last_report.program_family == "revolve"
    assert not backend.last_report.profile.shell
    selected = next(
        candidate
        for candidate in backend.last_report.axis_candidates
        if candidate.axis == backend.last_report.selected_axis
    )
    assert selected.radial_symmetry_score >= _config().minimum_radial_symmetry_score
    assert selected.angular_coverage_fraction >= 0.5
    assert _interior_step_hypothesis(selected, _config()) is None
    assert program.parameters["revolve_angle_degrees"] == 360.0
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error
    shape = cq.importers.importStep(str(tmp_path / "model.step")).val()
    assert "BSPLINE" not in {face.geomType() for face in shape.Faces()}
    assert {face.geomType() for face in shape.Faces()} <= {"PLANE", "CONE", "CYLINDER"}


def test_interior_step_grammar_recovers_sharp_shoulders_from_masks() -> None:
    points = _revolved_surface(lambda axial: np.where(np.abs(axial) <= 0.2, 0.8, 0.4))
    canonical = _canonical(points)
    initial_config = _config().model_copy(
        update={
            "profile_smoothing_sigma_bins": 2.0,
            "step_profile_refinement_enabled": False,
        }
    )
    backend = RevolveCadBackend(initial_config)
    backend.generate(canonical, seed=7)
    assert backend.last_report is not None
    candidate = backend.last_report.axis_candidates[0]
    hypothesis = _interior_step_hypothesis(candidate, initial_config)
    assert hypothesis is not None
    target_profile = _step_axial_profile(
        candidate,
        hypothesis,
        center_fraction=0.5,
        width_fraction=0.2,
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
    target_rings = _capture_revolve_rings(canonical, candidate, target_profile)
    masks = np.stack(
        [
            _render_revolve_silhouette(
                target_rings,
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
        backend="synthetic-step-regression",
    )
    config = initial_config.model_copy(
        update={
            "step_profile_refinement_enabled": True,
            "step_profile_center_minimum_fraction": 0.35,
            "step_profile_center_maximum_fraction": 0.65,
            "step_profile_center_steps": 13,
            "step_profile_width_minimum_fraction": 0.10,
            "step_profile_width_maximum_fraction": 0.40,
            "step_profile_width_steps": 13,
            "step_profile_pose_offset_fraction": 0.04,
            "step_profile_pose_offset_steps": 5,
            "step_profile_minimum_score_gain": 0.0,
        }
    )
    refined = _revolve_step_refinement(
        canonical,
        prediction,
        masks,
        candidate,
        config,
    )

    assert refined.step_refinement is not None
    assert refined.step_refinement.applied
    assert refined.step_refinement.selected_center_fraction == pytest.approx(0.5, abs=0.03)
    assert refined.step_refinement.selected_width_fraction == pytest.approx(0.2, abs=0.04)
    assert len(refined.profile.outer_points) == 6
    assert refined.step_refinement.as_dict()["ground_truth_access"] is False


def _synthetic_revolve_prediction(
    canonical,
    candidate,
    profile: AxialProfile,
) -> tuple[DepthPrediction, np.ndarray]:
    assert canonical.orientation is not None
    base = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    local = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    axis_world = (base @ local)[:, candidate.axis]
    axis_world /= np.linalg.norm(axis_world)
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
    rings = _capture_revolve_rings(canonical, candidate, profile)
    masks = np.stack(
        [
            _render_revolve_silhouette(
                rings,
                intrinsics[index],
                extrinsics[index],
                (128, 128),
            )
            for index in range(len(directions))
        ]
    )
    depth = np.ones(masks.shape, dtype=np.float32)
    return (
        DepthPrediction(
            depth=depth,
            confidence=np.ones_like(depth),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(np.zeros((128, 128, 3), dtype=np.uint8) for _ in directions),
            backend="synthetic-cad-conditioned-revolve",
        ),
        masks,
    )


def test_cad_conditioned_refinement_improves_all_views_and_preserves_through() -> None:
    canonical = _canonical(_revolved_surface(lambda axial: np.full_like(axial, 0.65)))
    config = _config().model_copy(update={"cad_refinement_enabled": False})
    backend = RevolveCadBackend(config)
    backend.generate(canonical, seed=7)
    assert backend.last_report is not None
    candidate = next(
        item
        for item in backend.last_report.axis_candidates
        if item.axis == backend.last_report.selected_axis
        and item.profile == backend.last_report.profile
    )
    inner = (
        (0.08, candidate.profile.outer_points[0][1]),
        (0.08, candidate.profile.outer_points[-1][1]),
    )
    through_profile = AxialProfile(
        points=(*candidate.profile.outer_points, *reversed(inner)),
        outer_points=candidate.profile.outer_points,
        inner_points=inner,
        shell=True,
        opening="through",
    )
    target = replace(candidate, profile=through_profile)
    prediction, masks = _synthetic_revolve_prediction(canonical, target, through_profile)
    rotation = [0.0, 0.0, 0.0]
    rotation[next(index for index in range(3) if index != target.axis)] = 3.0
    perturbed = _candidate_with_cad_refinement(
        target,
        axial_scale=0.90,
        radial_scale=0.90,
        axial_offset_fraction=0.02,
        rotation_degrees=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
    )
    refine_config = config.model_copy(
        update={
            "cad_refinement_enabled": True,
            "cad_refinement_axial_scale_minimum": 0.80,
            "cad_refinement_axial_scale_maximum": 1.30,
            "cad_refinement_axial_scale_steps": 11,
            "cad_refinement_radial_scale_minimum": 0.80,
            "cad_refinement_radial_scale_maximum": 1.30,
            "cad_refinement_radial_scale_steps": 11,
            "cad_refinement_axial_offset_fraction": 0.08,
            "cad_refinement_axial_offset_steps": 9,
            "cad_refinement_pose_maximum_degrees": 5.0,
            "cad_refinement_pose_coarse_step_degrees": 2.5,
            "cad_refinement_pose_fine_step_degrees": 0.5,
            "cad_refinement_minimum_score_gain": 0.0,
            "cad_refinement_regularization_weight": 0.0,
            "cad_refinement_maximum_view_iou_drop": 0.0,
            "cad_refinement_maximum_surface_residual_ratio": 1.5,
            "cad_refinement_surface_tolerance_fraction": 0.10,
            "cad_refinement_surface_samples": 1024,
        }
    )
    refined = _cad_conditioned_revolve_refinement(
        canonical,
        prediction,
        masks,
        perturbed,
        refine_config,
    )

    assert refined.cad_refinement is not None
    report = refined.cad_refinement
    assert report.applied, report.as_dict()
    assert report.topology == "through"
    assert report.topology_preserved
    assert refined.profile.opening == "through"
    assert len(refined.profile.inner_points) == len(perturbed.profile.inner_points)
    assert report.selected_mean_iou is not None
    assert report.baseline_mean_iou is not None
    assert report.selected_mean_iou > report.baseline_mean_iou
    assert all(
        selected >= baseline
        for selected, baseline in zip(
            report.selected_view_iou,
            report.baseline_view_iou,
            strict=True,
        )
    )
    assert report.as_dict()["ground_truth_access"] is False


def test_plateau_simplification_removes_projection_taper_and_preserves_bore_ratio() -> None:
    canonical = _canonical(_revolved_surface(lambda axial: np.full_like(axial, 0.65)))
    config = _config().model_copy(
        update={
            "cad_refinement_enabled": False,
            "plateau_simplification_enabled": False,
        }
    )
    backend = RevolveCadBackend(config)
    backend.generate(canonical, seed=7)
    assert backend.last_report is not None
    candidate = next(
        item
        for item in backend.last_report.axis_candidates
        if item.axis == backend.last_report.selected_axis
        and item.profile == backend.last_report.profile
    )
    first = candidate.profile.outer_points[0][1]
    last = candidate.profile.outer_points[-1][1]
    span = last - first
    radius = float(np.median([point[0] for point in candidate.profile.outer_points]))
    ratio = 0.30
    inner = ((ratio * radius, first), (ratio * radius, last))
    target_profile = AxialProfile(
        points=((radius, first), (radius, last), *reversed(inner)),
        outer_points=((radius, first), (radius, last)),
        inner_points=inner,
        shell=True,
        opening="through",
    )
    target = replace(candidate, profile=target_profile)
    prediction, masks = _synthetic_revolve_prediction(canonical, target, target_profile)
    tapered_outer = (
        (0.85 * radius, first),
        (radius, first + 0.15 * span),
        (radius, last - 0.15 * span),
        (0.85 * radius, last),
    )
    tapered_profile = AxialProfile(
        points=(*tapered_outer, *reversed(inner)),
        outer_points=tapered_outer,
        inner_points=inner,
        shell=True,
        opening="through",
    )
    tapered = replace(
        candidate,
        profile=tapered_profile,
        evidence_source="multi-view-silhouette-invariance+rgb-depth-concentric-interior-through",
        silhouette_metrics={"interior_radius_ratio": ratio},
    )
    refined = _revolve_plateau_simplification(
        canonical,
        prediction,
        masks,
        tapered,
        config.model_copy(
            update={
                "plateau_simplification_enabled": True,
                "plateau_simplification_minimum_mask_gain": 0.0,
                "plateau_simplification_minimum_surface_p90_gain": 0.0,
                "cad_refinement_surface_samples": 1024,
            }
        ),
    )

    assert refined.plateau_simplification is not None
    report = refined.plateau_simplification
    assert report.applied, report.as_dict()
    assert report.topology == "through"
    assert report.topology_preserved
    assert refined.profile.opening == "through"
    assert len(refined.profile.outer_points) == 2
    assert refined.profile.inner_points[0][0] / refined.profile.outer_points[0][0] == pytest.approx(
        ratio
    )
    assert all(
        selected >= baseline
        for selected, baseline in zip(
            report.selected_view_iou,
            report.baseline_view_iou,
            strict=True,
        )
    )


def test_cad_conditioned_refinement_rolls_back_when_input_masks_match() -> None:
    canonical = _canonical(_revolved_surface(lambda axial: np.full_like(axial, 0.65)))
    config = _config().model_copy(update={"cad_refinement_enabled": False})
    backend = RevolveCadBackend(config)
    backend.generate(canonical, seed=7)
    assert backend.last_report is not None
    candidate = next(
        item
        for item in backend.last_report.axis_candidates
        if item.axis == backend.last_report.selected_axis
        and item.profile == backend.last_report.profile
    )
    prediction, masks = _synthetic_revolve_prediction(canonical, candidate, candidate.profile)
    refined = _cad_conditioned_revolve_refinement(
        canonical,
        prediction,
        masks,
        candidate,
        config.model_copy(
            update={
                "cad_refinement_enabled": True,
                "cad_refinement_surface_samples": 1024,
            }
        ),
    )

    assert refined.cad_refinement is not None
    assert not refined.cad_refinement.applied
    assert refined.profile == candidate.profile
    assert refined.frame_canonical_columns == candidate.frame_canonical_columns


def test_visible_inner_wall_emits_valid_shell(tmp_path: Path) -> None:
    points = _revolved_surface(
        lambda axial: np.full_like(axial, 0.65),
        inner_radius=0.5,
        cavity_floor=-0.45,
    )
    backend = RevolveCadBackend(_config())
    program = backend.generate(_canonical(points), seed=7)

    assert backend.last_report is not None
    assert backend.last_report.program_family == "revolve-shell"
    assert backend.last_report.profile.shell
    assert backend.last_report.profile.opening in {"lower", "upper"}
    assert len(backend.last_report.profile.inner_points) >= 2
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error


def test_metric_scaling_never_scales_angle_or_profile_fractions() -> None:
    points = _revolved_surface(lambda axial: np.full_like(axial, 0.55))
    backend = RevolveCadBackend(_config())
    program = backend.generate(_canonical(points, metric=True), seed=7)

    assert backend.last_report is not None
    factor = backend.last_report.scale.millimeters_per_unit
    assert factor is not None
    assert program.parameters["revolve_angle_degrees"] == 360.0
    assert program.parameters["body_height"] == pytest.approx(
        backend.last_report.parameters_normalized["body_height"] * factor
    )
    fractions = [name for name in program.parameters if name.endswith("_axis_fraction")]
    assert fractions
    for name in fractions:
        assert program.parameters[name] == pytest.approx(
            backend.last_report.parameters_normalized[name]
        )


def test_known_body_height_sets_metric_scale() -> None:
    points = _revolved_surface(lambda axial: np.full_like(axial, 0.55))
    backend = RevolveCadBackend(_config())
    program = backend.generate(
        _canonical(points),
        seed=7,
        known_dimension=KnownDimension.parse("body_height=80mm"),
    )

    assert program.parameters["body_height"] == pytest.approx(80.0)
    assert program.parameters["revolve_angle_degrees"] == 360.0


def test_consistent_silhouettes_enable_revolve_when_partial_3d_is_not_radial(
    tmp_path: Path,
) -> None:
    backend = RevolveCadBackend(_config())
    program = backend.generate(
        _canonical(_box_surface()),
        seed=7,
        masks=_bottle_silhouettes(),
    )

    assert backend.last_report is not None
    selected = next(
        candidate
        for candidate in backend.last_report.axis_candidates
        if candidate.axis == backend.last_report.selected_axis
    )
    assert selected.evidence_source == "multi-view-silhouette-invariance"
    assert selected.silhouette_metrics is not None
    assert selected.silhouette_metrics["width_over_height_cv"] < 1e-6
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error


def test_concentric_rgb_depth_rims_add_only_an_observed_side_cavity(
    tmp_path: Path,
) -> None:
    masks = _bottle_silhouettes()
    canonical = _canonical(_box_surface())
    backend = RevolveCadBackend(_config())

    flat = _concentric_rim_prediction(masks, depth_discontinuity=False)
    flat_program = backend.generate(canonical, seed=7, prediction=flat, masks=masks)
    assert flat_program.program_family == "revolve"
    assert backend.last_report is not None
    assert not backend.last_report.profile.shell

    cavity = _concentric_rim_prediction(masks, depth_discontinuity=True)
    cavity_program = backend.generate(canonical, seed=7, prediction=cavity, masks=masks)
    assert cavity_program.program_family == "revolve-shell"
    assert backend.last_report is not None
    assert backend.last_report.profile.shell
    assert backend.last_report.profile.opening in {"lower", "upper"}
    selected = backend.last_report.axis_candidates[0]
    assert selected.silhouette_metrics is not None
    assert selected.silhouette_metrics["interior_through_hole_claimed"] == 0.0
    validation = validate_and_export(cavity_program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error


def _constant_cylinder_silhouettes(view_count: int = 4) -> np.ndarray:
    height, width = 160, 120
    yy, xx = np.mgrid[:height, :width]
    mask = (yy >= 15) & (yy <= 145) & (np.abs(xx - width / 2.0) <= 24)
    return np.repeat(mask[None, ...], view_count, axis=0)


def test_opposite_endpoint_rims_select_through_hole_without_gt(
    tmp_path: Path,
) -> None:
    masks = _constant_cylinder_silhouettes()
    endpoint_signs = (-1, -1, 1, 1)
    prediction = _concentric_rim_prediction(
        masks,
        depth_discontinuity=True,
        endpoint_signs=endpoint_signs,
        opposite_camera_groups=True,
    )
    config = _config().model_copy(update={"silhouette_minimum_views": 4})
    backend = RevolveCadBackend(config)
    program = backend.generate(
        _canonical(_box_surface()),
        seed=7,
        prediction=prediction,
        masks=masks,
    )

    assert program.program_family == "revolve-shell"
    assert backend.last_report is not None
    assert backend.last_report.profile.opening == "through"
    selected = backend.last_report.axis_candidates[0]
    assert selected.silhouette_metrics is not None
    assert selected.silhouette_metrics["interior_endpoint_negative_views"] >= 1.0
    assert selected.silhouette_metrics["interior_endpoint_positive_views"] >= 1.0
    assert (
        selected.silhouette_metrics["interior_endpoint_maximum_camera_angle_degrees"]
        >= config.interior_ellipse.concentric_through_minimum_camera_angle_degrees
    )
    assert selected.silhouette_metrics["interior_through_hole_claimed"] == 1.0
    hypotheses = {item.topology: item for item in selected.topology_hypotheses}
    assert hypotheses["through"].admitted
    assert hypotheses["through"].selected
    assert hypotheses["through"].as_dict()["ground_truth_access"] is False
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error
    shape = cq.importers.importStep(str(tmp_path / "model.step")).val()
    assert sum(face.geomType() == "CYLINDER" for face in shape.Faces()) >= 2


def test_opposite_image_endpoints_without_camera_separation_do_not_claim_through() -> None:
    masks = _constant_cylinder_silhouettes()
    prediction = _concentric_rim_prediction(
        masks,
        depth_discontinuity=True,
        endpoint_signs=(-1, -1, 1, 1),
        opposite_camera_groups=False,
    )
    config = _config().model_copy(update={"silhouette_minimum_views": 4})
    backend = RevolveCadBackend(config)
    backend.generate(
        _canonical(_box_surface()),
        seed=7,
        prediction=prediction,
        masks=masks,
    )

    assert backend.last_report is not None
    assert backend.last_report.profile.opening != "through"
    selected = backend.last_report.axis_candidates[0]
    hypotheses = {item.topology: item for item in selected.topology_hypotheses}
    assert not hypotheses["through"].admitted
    assert not hypotheses["through"].selected


def test_varying_silhouette_widths_do_not_invent_revolve() -> None:
    backend = RevolveCadBackend(_config())
    with pytest.raises(UnsupportedProfileError, match="silhouette invariance rejected"):
        backend.generate(
            _canonical(_box_surface()),
            seed=7,
            masks=_bottle_silhouettes(varying_width=True),
        )


def test_rectangular_prism_is_not_silently_accepted_as_revolve() -> None:
    backend = RevolveCadBackend(_config())
    with pytest.raises(UnsupportedProfileError, match="body of revolution"):
        backend.generate(_canonical(_box_surface()), seed=7)


def _grammar() -> ConstructionGrammarCadBackend:
    return ConstructionGrammarCadBackend(
        ConstructionGrammarConfig(),
        SketchExtrusionConfig(silhouette_profile_enabled=False),
        _config(),
    )


def test_construction_grammar_selects_revolve_for_varying_radius() -> None:
    points = _revolved_surface(lambda axial: 0.35 + 0.35 * (axial + 1.0) / 2.0)
    backend = _grammar()
    program = backend.generate(_canonical(points), seed=7)

    assert backend.last_report is not None
    assert backend.last_report.selected_family == "revolve"
    assert program.backend == "construction-grammar-v1"
    assert program.program_family == "revolve"
    payload = backend.last_report.as_dict()
    assert payload["gt_blind"] is True
    assert payload["gt_or_mesh_argument_available"] is False


def test_construction_grammar_selects_extrude_for_rectangular_prism() -> None:
    backend = _grammar()
    program = backend.generate(_canonical(_box_surface()), seed=7)

    assert backend.last_report is not None
    assert backend.last_report.selected_family == "extrude"
    assert program.backend == "construction-grammar-v1"
    assert program.program_family == "sketch-extrusion"
    assert any(item["family"] == "revolve" for item in backend.last_report.rejections)


def test_construction_grammar_selects_inner_profile_revolve() -> None:
    points = _revolved_surface(
        lambda axial: np.full_like(axial, 0.65),
        inner_radius=0.5,
        cavity_floor=-0.45,
    )
    backend = _grammar()
    program = backend.generate(_canonical(points), seed=7)

    assert backend.last_report is not None
    assert backend.last_report.selected_family == "revolve"
    assert program.program_family == "revolve-shell"
