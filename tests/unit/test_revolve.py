from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cadquery as cq
import numpy as np
import pytest

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.backends.revolve import (
    RevolveCadBackend,
    _capture_revolve_rings,
    _interior_step_hypothesis,
    _render_revolve_silhouette,
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
