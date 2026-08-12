from __future__ import annotations

import numpy as np
import pytest

from da3_cad.backends.geometric_fitter import (
    GeometricCadBackend,
    _detect_circular_void,
)
from da3_cad.cad.program import extract_parameters
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import (
    CanonicalizerConfig,
    GeometricFitterConfig,
    SandboxConfig,
)
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.geometry.scale import KnownDimension


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


def _canonical_plate(*, with_hole: bool, metric: bool = False):
    one_view = _plate_surface(with_hole=with_hole)
    points = np.concatenate((one_view, one_view))
    count = len(one_view)
    views = np.repeat(np.asarray([0, 1], dtype=np.int32), count)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic-mesh-surface",
        require_confidence=True,
        views=(
            ViewFusionStats(0, count, count, count, count, count),
            ViewFusionStats(1, count, count, count, count, count),
        ),
    )
    cloud = FusedPointCloud(
        points=points,
        colors=np.zeros((len(points), 3), dtype=np.uint8),
        confidences=np.ones(len(points), dtype=np.float32),
        view_indices=views,
        pixel_xy=np.zeros((len(points), 2), dtype=np.int32),
        report=report,
        scale=(
            ScaleChannel(
                status="known",
                units="calibrated-world-unit",
                world_units_to_mm=10.0,
                source="synthetic-calibrated-cameras",
                evidence={"test": True},
            )
            if metric
            else ScaleChannel()
        ),
    )
    config = CanonicalizerConfig(
        confidence_percentile=0.0,
        outlier_enabled=False,
        consistency_radius_fraction=0.001,
        plane_ransac_iterations=64,
    )
    return PointCloudCanonicalizer(config).run(cloud, seed=3)


def test_void_detector_requires_supported_circular_boundary() -> None:
    points = _plate_surface(with_hole=True)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    detected = _detect_circular_void(points, minimum, maximum, GeometricFitterConfig())
    assert detected.accepted is True
    assert detected.radius == pytest.approx(0.2, abs=0.02)
    assert detected.angular_coverage >= 0.9


def test_void_detector_skips_a_larger_unsupported_surface_gap() -> None:
    points = _plate_surface(with_hole=True)
    keep = ~(
        (points[:, 2] > 0.05)
        & (points[:, 0] > 0.45)
        & (points[:, 1] > 0.05)
    )
    incomplete = points[keep]
    detected = _detect_circular_void(
        incomplete,
        incomplete.min(axis=0),
        incomplete.max(axis=0),
        GeometricFitterConfig(),
    )
    assert detected.accepted is True
    assert detected.center_x == pytest.approx(0.0, abs=0.04)
    assert detected.center_y == pytest.approx(0.0, abs=0.04)


def test_geometric_fitter_emits_valid_parameterized_through_hole(tmp_path) -> None:
    canonical = _canonical_plate(with_hole=True)
    fitter = GeometricCadBackend(GeometricFitterConfig())
    program = fitter.generate(canonical, seed=3)

    assert fitter.last_report is not None
    assert fitter.last_report.template == "rectangular-extrusion-through-hole"
    parameters = extract_parameters(program.source)
    assert parameters["body_width"] == pytest.approx(2.0, abs=0.08)
    assert parameters["body_depth"] == pytest.approx(1.4, abs=0.08)
    assert parameters["hole_1_diameter"] == pytest.approx(0.4, abs=0.06)
    validation = validate_and_export(
        program.source,
        tmp_path,
        SandboxConfig(),
    )
    assert validation.valid, validation.error
    assert validation.volume is not None and validation.volume > 0.0


def test_known_dimension_scales_every_geometric_parameter() -> None:
    canonical = _canonical_plate(with_hole=True)
    fitter = GeometricCadBackend(GeometricFitterConfig())
    program = fitter.generate(
        canonical,
        seed=3,
        known_dimension=KnownDimension.parse("hole_1_diameter=8mm"),
    )
    assert fitter.last_report is not None
    assert fitter.last_report.scale.status == "known"
    assert program.parameters["hole_1_diameter"] == pytest.approx(8.0)
    assert program.parameters["body_width"] == pytest.approx(40.0, abs=3.0)


def test_metric_camera_scale_reaches_emitted_cad_parameters() -> None:
    canonical = _canonical_plate(with_hole=False, metric=True)
    assert canonical.normalization is not None
    expected_factor = canonical.normalization.largest_extent * 10.0 / 2.0
    assert canonical.scale.status == "known"
    assert canonical.scale.source == "camera-bundle"
    assert canonical.scale.millimeters_per_unit == pytest.approx(expected_factor)

    fitter = GeometricCadBackend(GeometricFitterConfig())
    program = fitter.generate(canonical, seed=3)

    assert fitter.last_report is not None
    report = fitter.last_report
    assert report.scale.status == "known"
    assert report.parameters_emitted["body_width"] == pytest.approx(
        report.parameters_normalized["body_width"] * expected_factor
    )
    assert program.parameters["body_width"] == pytest.approx(20.0, abs=0.8)


def test_no_void_is_not_silently_invented() -> None:
    canonical = _canonical_plate(with_hole=False)
    fitter = GeometricCadBackend(GeometricFitterConfig())
    fitter.generate(canonical, seed=3)
    assert fitter.last_report is not None
    assert fitter.last_report.template == "rectangular-extrusion"
    assert "hole_1_diameter" not in fitter.last_report.parameters_emitted
