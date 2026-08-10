from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.canonicalizer import (
    PointCloudCanonicalizer,
    write_canonicalizer_artifacts,
)
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
)
from da3_cad.geometry.scale import KnownDimension


def _plate_cloud(*, scale: float = 1.0, offset: float = 0.0) -> FusedPointCloud:
    x, y = np.meshgrid(
        np.linspace(-2.0, 2.0, 40),
        np.linspace(-1.0, 1.0, 30),
        indexing="ij",
    )
    bottom = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -0.02)))
    top = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.02)))
    one_view = np.concatenate((bottom, top)).astype(np.float32)
    points = np.concatenate((one_view, one_view))
    points = points * scale + offset
    count_per_view = len(one_view)
    total = len(points)
    views = np.repeat(np.asarray([0, 1], dtype=np.int32), count_per_view)
    confidence = np.linspace(1.0, 2.0, total, dtype=np.float32)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic",
        require_confidence=True,
        views=(
            ViewFusionStats(
                0, count_per_view, count_per_view, count_per_view, count_per_view, count_per_view
            ),
            ViewFusionStats(
                1, count_per_view, count_per_view, count_per_view, count_per_view, count_per_view
            ),
        ),
    )
    return FusedPointCloud(
        points=points.astype(np.float32),
        colors=np.zeros((total, 3), dtype=np.uint8),
        confidences=confidence,
        view_indices=views,
        pixel_xy=np.zeros((total, 2), dtype=np.int32),
        report=report,
    )


def _config() -> CanonicalizerConfig:
    return CanonicalizerConfig(
        confidence_percentile=0.0,
        outlier_enabled=False,
        consistency_radius_fraction=0.001,
        symmetry_completion_enabled=False,
        plane_ransac_iterations=64,
    )


def test_full_canonicalizer_contract_and_similarity_invariance() -> None:
    canonicalizer = PointCloudCanonicalizer(_config())
    first = canonicalizer.run(_plate_cloud(), seed=123)
    repeated = canonicalizer.run(_plate_cloud(), seed=123)
    transformed = canonicalizer.run(
        _plate_cloud(scale=7.0, offset=11.0),
        seed=123,
        known_dimension=KnownDimension.parse("hole_1_diameter=8mm"),
    )

    assert first.decoder_tensor.shape == (1, 256, 3)
    assert first.decoder_points.dtype == np.float32
    assert np.isfinite(first.decoder_points).all()
    assert first.orientation is not None
    assert first.orientation.method == "planar-dominance-symmetry"
    assert first.orientation.determinant == pytest.approx(1.0)
    assert np.array_equal(first.decoder_points, repeated.decoder_points)
    np.testing.assert_allclose(first.decoder_points, transformed.decoder_points, atol=2e-5)
    assert transformed.scale.status == "pending"

    unit_min = first.unit_points.min(axis=0)
    unit_max = first.unit_points.max(axis=0)
    assert float(np.max(unit_max - unit_min)) == pytest.approx(1.0)
    np.testing.assert_allclose(unit_min + unit_max, np.ones(3), atol=1e-6)
    assert [stage.name for stage in first.stages] == [
        "input",
        "confidence",
        "outliers",
        "multi-view-consistency",
        "symmetry",
        "orientation",
        "sampling",
        "normalization",
        "scale-channel",
    ]
    assert any("planar degeneracy" in warning for warning in first.warnings)


def test_every_stage_is_explicitly_ablatable() -> None:
    config = CanonicalizerConfig(
        confidence_enabled=False,
        outlier_enabled=False,
        consistency_enabled=False,
        symmetry_detection_enabled=False,
        symmetry_completion_enabled=False,
        orientation_enabled=False,
        sampling_enabled=False,
        normalization_enabled=False,
    )
    result = PointCloudCanonicalizer(config).run(_plate_cloud(), seed=4)
    enabled = {stage.name: stage.enabled for stage in result.stages}
    assert enabled["confidence"] is False
    assert enabled["outliers"] is False
    assert enabled["multi-view-consistency"] is False
    assert enabled["symmetry"] is False
    assert enabled["orientation"] is False
    assert enabled["sampling"] is False
    assert enabled["normalization"] is False
    assert result.decoder_points.shape == (256, 3)
    assert result.normalization is None


def test_exact_preselected_cloud_is_preserved_when_sampling_is_disabled() -> None:
    source = _plate_cloud()
    keep = np.arange(256)
    cloud = FusedPointCloud(
        points=source.points[keep],
        colors=source.colors[keep],
        confidences=source.confidences[keep],
        view_indices=source.view_indices[keep],
        pixel_xy=source.pixel_xy[keep],
        report=source.report,
        scale=source.scale,
    )
    config = CanonicalizerConfig(
        confidence_enabled=False,
        outlier_enabled=False,
        consistency_enabled=False,
        symmetry_detection_enabled=False,
        orientation_enabled=False,
        sampling_enabled=False,
        normalization_enabled=False,
    )

    result = PointCloudCanonicalizer(config).run(cloud, seed=5)

    assert np.array_equal(result.decoder_points, cloud.points)
    sampling = next(stage for stage in result.stages if stage.name == "sampling")
    assert sampling.report["method"] == "identity-exact-contract"
    assert not any("stable-index" in warning for warning in result.warnings)


def test_canonicalizer_artifacts_contain_exact_decoder_tensor(tmp_path: Path) -> None:
    result = PointCloudCanonicalizer(_config()).run(_plate_cloud(), seed=9)
    output = tmp_path / "canonical"
    write_canonicalizer_artifacts(output, result)
    tensor = np.load(output / "decoder_input.npy", allow_pickle=False)
    assert np.array_equal(tensor, result.decoder_tensor)
    assert (output / "canonicalizer_trace.json").is_file()
    assert len(list(output.glob("*.npz"))) == len(result.stages)


def test_seed_repeatability_across_processes(tmp_path: Path) -> None:
    cloud = _plate_cloud()
    input_path = tmp_path / "cloud.npz"
    np.savez(
        input_path,
        points=cloud.points,
        confidence=cloud.confidences,
        views=cloud.view_indices,
    )
    probe = r"""
import hashlib
import sys
import numpy as np
from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, ViewFusionStats

with np.load(sys.argv[1], allow_pickle=False) as data:
    points = data["points"]
    confidence = data["confidence"]
    views = data["views"]
n = len(points) // 2
report = FusionReport(
    confidence_percentile=0.0,
    confidence_scope="per-view",
    confidence_thresholds=(1.0, 1.0),
    mask_source="process-probe",
    require_confidence=True,
    views=(
        ViewFusionStats(0, n, n, n, n, n),
        ViewFusionStats(1, n, n, n, n, n),
    ),
)
cloud = FusedPointCloud(
    points=points,
    colors=np.zeros((len(points), 3), dtype=np.uint8),
    confidences=confidence,
    view_indices=views,
    pixel_xy=np.zeros((len(points), 2), dtype=np.int32),
    report=report,
)
config = CanonicalizerConfig(
    confidence_percentile=0.0,
    outlier_enabled=False,
    consistency_radius_fraction=0.001,
    plane_ransac_iterations=64,
)
result = PointCloudCanonicalizer(config).run(cloud, seed=123)
print(hashlib.sha256(result.decoder_points.tobytes()).hexdigest())
"""
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    hashes = [
        subprocess.run(
            [sys.executable, "-c", probe, str(input_path)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert hashes[0] == hashes[1]
    assert len(hashes[0]) == hashlib.sha256().digest_size * 2
