from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from da3_cad.backends.da3 import get_da3_model_spec
from da3_cad.config import AppConfig, Da3Config
from da3_cad.geometry.cameras import CameraBundle
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.models import DepthPrediction, ObservationSet
from da3_cad.observations import load_observations


class _FakeDa3Backend:
    received_intrinsics: np.ndarray | None = None
    received_extrinsics: np.ndarray | None = None
    received_align: bool | None = None

    def __init__(self, **_kwargs: Any) -> None:
        self.spec = get_da3_model_spec("base")
        self.last_runtime_report: dict[str, object] | None = None
        self.last_lifecycle: object | None = None

    def predict(
        self,
        observations: ObservationSet,
        *,
        device: str,
        seed: int,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
    ) -> DepthPrediction:
        del device, seed
        if extrinsics is None or intrinsics is None:
            raise AssertionError("test expected camera-conditioned DA3")
        type(self).received_intrinsics = intrinsics.copy()
        type(self).received_extrinsics = extrinsics.copy()
        type(self).received_align = align_to_input_ext_scale
        count = len(observations.images)
        images: list[np.ndarray] = []
        for _ in range(count):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[2:6, 2:6] = (80, 140, 220)
            images.append(image)
        self.last_lifecycle = object()
        self.last_runtime_report = {"camera_conditioning": True, "test_double": True}
        return DepthPrediction(
            depth=np.ones((count, 8, 8), dtype=np.float32),
            confidence=np.ones((count, 8, 8), dtype=np.float32),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(images),
            backend="fake-da3-camera-conditioned",
        )


class _FakeUnposedDa3Backend(_FakeDa3Backend):
    def predict(
        self,
        observations: ObservationSet,
        *,
        device: str,
        seed: int,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
    ) -> DepthPrediction:
        del device, seed, align_to_input_ext_scale
        if extrinsics is not None or intrinsics is not None:
            raise AssertionError("test expected DA3-estimated cameras")
        count = len(observations.images)
        images: list[np.ndarray] = []
        for _ in range(count):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[2:6, 2:6] = (80, 140, 220)
            images.append(image)
        predicted_intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
        predicted_intrinsics[:, 0, 0] = 100.0
        predicted_intrinsics[:, 1, 1] = 100.0
        predicted_intrinsics[:, 0, 2] = 4.0
        predicted_intrinsics[:, 1, 2] = 4.0
        predicted_extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
        predicted_extrinsics[-1, 0, 3] = 5.0
        self.last_lifecycle = object()
        self.last_runtime_report = {"camera_conditioning": False, "test_double": True}
        return DepthPrediction(
            depth=np.ones((count, 8, 8), dtype=np.float32),
            confidence=np.ones((count, 8, 8), dtype=np.float32),
            intrinsics=predicted_intrinsics,
            extrinsics=predicted_extrinsics,
            processed_images=tuple(images),
            backend="fake-da3-one-pose-island",
        )


class _FakeBoundedPoseIslandDa3Backend(_FakeUnposedDa3Backend):
    def predict(self, *args: Any, **kwargs: Any) -> DepthPrediction:
        prediction = super().predict(*args, **kwargs)
        extrinsics = prediction.extrinsics.copy()
        extrinsics[-1, 0, 3] = 0.05
        return DepthPrediction(
            depth=prediction.depth,
            confidence=prediction.confidence,
            intrinsics=prediction.intrinsics,
            extrinsics=extrinsics,
            processed_images=prediction.processed_images,
            backend="fake-da3-bounded-pose-island",
        )


def _bundle(observations: ObservationSet) -> CameraBundle:
    count = len(observations.images)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 4.0
    intrinsics[:, 1, 2] = 4.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    for index in range(count):
        angle = 2.0 * np.pi * index / count
        extrinsics[index, 0, 3] = np.cos(angle)
        extrinsics[index, 1, 3] = np.sin(angle)
    return CameraBundle(
        image_names=tuple(item.relative_path for item in observations.images),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        source="unit-test-metric-rig",
        scale_status="known",
        world_units="centimetre",
        world_units_to_mm=10.0,
        details={"rig": "synthetic"},
    )


def test_geometry_passes_external_cameras_and_scale_to_fusion(
    sample_case: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = sample_case / "views"
    observations = load_observations(input_dir)
    bundle = _bundle(observations)
    bundle_path = tmp_path / "cameras.npz"
    bundle.save(bundle_path)
    monkeypatch.setattr("da3_cad.geometry_pipeline.Da3Backend", _FakeDa3Backend)
    config = AppConfig(
        profile="research",
        device="cpu",
        depth_backend="da3-base",
        da3=Da3Config(checkpoint="base"),
    )

    result = run_geometry(
        input_dir,
        tmp_path / "geometry",
        config,
        accepted_noncommercial=False,
        camera_bundle_path=bundle_path,
    )

    np.testing.assert_array_equal(_FakeDa3Backend.received_intrinsics, bundle.intrinsics)
    np.testing.assert_array_equal(_FakeDa3Backend.received_extrinsics, bundle.extrinsics)
    assert _FakeDa3Backend.received_align is True
    assert result.cloud.scale.status == "known"
    assert result.cloud.scale.world_units_to_mm == pytest.approx(10.0)
    assert result.report["camera_conditioning"]["source"] == "unit-test-metric-rig"
    assert result.report["pose_admission"]["status"] == "bypassed-external-cameras"
    assert (result.output_dir / "geometry_report.json").is_file()
    assert not (
        result.output_dir / "artefacts" / "camera_prediction_before_pose_admission.npz"
    ).exists()
    with np.load(result.output_dir / "artefacts" / "camera_prediction.npz") as saved:
        assert "processed_images" in saved.files
        np.testing.assert_array_equal(
            saved["processed_images"],
            np.stack(result.prediction.processed_images),
        )
    assert result.report["camera_conditioning"]["sha256"]


def test_geometry_rejects_pose_island_and_preserves_pre_admission_prediction(
    sample_case: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("da3_cad.geometry_pipeline.Da3Backend", _FakeUnposedDa3Backend)
    config = AppConfig(
        profile="research",
        device="cpu",
        depth_backend="da3-base",
        da3=Da3Config(checkpoint="base"),
    )

    result = run_geometry(
        sample_case / "views",
        tmp_path / "geometry-pose-island",
        config,
        accepted_noncommercial=False,
    )

    assert result.report["pose_admission"]["status"] == "rejected-outliers"
    assert result.report["pose_admission"]["rejected_input_views"] == [3]
    assert result.prediction.depth.shape[0] == 3
    before_path = result.output_dir / "artefacts" / "camera_prediction_before_pose_admission.npz"
    assert before_path.is_file()
    with np.load(before_path) as before:
        assert before["depth"].shape[0] == 4
        np.testing.assert_array_equal(before["local_to_input_view"], np.arange(4))
    with np.load(result.output_dir / "artefacts" / "camera_prediction.npz") as after:
        assert after["depth"].shape[0] == 3


def test_geometry_refines_bounded_pose_island_and_retains_all_views(
    sample_case: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "da3_cad.geometry_pipeline.Da3Backend",
        _FakeBoundedPoseIslandDa3Backend,
    )
    config = AppConfig(
        profile="research",
        device="cpu",
        depth_backend="da3-base",
        da3=Da3Config(checkpoint="base"),
    )

    result = run_geometry(
        sample_case / "views",
        tmp_path / "geometry-refined-pose-island",
        config,
        accepted_noncommercial=False,
    )

    report = result.report["pose_admission"]
    assert report["status"] == "all-consistent-after-refinement"
    assert report["initial_rejected_input_views"] == [3]
    assert report["refined_input_views"] == [3]
    assert report["rejected_input_views"] == []
    assert result.prediction.depth.shape[0] == 4
    before_path = result.output_dir / "artefacts" / "camera_prediction_before_pose_admission.npz"
    assert before_path.is_file()
    with np.load(result.output_dir / "artefacts" / "pose_admission_samples.npz") as samples:
        assert samples["points_before_refinement"].shape == (4 * 16, 3)
        assert samples["points_after_refinement"].shape == (4 * 16, 3)
        np.testing.assert_array_equal(samples["refined_local_view_indices"], [3])
    with np.load(result.output_dir / "artefacts" / "camera_prediction.npz") as after:
        assert after["depth"].shape[0] == 4
        assert abs(float(after["extrinsics"][-1, 0, 3])) < 1e-6
