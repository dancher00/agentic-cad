from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.multiview_depth_alignment import (
    align_multiview_depths,
    select_depth_hypothesis,
)
from da3_cad.models import BoolArray, DepthPrediction


def _sloped_plane_prediction() -> tuple[
    DepthPrediction,
    BoolArray,
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    size = 40
    focal = 45.0
    principal = (size - 1.0) / 2.0
    intrinsic = np.asarray(
        [[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.28, 0.02, 0.0],
            [-0.24, 0.18, 0.04],
            [0.04, -0.26, -0.03],
        ],
        dtype=np.float64,
    )
    expected_scales = np.asarray([1.0, 1.25, 0.8, 1.1], dtype=np.float64)
    expected_shifts = np.asarray([0.0, -0.14, 0.11, 0.06], dtype=np.float64)
    normal = np.asarray([0.38, -0.24, 1.0], dtype=np.float64)
    plane_offset = 3.0
    yy, xx = np.indices((size, size), dtype=np.float64)
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    camera_rays = pixels @ np.linalg.inv(intrinsic).T

    depths: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
    extrinsics: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
    for center, scale, shift in zip(
        camera_centers,
        expected_scales,
        expected_shifts,
        strict=True,
    ):
        true_depth = (plane_offset - float(normal @ center)) / (camera_rays @ normal)
        depths.append((true_depth - shift) / scale)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, 3] = -center
        extrinsics.append(world_to_camera)

    confidence = 1.0 + 0.4 * xx / size + 0.2 * yy / size
    masks = np.zeros((len(depths), size, size), dtype=np.bool_)
    masks[:, 2:-2, 2:-2] = True
    prediction = DepthPrediction(
        depth=np.stack(depths).astype(np.float32),
        confidence=np.repeat(confidence[None, :, :], len(depths), axis=0).astype(np.float32),
        intrinsics=np.repeat(intrinsic[None, :, :], len(depths), axis=0).astype(np.float32),
        extrinsics=np.stack(extrinsics).astype(np.float32),
        processed_images=tuple(np.zeros((size, size, 3), dtype=np.uint8) for _ in depths),
        backend="synthetic-sloped-plane",
    )
    return prediction, masks, expected_scales, expected_shifts


@pytest.mark.parametrize("criterion", ["projected-local-depth", "fixed-local-plane"])
def test_alignment_recovers_known_affines_deterministically(criterion: str) -> None:
    prediction, masks, expected_scales, expected_shifts = _sloped_plane_prediction()

    first = align_multiview_depths(
        prediction,
        masks,
        criterion=criterion,  # type: ignore[arg-type]
        seed=1123,
    )
    second = align_multiview_depths(
        prediction,
        masks,
        criterion=criterion,  # type: ignore[arg-type]
        seed=1123,
    )

    measured_scales = np.asarray([value.scale for value in first.parameters])
    measured_shifts = np.asarray([value.shift_b for value in first.parameters])
    assert measured_scales == pytest.approx(expected_scales, abs=0.16)
    assert measured_shifts == pytest.approx(expected_shifts, abs=0.16)
    assert first.parameters[0].fixed_reference is True
    assert first.parameters[0].scale == 1.0
    assert first.parameters[0].shift_b == 0.0
    assert first.report["gt_blind"] is True
    assert first.report["gt_or_mesh_argument_available"] is False
    assert first.report["loss"]["final"] < first.report["loss"]["initial"]  # type: ignore[index]
    assert first.report["parameter_sha256"] == second.report["parameter_sha256"]
    assert np.array_equal(first.prediction.depth, second.prediction.depth)
    assert np.array_equal(first.prediction.intrinsics, prediction.intrinsics)
    assert np.array_equal(first.prediction.extrinsics, prediction.extrinsics)
    assert np.array_equal(first.prediction.depth[~masks], prediction.depth[~masks])


def test_single_view_is_an_explicit_identity_noop() -> None:
    prediction, masks, _, _ = _sloped_plane_prediction()
    single = DepthPrediction(
        depth=prediction.depth[:1],
        confidence=prediction.confidence[:1] if prediction.confidence is not None else None,
        intrinsics=prediction.intrinsics[:1],
        extrinsics=prediction.extrinsics[:1],
        processed_images=prediction.processed_images[:1],
        backend=prediction.backend,
    )
    result = align_multiview_depths(
        single,
        masks[:1],
        criterion="projected-local-depth",
        seed=3,
    )

    assert result.report["status"] == "not-applicable-single-view"
    assert result.report["depth_changed"] is False
    assert np.array_equal(result.prediction.depth, single.depth)

    hypothesis = select_depth_hypothesis(
        single,
        masks[:1],
        criterion="projected-local-depth",
        selection="auto",
        seed=3,
    )
    assert hypothesis.selected == "identity"
    assert hypothesis.prediction is single
    assert result.report["input_depth_sha256"] == result.report["output_depth_sha256"]


def test_auto_hypothesis_accepts_safe_observation_improvement() -> None:
    prediction, masks, _, _ = _sloped_plane_prediction()

    result = select_depth_hypothesis(
        prediction,
        masks,
        criterion="fixed-local-plane",
        selection="auto",
        seed=1123,
        maximum_scale_ratio=1.5,
        maximum_center_ratio_deviation=0.25,
    )

    assert result.selected == "aligned"
    assert result.prediction is result.aligned_prediction
    assert result.identity_prediction is prediction
    assert result.report["gt_blind"] is True
    assert result.report["gt_or_mesh_argument_available"] is False


def test_auto_hypothesis_keeps_identity_when_safety_gate_fails() -> None:
    prediction, masks, _, _ = _sloped_plane_prediction()

    result = select_depth_hypothesis(
        prediction,
        masks,
        criterion="fixed-local-plane",
        selection="auto",
        seed=1123,
        maximum_scale_ratio=1.01,
    )

    assert result.selected == "identity"
    assert result.prediction is prediction
    aligned = result.report["candidate_evidence"]["aligned"]  # type: ignore[index]
    assert aligned["checks"]["scale_ratio_at_most_limit"] is False  # type: ignore[index]
