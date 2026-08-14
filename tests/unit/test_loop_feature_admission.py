from __future__ import annotations

import cv2
import numpy as np
import pytest

from da3_cad.backends.axial_shell_loop import LoopViewEvidence, _ViewAnalysis
from da3_cad.config import AxialShellLoopConfig, LoopFeatureAdmissionConfig
from da3_cad.geometry.loop_feature_admission import (
    _largest_pairwise_component,
    admit_loop_feature_geometry,
)
from da3_cad.models import DepthPrediction


def _prediction() -> tuple[DepthPrediction, np.ndarray]:
    count, height, width = 4, 64, 64
    masks = np.ones((count, height, width), dtype=np.bool_)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    extrinsics[1, 0, 3] = 0.001
    extrinsics[2, 1, 3] = -0.001
    extrinsics[3, 0, 3] = 2.0
    images = tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count))
    return (
        DepthPrediction(
            depth=np.ones((count, height, width), dtype=np.float32),
            confidence=np.ones((count, height, width), dtype=np.float32),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=images,
            backend="loop-feature-test",
        ),
        masks,
    )


def _analysis(view_index: int) -> _ViewAnalysis:
    contour = np.asarray(
        [[[38, 20]], [[56, 20]], [[56, 44]], [[38, 44]]],
        dtype=np.int32,
    )
    body = np.zeros((64, 64), dtype=np.bool_)
    body[8:56, 8:42] = True
    evidence = LoopViewEvidence(
        view_index=view_index,
        body_bbox=(8, 8, 42, 56),
        body_aspect=34.0 / 48.0,
        body_height_over_diameter=48.0 / 34.0,
        side_like=True,
        loop_supported=True,
        hole_bbox=tuple(int(value) for value in cv2.boundingRect(contour)),
        hole_center=(47.0, 32.0),
        hole_area_fraction=0.1,
        hole_center_offset_fraction=0.6,
    )
    return _ViewAnalysis(evidence=evidence, body=body, hole_contour=contour)


def test_loop_feature_admission_keeps_largest_registered_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, masks = _prediction()
    monkeypatch.setattr(
        "da3_cad.geometry.loop_feature_admission._analyze_view",
        lambda index, _mask, _config: _analysis(index),
    )

    result = admit_loop_feature_geometry(
        prediction,
        masks,
        config=LoopFeatureAdmissionConfig(surface_distance_fraction=0.03),
        loop_config=AxialShellLoopConfig(),
    )

    assert result.report["status"] == "admitted-consistent-component"
    assert result.report["admitted_loop_geometry_views"] == [0, 1, 2]
    assert result.report["suppressed_loop_geometry_views"] == [3]
    assert np.array_equal(result.geometry_masks[:3], masks[:3])
    assert int(result.geometry_masks[3].sum()) < int(masks[3].sum())
    assert result.geometry_masks[3, 20, 55] == 0
    assert result.geometry_masks[3, 20, 20] == 1


def test_loop_feature_admission_bypasses_external_cameras() -> None:
    prediction, masks = _prediction()

    result = admit_loop_feature_geometry(
        prediction,
        masks,
        config=LoopFeatureAdmissionConfig(),
        loop_config=AxialShellLoopConfig(),
        external_cameras=True,
    )

    assert result.report["status"] == "bypassed-external-cameras"
    np.testing.assert_array_equal(result.geometry_masks, masks)


def test_pairwise_component_does_not_admit_transitive_bridge() -> None:
    adjacency = {
        0: {0, 1},
        1: {0, 1, 2},
        2: {1, 2},
    }

    assert _largest_pairwise_component(adjacency) == (0, 1)
