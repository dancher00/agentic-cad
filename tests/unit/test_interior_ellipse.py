from __future__ import annotations

import cv2
import numpy as np

from da3_cad.config import InteriorEllipseConfig
from da3_cad.geometry.interior_ellipse import (
    detect_concentric_interior,
    detect_interior_ellipses,
)
from da3_cad.models import DepthPrediction


def _ellipse_observations(
    *,
    depth_discontinuity: bool,
    outer_rim: bool = True,
    view_count: int = 3,
) -> tuple[DepthPrediction, np.ndarray]:
    height = width = 160
    masks = np.zeros((view_count, height, width), dtype=np.bool_)
    images: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    for view_index in range(view_count):
        mask = masks[view_index]
        mask[15:145, 15:145] = True
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[mask] = 220
        center = (80 + (view_index % 2), 80)
        if outer_rim:
            cv2.ellipse(image, center, (45, 28), 25.0, 0.0, 360.0, (35, 35, 35), 2)
        cv2.ellipse(image, center, (18, 12), 25.0, 0.0, 360.0, (20, 20, 20), -1)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        if depth_discontinuity:
            inner = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(inner, center, (17, 11), 25.0, 0.0, 360.0, 1, -1)
            depth[inner.astype(bool)] += 0.08
        images.append(image)
        depths.append(depth)
    intrinsics = np.repeat(
        np.asarray([[[120.0, 0.0, 80.0], [0.0, 120.0, 80.0], [0.0, 0.0, 1.0]]]),
        view_count,
        axis=0,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], view_count, axis=0)
    depth_values = np.stack(depths)
    return (
        DepthPrediction(
            depth=depth_values,
            confidence=np.ones_like(depth_values),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(images),
            backend="synthetic-interior-ellipse",
        ),
        masks,
    )


def test_rgb_ellipse_with_nonplanar_depth_is_admitted() -> None:
    prediction, masks = _ellipse_observations(depth_discontinuity=True)

    evidence = detect_interior_ellipses(
        prediction,
        masks,
        InteriorEllipseConfig(),
    )

    assert tuple(item.view_index for item in evidence) == (0, 1, 2)
    assert all(item.angular_coverage >= 0.75 for item in evidence)
    assert all(item.depth_plane_excess_fraction > 0.01 for item in evidence)


def test_painted_circle_on_planar_depth_cannot_change_topology() -> None:
    prediction, masks = _ellipse_observations(depth_discontinuity=False)

    assert detect_interior_ellipses(prediction, masks, InteriorEllipseConfig()) == ()
    assert detect_concentric_interior(prediction, masks, InteriorEllipseConfig()) is None


def test_repeated_concentric_rims_measure_axial_cavity_ratio() -> None:
    prediction, masks = _ellipse_observations(depth_discontinuity=True)

    evidence = detect_concentric_interior(
        prediction,
        masks,
        InteriorEllipseConfig(),
    )

    assert evidence is not None
    assert evidence.supporting_views == (0, 1, 2)
    np.testing.assert_allclose(evidence.radius_ratio, 0.41, atol=0.05)
