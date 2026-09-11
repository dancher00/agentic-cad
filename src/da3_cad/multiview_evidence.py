"""Cross-view diagnostics for jointly predicted cameras, masks and relative depth."""

from __future__ import annotations

from typing import Any

import numpy as np


def camera_consistency(depth: Any, masks: Any, intrinsics: Any, extrinsics: Any) -> dict[str, Any]:
    """Reproject object depth into other views, ignoring depth-occluded samples.

    This measures internal agreement of predictions, not calibrated camera accuracy.
    Low support is kept explicit rather than scored as a successful comparison.
    """
    count = len(depth)
    if any(len(value) != count for value in (masks, intrinsics, extrinsics)):
        raise ValueError("Camera, depth and mask view counts must agree")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("Non-finite estimated cameras")
    clouds = []
    centers = []
    for i in range(count):
        rotation, translation = extrinsics[i, :3, :3], extrinsics[i, :3, 3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=0.02) or not np.isclose(
            np.linalg.det(rotation), 1, atol=0.02
        ):
            raise ValueError("Estimated camera rotation is not a proper rotation")
        ys, xs = np.where(masks[i] & np.isfinite(depth[i]) & (depth[i] > 0))
        if not len(xs):
            raise ValueError("No finite object depth for camera consistency")
        selection = np.linspace(0, len(xs) - 1, min(1024, len(xs))).astype(int)
        xs, ys = xs[selection], ys[selection]
        camera = np.column_stack([xs, ys, np.ones(len(xs))]) @ np.linalg.inv(intrinsics[i]).T
        camera *= depth[i, ys, xs, None]
        clouds.append((camera - translation) @ rotation)
        centers.append(-translation @ rotation)
    object_center = np.median(np.concatenate(clouds), axis=0)
    directions = np.asarray(centers) - object_center
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    pairs = []
    for source, cloud in enumerate(clouds):
        for target in range(count):
            if source == target:
                continue
            ext = extrinsics[target]
            camera = cloud @ ext[:3, :3].T + ext[:3, 3]
            projected = camera @ intrinsics[target].T
            xy = np.rint(projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)).astype(int)
            h, w = depth[target].shape
            inside = (camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w)
            inside &= (xy[:, 1] >= 0) & (xy[:, 1] < h)
            x, y = xy[inside].T
            observed = depth[target, y, x]
            # Points behind the visible depth surface are occluded, not missing.
            visible = np.isfinite(observed) & (observed > 0)
            visible &= camera[inside, 2] <= observed * 1.1
            support = int(visible.sum())
            pairs.append(
                {
                    "source_view": source + 1,
                    "target_view": target + 1,
                    "visible_samples": support,
                    "mask_containment": float(masks[target, y[visible], x[visible]].mean())
                    if support >= 32
                    else None,
                }
            )
    return {
        "view_count": count,
        "estimated_max_view_separation_degrees": float(angles.max()),
        "pairs": pairs,
        "contract": "Predicted-camera consistency only; occluded samples excluded; no metric scale",
    }
