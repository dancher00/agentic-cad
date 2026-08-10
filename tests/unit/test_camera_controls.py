from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.benchmark.camera_controls import (
    load_renderer_camera_batch,
    pairwise_pose_diagnostics,
)
from da3_cad.benchmark.cameras import master_schedule
from da3_cad.observations import load_observations


def test_renderer_batch_requires_exact_frozen_image_hashes(tmp_path: Path) -> None:
    render = tmp_path / "render"
    views = tmp_path / "views"
    render.mkdir()
    views.mkdir()
    camera = master_schedule(image_size=504)[0]
    image_path = views / "view_000.png"
    Image.new("RGB", (504, 504), (10, 20, 30)).save(image_path)
    observations = load_observations(views)
    record = camera.as_dict()
    record["image_sha256"] = observations.images[0].sha256
    (render / "render_manifest.json").write_text(
        json.dumps({"camera_schedule": [record]}),
        encoding="utf-8",
    )

    batch = load_renderer_camera_batch(render, observations)

    assert batch.intrinsics.shape == (1, 3, 3)
    assert batch.extrinsics.shape == (1, 4, 4)
    assert np.linalg.det(batch.extrinsics[0, :3, :3]) == pytest.approx(1.0)

    record["image_sha256"] = "0" * 64
    (render / "render_manifest.json").write_text(
        json.dumps({"camera_schedule": [record]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_renderer_camera_batch(render, observations)


def test_pairwise_pose_diagnostics_are_world_frame_invariant() -> None:
    cameras = master_schedule()[:4]
    gt = []
    for camera in cameras:
        value = np.diag([1.0, -1.0, 1.0, 1.0]) @ camera.world_to_camera
        gt.append(value)
    gt_array = np.stack(gt)
    world_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    world_transform = np.eye(4)
    world_transform[:3, :3] = world_rotation
    predicted = gt_array @ world_transform
    predicted[:, :3, 3] *= 0.5

    diagnostics = pairwise_pose_diagnostics(predicted, gt_array)

    assert diagnostics["identifiable"] is True
    assert diagnostics["relative_rotation_error_degrees"]["max"] == pytest.approx(0.0)
    assert diagnostics["predicted_to_gt_baseline_scale"] == pytest.approx(2.0)
    assert diagnostics["baseline_rmse_over_gt_median"] == pytest.approx(0.0, abs=1e-12)


def test_single_camera_pose_scale_is_explicitly_unidentifiable() -> None:
    extrinsic = np.eye(4, dtype=np.float32)[None]

    diagnostics = pairwise_pose_diagnostics(extrinsic, extrinsic)

    assert diagnostics == {
        "identifiable": False,
        "views": 1,
        "reason": "one camera has no relative pose or baseline scale",
    }
