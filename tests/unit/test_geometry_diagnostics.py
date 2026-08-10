from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from da3_cad.geometry.diagnostics import write_geometry_diagnostics
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.models import DepthPrediction


def test_geometry_diagnostics_write_inspectable_artifacts(tmp_path: Path) -> None:
    prediction = DepthPrediction(
        depth=np.ones((1, 2, 2), dtype=np.float32),
        confidence=np.array([[[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None],
        processed_images=(np.full((2, 2, 3), 127, dtype=np.uint8),),
        backend="test",
    )
    masks = np.ones((1, 2, 2), dtype=np.bool_)
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="unit-test",
        confidence_percentile=None,
    )

    write_geometry_diagnostics(tmp_path, prediction, masks, cloud)

    expected = {
        "depth_000.png",
        "confidence_000.png",
        "mask_000.png",
        "mask_overlay_000.png",
        "fused_cloud.npz",
        "fused_cloud.ply",
        "fusion_report.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    report = json.loads((tmp_path / "fusion_report.json").read_text())
    assert report["fusion"]["fused_points"] == 4
    assert report["scale"]["status"] == "unresolved"
