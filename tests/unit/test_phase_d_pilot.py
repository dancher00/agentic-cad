from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from da3_cad.backends import cadrille as cadrille_module
from da3_cad.backends import da3 as da3_module
from da3_cad.benchmark.pilot import image_set_digest, json_digest, load_fused_cloud


def _write_fused_artifacts(root: Path, *, scale_status: str) -> None:
    artifact_root = root / "artefacts"
    artifact_root.mkdir(parents=True)
    np.savez(
        artifact_root / "fused_cloud.npz",
        points=np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float32),
        colors=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        confidence=np.asarray([0.7, 0.8], dtype=np.float32),
        view_indices=np.asarray([0, 1], dtype=np.int32),
        pixel_xy=np.asarray([[10, 11], [12, 13]], dtype=np.int32),
    )
    payload = {
        "fusion": {
            "confidence_percentile": 70.0,
            "confidence_thresholds": [0.5, 0.6],
            "mask_source": "rendered-object-mask",
            "require_confidence": True,
            "views": [
                {
                    "view_index": 0,
                    "pixels": 20,
                    "finite_positive_depth": 18,
                    "mask_selected": 10,
                    "confidence_selected": 8,
                    "fused": 8,
                },
                {
                    "view_index": 1,
                    "pixels": 20,
                    "finite_positive_depth": 19,
                    "mask_selected": 11,
                    "confidence_selected": 9,
                    "fused": 9,
                },
            ],
        },
        "scale": {
            "status": scale_status,
            "units": "normalized",
            "world_units_to_mm": None,
        },
    }
    (artifact_root / "fusion_report.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_pilot_digests_are_stable_and_content_sensitive(tmp_path: Path) -> None:
    assert json_digest({"b": 2, "a": 1}) == json_digest({"a": 1, "b": 2})
    images = tmp_path / "images"
    images.mkdir()
    (images / "view_01.png").write_bytes(b"one")
    first = image_set_digest(images)
    (images / "view_01.png").write_bytes(b"two")
    assert image_set_digest(images) != first


def test_load_fused_cloud_restores_contract_and_rejects_unknown_scale(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    _write_fused_artifacts(valid, scale_status="unresolved")
    cloud = load_fused_cloud(valid)
    assert cloud.points.shape == (2, 3)
    assert cloud.report.confidence_scope == "per-view"
    assert cloud.report.fused_points == 17
    assert cloud.scale.status == "unresolved"

    invalid = tmp_path / "invalid"
    _write_fused_artifacts(invalid, scale_status="invented")
    with pytest.raises(ValueError, match="scale status"):
        load_fused_cloud(invalid)


@pytest.mark.parametrize("module", [cadrille_module, da3_module])
def test_checkpoint_hash_cache_invalidates_on_file_stat(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    path = (tmp_path / "weights.bin").resolve()
    path.write_bytes(b"a")
    first_stat = path.stat()
    module._sha256_for_stat.cache_clear()
    first = module._sha256_for_stat(path, first_stat.st_size, first_stat.st_mtime_ns)

    path.write_bytes(b"changed-size")
    second_stat = path.stat()
    second = module._sha256_for_stat(path, second_stat.st_size, second_stat.st_mtime_ns)
    assert second != first
