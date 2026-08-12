from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
from da3_cad.observations import load_observations


def _camera_arrays(count: int) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 700.0
    intrinsics[:, 1, 1] = 710.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 48.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    positions = np.asarray(
        [(float(index), float(index % 2), 0.25 * index) for index in range(count)],
        dtype=np.float32,
    )
    extrinsics[:, :3, 3] = -positions
    return intrinsics, extrinsics


def test_camera_bundle_roundtrip_is_name_ordered_and_non_pickled(
    sample_case: Path,
    tmp_path: Path,
) -> None:
    observations = load_observations(sample_case / "views")
    observation_names = tuple(item.relative_path for item in observations.images)
    stored_names = tuple(reversed(observation_names))
    intrinsics, extrinsics = _camera_arrays(len(stored_names))
    bundle = CameraBundle(
        image_names=stored_names,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        source="unit-test-calibration",
        scale_status="known",
        world_units="metre",
        world_units_to_mm=1000.0,
        details={"board": "synthetic"},
    )
    path = tmp_path / "cameras.npz"

    bundle.save(path)
    loaded = load_camera_bundle(path, observations)

    assert loaded.image_names == observation_names
    expected_indices = [stored_names.index(name) for name in observation_names]
    np.testing.assert_array_equal(loaded.intrinsics, intrinsics[expected_indices])
    np.testing.assert_array_equal(loaded.extrinsics, extrinsics[expected_indices])
    assert loaded.scale.status == "known"
    assert loaded.scale.world_units_to_mm == pytest.approx(1000.0)
    assert loaded.scale.source == "unit-test-calibration"
    assert path.with_suffix(".json").is_file()
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "image_names",
            "intrinsics",
            "extrinsics",
            "metadata",
        }


def test_camera_bundle_rejects_name_and_scale_ambiguity() -> None:
    intrinsics, extrinsics = _camera_arrays(3)
    bundle = CameraBundle(
        image_names=("a.png", "b.png", "c.png"),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        source="unit-test",
    )

    with pytest.raises(ValueError, match="names differ"):
        bundle.reordered(("a.png", "b.png", "missing.png"))
    with pytest.raises(ValueError, match="known camera scale"):
        CameraBundle(
            image_names=bundle.image_names,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            source="unit-test",
            scale_status="known",
        )
    with pytest.raises(ValueError, match="orthonormal"):
        invalid = extrinsics.copy()
        invalid[0, 0, 0] = 2.0
        CameraBundle(
            image_names=bundle.image_names,
            intrinsics=intrinsics,
            extrinsics=invalid,
            source="unit-test",
        )
