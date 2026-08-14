from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from da3_cad.geometry.cameras import (
    CameraBundle,
    _match_colmap_features,
    _resolve_colmap_device,
    _write_camera_recovery_report,
    load_camera_bundle,
)
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


class _FakeDevice:
    cpu = "cpu"
    cuda = "cuda"


class _FakeSequentialPairingOptions:
    overlap = 0
    quadratic_overlap = False
    loop_detection = True


class _FakeExhaustivePairingOptions:
    block_size = 50


class _FakePycolmap:
    Device = _FakeDevice
    SequentialPairingOptions = _FakeSequentialPairingOptions
    ExhaustivePairingOptions = _FakeExhaustivePairingOptions

    def __init__(self, *, has_cuda: bool) -> None:
        self.has_cuda = has_cuda
        self.calls: list[tuple[str, dict[str, object]]] = []

    def match_sequential(self, **kwargs: object) -> None:
        self.calls.append(("sequential", kwargs))

    def match_exhaustive(self, **kwargs: object) -> None:
        self.calls.append(("exhaustive", kwargs))


def test_colmap_device_auto_selects_available_backend_and_rejects_missing_cuda() -> None:
    cpu_pycolmap = _FakePycolmap(has_cuda=False)
    resolved, selected, has_cuda = _resolve_colmap_device(cpu_pycolmap, "auto")

    assert resolved == "cpu"
    assert selected == "cpu"
    assert has_cuda is False
    with pytest.raises(RuntimeError, match="no CUDA support"):
        _resolve_colmap_device(cpu_pycolmap, "cuda")

    cuda_pycolmap = _FakePycolmap(has_cuda=True)
    resolved, selected, has_cuda = _resolve_colmap_device(cuda_pycolmap, "auto")
    assert resolved == "cuda"
    assert selected == "cuda"
    assert has_cuda is True


def test_colmap_pairing_dispatches_sequential_and_exhaustive(tmp_path: Path) -> None:
    pycolmap = _FakePycolmap(has_cuda=False)
    matching_options = object()
    database = tmp_path / "database.db"

    sequential = _match_colmap_features(
        pycolmap,
        database=database,
        matching_options=matching_options,
        pairing="sequential",
        image_count=8,
        device="cpu",
    )
    exhaustive = _match_colmap_features(
        pycolmap,
        database=database,
        matching_options=matching_options,
        pairing="exhaustive",
        image_count=8,
        device="cpu",
    )

    assert sequential == "sequential-overlap-7-quadratic-no-loop-detection"
    assert exhaustive == "exhaustive-all-pairs-block-50"
    assert [name for name, _ in pycolmap.calls] == ["sequential", "exhaustive"]
    sequential_options = pycolmap.calls[0][1]["pairing_options"]
    assert isinstance(sequential_options, _FakeSequentialPairingOptions)
    assert sequential_options.overlap == 7
    assert sequential_options.quadratic_overlap is True
    assert sequential_options.loop_detection is False


def test_camera_recovery_report_is_strict_json_with_real_newline(tmp_path: Path) -> None:
    path = tmp_path / "camera_recovery.json"
    report: dict[str, object] = {"schema_version": "1.1", "status": "abstained"}

    _write_camera_recovery_report(path, report)

    raw = path.read_text(encoding="utf-8")
    assert json.loads(raw) == report
    assert raw.endswith("\n")
    assert not raw.endswith("\\n")
