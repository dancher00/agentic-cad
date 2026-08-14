from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh

from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.observations import load_observations
from da3_cad.sample import build_sample_case, build_typical_parts_benchmark


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_sample_case_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_sample_case(first)
    build_sample_case(second)

    assert _tree_digest(first) == _tree_digest(second)


def test_typical_parts_benchmark_is_deterministic_and_valid(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_typical_parts_benchmark(first)
    second_manifest = build_typical_parts_benchmark(second)

    assert _tree_digest(first) == _tree_digest(second)
    assert [case["id"] for case in first_manifest["cases"]] == [
        "block",
        "flange",
        "l_bracket",
    ]
    assert all(
        case["expected_operation_family"] == "sketch-extrusion" for case in first_manifest["cases"]
    )
    assert first_manifest == second_manifest
    for case in first_manifest["cases"]:
        case_root = first / str(case["id"])
        assert len(tuple((case_root / "views").glob("*.png"))) == 8
        observations = load_observations(case_root / "views")
        cameras = load_camera_bundle(case_root / "cameras.npz", observations)
        assert cameras.scale.status == "known"
        assert cameras.scale.world_units_to_mm == 1.0
        assert cameras.source == "synthetic-fixed-pinhole-v1"
        assert np.allclose(cameras.intrinsics[:, 0, 0], 180.0)
        assert np.allclose(np.linalg.det(cameras.extrinsics[:, :3, :3]), 1.0)
        assert (case_root / "gt.step").stat().st_size > 1000
        mesh = trimesh.load_mesh(case_root / "gt.stl", process=True)
        assert mesh.is_watertight
        assert mesh.volume > 0.0
