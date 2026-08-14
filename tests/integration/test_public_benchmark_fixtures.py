from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.benchmark.public_cases import PUBLIC_CASES, build_public_release_benchmark
from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.observations import load_observations


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_public_benchmark_fixtures_are_deterministic_and_valid(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_public_release_benchmark(first)
    second_manifest = build_public_release_benchmark(second)

    assert first_manifest == second_manifest
    assert _tree_digest(first) == _tree_digest(second)
    assert len(first_manifest["cases"]) == 10
    assert len(PUBLIC_CASES) == 10
    assert sum(int(case["input_views"]) for case in first_manifest["cases"]) == 120
    assert {case["expected_operation_family"] for case in first_manifest["cases"]} == {
        "extrude",
        "extrude+cut",
        "extrude+2cut",
        "revolve",
    }
    for case in first_manifest["cases"]:
        root = first / str(case["id"])
        observations = load_observations(root / "views")
        cameras = load_camera_bundle(root / "cameras.npz", observations)
        assert cameras.scale.status == "known"
        assert cameras.scale.world_units_to_mm == 1.0
        assert len(observations.images) == 12
        assert len(tuple((root / "masks").glob("*.png"))) == 12
        assert np.allclose(cameras.intrinsics[:, 0, 0], 300.0)
        for mask_path in (root / "masks").glob("*.png"):
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            assert 0.01 < float(mask.mean()) < 0.75
        mesh = trimesh.load_mesh(root / "gt.stl", process=True)
        assert mesh.is_watertight
        assert mesh.volume > 0.0
