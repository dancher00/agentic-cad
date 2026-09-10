from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from da3_cad.ray_sections import RayBundle


def test_ray_bundle_rejects_reflected_camera() -> None:
    extrinsics = np.tile(np.eye(4), (3, 1, 1))
    extrinsics[0, 0, 0] = -1
    bundle = RayBundle(
        np.ones((3, 8, 8), bool),
        np.ones((3, 8, 8)),
        np.tile(np.eye(3), (3, 1, 1)),
        extrinsics,
        ("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="right-handed"):
        bundle.validate()


def test_bundle_roundtrip_and_view_separation(tmp_path: Path) -> None:
    bundle = RayBundle(
        np.ones((4, 8, 8), bool),
        np.ones((4, 8, 8)),
        np.tile(np.eye(3), (4, 1, 1)),
        np.tile(np.eye(4), (4, 1, 1)),
        ("a", "b", "c", "heldout"),
    )
    bundle.save(tmp_path / "rays.npz")
    fit = RayBundle.load(tmp_path / "rays.npz").subset(np.arange(3))
    fit.validate()
    assert "heldout" not in fit.names
    assert fit.depths.shape[0] == 3


def test_dp_matches_exhaustive_interval_search() -> None:
    pytest.importorskip("torch")
    from da3_cad.ray_sections import partition

    rng = np.random.default_rng(123)
    volume = rng.random((6, 4, 4)) > 0.35
    penalty = 0.03
    sections, objective = partition(volume, 0, maximum_sections=3, penalty=penalty)
    best = float("inf")
    for count in range(1, 4):
        for breaks in combinations(range(1, 6), count - 1):
            boundaries = (0, *breaks, 6)
            error = 0
            for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
                counts = volume[a:b].sum(0)
                error += np.minimum(counts, b - a - counts).sum()
            best = min(best, error / volume.sum() + penalty * count)
    assert objective == pytest.approx(best, abs=1e-6)
    assert 1 <= len(sections) <= 3


def test_section_export_preserves_through_hole_and_world_rotation() -> None:
    pytest.importorskip("cv2")
    import cadquery as cq
    from scipy.spatial.transform import Rotation

    from da3_cad.ray_sections import compile_sections

    profile = np.zeros((24, 24), bool)
    profile[2:22, 2:22] = True
    profile[8:16, 8:16] = False
    frame = Rotation.from_euler("xyz", [21, 34, -17], degrees=True).as_matrix()
    result, source, _ = compile_sections([(2, 12, profile)], 2, frame, np.zeros(3), 1.0)
    shape = result.val()
    assert isinstance(shape, cq.Shape)
    assert shape.isValid() and len(shape.Solids()) == 1
    center = frame @ np.array([12.0, 12.0, 7.0])
    assert not shape.isInside(cq.Vector(*center), 1e-6)
    corner = frame @ np.array([4.0, 4.0, 7.0])
    assert shape.isInside(cq.Vector(*corner), 1e-6)
    assert "reference" not in source


def test_free_space_does_not_erase_space_behind_hit() -> None:
    pytest.importorskip("torch")
    from da3_cad.ray_sections import carve

    masks = np.ones((3, 16, 16), bool)
    depths = np.full((3, 16, 16), 2.0, np.float32)
    k = np.array([[8, 0, 8], [0, 8, 8], [0, 0, 1.0]])
    bundle = RayBundle(
        masks, depths, np.tile(k, (3, 1, 1)), np.tile(np.eye(4), (3, 1, 1)), ("a", "b", "c")
    )
    points = np.array([[x, y, z] for x in [-0.4, 0.4] for y in [-0.4, 0.4] for z in [1.0, 3.0]])
    volume, lower, step, _ = carve(bundle, points, np.eye(3), resolution=32, device="cpu")

    def occupied(z: float) -> bool:
        index = np.floor((np.array([0, 0, z]) - lower) / step).astype(int)
        return bool(volume[tuple(index)])

    assert not occupied(1.5)
    assert occupied(2.5)


def test_budget_search_recovers_when_best_partition_is_not_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from da3_cad import ray_section_search as search

    bundle = RayBundle(
        np.ones((3, 8, 8), bool),
        np.ones((3, 8, 8)),
        np.tile(np.eye(3), (3, 1, 1)),
        np.tile(np.eye(4), (3, 1, 1)),
        ("a", "b", "c"),
    )
    volume = np.zeros((16, 16, 16), bool)
    volume[2:8, 2:12, 2:12] = True
    volume[8:14, 5:14, 5:14] = True
    monkeypatch.setattr(search, "observed_points", lambda _: np.ones((32, 3)))
    monkeypatch.setattr(search, "evidence_frames", lambda _: [("world", np.eye(3))])
    monkeypatch.setattr(search, "carve", lambda *a, **kw: (volume, np.zeros(3), 1.0, {}))

    def kernel_oracle(payload: object, output: Path, timeout: float) -> dict[str, object]:
        del timeout
        if len(payload[0]) > 1:
            raise ValueError("controlled kernel rejection")
        (output / "candidate.step").write_text("controlled test fixture")
        return {"extrusions": 1, "faces": 6, "edges": 12, "volume": 1.0}

    monkeypatch.setattr(search, "_bounded_compile", kernel_oracle)
    report = search.reconstruct_sections(
        bundle, tmp_path / "run", resolution=16, maximum_sections=2, device="cpu"
    )
    assert report["kernel_valid"]
    assert report["sections"] == 1
    assert report["failed_compilations"]
