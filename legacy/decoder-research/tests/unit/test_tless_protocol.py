from __future__ import annotations

import numpy as np
import trimesh

from da3_cad.benchmark.tless_protocol import (
    TlessViewCandidate,
    camera_direction_in_model_frame,
    nested_maxmin_views,
    repair_official_cad,
)
from da3_cad.evaluation.mesh import validate_mesh


def _view(frame: int, direction: tuple[float, float, float]) -> TlessViewCandidate:
    return TlessViewCandidate(1, frame, 0, 0.9, 0.1, 0.1, direction, "rgb", "mask")


def test_camera_direction_is_camera_center_in_model_frame() -> None:
    direction = camera_direction_in_model_frame(np.eye(3), [0.0, 0.0, 10.0])
    assert np.allclose(direction, [0.0, 0.0, -1.0])


def test_nested_maxmin_views_is_deterministic_and_spread() -> None:
    candidates = (
        _view(0, (1.0, 0.0, 0.0)),
        _view(1, (-1.0, 0.0, 0.0)),
        _view(2, (0.0, 1.0, 0.0)),
        _view(3, (0.0, -1.0, 0.0)),
    )
    first = nested_maxmin_views(candidates, count=4, object_id=1, seed=7)
    second = nested_maxmin_views(candidates, count=4, object_id=1, seed=7)
    assert first == second
    assert np.dot(first[0].direction, first[1].direction) == -1.0


def test_repair_fills_simple_hole() -> None:
    box = trimesh.creation.box()
    damaged = box.copy()
    damaged.update_faces(np.arange(len(damaged.faces) - 2))
    repaired, report = repair_official_cad(damaged)
    assert validate_mesh(repaired).valid
    assert report["repair_applied"] is True
