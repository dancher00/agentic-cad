from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any


def _load_candidates() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_tless_multiview_case.py"
    return runpy.run_path(str(script))["_candidates"]


_candidates = _load_candidates()


def _write_scene(root: Path) -> None:
    camera = {
        str(index): {"cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]} for index in range(3)
    }
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ground_truth = {
        str(index): [
            {"obj_id": 4, "cam_R_m2c": identity, "cam_t_m2c": [index, 0.0, 1.0]},
            {"obj_id": 4, "cam_R_m2c": identity, "cam_t_m2c": [index, 1.0, 1.0]},
        ]
        for index in range(3)
    }
    information = {
        "0": [{"visib_fract": 0.95}, {"visib_fract": 0.91}],
        "1": [{"visib_fract": 0.94}, {"visib_fract": 0.20}],
        "2": [{"visib_fract": 0.93}, {"visib_fract": 0.92}],
    }
    for name, payload in (
        ("scene_camera.json", camera),
        ("scene_gt.json", ground_truth),
        ("scene_gt_info.json", information),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_candidates_keep_one_stable_instance_across_every_frame(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    views, selected, counts = _candidates(tmp_path, 4, 0.90)

    assert selected == 0
    assert [view.gt_index for view in views] == [0, 0, 0]
    assert [view.image_id for view in views] == [0, 1, 2]
    assert counts == {0: 3, 1: 3}


def test_candidates_honor_explicit_instance_index(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    views, selected, _ = _candidates(tmp_path, 4, 0.90, gt_index=1)

    assert selected == 1
    assert [view.gt_index for view in views] == [1, 1]
    assert [view.image_id for view in views] == [0, 2]
