from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.observations import load_observations
from da3_cad.segmentation.sam2_box import (
    _box_refinement_points,
    _select_prompt_instance,
    load_box_prompts,
)


def _images(root: Path) -> Path:
    root.mkdir()
    for index in range(3):
        Image.fromarray(np.full((50, 100, 3), 50 + index, dtype=np.uint8)).save(
            root / f"view_{index}.png"
        )
    return root


def test_box_prompt_loader_converts_normalized_xyxy_and_clips_to_images(
    tmp_path: Path,
) -> None:
    images = _images(tmp_path / "images")
    boxes = tmp_path / "boxes.json"
    boxes.write_text(
        json.dumps(
            {
                "coordinate_space": "normalized-exif-corrected-xyxy",
                "views": [
                    {
                        "image": f"view_{index}.png",
                        "xyxy": [-0.1, 0.2, 0.8, 1.1],
                        "frame_index": index * 4,
                    }
                    for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    observations = load_observations(images)

    prompts = load_box_prompts(boxes, observations.images)

    assert prompts[0].xyxy_pixels == (0.0, 10.0, 80.0, 50.0)
    assert prompts[0].xyxy_normalized == (-0.1, 0.2, 0.8, 1.1)
    assert prompts[0].metadata["clipped_to_source"] is True
    assert prompts[2].metadata["frame_index"] == 8


def test_box_prompt_loader_requires_exact_image_set(tmp_path: Path) -> None:
    images = _images(tmp_path / "images")
    boxes = tmp_path / "boxes.json"
    boxes.write_text(
        json.dumps(
            {
                "coordinate_space": "normalized-exif-corrected-xyxy",
                "views": [{"image": "view_0.png", "xyxy": [0.1, 0.1, 0.9, 0.9]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompts and images differ"):
        load_box_prompts(boxes, load_observations(images).images)


def test_prompt_instance_is_box_bounded_and_keeps_one_component() -> None:
    mask = np.zeros((80, 100), dtype=np.bool_)
    mask[20:60, 30:70] = True
    mask[0:10, 0:10] = True
    mask[25:35, 75:90] = True
    mask[35:45, 35:42] = False  # A real hole must remain a hole.

    selected, report = _select_prompt_instance(mask, (25.0, 15.0, 72.0, 65.0))

    assert selected[30, 40]
    assert not selected[5, 5]
    assert not selected[30, 80]
    assert not selected[40, 38]
    assert report["connected_foreground_components"] == 1
    assert report["component_selection"] == "component-containing-box-center"


def test_prompt_instance_uses_largest_component_when_box_centre_is_background() -> None:
    mask = np.zeros((50, 50), dtype=np.bool_)
    mask[10:25, 8:20] = True
    mask[30:35, 35:40] = True

    selected, report = _select_prompt_instance(mask, (5.0, 5.0, 45.0, 45.0))

    assert int(selected.sum()) == 15 * 12
    assert report["connected_foreground_components"] == 2
    assert report["component_selection"] == "largest-component"


def test_box_refinement_points_have_explicit_labels_and_inset() -> None:
    points, labels = _box_refinement_points((10.0, 20.0, 110.0, 220.0))

    np.testing.assert_allclose(
        points,
        np.asarray(((60, 120), (18, 36), (102, 36), (18, 204), (102, 204))),
    )
    np.testing.assert_array_equal(labels, np.asarray((1, 0, 0, 0, 0)))
