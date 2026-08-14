from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from da3_cad.objectron import objectron_boxes_for_capture
from da3_cad.objectron_annotation_subset_pb2 import Sequence  # type: ignore[attr-defined]


def test_objectron_keypoints_map_selected_frame_indices_to_clipped_xyxy(tmp_path: Path) -> None:
    images = tmp_path / "frames"
    images.mkdir()
    selected: list[dict[str, object]] = []
    sequence = Sequence()
    for index, frame_id in enumerate((2, 7, 11)):
        name = f"view_{index:03d}.png"
        Image.fromarray(np.full((40, 80, 3), 40 + index, dtype=np.uint8)).save(images / name)
        selected.append({"output_name": name, "frame_index": frame_id})
        frame = sequence.frame_annotations.add()
        frame.frame_id = frame_id
        annotation = frame.annotations.add()
        annotation.object_id = 0
        annotation.visibility = 0.75
        for point_id, (x, y) in enumerate(((-0.1, 0.2), (0.4, 0.1), (1.1, 0.8), (0.7, 0.9))):
            point = annotation.keypoints.add()
            point.id = point_id
            point.point_2d.x = x
            point.point_2d.y = y
    annotation_path = tmp_path / "annotation.pbdata"
    annotation_path.write_bytes(sequence.SerializeToString())
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    output = tmp_path / "boxes.json"

    payload = objectron_boxes_for_capture(annotation_path, capture, images, output)

    assert payload["coordinate_space"] == "normalized-exif-corrected-xyxy"
    views = payload["views"]
    assert isinstance(views, list)
    assert [item["frame_index"] for item in views] == [2, 7, 11]
    assert views[0]["xyxy"] == [0.0, 0.10000000149011612, 1.0, 0.8999999761581421]
    assert views[0]["clipped_to_image"] is True
    assert output.is_file()
