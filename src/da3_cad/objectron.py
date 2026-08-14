"""Import Objectron projected 3D boxes as explicit target-localization prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.objectron_annotation_subset_pb2 import Sequence  # type: ignore[attr-defined]
from da3_cad.observations import load_observations

OBJECTRON_PAGE = "https://github.com/google-research-datasets/Objectron"
OBJECTRON_SCHEMA_URL = (
    "https://github.com/google-research-datasets/Objectron/blob/master/"
    "objectron/schema/annotation_data.proto"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_capture_frames(capture_manifest: Path) -> list[dict[str, object]]:
    payload = json.loads(capture_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("selected"), list):
        raise ValueError("capture manifest must contain a selected frame list")
    selected: list[dict[str, object]] = []
    for raw in payload["selected"]:
        if not isinstance(raw, dict):
            raise ValueError("capture selected entries must be JSON objects")
        if not isinstance(raw.get("output_name"), str) or not isinstance(
            raw.get("frame_index"), int
        ):
            raise ValueError("capture selected entries require output_name and frame_index")
        selected.append(raw)
    return selected


def _frame_lookup(annotation_path: Path) -> dict[int, Any]:
    sequence = Sequence()
    sequence.ParseFromString(annotation_path.read_bytes())
    lookup: dict[int, Any] = {}
    for frame in sequence.frame_annotations:
        frame_id = int(frame.frame_id)
        if frame_id in lookup:
            raise ValueError(f"duplicate Objectron frame annotation: {frame_id}")
        lookup[frame_id] = frame
    if not lookup:
        raise ValueError(f"Objectron annotation sequence contains no frames: {annotation_path}")
    return lookup


def objectron_boxes_for_capture(
    annotation_path: Path,
    capture_manifest: Path,
    images_dir: Path,
    output_path: Path,
    *,
    object_id: int = 0,
) -> dict[str, object]:
    """Map Objectron normalized keypoints to selected key-frame XYXY prompts.

    The output is target localization, not a silhouette oracle.  SAM2 must
    still infer the visible instance mask inside each projected 3D box.
    """

    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing box prompts: {output_path}")
    observations = load_observations(images_dir)
    observation_lookup = {item.relative_path: item for item in observations.images}
    selected = _selected_capture_frames(capture_manifest)
    selected_names = [str(item["output_name"]) for item in selected]
    if set(selected_names) != set(observation_lookup):
        raise ValueError("capture output names do not exactly match the supplied image directory")
    frames = _frame_lookup(annotation_path)

    views: list[dict[str, object]] = []
    for selected_frame in selected:
        image_name = str(selected_frame["output_name"])
        frame_index_value = selected_frame["frame_index"]
        if not isinstance(frame_index_value, int):
            raise ValueError("capture frame_index must be an integer")
        frame_index = frame_index_value
        frame = frames.get(frame_index)
        if frame is None:
            raise ValueError(f"Objectron annotation is missing selected frame {frame_index}")
        matching = [item for item in frame.annotations if int(item.object_id) == object_id]
        if len(matching) != 1:
            raise ValueError(
                f"expected one Objectron object_id={object_id} annotation in frame "
                f"{frame_index}, got {len(matching)}"
            )
        annotation = matching[0]
        points = np.asarray(
            [
                (float(keypoint.point_2d.x), float(keypoint.point_2d.y))
                for keypoint in annotation.keypoints
            ],
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 2:
            raise ValueError(f"Objectron frame {frame_index} has insufficient 2D keypoints")
        if not np.isfinite(points).all():
            raise ValueError(f"Objectron frame {frame_index} has non-finite 2D keypoints")
        raw_xyxy = np.asarray(
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()),
            dtype=np.float64,
        )
        clipped = np.clip(raw_xyxy, 0.0, 1.0)
        if clipped[2] - clipped[0] <= 1e-4 or clipped[3] - clipped[1] <= 1e-4:
            raise ValueError(f"Objectron frame {frame_index} has no usable in-frame box")
        views.append(
            {
                "image": image_name,
                "xyxy": clipped.tolist(),
                "frame_index": frame_index,
                "object_id": object_id,
                "annotation_visibility": float(annotation.visibility),
                "projected_keypoint_count": int(points.shape[0]),
                "unclipped_xyxy": raw_xyxy.tolist(),
                "clipped_to_image": bool(not np.array_equal(raw_xyxy, clipped)),
            }
        )

    payload: dict[str, object] = {
        "schema_version": "da3-cad-target-boxes-v1",
        "coordinate_space": "normalized-exif-corrected-xyxy",
        "selection_source": "dataset-box-oracle",
        "dataset": {
            "name": "Google Objectron",
            "page": OBJECTRON_PAGE,
            "license": "C-UDA-1.0",
            "annotation_path": str(annotation_path.resolve()),
            "annotation_sha256": _sha256(annotation_path),
            "annotation_schema": OBJECTRON_SCHEMA_URL,
            "capture_manifest": str(capture_manifest.resolve()),
            "capture_manifest_sha256": _sha256(capture_manifest),
        },
        "contract": (
            "projected Objectron 3D bounding boxes provide target localization only; "
            "they are not pixel masks and are not CAD ground truth"
        ),
        "views": views,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
