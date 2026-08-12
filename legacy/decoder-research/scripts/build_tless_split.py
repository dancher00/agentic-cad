#!/usr/bin/env python3
"""Freeze nested, target-visible T-LESS Primesense views for all 30 objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from da3_cad.benchmark.splits import GLOBAL_SEED
from da3_cad.benchmark.tless import sha256_file
from da3_cad.benchmark.tless_protocol import (
    TlessViewCandidate,
    camera_direction_in_model_frame,
    nested_maxmin_views,
)

VIEW_COUNTS = (1, 2, 4, 8, 16)
VISIBILITY_THRESHOLDS = (0.80, 0.75, 0.70, 0.60, 0.0)


def _candidate(
    root: Path,
    scene_id: int,
    frame_id: int,
    gt_index: int,
    gt: dict[str, Any],
    info: dict[str, Any],
) -> TlessViewCandidate:
    rgb = root / f"test_primesense/{scene_id:06d}/rgb/{frame_id:06d}.png"
    mask = root / (
        f"test_primesense/{scene_id:06d}/mask_visib/"
        f"{frame_id:06d}_{gt_index:06d}.png"
    )
    with Image.open(rgb) as image:
        width, height = image.size
    x, y, box_width, box_height = (float(value) for value in info["bbox_visib"])
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    center_distance = (
        ((center_x - width / 2.0) / width) ** 2
        + ((center_y - height / 2.0) / height) ** 2
    ) ** 0.5
    return TlessViewCandidate(
        scene_id=scene_id,
        frame_id=frame_id,
        gt_index=gt_index,
        visible_fraction=float(info["visib_fract"]),
        visible_area_fraction=(box_width * box_height) / float(width * height),
        center_distance_fraction=float(center_distance),
        direction=camera_direction_in_model_frame(gt["cam_R_m2c"], gt["cam_t_m2c"]),
        rgb_path=rgb.relative_to(root).as_posix(),
        mask_path=mask.relative_to(root).as_posix(),
    )


def _split_digest(objects: list[dict[str, Any]]) -> str:
    encoded = json.dumps(objects, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tless/extracted"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/splits/tless_primesense.json"),
    )
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    scene_root = args.data_root / "test_primesense"
    grouped: dict[int, dict[tuple[int, int], list[TlessViewCandidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for scene in sorted(path for path in scene_root.iterdir() if path.is_dir()):
        scene_id = int(scene.name)
        gt_by_frame = json.loads((scene / "scene_gt.json").read_text(encoding="utf-8"))
        info_by_frame = json.loads(
            (scene / "scene_gt_info.json").read_text(encoding="utf-8")
        )
        for frame_text, entries in gt_by_frame.items():
            frame_id = int(frame_text)
            for gt_index, gt in enumerate(entries):
                object_id = int(gt["obj_id"])
                candidate = _candidate(
                    args.data_root,
                    scene_id,
                    frame_id,
                    gt_index,
                    gt,
                    info_by_frame[frame_text][gt_index],
                )
                grouped[object_id][(scene_id, gt_index)].append(candidate)

    objects: list[dict[str, Any]] = []
    for object_id in range(1, 31):
        chosen_threshold: float | None = None
        eligible_groups: list[tuple[tuple[int, int], tuple[TlessViewCandidate, ...]]] = []
        for threshold in VISIBILITY_THRESHOLDS:
            eligible_groups = []
            for key, values in grouped[object_id].items():
                filtered = tuple(item for item in values if item.visible_fraction >= threshold)
                if len(filtered) >= max(VIEW_COUNTS):
                    eligible_groups.append((key, filtered))
            if eligible_groups:
                chosen_threshold = threshold
                break
        if chosen_threshold is None:
            raise ValueError(f"object {object_id} has no 16-view scene/instance")
        group_key, candidates = max(
            eligible_groups,
            key=lambda item: (
                median(value.quality for value in item[1]),
                len(item[1]),
                -item[0][0],
                -item[0][1],
            ),
        )
        selected = nested_maxmin_views(
            candidates,
            count=max(VIEW_COUNTS),
            object_id=object_id,
            seed=args.seed,
        )
        views = []
        for rank, view in enumerate(selected):
            rgb = args.data_root / view.rgb_path
            mask = args.data_root / view.mask_path
            views.append(
                {
                    "rank": rank,
                    "scene_id": view.scene_id,
                    "frame_id": view.frame_id,
                    "gt_index": view.gt_index,
                    "rgb_path": view.rgb_path,
                    "rgb_sha256": sha256_file(rgb),
                    "audit_mask_path": view.mask_path,
                    "audit_mask_sha256": sha256_file(mask),
                    "visible_fraction": view.visible_fraction,
                    "visible_area_fraction": view.visible_area_fraction,
                    "center_distance_fraction": view.center_distance_fraction,
                    "selection_only_camera_direction_model_frame": list(view.direction),
                }
            )
        gt_mesh = args.data_root / f"models_cad/obj_{object_id:06d}.ply"
        objects.append(
            {
                "object_id": object_id,
                "item_id": f"tless-{object_id:02d}",
                "scene_id": group_key[0],
                "gt_index": group_key[1],
                "visibility_threshold": chosen_threshold,
                "eligible_views_in_group": len(candidates),
                "gt_mesh_path": gt_mesh.relative_to(args.data_root).as_posix(),
                "gt_mesh_sha256": sha256_file(gt_mesh),
                "views": views,
            }
        )
    payload = {
        "schema_version": "1.0",
        "protocol": "da3-cad-tless-primesense-v1",
        "global_seed": args.seed,
        "view_counts": list(VIEW_COUNTS),
        "object_count": len(objects),
        "selection_gt_use": (
            "object/instance identity, visibility, bbox centrality and camera direction only"
        ),
        "reconstruction_gt_access": False,
        "reconstruction_inputs": "full-frame RGB only; no crop, masks, depth, K or poses",
        "objects": objects,
    }
    payload["objects_sha256"] = _split_digest(objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "objects": len(objects),
                "sha256": payload["objects_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
