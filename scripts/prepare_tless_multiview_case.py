#!/usr/bin/env python3
"""Prepare a controlled real-RGB T-LESS case with evaluator-only object poses.

The reference CAD mesh is deliberately not read.  BOP object-to-camera poses
are used only to isolate dense reconstruction/CAD fitting from camera recovery;
this is a controlled benchmark path, not the public photo-product path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from da3_cad.geometry.cameras import CameraBundle
from da3_cad.integrations.gaussian_scene import prepare_gaussian_scene
from da3_cad.target_preparation import prepare_target_from_masks


@dataclass(frozen=True, slots=True)
class View:
    image_id: int
    gt_index: int
    visibility: float
    intrinsic: np.ndarray
    extrinsic: np.ndarray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("scene_id", type=int)
    parser.add_argument("object_id", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.97)
    parser.add_argument("--views", type=int, default=40)
    parser.add_argument("--held-out-views", type=int, default=6)
    parser.add_argument("--initial-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _view(
    image_id: int,
    gt_index: int,
    visibility: float,
    camera_row: dict[str, object],
    gt_row: dict[str, object],
) -> View:
    intrinsic = np.asarray(camera_row["cam_K"], dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(gt_row["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(gt_row["cam_t_m2c"], dtype=np.float64).reshape(3)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    return View(image_id, gt_index, visibility, intrinsic, extrinsic)


def _candidates(scene: Path, object_id: int, minimum_visibility: float) -> list[View]:
    cameras = _load_json(scene / "scene_camera.json")
    ground_truth = _load_json(scene / "scene_gt.json")
    information = _load_json(scene / "scene_gt_info.json")
    result: list[View] = []
    for key in sorted(ground_truth, key=int):
        gt_rows = ground_truth[key]
        info_rows = information[key]
        if not isinstance(gt_rows, list) or not isinstance(info_rows, list):
            raise ValueError("malformed BOP scene GT arrays")
        matches = [
            (index, row, info_rows[index])
            for index, row in enumerate(gt_rows)
            if isinstance(row, dict) and int(row.get("obj_id", -1)) == object_id
        ]
        if not matches:
            continue
        index, gt_row, info_row = max(
            matches,
            key=lambda item: float(item[2].get("visib_fract", 0.0)),
        )
        visibility = float(info_row.get("visib_fract", 0.0))
        if visibility < minimum_visibility:
            continue
        camera_row = cameras[key]
        if not isinstance(camera_row, dict):
            raise ValueError("malformed BOP camera row")
        result.append(_view(int(key), index, visibility, camera_row, gt_row))
    return result


def _camera_centres(views: list[View]) -> np.ndarray:
    return np.stack([-(view.extrinsic[:3, :3].T @ view.extrinsic[:3, 3]) for view in views])


def _angular_subset(views: list[View], count: int) -> list[View]:
    if len(views) <= count:
        return views
    centres = _camera_centres(views)
    directions = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    selected = [int(np.argmax(directions[:, 2]))]
    while len(selected) < count:
        similarities = directions @ directions[np.asarray(selected)].T
        nearest = similarities.max(axis=1)
        nearest[np.asarray(selected)] = np.inf
        selected.append(int(np.argmin(nearest)))
    return [views[index] for index in sorted(selected)]


def main() -> None:
    args = _parser().parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    if not 0.0 < args.minimum_visible_fraction <= 1.0:
        raise ValueError("minimum visible fraction must be in (0,1]")
    if args.views < 9 or args.held_out_views < 1 or args.held_out_views > args.views - 3:
        raise ValueError("view budgets must leave at least three fitted views")
    scene = args.dataset_root.resolve() / f"{args.scene_id:06d}"
    if not scene.is_dir():
        raise ValueError(f"T-LESS scene does not exist: {scene}")
    candidates = _candidates(scene, args.object_id, args.minimum_visible_fraction)
    if len(candidates) < min(args.views, 9):
        raise RuntimeError(
            f"only {len(candidates)} views meet visibility >= {args.minimum_visible_fraction:.3f}"
        )
    selected = _angular_subset(candidates, args.views)

    root = args.output_dir.resolve()
    source_images = root / "source" / "images"
    source_masks = root / "source" / "masks"
    source_images.mkdir(parents=True)
    source_masks.mkdir(parents=True)
    names: list[str] = []
    for output_index, view in enumerate(selected):
        name = f"view_{output_index:03d}.png"
        image = scene / "rgb" / f"{view.image_id:06d}.png"
        mask = scene / "mask_visib" / f"{view.image_id:06d}_{view.gt_index:06d}.png"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"missing T-LESS RGB/mask pair: {image}, {mask}")
        shutil.copy2(image, source_images / name)
        shutil.copy2(mask, source_masks / name)
        names.append(name)

    bundle = CameraBundle(
        image_names=tuple(names),
        intrinsics=np.stack([view.intrinsic for view in selected]),
        extrinsics=np.stack([view.extrinsic for view in selected]),
        source="tless-bop-object-pose-evaluator-only",
        scale_status="known",
        world_units="millimetre",
        world_units_to_mm=1.0,
        details={
            "dataset": "T-LESS test_primesense",
            "scene_id": args.scene_id,
            "object_id": args.object_id,
            "pose_usage": "controlled benchmark only; not product inference",
            "reference_geometry_access": False,
        },
    )
    source_cameras = root / "source" / "cameras.npz"
    bundle.save(source_cameras)
    prepared = prepare_target_from_masks(
        source_images,
        root / "prepared",
        source_masks,
        camera_bundle_path=source_cameras,
        margin_fraction=0.15,
        selection_source="dataset-mask-oracle",
    )
    assert prepared.camera_bundle_path is not None
    gaussian = prepare_gaussian_scene(
        prepared.images_dir,
        prepared.masks_dir,
        prepared.camera_bundle_path,
        root / "gaussian_scene",
        held_out_views=args.held_out_views,
        initial_points=args.initial_points,
        seed=args.seed,
    )
    centres = _camera_centres(selected)
    singular = np.linalg.svd(centres - centres.mean(axis=0), compute_uv=False)
    report = {
        "schema_version": "da3-cad-tless-controlled-case-v1",
        "scene_id": args.scene_id,
        "object_id": args.object_id,
        "candidate_views": len(candidates),
        "selected_views": len(selected),
        "minimum_visible_fraction": args.minimum_visible_fraction,
        "selected_visibility_minimum": min(view.visibility for view in selected),
        "camera_center_singular_values_mm": singular.tolist(),
        "prepared_target": str(prepared.root),
        "gaussian_scene": str(gaussian.output_dir),
        "reference_geometry_access": False,
        "evidence_policy": (
            "BOP object poses and visible masks isolate RGB-to-surface fitting from SfM; "
            "the reference CAD mesh is reserved for post-hoc evaluation"
        ),
    }
    report_path = root / "controlled_case.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
