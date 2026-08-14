#!/usr/bin/env python3
"""Adapt one redistributable DA3-CAD fixture to a sparse BrepGaussian scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_dilation, binary_erosion

from da3_cad.integrations.gaussian_depth_prior import (
    greedy_camera_subset,
    spherical_camera_coverage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-views", type=int, default=5)
    parser.add_argument("--held-out-views", type=int, default=4)
    parser.add_argument("--normalized-maximum-extent", type=float, default=2.0)
    return parser


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"reference is not a mesh: {path}")
    return loaded


def _write_initial_point_cloud(path: Path) -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.3, 1.3, size=(100_000, 3)).astype(np.float32)
    attributes = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    attributes["x"], attributes["y"], attributes["z"] = points.T
    attributes["nx"] = attributes["ny"] = attributes["nz"] = 0.0
    attributes["red"] = attributes["green"] = attributes["blue"] = 127
    PlyData([PlyElement.describe(attributes, "vertex")]).write(path)


def _normalized_extrinsics(
    extrinsics: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    result = np.asarray(extrinsics, dtype=np.float64).copy()
    for index, world_to_camera in enumerate(result):
        rotation = world_to_camera[:3, :3]
        camera_center = -(rotation.T @ world_to_camera[:3, 3])
        normalized_center = (camera_center - center) * scale
        result[index, :3, 3] = -(rotation @ normalized_center)
    return result


def _frame(
    output: Path,
    case: Path,
    image_name: str,
    directory: str,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    companions: bool,
) -> dict[str, object]:
    source_image = case / "views" / image_name
    source_mask = case / "masks" / image_name
    with Image.open(source_image) as raw:
        rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    with Image.open(source_mask) as raw:
        mask = np.asarray(raw.convert("L"), dtype=np.uint8) >= 128
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"image/mask shape mismatch for {image_name}")
    masked = rgb.copy()
    masked[~mask] = 0
    destination = output / directory / image_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked, mode="RGB").save(destination)

    if companions:
        stem = Path(image_name).stem
        edge = mask & ~binary_erosion(mask, iterations=1)
        edge = binary_dilation(edge, iterations=1)
        values = np.asarray(edge, dtype=np.uint8) * 255
        for companion, payload in (
            ("edge_img", values),
            ("mask_img", np.asarray(mask, dtype=np.uint8) * 255),
            ("corner_img", np.zeros(mask.shape, dtype=np.uint8)),
            ("line_mask_img", np.zeros(mask.shape, dtype=np.uint8)),
        ):
            target = output / companion / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(payload, mode="L").save(target)

    camera_to_world = np.linalg.inv(world_to_camera)
    camera_to_world[:3, 1:3] *= -1.0
    return {
        "file_path": f"./{directory}/{Path(image_name).stem}",
        "rotation": 0,
        "camera_intrinsics": intrinsic.tolist(),
        "transform_matrix": camera_to_world.tolist(),
    }


def _transforms(frames: list[dict[str, object]], width: int, focal: float) -> dict[str, object]:
    return {
        "camera_angle_x": float(2.0 * np.arctan(0.5 * width / focal)),
        "frames": frames,
    }


def main() -> None:
    args = _parser().parse_args()
    case = args.case_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing scene: {output}")
    if args.train_views < 2 or args.held_out_views < 1:
        raise ValueError("need at least two fitted and one held-out view")

    with np.load(case / "cameras.npz", allow_pickle=False) as payload:
        image_names = tuple(str(item) for item in payload["image_names"].tolist())
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
        extrinsics = np.asarray(payload["extrinsics"], dtype=np.float64)
    count = len(image_names)
    if args.train_views + args.held_out_views > count:
        raise ValueError("fitted and held-out splits exceed available cameras")

    reference = _mesh(case / "gt.stl")
    bounds = np.asarray(reference.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    maximum_extent = float(np.ptp(bounds, axis=0).max())
    if maximum_extent <= 0.0:
        raise ValueError("reference mesh has zero extent")
    scale = args.normalized_maximum_extent / maximum_extent
    normalized_extrinsics = _normalized_extrinsics(extrinsics, center, scale)

    train_indices = greedy_camera_subset(normalized_extrinsics, args.train_views)
    remaining = tuple(index for index in range(count) if index not in set(train_indices))
    held_local = greedy_camera_subset(
        normalized_extrinsics[np.asarray(remaining, dtype=np.int64)],
        args.held_out_views,
    )
    held_indices = tuple(remaining[index] for index in held_local)

    output.mkdir(parents=True)
    with Image.open(case / "views" / image_names[0]) as first:
        width = first.width
    fitted_frames = [
        _frame(
            output,
            case,
            image_names[index],
            "train_img",
            intrinsics[index],
            normalized_extrinsics[index],
            companions=True,
        )
        for index in train_indices
    ]
    held_frames = [
        _frame(
            output,
            case,
            image_names[index],
            "heldout_img",
            intrinsics[index],
            normalized_extrinsics[index],
            companions=False,
        )
        for index in held_indices
    ]
    (output / "transforms_train.json").write_text(
        json.dumps(_transforms(fitted_frames, width, float(intrinsics[0, 0, 0])), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "transforms_heldout.json").write_text(
        json.dumps(_transforms(held_frames, width, float(intrinsics[0, 0, 0])), indent=2) + "\n",
        encoding="utf-8",
    )

    normalized_reference = reference.copy()
    normalized_reference.apply_translation(-center)
    normalized_reference.apply_scale(scale)
    evaluation_dir = output / "evaluator_only"
    evaluation_dir.mkdir()
    normalized_reference.export(evaluation_dir / "reference_mesh.ply")

    _write_initial_point_cloud(output / "points3d.ply")

    report = {
        "schema_version": "da3-cad-brepgaussian-fixture-v1",
        "source_case": str(case),
        "fitted_indices": list(train_indices),
        "held_out_indices": list(held_indices),
        "fitted_views": len(train_indices),
        "held_out_views": len(held_indices),
        "spherical_coverage_fraction": spherical_camera_coverage(
            normalized_extrinsics[np.asarray(train_indices, dtype=np.int64)]
        ),
        "normalization": {
            "source_center": center.tolist(),
            "source_to_normalized_scale": scale,
            "normalized_maximum_extent": args.normalized_maximum_extent,
        },
        "reference_mesh": "evaluator_only/reference_mesh.ply",
        "reference_visible_to_training": False,
        "initial_point_cloud_seed": 0,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
