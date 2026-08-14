#!/usr/bin/env python3
"""Evaluate a BrepGaussian checkpoint on disjoint fitted/held-out views and GT mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted
from da3_cad.integrations.gaussian_depth_prior import load_nerf_camera_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("baseline", "da3"), required=True)
    parser.add_argument("--held-out-transforms", type=Path, required=True)
    parser.add_argument("--reference-mesh", type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.5)
    parser.add_argument("--surface-samples", type=int, default=100_000)
    return parser


def _upstream_imports(checkout: Path) -> tuple[Any, Any, Any, Any]:
    root = checkout.resolve() / "GS"
    paths = (
        root,
        root / "submodules" / "diff-surfel-segment-rasterization",
        root / "submodules" / "simple-knn",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from scene.cameras import Camera

    return render, GaussianModel, Scene, Camera


def _rgb_tensor(path: Path) -> Any:
    import torch

    with Image.open(path) as raw:
        rgba = np.asarray(raw.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4]
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _held_out_cameras(transforms: Path, camera_class: Any) -> list[Any]:
    dataset = load_nerf_camera_dataset(transforms)
    cameras: list[Any] = []
    for index, (path, name, intrinsic, world_to_camera) in enumerate(
        zip(
            dataset.image_paths,
            dataset.image_names,
            dataset.intrinsics,
            dataset.extrinsics,
            strict=True,
        )
    ):
        image = _rgb_tensor(path)
        height, width = image.shape[-2:]
        dummy = image.new_zeros((1, height, width))
        fov_x = float(2.0 * np.arctan(0.5 * width / float(intrinsic[0, 0])))
        fov_y = float(2.0 * np.arctan(0.5 * height / float(intrinsic[1, 1])))
        cameras.append(
            camera_class(
                colmap_id=index,
                R=np.asarray(world_to_camera[:3, :3].T, dtype=np.float32),
                T=np.asarray(world_to_camera[:3, 3], dtype=np.float32),
                FoVx=fov_x,
                FoVy=fov_y,
                image=image,
                corner=dummy,
                edge=dummy,
                mask=dummy,
                line_mask=dummy,
                gt_alpha_mask=None,
                image_name=name,
                uid=index,
                data_device="cuda",
            )
        )
    return cameras


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _evaluate_views(
    cameras: list[Any],
    *,
    split: str,
    render: Any,
    gaussians: Any,
    pipeline: Any,
    background: Any,
) -> dict[str, object]:
    import torch

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for camera in cameras:
            package = render(camera, gaussians, pipeline, background, feature_tag="stage1")
            prediction = package["render"].clamp(0.0, 1.0)
            target = camera.original_image.cuda().clamp(0.0, 1.0)
            foreground = target.abs().sum(dim=0) > 1e-3
            squared = (prediction - target) ** 2
            all_mse = squared.mean()
            if bool(foreground.any()):
                foreground_mse = squared[:, foreground].mean()
            else:
                foreground_mse = all_mse.new_tensor(float("nan"))
            alpha = package["rend_alpha"][0]
            predicted_mask = alpha >= 0.5
            union = predicted_mask | foreground
            intersection = predicted_mask & foreground
            silhouette_iou = (
                float(intersection.sum().item() / union.sum().item()) if bool(union.any()) else 1.0
            )
            rows.append(
                {
                    "image_name": camera.image_name,
                    "rgb_l1": float(torch.abs(prediction - target).mean().item()),
                    "rgb_psnr": float((-10.0 * torch.log10(all_mse.clamp_min(1e-12))).item()),
                    "foreground_rgb_psnr": float(
                        (-10.0 * torch.log10(foreground_mse.clamp_min(1e-12))).item()
                    ),
                    "silhouette_iou": silhouette_iou,
                }
            )
    return {
        "split": split,
        "view_count": len(rows),
        "means": {
            "rgb_l1": _mean(rows, "rgb_l1"),
            "rgb_psnr": _mean(rows, "rgb_psnr"),
            "foreground_rgb_psnr": _mean(rows, "foreground_rgb_psnr"),
            "silhouette_iou": _mean(rows, "silhouette_iou"),
        },
        "views": rows,
    }


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"reference is not a mesh: {path}")
    return loaded


def _geometry_metrics(
    gaussians: Any,
    reference_path: Path,
    *,
    opacity_threshold: float,
    samples: int,
) -> dict[str, object]:
    points = np.asarray(gaussians.get_xyz.detach().cpu(), dtype=np.float64)
    opacity = np.asarray(gaussians.get_opacity.detach().cpu(), dtype=np.float64).reshape(-1)
    selected = points[opacity >= opacity_threshold]
    if len(selected) == 0:
        raise ValueError("opacity threshold removed every Gaussian")
    reference = _load_mesh(reference_path)
    sampled = sample_surface_area_weighted(reference, samples, seed=1729).points
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(reference.vertices), axis=0)))
    if diagonal <= 0.0:
        raise ValueError("reference mesh has zero diagonal")
    prediction_to_reference = cKDTree(sampled).query(selected, workers=-1)[0]
    reference_to_prediction = cKDTree(selected).query(sampled, workers=-1)[0]
    both = np.concatenate((prediction_to_reference, reference_to_prediction))
    return {
        "reference": str(reference_path.resolve()),
        "reference_is_ground_truth": True,
        "opacity_probability_threshold": opacity_threshold,
        "gaussian_count": int(len(points)),
        "selected_gaussian_count": int(len(selected)),
        "surface_sample_count": int(samples),
        "reference_bbox_diagonal": diagonal,
        "normalized_symmetric_chamfer": float(
            0.5 * (prediction_to_reference.mean() + reference_to_prediction.mean()) / diagonal
        ),
        "normalized_p95_bidirectional_distance": float(np.quantile(both, 0.95) / diagonal),
        "normalized_prediction_to_reference_mean": float(prediction_to_reference.mean() / diagonal),
        "normalized_reference_to_prediction_mean": float(reference_to_prediction.mean() / diagonal),
    }


def main() -> None:
    args = _parser().parse_args()
    render, gaussian_model_class, scene_class, camera_class = _upstream_imports(args.checkout)
    import torch

    model_dir = args.model_dir.resolve()
    dataset_args = SimpleNamespace(
        sh_degree=3,
        source_path=str(args.dataset.resolve()),
        model_path=str(model_dir),
        images="images",
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=False,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )
    gaussians = gaussian_model_class(dataset_args.sh_degree)
    scene = scene_class(
        dataset_args,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        feature_tag="stage1",
    )
    checkpoint = model_dir / f"chkpnt{args.iteration}.pth"
    captured = torch.load(checkpoint, map_location="cuda", weights_only=False)
    gaussians.restore(captured, SimpleNamespace(), "stage1")
    gaussians._extra_features = torch.zeros(
        (len(gaussians.get_xyz), gaussians.extra_feature_degree),
        dtype=torch.float32,
        device="cuda",
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    fitted = _evaluate_views(
        scene.getTrainCameras(),
        split="fitted",
        render=render,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
    )
    held_out = _evaluate_views(
        _held_out_cameras(args.held_out_transforms.resolve(), camera_class),
        split="held_out",
        render=render,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
    )
    geometry = (
        _geometry_metrics(
            gaussians,
            args.reference_mesh.resolve(),
            opacity_threshold=args.opacity_threshold,
            samples=args.surface_samples,
        )
        if args.reference_mesh is not None
        else None
    )
    result = {
        "schema_version": "da3-cad-brepgaussian-held-out-evaluation-v1",
        "variant": args.variant,
        "optimization_seed": args.seed,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "iteration": args.iteration,
        "fitted": fitted,
        "held_out": held_out,
        "geometry": geometry,
        "evidence_policy": (
            "held_out views are disjoint from optimization; fitted-view PSNR is diagnostic only"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
