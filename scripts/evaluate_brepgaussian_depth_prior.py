#!/usr/bin/env python3
"""Evaluate an external BrepGaussian Stage 1 checkpoint against a DA3 prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from da3_cad.integrations.gaussian_depth_prior import (
    DepthPriorBundle,
    confidence_aware_depth_prior_loss,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--maximum-rendered-views", type=int, default=4)
    return parser


def _upstream_imports(checkout: Path) -> tuple[Any, Any, Any]:
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

    return render, GaussianModel, Scene


def _visual(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(values)
    result = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return result
    low, high = np.quantile(values[valid], (0.02, 0.98))
    normalized = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    result[..., 0] = np.asarray(255.0 * normalized, dtype=np.uint8)
    result[..., 1] = np.asarray(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)), dtype=np.uint8)
    result[..., 2] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    result[~valid] = 0
    return result


def _save_diagnostics(
    output_dir: Path,
    image_name: str,
    rgb: np.ndarray,
    rendered_depth: np.ndarray,
    prior_depth: np.ndarray,
    mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_dir / f"{image_name}-rgb.png")
    Image.fromarray(_visual(rendered_depth, mask)).save(
        output_dir / f"{image_name}-rendered-depth.png"
    )
    Image.fromarray(_visual(prior_depth, mask)).save(output_dir / f"{image_name}-da3-depth.png")
    log_residual = np.log(np.maximum(rendered_depth, 1e-8)) - np.log(np.maximum(prior_depth, 1e-8))
    if np.any(mask):
        log_residual -= np.median(log_residual[mask])
    Image.fromarray(_visual(np.abs(log_residual), mask)).save(
        output_dir / f"{image_name}-absolute-log-residual.png"
    )


def main() -> None:
    args = _parser().parse_args()
    render, gaussian_model_class, scene_class = _upstream_imports(args.checkout)
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
    prior = DepthPriorBundle.load(args.prior)
    reports: list[dict[str, object]] = []
    with torch.no_grad():
        for camera in scene.getTrainCameras():
            package = render(camera, gaussians, pipeline, background, feature_tag="stage1")
            rendered_rgb = package["render"].clamp(0.0, 1.0)
            ground_truth = camera.original_image.cuda().clamp(0.0, 1.0)
            mse = torch.mean((rendered_rgb - ground_truth) ** 2)
            report: dict[str, object] = {
                "image_name": camera.image_name,
                "rgb_l1": float(torch.mean(torch.abs(rendered_rgb - ground_truth)).item()),
                "rgb_psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
            }
            if camera.image_name in prior.image_names:
                rendered_depth = package["surf_depth"]
                prior_depth, prior_confidence = prior.torch_view(
                    camera.image_name,
                    height=rendered_depth.shape[-2],
                    width=rendered_depth.shape[-1],
                    device=rendered_depth.device,
                )
                foreground = ground_truth.abs().sum(dim=0, keepdim=True) > 1e-3
                terms = confidence_aware_depth_prior_loss(
                    rendered_depth,
                    prior_depth,
                    prior_confidence,
                    foreground_mask=foreground,
                )
                report.update(
                    {
                        "prior_total": float(terms.total.item()),
                        "prior_depth": float(terms.depth.item()),
                        "prior_gradient": float(terms.gradient.item()),
                        "prior_valid_fraction": terms.valid_fraction,
                        "prior_consensus_fraction": terms.consensus_fraction,
                        "prior_log_scale_offset": terms.log_scale_offset,
                    }
                )
                if args.render_dir is not None and len(reports) < args.maximum_rendered_views:
                    rgb = np.asarray(
                        (ground_truth.permute(1, 2, 0) * 255.0).byte().cpu(), dtype=np.uint8
                    )
                    mask = np.asarray(foreground[0].cpu(), dtype=np.bool_)
                    _save_diagnostics(
                        args.render_dir.resolve(),
                        camera.image_name,
                        rgb,
                        np.asarray(rendered_depth[0].cpu(), dtype=np.float32),
                        np.asarray(prior_depth[0].cpu(), dtype=np.float32),
                        mask,
                    )
            reports.append(report)

    numeric_keys = (
        "rgb_l1",
        "rgb_psnr",
        "prior_total",
        "prior_depth",
        "prior_gradient",
        "prior_valid_fraction",
        "prior_consensus_fraction",
        "prior_log_scale_offset",
    )
    means = {
        key: float(np.mean([float(row[key]) for row in reports if key in row]))
        for key in numeric_keys
        if any(key in row for row in reports)
    }
    result = {
        "schema_version": "da3-cad-brepgaussian-prior-evaluation-v1",
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "prior": str(args.prior.resolve()),
        "iteration": args.iteration,
        "gaussian_count": int(len(gaussians.get_xyz)),
        "view_count": len(reports),
        "prior_view_count": sum("prior_total" in row for row in reports),
        "means": means,
        "views": reports,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
