#!/usr/bin/env python3
"""Export a camera-conditioned DA3 prior for a NeRF/BrepGaussian dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from PIL import Image

from da3_cad.backends.da3 import Da3Backend
from da3_cad.integrations.gaussian_depth_prior import (
    DepthPriorBundle,
    greedy_camera_subset,
    load_nerf_camera_dataset,
    spherical_camera_coverage,
)
from da3_cad.observations import load_observations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DA3 with the exact NeRF cameras and save a versioned depth/confidence "
            "bundle for an external Gaussian-surface optimizer."
        )
    )
    parser.add_argument("transforms", type=Path, help="transforms_train.json")
    parser.add_argument("output", type=Path, help="output .npz bundle")
    parser.add_argument("--views", type=int, default=10, help="coverage-greedy view count")
    parser.add_argument(
        "--checkpoint",
        choices=("base", "large-1.1", "large"),
        default="base",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/upstream/Depth-Anything-3"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--process-resolution", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parser().parse_args()
    transforms = args.transforms.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing prior: {output}")
    cameras = load_nerf_camera_dataset(transforms)
    selected_indices = greedy_camera_subset(cameras.extrinsics, args.views)
    selected = cameras.subset(selected_indices)
    spherical_coverage = spherical_camera_coverage(selected.extrinsics)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".da3-prior-", dir=output.parent) as staging_raw:
        staging = Path(staging_raw)
        for order, source in enumerate(selected.image_paths):
            destination = staging / f"{order:04d}_{source.stem}.png"
            with Image.open(source) as raw:
                rgba = raw.convert("RGBA")
            value = 0 if args.background == "black" else 255
            background = Image.new("RGBA", rgba.size, (value, value, value, 255))
            Image.alpha_composite(background, rgba).convert("RGB").save(destination)
        observations = load_observations(staging)
        backend = Da3Backend(
            checkpoint=args.checkpoint,
            source_dir=args.source_dir.resolve(),
            cache_dir=args.cache_dir.resolve(),
            process_resolution=args.process_resolution,
            process_resolution_method="upper_bound_resize",
            local_files_only=True,
            accepted_noncommercial=args.accept_noncommercial_weights,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
            export_feature_layer=None,
        )
        prediction = backend.predict(
            observations,
            device=args.device,
            seed=args.seed,
            extrinsics=selected.extrinsics,
            intrinsics=selected.intrinsics,
            align_to_input_ext_scale=True,
        )

    bundle = DepthPriorBundle.from_prediction(
        prediction,
        selected.image_names,
        metadata={
            "source_schema": "nerf-transforms",
            "transforms_path": str(transforms),
            "transforms_sha256": _sha256(transforms),
            "pool_view_count": len(cameras.image_paths),
            "selected_indices": list(selected_indices),
            "selected_image_names": list(selected.image_names),
            "selection": "greedy-spherical-camera-coverage",
            "spherical_coverage_fraction": spherical_coverage,
            "surface_cone_degrees": 35.0,
            "camera_conditioning": True,
            "align_to_input_ext_scale": True,
            "process_resolution": args.process_resolution,
            "alpha_composite_background": args.background,
            "seed": args.seed,
        },
    )
    bundle.save(output)
    report = {
        "schema_version": "da3-cad-depth-prior-export-v1",
        "output": str(output),
        "output_sha256": _sha256(output),
        "backend": bundle.backend,
        "views": len(bundle.image_names),
        "depth_shape": list(bundle.depth.shape),
        "finite_positive_depth_fraction": float(
            ((bundle.depth > 0.0) & (bundle.depth < float("inf"))).mean()
        ),
        "mean_confidence": float(bundle.confidence.mean()),
        "metadata": bundle.metadata,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
