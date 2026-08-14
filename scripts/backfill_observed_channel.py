"""Backfill the observed masked-depth channel from a saved DA3 run.

This is a deterministic diagnostics migration. It never runs DA3 and never
changes the trusted cloud or CAD result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.config import ObservationCoverageConfig
from da3_cad.geometry.coverage import analyze_camera_coverage
from da3_cad.geometry.fusion import ScaleChannel, fuse_prediction
from da3_cad.models import DepthPrediction, UInt8Array


def _load_images(
    image_dir: Path,
    shape: tuple[int, int],
    names: tuple[str, ...] | None = None,
) -> tuple[UInt8Array, ...]:
    height, width = shape
    paths = (
        [image_dir / name for name in names]
        if names is not None
        else sorted(image_dir.glob("*.png"), key=lambda path: path.name)
    )
    if not paths:
        raise FileNotFoundError(f"no PNG inputs in {image_dir}")
    images: list[UInt8Array] = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(
                (width, height),
                resample=Image.Resampling.BILINEAR,
            )
            images.append(np.asarray(resized, dtype=np.uint8))
    return tuple(images)


def _scale(report: dict[str, object]) -> ScaleChannel:
    cloud = report.get("cloud")
    if not isinstance(cloud, dict):
        return ScaleChannel()
    if cloud.get("scale_status") != "known":
        return ScaleChannel()
    factor = cloud.get("world_units_to_mm")
    if not isinstance(factor, (float, int)):
        raise ValueError("known saved scale has no world_units_to_mm")
    evidence = cloud.get("scale_evidence")
    return ScaleChannel(
        status="known",
        units=str(cloud.get("units", "world")),
        world_units_to_mm=float(factor),
        source=str(cloud.get("scale_source", "saved-run")),
        evidence=evidence if isinstance(evidence, dict) else {},
    )


def backfill(run_dir: Path, image_dir: Path, *, overwrite: bool) -> None:
    geometry_root = run_dir / "artefacts" / "geometry"
    artifact_root = geometry_root / "artefacts"
    prediction_path = artifact_root / "camera_prediction.npz"
    geometry_report_path = geometry_root / "geometry_report.json"
    output_npz = artifact_root / "observed_cloud.npz"
    output_ply = artifact_root / "observed_cloud.ply"
    coverage_path = run_dir / "artefacts" / "camera_coverage.json"
    for path in (prediction_path, geometry_report_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing saved-run artifact: {path}")
    if not overwrite and (output_npz.exists() or output_ply.exists()):
        raise FileExistsError(
            f"observed channel already exists under {artifact_root}; pass --overwrite"
        )

    with np.load(prediction_path, allow_pickle=False) as payload:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=np.bool_)
        cached_images = (
            np.asarray(payload["processed_images"], dtype=np.uint8)
            if "processed_images" in payload.files
            else None
        )
    selection_path = artifact_root / "view_selection.json"
    selected_names: tuple[str, ...] | None = None
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        raw_names = selection.get("selected_image_names")
        if isinstance(raw_names, list) and len(raw_names) == len(depth):
            selected_names = tuple(str(name) for name in raw_names)
    images = (
        tuple(cached_images[index] for index in range(len(cached_images)))
        if cached_images is not None
        else _load_images(
            image_dir,
            (depth.shape[1], depth.shape[2]),
            selected_names,
        )
    )
    if len(images) != len(depth):
        raise ValueError(
            f"saved prediction has {len(depth)} views but {image_dir} has {len(images)} PNGs"
        )
    prediction = DepthPrediction(
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=images,
        backend="saved-da3-diagnostics",
        warnings=("reloaded from camera_prediction.npz; DA3 was not rerun",),
    )
    geometry_report = json.loads(geometry_report_path.read_text(encoding="utf-8"))
    segmentation = geometry_report.get("segmentation")
    mask_source = (
        str(segmentation.get("backend", "saved-mask"))
        if isinstance(segmentation, dict)
        else "saved-mask"
    )
    observed = fuse_prediction(
        prediction,
        masks,
        mask_source=mask_source,
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        scale=_scale(geometry_report),
    )
    np.savez_compressed(
        output_npz,
        points=observed.points,
        colors=observed.colors,
        confidence=observed.confidences,
        view_indices=observed.view_indices,
        pixel_xy=observed.pixel_xy,
        derived_from=np.asarray(str(prediction_path)),
    )
    point_cloud = trimesh.points.PointCloud(  # type: ignore[no-untyped-call]
        vertices=observed.points,
        colors=observed.colors,
    )
    point_cloud.export(output_ply)  # type: ignore[no-untyped-call]

    trusted_path = artifact_root / "trusted_geometry.npz"
    if not trusted_path.is_file():
        trusted_path = artifact_root / "fused_cloud.npz"
    with np.load(trusted_path, allow_pickle=False) as trusted:
        trusted_points = np.asarray(trusted["points"], dtype=np.float32)
    coverage = analyze_camera_coverage(
        prediction,
        trusted_points,
        ObservationCoverageConfig(),
    )
    coverage_path.write_text(
        json.dumps(coverage.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{run_dir.name}: observed={len(observed.points)} "
        f"trusted={len(trusted_points)} coverage={coverage.status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    backfill(args.run, args.images, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
