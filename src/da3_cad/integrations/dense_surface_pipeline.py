"""Reproducible calibrated RGB-to-measured-surface orchestration."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from da3_cad.integrations.colmap_mvs import prepare_colmap_mvs_workspace
from da3_cad.integrations.depth_fusion import fuse_colmap_depth_maps
from da3_cad.integrations.surface_mesh import build_poisson_surface


@dataclass(frozen=True, slots=True)
class DenseSurfacePipelineResult:
    output_dir: Path
    cloud_path: Path
    surface_path: Path
    report_path: Path
    report: dict[str, object]


def run_dense_surface_pipeline(
    images_dir: Path,
    masks_dir: Path,
    camera_bundle_path: Path,
    output_dir: Path,
    *,
    mvs_python: Path,
    source_views: int = 6,
    maximum_image_size: int = 800,
    patchmatch_iterations: int = 3,
    minimum_spherical_coverage: float = 0.25,
) -> DenseSurfacePipelineResult:
    """Run masked calibrated PatchMatch, verified fusion and measured surface meshing."""

    if output_dir.exists():
        raise ValueError(f"dense surface output already exists: {output_dir}")
    if not mvs_python.is_file():
        raise ValueError(f"dedicated MVS Python does not exist: {mvs_python}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    workspace = prepare_colmap_mvs_workspace(
        images_dir,
        masks_dir,
        camera_bundle_path,
        output_dir / "mvs",
        source_views=source_views,
        minimum_spherical_coverage=minimum_spherical_coverage,
    )
    preparation_seconds = time.monotonic() - started
    patchmatch_started = time.monotonic()
    # Do not resolve this symlink: Python detects its virtual environment from
    # the invoked executable path. Resolving it silently falls back to the base
    # interpreter and can load the CPU-only PyCOLMAP build.
    mvs_executable = mvs_python if mvs_python.is_absolute() else Path.cwd() / mvs_python
    worker = Path(__file__).with_name("patchmatch_worker.py").resolve()
    command = [
        str(mvs_executable),
        str(worker),
        str(workspace.output_dir.resolve()),
        "--max-image-size",
        str(maximum_image_size),
        "--iterations",
        str(patchmatch_iterations),
    ]
    patchmatch_log = output_dir / "patchmatch.log"
    try:
        with patchmatch_log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"isolated PatchMatch worker failed with exit code {error.returncode}; "
            f"partial evidence kept at {workspace.output_dir}"
        ) from error
    patchmatch_seconds = time.monotonic() - patchmatch_started
    fusion_started = time.monotonic()
    cloud_path = output_dir / "fused_cloud.ply"
    fusion = fuse_colmap_depth_maps(
        workspace.output_dir,
        camera_bundle_path,
        cloud_path,
        minimum_confirmations=1,
        relative_depth_tolerance=0.015,
        voxel_divisions=400,
    )
    fusion_seconds = time.monotonic() - fusion_started
    surface_started = time.monotonic()
    surface_path = output_dir / "surface.ply"
    surface = build_poisson_surface(cloud_path, surface_path)
    surface_seconds = time.monotonic() - surface_started
    report: dict[str, object] = {
        "schema_version": "da3-cad-dense-surface-pipeline-v1",
        "images": str(images_dir.resolve()),
        "masks": str(masks_dir.resolve()),
        "cameras": str(camera_bundle_path.resolve()),
        "workspace": workspace.report,
        "patchmatch_report": str((workspace.output_dir / "patchmatch_report.json").resolve()),
        "patchmatch_log": str(patchmatch_log.resolve()),
        "fusion": fusion.report,
        "surface": surface.report,
        "timing_seconds": {
            "workspace_preparation": preparation_seconds,
            "patchmatch": patchmatch_seconds,
            "fusion": fusion_seconds,
            "surface": surface_seconds,
            "total": time.monotonic() - started,
        },
        "reference_geometry_access": False,
    }
    report_path = output_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DenseSurfacePipelineResult(
        output_dir=output_dir,
        cloud_path=cloud_path,
        surface_path=surface_path,
        report_path=report_path,
        report=report,
    )
