"""Validation and compact reporting for the real DA3 Phase B smoke gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(run_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((run_dir / "geometry_report.json").read_text(encoding="utf-8")),
    )


def _exact_npz_match(left: Path, right: Path) -> dict[str, bool]:
    with (
        np.load(left, allow_pickle=False) as left_data,
        np.load(right, allow_pickle=False) as right_data,
    ):
        if set(left_data.files) != set(right_data.files):
            raise ValueError(f"NPZ fields differ: {left} versus {right}")
        return {
            field: bool(np.array_equal(left_data[field], right_data[field]))
            for field in sorted(left_data.files)
        }


def _summarize_run(run_dir: Path, repeat_dir: Path) -> dict[str, object]:
    report = _load_report(run_dir)
    repeat = _load_report(repeat_dir)
    model = report["da3"]["model"]
    lifecycle = report["da3"]["lifecycle"]
    roundtrips = report["runtime_unprojection_roundtrip"]
    camera_exact = _exact_npz_match(
        run_dir / "artefacts" / "camera_prediction.npz",
        repeat_dir / "artefacts" / "camera_prediction.npz",
    )
    cloud_exact = _exact_npz_match(
        run_dir / "artefacts" / "fused_cloud.npz",
        repeat_dir / "artefacts" / "fused_cloud.npz",
    )
    primary_ply = run_dir / "artefacts" / "fused_cloud.ply"
    repeat_ply = repeat_dir / "artefacts" / "fused_cloud.ply"
    ply_sha256 = _sha256(primary_ply)
    repeat_ply_sha256 = _sha256(repeat_ply)

    if report["input"]["digest"] != repeat["input"]["digest"]:
        raise ValueError("repeat used different input bytes")
    if not all(camera_exact.values()) or not all(cloud_exact.values()):
        raise ValueError(f"{model['key']} repeat is not array-exact")
    if ply_sha256 != repeat_ply_sha256:
        raise ValueError(f"{model['key']} repeat PLY differs")
    if lifecycle["model_tensors_off_cuda"] is not True:
        raise ValueError(f"{model['key']} retained model tensors on CUDA")
    if "sm_120" not in lifecycle["compiled_architectures"]:
        raise ValueError("torch build does not contain sm_120")
    if report["da3"]["checkpoint_file"]["sha256_verified"] is not True:
        raise ValueError(f"{model['key']} checkpoint bytes were not verified")
    if report["cloud"]["finite"] is not True or report["cloud"]["point_count"] <= 0:
        raise ValueError(f"{model['key']} did not produce a finite non-empty cloud")

    return {
        "model": model,
        "checkpoint_file": report["da3"]["checkpoint_file"],
        "input_digest": report["input"]["digest"],
        "input_views": report["da3"]["input_views"],
        "process_resolution": report["da3"]["process_resolution"],
        "output_shapes": report["da3"]["output_shapes"],
        "depth": report["da3"]["depth"],
        "confidence": report["da3"]["confidence"],
        "is_metric": report["da3"]["is_metric"],
        "pose": report["runtime_pose_validation"],
        "unprojection_roundtrip": {
            "max_pixel_abs_error": max(item["max_pixel_abs_error"] for item in roundtrips),
            "max_z_depth_abs_error": max(item["max_z_depth_abs_error"] for item in roundtrips),
        },
        "segmentation": report["segmentation"],
        "fusion": report["fusion"],
        "cloud": report["cloud"],
        "lifecycle": lifecycle,
        "determinism_repeat": {
            "camera_arrays_exact": camera_exact,
            "cloud_arrays_exact": cloud_exact,
            "ply_sha256": ply_sha256,
            "repeat_ply_sha256": repeat_ply_sha256,
            "exact": True,
        },
    }


def build_da3_smoke_summary(
    *,
    base_dir: Path,
    base_repeat_dir: Path,
    large_dir: Path,
    large_repeat_dir: Path,
    repository_commit: str,
) -> dict[str, object]:
    """Build the committed stop-point evidence from real ignored run artifacts."""

    base = _summarize_run(base_dir, base_repeat_dir)
    large = _summarize_run(large_dir, large_repeat_dir)
    if base["model"]["key"] != "base" or large["model"]["key"] != "large":  # type: ignore[index]
        raise ValueError("BASE/LARGE run directories are swapped or mislabeled")
    return {
        "schema_version": "1.0",
        "status": "real-smoke-not-quality-benchmark",
        "repository_commit": repository_commit,
        "input": "committed sample_data/plate/views (four distinct rendered views)",
        "notes": [
            "timings are a compatibility smoke, not throughput claims",
            "another process occupied the same GPU; current-process torch peaks remain valid",
            "post-unload allocator residual is reported separately from zero model CUDA tensors",
            "global scale is unresolved and no CAD decoder ran",
        ],
        "runs": {"base": base, "large": large},
    }
