"""Empirical decoder-normalization parity checks on external test meshes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh

Float64Array = npt.NDArray[np.float64]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_bounds(path: Path) -> Float64Array:
    loaded = trimesh.load_mesh(path, process=False)
    if len(loaded.vertices) == 0:
        raise ValueError(f"not a non-empty triangle mesh: {path}")
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"invalid mesh bounds: {path}")
    return bounds


def classify_bounds(bounds: Float64Array, *, tolerance: float = 5e-4) -> dict[str, object]:
    """Test competing normalization hypotheses for one mesh bbox."""

    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2, 3):
        raise ValueError("bounds must have shape (2,3)")
    extents = values[1] - values[0]
    center = values.mean(axis=0)
    if np.any(extents <= 0.0):
        raise ValueError("mesh bounds are degenerate")
    mapped = (values - 0.5) * 2.0
    return {
        "bounds": values.tolist(),
        "extents": extents.tolist(),
        "center": center.tolist(),
        "largest_extent": float(extents.max()),
        "largest_extent_is_one": bool(np.isclose(extents.max(), 1.0, atol=tolerance)),
        "bbox_center_is_half": bool(np.allclose(center, 0.5, atol=tolerance)),
        "per_axis_unit_extent": bool(np.allclose(extents, 1.0, atol=tolerance)),
        "corner_anchored_at_zero": bool(np.allclose(values[0], 0.0, atol=tolerance)),
        "decoder_mapped_bounds": mapped.tolist(),
        "decoder_mapped_extents": (mapped[1] - mapped[0]).tolist(),
        "short_axis_ratios": (extents / extents.max()).tolist(),
    }


def audit_manifest(manifest_path: Path, *, data_root: Path | None = None) -> dict[str, object]:
    """Verify bytes and aggregate hypotheses across pinned external samples."""

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = data_root or Path(str(manifest["data_root"]))
    records: list[dict[str, object]] = []
    for dataset in manifest["datasets"]:
        dataset_name = str(dataset["name"])
        for sample in dataset["samples"]:
            relative_path = str(sample["path"])
            mesh_path = root / dataset_name / relative_path
            if not mesh_path.is_file():
                raise FileNotFoundError(f"external audit mesh is missing: {mesh_path}")
            actual_sha = _sha256(mesh_path)
            expected_sha = str(sample["sha256"])
            if actual_sha != expected_sha:
                raise ValueError(
                    f"SHA-256 mismatch for {mesh_path}: expected {expected_sha}, got {actual_sha}"
                )
            classification = classify_bounds(_mesh_bounds(mesh_path))
            records.append(
                {
                    "dataset": dataset_name,
                    "repo_id": str(dataset["repo_id"]),
                    "revision": str(dataset["revision"]),
                    "path": relative_path,
                    "sha256": actual_sha,
                    **classification,
                }
            )

    isotropic_centered = all(
        bool(record["largest_extent_is_one"]) and bool(record["bbox_center_is_half"])
        for record in records
    )
    per_axis = all(bool(record["per_axis_unit_extent"]) for record in records)
    corner_anchored = all(bool(record["corner_anchored_at_zero"]) for record in records)
    if isotropic_centered and not per_axis and not corner_anchored:
        conclusion = "isotropic-largest-extent-with-centered-short-axes"
    else:
        conclusion = "unresolved-or-mixed"
    return {
        "schema_version": "1.0",
        "manifest": manifest_path.as_posix(),
        "sample_count": len(records),
        "hypotheses": {
            "isotropic_largest_extent_centered": isotropic_centered,
            "per_axis_scaling": per_axis,
            "corner_anchored": corner_anchored,
        },
        "conclusion": conclusion,
        "decoder_test_path_transform": "d = (xyz - 0.5) * 2",
        "arbitrary_cloud_canonical_transform": (
            "u = (p - bbox_center) / max_bbox_extent + 0.5; d = (u - 0.5) * 2"
        ),
        "records": records,
    }
