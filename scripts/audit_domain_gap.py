#!/usr/bin/env python3
"""Measure the frozen DA3-versus-upstream-Cadrille decoder-input domain gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.domain_gap import measure_domain_gap
from da3_cad.benchmark.splits import read_split
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    validate_mesh,
    verify_official_test_mesh_frame,
)


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("domain-gap audit requires a clean tracked working tree")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite divergent audit artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _load_gt_mesh(path: Path) -> tuple[trimesh.Trimesh, list[list[float]]]:
    mesh = load_mesh(path, TessellationConfig())
    native_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    validation = validate_mesh(mesh)
    if not validation.valid:
        raise ValueError(f"invalid GT mesh {path}: {validation.reason}")
    verify_official_test_mesh_frame(mesh)
    mesh.apply_translation((-0.5, -0.5, -0.5))
    mesh.apply_scale(2.0)  # type: ignore[no-untyped-call]
    return mesh, native_bounds.tolist()


def _trace_summary(path: Path) -> dict[str, object]:
    trace = _json(path)
    orientation = cast(dict[str, Any], trace["orientation"])
    stages = cast(list[dict[str, Any]], trace["stages"])
    return {
        "orientation_method": orientation["method"],
        "planar_extent_ratio": orientation["planar_extent_ratio"],
        "planar_threshold": orientation["planar_threshold"],
        "orientation_determinant": orientation["determinant"],
        "stage_point_counts": {str(stage["name"]): int(stage["point_count"]) for stage in stages},
        "stage_inferred_points": {
            str(stage["name"]): int(stage["inferred_points"]) for stage in stages
        },
    }


def _canonical_artifacts(root: Path) -> tuple[tuple[str, str, int, Path], ...]:
    artifacts: list[tuple[str, str, int, Path]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in sorted(root.rglob("decoder_input.npy")):
        parts = path.relative_to(root).parts
        if len(parts) != 5 or not parts[2].startswith("n"):
            raise ValueError(f"unexpected canonical artifact layout: {path}")
        dataset, item_id, view_component = parts[:3]
        view_count = int(view_component[1:])
        key = (dataset, item_id, view_count)
        if key in seen:
            raise ValueError(f"duplicate canonical artifact for {key}")
        seen.add(key)
        artifacts.append((dataset, item_id, view_count, path))
    return tuple(artifacts)


def _nested(record: dict[str, object], *path: str) -> Any:
    value: Any = record["diagnostics"]
    for component in path:
        value = value[component]
    return value


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("aggregate inputs must be non-empty and finite")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _path_stats(records: list[dict[str, object]], *path: str) -> dict[str, float | int]:
    return _stats([float(_nested(record, *path)) for record in records])


def _relation_summary(
    records: list[dict[str, object]], frame: str, *, oracle: bool = False
) -> dict[str, object]:
    prefix = ("gt_oracle_proper_axis_frame", "metrics") if oracle else (frame,)
    return {
        "sample_chamfer_x1000": _path_stats(records, *prefix, "sample_chamfer_x1000"),
        "surface_coverage_fraction": {
            threshold: _path_stats(
                records,
                *prefix,
                "thresholds_decoder_coordinates",
                threshold,
                "surface_coverage_fraction",
            )
            for threshold in ("0.02", "0.05", "0.10")
        },
        "point_precision_fraction": {
            threshold: _path_stats(
                records,
                *prefix,
                "thresholds_decoder_coordinates",
                threshold,
                "point_precision_fraction",
            )
            for threshold in ("0.02", "0.05", "0.10")
        },
        "exact_surface_distance_mean": _path_stats(
            records, *prefix, "point_to_exact_surface", "mean"
        ),
        "absolute_normal_residual_mean": _path_stats(
            records, *prefix, "normal_residual_absolute", "mean"
        ),
        "interior_fraction_beyond_0.02": _path_stats(
            records, *prefix, "signed_side", "interior_fraction_beyond_0.02"
        ),
        "exterior_fraction_beyond_0.02": _path_stats(
            records, *prefix, "signed_side", "exterior_fraction_beyond_0.02"
        ),
        "near_surface_fraction_within_0.02": _path_stats(
            records, *prefix, "signed_side", "near_surface_fraction_within_0.02"
        ),
    }


def _transform_key(record: dict[str, object]) -> str:
    transform = cast(
        dict[str, Any],
        _nested(record, "gt_oracle_proper_axis_frame", "transform"),
    )
    permutation = ",".join(str(value) for value in transform["permutation"])
    signs = ",".join(str(value) for value in transform["signs"])
    return f"permutation={permutation};signs={signs}"


def _canonicalizer_method(record: dict[str, object]) -> str:
    canonicalizer = cast(dict[str, object], record["canonicalizer"])
    return str(canonicalizer["orientation_method"])


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    ratios = [
        float(
            _nested(
                record,
                "orientation_diagnostic",
                "oracle_to_emitted_ratio",
            )
        )
        for record in records
    ]
    density_ratios = [
        float(_nested(record, "density", "da3_canonical_fps", "nearest_neighbor_mean"))
        / float(_nested(record, "density", "gt_upstream_fps", "nearest_neighbor_mean"))
        for record in records
    ]
    transforms = Counter(_transform_key(record) for record in records)
    methods = Counter(_canonicalizer_method(record) for record in records)
    return {
        "records": len(records),
        "density": {
            "gt_nearest_neighbor_mean": _path_stats(
                records, "density", "gt_upstream_fps", "nearest_neighbor_mean"
            ),
            "da3_nearest_neighbor_mean": _path_stats(
                records, "density", "da3_canonical_fps", "nearest_neighbor_mean"
            ),
            "da3_to_gt_nearest_neighbor_mean_ratio": _stats(density_ratios),
            "gt_occupied_voxel_fraction_of_points": _path_stats(
                records,
                "density",
                "gt_upstream_fps",
                "occupied_voxel_fraction_of_points",
            ),
            "da3_occupied_voxel_fraction_of_points": _path_stats(
                records,
                "density",
                "da3_canonical_fps",
                "occupied_voxel_fraction_of_points",
            ),
        },
        "surface_relation": {
            "gt_upstream_control": _relation_summary(records, "gt_upstream_frame"),
            "da3_emitted_frame": _relation_summary(records, "emitted_frame"),
            "da3_gt_axis_oracle": _relation_summary(records, "", oracle=True),
        },
        "orientation_diagnostic": {
            "oracle_to_emitted_chamfer_ratio": _stats(ratios),
            "oracle_strictly_improves_fraction": float(
                np.mean(np.asarray(ratios, dtype=np.float64) < 1.0)
            ),
            "identity_transform_fraction": transforms.get("permutation=0,1,2;signs=1,1,1", 0)
            / len(records),
            "selected_transform_histogram": dict(transforms),
        },
        "canonicalizer_orientation_methods": dict(methods),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-canonical-root",
        type=Path,
        default=Path("data/benchmark_runs/pilot_33c0003/canonical"),
    )
    parser.add_argument(
        "--gt-control-input-root",
        type=Path,
        default=Path("data/benchmark_runs/gt_cloud_control_69f71fc/inputs"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/domain_gap/report.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    artifacts = _canonical_artifacts(args.pilot_canonical_root)
    if not artifacts:
        raise ValueError("no frozen DA3 canonical decoder inputs were found")
    if not args.dry_run:
        _clean_repository(root)
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    expected_items = {
        (dataset, item_id) for dataset, item_ids in splits.items() for item_id in item_ids
    }
    actual_items = {(dataset, item_id) for dataset, item_id, _, _ in artifacts}
    if actual_items != expected_items:
        raise ValueError("canonical artifacts do not cover exactly the frozen pilot items")
    counts = Counter(view_count for _, _, view_count, _ in artifacts)
    if not set(counts).issubset({1, 2, 4, 8, 16}):
        raise ValueError("canonical artifacts contain an unapproved view count")
    summary: dict[str, object] = {
        "repository_commit": repository_commit(root),
        "records": len(artifacts),
        "items": len(actual_items),
        "records_by_view_count": {str(key): counts[key] for key in sorted(counts)},
        "inference": "none; frozen point-cloud artifacts only",
    }
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        return 0

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for index, (dataset, item_id, view_count, da3_path) in enumerate(artifacts, 1):
        gt_path = args.gt_control_input_root / dataset / f"{item_id}.npz"
        mesh_path = args.data_root / dataset / f"{item_id}.stl"
        trace_path = da3_path.parent / "canonicalizer_trace.json"
        da3_stored = np.load(da3_path, allow_pickle=False)
        if da3_stored.shape != (1, 256, 3) or da3_stored.dtype != np.float32:
            raise ValueError(f"unexpected DA3 decoder tensor contract: {da3_path}")
        with np.load(gt_path, allow_pickle=False) as gt_payload:
            if set(gt_payload.files) != {
                "surface_points",
                "surface_face_indices",
                "fps_indices",
                "decoder_points",
            }:
                raise ValueError(f"unexpected GT-control NPZ contract: {gt_path}")
            gt_surface_native = np.asarray(gt_payload["surface_points"], dtype=np.float64)
            gt_decoder = np.asarray(gt_payload["decoder_points"], dtype=np.float32)
        if gt_surface_native.shape != (8192, 3) or gt_decoder.shape != (256, 3):
            raise ValueError(f"unexpected GT-control array shapes: {gt_path}")
        gt_mesh, native_bounds = _load_gt_mesh(mesh_path)
        diagnostics = measure_domain_gap(
            np.asarray(da3_stored[0], dtype=np.float32),
            gt_decoder,
            (gt_surface_native - 0.5) * 2.0,
            gt_mesh,
        )
        records.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "view_count": view_count,
                "diagnostics": diagnostics,
                "canonicalizer": _trace_summary(trace_path),
                "artifacts": {
                    "da3_decoder_input": str(da3_path),
                    "da3_decoder_input_sha256": _sha256(da3_path),
                    "canonicalizer_trace": str(trace_path),
                    "canonicalizer_trace_sha256": _sha256(trace_path),
                    "gt_control_npz": str(gt_path),
                    "gt_control_npz_sha256": _sha256(gt_path),
                    "gt_mesh": str(mesh_path),
                    "gt_mesh_sha256": _sha256(mesh_path),
                    "gt_mesh_native_bbox": native_bounds,
                },
            }
        )
        if index % 10 == 0 or index == len(artifacts):
            print(json.dumps({"completed": index, "total": len(artifacts)}))

    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_dataset_view: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        view_count = int(cast(int, record["view_count"]))
        dataset = str(record["dataset"])
        by_view[view_count].append(record)
        by_dataset_view[(dataset, view_count)].append(record)

    report: dict[str, object] = {
        "schema_version": "da3-cad-domain-gap-audit-v1",
        "status": "measured-frozen-da3-domain-gap",
        **summary,
        "coordinate_contract": {
            "gt_surface_and_mesh": ("released [0,1]^3 values mapped by (xyz-0.5)*2"),
            "gt_decoder": ("frozen upstream 8192 -> fixed-start FP32 FPS 256 -> (xyz-0.5)*2"),
            "da3_decoder": ("frozen canonicalizer output, float32 shape [1,256,3] in [-1,1]^3"),
            "metric_role": ("decoder-input distribution diagnostic, not evaluator CD/IoU"),
        },
        "surface_proxy": (
            "8192 frozen area-weighted GT samples; coverage is the fraction "
            "within a decoder-coordinate threshold of any decoder-input point"
        ),
        "orientation_oracle_policy": {
            "transforms": ("24 orientation-preserving signed axis permutations"),
            "uses_gt": True,
            "allowed_in_inference": False,
            "allowed_in_candidate_selection": False,
            "purpose": ("separate discrete orientation error from coverage/noise"),
        },
        "aggregates": {
            "overall": _aggregate(records),
            "by_view_count": {
                str(view_count): _aggregate(by_view[view_count]) for view_count in sorted(by_view)
            },
            "by_dataset_and_view_count": [
                {
                    "dataset": dataset,
                    "view_count": view_count,
                    **_aggregate(group),
                }
                for (dataset, view_count), group in sorted(by_dataset_view.items())
            ],
        },
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "long_campaign_started": False,
            "readme_updated": False,
            "gt_oracle_is_diagnostic_only": True,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "records": len(records),
                "runtime_seconds": report["runtime_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
