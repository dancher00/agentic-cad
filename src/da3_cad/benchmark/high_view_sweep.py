"""Shared contracts for the bounded N=24/32 geometry diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.camera_controls import pairwise_pose_diagnostics
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment
from da3_cad.benchmark.precision_distribution import (
    PRECISION_THRESHOLDS,
    distribution_summary,
    point_precision_curve,
    threshold_key,
)
from da3_cad.benchmark.scale_oracle import symmetric_sample_chamfer_x1000
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.models import BoolArray, FloatArray

PROTOCOL_VERSION = "da3-cad-high-view-sweep-v1"
NEW_VIEW_COUNTS = (24, 32)
CONTEXT_VIEW_COUNTS = (8, 16, 24, 32)
MATERIAL_GAIN = 0.05
PLATEAU_GAIN = 0.03


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_render_prefix(
    legacy_render_output: Path,
    extended_render_output: Path,
    *,
    prefix_count: int = 16,
) -> dict[str, object]:
    """Prove that extending the master changed neither old cameras nor pixels."""

    legacy_path = legacy_render_output / "render_manifest.json"
    extended_path = extended_render_output / "render_manifest.json"
    legacy = cast(dict[str, Any], json.loads(legacy_path.read_text(encoding="utf-8")))
    extended = cast(dict[str, Any], json.loads(extended_path.read_text(encoding="utf-8")))
    legacy_schedule = cast(list[dict[str, Any]], legacy["camera_schedule"])
    extended_schedule = cast(list[dict[str, Any]], extended["camera_schedule"])
    if len(legacy_schedule) < prefix_count or len(extended_schedule) < prefix_count:
        raise ValueError("render schedules are shorter than the frozen prefix")
    fields = (
        "index",
        "azimuth_deg",
        "elevation_deg",
        "position",
        "intrinsics",
        "world_to_camera",
        "focal_jitter_fraction",
        "blur_radius",
        "jpeg_quality",
        "image_sha256",
        "mask_sha256",
    )
    mismatches: list[dict[str, object]] = []
    for index in range(prefix_count):
        left = legacy_schedule[index]
        right = extended_schedule[index]
        changed = [field for field in fields if left.get(field) != right.get(field)]
        if changed:
            mismatches.append({"view_index": index, "fields": changed})
    if mismatches:
        raise ValueError(f"extended render changed the frozen 16-view prefix: {mismatches}")
    material = [
        {field: extended_schedule[index].get(field) for field in fields}
        for index in range(prefix_count)
    ]
    return {
        "status": "exact-match",
        "prefix_views": prefix_count,
        "fields": list(fields),
        "prefix_sha256": json_digest(material),
        "legacy_manifest": str(legacy_path),
        "legacy_manifest_sha256": sha256_path(legacy_path),
        "extended_manifest": str(extended_path),
        "extended_manifest_sha256": sha256_path(extended_path),
    }


def evaluate_cloud(
    cloud: FusedPointCloud,
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
    gt_surface_decoder: FloatArray,
    gt_extrinsics: FloatArray,
    *,
    seed: int,
    canonicalizer: PointCloudCanonicalizer,
) -> tuple[FloatArray, dict[str, object]]:
    """Apply the frozen reliability/canonical/precision contract to one cloud."""

    selection = select_reliable_points(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
        seed=seed,
    )
    canonical = canonicalizer.run(selection.cloud, seed=seed)
    emitted = canonical.decoder_points
    axis_oracle, transform, axis_chamfer = best_proper_axis_alignment(
        emitted,
        gt_surface_decoder,
    )
    digest = hashlib.sha256(
        np.asarray(emitted, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    return emitted, {
        "status": "complete",
        "decoder_sha256": digest,
        "curves": {
            "emitted": point_precision_curve(emitted, gt_surface_decoder),
            "axis_oracle": point_precision_curve(axis_oracle, gt_surface_decoder),
        },
        "diagnostic_chamfer_x1000": {
            "emitted": symmetric_sample_chamfer_x1000(emitted, gt_surface_decoder),
            "axis_oracle": axis_chamfer,
        },
        "axis_oracle_transform": transform.as_dict(),
        "pose_diagnostics": pairwise_pose_diagnostics(extrinsics, gt_extrinsics),
        "raw_fused_points": len(cloud.points),
        "fusion": cloud.report.as_dict(),
        "reliability_selection": selection.report,
        "canonicalizer": {
            "orientation": (
                canonical.orientation.as_dict() if canonical.orientation is not None else None
            ),
            "normalization": (
                canonical.normalization.as_dict() if canonical.normalization is not None else None
            ),
        },
    }


def _oracle_precision(record: dict[str, Any]) -> float:
    return float(
        record["per_view_depth_oracle"]["curves"]["axis_oracle"]["0.05"]
    )


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["dataset"]), str(record["item_id"])


def _complete_new(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("status") == "complete"]


def _row_aggregate(
    records: list[dict[str, Any]],
    row: str,
) -> dict[str, object]:
    complete = [
        record
        for record in _complete_new(records)
        if cast(dict[str, Any], record[row]).get("status") == "complete"
    ]
    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in complete:
        by_view[int(record["view_count"])].append(record)
    return {
        "records_complete": len(complete),
        "axis_oracle_precision_by_view_count": {
            str(view_count): {
                threshold_key(threshold): distribution_summary(
                    [
                        float(
                            record[row]["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                        )
                        for record in group
                    ]
                )
                for threshold in PRECISION_THRESHOLDS
            }
            for view_count, group in sorted(by_view.items())
        },
        "diagnostic_chamfer_x1000_by_view_count": {
            str(view_count): distribution_summary(
                [
                    float(record[row]["diagnostic_chamfer_x1000"]["axis_oracle"])
                    for record in group
                ]
            )
            for view_count, group in sorted(by_view.items())
        },
    }


def _frozen_records(frozen: dict[str, Any]) -> dict[int, dict[tuple[str, str], dict[str, Any]]]:
    result: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for value in cast(list[dict[str, Any]], frozen["records_detail"]):
        view_count = int(value["view_count"])
        if view_count in (8, 16):
            result[view_count][_record_key(value)] = value
    return dict(result)


def _paired_steps(
    records: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    by_view = _frozen_records(frozen)
    for record in _complete_new(records):
        by_view.setdefault(int(record["view_count"]), {})[_record_key(record)] = record
    missing_counts = [value for value in CONTEXT_VIEW_COUNTS if value not in by_view]
    if missing_counts:
        raise ValueError(f"high-view report is missing context view counts: {missing_counts}")
    common = set(by_view[CONTEXT_VIEW_COUNTS[0]])
    for view_count in CONTEXT_VIEW_COUNTS[1:]:
        common &= set(by_view[view_count])
    ordered = tuple(sorted(common))
    if not ordered:
        raise ValueError("high-view curve has no common valid object")
    steps: dict[str, object] = {}
    for lower, upper in zip(CONTEXT_VIEW_COUNTS, CONTEXT_VIEW_COUNTS[1:], strict=False):
        lower_values = np.asarray(
            [_oracle_precision(by_view[lower][key]) for key in ordered],
            dtype=np.float64,
        )
        upper_values = np.asarray(
            [_oracle_precision(by_view[upper][key]) for key in ordered],
            dtype=np.float64,
        )
        delta = upper_values - lower_values
        steps[f"{lower}->{upper}"] = {
            "common_records": len(ordered),
            "lower": distribution_summary(lower_values.tolist()),
            "upper": distribution_summary(upper_values.tolist()),
            "paired_change": distribution_summary(delta.tolist()),
            "improved": int(np.sum(delta > 0.0)),
            "unchanged": int(np.sum(delta == 0.0)),
            "worsened": int(np.sum(delta < 0.0)),
            "material_gain": bool(float(np.median(delta)) >= MATERIAL_GAIN),
        }
    return steps, ordered


def _plateau_decision(steps: dict[str, Any]) -> dict[str, object]:
    deltas = {
        name: float(cast(dict[str, Any], value["paired_change"])["median"])
        for name, value in steps.items()
    }
    non_monotone = any(value < 0.0 for value in deltas.values())
    final_gain = deltas["24->32"]
    if non_monotone:
        conclusion = "non-monotone-no-numeric-capture-threshold"
        plateau = None
    elif final_gain >= PLATEAU_GAIN:
        conclusion = "at-least-32-upper-bound-unmeasured"
        plateau = None
    else:
        conclusion = "plateau-begins-at-24-under-frozen-rule"
        plateau = 24
    return {
        "paired_median_changes": deltas,
        "material_gain_threshold_absolute": MATERIAL_GAIN,
        "n24_to_n32_non_plateau_threshold_absolute": PLATEAU_GAIN,
        "non_monotone": non_monotone,
        "plateau_view_count": plateau,
        "conclusion": conclusion,
        "causal_limit": (
            "nested max-min schedule changes image count and angular fill together; "
            "this run cannot isolate which causes a gain"
        ),
    }


def _vram_aggregate(records: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for view_count in NEW_VIEW_COUNTS:
        selected = [
            record
            for record in _complete_new(records)
            if int(record["view_count"]) == view_count
        ]
        rows: dict[str, object] = {}
        for key in ("uncalibrated", "exact_pose"):
            lifecycles = [cast(dict[str, Any], record["runtime"][key]) for record in selected]
            allocated = [
                float(value["peak_allocated_bytes"])
                for value in lifecycles
                if value.get("peak_allocated_bytes") is not None
            ]
            reserved = [
                float(value["peak_reserved_bytes"])
                for value in lifecycles
                if value.get("peak_reserved_bytes") is not None
            ]
            rows[key] = {
                "records": len(lifecycles),
                "peak_allocated_bytes": distribution_summary(allocated) if allocated else None,
                "peak_reserved_bytes": distribution_summary(reserved) if reserved else None,
                "all_model_tensors_off_cuda": bool(lifecycles)
                and all(value.get("model_tensors_off_cuda") is True for value in lifecycles),
            }
        result[str(view_count)] = rows
    return result


def aggregate_high_view_report(
    records: list[dict[str, Any]],
    frozen_oracle_report: dict[str, Any],
) -> dict[str, object]:
    """Aggregate complete and failed rows without changing the frozen population."""

    planned_by_view: dict[int, int] = defaultdict(int)
    complete_by_view: dict[int, int] = defaultdict(int)
    failures: list[dict[str, object]] = []
    for record in records:
        view_count = int(record["view_count"])
        planned_by_view[view_count] += 1
        if record.get("status") == "complete":
            complete_by_view[view_count] += 1
        else:
            failures.append(
                {
                    "dataset": record["dataset"],
                    "item_id": record["item_id"],
                    "view_count": view_count,
                    "error": record.get("error"),
                }
            )
    paired_error: str | None = None
    try:
        steps, common = _paired_steps(records, frozen_oracle_report)
    except ValueError as error:
        steps = {}
        common = ()
        paired_error = str(error)
    return {
        "planned_records_by_view_count": {
            str(key): value for key, value in sorted(planned_by_view.items())
        },
        "complete_records_by_view_count": {
            str(key): complete_by_view.get(key, 0) for key in sorted(planned_by_view)
        },
        "failures": failures,
        "rows": {
            row: _row_aggregate(records, row)
            for row in ("uncalibrated", "exact_pose", "per_view_depth_oracle")
        },
        "common_curve_objects": [list(value) for value in common],
        "paired_curve": steps,
        "paired_curve_error": paired_error,
        "plateau_decision": (
            _plateau_decision(cast(dict[str, Any], steps)) if steps else None
        ),
        "vram": _vram_aggregate(records),
    }
