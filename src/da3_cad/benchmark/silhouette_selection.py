"""GT-blind candidate reranking with input masks and recovered cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from da3_cad.benchmark.candidates import CandidateArtifact, CandidateSelection
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    normalize_prediction_mesh,
    validate_mesh,
)
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import BoolArray, DepthPrediction, FloatArray


@dataclass(frozen=True, slots=True)
class ViewSilhouetteScore:
    view_index: int
    target_pixels: int
    rendered_pixels: int
    intersection_pixels: int
    union_pixels: int
    iou: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "target_pixels": self.target_pixels,
            "rendered_pixels": self.rendered_pixels,
            "intersection_pixels": self.intersection_pixels,
            "union_pixels": self.union_pixels,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class CandidateSilhouetteScore:
    index: int
    valid: bool
    mean_iou: float | None
    median_iou: float | None
    trimmed_mean_iou: float | None
    views: tuple[ViewSilhouetteScore, ...]
    invalid_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "valid": self.valid,
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "trimmed_mean_iou": self.trimmed_mean_iou,
            "invalid_reason": self.invalid_reason,
            "views": [view.as_dict() for view in self.views],
        }


@dataclass(frozen=True, slots=True)
class MultiviewSelectionRecord:
    index: int
    valid: bool
    input_cd_squared: float | None
    input_cd_ratio_to_best: float | None
    silhouette_trimmed_mean_iou: float | None
    silhouette_error: float | None
    combined_cost: float | None
    invalid_selection_cost: str | None
    invalid_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "valid": self.valid,
            "input_cd_squared": self.input_cd_squared,
            "input_cd_ratio_to_best": self.input_cd_ratio_to_best,
            "silhouette_trimmed_mean_iou": self.silhouette_trimmed_mean_iou,
            "silhouette_error": self.silhouette_error,
            "combined_cost": self.combined_cost,
            "invalid_selection_cost": self.invalid_selection_cost,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True, slots=True)
class MultiviewCandidateSelection:
    selected_index: int | None
    silhouette_weight: float
    input_selection: CandidateSelection
    records: tuple[MultiviewSelectionRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": ("minimum input-CD ratio plus weighted trimmed multiview silhouette error"),
            "formula": (
                "cost = input_cd / best_valid_input_cd + silhouette_weight * (1 - trimmed_mean_iou)"
            ),
            "ground_truth_access": False,
            "mask_access": "reconstruction input masks only",
            "selected_index": self.selected_index,
            "silhouette_weight": self.silhouette_weight,
            "invalid_candidate_cost": "infinity",
            "input_selection": self.input_selection.as_dict(),
            "records": [record.as_dict() for record in self.records],
        }


def _orientation_stage_points(canonical: CanonicalCloud) -> FloatArray:
    stages = [stage for stage in canonical.stages if stage.name == "orientation"]
    if len(stages) != 1:
        raise ValueError("canonical trace must contain exactly one orientation stage")
    points = np.asarray(stages[0].points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("canonical orientation stage must contain finite XYZ points")
    if not np.isfinite(points).all():
        raise ValueError("canonical orientation stage contains non-finite points")
    return points


def candidate_mesh_in_world(
    mesh_input: Path | trimesh.Trimesh,
    canonical: CanonicalCloud,
    *,
    tessellation: TessellationConfig | None = None,
) -> trimesh.Trimesh:
    """Map a bbox-normalized CAD mesh back to the DA3/COLMAP world frame."""

    config = tessellation if tessellation is not None else TessellationConfig()
    mesh = load_mesh(mesh_input, config)
    validation = validate_mesh(mesh)
    if not validation.valid:
        raise ValueError(validation.reason or "candidate mesh validation failed")
    normalized = normalize_prediction_mesh(mesh)
    oriented_input = _orientation_stage_points(canonical)
    minimum = oriented_input.min(axis=0)
    maximum = oriented_input.max(axis=0)
    midpoint = (minimum + maximum) / 2.0
    largest_extent = float(np.max(maximum - minimum))
    oriented_vertices = (
        np.asarray(normalized.vertices, dtype=np.float64) * largest_extent + midpoint
    )

    orientation = canonical.orientation
    if orientation is None:
        world_vertices = oriented_vertices
    else:
        axes = np.column_stack(
            tuple(np.asarray(axis, dtype=np.float64) for axis in orientation.axes_world)
        )
        center = np.asarray(orientation.center_world, dtype=np.float64)
        if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
            raise ValueError("canonical orientation axes are not orthonormal")
        world_vertices = oriented_vertices @ axes.T + center

    result = normalized.copy()
    result.vertices = world_vertices
    return result


def render_opencv_silhouette(
    mesh: trimesh.Trimesh,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    image_shape: tuple[int, int],
) -> BoolArray:
    """Rasterize a mesh silhouette using DA3's OpenCV x-right/y-down convention."""

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("silhouette image dimensions must be positive")
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("silhouette intrinsics must be one finite 3x3 matrix")
    world_to_camera = as_homogeneous_extrinsic(extrinsics)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    camera = homogeneous @ world_to_camera.T
    camera_xyz = camera[:, :3] / camera[:, 3:4]
    projected_h = camera_xyz @ intrinsic.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    depths = camera_xyz[:, 2]

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    rendered_faces = 0
    for face in np.asarray(mesh.faces, dtype=np.int64):
        if not np.all(depths[face] > 1e-6):
            continue
        polygon = projected[face]
        if not np.isfinite(polygon).all():
            continue
        draw.polygon([tuple(float(value) for value in point) for point in polygon], fill=255)
        rendered_faces += 1
    if rendered_faces == 0:
        raise ValueError("candidate has no triangles fully in front of the camera")
    return (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.bool_)


def _trimmed_mean(values: FloatArray, trim_fraction: float) -> float:
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("silhouette trim fraction must be in [0,0.5)")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(np.floor(len(ordered) * trim_fraction))
    retained = ordered[trim : len(ordered) - trim] if trim > 0 else ordered
    if len(retained) == 0:
        raise ValueError("silhouette trimming removed every view")
    return float(np.mean(retained, dtype=np.float64))


def score_candidate_silhouettes(
    candidates: tuple[CandidateArtifact, ...],
    canonical: CanonicalCloud,
    prediction: DepthPrediction,
    target_masks: BoolArray,
    *,
    trim_fraction: float = 0.1,
    output_root: Path | None = None,
    tessellation: TessellationConfig | None = None,
) -> tuple[CandidateSilhouetteScore, ...]:
    """Compare projected CAD silhouettes with masks used to build the input cloud."""

    masks = np.asarray(target_masks, dtype=np.bool_)
    expected = prediction.depth.shape
    if masks.shape != expected:
        raise ValueError(f"silhouette masks must have shape {expected}, got {masks.shape}")
    if len(candidates) == 0 or tuple(item.index for item in candidates) != tuple(
        range(len(candidates))
    ):
        raise ValueError("silhouette candidates must be consecutively indexed")
    if any(not np.any(mask) for mask in masks):
        raise ValueError("every silhouette target mask must contain foreground pixels")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=False)

    results: list[CandidateSilhouetteScore] = []
    for candidate in candidates:
        if candidate.mesh is None or candidate.invalid_reason is not None:
            results.append(
                CandidateSilhouetteScore(
                    index=candidate.index,
                    valid=False,
                    mean_iou=None,
                    median_iou=None,
                    trimmed_mean_iou=None,
                    views=(),
                    invalid_reason=candidate.invalid_reason or "candidate mesh is missing",
                )
            )
            continue
        try:
            mesh = candidate_mesh_in_world(
                candidate.mesh,
                canonical,
                tessellation=tessellation,
            )
            candidate_output = (
                output_root / f"candidate_{candidate.index:02d}"
                if output_root is not None
                else None
            )
            if candidate_output is not None:
                candidate_output.mkdir(parents=True, exist_ok=False)
            views: list[ViewSilhouetteScore] = []
            for view_index in range(len(masks)):
                rendered = render_opencv_silhouette(
                    mesh,
                    prediction.intrinsics[view_index],
                    prediction.extrinsics[view_index],
                    (int(masks.shape[1]), int(masks.shape[2])),
                )
                target = masks[view_index]
                intersection = int(np.logical_and(rendered, target).sum())
                union = int(np.logical_or(rendered, target).sum())
                rendered_pixels = int(rendered.sum())
                target_pixels = int(target.sum())
                if union == 0 or rendered_pixels == 0 or target_pixels == 0:
                    raise ValueError(f"view {view_index} produced an empty silhouette metric")
                views.append(
                    ViewSilhouetteScore(
                        view_index=view_index,
                        target_pixels=target_pixels,
                        rendered_pixels=rendered_pixels,
                        intersection_pixels=intersection,
                        union_pixels=union,
                        iou=intersection / union,
                        precision=intersection / rendered_pixels,
                        recall=intersection / target_pixels,
                    )
                )
                if candidate_output is not None:
                    Image.fromarray(rendered.astype(np.uint8) * 255, mode="L").save(
                        candidate_output / f"view_{view_index:03d}.png"
                    )
            ious = np.asarray([view.iou for view in views], dtype=np.float64)
            results.append(
                CandidateSilhouetteScore(
                    index=candidate.index,
                    valid=True,
                    mean_iou=float(np.mean(ious, dtype=np.float64)),
                    median_iou=float(np.median(ious)),
                    trimmed_mean_iou=_trimmed_mean(ious, trim_fraction),
                    views=tuple(views),
                    invalid_reason=None,
                )
            )
        except Exception as error:
            results.append(
                CandidateSilhouetteScore(
                    index=candidate.index,
                    valid=False,
                    mean_iou=None,
                    median_iou=None,
                    trimmed_mean_iou=None,
                    views=(),
                    invalid_reason=(
                        f"candidate silhouette failed: {type(error).__name__}: {error}"
                    ),
                )
            )
    return tuple(results)


def select_by_input_and_silhouette(
    input_selection: CandidateSelection,
    silhouette_scores: tuple[CandidateSilhouetteScore, ...],
    *,
    silhouette_weight: float = 1.0,
) -> MultiviewCandidateSelection:
    """Rerank input-valid candidates without opening a ground-truth CAD model."""

    if not np.isfinite(silhouette_weight) or silhouette_weight < 0.0:
        raise ValueError("silhouette weight must be finite and non-negative")
    if len(input_selection.records) != len(silhouette_scores):
        raise ValueError("input and silhouette candidate counts differ")
    if tuple(score.index for score in silhouette_scores) != tuple(
        record.index for record in input_selection.records
    ):
        raise ValueError("input and silhouette candidate indices differ")
    finite_input_costs = [
        float(record.input_cd_squared)
        for record in input_selection.records
        if record.valid and record.input_cd_squared is not None
    ]
    if not finite_input_costs:
        return MultiviewCandidateSelection(
            selected_index=None,
            silhouette_weight=silhouette_weight,
            input_selection=input_selection,
            records=tuple(
                MultiviewSelectionRecord(
                    index=record.index,
                    valid=False,
                    input_cd_squared=None,
                    input_cd_ratio_to_best=None,
                    silhouette_trimmed_mean_iou=None,
                    silhouette_error=None,
                    combined_cost=None,
                    invalid_selection_cost="infinity",
                    invalid_reason=record.invalid_reason,
                )
                for record in input_selection.records
            ),
        )
    baseline = max(min(finite_input_costs), 1e-12)
    combined: list[tuple[float, int]] = []
    records: list[MultiviewSelectionRecord] = []
    for input_record, silhouette in zip(
        input_selection.records,
        silhouette_scores,
        strict=True,
    ):
        if (
            input_record.valid
            and input_record.input_cd_squared is not None
            and silhouette.valid
            and silhouette.trimmed_mean_iou is not None
        ):
            ratio = float(input_record.input_cd_squared) / baseline
            silhouette_error = 1.0 - float(silhouette.trimmed_mean_iou)
            cost = ratio + silhouette_weight * silhouette_error
            combined.append((cost, input_record.index))
            records.append(
                MultiviewSelectionRecord(
                    index=input_record.index,
                    valid=True,
                    input_cd_squared=float(input_record.input_cd_squared),
                    input_cd_ratio_to_best=ratio,
                    silhouette_trimmed_mean_iou=float(silhouette.trimmed_mean_iou),
                    silhouette_error=silhouette_error,
                    combined_cost=cost,
                    invalid_selection_cost=None,
                    invalid_reason=None,
                )
            )
        else:
            reasons = list(
                dict.fromkeys(
                    reason
                    for reason in (input_record.invalid_reason, silhouette.invalid_reason)
                    if reason
                )
            )
            records.append(
                MultiviewSelectionRecord(
                    index=input_record.index,
                    valid=False,
                    input_cd_squared=input_record.input_cd_squared,
                    input_cd_ratio_to_best=None,
                    silhouette_trimmed_mean_iou=silhouette.trimmed_mean_iou,
                    silhouette_error=None,
                    combined_cost=None,
                    invalid_selection_cost="infinity",
                    invalid_reason="; ".join(reasons) or "candidate metric is unavailable",
                )
            )
    return MultiviewCandidateSelection(
        selected_index=min(combined)[1] if combined else None,
        silhouette_weight=silhouette_weight,
        input_selection=input_selection,
        records=tuple(records),
    )
