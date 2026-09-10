"""Score a CAD mesh by rendering it back into calibrated source views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, sobel

from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.integrations.depth_fusion import _read_colmap_array
from da3_cad.models import BoolArray, FloatArray, IntArray, UInt8Array
from da3_cad.observations import load_observations


@dataclass(frozen=True, slots=True)
class SourceViewScore:
    score: float
    silhouette_iou: float
    depth_inlier_fraction: float
    depth_observed_coverage: float
    median_relative_depth_error: float | None
    views: tuple[dict[str, object], ...]
    appearance_edge_precision: float | None = None
    appearance_edge_recall: float | None = None
    appearance_edge_pixels: int = 0
    rendered_geometry_edge_pixels: int = 0
    input_view_count: int = 0
    used_view_count: int = 0
    rejected_views: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "silhouette_iou": self.silhouette_iou,
            "depth_inlier_fraction": self.depth_inlier_fraction,
            "depth_observed_coverage": self.depth_observed_coverage,
            "median_relative_depth_error": self.median_relative_depth_error,
            "appearance_edge_precision": self.appearance_edge_precision,
            "appearance_edge_recall": self.appearance_edge_recall,
            "appearance_edge_pixels": self.appearance_edge_pixels,
            "rendered_geometry_edge_pixels": self.rendered_geometry_edge_pixels,
            "input_view_count": self.input_view_count,
            "used_view_count": self.used_view_count,
            "rejected_views": list(self.rejected_views),
            "views": list(self.views),
        }


@dataclass(frozen=True, slots=True)
class SourceViewDecision:
    decision: str
    reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.decision == "ACCEPT"

    def as_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "accepted": self.accepted, "reasons": list(self.reasons)}


def decide_source_view_progress(
    candidate: SourceViewScore,
    previous: SourceViewScore | None,
    *,
    minimum_score_improvement: float = 0.001,
    minimum_topology_recall_improvement: float = 0.05,
    maximum_topology_score_regression: float = 0.05,
    minimum_appearance_edge_precision: float = 0.45,
) -> tuple[bool, str]:
    """Admit either geometric progress or a bounded, RGB-supported topology step."""

    if previous is None:
        return True, "initial-valid-solid"
    if candidate.score >= previous.score + minimum_score_improvement:
        return True, "source-view-score-improvement"
    previous_recall = previous.appearance_edge_recall or 0.0
    candidate_recall = candidate.appearance_edge_recall or 0.0
    candidate_precision = candidate.appearance_edge_precision or 0.0
    topology_supported = (
        candidate.appearance_edge_pixels > 0
        and candidate_recall >= previous_recall + minimum_topology_recall_improvement
        and candidate_precision >= minimum_appearance_edge_precision
        and candidate.score >= previous.score - maximum_topology_score_regression
        and candidate.silhouette_iou >= previous.silhouette_iou - maximum_topology_score_regression
        and candidate.depth_inlier_fraction
        >= previous.depth_inlier_fraction - maximum_topology_score_regression
    )
    if topology_supported:
        return True, "bounded-appearance-topology-improvement"
    return False, "no-supported-geometric-or-topology-progress"


def appearance_topology_regressions(
    baseline: SourceViewScore,
    candidate: SourceViewScore,
    *,
    maximum_regression: float = 0.005,
) -> tuple[str, ...]:
    """Report visible-edge evidence lost by a supposedly cosmetic rewrite."""

    if not 0.0 <= maximum_regression <= 1.0:
        raise ValueError("maximum appearance topology regression must be in [0, 1]")
    regressions: list[str] = []
    for metric in ("appearance_edge_precision", "appearance_edge_recall"):
        baseline_value = getattr(baseline, metric)
        candidate_value = getattr(candidate, metric)
        if baseline_value is None:
            continue
        if candidate_value is None:
            regressions.append(f"{metric} missing")
        elif candidate_value + maximum_regression < baseline_value:
            regressions.append(f"{metric} {baseline_value:.4f}->{candidate_value:.4f}")
    return tuple(regressions)


def decide_source_view_score(
    score: SourceViewScore,
    *,
    minimum_silhouette_iou: float = 0.87,
    minimum_depth_inlier_fraction: float = 0.90,
    minimum_appearance_edge_precision: float = 0.45,
    minimum_appearance_edge_recall: float = 0.12,
    minimum_appearance_edge_pixels: int = 128,
) -> SourceViewDecision:
    """Accept only CAD supported by silhouettes, depth and visible geometry edges."""

    for name, value in (
        ("minimum_silhouette_iou", minimum_silhouette_iou),
        ("minimum_depth_inlier_fraction", minimum_depth_inlier_fraction),
        ("minimum_appearance_edge_precision", minimum_appearance_edge_precision),
        ("minimum_appearance_edge_recall", minimum_appearance_edge_recall),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if minimum_appearance_edge_pixels < 0:
        raise ValueError("minimum_appearance_edge_pixels must be non-negative")
    reasons: list[str] = []
    if score.silhouette_iou < minimum_silhouette_iou:
        reasons.append(f"silhouette IoU {score.silhouette_iou:.4f} < {minimum_silhouette_iou:.4f}")
    if score.depth_inlier_fraction < minimum_depth_inlier_fraction:
        reasons.append(
            "depth inlier fraction "
            f"{score.depth_inlier_fraction:.4f} < {minimum_depth_inlier_fraction:.4f}"
        )
    if score.appearance_edge_pixels >= minimum_appearance_edge_pixels:
        edge_precision = score.appearance_edge_precision or 0.0
        edge_recall = score.appearance_edge_recall or 0.0
        if edge_precision < minimum_appearance_edge_precision:
            reasons.append(
                "appearance-edge precision "
                f"{edge_precision:.4f} < {minimum_appearance_edge_precision:.4f}"
            )
        if edge_recall < minimum_appearance_edge_recall:
            reasons.append(
                f"appearance-edge recall {edge_recall:.4f} < {minimum_appearance_edge_recall:.4f}"
            )
    return SourceViewDecision(
        decision="ABSTAIN" if reasons else "ACCEPT",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class _ViewEvidence:
    name: str
    rays: FloatArray
    mask: BoolArray
    depth: FloatArray
    appearance_edges: BoolArray


def _depth_view_admission(
    depth: FloatArray,
    mask: BoolArray,
    *,
    minimum_pixels: int,
    minimum_mask_fraction: float,
) -> tuple[bool, dict[str, object]]:
    """Admit only views with enough cross-view-confirmed depth to verify CAD."""

    values = np.asarray(depth, dtype=np.float32)
    target = np.asarray(mask, dtype=np.bool_)
    if values.shape != target.shape:
        raise ValueError("depth admission requires matching depth and mask shapes")
    if minimum_pixels < 1:
        raise ValueError("minimum depth pixels must be positive")
    if not 0.0 < minimum_mask_fraction <= 1.0:
        raise ValueError("minimum depth mask fraction must be in (0, 1]")
    mask_pixels = int(target.sum())
    measured = target & np.isfinite(values) & (values > 0.0)
    measured_pixels = int(measured.sum())
    fraction = float(measured_pixels / max(mask_pixels, 1))
    reasons: list[str] = []
    if measured_pixels < minimum_pixels:
        reasons.append(f"{measured_pixels} measured depth pixels < {minimum_pixels}")
    if fraction < minimum_mask_fraction:
        reasons.append(
            f"measured depth covers {fraction:.4f} of target mask < {minimum_mask_fraction:.4f}"
        )
    return not reasons, {
        "mask_pixels": mask_pixels,
        "measured_depth_pixels": measured_pixels,
        "measured_mask_fraction": fraction,
        "minimum_depth_pixels": minimum_pixels,
        "minimum_depth_mask_fraction": minimum_mask_fraction,
        "reasons": reasons,
    }


def _appearance_edges(image: UInt8Array, mask: BoolArray) -> BoolArray:
    """Return high-contrast boundaries away from the segmentation silhouette."""

    values = np.asarray(image)
    binary = np.asarray(mask, dtype=np.bool_)
    if values.shape[:2] != binary.shape:
        raise ValueError("RGB image and mask must have the same shape")
    grayscale = (
        values.astype(np.float32).mean(axis=2) if values.ndim == 3 else values.astype(np.float32)
    )
    gradient = np.hypot(sobel(grayscale, axis=0), sobel(grayscale, axis=1))
    interior = binary_erosion(binary, iterations=3)
    selected = gradient[interior]
    if len(selected) == 0:
        return np.zeros_like(binary)
    threshold = max(float(np.percentile(selected, 88.0)), 8.0)
    return np.asarray(interior & (gradient >= threshold), dtype=np.bool_)


def _rendered_geometry_edges(
    depth: FloatArray,
    rendered: BoolArray,
    target_mask: BoolArray,
    surface_ids: IntArray | None = None,
) -> BoolArray:
    """Extract visible CAD boundaries without exposing mesh triangulation.

    Occupancy and depth discontinuities recover silhouettes and cavities. A
    smooth-surface id test additionally recovers sharp creases whose depth is
    continuous in the image (for example a cylinder-to-cone transition).
    Triangle ids are deliberately not used: tessellation is an implementation
    detail, not CAD topology.
    """

    hit_depth = np.asarray(depth, dtype=np.float32)
    occupancy = np.asarray(rendered, dtype=np.bool_)
    mask = np.asarray(target_mask, dtype=np.bool_)
    if hit_depth.shape != occupancy.shape or occupancy.shape != mask.shape:
        raise ValueError("rendered depth, occupancy and target mask must have the same shape")
    rendered_surface_ids: IntArray | None = None
    if surface_ids is not None:
        rendered_surface_ids = np.asarray(surface_ids, dtype=np.int64)
        if rendered_surface_ids.shape != occupancy.shape:
            raise ValueError("rendered surface ids must match the depth image")
    edges = np.zeros_like(occupancy)
    for axis in (0, 1):
        left_slices = [slice(None), slice(None)]
        right_slices = [slice(None), slice(None)]
        left_slices[axis] = slice(1, None)
        right_slices[axis] = slice(None, -1)
        left = tuple(left_slices)
        right = tuple(right_slices)
        occupancy_change = occupancy[left] != occupancy[right]
        both = occupancy[left] & occupancy[right]
        relative_jump = np.zeros_like(occupancy_change, dtype=np.float32)
        relative_jump[both] = np.abs(hit_depth[left][both] - hit_depth[right][both])
        relative_jump[both] /= np.maximum(
            np.minimum(hit_depth[left][both], hit_depth[right][both]),
            1e-6,
        )
        surface_change = np.zeros_like(occupancy_change, dtype=np.bool_)
        if rendered_surface_ids is not None:
            surface_change = both & (rendered_surface_ids[left] != rendered_surface_ids[right])
        boundary = occupancy_change | (relative_jump > 0.008) | surface_change
        edges[left] |= boundary
        edges[right] |= boundary
    return np.asarray(edges & binary_erosion(mask, iterations=3), dtype=np.bool_)


def _smooth_face_groups(
    mesh: trimesh.Trimesh,
    *,
    maximum_smooth_angle_degrees: float = 15.0,
) -> IntArray:
    """Group local tessellation facets into renderable smooth surfaces."""

    if not 0.0 < maximum_smooth_angle_degrees < 180.0:
        raise ValueError("maximum smooth angle must be in (0, 180)")
    face_count = len(mesh.faces)
    parents = np.arange(face_count, dtype=np.int64)

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = int(parents[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    threshold = float(np.deg2rad(maximum_smooth_angle_degrees))
    for (left, right), angle in zip(
        np.asarray(mesh.face_adjacency, dtype=np.int64),
        np.asarray(mesh.face_adjacency_angles, dtype=np.float64),
        strict=True,
    ):
        if np.isfinite(angle) and angle <= threshold:
            union(int(left), int(right))
    roots = np.asarray([find(index) for index in range(face_count)], dtype=np.int64)
    _, groups = np.unique(roots, return_inverse=True)
    return np.asarray(groups, dtype=np.int64)


class SourceViewVerifier:
    """Immutable calibrated evidence and a deterministic Open3D ray-cast scorer."""

    def __init__(
        self,
        views: tuple[_ViewEvidence, ...],
        *,
        silhouette_weight: float,
        depth_inlier_tolerance: float,
        appearance_edge_tolerance_pixels: float,
        input_view_count: int | None = None,
        rejected_views: tuple[dict[str, object], ...] = (),
    ) -> None:
        if not views:
            raise ValueError("source-view verifier requires at least one view")
        if not 0.0 <= silhouette_weight <= 1.0:
            raise ValueError("silhouette_weight must be in [0, 1]")
        if not 0.0 < depth_inlier_tolerance < 1.0:
            raise ValueError("depth_inlier_tolerance must be in (0, 1)")
        if appearance_edge_tolerance_pixels <= 0.0:
            raise ValueError("appearance_edge_tolerance_pixels must be positive")
        self._views = views
        self._silhouette_weight = silhouette_weight
        self._depth_inlier_tolerance = depth_inlier_tolerance
        self._appearance_edge_tolerance_pixels = appearance_edge_tolerance_pixels
        self._input_view_count = len(views) if input_view_count is None else input_view_count
        self._rejected_views = rejected_views
        if self._input_view_count != len(views) + len(rejected_views):
            raise ValueError("source-view admission counts do not match")

    @classmethod
    def from_colmap_workspace(
        cls,
        workspace_dir: Path,
        camera_bundle_path: Path,
        *,
        maximum_image_dimension: int = 240,
        input_type: str = "geometric",
        silhouette_weight: float = 0.65,
        depth_inlier_tolerance: float = 0.03,
        appearance_edge_tolerance_pixels: float = 2.0,
        minimum_observed_depth_pixels: int = 128,
        minimum_observed_depth_mask_fraction: float = 0.10,
    ) -> SourceViewVerifier:
        if maximum_image_dimension < 64:
            raise ValueError("maximum_image_dimension must be at least 64")
        if input_type not in {"geometric", "photometric"}:
            raise ValueError("input_type must be geometric or photometric")
        observations = load_observations(workspace_dir / "images")
        bundle = load_camera_bundle(camera_bundle_path, observations)
        views: list[_ViewEvidence] = []
        rejected_views: list[dict[str, object]] = []
        for index, observation in enumerate(observations.images):
            depth = _read_colmap_array(
                workspace_dir
                / "stereo"
                / "depth_maps"
                / f"{observation.relative_path}.{input_type}.bin"
            )
            with Image.open(workspace_dir / "masks" / observation.relative_path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
            if depth.shape != mask.shape:
                raise ValueError(
                    f"source-view depth/mask shape mismatch: {observation.relative_path}"
                )
            admitted, admission = _depth_view_admission(
                depth,
                mask,
                minimum_pixels=minimum_observed_depth_pixels,
                minimum_mask_fraction=minimum_observed_depth_mask_fraction,
            )
            if not admitted:
                rejected_views.append(
                    {
                        "image": observation.relative_path,
                        **admission,
                    }
                )
                continue
            height, width = mask.shape
            scale = min(1.0, maximum_image_dimension / max(height, width))
            output_height = max(1, int(round(height * scale)))
            output_width = max(1, int(round(width * scale)))
            row_indices = np.clip(
                np.rint(np.linspace(0, height - 1, output_height)).astype(np.int64),
                0,
                height - 1,
            )
            column_indices = np.clip(
                np.rint(np.linspace(0, width - 1, output_width)).astype(np.int64),
                0,
                width - 1,
            )
            resized_mask = mask[np.ix_(row_indices, column_indices)]
            resized_depth = np.asarray(depth[np.ix_(row_indices, column_indices)], dtype=np.float32)
            with Image.open(workspace_dir / "images" / observation.relative_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != mask.shape:
                raise ValueError(
                    f"source-view RGB/mask shape mismatch: {observation.relative_path}"
                )
            resized_rgb = rgb[np.ix_(row_indices, column_indices)]
            intrinsic = np.asarray(bundle.intrinsics[index], dtype=np.float64).copy()
            scale_x = output_width / width
            scale_y = output_height / height
            intrinsic[0, :] *= scale_x
            intrinsic[1, :] *= scale_y
            columns, rows = np.meshgrid(
                np.arange(output_width, dtype=np.float64),
                np.arange(output_height, dtype=np.float64),
            )
            camera_directions = np.stack(
                (
                    (columns - intrinsic[0, 2]) / intrinsic[0, 0],
                    (rows - intrinsic[1, 2]) / intrinsic[1, 1],
                    np.ones_like(columns),
                ),
                axis=-1,
            )
            extrinsic = np.asarray(bundle.extrinsics[index], dtype=np.float64)
            centre = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
            world_directions = camera_directions @ extrinsic[:3, :3]
            origins = np.broadcast_to(centre, world_directions.shape)
            rays = np.concatenate((origins, world_directions), axis=-1).astype(np.float32)
            views.append(
                _ViewEvidence(
                    name=observation.relative_path,
                    rays=rays,
                    mask=np.asarray(resized_mask, dtype=np.bool_),
                    depth=resized_depth,
                    appearance_edges=_appearance_edges(resized_rgb, resized_mask),
                )
            )
        return cls(
            tuple(views),
            silhouette_weight=silhouette_weight,
            depth_inlier_tolerance=depth_inlier_tolerance,
            appearance_edge_tolerance_pixels=appearance_edge_tolerance_pixels,
            input_view_count=len(observations.images),
            rejected_views=tuple(rejected_views),
        )

    def score(self, mesh: trimesh.Trimesh) -> SourceViewScore:
        if mesh.is_empty or len(mesh.faces) == 0:
            raise ValueError("source-view verifier requires a non-empty triangle mesh")
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("source-view CAD verification requires open3d>=0.19") from error
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        smooth_face_groups = _smooth_face_groups(mesh)
        rows: list[dict[str, object]] = []
        silhouette_values: list[float] = []
        inlier_numerator = 0
        observed_denominator = 0
        covered_numerator = 0
        all_relative_errors: list[FloatArray] = []
        appearance_edge_pixels = 0
        geometry_edge_pixels = 0
        matched_appearance_edges = 0
        matched_geometry_edges = 0
        for view in self._views:
            rays = o3d.core.Tensor(view.rays, dtype=o3d.core.Dtype.Float32)
            ray_hits = scene.cast_rays(rays)
            hit_depth = ray_hits["t_hit"].numpy()
            rendered = np.isfinite(hit_depth)
            primitive_ids = ray_hits["primitive_ids"].numpy().astype(np.int64)
            rendered_surface_ids = np.full(rendered.shape, -1, dtype=np.int64)
            rendered_surface_ids[rendered] = smooth_face_groups[primitive_ids[rendered]]
            geometry_edges = _rendered_geometry_edges(
                hit_depth,
                rendered,
                view.mask,
                rendered_surface_ids,
            )
            observed_edges = view.appearance_edges
            observed_count = int(observed_edges.sum())
            geometry_count = int(geometry_edges.sum())
            if observed_count and geometry_count:
                observed_distance = distance_transform_edt(~observed_edges)
                geometry_distance = distance_transform_edt(~geometry_edges)
                observed_matches = int(
                    (
                        geometry_distance[observed_edges] <= self._appearance_edge_tolerance_pixels
                    ).sum()
                )
                geometry_matches = int(
                    (
                        observed_distance[geometry_edges] <= self._appearance_edge_tolerance_pixels
                    ).sum()
                )
            else:
                observed_matches = 0
                geometry_matches = 0
            appearance_edge_pixels += observed_count
            geometry_edge_pixels += geometry_count
            matched_appearance_edges += observed_matches
            matched_geometry_edges += geometry_matches
            intersection = int(np.logical_and(rendered, view.mask).sum())
            union = int(np.logical_or(rendered, view.mask).sum())
            silhouette_iou = intersection / max(union, 1)
            silhouette_values.append(silhouette_iou)
            observed_valid = view.mask & np.isfinite(view.depth) & (view.depth > 0.0)
            both = observed_valid & rendered
            relative = np.abs(hit_depth[both] - view.depth[both])
            relative /= np.maximum(view.depth[both], 1e-12)
            observed = int(observed_valid.sum())
            inliers = int((relative <= self._depth_inlier_tolerance).sum())
            observed_denominator += observed
            covered_numerator += int(both.sum())
            inlier_numerator += inliers
            if len(relative):
                all_relative_errors.append(relative)
            rows.append(
                {
                    "image": view.name,
                    "silhouette_iou": silhouette_iou,
                    "observed_depth_pixels": observed,
                    "rendered_depth_coverage": float(both.sum() / max(observed, 1)),
                    "depth_inlier_fraction": float(inliers / max(observed, 1)),
                    "appearance_edge_pixels": observed_count,
                    "rendered_geometry_edge_pixels": geometry_count,
                    "appearance_edge_precision": float(geometry_matches / max(geometry_count, 1)),
                    "appearance_edge_recall": float(observed_matches / max(observed_count, 1)),
                }
            )
        silhouette_iou = float(np.mean(silhouette_values))
        depth_inlier_fraction = float(inlier_numerator / max(observed_denominator, 1))
        depth_coverage = float(covered_numerator / max(observed_denominator, 1))
        score = (
            self._silhouette_weight * silhouette_iou
            + (1.0 - self._silhouette_weight) * depth_inlier_fraction
        )
        relative_values = (
            np.concatenate(all_relative_errors) if all_relative_errors else np.empty(0)
        )
        return SourceViewScore(
            score=score,
            silhouette_iou=silhouette_iou,
            depth_inlier_fraction=depth_inlier_fraction,
            depth_observed_coverage=depth_coverage,
            median_relative_depth_error=(
                float(np.median(relative_values)) if len(relative_values) else None
            ),
            views=tuple(rows),
            appearance_edge_precision=(
                float(matched_geometry_edges / geometry_edge_pixels)
                if geometry_edge_pixels
                else None
            ),
            appearance_edge_recall=(
                float(matched_appearance_edges / appearance_edge_pixels)
                if appearance_edge_pixels
                else None
            ),
            appearance_edge_pixels=appearance_edge_pixels,
            rendered_geometry_edge_pixels=geometry_edge_pixels,
            input_view_count=self._input_view_count,
            used_view_count=len(self._views),
            rejected_views=self._rejected_views,
        )
