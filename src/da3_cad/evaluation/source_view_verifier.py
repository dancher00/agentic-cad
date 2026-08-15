"""Score a CAD mesh by rendering it back into calibrated source views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.integrations.depth_fusion import _read_colmap_array
from da3_cad.models import BoolArray, FloatArray
from da3_cad.observations import load_observations


@dataclass(frozen=True, slots=True)
class SourceViewScore:
    score: float
    silhouette_iou: float
    depth_inlier_fraction: float
    depth_observed_coverage: float
    median_relative_depth_error: float | None
    views: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "silhouette_iou": self.silhouette_iou,
            "depth_inlier_fraction": self.depth_inlier_fraction,
            "depth_observed_coverage": self.depth_observed_coverage,
            "median_relative_depth_error": self.median_relative_depth_error,
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


def decide_source_view_score(
    score: SourceViewScore,
    *,
    minimum_silhouette_iou: float = 0.87,
    minimum_depth_inlier_fraction: float = 0.90,
) -> SourceViewDecision:
    """Conservatively accept only CAD supported by both RGB masks and measured depth."""

    for name, value in (
        ("minimum_silhouette_iou", minimum_silhouette_iou),
        ("minimum_depth_inlier_fraction", minimum_depth_inlier_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    reasons: list[str] = []
    if score.silhouette_iou < minimum_silhouette_iou:
        reasons.append(f"silhouette IoU {score.silhouette_iou:.4f} < {minimum_silhouette_iou:.4f}")
    if score.depth_inlier_fraction < minimum_depth_inlier_fraction:
        reasons.append(
            "depth inlier fraction "
            f"{score.depth_inlier_fraction:.4f} < {minimum_depth_inlier_fraction:.4f}"
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


class SourceViewVerifier:
    """Immutable calibrated evidence and a deterministic Open3D ray-cast scorer."""

    def __init__(
        self,
        views: tuple[_ViewEvidence, ...],
        *,
        silhouette_weight: float,
        depth_inlier_tolerance: float,
    ) -> None:
        if not views:
            raise ValueError("source-view verifier requires at least one view")
        if not 0.0 <= silhouette_weight <= 1.0:
            raise ValueError("silhouette_weight must be in [0, 1]")
        if not 0.0 < depth_inlier_tolerance < 1.0:
            raise ValueError("depth_inlier_tolerance must be in (0, 1)")
        self._views = views
        self._silhouette_weight = silhouette_weight
        self._depth_inlier_tolerance = depth_inlier_tolerance

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
    ) -> SourceViewVerifier:
        if maximum_image_dimension < 64:
            raise ValueError("maximum_image_dimension must be at least 64")
        if input_type not in {"geometric", "photometric"}:
            raise ValueError("input_type must be geometric or photometric")
        observations = load_observations(workspace_dir / "images")
        bundle = load_camera_bundle(camera_bundle_path, observations)
        views: list[_ViewEvidence] = []
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
                )
            )
        return cls(
            tuple(views),
            silhouette_weight=silhouette_weight,
            depth_inlier_tolerance=depth_inlier_tolerance,
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
        rows: list[dict[str, object]] = []
        silhouette_values: list[float] = []
        inlier_numerator = 0
        observed_denominator = 0
        covered_numerator = 0
        all_relative_errors: list[FloatArray] = []
        for view in self._views:
            rays = o3d.core.Tensor(view.rays, dtype=o3d.core.Dtype.Float32)
            hit_depth = scene.cast_rays(rays)["t_hit"].numpy()
            rendered = np.isfinite(hit_depth)
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
        )
