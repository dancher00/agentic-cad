"""Confidence-aware DA3 priors for external Gaussian-surface optimizers.

The module deliberately does not depend on a particular Gaussian Splatting
implementation.  External projects exchange one versioned NPZ bundle and call
one small, differentiable loss function from their own training loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from da3_cad.geometry.coverage import camera_centers_world
from da3_cad.models import DepthPrediction, FloatArray

AlignmentMode = Literal["none", "median-log-scale"]


@dataclass(frozen=True, slots=True)
class DepthPriorBundle:
    """Portable depth/confidence bundle indexed by external view name."""

    depth: FloatArray
    confidence: FloatArray
    intrinsics: FloatArray
    extrinsics: FloatArray
    image_names: tuple[str, ...]
    backend: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.image_names)
        if count == 0 or len(set(self.image_names)) != count:
            raise ValueError("depth-prior image_names must be non-empty and unique")
        if self.depth.ndim != 3 or self.depth.shape[0] != count:
            raise ValueError("depth-prior depth must have shape (N,H,W)")
        if self.confidence.shape != self.depth.shape:
            raise ValueError("depth-prior confidence must match depth")
        if self.intrinsics.shape != (count, 3, 3):
            raise ValueError("depth-prior intrinsics must have shape (N,3,3)")
        if self.extrinsics.shape != (count, 4, 4):
            raise ValueError("depth-prior extrinsics must have shape (N,4,4)")
        if not np.isfinite(self.intrinsics).all() or not np.isfinite(self.extrinsics).all():
            raise ValueError("depth-prior cameras must be finite")
        valid_depth = np.isfinite(self.depth) & (self.depth > 0.0)
        if not np.any(valid_depth):
            raise ValueError("depth-prior bundle contains no finite positive depth")
        if not np.any(np.isfinite(self.confidence)):
            raise ValueError("depth-prior bundle contains no finite confidence")

    @classmethod
    def from_prediction(
        cls,
        prediction: DepthPrediction,
        image_names: tuple[str, ...],
        *,
        metadata: dict[str, object] | None = None,
    ) -> DepthPriorBundle:
        """Adapt a DA3 prediction without changing its camera convention."""

        if prediction.confidence is None:
            raise ValueError("confidence-aware prior requires backend confidence")
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
        if extrinsics.shape[1:] == (3, 4):
            homogeneous = np.repeat(np.eye(4, dtype=np.float32)[None], len(extrinsics), axis=0)
            homogeneous[:, :3, :] = extrinsics
            extrinsics = homogeneous
        return cls(
            depth=np.asarray(prediction.depth, dtype=np.float32),
            confidence=np.asarray(prediction.confidence, dtype=np.float32),
            intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32),
            extrinsics=extrinsics,
            image_names=image_names,
            backend=prediction.backend,
            metadata={} if metadata is None else dict(metadata),
        )

    def save(self, path: Path) -> None:
        """Atomically save a bundle without pickle-backed arrays."""

        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        payload = {
            "schema_version": "da3-cad-depth-prior-v1",
            "backend": self.backend,
            "metadata": self.metadata,
        }
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                depth=self.depth.astype(np.float32, copy=False),
                confidence=self.confidence.astype(np.float32, copy=False),
                intrinsics=self.intrinsics.astype(np.float32, copy=False),
                extrinsics=self.extrinsics.astype(np.float32, copy=False),
                image_names=np.asarray(self.image_names, dtype=np.str_),
                manifest=np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_),
            )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: Path) -> DepthPriorBundle:
        """Load and validate a versioned bundle."""

        with np.load(path, allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest"].item()))
            if manifest.get("schema_version") != "da3-cad-depth-prior-v1":
                raise ValueError("unsupported depth-prior schema")
            return cls(
                depth=np.asarray(payload["depth"], dtype=np.float32),
                confidence=np.asarray(payload["confidence"], dtype=np.float32),
                intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
                extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
                image_names=tuple(str(item) for item in payload["image_names"].tolist()),
                backend=str(manifest["backend"]),
                metadata=dict(manifest.get("metadata", {})),
            )

    def index(self, image_name: str) -> int:
        try:
            return self.image_names.index(image_name)
        except ValueError as error:
            raise KeyError(f"depth prior has no view named {image_name!r}") from error

    def torch_view(
        self,
        image_name: str,
        *,
        height: int,
        width: int,
        device: Any,
    ) -> tuple[Any, Any]:
        """Return depth/confidence tensors resized to one renderer view."""

        if height <= 0 or width <= 0:
            raise ValueError("target depth-prior raster must be positive")
        import torch
        import torch.nn.functional as functional

        index = self.index(image_name)
        depth = torch.as_tensor(self.depth[index], dtype=torch.float32, device=device)[None, None]
        confidence = torch.as_tensor(self.confidence[index], dtype=torch.float32, device=device)[
            None, None
        ]
        if depth.shape[-2:] != (height, width):
            depth = functional.interpolate(
                depth,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            confidence = functional.interpolate(
                confidence,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return depth[0], confidence[0]


@dataclass(frozen=True, slots=True)
class NerfCameraDataset:
    """Images and world-to-camera matrices from a NeRF transforms JSON."""

    image_paths: tuple[Path, ...]
    image_names: tuple[str, ...]
    intrinsics: FloatArray
    extrinsics: FloatArray

    def __post_init__(self) -> None:
        count = len(self.image_paths)
        if count == 0 or len(self.image_names) != count:
            raise ValueError("NeRF camera dataset must contain named images")
        if len(set(self.image_names)) != count:
            raise ValueError("NeRF image stems must be unique")
        if self.intrinsics.shape != (count, 3, 3):
            raise ValueError("NeRF intrinsics must have shape (N,3,3)")
        if self.extrinsics.shape != (count, 4, 4):
            raise ValueError("NeRF extrinsics must have shape (N,4,4)")

    def subset(self, indices: tuple[int, ...]) -> NerfCameraDataset:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("camera subset requires at least one index")
        if len(set(int(item) for item in selected)) != len(selected):
            raise ValueError("camera subset indices must be unique")
        if int(selected.min()) < 0 or int(selected.max()) >= len(self.image_paths):
            raise ValueError("camera subset index is out of range")
        return NerfCameraDataset(
            image_paths=tuple(self.image_paths[int(index)] for index in selected),
            image_names=tuple(self.image_names[int(index)] for index in selected),
            intrinsics=self.intrinsics[selected].copy(),
            extrinsics=self.extrinsics[selected].copy(),
        )


def load_nerf_camera_dataset(
    transforms_path: Path,
    *,
    image_extension: str = ".png",
) -> NerfCameraDataset:
    """Load Blender/NeRF camera-to-world transforms as COLMAP-style w2c."""

    source = transforms_path.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("NeRF transforms JSON contains no frames")
    image_paths: list[Path] = []
    image_names: list[str] = []
    intrinsics: list[FloatArray] = []
    extrinsics: list[FloatArray] = []
    for frame in frames:
        relative = str(frame["file_path"])
        raw_path = (source.parent / relative).resolve()
        image_path = raw_path if raw_path.suffix else raw_path.with_suffix(image_extension)
        if not image_path.is_file():
            raise FileNotFoundError(f"NeRF frame image is missing: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        if "camera_intrinsics" in frame:
            intrinsic = np.asarray(frame["camera_intrinsics"], dtype=np.float64)
        else:
            fov_x = float(payload["camera_angle_x"])
            focal = 0.5 * width / np.tan(0.5 * fov_x)
            intrinsic = np.asarray(
                [[focal, 0.0, 0.5 * (width - 1)], [0.0, focal, 0.5 * (height - 1)], [0, 0, 1]],
                dtype=np.float64,
            )
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"invalid camera intrinsics for {image_path}")
        camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
            raise ValueError(f"invalid camera transform for {image_path}")
        camera_to_world = camera_to_world.copy()
        camera_to_world[:3, 1:3] *= -1.0
        world_to_camera = np.linalg.inv(camera_to_world)
        image_paths.append(image_path)
        image_names.append(image_path.stem)
        intrinsics.append(intrinsic)
        extrinsics.append(world_to_camera)
    return NerfCameraDataset(
        image_paths=tuple(image_paths),
        image_names=tuple(image_names),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
    )


def _fibonacci_sphere(count: int = 2048) -> FloatArray:
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = golden * indices
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def spherical_camera_coverage(
    extrinsics: FloatArray,
    *,
    object_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_cone_degrees: float = 35.0,
) -> float:
    """Return the fraction of a sampled sphere covered by camera directions."""

    centers = camera_centers_world(extrinsics)
    center = np.asarray(object_center, dtype=np.float64)
    directions = centers - center[None]
    lengths = np.linalg.norm(directions, axis=1)
    valid = np.isfinite(directions).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-9)
    if not np.any(valid):
        raise ValueError("camera coverage requires at least one usable camera direction")
    directions = directions[valid] / lengths[valid, None]
    sphere = _fibonacci_sphere()
    cosine = float(np.cos(np.deg2rad(surface_cone_degrees)))
    return float((np.max(sphere @ directions.T, axis=1) >= cosine).mean())


def greedy_camera_subset(
    extrinsics: FloatArray,
    count: int,
    *,
    object_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_cone_degrees: float = 35.0,
) -> tuple[int, ...]:
    """Select views by marginal spherical coverage, then angular separation."""

    centers = camera_centers_world(extrinsics)
    center = np.asarray(object_center, dtype=np.float64)
    directions = centers - center[None]
    lengths = np.linalg.norm(directions, axis=1)
    valid = np.isfinite(directions).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-9)
    available = [int(index) for index in np.flatnonzero(valid)]
    if count <= 0 or count > len(available):
        raise ValueError("requested camera subset size is outside the usable view count")
    directions[valid] /= lengths[valid, None]
    sphere = _fibonacci_sphere()
    cosine = float(np.cos(np.deg2rad(surface_cone_degrees)))
    selected: list[int] = []
    while len(selected) < count:
        best_index = available[0]
        best_score: tuple[float, float, int] | None = None
        for index in available:
            if index in selected:
                continue
            candidate = [*selected, index]
            candidate_directions = directions[np.asarray(candidate, dtype=np.int64)]
            covered = np.max(sphere @ candidate_directions.T, axis=1) >= cosine
            separation = (
                2.0
                if not selected
                else float(1.0 - np.max(directions[index] @ directions[selected].T))
            )
            score = (float(covered.mean()), separation, -index)
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        selected.append(best_index)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class DepthPriorLoss:
    """Differentiable loss terms plus detached diagnostics."""

    total: Any
    depth: Any
    gradient: Any
    valid_fraction: float
    consensus_fraction: float
    log_scale_offset: float


def confidence_aware_depth_prior_loss(
    rendered_depth: Any,
    prior_depth: Any,
    confidence: Any,
    *,
    foreground_mask: Any | None = None,
    edge_strength: Any | None = None,
    alignment: AlignmentMode = "median-log-scale",
    gradient_weight: float = 0.25,
    confidence_floor_quantile: float = 0.2,
    confidence_ceiling_quantile: float = 0.9,
    edge_attenuation: float = 0.75,
    huber_delta: float = 0.05,
    minimum_consensus_log_error: float = 0.15,
) -> DepthPriorLoss:
    """Robust DA3 supervision for a rendered Gaussian-surface depth map.

    Confidence is normalized per view, discontinuities may be attenuated, and
    outliers are trimmed from detached residual statistics.  This prevents a
    noisy monocular prior from becoming an unconditional geometric target.
    """

    import torch
    import torch.nn.functional as functional

    if rendered_depth.shape != prior_depth.shape or rendered_depth.shape != confidence.shape:
        raise ValueError("rendered depth, prior depth and confidence must have equal shapes")
    if rendered_depth.ndim not in {2, 3}:
        raise ValueError("depth-prior loss expects (H,W) or (C,H,W) tensors")
    if not 0.0 <= confidence_floor_quantile < confidence_ceiling_quantile <= 1.0:
        raise ValueError("confidence quantiles must be ordered within [0,1]")
    if gradient_weight < 0.0 or huber_delta <= 0.0:
        raise ValueError("loss weights must be non-negative and huber_delta positive")

    rendered = rendered_depth.float()
    prior = prior_depth.to(device=rendered.device, dtype=rendered.dtype)
    conf = confidence.to(device=rendered.device, dtype=rendered.dtype)
    valid = torch.isfinite(rendered) & torch.isfinite(prior) & torch.isfinite(conf)
    valid = valid & (rendered > 1e-8) & (prior > 1e-8)
    if foreground_mask is not None:
        if foreground_mask.shape != rendered.shape:
            raise ValueError("foreground mask must match depth raster")
        valid = valid & foreground_mask.to(device=rendered.device, dtype=torch.bool)
    zero = rendered.sum() * 0.0
    if not bool(valid.any().item()):
        return DepthPriorLoss(zero, zero, zero, 0.0, 0.0, 0.0)

    valid_confidence = conf[valid].detach()
    low = torch.quantile(valid_confidence, confidence_floor_quantile)
    high = torch.quantile(valid_confidence, confidence_ceiling_quantile)
    span = high - low
    if float(span.item()) <= 1e-8:
        weights = torch.ones_like(conf)
    else:
        weights = ((conf - low) / span).clamp(0.0, 1.0)
        weights = 0.05 + 0.95 * weights
    if edge_strength is not None:
        if edge_strength.shape != rendered.shape:
            raise ValueError("edge strength must match depth raster")
        edges = edge_strength.to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)
        weights = weights * (1.0 - edge_attenuation * edges)
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    log_rendered = torch.log(rendered.clamp_min(1e-8))
    log_prior = torch.log(prior.clamp_min(1e-8))
    raw_residual = log_rendered - log_prior
    if alignment == "median-log-scale":
        offset = torch.median(raw_residual[valid].detach())
    elif alignment == "none":
        offset = torch.zeros((), device=rendered.device, dtype=rendered.dtype)
    else:
        raise ValueError(f"unsupported depth-prior alignment: {alignment}")
    residual = raw_residual - offset

    residual_detached = residual[valid].detach()
    residual_center = torch.median(residual_detached)
    mad = torch.median(torch.abs(residual_detached - residual_center))
    cutoff = max(minimum_consensus_log_error, 3.0 * 1.4826 * float(mad.item()))
    consensus = torch.abs(residual.detach() - residual_center) <= cutoff
    consensus_valid = valid & consensus
    retained_fraction = float(consensus_valid.float().sum().item() / valid.float().sum().item())
    if retained_fraction < 0.8:
        consensus_valid = valid
    weights = torch.where(consensus_valid, weights, torch.zeros_like(weights))

    depth_penalty = functional.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=huber_delta,
        reduction="none",
    )
    weight_sum = weights.sum().clamp_min(1e-8)
    depth_loss = (depth_penalty * weights).sum() / weight_sum

    dx = residual[..., :, 1:] - residual[..., :, :-1]
    dy = residual[..., 1:, :] - residual[..., :-1, :]
    weight_x = torch.minimum(weights[..., :, 1:], weights[..., :, :-1])
    weight_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
    penalty_x = functional.smooth_l1_loss(
        dx,
        torch.zeros_like(dx),
        beta=huber_delta,
        reduction="none",
    )
    penalty_y = functional.smooth_l1_loss(
        dy,
        torch.zeros_like(dy),
        beta=huber_delta,
        reduction="none",
    )
    gradient_loss = ((penalty_x * weight_x).sum() + (penalty_y * weight_y).sum()) / (
        weight_x.sum() + weight_y.sum()
    ).clamp_min(1e-8)
    total = depth_loss + gradient_weight * gradient_loss
    return DepthPriorLoss(
        total=total,
        depth=depth_loss,
        gradient=gradient_loss,
        valid_fraction=float(valid.float().mean().item()),
        consensus_fraction=float(consensus_valid.float().sum().item() / valid.float().sum().item()),
        log_scale_offset=float(offset.item()),
    )
