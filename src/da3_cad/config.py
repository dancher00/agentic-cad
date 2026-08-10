"""Validated configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_limit_mb: int = Field(default=8192, ge=512)
    cpu_seconds: int = Field(default=30, ge=1)
    wall_seconds: int = Field(default=45, ge=1)


class Da3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Literal["base", "large"] = "base"
    source_dir: Path = Path("data/upstream/Depth-Anything-3")
    cache_dir: Path = Path("data/hf")
    process_resolution: int = Field(default=504, ge=56, le=2016)
    process_resolution_method: Literal["upper_bound_resize", "lower_bound_resize"] = (
        "upper_bound_resize"
    )
    local_files_only: bool = False
    use_ray_pose: bool = False


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentation_backend: Literal["border-color", "depth-confidence"] = "border-color"
    segmentation_confidence_percentile: float = Field(default=25.0, ge=0.0, le=100.0)
    segmentation_depth_percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    fusion_confidence_percentile: float = Field(default=40.0, ge=0.0, le=100.0)
    minimum_confidence: float | None = None
    depth_alignment_criterion: Literal["projected-local-depth", "fixed-local-plane"] | None = None


class CanonicalizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_enabled: bool = True
    confidence_percentile: float = Field(default=40.0, ge=0.0, le=100.0)
    outlier_enabled: bool = True
    statistical_neighbors: int = Field(default=16, ge=1, le=128)
    statistical_std_ratio: float = Field(default=2.5, ge=0.0, le=10.0)
    outlier_radius_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    outlier_radius_min_neighbors: int = Field(default=3, ge=1, le=128)
    consistency_enabled: bool = True
    consistency_minimum_views: int = Field(default=2, ge=1, le=32)
    consistency_radius_fraction: float = Field(default=0.03, gt=0.0, le=1.0)
    symmetry_detection_enabled: bool = True
    symmetry_completion_enabled: bool = False
    symmetry_tolerance_fraction: float = Field(default=0.04, gt=0.0, le=1.0)
    symmetry_duplicate_radius_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    symmetry_evaluation_points: int = Field(default=4096, ge=256, le=65536)
    orientation_enabled: bool = True
    planar_extent_ratio_threshold: float = Field(default=0.20, gt=0.0, lt=1.0)
    plane_distance_fraction: float = Field(default=0.02, gt=0.0, le=0.25)
    plane_ransac_iterations: int = Field(default=256, ge=16, le=4096)
    eigenvalue_tie_tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    sampling_enabled: bool = True
    point_count: Literal[256] = 256
    normalization_enabled: bool = True


class GeometricFitterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robust_bounds_quantile: float = Field(default=0.005, ge=0.0, lt=0.25)
    minimum_extent: float = Field(default=1e-4, gt=0.0)
    hole_detection_enabled: bool = True
    top_surface_quantile: float = Field(default=0.65, ge=0.5, lt=1.0)
    hole_grid_resolution: int = Field(default=80, ge=24, le=256)
    hole_search_margin_fraction: float = Field(default=0.18, ge=0.05, lt=0.5)
    hole_min_radius_fraction: float = Field(default=0.04, gt=0.0, lt=0.5)
    hole_spacing_multiplier: float = Field(default=4.0, ge=1.0, le=20.0)
    hole_angular_bins: int = Field(default=24, ge=8, le=128)
    hole_min_angular_coverage: float = Field(default=0.5, gt=0.0, le=1.0)
    cylinder_max_aspect: float = Field(default=1.2, ge=1.0, le=3.0)
    cylinder_residual_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    annular_center_tolerance_fraction: float = Field(default=0.05, ge=0.0, le=0.5)


class CadrilleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Literal["sft", "rl"] = "rl"
    cache_dir: Path = Path("data/hf")
    local_files_only: bool = False
    max_new_tokens: int = Field(default=768, ge=1, le=2048)
    attn_implementation: Literal["sdpa"] = "sdpa"
    use_cache: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["stub", "research", "permissive"] = "stub"
    device: str = "cpu"
    seed: int = Field(default=20260810, ge=0)
    debug_artefacts: bool = True
    depth_backend: str = "stub"
    cad_backend: str = "stub"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    da3: Da3Config = Field(default_factory=Da3Config)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    canonicalizer: CanonicalizerConfig = Field(default_factory=CanonicalizerConfig)
    geometric_fitter: GeometricFitterConfig = Field(default_factory=GeometricFitterConfig)
    cadrille: CadrilleConfig = Field(default_factory=CadrilleConfig)


def load_config(
    path: Path | None,
    *,
    device: str | None = None,
    seed: int | None = None,
) -> AppConfig:
    """Load YAML configuration and apply explicit CLI overrides."""

    data: dict[str, object] = {}
    if path is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")
        data = loaded
    if device is not None:
        data["device"] = device
    if seed is not None:
        data["seed"] = seed
    return AppConfig.model_validate(data)
