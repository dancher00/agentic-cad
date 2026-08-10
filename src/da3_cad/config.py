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


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentation_backend: Literal["border-color", "depth-confidence"] = "border-color"
    segmentation_confidence_percentile: float = Field(default=25.0, ge=0.0, le=100.0)
    segmentation_depth_percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    fusion_confidence_percentile: float = Field(default=40.0, ge=0.0, le=100.0)
    minimum_confidence: float | None = None


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
