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


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["stub", "research", "permissive"] = "stub"
    device: str = "cpu"
    seed: int = Field(default=20260810, ge=0)
    debug_artefacts: bool = True
    depth_backend: str = "stub"
    cad_backend: str = "stub"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


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
