from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from da3_cad.config import AppConfig, VisualHullConfig, load_config


def test_stub_config_loads_with_overrides() -> None:
    config = load_config(
        Path("configs/stub.yaml"),
        device="cpu",
        seed=17,
    )

    assert config.device == "cpu"
    assert config.seed == 17
    assert config.depth_backend == "stub"
    assert config.cad_backend == "stub"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "device": "cpu",
                "seed": 0,
                "surprise": True,
                "depth_backend": "stub",
                "cad_backend": "stub",
                "sandbox": {},
            }
        )


def test_visual_hull_resolution_range_is_validated() -> None:
    with pytest.raises(ValidationError, match="minimum_grid_resolution"):
        VisualHullConfig(grid_resolution=10, minimum_grid_resolution=12)


@pytest.mark.parametrize(
    ("path", "segmentation_backend"),
    [
        ("configs/internet_photo.yaml", "internet-object"),
        ("configs/internet_photo_masked.yaml", "explicit-mask"),
    ],
)
def test_internet_photo_profiles_load(
    path: str,
    segmentation_backend: str,
) -> None:
    config = load_config(Path(path))
    assert config.depth_backend == "da3-large-1.1"
    assert config.cad_backend == "visual-hull"
    assert config.geometry.segmentation_backend == segmentation_backend
    assert config.visual_hull.depth_carving_enabled is True
