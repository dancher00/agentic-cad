from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from da3_cad.config import (
    AdaptiveViewSelectionConfig,
    AppConfig,
    AxialShellLoopConfig,
    InteriorEllipseConfig,
    VisualHullConfig,
    load_config,
)


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
    assert config.cad_backend == "construction-grammar"
    assert config.geometry.segmentation_backend == segmentation_backend
    assert config.geometry.depth_alignment_criterion == "fixed-local-plane"
    assert config.geometry.depth_alignment_selection == "auto"
    assert config.visual_hull.depth_carving_enabled is True
    assert config.view_selection.enabled is True
    assert config.view_selection.minimum_views == 24
    assert config.view_selection.maximum_views == 40
    assert config.view_selection.minimum_mask_fraction == 0.001
    assert config.pose_admission.refinement_enabled is True
    assert config.pose_admission.global_recentering_maximum_translation_fraction == 2.5
    assert config.pose_admission.refinement_maximum_translation_fraction == 1.5
    assert config.pose_admission.refinement_maximum_rotation_degrees == 15.0
    assert config.pose_admission.refinement_maximum_surface_distance_fraction == 0.12
    assert config.pose_admission.refinement_maximum_held_out_residual_ratio == 0.75
    assert config.pose_admission.refinement_minimum_reprojection_samples == 32
    assert config.pose_admission.refinement_translation_preference_ratio_tolerance == 0.01
    assert config.pose_admission.refinement_maximum_extent_ratio == 1.35


def test_interior_ellipse_ranges_are_validated() -> None:
    with pytest.raises(ValidationError, match="concentric_minimum_depth"):
        InteriorEllipseConfig(
            minimum_depth_plane_excess_fraction=0.001,
            concentric_minimum_depth_plane_excess_fraction=0.002,
        )


def test_adaptive_view_range_is_validated() -> None:
    with pytest.raises(ValidationError, match="minimum_views"):
        AdaptiveViewSelectionConfig(minimum_views=40, maximum_views=24)


@pytest.mark.parametrize(
    "values",
    [
        {
            "handle_band_minimum_diameter_fraction": 0.2,
            "handle_band_maximum_diameter_fraction": 0.1,
        },
        {
            "handle_depth_minimum_diameter_fraction": 0.3,
            "handle_depth_maximum_diameter_fraction": 0.2,
        },
    ],
)
def test_variable_handle_measurement_ranges_are_validated(
    values: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        AxialShellLoopConfig(**values)
