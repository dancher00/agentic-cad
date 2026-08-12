from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from da3_cad.config import AppConfig, CadrilleConfig, VisualHullConfig, load_config


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


def test_cadrille_candidate_budget_is_bounded_and_5080_safe_by_default() -> None:
    config = CadrilleConfig()
    assert (
        AppConfig(geometry={"segmentation_backend": "explicit-mask"}).geometry.segmentation_backend
        == "explicit-mask"
    )
    assert config.candidate_count == 1
    assert config.image_candidate_count == 0
    assert config.max_decode_batch_size == 1
    assert config.selection_mode == "input-chamfer"

    with pytest.raises(ValidationError):
        CadrilleConfig(candidate_count=0)
    with pytest.raises(ValidationError):
        CadrilleConfig(candidate_count=11)
    with pytest.raises(ValidationError):
        CadrilleConfig(image_candidate_count=5)
    with pytest.raises(ValidationError):
        CadrilleConfig(max_decode_batch_size=0)
    with pytest.raises(ValidationError):
        CadrilleConfig(silhouette_trim_fraction=0.5)

    with pytest.raises(ValidationError, match="minimum_grid_resolution"):
        VisualHullConfig(grid_resolution=10, minimum_grid_resolution=12)


@pytest.mark.parametrize(
    ("path", "cad_backend", "candidate_count"),
    [
        ("configs/video_geometric.yaml", "geometric-fitter", 1),
        ("configs/video_cadrille.yaml", "cadrille-rl", 4),
    ],
)
def test_video_profiles_load(path: str, cad_backend: str, candidate_count: int) -> None:
    config = load_config(Path(path))
    assert config.geometry.segmentation_backend == "explicit-mask"
    assert (config.cad_backend, config.cadrille.candidate_count) == (cad_backend, candidate_count)
    if path.endswith("video_cadrille.yaml"):
        assert config.cadrille.image_candidate_count == 4
        assert config.cadrille.selection_mode == "input-chamfer-silhouette"
    else:
        assert config.cadrille.image_candidate_count == 0
        assert config.cadrille.selection_mode == "input-chamfer"


@pytest.mark.parametrize(
    ("path", "cad_backend", "image_candidate_count"),
    [
        ("configs/photo_geometric.yaml", "geometric-fitter", 0),
        ("configs/photo_cadrille.yaml", "cadrille-rl", 1),
    ],
)
def test_photo_profiles_load(
    path: str,
    cad_backend: str,
    image_candidate_count: int,
) -> None:
    config = load_config(Path(path))
    assert config.depth_backend == "da3-large"
    assert config.geometry.segmentation_backend == "border-color"
    assert config.cad_backend == cad_backend
    assert config.cadrille.image_candidate_count == image_candidate_count
    if cad_backend == "geometric-fitter":
        assert config.geometric_fitter.hole_min_angular_coverage == 0.35
    else:
        assert config.cadrille.selection_mode == "input-chamfer-silhouette"


@pytest.mark.parametrize(
    ("path", "segmentation_backend"),
    [
        ("configs/internet_photo.yaml", "internet-object"),
        ("configs/internet_photo_masked.yaml", "explicit-mask"),
    ],
)
def test_cadrille_free_internet_photo_profiles_load(
    path: str,
    segmentation_backend: str,
) -> None:
    config = load_config(Path(path))
    assert config.depth_backend == "da3-large"
    assert config.cad_backend == "visual-hull"
    assert config.geometry.segmentation_backend == segmentation_backend
    assert config.visual_hull.depth_carving_enabled is True
    assert config.cadrille.image_candidate_count == 0
