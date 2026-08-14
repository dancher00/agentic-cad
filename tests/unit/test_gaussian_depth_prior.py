from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.integrations.gaussian_depth_prior import (
    DepthPriorBundle,
    confidence_aware_depth_prior_loss,
    greedy_camera_subset,
    load_nerf_camera_dataset,
    spherical_camera_coverage,
)
from da3_cad.models import DepthPrediction


def _prediction(count: int = 2) -> DepthPrediction:
    return DepthPrediction(
        depth=np.full((count, 4, 5), 2.0, dtype=np.float32),
        confidence=np.linspace(0.1, 1.0, count * 20, dtype=np.float32).reshape(count, 4, 5),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0),
        processed_images=tuple(np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(count)),
        backend="synthetic-da3",
    )


def test_depth_prior_bundle_round_trip_without_pickle(tmp_path: Path) -> None:
    bundle = DepthPriorBundle.from_prediction(
        _prediction(),
        ("front", "back"),
        metadata={"selected_indices": [3, 7]},
    )
    path = tmp_path / "prior.npz"
    bundle.save(path)
    restored = DepthPriorBundle.load(path)

    assert restored.backend == "synthetic-da3"
    assert restored.image_names == ("front", "back")
    assert restored.metadata == {"selected_indices": [3, 7]}
    np.testing.assert_array_equal(restored.depth, bundle.depth)
    np.testing.assert_array_equal(restored.confidence, bundle.confidence)


def test_depth_prior_rejects_missing_or_duplicate_view_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        DepthPriorBundle.from_prediction(_prediction(), ("same", "same"))


def test_nerf_camera_loader_matches_blender_to_colmap_conversion(tmp_path: Path) -> None:
    image_dir = tmp_path / "train_img"
    image_dir.mkdir()
    Image.new("RGB", (20, 10), "white").save(image_dir / "view.png")
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]
    transforms = tmp_path / "transforms_train.json"
    transforms.write_text(
        json.dumps(
            {
                "camera_angle_x": 0.8,
                "frames": [
                    {
                        "file_path": "./train_img/view",
                        "transform_matrix": camera_to_world.tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_nerf_camera_dataset(transforms)

    assert dataset.image_names == ("view",)
    np.testing.assert_allclose(dataset.extrinsics[0, :3, 3], [-1.0, 2.0, 3.0])
    assert dataset.intrinsics[0, 0, 0] > 0.0
    assert dataset.intrinsics[0, 0, 2] == pytest.approx(9.5)
    assert dataset.intrinsics[0, 1, 2] == pytest.approx(4.5)


def test_greedy_subset_prefers_opposite_camera_directions() -> None:
    centers = np.asarray(
        [[3.0, 0.0, 0.0], [3.1, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float32,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], len(centers), axis=0)
    extrinsics[:, :3, 3] = -centers

    selected = greedy_camera_subset(extrinsics, 3)

    assert set(selected) == {0, 2, 3}
    assert spherical_camera_coverage(extrinsics[np.asarray(selected)]) > spherical_camera_coverage(
        extrinsics[np.asarray([0, 1])]
    )


def test_confidence_aware_loss_is_scale_invariant_and_rejects_outlier() -> None:
    torch = pytest.importorskip("torch")
    prior = torch.tensor(
        [[[1.0, 1.1, 1.2], [1.0, 1.1, 1.2], [1.0, 1.1, 1.2]]],
        device="cpu",
    )
    rendered = (prior * 2.5).clone().requires_grad_(True)
    rendered.data[0, 0, 0] = 20.0
    confidence = torch.ones_like(prior)
    confidence[0, 0, 0] = 0.0

    terms = confidence_aware_depth_prior_loss(rendered, prior, confidence)
    terms.total.backward()

    assert float(terms.total.item()) < 1e-5
    assert terms.consensus_fraction < 1.0
    assert rendered.grad is not None


def test_confidence_aware_loss_penalizes_nonuniform_shape_error() -> None:
    torch = pytest.importorskip("torch")
    prior = torch.ones((1, 4, 4), dtype=torch.float32)
    rendered = prior.clone()
    rendered[:, :, 2:] *= 1.5
    rendered.requires_grad_(True)

    terms = confidence_aware_depth_prior_loss(rendered, prior, torch.ones_like(prior))

    assert float(terms.depth.item()) > 0.0
    assert float(terms.gradient.item()) > 0.0
