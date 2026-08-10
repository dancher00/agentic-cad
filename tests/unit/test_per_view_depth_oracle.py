from __future__ import annotations

import numpy as np
import pytest

from da3_cad.benchmark.per_view_depth_oracle import (
    DepthAffineParameters,
    ViewRaySamples,
    apply_depth_affines,
    build_view_ray_samples,
    coefficient_dispersion,
    fit_per_view_depth_oracle,
)
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.models import DepthPrediction


def _view(
    view_index: int,
    *,
    origin: tuple[float, float, float],
    raw_depths: np.ndarray,
    rays: np.ndarray,
) -> ViewRaySamples:
    return ViewRaySamples(
        view_index=view_index,
        origin_world=np.asarray(origin, dtype=np.float64),
        rays_world=np.asarray(rays, dtype=np.float64),
        raw_depths=np.asarray(raw_depths, dtype=np.float64),
        raw_median=float(np.median(raw_depths)),
        eligible_depth_bounds=(float(raw_depths.min()), float(raw_depths.max())),
        center_depth_bounds=(1.0, 3.0),
        source_indices=np.arange(len(raw_depths), dtype=np.int64) + view_index * 100,
        unprojection_max_abs_error=0.0,
    )


def test_per_view_oracle_recovers_distinct_depth_affines_deterministically() -> None:
    grid = np.linspace(-0.25, 0.25, 5)
    xy = np.asarray([(x, y) for y in grid for x in grid], dtype=np.float64)
    rays_first = np.column_stack((xy, np.ones(len(xy))))
    rays_second = np.column_stack((-xy[:, 0], xy[:, 1], np.ones(len(xy))))
    raw_first = 1.8 + 0.3 * xy[:, 0] - 0.2 * xy[:, 1]
    raw_second = 2.2 - 0.25 * xy[:, 0] + 0.15 * xy[:, 1]
    views = (
        _view(0, origin=(-0.4, 0.0, 0.0), raw_depths=raw_first, rays=rays_first),
        _view(1, origin=(0.4, 0.0, 0.0), raw_depths=raw_second, rays=rays_second),
    )
    expected = ((1.35, 2.15), (0.72, 1.75))
    gt = np.concatenate(
        [
            view.corrected_points(scale, center)
            for view, (scale, center) in zip(views, expected, strict=True)
        ]
    )

    first = fit_per_view_depth_oracle(views, gt, scale_bounds=(0.5, 2.0))
    second = fit_per_view_depth_oracle(views, gt, scale_bounds=(0.5, 2.0))

    assert first.chamfer_after_x1000 < 0.01 * first.chamfer_before_x1000
    assert first.chamfer_after_x1000 == pytest.approx(second.chamfer_after_x1000)
    assert [value.scale for value in first.parameters] == pytest.approx(
        [value.scale for value in second.parameters]
    )
    assert [value.center_depth for value in first.parameters] == pytest.approx(
        [value.center_depth for value in second.parameters]
    )
    assert [value.scale for value in first.parameters] == pytest.approx(
        [value[0] for value in expected], abs=0.06
    )
    assert [value.center_depth for value in first.parameters] == pytest.approx(
        [value[1] for value in expected], abs=0.04
    )
    assert first.as_dict()["precision_used_by_optimizer"] is False
    assert first.as_dict()["allowed_in_benchmark_inference"] is False


def test_apply_depth_affines_preserves_non_depth_values_and_uses_ray_formula() -> None:
    depth = np.asarray([[[1.0, 2.0], [np.nan, -1.0]]], dtype=np.float32)
    masks = np.asarray([[[True, True], [False, False]]], dtype=np.bool_)
    parameter = DepthAffineParameters(
        view_index=0,
        scale=2.0,
        center_depth=3.0,
        raw_median=1.5,
        scale_bounds=(0.5, 2.0),
        center_depth_bounds=(1.0, 3.0),
    )

    corrected = apply_depth_affines(depth, (parameter,), masks)

    assert corrected[0, 0].tolist() == pytest.approx([2.0, 4.0])
    assert np.isnan(corrected[0, 1, 0])
    assert corrected[0, 1, 1] == -1.0
    assert parameter.shift == pytest.approx(0.0)
    assert parameter.median_depth_correction == pytest.approx(1.5)


@pytest.mark.parametrize("extrinsic_rows", [3, 4])
def test_build_view_ray_samples_reproduces_frozen_unprojection(
    extrinsic_rows: int,
) -> None:
    depth = np.linspace(1.8, 2.2, 16, dtype=np.float32).reshape(1, 4, 4)
    intrinsics = np.asarray(
        [[[3.0, 0.0, 1.5], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    extrinsic = np.eye(4, dtype=np.float32)[None, ...]
    if extrinsic_rows == 3:
        extrinsic = extrinsic[:, :3, :]
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsic,
        processed_images=(image,),
        backend="synthetic",
    )
    masks = np.ones_like(depth, dtype=np.bool_)
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="synthetic",
        confidence_percentile=None,
    )
    gt_surface = np.asarray(
        [[-0.5, -0.5, 1.5], [0.5, -0.5, 1.5], [0.5, 0.5, 2.5], [-0.5, 0.5, 2.5]],
        dtype=np.float64,
    )

    samples = build_view_ray_samples(
        cloud,
        depth,
        intrinsics,
        extrinsic,
        masks,
        gt_surface,
        points_per_view=4,
    )

    assert len(samples) == 1
    assert samples[0].unprojection_max_abs_error < 1e-6
    assert np.allclose(
        samples[0].corrected_points(1.0, samples[0].raw_median),
        cloud.points[samples[0].source_indices],
        atol=1e-6,
    )


def test_depth_affine_rejects_a_nonpositive_mask_eligible_result() -> None:
    depth = np.asarray([[[1.0, 3.0]]], dtype=np.float32)
    masks = np.ones_like(depth, dtype=np.bool_)
    parameter = DepthAffineParameters(
        view_index=0,
        scale=2.0,
        center_depth=0.5,
        raw_median=2.0,
        scale_bounds=(0.1, 10.0),
        center_depth_bounds=(0.1, 3.0),
    )

    with pytest.raises(ValueError, match="invalidated"):
        apply_depth_affines(depth, (parameter,), masks)


def test_coefficient_dispersion_reports_scale_and_shift_spread() -> None:
    parameters = (
        DepthAffineParameters(0, 0.5, 2.0, 2.0, (0.5, 2.0), (1.0, 3.0)),
        DepthAffineParameters(1, 2.0, 2.5, 2.0, (0.5, 2.0), (1.0, 3.0)),
    )

    report = coefficient_dispersion(parameters, gt_largest_extent=1.0)

    assert report["scale"]["max_over_min"] == pytest.approx(4.0)  # type: ignore[index]
    assert report["shift_b"]["range"] == pytest.approx(2.5)  # type: ignore[index]
    assert report["median_depth_correction"]["range"] == pytest.approx(0.5)  # type: ignore[index]
    assert report["boundary_hits"] == {"scale": 2, "center_depth": 0}


def test_per_view_oracle_rejects_invalid_scale_bounds() -> None:
    raw = np.asarray([1.0, 1.1, 1.2], dtype=np.float64)
    rays = np.column_stack((np.zeros((3, 2)), np.ones(3)))
    view = _view(0, origin=(0.0, 0.0, 0.0), raw_depths=raw, rays=rays)

    with pytest.raises(ValueError, match="bounds"):
        fit_per_view_depth_oracle((view,), view.corrected_points(1.0, 1.1), scale_bounds=(1, 1))
