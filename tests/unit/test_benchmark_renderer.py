from __future__ import annotations

from pathlib import Path

from da3_cad.benchmark.cameras import (
    MASTER_ANGLES,
    master_schedule,
    minimum_pairwise_angle_degrees,
)
from da3_cad.benchmark.renderer import RenderConfig, materialize_view_subset, render_item


def test_camera_schedule_is_nested_and_well_separated() -> None:
    cameras = master_schedule(image_size=128)
    assert len(cameras) == 16
    assert len(set(MASTER_ANGLES)) == 16
    for count in (1, 2, 4, 8, 16):
        assert [camera.index for camera in cameras[:count]] == list(range(count))
    assert minimum_pairwise_angle_degrees(cameras[:2]) > 140.0
    assert minimum_pairwise_angle_degrees(cameras[:4]) > 75.0
    assert minimum_pairwise_angle_degrees(cameras) > 25.0
    for camera in cameras:
        assert camera.intrinsics.shape == (3, 3)
        assert camera.world_to_camera.shape == (4, 4)


def test_renderer_is_repeatable_and_keeps_gt_masks_outside_views(tmp_path: Path) -> None:
    mesh = Path("sample_data/plate/gt.stl")
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = RenderConfig(image_size=128, profile="normal")
    first_report = render_item(
        mesh,
        first,
        item_id="plate",
        dataset="synthetic",
        item_seed=123,
        config=config,
    )
    second_report = render_item(
        mesh,
        second,
        item_id="plate",
        dataset="synthetic",
        item_seed=123,
        config=config,
    )
    assert first_report["camera_schedule"] == second_report["camera_schedule"]
    assert len(tuple((first / "views").glob("*.png"))) == 16
    assert len(tuple((first / "gt_masks_not_for_reconstruction").glob("*.png"))) == 16
    assert not any("mask" in path.name for path in (first / "views").iterdir())

    subset = tmp_path / "n4"
    paths = materialize_view_subset(first / "views", subset, 4)
    assert [path.name for path in paths] == [f"view_{index:03d}.png" for index in range(4)]
