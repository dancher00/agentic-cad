from pathlib import Path

import numpy as np
import pytest

from da3_cad.integrations.brepgaussian_surface import (
    BrepGaussianSurface,
    load_brepgaussian_surface,
    rectify_labelled_axis_planes,
)


def _write_pcd(path: Path, *, points: int = 256, fields: str = "x y z label edge") -> None:
    rows = "\n".join(f"{index} 0 1 {index % 3} 0.25" for index in range(points))
    path.write_text(
        "\n".join(
            (
                "# .PCD v0.7",
                "VERSION 0.7",
                f"FIELDS {fields}",
                "SIZE 4 4 4 4 4",
                "TYPE F F F I F",
                "COUNT 1 1 1 1 1",
                f"WIDTH {points}",
                "HEIGHT 1",
                f"POINTS {points}",
                "DATA ascii",
                rows,
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_brepgaussian_surface_preserves_labels_and_scale(tmp_path: Path) -> None:
    path = tmp_path / "merged.pcd"
    _write_pcd(path)

    result = load_brepgaussian_surface(path, world_units_to_mm=26.0)

    assert result.cloud.points.shape == (256, 3)
    assert result.cloud.scale.world_units_to_mm == 26.0
    assert result.cloud.report.mask_source == "brepgaussian-stage2-merged-pcd"
    assert np.unique(result.labels).tolist() == [0, 1, 2]
    assert result.as_dict()["edge_score"]["median"] == pytest.approx(0.25)


def test_load_brepgaussian_surface_rejects_unknown_layout(tmp_path: Path) -> None:
    path = tmp_path / "merged.pcd"
    _write_pcd(path, fields="x y z edge label")

    with pytest.raises(ValueError, match="expected PCD fields"):
        load_brepgaussian_surface(path, world_units_to_mm=1.0)


def test_rectify_labelled_axis_planes_projects_reliable_patch(tmp_path: Path) -> None:
    path = tmp_path / "merged.pcd"
    _write_pcd(path)
    loaded = load_brepgaussian_surface(path, world_units_to_mm=1.0)
    points = loaded.cloud.points.copy()
    points[:, 0] = np.linspace(-1.0, 1.0, len(points), dtype=np.float32)
    points[:, 1] = np.tile(np.linspace(-0.5, 0.5, 3, dtype=np.float32), 86)[: len(points)]
    points[:, 2] = 0.2 * points[:, 0] + 0.01 * np.sin(np.arange(len(points)))
    surface = BrepGaussianSurface(
        cloud=type(loaded.cloud)(
            points=points,
            colors=loaded.cloud.colors,
            confidences=loaded.cloud.confidences,
            view_indices=loaded.cloud.view_indices,
            pixel_xy=loaded.cloud.pixel_xy,
            report=loaded.cloud.report,
            scale=loaded.cloud.scale,
        ),
        labels=np.zeros(len(points), dtype=np.int32),
        edge_scores=loaded.edge_scores,
    )

    rectified, report = rectify_labelled_axis_planes(surface)

    assert report.rectified_labels == (0,)
    assert np.ptp(rectified.cloud.points[:, 2]) == pytest.approx(0.0)
