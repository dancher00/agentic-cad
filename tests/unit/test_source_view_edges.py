from __future__ import annotations

import cv2
import numpy as np
import trimesh

from da3_cad.evaluation.source_view_verifier import (
    SourceViewScore,
    _appearance_edges,
    _depth_view_admission,
    _rendered_geometry_edges,
    _smooth_face_groups,
    appearance_topology_regressions,
)


def test_appearance_edges_ignore_segmentation_boundary_but_keep_inner_rim() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.bool_)
    mask[8:88, 8:88] = True
    image[mask] = 220
    cv2.circle(image, (48, 48), 14, (20, 20, 20), 2)

    edges = _appearance_edges(image, mask)

    assert int(edges.sum()) > 40
    assert not edges[8:12].any()
    assert not edges[:, 8:12].any()


def test_rendered_geometry_edges_include_internal_occupancy_boundary() -> None:
    mask = np.zeros((96, 96), dtype=np.bool_)
    mask[8:88, 8:88] = True
    rendered = mask.copy()
    cavity = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(cavity, (48, 48), 14, 1, -1)
    rendered[cavity.astype(bool)] = False
    depth = np.full(mask.shape, np.inf, dtype=np.float32)
    depth[rendered] = 2.0

    edges = _rendered_geometry_edges(depth, rendered, mask)

    assert int(edges.sum()) > 60
    assert edges[48, 33]
    assert edges[48, 63]


def test_rendered_geometry_edges_include_crease_but_not_coplanar_triangles() -> None:
    mask = np.ones((48, 48), dtype=np.bool_)
    rendered = mask.copy()
    depth = np.full(mask.shape, 2.0, dtype=np.float32)
    surfaces = np.zeros(mask.shape, dtype=np.int64)
    surfaces[:, 24:] = 1

    crease = _rendered_geometry_edges(depth, rendered, mask, surfaces)
    coplanar = _rendered_geometry_edges(
        depth,
        rendered,
        mask,
        np.zeros(mask.shape, dtype=np.int64),
    )

    assert crease[8:-8, 23:25].all()
    assert not coplanar.any()


def test_smooth_face_groups_hide_box_face_triangulation() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))

    groups = _smooth_face_groups(mesh)

    assert len(np.unique(groups)) == 6


def _score(*, precision: float | None, recall: float | None) -> SourceViewScore:
    return SourceViewScore(
        score=0.9,
        silhouette_iou=0.9,
        depth_inlier_fraction=0.95,
        depth_observed_coverage=0.95,
        median_relative_depth_error=0.01,
        views=(),
        appearance_edge_precision=precision,
        appearance_edge_recall=recall,
        appearance_edge_pixels=256,
        rendered_geometry_edge_pixels=128,
    )


def test_cosmetic_rewrite_cannot_erase_visible_topology() -> None:
    baseline = _score(precision=0.56, recall=0.077)

    regressions = appearance_topology_regressions(
        baseline,
        _score(precision=0.21, recall=0.027),
    )

    assert regressions == (
        "appearance_edge_precision 0.5600->0.2100",
        "appearance_edge_recall 0.0770->0.0270",
    )
    assert not appearance_topology_regressions(
        baseline,
        _score(precision=0.57, recall=0.076),
    )


def test_depth_view_admission_rejects_unmeasured_patchmatch_view() -> None:
    mask = np.ones((20, 20), dtype=np.bool_)
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[:5, :5] = 2.0

    admitted, report = _depth_view_admission(
        depth,
        mask,
        minimum_pixels=128,
        minimum_mask_fraction=0.10,
    )

    assert not admitted
    assert report["measured_depth_pixels"] == 25
    assert report["measured_mask_fraction"] == 0.0625
    assert len(report["reasons"]) == 2


def test_depth_view_admission_accepts_cross_view_confirmed_depth() -> None:
    mask = np.ones((20, 20), dtype=np.bool_)
    depth = np.full((20, 20), 2.0, dtype=np.float32)

    admitted, report = _depth_view_admission(
        depth,
        mask,
        minimum_pixels=128,
        minimum_mask_fraction=0.10,
    )

    assert admitted
    assert report["measured_depth_pixels"] == 400
    assert report["reasons"] == []
