from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import trimesh

from da3_cad.benchmark.candidates import (
    CandidateArtifact,
    CandidateSelection,
    CandidateSelectionRecord,
)
from da3_cad.benchmark.silhouette_selection import (
    candidate_mesh_in_world,
    render_opencv_silhouette,
    score_candidate_silhouettes,
    select_by_input_and_silhouette,
)
from da3_cad.models import DepthPrediction


def _canonical() -> SimpleNamespace:
    corners = np.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-0.5, 0.5) for z in (4.5, 5.5)],
        dtype=np.float32,
    )
    return SimpleNamespace(
        stages=(SimpleNamespace(name="orientation", points=corners),),
        orientation=None,
    )


def _prediction() -> DepthPrediction:
    intrinsics = np.asarray(
        [[[80.0, 0.0, 64.0], [0.0, 80.0, 64.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)[None, ...]
    return DepthPrediction(
        depth=np.ones((1, 128, 128), dtype=np.float32),
        confidence=np.ones((1, 128, 128), dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=(np.zeros((128, 128, 3), dtype=np.uint8),),
        backend="synthetic",
    )


def test_world_mapping_and_projection_preserve_opencv_camera_convention() -> None:
    canonical = _canonical()
    prediction = _prediction()
    mesh = candidate_mesh_in_world(
        trimesh.creation.box(extents=(2.0, 1.0, 1.0)),
        canonical,  # type: ignore[arg-type]
    )

    assert np.allclose(mesh.bounds, [[-1.0, -0.5, 4.5], [1.0, 0.5, 5.5]])
    rendered = render_opencv_silhouette(
        mesh,
        prediction.intrinsics[0],
        prediction.extrinsics[0],
        (128, 128),
    )
    ys, xs = np.nonzero(rendered)
    assert len(xs) > 0
    assert abs(float(xs.mean()) - 64.0) < 1.0
    assert abs(float(ys.mean()) - 64.0) < 1.0
    assert np.ptp(xs) > np.ptp(ys)


def test_multiview_silhouette_score_reranks_without_ground_truth(tmp_path) -> None:
    canonical = _canonical()
    prediction = _prediction()
    correct = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    wrong = trimesh.creation.box(extents=(1.0, 2.0, 1.0))
    target_mesh = candidate_mesh_in_world(correct, canonical)  # type: ignore[arg-type]
    target = render_opencv_silhouette(
        target_mesh,
        prediction.intrinsics[0],
        prediction.extrinsics[0],
        (128, 128),
    )[None, ...]
    candidates = (
        CandidateArtifact(index=0, mesh=correct),
        CandidateArtifact(index=1, mesh=wrong),
    )

    scores = score_candidate_silhouettes(
        candidates,
        canonical,  # type: ignore[arg-type]
        prediction,
        target,
        output_root=tmp_path / "renders",
    )

    assert scores[0].valid
    assert scores[0].trimmed_mean_iou == 1.0
    assert scores[1].valid
    assert scores[1].trimmed_mean_iou is not None
    assert scores[1].trimmed_mean_iou < 0.5
    assert (tmp_path / "renders/candidate_00/view_000.png").is_file()

    input_selection = CandidateSelection(
        selected_index=1,
        input_cloud_sha256="synthetic",
        input_point_count=8,
        selection_surface_points=8192,
        records=(
            CandidateSelectionRecord(0, True, 2.0, None, 1, None),
            CandidateSelectionRecord(1, True, 1.0, None, 2, None),
        ),
    )
    selection = select_by_input_and_silhouette(
        input_selection,
        scores,
        silhouette_weight=2.0,
    )

    assert selection.selected_index == 0
    assert selection.as_dict()["ground_truth_access"] is False
