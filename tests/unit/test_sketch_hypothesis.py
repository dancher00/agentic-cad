from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from da3_cad.backends.sketch_extrusion import UnsupportedProfileError
from da3_cad.config import AppConfig
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.geometry.multiview_depth_alignment import DepthHypothesisResult
from da3_cad.geometry.sketch_hypothesis import (
    UnsupportedDepthHypothesesError,
    rerank_sketch_depth_hypotheses,
)
from da3_cad.models import DepthPrediction


def _prediction(name: str) -> DepthPrediction:
    depth = np.ones((1, 4, 4), dtype=np.float32)
    return DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=np.eye(3, dtype=np.float32)[None, ...],
        extrinsics=np.eye(4, dtype=np.float32)[None, ...],
        processed_images=(np.zeros((4, 4, 3), dtype=np.uint8),),
        backend=name,
    )


def _cloud(cost: float) -> FusedPointCloud:
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0,),
        mask_source="synthetic",
        require_confidence=True,
        views=(ViewFusionStats(0, 1, 1, 1, 1, 1),),
    )
    return FusedPointCloud(
        points=np.asarray([[cost, 0.0, 0.0]], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        confidences=np.ones(1, dtype=np.float32),
        view_indices=np.zeros(1, dtype=np.int32),
        pixel_xy=np.zeros((1, 2), dtype=np.int32),
        report=report,
    )


def test_grammar_rerank_can_restore_identity(monkeypatch) -> None:
    identity = _prediction("identity")
    aligned = _prediction("aligned")
    hypotheses = DepthHypothesisResult(
        prediction=aligned,
        identity_prediction=identity,
        aligned_prediction=aligned,
        selected="aligned",
        report={"gt_blind": True},
    )

    def fake_fuse(prediction, masks, **kwargs):  # noqa: ANN001, ANN003
        del masks, kwargs
        return _cloud(0.03 if prediction.backend == "identity" else 0.05)

    def fake_canonicalize(  # noqa: ANN001
        self,
        cloud,
        *,
        seed,
        known_dimension=None,
        observed_cloud=None,
    ):
        del self, seed, known_dimension
        assert observed_cloud is not None
        return SimpleNamespace(cost=float(cloud.points[0, 0]))

    def fake_generate(  # noqa: ANN001
        self,
        canonical,
        *,
        seed,
        known_dimension=None,
        prediction=None,
        masks=None,
    ):
        del seed, known_dimension, prediction, masks
        axis = SimpleNamespace(
            axis=2,
            normalized_surface_p90=canonical.cost,
            surface_median=canonical.cost / 2.0,
            profile_occupancy_iou=0.95,
            profile_area_fraction=0.5,
        )
        self.last_report = SimpleNamespace(
            selected_axis=2,
            axis_candidates=(axis,),
            outer_loop=SimpleNamespace(kind="polyline", points=((0.0, 0.0),) * 4),
            apertures=(),
            input_points=256,
            profile_evidence_points=512,
        )
        return SimpleNamespace()

    monkeypatch.setattr(
        "da3_cad.geometry.sketch_hypothesis._fuse",
        fake_fuse,
    )
    monkeypatch.setattr(
        "da3_cad.geometry.sketch_hypothesis.PointCloudCanonicalizer.run",
        fake_canonicalize,
    )
    monkeypatch.setattr(
        "da3_cad.geometry.sketch_hypothesis.SketchExtrusionCadBackend.generate",
        fake_generate,
    )
    config = AppConfig.model_validate(
        {
            "profile": "research",
            "depth_backend": "da3-large-1.1",
            "cad_backend": "sketch-extrusion",
            "geometry": {
                "depth_alignment_criterion": "fixed-local-plane",
                "depth_alignment_selection": "auto",
            },
        }
    )

    result = rerank_sketch_depth_hypotheses(
        hypotheses,
        np.ones((1, 4, 4), dtype=np.bool_),
        mask_source="synthetic",
        scale=ScaleChannel(),
        config=config,
    )

    assert result.selected == "identity"
    assert result.prediction is identity
    assert result.report["gt_blind"] is True
    assert result.report["schema_version"] == "da3-cad-sketch-depth-selection-v2"
    assert [record["cost"] for record in result.report["records"]] == pytest.approx(  # type: ignore[index]
        [
            0.061,
            0.081,
        ]
    )


def test_grammar_rerank_retains_diagnostics_when_every_candidate_is_unsupported(
    monkeypatch,
) -> None:
    identity = _prediction("identity")
    aligned = _prediction("aligned")
    hypotheses = DepthHypothesisResult(
        prediction=aligned,
        identity_prediction=identity,
        aligned_prediction=aligned,
        selected="aligned",
        report={"gt_blind": True},
    )

    def fake_fuse(prediction, masks, **kwargs):  # noqa: ANN001, ANN003
        del masks, kwargs
        return _cloud(0.03 if prediction.backend == "identity" else 0.05)

    def reject(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise UnsupportedProfileError("constant-section residual 0.0820 > 0.0800")

    monkeypatch.setattr("da3_cad.geometry.sketch_hypothesis._fuse", fake_fuse)
    monkeypatch.setattr(
        "da3_cad.geometry.sketch_hypothesis.PointCloudCanonicalizer.run",
        lambda self, cloud, *, seed, observed_cloud=None: SimpleNamespace(
            cloud=cloud,
            seed=seed,
            observed_cloud=observed_cloud,
        ),
    )
    monkeypatch.setattr(
        "da3_cad.geometry.sketch_hypothesis.SketchExtrusionCadBackend.generate",
        reject,
    )
    config = AppConfig.model_validate(
        {
            "profile": "research",
            "depth_backend": "da3-large-1.1",
            "cad_backend": "sketch-extrusion",
            "geometry": {
                "depth_alignment_criterion": "fixed-local-plane",
                "depth_alignment_selection": "auto",
            },
        }
    )

    with pytest.raises(UnsupportedDepthHypothesesError) as captured:
        rerank_sketch_depth_hypotheses(
            hypotheses,
            np.ones((1, 4, 4), dtype=np.bool_),
            mask_source="synthetic",
            scale=ScaleChannel(),
            config=config,
        )

    error = captured.value
    assert error.prediction is identity
    assert error.cloud.points[0, 0] == pytest.approx(0.03)
    assert error.report["status"] == "unsupported"
    assert error.report["selected_hypothesis"] is None
    records = error.report["records"]
    assert isinstance(records, list)
    assert [record["hypothesis"] for record in records] == ["identity", "aligned"]
    assert all(record["valid"] is False for record in records)
