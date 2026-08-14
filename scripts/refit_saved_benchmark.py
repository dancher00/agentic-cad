#!/usr/bin/env python3
"""Refit CAD from saved DA3 evidence without rerunning the depth network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import load_config
from da3_cad.evaluation.evaluator import Evaluator
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.models import DepthPrediction

ROOT = Path(__file__).resolve().parents[1]


def _load_case(
    run: Path,
    *,
    cloud_name: str = "fused_cloud.npz",
) -> tuple[FusedPointCloud, DepthPrediction, np.ndarray]:
    geometry = run / "artefacts" / "geometry" / "artefacts"
    with np.load(geometry / cloud_name) as payload:
        points = np.asarray(payload["points"], dtype=np.float32)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        view_indices = np.asarray(payload["view_indices"], dtype=np.int32)
        pixel_xy = np.asarray(payload["pixel_xy"], dtype=np.int32)
    with np.load(geometry / "camera_prediction.npz") as payload:
        masks = np.asarray(payload["masks"], dtype=np.bool_)
        prediction = DepthPrediction(
            depth=np.asarray(payload["depth"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            processed_images=tuple(
                np.asarray(image, dtype=np.uint8) for image in payload["processed_images"]
            ),
            backend="saved-da3-evidence",
        )
    view_stats = tuple(
        ViewFusionStats(
            view_index=index,
            pixels=int(masks.shape[1] * masks.shape[2]),
            finite_positive_depth=int(
                np.count_nonzero(
                    np.isfinite(prediction.depth[index]) & (prediction.depth[index] > 0)
                )
            ),
            mask_selected=int(np.count_nonzero(masks[index])),
            confidence_selected=int(np.count_nonzero(view_indices == index)),
            fused=int(np.count_nonzero(view_indices == index)),
        )
        for index in range(len(masks))
    )
    report = FusionReport(
        confidence_percentile=35.0,
        confidence_scope="per-view",
        confidence_thresholds=tuple(None for _ in range(len(masks))),
        mask_source="saved-explicit-target-masks",
        require_confidence=True,
        views=view_stats,
    )
    cloud = FusedPointCloud(
        points=points,
        colors=colors,
        confidences=confidence,
        view_indices=view_indices,
        pixel_xy=pixel_xy,
        report=report,
        scale=ScaleChannel(
            status="known",
            units="millimetre",
            world_units_to_mm=1.0,
            source="saved-external-camera-bundle",
        ),
    )
    return cloud, prediction, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "public-benchmark-v2")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "sample_data" / "public_benchmark_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "public_benchmark_v2.yaml",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--cloud",
        choices=("fused", "observed"),
        default="fused",
        help="saved point evidence to refit (default: current filtered fused cloud)",
    )
    parser.add_argument("--disable-revolve-shell", action="store_true")
    parser.add_argument("--disable-extrude-axis-refinement", action="store_true")
    parser.add_argument("--disable-extrude-length-refinement", action="store_true")
    parser.add_argument("--disable-extrude-pose-refinement", action="store_true")
    parser.add_argument("--revolve-profile-smoothing-sigma", type=float)
    parser.add_argument("--silhouette-reprojection-weight", type=float)
    parser.add_argument(
        "--no-observed-channel",
        action="store_true",
        help="ablate the secondary per-view observed evidence used by revolve",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    cases = tuple(args.cases or [str(item["id"]) for item in manifest["cases"]])
    if args.output.exists():
        raise FileExistsError(f"refit output already exists: {args.output}")
    args.output.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for case_id in cases:
        destination = args.output / case_id
        destination.mkdir()
        report_payload: dict[str, object] | None = None
        try:
            cloud, prediction, masks = _load_case(
                args.source / case_id,
                cloud_name=f"{args.cloud}_cloud.npz",
            )
            observed_cloud = None
            if not args.no_observed_channel:
                observed_cloud, _, _ = _load_case(
                    args.source / case_id,
                    cloud_name="observed_cloud.npz",
                )
            canonical = PointCloudCanonicalizer(config.canonicalizer).run(
                cloud,
                seed=config.seed,
                observed_cloud=observed_cloud,
            )
            revolve_config = (
                config.revolve.model_copy(update={"shell_enabled": False})
                if args.disable_revolve_shell
                else config.revolve
            )
            sketch_config = config.sketch_extrusion.model_copy(
                update={
                    **(
                        {"axis_refinement_enabled": False}
                        if args.disable_extrude_axis_refinement
                        else {}
                    ),
                    **(
                        {"silhouette_length_refinement_enabled": False}
                        if args.disable_extrude_length_refinement
                        else {}
                    ),
                    **(
                        {"silhouette_pose_refinement_enabled": False}
                        if args.disable_extrude_pose_refinement
                        else {}
                    ),
                    **(
                        {"silhouette_reprojection_weight": args.silhouette_reprojection_weight}
                        if args.silhouette_reprojection_weight is not None
                        else {}
                    ),
                }
            )
            if args.revolve_profile_smoothing_sigma is not None:
                revolve_config = revolve_config.model_copy(
                    update={"profile_smoothing_sigma_bins": (args.revolve_profile_smoothing_sigma)}
                )
            backend = ConstructionGrammarCadBackend(
                config.construction_grammar,
                sketch_config,
                revolve_config,
                config.axial_shell_loop,
            )
            program = backend.generate(
                canonical,
                seed=config.seed,
                prediction=prediction,
                masks=masks,
            )
            report = backend.last_report
            if report is None:
                raise RuntimeError("grammar backend did not produce a report")
            report_payload = report.as_dict()
            (destination / "grammar_report.json").write_text(
                json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (destination / "model.py").write_text(program.source, encoding="utf-8")
            validation = validate_and_export(program.source, destination, config.sandbox)
            if not validation.valid or validation.step_path is None:
                raise RuntimeError(validation.error or "CAD validation failed")
            metrics = Evaluator().evaluate(
                f"saved-refit-{case_id}",
                validation.step_path,
                args.fixtures / case_id / "gt.step",
            )
            generator = next(
                item for item in report.candidates if item.family == report.selected_family
            ).generator_report
            record = {
                "id": case_id,
                "status": "step",
                "family": report.selected_family,
                "iou_percent": metrics.iou.percent if metrics.iou is not None else None,
                "chamfer_squared_x1000": (
                    metrics.chamfer.scaled_bidirectional if metrics.chamfer is not None else None
                ),
                "apertures": len(generator.get("apertures", [])),
                "report": report_payload,
            }
        except Exception as error:
            record = {
                "id": case_id,
                "status": "abstain",
                "error": f"{type(error).__name__}: {error}",
                "report": report_payload,
            }
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "report"}))
    (args.output / "refit.json").write_text(
        json.dumps({"cases": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
