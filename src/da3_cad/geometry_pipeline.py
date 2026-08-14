"""Real DA3-to-world-point-cloud pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from da3_cad.backends.da3 import Da3Backend, da3_license_notice
from da3_cad.config import AppConfig
from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.geometry.diagnostics import write_geometry_diagnostics
from da3_cad.geometry.fusion import FusedPointCloud, ScaleChannel, fuse_prediction
from da3_cad.geometry.loop_feature_admission import admit_loop_feature_geometry
from da3_cad.geometry.multiview_depth_alignment import select_depth_hypothesis
from da3_cad.geometry.pose_admission import (
    admit_consistent_views,
    refine_disconnected_view_poses,
)
from da3_cad.geometry.sketch_hypothesis import (
    UnsupportedDepthHypothesesError,
    rerank_sketch_depth_hypotheses,
)
from da3_cad.geometry.unprojection import (
    as_homogeneous_extrinsic,
    unprojection_roundtrip_errors,
)
from da3_cad.geometry.view_selection import select_adaptive_views
from da3_cad.models import BoolArray, DepthPrediction, ObservationSet
from da3_cad.observations import doctor_report, load_observations
from da3_cad.segmentation.border_foreground import segment_border_foreground
from da3_cad.segmentation.depth_foreground import SegmentationResult, segment_depth_foreground
from da3_cad.segmentation.explicit_mask import segment_explicit_masks
from da3_cad.segmentation.internet_object import segment_internet_object


@dataclass(frozen=True, slots=True)
class GeometryRunResult:
    output_dir: Path
    cloud: FusedPointCloud
    prediction: DepthPrediction
    masks: BoolArray
    report: dict[str, object]
    observed_cloud: FusedPointCloud | None = None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _subset_prediction(
    prediction: DepthPrediction,
    indices: tuple[int, ...],
) -> DepthPrediction:
    selected = np.asarray(indices, dtype=np.int64)
    return DepthPrediction(
        depth=prediction.depth[selected].copy(),
        confidence=(
            prediction.confidence[selected].copy() if prediction.confidence is not None else None
        ),
        intrinsics=prediction.intrinsics[selected].copy(),
        extrinsics=prediction.extrinsics[selected].copy(),
        processed_images=tuple(prediction.processed_images[index].copy() for index in indices),
        backend=prediction.backend,
        warnings=(
            *prediction.warnings,
            f"whole-view pose admission retained {len(indices)}/{len(prediction.depth)} views",
        ),
    )


def _pose_report(prediction: DepthPrediction) -> dict[str, object]:
    homogeneous = np.stack(
        [as_homogeneous_extrinsic(extrinsic) for extrinsic in prediction.extrinsics]
    )
    rotations = homogeneous[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality_errors = [
        float(np.max(np.abs(rotation @ rotation.T - identity))) for rotation in rotations
    ]
    determinants = [float(np.linalg.det(rotation)) for rotation in rotations]
    intrinsic_determinants = [
        float(np.linalg.det(intrinsic)) for intrinsic in prediction.intrinsics
    ]
    return {
        "extrinsic_shape": list(prediction.extrinsics.shape),
        "intrinsic_shape": list(prediction.intrinsics.shape),
        "extrinsic_convention": "world_to_camera",
        "rotation_determinants": determinants,
        "max_rotation_orthogonality_error": max(orthogonality_errors),
        "intrinsic_determinants": intrinsic_determinants,
    }


def _inference_pass_report(
    runtime_report: dict[str, object],
    prediction: DepthPrediction,
) -> dict[str, object]:
    """Normalize optional backend telemetry around the required prediction contract."""

    output_shapes = runtime_report.get("output_shapes")
    if not isinstance(output_shapes, dict):
        output_shapes = {
            "depth": list(prediction.depth.shape),
            "confidence": (
                list(prediction.confidence.shape) if prediction.confidence is not None else None
            ),
            "intrinsics": list(prediction.intrinsics.shape),
            "extrinsics": list(prediction.extrinsics.shape),
            "processed_images": [list(image.shape) for image in prediction.processed_images],
        }
    input_views = runtime_report.get("input_views")
    if not isinstance(input_views, int):
        input_views = int(prediction.depth.shape[0])
    return {"input_views": input_views, "output_shapes": output_shapes}


def _segment_prediction(
    prediction: DepthPrediction,
    config: AppConfig,
    mask_paths: tuple[Path, ...] | None,
) -> SegmentationResult:
    if config.geometry.segmentation_backend in {"explicit-mask", "gt-mask-oracle"}:
        if mask_paths is None:
            raise ValueError("explicit segmentation requires matched mask paths")
        return segment_explicit_masks(
            prediction,
            mask_paths,
            oracle=config.geometry.segmentation_backend == "gt-mask-oracle",
        )
    if config.geometry.segmentation_backend == "border-color":
        return segment_border_foreground(prediction)
    if config.geometry.segmentation_backend == "internet-object":
        return segment_internet_object(
            prediction,
            minimum_fraction=config.geometry.segmentation_minimum_fraction,
            maximum_seed_fraction=config.geometry.segmentation_maximum_seed_fraction,
            confidence_percentile=config.geometry.segmentation_confidence_percentile,
            depth_percentile=config.geometry.segmentation_depth_percentile,
        )
    return segment_depth_foreground(
        prediction,
        confidence_percentile=config.geometry.segmentation_confidence_percentile,
        depth_percentile=config.geometry.segmentation_depth_percentile,
    )


def run_geometry(
    input_dir: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    accepted_noncommercial: bool,
    segmentation_mask_dir: Path | None = None,
    camera_bundle_path: Path | None = None,
    cad_grammar_rerank: bool = True,
) -> GeometryRunResult:
    """Run real DA3 inference, segmentation, unprojection and gated fusion."""

    supported_depth = {"da3-base", "da3-large-1.1", "da3-large"}
    if config.depth_backend not in supported_depth:
        raise ValueError(f"geometry command requires depth_backend in {sorted(supported_depth)}")
    expected_checkpoint = config.depth_backend.removeprefix("da3-")
    if config.da3.checkpoint != expected_checkpoint:
        raise ValueError(
            "depth_backend and da3.checkpoint disagree: "
            f"{config.depth_backend} versus {config.da3.checkpoint}"
        )
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    observations = load_observations(input_dir)
    camera_bundle = (
        load_camera_bundle(camera_bundle_path, observations)
        if camera_bundle_path is not None
        else None
    )
    camera_conditioning: dict[str, object] = (
        {
            "status": "external",
            "path": str(camera_bundle_path.resolve()),
            "sha256": hashlib.sha256(camera_bundle_path.read_bytes()).hexdigest(),
            **camera_bundle.as_dict(),
        }
        if camera_bundle is not None and camera_bundle_path is not None
        else {"status": "unposed-da3"}
    )
    output_dir.mkdir(parents=True)
    backend = Da3Backend(
        checkpoint=config.da3.checkpoint,
        source_dir=config.da3.source_dir,
        cache_dir=config.da3.cache_dir,
        process_resolution=config.da3.process_resolution,
        process_resolution_method=config.da3.process_resolution_method,
        local_files_only=config.da3.local_files_only,
        accepted_noncommercial=accepted_noncommercial,
        use_ray_pose=config.da3.use_ray_pose,
        ref_view_strategy=config.da3.ref_view_strategy,
    )
    if camera_bundle is None:
        prediction = backend.predict(observations, device=config.device, seed=config.seed)
    else:
        prediction = backend.predict(
            observations,
            device=config.device,
            seed=config.seed,
            extrinsics=camera_bundle.extrinsics,
            intrinsics=camera_bundle.intrinsics,
            align_to_input_ext_scale=True,
        )
    source_masks: list[dict[str, str]] = []
    mask_paths: tuple[Path, ...] | None = None
    explicit_backends = {"explicit-mask", "gt-mask-oracle"}
    if config.geometry.segmentation_backend in explicit_backends:
        if segmentation_mask_dir is None:
            raise ValueError(
                f"geometry.segmentation_backend={config.geometry.segmentation_backend} "
                "requires segmentation_mask_dir"
            )
        available: dict[str, Path] = {}
        for path in sorted(segmentation_mask_dir.glob("*.png"), key=lambda item: item.name):
            if path.stem in available:
                raise ValueError(f"duplicate explicit-mask stem: {path.stem}")
            available[path.stem] = path
        expected_stems = tuple(Path(image.relative_path).stem for image in observations.images)
        missing_stems = [stem for stem in expected_stems if stem not in available]
        if missing_stems:
            raise ValueError(
                "explicit masks must use the input image stems with PNG extension; "
                f"missing={missing_stems}"
            )
        mask_paths = tuple(available[stem] for stem in expected_stems)
        source_masks = [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in mask_paths
        ]
    elif segmentation_mask_dir is not None:
        raise ValueError(
            "segmentation_mask_dir requires the explicit-mask or gt-mask-oracle backend"
        )
    segmentation = _segment_prediction(prediction, config, mask_paths)
    pool_pass = _inference_pass_report(backend.last_runtime_report or {}, prediction)
    segmentation_masks = segmentation.masks
    if config.view_selection.enabled:
        provisional_observed = fuse_prediction(
            prediction,
            segmentation_masks,
            mask_source=segmentation.backend,
            confidence_percentile=None,
            minimum_confidence=None,
            require_confidence=False,
            extrinsic_convention="world_to_camera",
            scale=camera_bundle.scale if camera_bundle is not None else None,
        )
        selection_points = provisional_observed.points
    else:
        selection_points = np.zeros((3, 3), dtype=np.float32)
    view_selection = select_adaptive_views(
        prediction,
        segmentation_masks,
        selection_points,
        tuple(item.relative_path for item in observations.images),
        config.view_selection,
        config.coverage,
    )
    prediction = view_selection.prediction
    segmentation_masks = view_selection.masks
    pool_runtime_report = backend.last_runtime_report
    if pool_runtime_report is None:
        raise RuntimeError("DA3 pose-selection pass did not produce its runtime report")
    reran_selected_views = len(view_selection.selected_indices) < len(observations.images)
    if reran_selected_views:
        selected_images = tuple(
            observations.images[index] for index in view_selection.selected_indices
        )
        selected_digest = hashlib.sha256()
        for image in selected_images:
            selected_digest.update(image.relative_path.encode())
            selected_digest.update(b"\0")
            selected_digest.update(image.sha256.encode())
            selected_digest.update(b"\0")
        selected_observations = ObservationSet(
            root=observations.root,
            images=selected_images,
            digest=selected_digest.hexdigest(),
        )
        selected_array = np.asarray(view_selection.selected_indices, dtype=np.int64)
        if camera_bundle is None:
            prediction = backend.predict(
                selected_observations,
                device=config.device,
                seed=config.seed,
            )
        else:
            prediction = backend.predict(
                selected_observations,
                device=config.device,
                seed=config.seed,
                extrinsics=camera_bundle.extrinsics[selected_array],
                intrinsics=camera_bundle.intrinsics[selected_array],
                align_to_input_ext_scale=True,
            )
        selected_mask_paths = (
            tuple(mask_paths[index] for index in view_selection.selected_indices)
            if mask_paths is not None
            else None
        )
        segmentation = _segment_prediction(prediction, config, selected_mask_paths)
        segmentation_masks = segmentation.masks
    reconstruction_runtime_report = backend.last_runtime_report
    if reconstruction_runtime_report is None:
        raise RuntimeError("DA3 reconstruction pass did not produce its runtime report")
    reconstruction_pass = _inference_pass_report(reconstruction_runtime_report, prediction)
    view_selection.report["inference_passes"] = {
        "pose_selection": pool_pass,
        "reconstruction": {
            "reran_selected_views": reran_selected_views,
            **reconstruction_pass,
        },
        "reason": (
            "DA3 predictions are joint-context dependent; pose selection uses the full pool, "
            "then geometry is inferred again only on the selected views"
            if reran_selected_views
            else "the selector retained the complete pool, so one DA3 pass is sufficient"
        ),
    }
    (output_dir / "artefacts").mkdir(exist_ok=True)
    _write_json(output_dir / "artefacts" / "view_selection.json", view_selection.report)

    selected_input_indices = tuple(int(index) for index in view_selection.selected_indices)
    if len(selected_input_indices) != len(prediction.depth):
        raise RuntimeError("view selection index map no longer matches reconstruction prediction")
    pose_admission_report: dict[str, object]
    pose_admission_artifacts = False
    pre_admission_prediction_artifact = False
    if config.pose_admission.enabled and camera_bundle is None:
        pose_gate_cloud = fuse_prediction(
            prediction,
            segmentation_masks,
            mask_source=segmentation.backend,
            confidence_percentile=None,
            minimum_confidence=None,
            require_confidence=False,
            extrinsic_convention="world_to_camera",
        )
        initial_admission = admit_consistent_views(
            pose_gate_cloud,
            view_count=len(prediction.depth),
            minimum_views=config.pose_admission.minimum_views,
            samples_per_view=config.pose_admission.samples_per_view,
            center_distance_fraction=config.pose_admission.center_distance_fraction,
            surface_distance_fraction=config.pose_admission.surface_distance_fraction,
            minimum_component_fraction=config.pose_admission.minimum_component_fraction,
        )
        prediction_before_pose_refinement = prediction
        if config.pose_admission.refinement_enabled and initial_admission.rejected_view_indices:
            refinement = refine_disconnected_view_poses(
                prediction,
                segmentation_masks,
                initial_admission,
                minimum_views=config.pose_admission.minimum_views,
                samples_per_view=config.pose_admission.samples_per_view,
                center_distance_fraction=config.pose_admission.center_distance_fraction,
                surface_distance_fraction=config.pose_admission.surface_distance_fraction,
                minimum_component_fraction=config.pose_admission.minimum_component_fraction,
                maximum_translation_fraction=(
                    config.pose_admission.refinement_maximum_translation_fraction
                ),
                maximum_rotation_degrees=(
                    config.pose_admission.refinement_maximum_rotation_degrees
                ),
                maximum_surface_distance_fraction=(
                    config.pose_admission.refinement_maximum_surface_distance_fraction
                ),
                maximum_residual_ratio=(config.pose_admission.refinement_maximum_residual_ratio),
                maximum_held_out_residual_ratio=(
                    config.pose_admission.refinement_maximum_held_out_residual_ratio
                ),
                minimum_support_views=(config.pose_admission.refinement_minimum_support_views),
                held_out_fraction=config.pose_admission.refinement_held_out_fraction,
                optimization_iterations=(config.pose_admission.refinement_optimization_iterations),
                trim_fraction=config.pose_admission.refinement_trim_fraction,
                minimum_reprojection_samples=(
                    config.pose_admission.refinement_minimum_reprojection_samples
                ),
                minimum_reprojection_mask_overlap=(
                    config.pose_admission.refinement_minimum_reprojection_mask_overlap
                ),
                maximum_reprojection_residual_ratio=(
                    config.pose_admission.refinement_maximum_reprojection_residual_ratio
                ),
                translation_preference_ratio_tolerance=(
                    config.pose_admission.refinement_translation_preference_ratio_tolerance
                ),
                maximum_extent_ratio=(config.pose_admission.refinement_maximum_extent_ratio),
            )
            prediction = refinement.prediction
            admission = refinement.admission
        else:
            admission = initial_admission
        admitted_local = admission.admitted_view_indices
        rejected_local = admission.rejected_view_indices
        initial_rejected_local = initial_admission.rejected_view_indices
        refinement_report = admission.report.get("pose_refinement")
        refined_local = (
            tuple(int(value) for value in refinement_report.get("refined_views", []))
            if isinstance(refinement_report, dict)
            else ()
        )
        admitted_input = tuple(selected_input_indices[index] for index in admitted_local)
        rejected_input = tuple(selected_input_indices[index] for index in rejected_local)
        initial_rejected_input = tuple(
            selected_input_indices[index] for index in initial_rejected_local
        )
        refined_input = tuple(selected_input_indices[index] for index in refined_local)
        pose_admission_report = {
            **admission.report,
            "local_to_input_view": list(selected_input_indices),
            "admitted_input_views": list(admitted_input),
            "rejected_input_views": list(rejected_input),
            "initial_rejected_input_views": list(initial_rejected_input),
            "refined_input_views": list(refined_input),
            "admitted_image_names": [
                observations.images[index].relative_path for index in admitted_input
            ],
            "rejected_image_names": [
                observations.images[index].relative_path for index in rejected_input
            ],
            "initial_rejected_image_names": [
                observations.images[index].relative_path for index in initial_rejected_input
            ],
            "refined_image_names": [
                observations.images[index].relative_path for index in refined_input
            ],
        }
        np.savez_compressed(
            output_dir / "artefacts" / "pose_admission_samples.npz",
            points=initial_admission.sampled_points,
            points_before_refinement=initial_admission.sampled_points,
            points_after_refinement=admission.sampled_points,
            local_view_indices=initial_admission.sampled_view_indices,
            local_view_indices_after_refinement=admission.sampled_view_indices,
            input_view_indices=np.asarray(
                [selected_input_indices[index] for index in initial_admission.sampled_view_indices],
                dtype=np.int32,
            ),
            admitted_local_view_indices=np.asarray(admitted_local, dtype=np.int32),
            initial_rejected_local_view_indices=np.asarray(
                initial_rejected_local,
                dtype=np.int32,
            ),
            refined_local_view_indices=np.asarray(refined_local, dtype=np.int32),
        )
        pose_admission_artifacts = True
        if initial_rejected_local:
            confidence_before_admission = prediction_before_pose_refinement.confidence
            if confidence_before_admission is None:
                raise RuntimeError(
                    "validated DA3 prediction unexpectedly lost confidence before pose admission"
                )
            np.savez_compressed(
                output_dir / "artefacts" / "camera_prediction_before_pose_admission.npz",
                depth=prediction_before_pose_refinement.depth,
                confidence=confidence_before_admission,
                intrinsics=prediction_before_pose_refinement.intrinsics,
                extrinsics=prediction_before_pose_refinement.extrinsics,
                masks=segmentation_masks,
                processed_images=np.stack(prediction_before_pose_refinement.processed_images),
                local_to_input_view=np.asarray(selected_input_indices, dtype=np.int32),
            )
            pre_admission_prediction_artifact = True
        if rejected_local:
            prediction = _subset_prediction(prediction, admitted_local)
            selected = np.asarray(admitted_local, dtype=np.int64)
            segmentation_masks = segmentation_masks[selected].copy()
    elif camera_bundle is not None:
        pose_admission_report = {
            "schema_version": "da3-cad-pose-admission-v1",
            "status": "bypassed-external-cameras",
            "reason": "externally supplied calibrated poses retain authority",
            "input_views": len(prediction.depth),
            "admitted_views": list(range(len(prediction.depth))),
            "rejected_views": [],
            "local_to_input_view": list(selected_input_indices),
        }
    else:
        pose_admission_report = {
            "schema_version": "da3-cad-pose-admission-v1",
            "status": "disabled",
            "input_views": len(prediction.depth),
            "admitted_views": list(range(len(prediction.depth))),
            "rejected_views": [],
            "local_to_input_view": list(selected_input_indices),
        }
    _write_json(output_dir / "artefacts" / "pose_admission.json", pose_admission_report)
    view_selection.report["pose_admission"] = pose_admission_report
    _write_json(output_dir / "artefacts" / "view_selection.json", view_selection.report)

    alignment_report: dict[str, object] = {"status": "disabled"}
    fusion_prediction = prediction
    cloud: FusedPointCloud | None = None
    terminal_grammar_error: UnsupportedDepthHypothesesError | None = None
    if config.geometry.depth_alignment_criterion is not None:
        hypothesis = select_depth_hypothesis(
            prediction,
            segmentation_masks,
            criterion=config.geometry.depth_alignment_criterion,
            selection=config.geometry.depth_alignment_selection,
            seed=config.seed,
            maximum_loss_ratio=config.geometry.depth_alignment_maximum_loss_ratio,
            maximum_scale_ratio=config.geometry.depth_alignment_maximum_scale_ratio,
            maximum_center_ratio_deviation=(
                config.geometry.depth_alignment_maximum_center_ratio_deviation
            ),
        )
        fusion_prediction = hypothesis.prediction
        grammar_report: dict[str, object] | None = None
        if (
            cad_grammar_rerank
            and config.cad_backend in {"construction-grammar", "sketch-extrusion"}
            and config.geometry.depth_alignment_selection == "auto"
        ):
            try:
                grammar_selection = rerank_sketch_depth_hypotheses(
                    hypothesis,
                    segmentation_masks,
                    mask_source=segmentation.backend,
                    scale=(camera_bundle.scale if camera_bundle is not None else ScaleChannel()),
                    config=config,
                )
                fusion_prediction = grammar_selection.prediction
                cloud = grammar_selection.cloud
                grammar_report = grammar_selection.report
            except UnsupportedDepthHypothesesError as error:
                # Preserve inspectable geometry for an honest CAD abstention.
                # The error is re-raised only after all diagnostics are durable.
                fusion_prediction = error.prediction
                cloud = error.cloud
                grammar_report = error.report
                terminal_grammar_error = error
        alignment_report = {
            "schema_version": "da3-cad-depth-selection-pipeline-v1",
            "status": "unsupported" if terminal_grammar_error is not None else "selected",
            "gt_blind": True,
            "selected_hypothesis": (
                grammar_report["selected_hypothesis"]
                if grammar_report is not None
                else hypothesis.selected
            ),
            "observation_gate": hypothesis.report,
            "cad_grammar_rerank": grammar_report,
            "failure": (
                {
                    "stage": "cad-grammar-rerank",
                    "type": type(terminal_grammar_error).__name__,
                    "reason": str(terminal_grammar_error),
                }
                if terminal_grammar_error is not None
                else None
            ),
        }
    loop_feature_admission = admit_loop_feature_geometry(
        fusion_prediction,
        segmentation_masks,
        config=config.loop_feature_admission,
        loop_config=config.axial_shell_loop,
        external_cameras=camera_bundle is not None,
    )
    geometry_masks = loop_feature_admission.geometry_masks
    _write_json(
        output_dir / "artefacts" / "loop_feature_admission.json",
        loop_feature_admission.report,
    )
    # The earlier cloud, when present, belongs to depth-hypothesis scoring.  The
    # final trusted geometry is rebuilt from feature-admitted masks so repeated
    # but mutually inconsistent loop layers are never presented as one surface.
    cloud = fuse_prediction(
        fusion_prediction,
        geometry_masks,
        mask_source=f"{segmentation.backend}+loop-feature-admission",
        confidence_percentile=config.geometry.fusion_confidence_percentile,
        minimum_confidence=config.geometry.minimum_confidence,
        require_confidence=True,
        extrinsic_convention="world_to_camera",
        scale=camera_bundle.scale if camera_bundle is not None else None,
    )
    observed_cloud = fuse_prediction(
        fusion_prediction,
        segmentation_masks,
        mask_source=segmentation.backend,
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
        scale=camera_bundle.scale if camera_bundle is not None else None,
    )
    write_geometry_diagnostics(
        output_dir / "artefacts",
        fusion_prediction,
        segmentation_masks,
        cloud,
        observed_cloud=observed_cloud,
        pose_admission=pose_admission_report,
        geometry_masks=geometry_masks,
        loop_feature_admission=loop_feature_admission.report,
    )

    confidence = fusion_prediction.confidence
    if confidence is None:
        raise RuntimeError("validated DA3 prediction unexpectedly lost confidence")
    if config.geometry.depth_alignment_criterion is not None:
        _write_json(
            output_dir / "artefacts" / "depth_alignment_report.json",
            alignment_report,
        )
        np.savez_compressed(
            output_dir / "artefacts" / "camera_prediction.npz",
            depth=fusion_prediction.depth,
            confidence=confidence,
            intrinsics=fusion_prediction.intrinsics,
            extrinsics=fusion_prediction.extrinsics,
            masks=segmentation_masks,
            processed_images=np.stack(fusion_prediction.processed_images),
            raw_depth_before_alignment=prediction.depth,
        )
    else:
        np.savez_compressed(
            output_dir / "artefacts" / "camera_prediction.npz",
            depth=fusion_prediction.depth,
            confidence=confidence,
            intrinsics=fusion_prediction.intrinsics,
            extrinsics=fusion_prediction.extrinsics,
            masks=segmentation_masks,
            processed_images=np.stack(fusion_prediction.processed_images),
        )
    if backend.last_runtime_report is None or backend.last_lifecycle is None:
        raise RuntimeError("DA3 backend did not produce its required runtime report")

    cloud_bounds = np.stack((cloud.points.min(axis=0), cloud.points.max(axis=0)))
    roundtrips = [
        unprojection_roundtrip_errors(
            fusion_prediction.depth[index],
            fusion_prediction.intrinsics[index],
            fusion_prediction.extrinsics[index],
            convention="world_to_camera",
        )
        for index in range(fusion_prediction.depth.shape[0])
    ]
    report: dict[str, object] = {
        "schema_version": "1.0",
        "command": "geometry",
        "input": observations.as_dict(),
        "doctor": doctor_report(observations),
        "seed": config.seed,
        "profile": config.profile,
        "license": {
            "notice": da3_license_notice(backend.spec),
            "explicit_noncommercial_acceptance": (
                accepted_noncommercial if backend.spec.noncommercial else None
            ),
            "weights_redistributed": False,
        },
        "camera_conditioning": camera_conditioning,
        "da3": backend.last_runtime_report,
        "verified_source_contracts": {
            "depth": "z-depth multiplying K^-1 [u,v,1] in the pinned exporter",
            "confidence": "higher-is-better; pinned exporter retains values >= percentile",
            "extrinsics": "world-to-camera; adapter accepts and validates N×3×4 or N×4×4",
            "pixel_coordinates": "integer u=0..W-1, v=0..H-1 as in pinned exporter",
        },
        "runtime_pose_validation": _pose_report(fusion_prediction),
        "runtime_unprojection_roundtrip": roundtrips,
        "segmentation": {
            "backend": segmentation.backend,
            "oracle": config.geometry.segmentation_backend == "gt-mask-oracle",
            "source_masks": source_masks,
            "selected_pixels": [int(mask.sum()) for mask in segmentation_masks],
            "warnings": list(segmentation.warnings),
        },
        "view_selection": view_selection.report,
        "pose_admission": pose_admission_report,
        "loop_feature_admission": loop_feature_admission.report,
        "depth_alignment": alignment_report,
        "fusion": cloud.report.as_dict(),
        "geometry_channels": {
            "observed": {
                "point_count": int(len(observed_cloud.points)),
                "artifact": "artefacts/observed_cloud.npz",
                "used_for_cad_fitting": False,
            },
            "trusted": {
                "point_count": int(len(cloud.points)),
                "artifact": "artefacts/trusted_geometry.npz",
                "mask_artifact_pattern": "artefacts/geometry_mask_*.png",
                "used_for_cad_fitting": True,
            },
            "silhouettes": {
                "selected_pixels": [int(mask.sum()) for mask in segmentation_masks],
                "artifact_pattern": "artefacts/mask_*.png",
                "used_for_boundaries_and_topology": True,
            },
            "compatibility_alias": {
                "artifact": "artefacts/fused_cloud.npz",
                "channel": "trusted",
            },
        },
        "cloud": {
            "point_count": int(len(cloud.points)),
            "bounds": cloud_bounds.tolist(),
            "finite": bool(np.isfinite(cloud.points).all()),
            "scale_status": cloud.scale.status,
            "units": cloud.scale.units,
            "world_units_to_mm": cloud.scale.world_units_to_mm,
            "scale_source": cloud.scale.source,
            "scale_evidence": cloud.scale.evidence,
        },
        "artifacts": [
            "artefacts/camera_prediction.npz",
            "artefacts/view_selection.json",
            "artefacts/pose_admission.json",
            "artefacts/loop_feature_admission.json",
            "artefacts/depth_*.png",
            "artefacts/confidence_*.png",
            "artefacts/mask_*.png",
            "artefacts/mask_overlay_*.png",
            "artefacts/geometry_mask_*.png",
            "artefacts/observed_cloud.npz",
            "artefacts/observed_cloud.ply",
            "artefacts/trusted_geometry.npz",
            "artefacts/trusted_geometry.ply",
            "artefacts/fused_cloud.npz",
            "artefacts/fused_cloud.ply",
            "artefacts/fusion_report.json",
            *(["artefacts/pose_admission_samples.npz"] if pose_admission_artifacts else []),
            *(
                ["artefacts/camera_prediction_before_pose_admission.npz"]
                if pre_admission_prediction_artifact
                else []
            ),
            *(
                ["artefacts/depth_alignment_report.json"]
                if config.geometry.depth_alignment_criterion is not None
                else []
            ),
        ],
    }
    _write_json(output_dir / "geometry_report.json", report)
    if terminal_grammar_error is not None:
        raise terminal_grammar_error
    return GeometryRunResult(
        output_dir=output_dir,
        cloud=cloud,
        prediction=fusion_prediction,
        masks=segmentation_masks.copy(),
        report=report,
        observed_cloud=observed_cloud,
    )
