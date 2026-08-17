#!/usr/bin/env python3
"""Run the official CADENA-RL policy without a persistent vLLM server.

The official rollout uses one model proposal per operation, executes the
prefix, and retains it only when target-mesh IoU improves.  This adapter keeps
that algorithm but loads the 2B Qwen2-VL checkpoint directly so a single RTX
5080 can run one object without vLLM's serving overhead.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from da3_cad.cad_program import simplify_revolve_profiles
from da3_cad.cad_validation import validate_and_export_cadquery, validate_cadquery
from da3_cad.evaluation.source_view_verifier import (
    SourceViewScore,
    SourceViewVerifier,
    appearance_topology_regressions,
    decide_source_view_progress,
    decide_source_view_score,
)
from da3_cad.geometry.orientation import orient_canonical_frame
from da3_cad.geometry.symmetry import detect_symmetry_plane
from da3_cad.integrations.construction_graph import (
    ConstructionGraphLedger,
    bind_operation_to_patch,
    operation_name,
    operation_polarity,
    polarity_consistent,
    repair_polarity_step,
    residual_patch,
)
from da3_cad.integrations.measured_features import (
    AxialRevolvedAdd,
    AxialRevolvedCut,
    PlanarProfileAdd,
    PlanarProfileCut,
    fit_axial_revolved_add,
    fit_axial_revolved_cut,
    fit_planar_profile_add_candidates,
    fit_planar_profile_cut_candidates,
)
from da3_cad.integrations.proposal_proxy import (
    ProposalProxy,
    fit_revolve_proposal_proxy,
    fit_sketch_extrusion_proposal_proxy,
)
from da3_cad.integrations.render_canonicalization import (
    CADENA_PROXY_RENDER_DOWNSAMPLE_FACTOR,
    CADENA_PROXY_RENDER_QUANTIZATION_STEP,
    CADENA_PROXY_SPUR_FILTER_SIZE,
    CADENA_RENDER_DOWNSAMPLE_FACTOR,
    CADENA_RENDER_QUANTIZATION_STEP,
    canonicalize_rgb_render,
)
from da3_cad.models import FloatArray

ALLOWED_OPERATIONS = {
    "extrude",
    "gear",
    "hole",
    "loft",
    "orto_cut",
    "revolve",
    "shell",
    "spring",
    "sweep",
    "sweep_adv",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--measurements",
        type=Path,
        help="Optional raw cross-view-confirmed point cloud used for measured CAD fitting.",
    )
    parser.add_argument("--maximum-measurement-points", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--verification-workspace", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--verification-max-image-dimension", type=int, default=240)
    parser.add_argument("--expansions", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-score-improvement", type=float, default=0.001)
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.87)
    parser.add_argument("--minimum-depth-inlier-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-appearance-edge-precision", type=float, default=0.45)
    parser.add_argument("--minimum-appearance-edge-recall", type=float, default=0.12)
    parser.add_argument("--minimum-appearance-edge-pixels", type=int, default=128)
    parser.add_argument("--minimum-topology-recall-improvement", type=float, default=0.05)
    parser.add_argument("--maximum-topology-score-regression", type=float, default=0.05)
    parser.add_argument("--simplification-score-tolerance", type=float, default=0.001)
    return parser


def _load_plotter_without_forced_osmesa(checkout: Path) -> Any:
    """Load the official renderer after removing its unavailable hardcoded backend."""

    path = checkout.resolve() / "visualization.py"
    source = path.read_text(encoding="utf-8")
    hardcoded = "os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'"
    if source.count(hardcoded) != 1:
        raise RuntimeError("unsupported CADENA visualization.py backend configuration")
    source = source.replace(
        hardcoded,
        "os.environ.pop('VTK_DEFAULT_OPENGL_WINDOW', None)",
    )
    module = types.ModuleType("visualization")
    module.__file__ = str(path)
    sys.modules["visualization"] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.Plotter


def _imports(checkout: Path) -> tuple[Any, Any, str, Any, Any, Any]:
    sys.path.insert(0, str(checkout.resolve()))
    Plotter = _load_plotter_without_forced_osmesa(checkout)
    if not hasattr(np, "trapezoid"):
        np.trapezoid = np.trapz  # type: ignore[attr-defined]
    from utils.meshcad import (  # type: ignore[import-not-found]
        code_to_mesh,
        transform_mesh_0_1,
        transform_pred_mesh,
    )
    from utils.metrics import compute_gms, compute_iou  # type: ignore[import-not-found]
    from utils.pipeline import CODE_PREFIX, get_point  # type: ignore[import-not-found]

    return (
        Plotter,
        get_point,
        CODE_PREFIX
        + "\nfrom da3_cad.integrations.measured_features import ("
        + "\n    axial_revolved_add, axial_revolved_cut,"
        + "\n    planar_profile_add, planar_profile_cut,"
        + "\n)",
        code_to_mesh,
        transform_mesh_0_1,
        (
            transform_pred_mesh,
            compute_iou,
            compute_gms,
        ),
    )


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.Name):
        return node.id == "r"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _literal(node.operand)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _literal(key) and _literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


def _validated_step(raw: str) -> str:
    text = raw.strip().replace("```python", "").replace("```", "").strip()
    if text == "end" or text.startswith("end\n"):
        return "end"
    try:
        module = ast.parse(text, mode="exec")
    except SyntaxError as error:
        raise ValueError(f"CADENA emitted invalid Python: {error.msg}") from error
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        raise ValueError("CADENA step must be one assignment")
    assignment = module.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise ValueError("CADENA step must assign one name")
    if assignment.targets[0].id != "r" or not isinstance(assignment.value, ast.Call):
        raise ValueError("CADENA step must have the form r = operation(...)")
    call = assignment.value
    if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_OPERATIONS:
        raise ValueError("CADENA step called an operation outside the published DSL")
    if not all(_literal(argument) for argument in call.args):
        raise ValueError("CADENA positional arguments must be literals or the current solid r")
    if not all(keyword.arg is not None and _literal(keyword.value) for keyword in call.keywords):
        raise ValueError("CADENA keyword arguments must be literals")
    return ast.unparse(module).strip()


def _generate_steps(
    model: Any,
    processor: Any,
    image: Image.Image,
    point: str,
    *,
    max_new_tokens: int,
    expansions: int,
    temperature: float,
) -> tuple[str, ...]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": point},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generation_options: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": expansions > 1,
        "num_return_sequences": expansions,
        "use_cache": True,
    }
    if expansions > 1:
        generation_options.update({"temperature": temperature, "top_p": 0.95, "top_k": 0})
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_options)
    suffix = generated[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(
        suffix,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return tuple(dict.fromkeys(str(value).strip() for value in decoded))


def _score(
    target: trimesh.Trimesh,
    predicted: trimesh.Trimesh,
    transform_pred_mesh: Any,
    compute_iou: Any,
    compute_gms: Any,
    *,
    use_iou: bool,
) -> float:
    normalized = transform_pred_mesh(predicted)
    if use_iou:
        value, _, _ = compute_iou(target, normalized)
    else:
        _, value = compute_gms(target, normalized)
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _execute_result(source: str) -> object:
    namespace: dict[str, object] = {}
    exec(compile(source, "<cadena-validated-program>", "exec"), None, namespace)
    result = namespace.get("r")
    if result is None:
        raise RuntimeError("validated CADENA program produced no r solid")
    return result


def _execute_final(source: str, output_path: Path) -> dict[str, int | float | bool]:
    return validate_and_export_cadquery(_execute_result(source), output_path).as_dict()


def _fit_best_axial_revolved_cut(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> tuple[AxialRevolvedCut | None, tuple[AxialRevolvedCut, ...]]:
    candidates = tuple(
        feature
        for axis in (0, 1, 2)
        if (feature := fit_axial_revolved_cut(target_points, current_mesh, axis=axis)) is not None
    )
    if not candidates:
        return None, ()
    selected = max(
        candidates,
        key=lambda value: (
            value.angular_coverage_fraction,
            value.radial_growth_fraction,
            value.axial_span_fraction,
            -value.normalized_profile_residual,
            value.support_points,
        ),
    )
    return selected, candidates


def _fit_best_axial_revolved_add(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> tuple[AxialRevolvedAdd | None, tuple[AxialRevolvedAdd, ...]]:
    candidates = tuple(
        feature
        for axis in (0, 1, 2)
        if (feature := fit_axial_revolved_add(target_points, current_mesh, axis=axis)) is not None
    )
    if not candidates:
        return None, ()
    selected = max(
        candidates,
        key=lambda value: (
            value.angular_coverage_fraction,
            value.axial_span_fraction,
            -value.normalized_profile_residual,
            value.support_points,
        ),
    )
    return selected, candidates


def _fit_best_planar_profile_add(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> tuple[PlanarProfileAdd | None, tuple[PlanarProfileAdd, ...]]:
    candidates = fit_planar_profile_add_candidates(target_points, current_mesh)
    if not candidates:
        return None, ()
    selected = max(
        candidates,
        key=lambda value: (
            value.axial_coverage_fraction,
            value.profile_occupancy_iou,
            -value.constant_section_residual,
            value.support_points,
        ),
    )
    return selected, candidates


def _fit_best_planar_profile_cut(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> tuple[PlanarProfileCut | None, tuple[PlanarProfileCut, ...]]:
    candidates = fit_planar_profile_cut_candidates(target_points, current_mesh)
    if not candidates:
        return None, ()
    selected = max(
        candidates,
        key=lambda value: (
            value.axial_coverage_fraction,
            value.profile_occupancy_iou,
            -value.constant_section_residual,
            value.support_points,
        ),
    )
    return selected, candidates


def _load_measurement_points(
    path: Path,
    *,
    maximum_points: int,
    seed: int,
) -> tuple[FloatArray, dict[str, object]]:
    """Load a deterministic finite subset of measured point-cloud vertices."""

    if maximum_points < 256:
        raise ValueError("maximum measurement points must be at least 256")
    loaded = trimesh.load(path.resolve(), process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = [
            np.asarray(geometry.vertices, dtype=np.float64)
            for geometry in loaded.geometry.values()
            if hasattr(geometry, "vertices") and len(geometry.vertices)
        ]
        raw = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float64)
    elif hasattr(loaded, "vertices"):
        raw = np.asarray(loaded.vertices, dtype=np.float64)
    else:
        raw = np.empty((0, 3), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1:] != (3,):
        raise ValueError(f"measurement cloud has invalid vertex shape: {path}")
    finite = raw[np.isfinite(raw).all(axis=1)]
    if len(finite) < 256:
        raise ValueError(f"measurement cloud requires at least 256 finite points: {path}")
    input_points = int(len(finite))
    if input_points > maximum_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(input_points, size=maximum_points, replace=False))
        finite = finite[indices]
        sampling = "seeded-uniform-without-replacement"
    else:
        sampling = "all-finite-points"
    return finite, {
        "source": "cross-view-confirmed-point-cloud",
        "path": str(path.resolve()),
        "input_points": input_points,
        "used_points": int(len(finite)),
        "sampling": sampling,
        "seed": seed if sampling.startswith("seeded") else None,
    }


def _candidate_in_observation_frame(
    candidate: trimesh.Trimesh,
    *,
    target_extent: float,
    target_center: FloatArray,
    object_to_observation: FloatArray,
) -> trimesh.Trimesh:
    """Map CADENA's centred 200-unit solid back to calibrated observation space."""

    result = candidate.copy()
    result.apply_scale(target_extent / 200.0)  # type: ignore[no-untyped-call]
    result.apply_translation(np.asarray(target_center, dtype=np.float64))
    result.apply_transform(np.asarray(object_to_observation, dtype=np.float64))
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    if args.max_steps < 1 or args.max_new_tokens < 1:
        raise ValueError("step and token budgets must be positive")
    if args.maximum_measurement_points < 256:
        raise ValueError("maximum measurement points must be at least 256")
    if args.expansions < 1 or args.expansions > 16:
        raise ValueError("expansions must be in [1, 16]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if args.minimum_score_improvement < 0.0:
        raise ValueError("minimum score improvement must be non-negative")
    if args.simplification_score_tolerance < 0.0:
        raise ValueError("simplification score tolerance must be non-negative")
    for name in (
        "minimum_appearance_edge_precision",
        "minimum_appearance_edge_recall",
        "minimum_topology_recall_improvement",
        "maximum_topology_score_regression",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name.replace('_', ' ')} must be in [0, 1]")
    if args.minimum_appearance_edge_pixels < 0:
        raise ValueError("minimum appearance edge pixels must be non-negative")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    Plotter, get_point, code_prefix, code_to_mesh, transform_target, metric_tools = _imports(
        args.checkout
    )
    transform_pred_mesh, compute_iou, compute_gms = metric_tools

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    target_world = trimesh.load(args.mesh.resolve(), force="mesh", process=True)
    if not isinstance(target_world, trimesh.Trimesh) or target_world.is_empty:
        raise ValueError(f"target is not a non-empty triangle mesh: {args.mesh}")

    target_mesh_path = args.mesh.resolve()
    object_to_observation = np.eye(4, dtype=np.float64)
    measurement_points_world: FloatArray | None = None
    object_frame_report: dict[str, object] = {
        "enabled": False,
        "reason": "no separate cross-view measurement cloud was supplied",
        "object_to_observation": object_to_observation.tolist(),
    }
    if args.measurements is None:
        target_raw = target_world
    else:
        measurement_points_world, measurement_report = _load_measurement_points(
            args.measurements,
            maximum_points=args.maximum_measurement_points,
            seed=args.seed,
        )
        symmetry = detect_symmetry_plane(
            measurement_points_world,
            tolerance_fraction=0.04,
            seed=args.seed,
            maximum_evaluation_points=4096,
        )
        orientation = orient_canonical_frame(
            measurement_points_world,
            symmetry,
            seed=args.seed,
            planar_extent_ratio_threshold=0.20,
            plane_distance_fraction=0.02,
            plane_ransac_iterations=256,
            eigenvalue_tie_tolerance=0.05,
        )
        orientation_center = np.asarray(orientation.center_world, dtype=np.float64)
        axes_world_columns = np.asarray(orientation.axes_world, dtype=np.float64).T
        target_raw = target_world.copy()
        target_raw.vertices = (
            np.asarray(target_raw.vertices, dtype=np.float64) - orientation_center[None, :]
        ) @ axes_world_columns
        target_mesh_path = output / "target_canonical.ply"
        target_raw.export(target_mesh_path)
        object_to_observation[:3, :3] = axes_world_columns
        object_to_observation[:3, 3] = orientation_center
        object_frame_report = {
            "enabled": True,
            "method": "existing-da3-cad-orientation-v1",
            "orientation": orientation.as_dict(),
            "symmetry": symmetry.as_dict(),
            "object_to_observation": object_to_observation.tolist(),
            "conditioning_mesh": target_mesh_path.name,
            "reference_geometry_access": False,
        }

    use_iou = bool(target_raw.is_watertight and target_raw.is_volume)
    target = transform_target(target_raw)
    target_center = np.asarray(target_raw.bounds, dtype=np.float64).mean(axis=0)
    target_extent = float(np.max(target_raw.extents))
    if (args.verification_workspace is None) != (args.cameras is None):
        raise ValueError("--verification-workspace and --cameras must be supplied together")
    view_verifier = (
        SourceViewVerifier.from_colmap_workspace(
            args.verification_workspace,
            args.cameras,
            maximum_image_dimension=args.verification_max_image_dimension,
        )
        if args.verification_workspace is not None and args.cameras is not None
        else None
    )

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.checkpoint.resolve(),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            args.checkpoint.resolve(),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    plotter = Plotter()
    target_image = canonicalize_rgb_render(plotter.get_img(str(target_mesh_path), None))
    target_image.save(output / "target.png")

    if args.measurements is None:
        first_target_point, point_cloud = get_point(
            str(target_mesh_path),
            None,
            seed=args.seed,
        )
        measurement_report = {
            "source": "triangle-surface-samples",
            "path": str(target_mesh_path),
            "input_points": None,
            "used_points": int(len(point_cloud)),
            "sampling": "CADENA-area-weighted-surface",
            "seed": args.seed,
            "canonicalization": "CADENA mesh-bounds centre and maximum extent",
        }
    else:
        if measurement_points_world is None:
            raise RuntimeError("measurement point cloud was not loaded")
        frame = object_to_observation[:3, :3]
        frame_center = object_to_observation[:3, 3]
        oriented = (measurement_points_world - frame_center[None, :]) @ frame
        point_cloud = (oriented - target_center[None, :]) * (200.0 / target_extent)
        measurement_report.update(
            {
                "canonicalization": (
                    "DA3-CAD oriented object frame, then target-mesh bbox centre and maximum extent"
                ),
                "canonical_extent": target_extent,
                "canonical_bbox_center": target_center.tolist(),
            }
        )
        first_target_point, point_cloud = get_point(point_cloud, None, seed=args.seed)
    proposal_point = first_target_point
    proposal_image: Image.Image | None = None
    measured_root_candidates: list[
        tuple[str, trimesh.Trimesh, float, SourceViewScore | None, str]
    ] = []
    measured_root_reports: list[dict[str, object]] = []

    def register_measured_root(
        proxy: ProposalProxy,
        *,
        artifact_stem: str,
    ) -> tuple[trimesh.Trimesh | None, Image.Image | None, str | None]:
        root_report: dict[str, object] = {
            "attempted": True,
            "available": False,
            "role": "independent-measured-root",
            **proxy.as_dict(),
        }
        root_report["role"] = "independent-measured-root"
        proxy_execution = code_to_mesh(
            proxy.source,
            var_name="r",
            _tol=0.001,
            _ang=0.1,
        )
        proxy_mesh = proxy_execution.get("mesh")
        if not isinstance(proxy_mesh, trimesh.Trimesh) or proxy_mesh.is_empty:
            proxy_status = proxy_execution.get("status")
            root_report["reason"] = f"proxy CAD execution failed: {proxy_status}"
            measured_root_reports.append(root_report)
            return None, None, None
        try:
            proxy_kernel = validate_cadquery(_execute_result(proxy.source)).as_dict()
        except Exception as error:  # noqa: BLE001 - isolate one measured hypothesis
            root_report["reason"] = f"proxy CAD kernel rejected fit: {error}"
            measured_root_reports.append(root_report)
            return None, None, None
        preview = canonicalize_rgb_render(
            plotter.get_img(proxy_mesh.copy(), None),
            downsample_factor=CADENA_PROXY_RENDER_DOWNSAMPLE_FACTOR,
            quantization_step=CADENA_PROXY_RENDER_QUANTIZATION_STEP,
            spur_filter_size=CADENA_PROXY_SPUR_FILTER_SIZE,
        )
        render_path = output / f"{artifact_stem}.png"
        preview_path = output / f"{artifact_stem}.stl"
        preview.save(render_path)
        proxy_mesh.export(preview_path)
        conditioning_point, _ = get_point(proxy_mesh, None, seed=args.seed)
        proxy_view_score = None
        if view_verifier is None:
            proxy_score = _score(
                target,
                proxy_mesh,
                transform_pred_mesh,
                compute_iou,
                compute_gms,
                use_iou=use_iou,
            )
            selection_metric = "IoU" if use_iou else "GMS"
        else:
            proxy_world = _candidate_in_observation_frame(
                proxy_mesh,
                target_extent=target_extent,
                target_center=target_center,
                object_to_observation=object_to_observation,
            )
            proxy_view_score = view_verifier.score(proxy_world)
            proxy_score = proxy_view_score.score
            selection_metric = "source-view-multiobjective-v2"
        measured_root_candidates.append(
            (proxy.source, proxy_mesh, proxy_score, proxy_view_score, proxy.family)
        )
        root_report.update(
            {
                "available": True,
                "conditioning_point": conditioning_point,
                "kernel_validation": proxy_kernel,
                "preview_path": preview_path.name,
                "render_path": render_path.name,
                "candidate_score": proxy_score,
                "selection_metric": selection_metric,
                "source_view_score": (
                    proxy_view_score.as_dict() if proxy_view_score is not None else None
                ),
            }
        )
        measured_root_reports.append(root_report)
        return proxy_mesh, preview, conditioning_point

    proposal_proxy = fit_revolve_proposal_proxy(point_cloud, seed=args.seed)
    proposal_proxy_report: dict[str, object]
    if proposal_proxy is None:
        proposal_proxy_report = {
            "attempted": True,
            "available": False,
            "role": "proposal-conditioning-and-independent-root",
            "reason": "target evidence did not support a revolve proxy",
        }
        measured_root_reports.append(
            {
                "attempted": True,
                "available": False,
                "role": "independent-measured-root",
                "family": "revolve",
                "reason": "target evidence did not support a revolve proxy",
            }
        )
    else:
        _, proposal_image, fitted_point = register_measured_root(
            proposal_proxy,
            artifact_stem="proposal_proxy",
        )
        proposal_proxy_report = dict(measured_root_reports[-1])
        proposal_proxy_report["role"] = "proposal-conditioning-and-independent-root"
        if fitted_point is not None:
            proposal_point = fitted_point

    sketch_proxy = fit_sketch_extrusion_proposal_proxy(point_cloud, seed=args.seed)
    if sketch_proxy is None:
        measured_root_reports.append(
            {
                "attempted": True,
                "available": False,
                "role": "independent-measured-root",
                "family": "sketch-extrusion",
                "reason": "target evidence did not support one constant-section extrusion",
            }
        )
    else:
        register_measured_root(
            sketch_proxy,
            artifact_stem="sketch_extrusion_proxy",
        )

    prefix = code_prefix
    best_source: str | None = None
    best_mesh: trimesh.Trimesh | None = None
    best_iou = 0.0
    current: trimesh.Trimesh | None = None
    trajectory: list[dict[str, object]] = []
    best_view_score: SourceViewScore | None = None
    accepted_prefixes: list[tuple[str, trimesh.Trimesh, float, SourceViewScore | None, int]] = []
    construction_graph = ConstructionGraphLedger()
    for step_index in range(args.max_steps):
        if step_index == 0:
            point = first_target_point
        else:
            point, _ = get_point(point_cloud, current, seed=args.seed + step_index)
        parsed_point = ast.literal_eval(point)
        if (
            not isinstance(parsed_point, tuple)
            or len(parsed_point) != 3
            or not all(isinstance(value, int | float) for value in parsed_point)
        ):
            raise RuntimeError(f"CADENA residual selector returned an invalid point: {point}")
        patch = residual_patch(
            patch_id=f"residual-{step_index:02d}",
            point=(
                float(parsed_point[0]),
                float(parsed_point[1]),
                float(parsed_point[2]),
            ),
            target_points=point_cloud,
            current_mesh=current,
            current_result=_execute_result(prefix) if current is not None else None,
        )
        construction_graph.add_patch(patch)
        if current is None:
            image = target_image
        else:
            image = canonicalize_rgb_render(plotter.get_img(str(target_mesh_path), current))
        image.save(output / f"step_{step_index:02d}_input.png")
        generation_requests = [(image, point, "measured-target")]
        if step_index == 0 and proposal_image is not None:
            proposal_image.save(output / "step_00_proxy_input.png")
            generation_requests.append((proposal_image, proposal_point, "revolve-proxy"))
        raw_steps: list[tuple[str, str, str]] = []
        seen_steps: set[str] = set()
        for request_image, request_point, request_origin in generation_requests:
            generated = _generate_steps(
                model,
                processor,
                request_image,
                request_point,
                max_new_tokens=args.max_new_tokens,
                expansions=args.expansions,
                temperature=args.temperature,
            )
            for raw in generated:
                if raw in seen_steps:
                    continue
                seen_steps.add(raw)
                raw_steps.append((raw, request_origin, request_point))
        row: dict[str, object] = {
            "step": step_index,
            "point": point,
            "proposal_queries": [
                {"origin": origin, "point": request_point}
                for _, request_point, origin in generation_requests
            ],
            "residual_patch": patch.as_dict(),
            "previous_score": best_iou,
        }
        expansion_rows: list[dict[str, object]] = []
        best_expansion: (
            tuple[
                str,
                str,
                trimesh.Trimesh,
                float,
                str,
                SourceViewScore | None,
                str,
                str,
            ]
            | None
        )
        best_expansion = None
        best_expansion_rank: tuple[int, float, float] | None = None
        for expansion_index, (raw, proposal_origin, request_point) in enumerate(
            raw_steps,
        ):
            expansion_row = {
                "index": expansion_index,
                "raw": raw,
                "origin": proposal_origin,
                "proposal_point": request_point,
            }
            try:
                step = _validated_step(raw)
            except ValueError as error:
                expansion_row.update({"executable": False, "reason": str(error)})
                expansion_rows.append(expansion_row)
                continue
            expansion_row["validated"] = step
            if step == "end":
                expansion_row.update({"executable": False, "reason": "model-ended"})
                expansion_rows.append(expansion_row)
                continue
            model_step = step
            model_operation = operation_name(model_step)
            emitted_polarity = operation_polarity(model_operation)
            correspondence: dict[str, object] | None = None
            if current is not None:
                bound_step, correspondence = bind_operation_to_patch(
                    model_step,
                    patch,
                    current,
                )
                expansion_row["operation_patch_correspondence"] = correspondence
                if bound_step is None:
                    node_id = construction_graph.add_node(
                        step_index=step_index,
                        expansion_index=expansion_index,
                        patch=patch,
                        model_step=model_step,
                        executed_step=None,
                        model_operation=model_operation,
                        executed_operation=None,
                        emitted_polarity=emitted_polarity,
                        polarity_consistent=False,
                        repair=None,
                        correspondence=correspondence,
                        origin=proposal_origin,
                    )
                    reason = (
                        "operation does not correspond to residual patch: "
                        f"{correspondence['reason']}"
                    )
                    construction_graph.update_node(node_id, reason=reason)
                    expansion_row.update(
                        {
                            "executable": False,
                            "reason": reason,
                            "construction_graph_node": node_id,
                            "model_operation": model_operation,
                            "required_polarity": patch.required_polarity,
                        }
                    )
                    expansion_rows.append(expansion_row)
                    continue
                step = bound_step
                if step != model_step:
                    expansion_row.update(
                        {
                            "model_validated": model_step,
                            "validated": step,
                        }
                    )
            consistent = polarity_consistent(patch.required_polarity, emitted_polarity)
            repair: dict[str, object] | None = None
            if not consistent:
                repaired = (
                    repair_polarity_step(step, patch, current) if current is not None else None
                )
                if repaired is None:
                    node_id = construction_graph.add_node(
                        step_index=step_index,
                        expansion_index=expansion_index,
                        patch=patch,
                        model_step=model_step,
                        executed_step=None,
                        model_operation=model_operation,
                        executed_operation=None,
                        emitted_polarity=emitted_polarity,
                        polarity_consistent=False,
                        repair=None,
                        correspondence=correspondence,
                        origin=proposal_origin,
                    )
                    reason = (
                        f"operation polarity {emitted_polarity} contradicts residual "
                        f"polarity {patch.required_polarity}"
                    )
                    construction_graph.update_node(node_id, reason=reason)
                    expansion_row.update(
                        {
                            "executable": False,
                            "reason": reason,
                            "construction_graph_node": node_id,
                            "model_operation": model_operation,
                            "required_polarity": patch.required_polarity,
                        }
                    )
                    expansion_rows.append(expansion_row)
                    continue
                step, repair = repaired
                expansion_row.update(
                    {
                        "model_validated": model_step,
                        "validated": step,
                        "polarity_repair": repair,
                    }
                )
            executed_operation = operation_name(step)
            node_id = construction_graph.add_node(
                step_index=step_index,
                expansion_index=expansion_index,
                patch=patch,
                model_step=model_step,
                executed_step=step,
                model_operation=model_operation,
                executed_operation=executed_operation,
                emitted_polarity=emitted_polarity,
                polarity_consistent=consistent,
                repair=repair,
                correspondence=correspondence,
                origin=proposal_origin,
            )
            expansion_row.update(
                {
                    "construction_graph_node": node_id,
                    "model_operation": model_operation,
                    "executed_operation": executed_operation,
                    "required_polarity": patch.required_polarity,
                    "emitted_polarity": emitted_polarity,
                    "polarity_consistent": consistent,
                }
            )
            candidate_source = prefix.rstrip() + "\n" + step
            execution = code_to_mesh(candidate_source, var_name="r", _tol=0.001, _ang=0.1)
            candidate = execution.get("mesh")
            if not isinstance(candidate, trimesh.Trimesh) or candidate.is_empty:
                expansion_row.update(
                    {
                        "executable": False,
                        "reason": f"CAD execution failed: {execution.get('status')}",
                    }
                )
                construction_graph.update_node(
                    node_id,
                    reason=f"CAD execution failed: {execution.get('status')}",
                )
                expansion_rows.append(expansion_row)
                continue
            try:
                kernel_validation = validate_cadquery(_execute_result(candidate_source)).as_dict()
            except Exception as error:  # noqa: BLE001 - isolate one model proposal
                expansion_row.update(
                    {
                        "executable": False,
                        "reason": f"CAD kernel rejected candidate: {error}",
                    }
                )
                construction_graph.update_node(
                    node_id,
                    reason=f"CAD kernel rejected candidate: {error}",
                )
                expansion_rows.append(expansion_row)
                continue
            expansion_row["kernel_validation"] = kernel_validation
            construction_graph.update_node(node_id, kernel_validation=kernel_validation)
            candidate_view_score = None
            if view_verifier is None:
                candidate_score = _score(
                    target,
                    candidate,
                    transform_pred_mesh,
                    compute_iou,
                    compute_gms,
                    use_iou=use_iou,
                )
                selection_metric = "IoU" if use_iou else "GMS"
            else:
                candidate_world = _candidate_in_observation_frame(
                    candidate,
                    target_extent=target_extent,
                    target_center=target_center,
                    object_to_observation=object_to_observation,
                )
                candidate_view_score = view_verifier.score(candidate_world)
                candidate_score = candidate_view_score.score
                selection_metric = "source-view-multiobjective-v2"
                view_score_dict = candidate_view_score.as_dict()
                expansion_row["source_view_score"] = {
                    key: value for key, value in view_score_dict.items() if key != "views"
                }
            expansion_row.update(
                {
                    "executable": True,
                    "candidate_score": candidate_score,
                    "selection_metric": selection_metric,
                }
            )
            if view_verifier is None:
                progress_accepted = candidate_score >= best_iou + args.minimum_score_improvement
                progress_reason = (
                    "mesh-score-improvement" if progress_accepted else "no-mesh-score-improvement"
                )
            else:
                if candidate_view_score is None:
                    raise RuntimeError("source-view candidate lost its verification score")
                progress_accepted, progress_reason = decide_source_view_progress(
                    candidate_view_score,
                    best_view_score,
                    minimum_score_improvement=args.minimum_score_improvement,
                    minimum_topology_recall_improvement=(args.minimum_topology_recall_improvement),
                    maximum_topology_score_regression=args.maximum_topology_score_regression,
                    minimum_appearance_edge_precision=args.minimum_appearance_edge_precision,
                )
            expansion_row.update(
                {
                    "progress_accepted": progress_accepted,
                    "progress_reason": progress_reason,
                }
            )
            construction_graph.update_node(
                node_id,
                score=candidate_score,
                reason=progress_reason,
            )
            expansion_rows.append(expansion_row)
            if not progress_accepted:
                continue
            edge_recall = (
                candidate_view_score.appearance_edge_recall or 0.0
                if candidate_view_score is not None
                else 0.0
            )
            expansion_rank = (
                int(progress_reason != "bounded-appearance-topology-improvement"),
                candidate_score,
                edge_recall,
            )
            if best_expansion_rank is None or expansion_rank > best_expansion_rank:
                best_expansion = (
                    raw,
                    candidate_source,
                    candidate,
                    candidate_score,
                    selection_metric,
                    candidate_view_score,
                    progress_reason,
                    node_id,
                )
                best_expansion_rank = expansion_rank
        row["expansions"] = expansion_rows
        if best_expansion is None:
            row.update(
                {
                    "accepted": False,
                    "reason": (
                        "no-supported-CAD-expansion"
                        if any(item.get("executable") for item in expansion_rows)
                        else "no-executable-CAD-expansion"
                    ),
                }
            )
            trajectory.append(row)
            continue
        (
            raw,
            candidate_source,
            candidate,
            candidate_iou,
            selection_metric,
            candidate_view_score,
            progress_reason,
            selected_node_id,
        ) = best_expansion
        row.update(
            {
                "raw": raw,
                "validated": (
                    raw if "def residual_feature_" in raw else candidate_source.rsplit("\n", 1)[-1]
                ),
                "candidate_score": candidate_iou,
                "selection_metric": selection_metric,
            }
        )
        if candidate_view_score is not None:
            row["source_view_score"] = candidate_view_score.as_dict()
        prefix = candidate_source
        current = candidate
        best_source = candidate_source
        best_mesh = candidate.copy()
        best_iou = candidate_iou
        if view_verifier is not None:
            accepted_world = _candidate_in_observation_frame(
                candidate,
                target_extent=target_extent,
                target_center=target_center,
                object_to_observation=object_to_observation,
            )
            best_view_score = view_verifier.score(accepted_world)
        else:
            best_view_score = None
        row.update({"accepted": True, "reason": progress_reason})
        row["construction_graph_node"] = selected_node_id
        construction_graph.accept(selected_node_id)
        trajectory.append(row)
        accepted_prefixes.append(
            (best_source, best_mesh.copy(), best_iou, best_view_score, step_index)
        )
        best_mesh.export(output / f"step_{step_index:02d}.stl")
        if best_iou >= 0.98:
            break

    selection_candidates: list[
        tuple[str, trimesh.Trimesh, float, SourceViewScore | None, str, int | None]
    ] = [(*value[:4], "learned-cadena", value[4]) for value in accepted_prefixes]
    selection_candidates.extend(
        (*value[:4], f"measured-{value[4]}", None) for value in measured_root_candidates
    )
    if not selection_candidates:
        raise RuntimeError("CAD search produced no kernel-valid candidate")
    if view_verifier is not None:

        def prefix_key(
            value: tuple[
                str,
                trimesh.Trimesh,
                float,
                SourceViewScore | None,
                str,
                int | None,
            ],
        ) -> tuple[bool, float]:
            view_score = value[3]
            if view_score is None:
                return False, value[2]
            decision = decide_source_view_score(
                view_score,
                minimum_silhouette_iou=args.minimum_silhouette_iou,
                minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
                minimum_appearance_edge_precision=args.minimum_appearance_edge_precision,
                minimum_appearance_edge_recall=args.minimum_appearance_edge_recall,
                minimum_appearance_edge_pixels=args.minimum_appearance_edge_pixels,
            )
            return decision.accepted, view_score.score

        (
            best_source,
            best_mesh,
            best_iou,
            best_view_score,
            selected_candidate_origin,
            selected_step,
        ) = max(selection_candidates, key=prefix_key)
    else:
        (
            best_source,
            best_mesh,
            best_iou,
            best_view_score,
            selected_candidate_origin,
            selected_step,
        ) = max(selection_candidates, key=lambda value: value[2])
    if selected_step is not None:
        trajectory[selected_step]["selected_prefix"] = True
    selected_prefix_accepted = (
        decide_source_view_score(
            best_view_score,
            minimum_silhouette_iou=args.minimum_silhouette_iou,
            minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
            minimum_appearance_edge_precision=args.minimum_appearance_edge_precision,
            minimum_appearance_edge_recall=args.minimum_appearance_edge_recall,
            minimum_appearance_edge_pixels=args.minimum_appearance_edge_pixels,
        ).accepted
        if best_view_score is not None
        else False
    )
    measured_axial_cut, measured_axial_cut_candidates = _fit_best_axial_revolved_cut(
        point_cloud,
        best_mesh,
    )
    measured_axial_add, measured_axial_add_candidates = _fit_best_axial_revolved_add(
        point_cloud,
        best_mesh,
    )
    measured_planar_cut, measured_planar_cut_candidates = _fit_best_planar_profile_cut(
        point_cloud,
        best_mesh,
    )
    measured_planar_add, measured_planar_add_candidates = _fit_best_planar_profile_add(
        point_cloud,
        best_mesh,
    )
    measured_feature_report: dict[str, object] = {
        "iteration_limit": 2,
        "axial_revolved_cut": {
            "attempted_axes": [0, 1, 2],
            "candidates": [value.as_dict() for value in measured_axial_cut_candidates],
            "selected_fit": measured_axial_cut.as_dict() if measured_axial_cut else None,
            "role": "source-view-gated-constructive-hypothesis",
        },
        "axial_revolved_add": {
            "attempted_axes": [0, 1, 2],
            "candidates": [value.as_dict() for value in measured_axial_add_candidates],
            "selected_fit": measured_axial_add.as_dict() if measured_axial_add else None,
            "role": "source-view-gated-constructive-hypothesis",
        },
        "planar_profile_cut": {
            "attempted_axes": [0, 1, 2],
            "candidates": [value.as_dict() for value in measured_planar_cut_candidates],
            "selected_fit": measured_planar_cut.as_dict() if measured_planar_cut else None,
            "role": "source-view-gated-constructive-hypothesis",
        },
        "planar_profile_add": {
            "attempted_axes": [0, 1, 2],
            "candidates": [value.as_dict() for value in measured_planar_add_candidates],
            "selected_fit": measured_planar_add.as_dict() if measured_planar_add else None,
            "role": "source-view-gated-constructive-hypothesis",
        },
    }

    simplification_rows: list[dict[str, object]] = []
    simplified_candidates: list[
        tuple[int, float, str, trimesh.Trimesh, SourceViewScore | None, dict[str, object]]
    ] = []
    for tolerance in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        simplification = simplify_revolve_profiles(best_source, tolerance)
        if simplification.profiles == 0 and tolerance > 0.0:
            break
        if any(value[2] == simplification.source for value in simplified_candidates):
            continue
        execution = code_to_mesh(simplification.source, var_name="r", _tol=0.001, _ang=0.1)
        simplified_mesh = execution.get("mesh")
        if not isinstance(simplified_mesh, trimesh.Trimesh) or simplified_mesh.is_empty:
            simplification_rows.append(
                {
                    "tolerance": tolerance,
                    "executable": False,
                    "status": execution.get("status"),
                }
            )
            continue
        try:
            validate_cadquery(_execute_result(simplification.source))
        except Exception as error:  # noqa: BLE001 - isolate one simplification proposal
            simplification_rows.append(
                {
                    "tolerance": tolerance,
                    "executable": False,
                    "status": f"CAD kernel rejected simplification: {error}",
                }
            )
            continue
        if view_verifier is None:
            simplified_score = _score(
                target,
                simplified_mesh,
                transform_pred_mesh,
                compute_iou,
                compute_gms,
                use_iou=use_iou,
            )
            simplified_view_score = None
        else:
            simplified_world = _candidate_in_observation_frame(
                simplified_mesh,
                target_extent=target_extent,
                target_center=target_center,
                object_to_observation=object_to_observation,
            )
            simplified_view_score = view_verifier.score(simplified_world)
            simplified_score = simplified_view_score.score
        row = {
            "tolerance": tolerance,
            "executable": True,
            "feature": None,
            "profiles": simplification.profiles,
            "points_before": simplification.points_before,
            "points_after": simplification.points_after,
            "score": simplified_score,
            "source_view_score": simplified_view_score.as_dict() if simplified_view_score else None,
        }
        simplification_rows.append(row)
        simplified_candidates.append(
            (
                simplification.points_after,
                simplified_score,
                simplification.source,
                simplified_mesh,
                simplified_view_score,
                row,
            )
        )
        feature_frontier: list[
            tuple[
                str,
                trimesh.Trimesh,
                float,
                SourceViewScore | None,
                tuple[dict[str, object], ...],
            ]
        ] = [
            (
                simplification.source,
                simplified_mesh,
                simplified_score,
                simplified_view_score,
                (),
            )
        ]
        for feature_round in range(2):
            next_frontier: list[
                tuple[
                    str,
                    trimesh.Trimesh,
                    float,
                    SourceViewScore | None,
                    tuple[dict[str, object], ...],
                ]
            ] = []
            for (
                parent_source,
                parent_mesh,
                parent_score,
                parent_view_score,
                parent_sequence,
            ) in feature_frontier:
                round_cut, _ = _fit_best_axial_revolved_cut(point_cloud, parent_mesh)
                round_add, _ = _fit_best_axial_revolved_add(point_cloud, parent_mesh)
                round_planar_cut, _ = _fit_best_planar_profile_cut(point_cloud, parent_mesh)
                round_planar_add, _ = _fit_best_planar_profile_add(point_cloud, parent_mesh)
                measured_feature_options: list[
                    tuple[
                        str,
                        AxialRevolvedCut | AxialRevolvedAdd | PlanarProfileCut | PlanarProfileAdd,
                    ]
                ] = []
                if round_cut is not None:
                    measured_feature_options.append(("axial-revolved-cut", round_cut))
                if round_add is not None:
                    measured_feature_options.append(("axial-revolved-add", round_add))
                if round_planar_cut is not None:
                    measured_feature_options.append(("planar-profile-cut", round_planar_cut))
                if round_planar_add is not None:
                    measured_feature_options.append(("planar-profile-add", round_planar_add))
                for feature_kind, measured_feature in measured_feature_options:
                    axis_extent = float(parent_mesh.extents[measured_feature.axis])
                    if isinstance(measured_feature, AxialRevolvedCut):
                        feature_variants = (
                            ("measured", 1.0, 0.0),
                            (
                                "conservative",
                                0.85,
                                measured_feature.side * 0.015 * axis_extent,
                            ),
                        )
                    elif isinstance(measured_feature, AxialRevolvedAdd):
                        feature_variants = (
                            ("measured", 1.0, 1.0),
                            ("conservative", 0.85, 0.95),
                        )
                    elif isinstance(measured_feature, PlanarProfileCut):
                        feature_variants = (
                            ("measured", 1.0, 1.0),
                            ("conservative", 0.90, 1.0),
                        )
                    else:
                        feature_variants = (
                            ("measured", 1.0, 1.0),
                            ("conservative", 0.90, 0.95),
                        )
                    for variant, primary_scale, secondary_scale in feature_variants:
                        if isinstance(measured_feature, AxialRevolvedCut):
                            feature_step = measured_feature.step(
                                radial_scale=primary_scale,
                                floor_offset=secondary_scale,
                            )
                            feature_parameters = {
                                "radial_scale": primary_scale,
                                "floor_offset": secondary_scale,
                            }
                        elif isinstance(measured_feature, AxialRevolvedAdd):
                            feature_step = measured_feature.step(
                                radial_scale=primary_scale,
                                axial_scale=secondary_scale,
                            )
                            feature_parameters = {
                                "radial_scale": primary_scale,
                                "axial_scale": secondary_scale,
                            }
                        elif isinstance(measured_feature, PlanarProfileCut):
                            feature_step = measured_feature.step(profile_scale=primary_scale)
                            feature_parameters = {"profile_scale": primary_scale}
                        else:
                            feature_step = measured_feature.step(
                                profile_scale=primary_scale,
                                axial_scale=secondary_scale,
                            )
                            feature_parameters = {
                                "profile_scale": primary_scale,
                                "axial_scale": secondary_scale,
                            }
                        feature_source = parent_source.rstrip() + "\n" + feature_step
                        if any(value[2] == feature_source for value in simplified_candidates):
                            continue
                        sequence_entry: dict[str, object] = {
                            "round": feature_round + 1,
                            "feature": feature_kind,
                            "operation": feature_kind.replace("-", "_"),
                            "executed_step": feature_step,
                            "variant": variant,
                            "parameters": feature_parameters,
                            "evidence_fit": measured_feature.as_dict(),
                        }
                        feature_sequence = (*parent_sequence, sequence_entry)
                        feature_row: dict[str, object] = {
                            "tolerance": tolerance,
                            "feature": feature_kind,
                            "feature_variant": variant,
                            "feature_round": feature_round + 1,
                            "feature_count": len(feature_sequence),
                            "feature_sequence": list(feature_sequence),
                            "evidence_fit": measured_feature.as_dict(),
                            **feature_parameters,
                            "profiles": simplification.profiles,
                            "points_before": simplification.points_before,
                            "points_after": simplification.points_after,
                        }
                        execution = code_to_mesh(
                            feature_source,
                            var_name="r",
                            _tol=0.001,
                            _ang=0.1,
                        )
                        feature_mesh = execution.get("mesh")
                        if not isinstance(feature_mesh, trimesh.Trimesh) or feature_mesh.is_empty:
                            feature_row.update(
                                {
                                    "executable": False,
                                    "status": execution.get("status"),
                                }
                            )
                            simplification_rows.append(feature_row)
                            continue
                        try:
                            validate_cadquery(_execute_result(feature_source))
                        except Exception as error:  # noqa: BLE001
                            feature_row.update(
                                {
                                    "executable": False,
                                    "status": (f"CAD kernel rejected measured feature: {error}"),
                                }
                            )
                            simplification_rows.append(feature_row)
                            continue
                        if view_verifier is None:
                            feature_score = _score(
                                target,
                                feature_mesh,
                                transform_pred_mesh,
                                compute_iou,
                                compute_gms,
                                use_iou=use_iou,
                            )
                            feature_view_score = None
                        else:
                            feature_world = _candidate_in_observation_frame(
                                feature_mesh,
                                target_extent=target_extent,
                                target_center=target_center,
                                object_to_observation=object_to_observation,
                            )
                            feature_view_score = view_verifier.score(feature_world)
                            feature_score = feature_view_score.score
                        iterative_eligible = (
                            feature_score >= parent_score - args.simplification_score_tolerance
                        )
                        iterative_rejection: str | None = None
                        if (
                            iterative_eligible
                            and parent_view_score is not None
                            and feature_view_score is not None
                        ):
                            regressions = appearance_topology_regressions(
                                parent_view_score,
                                feature_view_score,
                            )
                            if regressions:
                                iterative_eligible = False
                                iterative_rejection = (
                                    "appearance-topology-regression: " + ", ".join(regressions)
                                )
                        if not iterative_eligible and iterative_rejection is None:
                            iterative_rejection = "parent-score-regression"
                        feature_row.update(
                            {
                                "executable": True,
                                "score": feature_score,
                                "source_view_score": (
                                    feature_view_score.as_dict() if feature_view_score else None
                                ),
                                "iterative_eligible": iterative_eligible,
                                "iterative_rejection_reason": iterative_rejection,
                            }
                        )
                        simplification_rows.append(feature_row)
                        simplified_candidates.append(
                            (
                                simplification.points_after,
                                feature_score,
                                feature_source,
                                feature_mesh,
                                feature_view_score,
                                feature_row,
                            )
                        )
                        if iterative_eligible:
                            next_frontier.append(
                                (
                                    feature_source,
                                    feature_mesh,
                                    feature_score,
                                    feature_view_score,
                                    feature_sequence,
                                )
                            )
            if not next_frontier:
                break
            strongest_round_score = max(value[2] for value in next_frontier)
            competitive = [
                value
                for value in next_frontier
                if value[2] >= strongest_round_score - args.simplification_score_tolerance
            ]

            def frontier_risk(
                value: tuple[
                    str,
                    trimesh.Trimesh,
                    float,
                    SourceViewScore | None,
                    tuple[dict[str, object], ...],
                ],
            ) -> tuple[int, int]:
                sequence = value[4]
                measured_variants = sum(item.get("variant") == "measured" for item in sequence)
                return len(sequence), measured_variants

            feature_frontier = [
                min(
                    competitive,
                    key=lambda value: (frontier_risk(value), -value[2]),
                )
            ]
    learned_best_source = best_source
    eligible_simplifications = []
    for value in simplified_candidates:
        candidate_row = value[5]
        if value[1] < best_iou - args.simplification_score_tolerance:
            candidate_row["eligible"] = False
            candidate_row["rejection_reason"] = "primary-score-regression"
            continue
        simplified_view_score = value[4]
        if best_view_score is not None and simplified_view_score is not None:
            edge_regressions = appearance_topology_regressions(
                best_view_score,
                simplified_view_score,
            )
            if edge_regressions:
                candidate_row["eligible"] = False
                candidate_row["rejection_reason"] = "appearance-topology-regression: " + ", ".join(
                    edge_regressions
                )
                continue
        if selected_prefix_accepted and (
            simplified_view_score is None
            or not decide_source_view_score(
                simplified_view_score,
                minimum_silhouette_iou=args.minimum_silhouette_iou,
                minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
                minimum_appearance_edge_precision=args.minimum_appearance_edge_precision,
                minimum_appearance_edge_recall=args.minimum_appearance_edge_recall,
                minimum_appearance_edge_pixels=args.minimum_appearance_edge_pixels,
            ).accepted
        ):
            candidate_row["eligible"] = False
            candidate_row["rejection_reason"] = "accepted-prefix-gate-regression"
            continue
        candidate_row["eligible"] = True
        eligible_simplifications.append(value)
    if eligible_simplifications:
        minimum_points = min(value[0] for value in eligible_simplifications)
        simplest = [value for value in eligible_simplifications if value[0] == minimum_points]
        strongest_score = max(value[1] for value in simplest)
        score_tied = [
            value
            for value in simplest
            if value[1] >= strongest_score - args.simplification_score_tolerance
        ]

        def construction_risk(value: tuple[object, ...]) -> int:
            row = value[5]
            if not isinstance(row, dict) or row.get("feature") is None:
                return 0
            feature_count = int(row.get("feature_count", 1))
            variant_risk = 1 if row.get("feature_variant") == "conservative" else 2
            return 3 * feature_count + variant_risk

        selected = min(
            score_tied,
            key=lambda value: (construction_risk(value), -value[1]),
        )
        _, best_iou, best_source, best_mesh, best_view_score, selected_row = selected
        selected_row["selected"] = True
        selected_sequence = selected_row.get("feature_sequence")
        selected_base_source = best_source.rstrip()
        if isinstance(selected_sequence, list):
            for selected_feature in reversed(selected_sequence):
                if not isinstance(selected_feature, dict):
                    raise RuntimeError("selected measured feature provenance is invalid")
                executed_step = selected_feature.get("executed_step")
                if not isinstance(executed_step, str) or not selected_base_source.endswith(
                    executed_step
                ):
                    raise RuntimeError("selected feature chain does not match final program")
                selected_base_source = selected_base_source[: -len(executed_step)].rstrip()
        selected_base_source += "\n"
        verified_prefix = selected_base_source.rstrip()
        if selected_base_source != learned_best_source:
            base_validation = validate_cadquery(_execute_result(selected_base_source)).as_dict()
            construction_graph.accept_program_rewrite(
                rewrite_index=0,
                source=selected_base_source,
                evidence={
                    "kind": "profile-simplification",
                    "tolerance": selected_row.get("tolerance"),
                    "points_before": selected_row.get("points_before"),
                    "points_after": selected_row.get("points_after"),
                },
                kernel_validation=base_validation,
            )
        if isinstance(selected_sequence, list):
            for feature_index, selected_feature in enumerate(selected_sequence):
                if not isinstance(selected_feature, dict):
                    raise RuntimeError("selected measured feature provenance is invalid")
                operation = selected_feature.get("operation")
                executed_step = selected_feature.get("executed_step")
                evidence_fit = selected_feature.get("evidence_fit")
                if (
                    not isinstance(operation, str)
                    or not isinstance(executed_step, str)
                    or not isinstance(evidence_fit, dict)
                ):
                    raise RuntimeError("selected measured feature is missing provenance")
                verified_prefix += "\n" + executed_step
                prefix_validation = validate_cadquery(_execute_result(verified_prefix)).as_dict()
                construction_graph.accept_trusted_feature(
                    feature_index=feature_index,
                    operation=operation,
                    step=executed_step,
                    evidence=evidence_fit,
                    kernel_validation=prefix_validation,
                )

    source_decision = (
        decide_source_view_score(
            best_view_score,
            minimum_silhouette_iou=args.minimum_silhouette_iou,
            minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
            minimum_appearance_edge_precision=args.minimum_appearance_edge_precision,
            minimum_appearance_edge_recall=args.minimum_appearance_edge_recall,
            minimum_appearance_edge_pixels=args.minimum_appearance_edge_pixels,
        )
        if best_view_score is not None
        else None
    )
    accepted = source_decision is None or source_decision.accepted
    artifact_stem = "model" if accepted else "candidate"
    program_path = output / f"{artifact_stem}.py"
    preview_path = output / f"{artifact_stem}.stl"
    step_path = output / f"{artifact_stem}.step"
    program_path.write_text(best_source.rstrip() + "\n", encoding="utf-8")
    best_mesh.export(preview_path)
    kernel_validation = _execute_final(best_source, step_path)
    report = {
        "schema_version": "da3-cad-cadena-direct-v9",
        "upstream_checkout": str(args.checkout.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "target_mesh": str(args.mesh.resolve()),
        "target_conditioning_mesh": str(target_mesh_path),
        "target_measurements": measurement_report,
        "object_frame": object_frame_report,
        "seed": args.seed,
        "maximum_steps": args.max_steps,
        "expansions_per_step": args.expansions,
        "sampling_temperature": args.temperature if args.expansions > 1 else None,
        "proxy_quantization_step": CADENA_PROXY_RENDER_QUANTIZATION_STEP,
        "renderer_canonicalization": {
            "method": "optional-lanczos-downsample-then-nearest-channel-quantization",
            "measured_downsample_factor": CADENA_RENDER_DOWNSAMPLE_FACTOR,
            "proposal_downsample_factor": CADENA_PROXY_RENDER_DOWNSAMPLE_FACTOR,
            "proposal_spur_filter_size": CADENA_PROXY_SPUR_FILTER_SIZE,
            "quantization_step": CADENA_RENDER_QUANTIZATION_STEP,
            "reason": "remove sub-pixel off-screen raster jitter before policy inference",
        },
        "proposal_proxy": proposal_proxy_report,
        "measured_root_candidates": measured_root_reports,
        "measured_features": measured_feature_report,
        "minimum_score_improvement": args.minimum_score_improvement,
        "profile_simplification": {
            "enabled": bool(simplification_rows),
            "score_tolerance": args.simplification_score_tolerance,
            "candidates": simplification_rows,
        },
        "accepted_steps": sum(bool(row.get("accepted")) for row in trajectory),
        "selected_prefix_step": selected_step,
        "selected_candidate_origin": selected_candidate_origin,
        "construction_graph": construction_graph.as_dict(),
        "search_policy": (
            "score measured revolve and sketch-extrusion roots beside learned "
            "proposals, condition the first learned proposal on an evidence-fitted "
            "revolve proxy, bind "
            "decoded operations to signed residual patches, then iteratively fit bounded "
            "additive or subtractive axial or planar profiles from measured target "
            "evidence; retain "
            "only kernel-valid source-view-non-regressing candidates and backtrack"
        ),
        "selection_metric": (
            "source-view-multiobjective-v3-surface-topology"
            if view_verifier is not None
            else ("IoU" if use_iou else "GMS")
        ),
        "verification_workspace": (
            str(args.verification_workspace.resolve())
            if args.verification_workspace is not None
            else None
        ),
        "final_selection_score": best_iou,
        "final_source_view_score": best_view_score.as_dict() if best_view_score else None,
        "decision": source_decision.as_dict()
        if source_decision
        else {
            "decision": "ACCEPT",
            "accepted": True,
            "reasons": [],
        },
        "source_view_thresholds": (
            {
                "minimum_silhouette_iou": args.minimum_silhouette_iou,
                "minimum_depth_inlier_fraction": args.minimum_depth_inlier_fraction,
                "minimum_appearance_edge_precision": (args.minimum_appearance_edge_precision),
                "minimum_appearance_edge_recall": args.minimum_appearance_edge_recall,
                "minimum_appearance_edge_pixels": args.minimum_appearance_edge_pixels,
                "status": "provisional-controlled-diagnostic-v2",
            }
            if view_verifier is not None
            else None
        ),
        "target_watertight": bool(target_raw.is_watertight),
        "preview_mesh_watertight": bool(best_mesh.is_watertight),
        "kernel_validation": kernel_validation,
        "kernel_step_exported": step_path.is_file(),
        "step_path": step_path.name,
        "program_path": program_path.name,
        "preview_path": preview_path.name,
        "trajectory": trajectory,
        "safety": (
            "measured revolve and sketch-extrusion roots are kernel-validated and "
            "source-view-scored before selection; each model response is restricted "
            "to one assignment calling a published CADENA "
            "DSL operation with literal arguments; trusted measured add/cut operations are "
            "not in that allowlist; axial features require circumferential signed-distance "
            "support, while planar features require 75% axial coverage and a constant-section "
            "residual at most 0.15; all require a valid single-solid B-Rep and source-view "
            "non-regression; "
            "arbitrary residual-component unions are not part of the direct runner"
        ),
        "mesh_contract": (
            "model.stl is a tessellated preview and may contain face-boundary seams; "
            "kernel_validation and model.step are authoritative"
        ),
    }
    (output / "cadena_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
