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
from da3_cad.cad_validation import validate_and_export_cadquery
from da3_cad.evaluation.source_view_verifier import (
    SourceViewScore,
    SourceViewVerifier,
    decide_source_view_score,
)
from da3_cad.integrations.render_canonicalization import (
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
        CODE_PREFIX,
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


def _execute_final(source: str, output_path: Path) -> dict[str, int | float | bool]:
    namespace: dict[str, object] = {}
    exec(compile(source, "<cadena-validated-program>", "exec"), None, namespace)
    result = namespace.get("r")
    if result is None:
        raise RuntimeError("validated CADENA program produced no r solid")
    return validate_and_export_cadquery(result, output_path).as_dict()


def main() -> None:
    args = _parser().parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    if args.max_steps < 1 or args.max_new_tokens < 1:
        raise ValueError("step and token budgets must be positive")
    if args.expansions < 1 or args.expansions > 16:
        raise ValueError("expansions must be in [1, 16]")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if args.minimum_score_improvement < 0.0:
        raise ValueError("minimum score improvement must be non-negative")
    if args.simplification_score_tolerance < 0.0:
        raise ValueError("simplification score tolerance must be non-negative")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Plotter, get_point, code_prefix, code_to_mesh, transform_target, metric_tools = _imports(
        args.checkout
    )
    transform_pred_mesh, compute_iou, compute_gms = metric_tools

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    target_raw = trimesh.load(args.mesh.resolve(), force="mesh", process=True)
    if not isinstance(target_raw, trimesh.Trimesh) or target_raw.is_empty:
        raise ValueError(f"target is not a non-empty triangle mesh: {args.mesh}")
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
    target_image = canonicalize_rgb_render(plotter.get_img(str(args.mesh.resolve()), None))
    target_image.save(output / "target.png")

    prefix = code_prefix
    best_source: str | None = None
    best_mesh: trimesh.Trimesh | None = None
    best_iou = 0.0
    current: trimesh.Trimesh | None = None
    point_cloud: FloatArray | None = None
    trajectory: list[dict[str, object]] = []
    best_view_score: SourceViewScore | None = None
    for step_index in range(args.max_steps):
        if point_cloud is None:
            point, point_cloud = get_point(str(args.mesh.resolve()), None, seed=args.seed)
        else:
            point, _ = get_point(point_cloud, current, seed=args.seed + step_index)
        image = (
            target_image
            if current is None
            else canonicalize_rgb_render(plotter.get_img(str(args.mesh.resolve()), current))
        )
        image.save(output / f"step_{step_index:02d}_input.png")
        raw_steps = _generate_steps(
            model,
            processor,
            image,
            point,
            max_new_tokens=args.max_new_tokens,
            expansions=args.expansions,
            temperature=args.temperature,
        )
        row: dict[str, object] = {
            "step": step_index,
            "point": point,
            "previous_score": best_iou,
        }
        expansion_rows: list[dict[str, object]] = []
        best_expansion: (
            tuple[str, str, trimesh.Trimesh, float, str, dict[str, object] | None] | None
        )
        best_expansion = None
        for expansion_index, raw in enumerate(raw_steps):
            expansion_row: dict[str, object] = {"index": expansion_index, "raw": raw}
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
                expansion_rows.append(expansion_row)
                continue
            view_score_dict: dict[str, object] | None = None
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
                candidate_world = candidate.copy()
                candidate_world.apply_scale(target_extent / 200.0)  # type: ignore[no-untyped-call]
                candidate_world.apply_translation(target_center)
                view_score = view_verifier.score(candidate_world)
                candidate_score = view_score.score
                selection_metric = "source-view-silhouette-and-depth"
                view_score_dict = view_score.as_dict()
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
            expansion_rows.append(expansion_row)
            if best_expansion is None or candidate_score > best_expansion[3]:
                best_expansion = (
                    raw,
                    candidate_source,
                    candidate,
                    candidate_score,
                    selection_metric,
                    view_score_dict,
                )
        row["expansions"] = expansion_rows
        if best_expansion is None:
            row.update({"accepted": False, "reason": "no-executable-CAD-expansion"})
            trajectory.append(row)
            break
        raw, candidate_source, candidate, candidate_iou, selection_metric, view_score_dict = (
            best_expansion
        )
        row.update(
            {
                "raw": raw,
                "validated": candidate_source.rsplit("\n", 1)[-1],
                "candidate_score": candidate_iou,
                "selection_metric": selection_metric,
            }
        )
        if view_score_dict is not None:
            row["source_view_score"] = view_score_dict
        if candidate_iou < best_iou + args.minimum_score_improvement:
            row.update(
                {
                    "accepted": False,
                    "reason": "operation-did-not-meet-minimum-score-improvement",
                    "minimum_score_improvement": args.minimum_score_improvement,
                }
            )
            trajectory.append(row)
            break
        prefix = candidate_source
        current = candidate
        best_source = candidate_source
        best_mesh = candidate.copy()
        best_iou = candidate_iou
        if view_verifier is not None:
            accepted_world = candidate.copy()
            accepted_world.apply_scale(target_extent / 200.0)  # type: ignore[no-untyped-call]
            accepted_world.apply_translation(target_center)
            best_view_score = view_verifier.score(accepted_world)
        else:
            best_view_score = None
        row.update({"accepted": True, "reason": "minimum-score-improvement-met"})
        trajectory.append(row)
        best_mesh.export(output / f"step_{step_index:02d}.stl")
        if best_iou >= 0.98:
            break

    if best_source is None or best_mesh is None:
        raise RuntimeError("CADENA produced no improving, executable CAD operation")
    simplification_rows: list[dict[str, object]] = []
    simplified_candidates: list[
        tuple[int, float, str, trimesh.Trimesh, SourceViewScore | None, dict[str, object]]
    ] = []
    for tolerance in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        simplification = simplify_revolve_profiles(best_source, tolerance)
        if simplification.profiles == 0:
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
            simplified_world = simplified_mesh.copy()
            simplified_world.apply_scale(target_extent / 200.0)  # type: ignore[no-untyped-call]
            simplified_world.apply_translation(target_center)
            simplified_view_score = view_verifier.score(simplified_world)
            simplified_score = simplified_view_score.score
        row = {
            "tolerance": tolerance,
            "executable": True,
            "profiles": simplification.profiles,
            "points_before": simplification.points_before,
            "points_after": simplification.points_after,
            "score": simplified_score,
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
    eligible_simplifications = [
        value
        for value in simplified_candidates
        if value[1] >= best_iou - args.simplification_score_tolerance
    ]
    if eligible_simplifications:
        selected = min(eligible_simplifications, key=lambda value: (value[0], -value[1]))
        _, best_iou, best_source, best_mesh, best_view_score, selected_row = selected
        selected_row["selected"] = True

    source_decision = (
        decide_source_view_score(
            best_view_score,
            minimum_silhouette_iou=args.minimum_silhouette_iou,
            minimum_depth_inlier_fraction=args.minimum_depth_inlier_fraction,
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
        "schema_version": "da3-cad-cadena-direct-v1",
        "upstream_checkout": str(args.checkout.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "target_mesh": str(args.mesh.resolve()),
        "seed": args.seed,
        "maximum_steps": args.max_steps,
        "expansions_per_step": args.expansions,
        "sampling_temperature": args.temperature if args.expansions > 1 else None,
        "renderer_canonicalization": {
            "method": "nearest-channel-quantization",
            "quantization_step": CADENA_RENDER_QUANTIZATION_STEP,
            "reason": "remove sub-quantization off-screen raster jitter before policy inference",
        },
        "minimum_score_improvement": args.minimum_score_improvement,
        "profile_simplification": {
            "enabled": bool(simplification_rows),
            "score_tolerance": args.simplification_score_tolerance,
            "candidates": simplification_rows,
        },
        "accepted_steps": sum(bool(row.get("accepted")) for row in trajectory),
        "selection_metric": (
            "source-view-silhouette-and-depth"
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
                "status": "provisional-controlled-benchmark-v1",
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
            "each model response is restricted to one assignment calling a published CADENA "
            "DSL operation with literal arguments before execution"
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
