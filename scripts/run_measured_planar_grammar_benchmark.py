#!/usr/bin/env python3
"""Exercise the trusted v6 planar grammar without learned-policy or GT leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cadquery as cq
import matplotlib
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

from da3_cad.cad_validation import validate_cadquery
from da3_cad.integrations.measured_features import (
    PlanarProfileAdd,
    PlanarProfileCut,
    fit_planar_profile_add_candidates,
    fit_planar_profile_cut_candidates,
    planar_profile_add,
    planar_profile_cut,
)
from da3_cad.models import FloatArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "measured-planar-grammar-v6"
DEFAULT_LEDGER = ROOT / "docs" / "results" / "measured-planar-grammar-v6.json"
SAMPLE_COUNT = 32_000
SEED = 20260816


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    operation: str
    axis: int
    root: cq.Workplane
    target: cq.Workplane


def _mesh(result: cq.Workplane) -> trimesh.Trimesh:
    shape = cast(cq.Shape, result.val())
    vertices, faces = shape.tessellate(0.002, 0.1)
    return trimesh.Trimesh(
        vertices=np.asarray([vertex.toTuple() for vertex in vertices], dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )


def _cases() -> tuple[Case, ...]:
    l_root = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    l_target = planar_profile_add(
        l_root,
        1,
        ((-0.8, 0.8), (-0.8, 2.2), (-0.2, 2.2), (-0.2, 1.4), (0.8, 1.4), (0.8, 0.8)),
        -1.0,
        1.0,
    )

    t_root = cq.Workplane("XY").box(3.0, 2.0, 1.0)
    t_target = planar_profile_add(
        t_root,
        2,
        (
            (-0.3, 0.8),
            (0.3, 0.8),
            (0.3, 1.5),
            (1.1, 1.5),
            (1.1, 2.1),
            (-1.1, 2.1),
            (-1.1, 1.5),
            (-0.3, 1.5),
        ),
        -0.5,
        0.5,
    )

    u_root = cq.Workplane("XY").box(4.0, 4.0, 2.0)
    u_target = planar_profile_cut(
        u_root,
        2,
        (
            (-1.3, -1.3),
            (-0.8, -1.3),
            (-0.8, 0.8),
            (0.8, 0.8),
            (0.8, -1.3),
            (1.3, -1.3),
            (1.3, 1.3),
            (-1.3, 1.3),
        ),
        -1.1,
        1.1,
    )

    hex_root = cq.Workplane("XY").box(4.0, 4.0, 2.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False) + np.pi / 6.0
    hex_profile = tuple((float(np.cos(a)), float(np.sin(a))) for a in angles)
    hex_target = planar_profile_cut(hex_root, 2, hex_profile, -1.1, 1.1)

    return (
        Case("l_add", "add", 1, l_root, l_target),
        Case("t_add", "add", 2, t_root, t_target),
        Case("u_cut", "cut", 2, u_root, u_target),
        Case("hex_cut", "cut", 2, hex_root, hex_target),
    )


def _rank(
    candidates: tuple[PlanarProfileAdd | PlanarProfileCut, ...],
) -> PlanarProfileAdd | PlanarProfileCut | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            value.axial_coverage_fraction,
            value.profile_occupancy_iou,
            -value.constant_section_residual,
            value.support_points,
        ),
    )


def _fit(
    case: Case,
    points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> PlanarProfileAdd | PlanarProfileCut | None:
    if case.operation == "add":
        return _rank(
            fit_planar_profile_add_candidates(
                points,
                current_mesh,
                minimum_surface_distance=0.03,
                minimum_support_points=64,
            )
        )
    return _rank(
        fit_planar_profile_cut_candidates(
            points,
            current_mesh,
            minimum_surface_distance=0.03,
            minimum_support_points=64,
        )
    )


def _execute(
    root: cq.Workplane,
    feature: PlanarProfileAdd | PlanarProfileCut,
) -> cq.Workplane:
    if isinstance(feature, PlanarProfileAdd):
        return planar_profile_add(root, feature.axis, feature.profile, feature.lower, feature.upper)
    return planar_profile_cut(root, feature.axis, feature.profile, feature.lower, feature.upper)


def _iou(prediction: cq.Workplane, target: cq.Workplane) -> float:
    prediction_shape = cast(cq.Shape, prediction.val())
    target_shape = cast(cq.Shape, target.val())
    intersection = float(prediction_shape.intersect(target_shape).Volume())
    union = float(prediction_shape.Volume() + target_shape.Volume() - intersection)
    return intersection / union


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(result: cq.Workplane, path: Path) -> None:
    cq.exporters.export(result, str(path))


def _draw_mesh(axis: Any, mesh: trimesh.Trimesh, color: str) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    light = np.asarray((0.45, -0.35, 0.82), dtype=np.float64)
    light /= np.linalg.norm(light)
    illumination = 0.58 + 0.42 * np.maximum(np.asarray(mesh.face_normals) @ light, 0.0)
    face_colors = np.clip(
        np.asarray(to_rgb(color), dtype=np.float64)[None, :] * illumination[:, None],
        0.0,
        1.0,
    )
    collection = Poly3DCollection(
        vertices[faces],
        facecolor=face_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.96,
    )
    axis.add_collection3d(collection)
    center = mesh.bounds.mean(axis=0)
    radius = 0.55 * float(np.max(mesh.extents))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24, azim=-56)
    axis.set_axis_off()


def _render_montage(
    rows: list[tuple[Case, trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh, float]],
    path: Path,
) -> None:
    figure = plt.figure(figsize=(10.5, 11.2), facecolor="#f5f2ea")
    titles = ("Known root", "Measured target", "Recovered B-Rep")
    colors = ("#8aa4b8", "#e1a85a", "#67a88b")
    for row_index, (case, root, target, prediction, iou) in enumerate(rows):
        for column, (title, mesh, color) in enumerate(
            zip(titles, (root, target, prediction), colors, strict=True)
        ):
            axis = figure.add_subplot(len(rows), 3, row_index * 3 + column + 1, projection="3d")
            _draw_mesh(axis, mesh, color)
            if row_index == 0:
                axis.set_title(title, fontsize=12, weight="bold", pad=2)
            if column == 0:
                axis.text2D(
                    -0.08,
                    0.5,
                    case.case_id.replace("_", " ").upper(),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
            if column == 2:
                axis.text2D(
                    0.5,
                    0.02,
                    f"exact volume IoU {iou:.3f}",
                    transform=axis.transAxes,
                    ha="center",
                    fontsize=9,
                )
    figure.suptitle(
        "DA3-CAD v6 · signed residual → arbitrary 2D profile → trusted add/cut",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.01, 0.98, 0.975))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _frustum_negative_control() -> tuple[cq.Workplane, cq.Workplane]:
    root = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    frustum = (
        cq.Workplane("XY", origin=(0.0, 0.0, 0.98))
        .rect(0.7, 0.7)
        .workplane(offset=1.8)
        .rect(2.0, 1.5)
        .loft(combine=True)
    )
    return root, root.union(frustum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    visual_rows: list[tuple[Case, trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh, float]] = []
    for index, case in enumerate(_cases()):
        case_output = args.output / case.case_id
        case_output.mkdir(parents=True)
        root_mesh = _mesh(case.root)
        target_mesh = _mesh(case.target)
        points, _ = trimesh.sample.sample_surface(target_mesh, SAMPLE_COUNT, seed=SEED + index)
        points = np.asarray(points, dtype=np.float64)
        feature = _fit(case, points, root_mesh)
        if feature is None:
            raise RuntimeError(f"no measured feature recovered for {case.case_id}")
        prediction = _execute(case.root, feature)
        validation = validate_cadquery(prediction)
        if not validation.valid or validation.solids != 1:
            raise RuntimeError(f"invalid B-Rep recovered for {case.case_id}: {validation}")
        prediction_mesh = _mesh(prediction)
        iou = _iou(prediction, case.target)

        root_path = case_output / "root.step"
        target_path = case_output / "target.step"
        model_path = case_output / "model.step"
        points_path = case_output / "measured_target.ply"
        _export(case.root, root_path)
        _export(case.target, target_path)
        _export(prediction, model_path)
        trimesh.PointCloud(points).export(points_path)  # type: ignore[no-untyped-call]
        records.append(
            {
                "id": case.case_id,
                "expected_operation": f"planar_profile_{case.operation}",
                "expected_axis": case.axis,
                "selected_axis": feature.axis,
                "feature": feature.as_dict(),
                "executed_step": feature.step(),
                "kernel_validation": validation.as_dict(),
                "exact_volume_iou": iou,
                "artifacts": {
                    "root_step": str(root_path.relative_to(ROOT)),
                    "target_step": str(target_path.relative_to(ROOT)),
                    "model_step": str(model_path.relative_to(ROOT)),
                    "measured_target": str(points_path.relative_to(ROOT)),
                    "model_step_sha256": _sha256(model_path),
                },
            }
        )
        visual_rows.append((case, root_mesh, target_mesh, prediction_mesh, iou))

    negative_root, negative_target = _frustum_negative_control()
    negative_mesh = _mesh(negative_target)
    negative_points, _ = trimesh.sample.sample_surface(
        negative_mesh,
        SAMPLE_COUNT,
        seed=SEED + 100,
    )
    negative_add = fit_planar_profile_add_candidates(
        np.asarray(negative_points, dtype=np.float64),
        _mesh(negative_root),
        minimum_surface_distance=0.03,
        minimum_support_points=64,
    )
    negative_control = {
        "id": "nonconstant_frustum_add",
        "expected": "reject planar constant-section hypothesis",
        "candidate_count": len(negative_add),
        "passed": len(negative_add) == 0,
    }
    if not negative_control["passed"]:
        raise RuntimeError("planar grammar admitted the non-constant frustum control")

    montage_path = args.output / "benchmark.png"
    _render_montage(visual_rows, montage_path)
    mean_iou = float(np.mean([float(record["exact_volume_iou"]) for record in records]))
    payload = {
        "schema_version": "da3-cad-measured-planar-grammar-v6",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": {
            "purpose": "deterministic grammar capability and false-positive control",
            "input": f"{SAMPLE_COUNT} sampled target-surface points plus a known root B-Rep",
            "fit_reference_cad_access": False,
            "reference_role": "post-hoc exact-volume evaluator and visualization only",
            "not_claimed": "end-to-end real-photo accuracy",
        },
        "method": {
            "residual": "signed target surface relative to the current watertight B-Rep",
            "profile": "largest supported raster component simplified to an arbitrary polyline",
            "operations": ["planar_profile_add", "planar_profile_cut"],
            "axis_search": [0, 1, 2],
            "minimum_axial_coverage": 0.75,
            "maximum_constant_section_residual": 0.15,
            "kernel_contract": "exactly one valid solid",
        },
        "summary": {
            "positive_cases": len(records),
            "valid_single_solid_steps": sum(
                bool(record["kernel_validation"]["valid"])
                and int(record["kernel_validation"]["solids"]) == 1
                for record in records
            ),
            "correct_axis_selections": sum(
                int(record["expected_axis"]) == int(record["selected_axis"]) for record in records
            ),
            "mean_exact_volume_iou": mean_iou,
            "negative_controls_passed": int(bool(negative_control["passed"])),
        },
        "cases": records,
        "negative_controls": [negative_control],
        "visual": str(montage_path.relative_to(ROOT)),
    }
    args.ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Ledger: {args.ledger}")
    print(f"Visual: {montage_path}")


if __name__ == "__main__":
    main()
