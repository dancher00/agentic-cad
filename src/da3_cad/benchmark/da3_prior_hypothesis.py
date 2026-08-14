"""Exact-CAD fixtures for the preregistered DA3-prior shape experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageDraw

from da3_cad.benchmark.public_cases import (
    CAMERA_SCHEDULE,
    IMAGE_SIZE,
    PUBLIC_CASES,
    PublicCaseSpec,
    _camera,
    _normalize_step,
    _render,
    _shape,
)
from da3_cad.geometry.cameras import CameraBundle
from da3_cad.integrations.gaussian_depth_prior import spherical_camera_coverage

LOW_COVERAGE_INDICES = (2, 4, 5, 8, 10)
MEDIUM_COVERAGE_INDICES = (0, 2, 6, 8, 11)
HIGH_COVERAGE_INDICES = (0, 4, 6, 8, 9)
COVERAGE_SUBSETS = {
    "low": LOW_COVERAGE_INDICES,
    "medium": MEDIUM_COVERAGE_INDICES,
    "high": HIGH_COVERAGE_INDICES,
}

HELD_OUT_SCHEDULE = (
    {"azimuth_deg": 25.0, "elevation_deg": -32.0},
    {"azimuth_deg": 115.0, "elevation_deg": 6.0},
    {"azimuth_deg": 205.0, "elevation_deg": -48.0},
    {"azimuth_deg": 295.0, "elevation_deg": 8.0},
)
HELD_OUT_INDICES = tuple(range(len(CAMERA_SCHEDULE), len(CAMERA_SCHEDULE) + len(HELD_OUT_SCHEDULE)))


@dataclass(frozen=True, slots=True)
class PriorHypothesisCase:
    spec: PublicCaseSpec
    group: str
    axial: bool


_PUBLIC_BY_ID = {spec.case_id: spec for spec in PUBLIC_CASES}
_NEW_CASES = (
    PublicCaseSpec(
        "hollow_sleeve",
        "Hollow sleeve",
        "revolve+cut",
        {"outer_diameter": 38.0, "inner_diameter": 22.0, "length": 46.0},
        (180, 92, 78),
        "axial interior concavity",
    ),
    PublicCaseSpec(
        "grooved_shaft",
        "Grooved shaft",
        "revolve+cut",
        {
            "diameter": 28.0,
            "length": 54.0,
            "groove_depth": 4.0,
            "groove_width": 9.0,
        },
        (78, 116, 177),
        "axial exterior concavity",
    ),
    PublicCaseSpec(
        "wedge",
        "Triangular wedge",
        "extrude",
        {"length": 48.0, "width": 32.0, "height": 24.0},
        (196, 131, 53),
        "non-axial convex control",
    ),
    PublicCaseSpec(
        "solid_cylinder",
        "Solid cylinder",
        "revolve",
        {"diameter": 34.0, "length": 46.0},
        (76, 146, 156),
        "axial convex control",
    ),
    PublicCaseSpec(
        "conical_frustum",
        "Conical frustum",
        "revolve",
        {"base_diameter": 42.0, "top_diameter": 24.0, "height": 42.0},
        (137, 101, 178),
        "tapered axial convex control",
    ),
)
_NEW_BY_ID = {spec.case_id: spec for spec in _NEW_CASES}

HYPOTHESIS_CASES = (
    PriorHypothesisCase(_PUBLIC_BY_ID["l_bracket"], "concave", False),
    PriorHypothesisCase(_PUBLIC_BY_ID["t_bracket"], "concave", False),
    PriorHypothesisCase(_PUBLIC_BY_ID["u_channel"], "concave", False),
    PriorHypothesisCase(_NEW_BY_ID["hollow_sleeve"], "concave", True),
    PriorHypothesisCase(_NEW_BY_ID["grooved_shaft"], "concave", True),
    PriorHypothesisCase(_PUBLIC_BY_ID["block"], "convex", False),
    PriorHypothesisCase(_PUBLIC_BY_ID["hex_prism"], "convex", False),
    PriorHypothesisCase(_NEW_BY_ID["wedge"], "convex", False),
    PriorHypothesisCase(_NEW_BY_ID["solid_cylinder"], "convex", True),
    PriorHypothesisCase(_NEW_BY_ID["conical_frustum"], "convex", True),
)


def hypothesis_shape(case_id: str) -> cq.Workplane:
    """Construct the exact reference solid for a preregistered case."""

    if case_id in _PUBLIC_BY_ID:
        return _shape(_PUBLIC_BY_ID[case_id])
    spec = _NEW_BY_ID[case_id]
    p = spec.parameters_mm
    if case_id == "hollow_sleeve":
        return (
            cq.Workplane("XY")
            .circle(p["outer_diameter"] / 2.0)
            .circle(p["inner_diameter"] / 2.0)
            .extrude(p["length"] / 2.0, both=True)
        )
    if case_id == "grooved_shaft":
        shaft = cq.Workplane("XY").circle(p["diameter"] / 2.0).extrude(p["length"] / 2.0, both=True)
        groove = (
            cq.Workplane("XY")
            .circle(p["diameter"])
            .circle(p["diameter"] / 2.0 - p["groove_depth"])
            .extrude(p["groove_width"] / 2.0, both=True)
        )
        return shaft.cut(groove)
    if case_id == "wedge":
        half_length = p["length"] / 2.0
        return (
            cq.Workplane("XZ")
            .polyline(
                (
                    (-half_length, -p["height"] / 2.0),
                    (half_length, -p["height"] / 2.0),
                    (-half_length, p["height"] / 2.0),
                )
            )
            .close()
            .extrude(p["width"] / 2.0, both=True)
        )
    if case_id == "solid_cylinder":
        return cq.Workplane("XY").circle(p["diameter"] / 2.0).extrude(p["length"] / 2.0, both=True)
    if case_id == "conical_frustum":
        return (
            cq.Workplane("XY")
            .circle(p["base_diameter"] / 2.0)
            .workplane(offset=p["height"])
            .circle(p["top_diameter"] / 2.0)
            .loft(combine=True)
            .translate((0.0, 0.0, -p["height"] / 2.0))
        )
    raise ValueError(f"unknown hypothesis case: {case_id}")


def _face_mesh(shape: cq.Workplane) -> tuple[trimesh.Trimesh, npt.NDArray[np.uint8]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    labels: list[int] = []
    for label, face in enumerate(shape.findSolid().Faces(), start=1):
        if label > 255:
            raise ValueError("face labels exceed the 8-bit Stage 2 mask format")
        face_vertices, face_triangles = face.tessellate(0.01)
        offset = len(vertices)
        vertices.extend(
            (float(vertex.x), float(vertex.y), float(vertex.z)) for vertex in face_vertices
        )
        triangles.extend(
            (
                offset + int(triangle[0]),
                offset + int(triangle[1]),
                offset + int(triangle[2]),
            )
            for triangle in face_triangles
        )
        labels.extend([label] * len(face_triangles))
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    return mesh, np.asarray(labels, dtype=np.uint8)


def _render_patch_mask(
    mesh: trimesh.Trimesh,
    face_labels: npt.NDArray[np.uint8],
    intrinsic: npt.NDArray[np.float64],
    extrinsic: npt.NDArray[np.float64],
) -> Image.Image:
    scale = 2
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    pixels_h = vertices @ intrinsic.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    pixels *= scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    order = np.argsort(triangles[..., 2].mean(axis=1))[::-1]
    image = Image.new("L", (IMAGE_SIZE * scale, IMAGE_SIZE * scale), 0)
    draw = ImageDraw.Draw(image, mode="L")
    for face_index in order:
        polygon = [tuple(float(value) for value in pixels[index]) for index in faces[face_index]]
        draw.polygon(polygon, fill=int(face_labels[face_index]))
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)


def build_da3_prior_hypothesis_fixtures(root: Path) -> dict[str, object]:
    """Build the frozen ten-object, sixteen-view exact-CAD fixture set."""

    if root.exists() and any(root.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty fixture root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schedules = (*CAMERA_SCHEDULE, *HELD_OUT_SCHEDULE)
    image_names = tuple(f"view_{index:03d}.png" for index in range(len(schedules)))
    records: list[dict[str, object]] = []
    for case in HYPOTHESIS_CASES:
        spec = case.spec
        case_root = root / spec.case_id
        for directory in ("views", "masks", "patch_masks"):
            (case_root / directory).mkdir(parents=True, exist_ok=True)
        shape = hypothesis_shape(spec.case_id)
        step_path = case_root / "gt.step"
        stl_path = case_root / "gt.stl"
        cq.exporters.export(shape, str(step_path))
        _normalize_step(step_path)
        cq.exporters.export(shape, str(stl_path), tolerance=0.01, angularTolerance=0.1)
        reference = trimesh.load_mesh(stl_path, process=True)
        if not isinstance(reference, trimesh.Trimesh) or not reference.is_watertight:
            raise ValueError(f"hypothesis reference is not watertight: {spec.case_id}")
        render_mesh, face_labels = _face_mesh(shape)
        target = np.asarray(reference.bounds, dtype=np.float64).mean(axis=0)
        intrinsics: list[npt.NDArray[np.float64]] = []
        extrinsics: list[npt.NDArray[np.float64]] = []
        for index, pose in enumerate(schedules):
            intrinsic, extrinsic = _camera(pose, target)
            image, mask = _render(render_mesh, intrinsic, extrinsic, spec.color_rgb)
            patch_mask = _render_patch_mask(render_mesh, face_labels, intrinsic, extrinsic)
            patch_values = np.asarray(patch_mask, dtype=np.uint8).copy()
            patch_values[np.asarray(mask, dtype=np.uint8) == 0] = 0
            image.save(case_root / "views" / image_names[index], optimize=True)
            mask.save(case_root / "masks" / image_names[index], optimize=True)
            Image.fromarray(patch_values, mode="L").save(
                case_root / "patch_masks" / image_names[index], optimize=True
            )
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
        extrinsic_array = np.stack(extrinsics)
        coverages = {
            name: spherical_camera_coverage(
                extrinsic_array[np.asarray(indices)], object_center=tuple(target)
            )
            for name, indices in COVERAGE_SUBSETS.items()
        }
        camera_bundle = CameraBundle(
            image_names=image_names,
            intrinsics=np.stack(intrinsics).astype(np.float32),
            extrinsics=extrinsic_array.astype(np.float32),
            source="da3-prior-shape-hypothesis-v1",
            scale_status="known",
            world_units="millimetre",
            world_units_to_mm=1.0,
            details={"coverage_subsets": COVERAGE_SUBSETS, "held_out_indices": HELD_OUT_INDICES},
        )
        camera_bundle.save(case_root / "cameras.npz")
        record: dict[str, object] = {
            "id": spec.case_id,
            "title": spec.title,
            "group": case.group,
            "axial": case.axial,
            "operation_family": spec.family,
            "parameters_mm": spec.parameters_mm,
            "coverage_subsets": {key: list(value) for key, value in COVERAGE_SUBSETS.items()},
            "coverage_fractions": coverages,
            "held_out_indices": list(HELD_OUT_INDICES),
            "oracle_patch_masks": "Stage 2 upper-bound only; not used by Stage 1",
        }
        (case_root / "manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(record)
    manifest: dict[str, object] = {
        "schema_version": "da3-prior-shape-hypothesis-v1",
        "preregistration": "docs/experiments/DA3_PRIOR_SHAPE_HYPOTHESIS.md",
        "cases": records,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
