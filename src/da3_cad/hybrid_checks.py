"""Mesh section and local material-chord diagnostics for generated CAD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def material_chords(mesh: Any, count: int = 96) -> dict[str, Any]:
    """Sample inward normal chords, not certified minimum wall thickness."""
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=0)
    directions = -mesh.face_normals[faces]
    epsilon = max(float(np.max(mesh.extents)) * 1e-7, 1e-8)
    triangles = np.asarray(mesh.triangles)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    lengths = []
    for origin, direction in zip(points, directions, strict=True):
        origin = origin + direction * epsilon
        h = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        usable = np.abs(determinant) > 1e-12
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
        offset = origin - triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", offset, h)
        q = np.cross(offset, edge1)
        v = inverse * (q @ direction)
        distance = inverse * np.einsum("ij,ij->i", edge2, q)
        hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > epsilon)
        if np.any(hit):
            lengths.append(float(distance[hit].min() + epsilon))
    return {
        "method": "area-weighted inward normal chords on exported STL; seed 0",
        "requested_samples": count,
        "hit_samples": len(lengths),
        "p10_p50_p90_mm": list(map(float, np.percentile(lengths, [10, 50, 90])))
        if lengths
        else None,
        "certified_minimum_wall_thickness": False,
        "scope": "Includes feet, ribs and handles; chords are not uniform wall measurements",
    }


def section_diagnostics(mesh_path: Path, output: Path) -> dict[str, Any]:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    report = material_chords(mesh)
    panels = []
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    for name, normal, axes in [("xz", [0, 1, 0], [0, 2]), ("yz", [1, 0, 0], [1, 2])]:
        section = mesh.section(plane_normal=normal, plane_origin=center)
        panel = Image.new("RGB", (480, 480), "#f5f5f5")
        draw = ImageDraw.Draw(panel)
        draw.text((16, 12), f"{name.upper()} section / mm", fill="#333333")
        if section is not None:
            bounds = mesh.bounds[:, axes]
            scale = 420 / max(float(np.max(bounds[1] - bounds[0])), 1e-8)
            midpoint = (bounds[0] + bounds[1]) / 2
            for polyline in section.discrete:
                xy = (polyline[:, axes] - midpoint) * scale
                xy[:, 1] *= -1
                xy += [240, 250]
                draw.line([tuple(point) for point in xy], fill="#333333", width=2)
        panels.append(panel)
    combined = Image.new("RGB", (960, 480), "#f5f5f5")
    for i, panel in enumerate(panels):
        combined.paste(panel, (480 * i, 0))
    combined.save(output / "sections.png")
    (output / "material-chords.json").write_text(json.dumps(report, indent=2))
    return report


def check_overall_height(candidate: Any, validation: Any) -> None:
    """Hybrid CAD is Z-up; enforce explicitly named overall height parameters."""
    if not validation.valid or validation.bbox is None:
        return
    actual = validation.bbox[5] - validation.bbox[2]
    for parameter in candidate.parameters:
        if (
            parameter.source == "specified"
            and parameter.unit == "mm"
            and parameter.name.lower() in {"target_height", "overall_height", "total_height"}
        ):
            tolerance = max(0.01, abs(parameter.value) * 0.001)
            if abs(actual - parameter.value) > tolerance:
                raise ValueError(
                    f"Specified {parameter.name}={parameter.value:g} mm, "
                    f"but actual Z extent is {actual:g} mm (tolerance {tolerance:g} mm)"
                )


def check_profile_order(reference: Any, candidate: Any) -> None:
    """Keep radial body-profile ordering during local parameter fitting.

    These are design-preservation constraints, not measurements from photographs.
    GPT may propose a new profile; a scalar fit must not silently invert that profile.
    """
    values = {p.name: p.value for p in candidate.parameters}
    for suffix in ("diameter", "radius"):
        profile = [
            p
            for p in reference.parameters
            if p.unit == "mm"
            and "body" in p.name.lower().split("_")
            and suffix in p.name.lower().split("_")
        ]
        for i, left in enumerate(profile):
            for right in profile[i + 1 :]:
                difference = left.value - right.value
                updated = values[left.name] - values[right.name]
                tolerance = 1e-7 * max(abs(left.value), abs(right.value), 1)
                if (
                    abs(difference) <= tolerance
                    and abs(updated) > tolerance
                    or difference * updated < -tolerance
                ):
                    raise ValueError(f"Fit would reverse body profile: {left.name}, {right.name}")


def observation_target_met(score: dict[str, Any], threshold: float) -> bool:
    views = score.get("view_ious", [score["mean_silhouette_iou"]])
    return bool(views) and min(views) >= threshold


def fit_improves(reference: dict[str, Any], trial: dict[str, Any]) -> bool:
    """Require an improvement without trading away another view or the depth prior."""
    return (
        trial["loss"] < reference["loss"] - 0.001
        and trial["mean_silhouette_iou"] > reference["mean_silhouette_iou"] + 0.001
        and trial["relative_depth_surface_residual"]
        <= reference["relative_depth_surface_residual"] + 0.01
        and all(
            after >= before - 0.002
            for before, after in zip(reference["view_ious"], trial["view_ious"], strict=True)
        )
    )


def candidate_rank(score: dict[str, Any], threshold: float) -> tuple[int, float]:
    """Prefer a candidate meeting the target in every view, then minimize loss."""
    return (0 if observation_target_met(score, threshold) else 1, float(score["loss"]))
