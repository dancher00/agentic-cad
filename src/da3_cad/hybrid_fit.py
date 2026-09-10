"""Shared CAD-to-camera registration and bounded editable-parameter fitting.

DA3 scale is a nuisance variable, never a source of measured CAD dimensions.
The objective measures consistency with observations, not ground-truth accuracy.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def silhouette_iou(left: Any, right: Any) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def rasterize(
    vertices: Any, faces: Any, intrinsics: Any, extrinsics: Any, shape: tuple[int, int]
) -> Any:
    import cv2

    camera = vertices @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    pixels = camera @ intrinsics.T
    xy = pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-8)
    mask = np.zeros(shape, dtype=np.uint8)
    visible = np.all(camera[faces, 2] > 1e-7, axis=1)
    triangles = np.clip(xy[faces[visible]], -10000, 10000).astype(np.int32)
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, (1,))
    return mask.astype(bool)


class GeometryObjective:
    def __init__(self, evidence: Path, resolution: int = 96):
        import cv2

        with np.load(evidence / "geometry.npz", allow_pickle=False) as data:
            self.extrinsics = data["extrinsics"].copy()
            self.intrinsics = data["intrinsics"].copy()
            depth, masks = data["depth"], data["masks"]
            clouds = []
            targets = []
            for i, mask in enumerate(masks):
                h, w = mask.shape
                ratio = resolution / max(h, w)
                size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
                targets.append(
                    cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(
                        bool
                    )
                )
                ys, xs = np.where(mask & np.isfinite(depth[i]) & (depth[i] > 0))
                if not len(xs):
                    raise ValueError("No finite object depth")
                selection = np.linspace(0, len(xs) - 1, min(len(xs), 800)).astype(int)
                xs, ys = xs[selection], ys[selection]
                rays = (
                    np.column_stack([xs, ys, np.ones(len(xs))])
                    @ np.linalg.inv(self.intrinsics[i]).T
                )
                points = rays * depth[i, ys, xs, None]
                ext = self.extrinsics[i]
                clouds.append((points - ext[:3, 3]) @ ext[:3, :3])
                self.intrinsics[i, 0] *= size[0] / w
                self.intrinsics[i, 1] *= size[1] / h
        self.targets = targets
        self.cloud = np.concatenate(clouds)
        self.center = np.median(self.cloud, axis=0)
        distances = np.linalg.norm(self.cloud - self.center, axis=1)
        keep = distances <= np.percentile(distances, 98)
        self.cloud = self.cloud[keep]
        self.radius = max(float(np.percentile(distances[keep], 90)), 1e-8)

    def load_mesh(self, path: Path, normalization: tuple[Any, float] | None = None) -> None:
        mesh = trimesh.load_mesh(path, process=True)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Expected a nonempty triangle mesh")
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces)
        self.mesh_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        self.mesh_radius = max(float(np.max(mesh.extents)) / 2, 1e-8)
        if normalization is not None:
            self.mesh_center, self.mesh_radius = normalization
        # Deterministic area-weighted samples avoid tessellation-density bias.
        points, _ = trimesh.sample.sample_surface(mesh, 4000, seed=0)
        self.surface_tree = cKDTree((points - self.mesh_center) / self.mesh_radius)

    def world_vertices(self, pose: Any) -> Any:
        rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
        scale = np.exp(pose[3]) * self.radius / self.mesh_radius
        return (self.vertices - self.mesh_center) @ rotation.T * scale + (
            self.center + np.asarray(pose[4:7]) * self.radius
        )

    def evaluate(self, pose: Any, *, panels: Path | None = None) -> dict[str, Any]:
        world = self.world_vertices(pose)
        ious = []
        for i, (target, intrinsics, extrinsics) in enumerate(
            zip(self.targets, self.intrinsics, self.extrinsics, strict=True)
        ):
            predicted = rasterize(world, self.faces, intrinsics, extrinsics, target.shape)
            ious.append(silhouette_iou(predicted, target))
            if panels is not None:
                # Gray overlap, blue missing material, red excess material.
                rgb = np.full((*target.shape, 3), 245, dtype=np.uint8)
                rgb[target & predicted] = [100, 100, 100]
                rgb[target & ~predicted] = [55, 115, 205]
                rgb[~target & predicted] = [210, 75, 65]
                Image.fromarray(rgb).resize((rgb.shape[1] * 4, rgb.shape[0] * 4)).save(
                    panels / f"comparison-{i:02d}.png"
                )
        rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
        local = ((self.cloud - self.center) / self.radius - pose[4:7]) @ rotation / np.exp(pose[3])
        distances = self.surface_tree.query(local)[0]
        residual = float(np.mean(np.minimum(distances, 0.5)))
        mean_iou = float(np.mean(ious))
        return {
            "loss": 1 - mean_iou + 0.1 * residual,
            "mean_silhouette_iou": mean_iou,
            "view_ious": ious,
            "relative_depth_surface_residual": residual,
            "pose": list(map(float, pose)),
            "contract": "shared similarity in DA3 camera frame; relative scale; depth prior",
        }

    def register(self, seed: Any = None) -> dict[str, Any]:
        if seed is not None:
            origin = np.asarray(seed, dtype=float)
            if origin.shape == (7,) and np.isfinite(origin).all():
                best = self.evaluate(origin)

                def loss(pose: Any) -> float:
                    nonlocal best
                    trial = self.evaluate(pose)
                    if trial["loss"] < best["loss"]:
                        best = trial
                    return float(trial["loss"])

                # Reuse an already strong alignment of the same views. All scoring
                # still uses the full exported mesh; poor seeds fall back to search.
                if min(best["view_ious"]) >= 0.95:
                    delta = np.asarray([0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
                    minimize(
                        loss,
                        origin,
                        method="Powell",
                        bounds=list(zip(origin - delta, origin + delta, strict=True)),
                        options={"maxfev": 80, "xtol": 0.005, "ftol": 0.001},
                    )
                    if min(best["view_ious"]) >= 0.95:
                        return {**best, "registration_method": "warm-start"}
        return {**self._register_global(), "registration_method": "global"}

    def _register_global(self) -> dict[str, Any]:
        _, _, basis = np.linalg.svd(self.cloud - self.center, full_matrices=False)
        rotations = []
        for order in itertools.permutations(range(3)):
            for signs in itertools.product([-1, 1], repeat=3):
                matrix = basis.T[:, order] * signs
                if np.linalg.det(matrix) > 0:
                    rotations.append(Rotation.from_matrix(matrix).as_rotvec())
        pca_rotations = rotations.copy()
        # Z-up CAD and an upright first photo give useful hypotheses even when
        # one-sided predicted depth has a misleading PCA (notably mug handles).
        camera_basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        for yaw in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for pitch in (-0.5, 0.0, 0.5):
                camera_rotation = (
                    Rotation.from_rotvec([pitch, 0, 0]).as_matrix()
                    @ camera_basis
                    @ Rotation.from_rotvec([0, 0, yaw]).as_matrix()
                )
                world_rotation = self.extrinsics[0, :3, :3].T @ camera_rotation
                rotations.append(Rotation.from_matrix(world_rotation).as_rotvec())
        starts = [np.r_[rotation, 0.0, 0.0, 0.0, 0.0] for rotation in rotations]
        forward = self.extrinsics[0, :3, :3].T @ np.array([0.0, 0.0, 0.35])
        starts += [np.r_[rotation, 0.0, forward] for rotation in rotations]
        ranked = sorted((self.evaluate(p) for p in starts), key=lambda item: item["loss"])
        best = ranked[0]

        # Track every evaluated pose: rasterized losses have plateaus and Powell's
        # returned point can be worse than an intermediate evaluation.
        def loss(pose: Any) -> float:
            nonlocal best
            trial = self.evaluate(pose)
            if trial["loss"] < best["loss"]:
                best = trial
            return float(trial["loss"])

        diverse: list[dict[str, Any]] = []
        for start in ranked:
            rotation = Rotation.from_rotvec(start["pose"][:3])
            if all(
                (rotation.inv() * Rotation.from_rotvec(other["pose"][:3])).magnitude() > 0.6
                for other in diverse
            ):
                diverse.append(start)
            if len(diverse) == 6:
                break
        # Keep the original PCA search basins as well. Adding camera hypotheses
        # must not evict a previously useful local optimization start.
        pca_ranked = sorted(
            (self.evaluate(np.r_[rotation, 0.0, 0.0, 0.0, 0.0]) for rotation in pca_rotations),
            key=lambda item: item["loss"],
        )
        starts_to_refine = pca_ranked[:3] + diverse
        for start in starts_to_refine:
            origin = np.asarray(start["pose"])
            delta = np.asarray([0.6, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6])
            minimize(
                loss,
                origin,
                method="Powell",
                bounds=list(zip(origin - delta, origin + delta, strict=True)),
                options={"maxfev": 280, "xtol": 0.005, "ftol": 0.001},
            )
        return best


def fit_candidate(
    candidate: Any,
    attempt_dir: Path,
    evidence: Path,
    settings: Any,
    max_parameters: int,
    registration_seed: Any = None,
) -> tuple[Any, Any, dict[str, Any]]:
    from da3_cad.cad.sandbox import validate_and_export
    from da3_cad.gpt_cad import parameterize

    objective = GeometryObjective(evidence)
    objective.load_mesh(attempt_dir / "model.stl")
    normalization = (objective.mesh_center.copy(), objective.mesh_radius)
    initial = (
        objective.register(registration_seed)
        if registration_seed is not None
        else objective.register()
    )
    verification = GeometryObjective(evidence, resolution=192)
    verification.load_mesh(attempt_dir / "model.stl", normalization)
    verified_initial = verification.evaluate(initial["pose"])
    best = candidate
    best_score = initial
    best_verified = verified_initial
    chosen_validation = None
    trials: list[dict[str, Any]] = []
    eligible = [
        p
        for p in candidate.parameters
        if p.source == "estimated"
        and p.unit == "mm"
        and p.value > 0
        and not any(
            word in p.name.lower() for word in ("thickness", "clearance", "tolerance", "wall")
        )
    ][:max_parameters]
    for parameter in eligible:
        anchor = next(p.value for p in best.parameters if p.name == parameter.name)
        for factor in (0.92, 1.08):
            import ast

            value = anchor * factor
            tree = ast.parse(best.code)
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id == parameter.name:
                        node.value = ast.Constant(value=value)
            params = [
                p.model_copy(update={"value": value}) if p.name == parameter.name else p
                for p in best.parameters
            ]
            trial = best.model_copy(
                update={"code": ast.unparse(ast.fix_missing_locations(tree)), "parameters": params}
            )
            trial_dir = attempt_dir / "fit" / f"{len(trials):02d}"
            trial_dir.mkdir(parents=True)
            from da3_cad.hybrid_checks import check_profile_order, fit_improves

            try:
                check_profile_order(candidate, trial)
            except ValueError as error:
                trials.append(
                    {
                        "parameter": parameter.name,
                        "value": value,
                        "valid": False,
                        "accepted": False,
                        "reason": str(error),
                    }
                )
                continue
            source = parameterize(trial)
            validation = validate_and_export(source, trial_dir, settings.sandbox)
            row: dict[str, Any] = {
                "parameter": parameter.name,
                "value": value,
                "valid": validation.valid,
                "accepted": False,
            }
            if validation.valid:
                from da3_cad.hybrid_checks import check_overall_height

                try:
                    check_overall_height(trial, validation)
                except ValueError as error:
                    row.update(valid=False, error=str(error))
                    trials.append(row)
                    continue
                objective.load_mesh(trial_dir / "model.stl", normalization)
                # Freeze CAD normalization as well as pose: otherwise changing a sole
                # diameter can be hidden by renormalizing the registration scale.
                score = objective.evaluate(best_score["pose"])
                row.update(score)
                if fit_improves(best_score, score):
                    verification.load_mesh(trial_dir / "model.stl", normalization)
                    verified = verification.evaluate(best_score["pose"])
                    row["verification"] = verified
                    if not fit_improves(best_verified, verified):
                        row["reason"] = "Improvement did not survive 192-pixel verification"
                        trials.append(row)
                        continue
                    best_verified = verified
                    best, best_score, chosen_validation = trial, score, validation
                    row["accepted"] = True
                    import shutil

                    for filename in ("model.step", "model.stl"):
                        shutil.copy2(trial_dir / filename, attempt_dir / filename)
                    (attempt_dir / "model.py").write_text(source)
            trials.append(row)
    objective.load_mesh(attempt_dir / "model.stl", normalization)
    final = objective.evaluate(best_score["pose"])
    verification.load_mesh(attempt_dir / "model.stl", normalization)
    verified_final = verification.evaluate(best_score["pose"], panels=attempt_dir)
    from da3_cad.hybrid_checks import section_diagnostics

    chords = section_diagnostics(attempt_dir / "model.stl", attempt_dir)
    report = {
        "material_chords": chords,
        "after": verified_final,
        "before": verified_initial,
        "search_before": initial,
        "search_after": final,
        "verification_resolution": 192,
        "registration_method": initial.get("registration_method", "global"),
        "trials": trials,
        "specified_dimensions_modified": False,
        "limitation": "Observation consistency only; hidden geometry and material are not measured",
    }
    (attempt_dir / "fitted-response.json").write_text(best.model_dump_json(indent=2))
    (attempt_dir / "geometry-review.json").write_text(json.dumps(report, indent=2))
    return best, chosen_validation, report
