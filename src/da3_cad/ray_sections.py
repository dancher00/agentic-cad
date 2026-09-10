"""Ray-evidence section compiler (experimental, no learned weights required).

Coordinates stay in the supplied camera frame. Depth is camera-z, not ray length.
Free-space carving does not label space behind a first hit as empty.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cadquery as cq
import numpy as np
from scipy import ndimage

from da3_cad.models import FloatArray


@dataclass(frozen=True)
class RayBundle:
    masks: Any
    depths: Any
    intrinsics: Any
    extrinsics: Any
    names: tuple[str, ...]

    def validate(self) -> None:
        n = len(self.names)
        if n < 3 or self.masks.ndim != 3 or self.masks.shape[0] != n:
            raise ValueError("require >=3 equally sized calibrated mask/depth images")
        if self.depths.shape != self.masks.shape:
            raise ValueError("depth and mask shapes differ")
        if self.intrinsics.shape != (n, 3, 3) or self.extrinsics.shape != (n, 4, 4):
            raise ValueError("invalid camera array shape")
        if len(set(self.names)) != n:
            raise ValueError("image names must be unique")
        if not np.isfinite(self.intrinsics).all() or not np.isfinite(self.extrinsics).all():
            raise ValueError("nonfinite cameras")
        if np.any(self.intrinsics[:, (0, 1), (0, 1)] <= 0):
            raise ValueError("focal lengths must be positive")
        rotation = self.extrinsics[:, :3, :3]
        if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3):
            raise ValueError("camera rotations must be orthonormal")
        if not np.allclose(np.linalg.det(rotation), 1, atol=1e-3):
            raise ValueError("camera rotations must be right-handed")

    @classmethod
    def load(cls, path: Path) -> RayBundle:
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                np.asarray(data["masks"], dtype=bool),
                np.asarray(data["depths"], dtype=np.float32),
                np.asarray(data["intrinsics"], dtype=np.float64),
                np.asarray(data["extrinsics"], dtype=np.float64),
                tuple(str(x) for x in data["names"]),
            )
        result.validate()
        return result

    def subset(self, indices: Any) -> RayBundle:
        return RayBundle(
            self.masks[indices],
            self.depths[indices],
            self.intrinsics[indices],
            self.extrinsics[indices],
            tuple(self.names[int(i)] for i in indices),
        )

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            masks=self.masks,
            depths=self.depths,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            names=np.asarray(self.names),
        )


def bundle_from_mvs(workspace: Path, cameras: Path) -> RayBundle:
    """Read existing calibrated PatchMatch outputs without accessing reference CAD."""
    from PIL import Image

    from da3_cad.geometry.cameras import load_camera_bundle
    from da3_cad.integrations.depth_fusion import _read_colmap_array
    from da3_cad.observations import load_observations

    observations = load_observations(workspace / "images")
    camera = load_camera_bundle(cameras, observations)
    masks, depths = [], []
    for name in camera.image_names:
        depths.append(_read_colmap_array(workspace / "stereo/depth_maps" / f"{name}.geometric.bin"))
        with Image.open(workspace / "masks" / name) as image:
            masks.append(np.asarray(image.convert("L")) >= 128)
    bundle = RayBundle(
        np.stack(masks), np.stack(depths), camera.intrinsics, camera.extrinsics, camera.image_names
    )
    bundle.validate()
    return bundle


def observed_points(bundle: RayBundle, maximum: int = 30000) -> FloatArray:
    points = []
    for mask, depth, k, e in zip(
        bundle.masks, bundle.depths, bundle.intrinsics, bundle.extrinsics, strict=True
    ):
        y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
        stride = max(1, len(y) // max(1, maximum // len(bundle.names)))
        y, x = y[::stride], x[::stride]
        z = depth[y, x]
        camera = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
        points.append((camera - e[:3, 3]) @ e[:3, :3])
    values = np.concatenate(points)
    if len(values) < 32:
        raise ValueError("insufficient measured depth for input-only bounds/frame estimation")
    return np.asarray(values, dtype=np.float64)


def evidence_frames(points: FloatArray) -> list[tuple[str, FloatArray]]:
    """Global, PCA and dominant-plane frames, estimated without reference geometry."""
    centered = points - np.median(points, axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    pca = vectors[:, ::-1].copy()
    pca[:, 2] = np.cross(pca[:, 0], pca[:, 1])
    frames = [("world", np.eye(3)), ("pca", pca)]
    # Dominant planes preserve extrusion axes when asymmetric profiles bias PCA.
    rng = np.random.default_rng(0)
    sample = centered[rng.choice(len(points), min(len(points), 6000), replace=False)]
    threshold = 0.008 * np.ptp(sample, axis=0).max()
    triples = sample[rng.integers(0, len(sample), (192, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    distances = np.abs(sample @ normals.T - np.sum(triples[:, 0] * normals, axis=1))
    counts = (distances < threshold).sum(axis=0)
    first = int(np.argmax(counts))
    normal = normals[first]
    inliers = sample[distances[:, first] < threshold]
    if len(inliers) >= 10:
        _, vec = np.linalg.eigh(np.cov(inliers.T))
        normal = vec[:, 0]
    perpendicular = np.abs(normals @ normal) < 0.15
    if perpendicular.any():
        second = int(np.argmax(np.where(perpendicular, counts, -1)))
        inliers = sample[distances[:, second] < threshold]
        _, vec = np.linalg.eigh(np.cov(inliers.T))
        x = vec[:, 0] - normal * np.dot(vec[:, 0], normal)
        x /= np.linalg.norm(x)
        frames.append(("planes", np.column_stack((x, np.cross(normal, x), normal))))
    return frames


def carve(
    bundle: RayBundle,
    points: FloatArray,
    frame: FloatArray,
    *,
    resolution: int,
    device: str,
    use_depth: bool = True,
) -> tuple[Any, FloatArray, float, dict[str, Any]]:
    import torch

    local = points @ frame
    low, high = np.quantile(local, [0.002, 0.998], axis=0)
    step = float(np.max(high - low) / (resolution - 6))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("degenerate input extent")
    center = (low + high) / 2
    lower = center - resolution * step / 2
    coords = torch.arange(resolution, device=device, dtype=torch.float32) + 0.5
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), -1).reshape(-1, 3)
    world = (
        grid * step + torch.as_tensor(lower, device=device, dtype=torch.float32)
    ) @ torch.as_tensor(frame.T, device=device, dtype=torch.float32)
    count = len(world)
    visible = torch.zeros(count, device=device)
    outside = torch.zeros_like(visible)
    free = torch.zeros_like(visible)
    depth_votes = torch.zeros_like(visible)
    for mask, depth, k, e in zip(
        bundle.masks, bundle.depths, bundle.intrinsics, bundle.extrinsics, strict=True
    ):
        h, w = mask.shape
        rotation = torch.as_tensor(e[:3, :3], device=device, dtype=torch.float32)
        camera = world @ rotation.T + torch.as_tensor(e[:3, 3], device=device, dtype=torch.float32)
        z = camera[:, 2]
        x = torch.round(k[0, 0] * camera[:, 0] / z.clamp_min(1e-8) + k[0, 2]).long()
        y = torch.round(k[1, 1] * camera[:, 1] / z.clamp_min(1e-8) + k[1, 2]).long()
        valid = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x.clamp(0, w - 1), y.clamp(0, h - 1)
        # One-pixel tolerance for discretized silhouettes; frozen across methods.
        tolerant_mask = ndimage.binary_dilation(mask, iterations=1)
        foreground = torch.as_tensor(tolerant_mask, device=device)[y, x]
        visible += valid
        outside += valid & ~foreground
        if use_depth:
            measured = torch.as_tensor(depth, device=device)[y, x]
            reliable = valid & foreground & torch.isfinite(measured) & (measured > 0)
            depth_votes += reliable
            # Depth is z: convert one world voxel to a conservative z tolerance.
            free += reliable & (z < measured - 1.5 * step)
    occupancy = (visible >= max(3, len(bundle.names) // 2)) & (outside <= visible * 0.05)
    occupancy &= ~((free >= 2) & (free > depth_votes * 0.15))
    volume = occupancy.reshape((resolution,) * 3).cpu().numpy()
    components, number = ndimage.label(volume)
    sizes = np.bincount(components.ravel())
    if number == 0:
        raise ValueError("ray constraints leave no occupied completion")
    sizes[0] = 0
    volume = components == int(sizes.argmax())
    return (
        volume,
        np.asarray(lower),
        step,
        {
            "occupied_voxels": int(volume.sum()),
            "components_before_filter": number,
            "discarded_voxels": int(occupancy.sum().item() - volume.sum()),
            "unknown_policy": "space behind first hits remains a completion hypothesis",
        },
    )


def partition(
    volume: Any,
    axis: int,
    *,
    maximum_sections: int = 8,
    penalty: float = 0.03,
    device: str = "cpu",
    mode: str = "adaptive",
) -> tuple[list[tuple[int, int, Any]], float]:
    """Exact DP for interval majority profiles under voxel XOR + section cost.

    Exactness applies to the discrete, unsimplified interval objective only.
    Polygon simplification and CAD-kernel execution are separate later stages.
    """
    import torch

    order = (axis, (axis + 1) % 3, (axis + 2) % 3)
    data = np.transpose(volume, order)
    nonempty = np.flatnonzero(data.any(axis=(1, 2)))
    if not len(nonempty):
        raise ValueError("empty volume")
    start, end = int(nonempty[0]), int(nonempty[-1]) + 1
    values = torch.as_tensor(data[start:end], device=device, dtype=torch.float32)
    length = len(values)
    cumulative = torch.cat((torch.zeros_like(values[:1]), values.cumsum(0)))
    costs = np.full((length + 1, length + 1), np.inf)
    denominator = max(1, int(volume.sum()))
    for i in range(length):
        counts = cumulative[i + 1 :] - cumulative[i]
        widths = torch.arange(1, length - i + 1, device=device)[:, None, None]
        errors = torch.minimum(counts, widths - counts).sum((1, 2)) / denominator
        costs[i, i + 1 :] = errors.cpu().numpy()
    if mode == "single":
        boundaries = [0, length]
    elif mode == "uniform":
        boundaries = (
            np.unique(np.rint(np.linspace(0, length, min(maximum_sections, length) + 1)))
            .astype(int)
            .tolist()
        )
    elif mode == "adaptive":
        dp = np.full((maximum_sections + 1, length + 1), np.inf)
        parent = np.zeros_like(dp, dtype=int)
        dp[0, 0] = 0
        for k in range(1, maximum_sections + 1):
            for j in range(1, length + 1):
                candidates = dp[k - 1, :j] + costs[:j, j] + penalty
                parent[k, j] = int(np.argmin(candidates))
                dp[k, j] = candidates[parent[k, j]]
        k = int(np.argmin(dp[:, length]))
        j = length
        boundaries = [j]
        while k:
            j = int(parent[k, j])
            boundaries.append(j)
            k -= 1
        boundaries.reverse()
    else:
        raise ValueError(f"unknown partition mode: {mode}")
    sections = []
    cost = 0.0
    for i, j in zip(boundaries[:-1], boundaries[1:], strict=True):
        profile = data[start + i : start + j].mean(0) >= 0.5
        if profile.any():
            sections.append((start + i, start + j, profile))
        cost += float(costs[i, j]) + penalty
    return sections, cost


def profile_loops(profile: Any, tolerance: float = 0.65) -> list[list[Any]]:
    import cv2

    contours, hierarchy = cv2.findContours(
        profile.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    groups = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] != -1 or cv2.contourArea(contour) < 3:
            continue
        loops = []
        indices = [i] + [j for j in range(len(contours)) if hierarchy[0, j, 3] == i]
        for j in indices:
            approximate = cv2.approxPolyDP(contours[j], tolerance, True)[:, 0]
            if len(approximate) >= 3:
                # OpenCV reports (column,row); the section axes are (row,column).
                loops.append(approximate[:, ::-1].astype(float) + 0.5)
        if loops:
            groups.append(loops)
    return groups


def compile_sections(
    sections: list[tuple[int, int, Any]],
    axis: int,
    frame: FloatArray,
    lower: FloatArray,
    step: float,
) -> tuple[cq.Workplane, str, int]:
    u = (axis + 1) % 3
    program = [
        "import cadquery as cq",
        "# Coordinates use the supplied camera world frame.",
        "# Inferred completion; no claim about hidden topology or physical units.",
        "SECTIONS = []",
    ]
    solids = []
    for section_index, (start, end, profile) in enumerate(sections):
        for groups in profile_loops(profile):
            origin = lower.copy()
            overlap = 0.001 * step if section_index else 0.0
            origin[axis] += start * step - overlap
            world_origin = frame @ origin
            loops = [[(float(p[0] * step), float(p[1] * step)) for p in loop] for loop in groups]
            spec = {
                "origin": world_origin.tolist(),
                "x_direction": frame[:, u].tolist(),
                "normal": frame[:, axis].tolist(),
                "height": (end - start) * step + overlap,
                "loops": loops,
            }
            program.append(f"SECTIONS.append({spec!r})")
            plane = cq.Plane(
                origin=tuple(world_origin), xDir=tuple(frame[:, u]), normal=tuple(frame[:, axis])
            )
            work = cq.Workplane(plane)
            for loop in loops:
                work = work.polyline(loop).close()
            solids.append(work.extrude((end - start) * step + overlap))
    if not solids:
        raise ValueError("no nondegenerate sketch loops")
    result = solids[0]
    for solid in solids[1:]:
        result = result.union(solid)
    shape = result.val()
    if not isinstance(shape, cq.Shape) or not shape.isValid() or len(shape.Solids()) != 1:
        raise ValueError("section union is not one kernel-valid solid")
    if shape.Volume() <= 0:
        raise ValueError("section solid has nonpositive volume")
    program.extend(
        [
            "parts = []",
            "for section in SECTIONS:",
            "    plane = cq.Plane(origin=section['origin'],",
            "                     xDir=section['x_direction'], normal=section['normal'])",
            "    work = cq.Workplane(plane)",
            "    for loop in section['loops']:",
            "        work = work.polyline(loop).close()",
            "    parts.append(work.extrude(section['height']))",
            "r = parts[0]",
            "for part in parts[1:]:",
            "    r = r.union(part)",
            "",
        ]
    )
    return result, "\n".join(program), len(solids)


def _compile_worker(payload: Any, output: str, connection: Any) -> None:
    """Isolate CAD-kernel hangs; only trusted numeric section data enters here."""
    try:
        from da3_cad.evaluation.mesh import tessellate_step

        result, source, operations = compile_sections(*payload)
        directory = Path(output)
        cq.exporters.export(result, str(directory / "candidate.step"))
        mesh = tessellate_step(directory / "candidate.step")
        if not mesh.is_volume:
            raise ValueError(
                "kernel-valid solid did not tessellate to a watertight oriented volume"
            )
        mesh.export(directory / "candidate.stl")
        (directory / "candidate.py").write_text(source, encoding="utf-8")
        shape = result.val()
        assert isinstance(shape, cq.Shape)
        connection.send(
            {
                "extrusions": operations,
                "faces": len(shape.Faces()),
                "edges": len(shape.Edges()),
                "volume": shape.Volume(),
            }
        )
    except Exception as error:
        connection.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _bounded_compile(payload: Any, output: Path, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    with TemporaryDirectory(prefix="compile-", dir=output) as temporary:
        process = context.Process(target=_compile_worker, args=(payload, temporary, send))
        process.start()
        send.close()
        process.join(timeout)
        if process.is_alive():
            process.kill()
            process.join()
            receive.close()
            raise TimeoutError(f"CAD compilation exceeded {timeout:g} seconds")
        try:
            result = (
                receive.recv() if receive.poll() else {"error": f"worker exit {process.exitcode}"}
            )
        except EOFError:
            result = {"error": f"worker exited without result: {process.exitcode}"}
        finally:
            receive.close()
        if "error" in result:
            raise ValueError(result["error"])
        for name in ("candidate.step", "candidate.stl", "candidate.py"):
            shutil.copyfile(Path(temporary) / name, output / name)
        return dict(result)


def reconstruct_sections(
    bundle: RayBundle,
    output: Path,
    *,
    resolution: int = 72,
    maximum_sections: int = 8,
    penalty: float = 0.03,
    device: str = "cuda",
    mode: str = "adaptive",
    use_depth: bool = True,
    frame_modes: tuple[str, ...] = ("world", "pca", "planes"),
    compile_timeout: float = 20.0,
) -> dict[str, Any]:
    import torch

    bundle.validate()
    if resolution < 16 or resolution > 256 or maximum_sections < 1 or maximum_sections > 32:
        raise ValueError("resolution must be 16..256 and maximum_sections 1..32")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    points = observed_points(bundle)
    candidates = []
    frames_report = []
    for name, frame in evidence_frames(points):
        if name not in frame_modes:
            continue
        volume, lower, step, carving = carve(
            bundle, points, frame, resolution=resolution, device=device, use_depth=use_depth
        )
        frames_report.append({"frame": name, **carving})
        for axis in range(3):
            sections, objective = partition(
                volume,
                axis,
                maximum_sections=maximum_sections,
                penalty=penalty,
                device=device,
                mode=mode,
            )
            candidates.append((objective, name, axis, sections, frame, lower, step))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    failures = []
    for objective, name, axis, sections, frame, lower, step in candidates:
        try:
            compiled = _bounded_compile(
                (sections, axis, frame, lower, step), output, compile_timeout
            )
        except Exception as error:
            failures.append({"frame": name, "axis": axis, "error": str(error)})
            continue
        report = {
            "schema": "ray-section-cad-v1",
            "status": "CANDIDATE",
            "kernel_valid": True,
            "decision_policy": "requires independent source-view assessment; not an ACCEPT",
            "reference_access": False,
            "views": list(bundle.names),
            "device": device,
            "resolution": resolution,
            "maximum_sections": maximum_sections,
            "penalty": penalty,
            "mode": mode,
            "use_depth": use_depth,
            "frame": name,
            "axis": axis,
            "frame_matrix": frame.tolist(),
            "lower": lower.tolist(),
            "voxel_size": step,
            "objective": objective,
            "sections": len(sections),
            **compiled,
            "compile_timeout_seconds": compile_timeout,
            "elapsed_seconds": time.perf_counter() - started,
            "failed_compilations": failures,
            "carving": frames_report,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "step_sha256": hashlib.sha256((output / "candidate.step").read_bytes()).hexdigest(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    report = {
        "status": "ABSTAIN",
        "kernel_valid": False,
        "failed_compilations": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
