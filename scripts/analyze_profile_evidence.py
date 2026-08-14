"""Compare profile-evidence channels against evaluator-only reference CAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import trimesh
from scipy import ndimage
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _largest(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=np.bool_)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _solid_occupancy(
    points: np.ndarray,
    *,
    axis: int,
    minimum: np.ndarray,
    span: np.ndarray,
    resolution: int,
) -> np.ndarray:
    transverse = [index for index in range(3) if index != axis]
    values = points[:, transverse]
    keep = np.all((values >= minimum) & (values <= minimum + span), axis=1)
    pixels = np.floor((values[keep] - minimum) / span * (resolution - 1)).astype(np.int32)
    pixels = np.clip(pixels, 0, resolution - 1)
    raster = np.zeros((resolution, resolution), dtype=np.bool_)
    raster[pixels[:, 1], pixels[:, 0]] = True
    raster = ndimage.binary_dilation(raster, iterations=1)
    raster = ndimage.binary_closing(raster, iterations=2)
    return ndimage.binary_fill_holes(_largest(raster)).astype(np.bool_)


def _mesh_profile(
    mesh_path: Path,
    *,
    center_world: np.ndarray,
    axes_world_columns: np.ndarray,
    midpoint: np.ndarray,
    extent: float,
    axis: int,
    minimum: np.ndarray,
    span: np.ndarray,
    resolution: int,
    frame_canonical_columns: np.ndarray,
) -> np.ndarray:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    world = np.asarray(mesh.vertices, dtype=np.float64)
    oriented = (world - center_world) @ axes_world_columns
    normalized = 2.0 * (oriented - midpoint) / extent
    normalized = normalized @ frame_canonical_columns
    transverse = [index for index in range(3) if index != axis]
    coordinates = normalized[:, transverse]
    pixels = np.rint((coordinates - minimum) / span * (resolution - 1)).astype(np.int32)
    raster = np.zeros((resolution, resolution), dtype=np.uint8)
    for face in np.asarray(mesh.faces, dtype=np.int64):
        cv2.fillPoly(raster, [pixels[face]], (1.0,))
    return raster.astype(np.bool_)


def _silhouette_support(
    *,
    masks: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    center_world: np.ndarray,
    axes_world_columns: np.ndarray,
    midpoint: np.ndarray,
    extent: float,
    axis: int,
    lower: np.ndarray,
    upper: np.ndarray,
    resolution: int,
    axis_samples: int,
    frame_canonical_columns: np.ndarray,
) -> np.ndarray:
    transverse = [index for index in range(3) if index != axis]
    minimum = lower[transverse]
    span = upper[transverse] - minimum
    first = minimum[0] + (np.arange(resolution) + 0.5) / resolution * span[0]
    second = minimum[1] + (np.arange(resolution) + 0.5) / resolution * span[1]
    xx, yy = np.meshgrid(first, second, indexing="xy")
    transverse_points = np.column_stack((xx.ravel(), yy.ravel()))
    along = np.linspace(lower[axis], upper[axis], axis_samples)
    canonical = np.zeros((len(transverse_points), axis_samples, 3), dtype=np.float64)
    canonical[:, :, axis] = along[None, :]
    canonical[:, :, transverse[0]] = transverse_points[:, 0, None]
    canonical[:, :, transverse[1]] = transverse_points[:, 1, None]
    canonical = canonical @ frame_canonical_columns.T
    oriented = canonical * (extent / 2.0) + midpoint
    world = oriented @ axes_world_columns.T + center_world
    homogeneous = np.concatenate((world, np.ones((*world.shape[:2], 1))), axis=2)
    fractions = np.zeros((len(masks), len(transverse_points)), dtype=np.float64)
    height, width = masks.shape[1:]
    for view_index, mask in enumerate(masks):
        camera = homogeneous @ extrinsics[view_index].T
        positive = camera[:, :, 2] > 1e-8
        pixels_h = camera @ intrinsics[view_index].T
        u = np.rint(pixels_h[:, :, 0] / np.maximum(pixels_h[:, :, 2], 1e-8)).astype(int)
        v = np.rint(pixels_h[:, :, 1] / np.maximum(pixels_h[:, :, 2], 1e-8)).astype(int)
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        admitted = np.zeros_like(inside)
        admitted[inside] = mask[v[inside], u[inside]]
        fractions[view_index] = admitted.mean(axis=1)
    return fractions.reshape(len(masks), resolution, resolution)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / max(int(union), 1))


def _rectilinear_candidate(mask: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    """Fit a generic orthogonal cell complex to a raster, for diagnosis only."""

    binary = np.asarray(mask, dtype=np.bool_)

    def levels(values: np.ndarray, *, other_extent: int) -> np.ndarray:
        smooth = ndimage.gaussian_filter1d(values.astype(np.float64), sigma=1.0)
        threshold = max(2.0, 0.10 * float(smooth.max()), 0.02 * other_extent)
        peaks, _ = find_peaks(smooth, height=threshold, distance=3)
        candidates = list(int(value) for value in peaks)
        for endpoint in (0, len(smooth) - 1):
            if smooth[endpoint] >= threshold:
                candidates.append(endpoint)
        return np.asarray(sorted(set(candidates)), dtype=np.int32)

    padded_x = np.pad(binary, ((0, 0), (1, 1)), constant_values=False)
    padded_y = np.pad(binary, ((1, 1), (0, 0)), constant_values=False)
    vertical = np.count_nonzero(padded_x[:, 1:] != padded_x[:, :-1], axis=0)
    horizontal = np.count_nonzero(padded_y[1:, :] != padded_y[:-1, :], axis=1)
    x_levels = levels(vertical, other_extent=binary.shape[0])
    y_levels = levels(horizontal, other_extent=binary.shape[1])
    report: dict[str, object] = {
        "x_levels": x_levels.tolist(),
        "y_levels": y_levels.tolist(),
    }
    if not 2 <= len(x_levels) <= 12 or not 2 <= len(y_levels) <= 12:
        return None, report
    rectified = np.zeros_like(binary)
    for x0, x1 in zip(x_levels[:-1], x_levels[1:], strict=True):
        for y0, y1 in zip(y_levels[:-1], y_levels[1:], strict=True):
            if x1 <= x0 or y1 <= y0:
                continue
            cell = binary[y0:y1, x0:x1]
            if cell.size and float(cell.mean()) >= 0.5:
                rectified[y0:y1, x0:x1] = True
    rectified = _largest(rectified)
    report["source_iou"] = _iou(rectified, binary)
    return rectified, report


def _simplification_table(mask: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    result: dict[str, object] = {}
    for fraction in (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02):
        simplified = cv2.approxPolyDP(contour, fraction * perimeter, True)
        raster = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(raster, [simplified], (1.0,))
        result[str(fraction)] = {
            "vertices": len(simplified),
            "reference_iou": _iou(raster.astype(np.bool_), reference),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report_payload = json.loads(
        (args.run / "artefacts" / "cad_report.json").read_text(encoding="utf-8")
    )
    report = report_payload["report"]
    if "axis_candidates" not in report:
        selected = next(
            item
            for item in report["candidates"]
            if item["selected"] and item["family"] == "extrude"
        )
        report = selected["generator_report"]
    axis = int(report["selected_axis"])
    candidate = next(item for item in report["axis_candidates"] if int(item["axis"]) == axis)
    frame = np.asarray(candidate["frame_canonical_columns"], dtype=np.float64)
    lower = np.asarray(candidate["lower"], dtype=np.float64)
    upper = np.asarray(candidate["upper"], dtype=np.float64)
    transverse = [index for index in range(3) if index != axis]
    minimum = lower[transverse]
    span = upper[transverse] - minimum
    resolution = int(candidate["raster_resolution"])

    trace = json.loads(
        (args.run / "artefacts" / "canonicalizer" / "canonicalizer_trace.json").read_text(
            encoding="utf-8"
        )
    )
    center_world = np.asarray(trace["orientation"]["center_world"], dtype=np.float64)
    axes_world_columns = np.asarray(trace["orientation"]["axes_world_columns"], dtype=np.float64).T
    midpoint = np.asarray(trace["normalization"]["midpoint"], dtype=np.float64)
    extent = float(trace["normalization"]["largest_extent"])

    def normalized_stage(name: str) -> np.ndarray:
        with np.load(args.run / "artefacts" / "canonicalizer" / name) as payload:
            world = np.asarray(payload["points"], dtype=np.float64)
        oriented = (world - center_world) @ axes_world_columns
        return (2.0 * (oriented - midpoint) / extent) @ frame

    raw = _solid_occupancy(
        normalized_stage("00_input.npz"),
        axis=axis,
        minimum=minimum,
        span=span,
        resolution=resolution,
    )
    filtered = _solid_occupancy(
        normalized_stage("03_multi-view-consistency.npz"),
        axis=axis,
        minimum=minimum,
        span=span,
        resolution=resolution,
    )
    reference = _mesh_profile(
        args.reference,
        center_world=center_world,
        axes_world_columns=axes_world_columns,
        midpoint=midpoint,
        extent=extent,
        axis=axis,
        minimum=minimum,
        span=span,
        resolution=resolution,
        frame_canonical_columns=frame,
    )
    profile_points = np.asarray(candidate["profile"]["points"], dtype=np.float64)
    profile_pixels = np.rint((profile_points - minimum) / span * (resolution - 1)).astype(np.int32)
    current_profile = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(current_profile, [profile_pixels], (1.0,))
    current_profile = current_profile.astype(np.bool_)
    current_rectilinear, current_rectilinear_report = _rectilinear_candidate(current_profile)

    prediction_path = args.run / "artefacts" / "geometry" / "artefacts" / "camera_prediction.npz"
    with np.load(prediction_path) as prediction:
        fractions = _silhouette_support(
            masks=np.asarray(prediction["masks"], dtype=np.bool_),
            intrinsics=np.asarray(prediction["intrinsics"], dtype=np.float64),
            extrinsics=np.asarray(prediction["extrinsics"], dtype=np.float64),
            center_world=center_world,
            axes_world_columns=axes_world_columns,
            midpoint=midpoint,
            extent=extent,
            axis=axis,
            lower=lower,
            upper=upper,
            resolution=resolution,
            axis_samples=9,
            frame_canonical_columns=frame,
        )

    candidates: list[tuple[float, int, np.ndarray, float]] = []
    for along_fraction in (0.5, 0.75, 1.0):
        per_view = fractions >= along_fraction
        for minimum_views in range(2, len(fractions) + 1):
            mask = per_view.sum(axis=0) >= minimum_views
            mask = ndimage.binary_closing(mask, iterations=1)
            mask = ndimage.binary_fill_holes(_largest(mask)).astype(np.bool_)
            candidates.append((along_fraction, minimum_views, mask, _iou(mask, reference)))
    best = max(candidates, key=lambda item: item[3])
    fixed_minimum_views = int(np.ceil(0.6 * len(fractions)))
    fixed = next(item for item in candidates if item[0] == 1.0 and item[1] == fixed_minimum_views)
    intersection = np.logical_and(raw, fixed[2])
    best_intersection = np.logical_and(raw, best[2])
    rectilinear, rectilinear_report = _rectilinear_candidate(best_intersection)
    output = args.output or args.run / "profile_evidence_evaluation.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=160)
    panels = (
        (raw, f"raw · IoU {_iou(raw, reference):.3f}"),
        (filtered, f"filtered · IoU {_iou(filtered, reference):.3f}"),
        (
            best[2],
            f"silhouettes · IoU {best[3]:.3f}\naxis≥{best[0]:.2f}, views≥{best[1]}",
        ),
        (reference, "reference (evaluation only)"),
    )
    for axis_plot, (mask, title) in zip(axes, panels, strict=True):
        axis_plot.imshow(mask, cmap="gray", origin="lower")
        axis_plot.set_title(title)
        axis_plot.axis("off")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    result = {
        "raw_iou": _iou(raw, reference),
        "filtered_iou": _iou(filtered, reference),
        "current_profile_iou": _iou(current_profile, reference),
        "current_rectilinear": {
            **current_rectilinear_report,
            "reference_iou": (
                _iou(current_rectilinear, reference) if current_rectilinear is not None else None
            ),
        },
        "best_silhouette_iou": best[3],
        "best_axis_fraction": best[0],
        "best_minimum_views": best[1],
        "fixed_60_percent_views": fixed_minimum_views,
        "fixed_silhouette_iou": fixed[3],
        "raw_silhouette_agreement_iou": _iou(raw, fixed[2]),
        "raw_intersection_iou": _iou(intersection, reference),
        "raw_intersection_simplification": _simplification_table(
            intersection,
            reference,
        ),
        "best_silhouette_simplification": _simplification_table(
            best[2],
            reference,
        ),
        "best_intersection_iou": _iou(best_intersection, reference),
        "rectilinear": {
            **rectilinear_report,
            "reference_iou": (_iou(rectilinear, reference) if rectilinear is not None else None),
        },
        "best_intersection_simplification": _simplification_table(
            best_intersection,
            reference,
        ),
        "full_axis_silhouette_iou_by_minimum_views": {
            str(minimum_views): iou
            for along_fraction, minimum_views, _, iou in candidates
            if along_fraction == 1.0
        },
        "note": "reference CAD is evaluator-only and never used by reconstruction",
        "output": str(output),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
