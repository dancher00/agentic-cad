#!/usr/bin/env python3
"""Generate every README visual from frozen benchmark artefacts.

The committed showcase manifest fixes the population, selection rule, IDs,
seed, checkpoints and expected metrics. Runtime benchmark data are deliberately
not redistributed; rerun the documented pilot before regenerating these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageOps

from da3_cad.cad.program import extract_parameters
from da3_cad.viewer import build_viewer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/showcase/manifest.json"
DEFAULT_RELEASE_FACTS = ROOT / "benchmarks/release_facts.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/readme"
DEFAULT_ASSET_MANIFEST = ROOT / "benchmarks/showcase/assets.json"

INK = "#172033"
MUTED = "#667085"
PAPER = "#F7F9FC"
PANEL = "#FFFFFF"
LINE = "#D8E0EC"
TEAL = "#16A085"
TEAL_DARK = "#0C6E62"
CYAN = "#3B82C4"
ORANGE = "#E67E22"
RED = "#C94141"
PURPLE = "#7C5CC4"

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class ShowcaseItem:
    role: str
    dataset: str
    item_id: str
    candidate_row: str
    selected_candidate: int | None
    valid: bool
    chamfer: float | None
    iou: float | None
    invalid_reason: str | None
    result_path: Path
    views: tuple[Path, ...]
    cloud_path: Path
    gt_path: Path
    model_path: Path | None
    program_path: Path | None


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise FileNotFoundError(f"deterministic figure font is missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _only(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {description}; found {len(paths)}")
    return paths[0]


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"not a non-empty triangle mesh: {path}")
    return loaded


def _result_path(pilot: Path, dataset: str, row: str, item_id: str) -> Path:
    return pilot / "results" / dataset / "n08" / row / "items" / f"{item_id}.json"


def _numeric_record(path: Path) -> tuple[bool, float | None, float | None, int | None, str | None]:
    payload = _load_json(path)
    normative = payload["normative"]
    selection = payload["selection"]
    valid = bool(normative["valid_prediction"])
    chamfer = (
        float(normative["chamfer"]["bidirectional_squared_x1000"])
        if normative["chamfer"] is not None
        else None
    )
    iou = float(normative["iou"]["percent"]) if normative["iou"] is not None else None
    candidate = selection["selected_index"]
    return valid, chamfer, iou, candidate, normative.get("invalid_reason")


def _validate_selection_rule(manifest: dict[str, Any], pilot: Path) -> None:
    rows: list[tuple[str, str, float, float]] = []
    invalid: list[tuple[str, str]] = []
    for dataset in ("deepcad", "fusion360"):
        best_root = pilot / "results" / dataset / "n08" / "best-of-10-input-CD" / "items"
        single_root = pilot / "results" / dataset / "n08" / "single-decode" / "items"
        for path in sorted(best_root.glob("*.json")):
            valid, chamfer, iou, _, _ = _numeric_record(path)
            if valid and chamfer is not None and iou is not None:
                rows.append((dataset, path.stem, iou, chamfer))
        for path in sorted(single_root.glob("*.json")):
            valid, _, _, _, _ = _numeric_record(path)
            if not valid:
                invalid.append((dataset, path.stem))
    if len(rows) != 20:
        raise ValueError(
            f"showcase selection requires all 20 valid best-of-10 rows; got {len(rows)}"
        )
    expected: dict[str, tuple[str, str]] = {}
    for dataset in ("deepcad", "fusion360"):
        winner = max((row for row in rows if row[0] == dataset), key=lambda row: (row[2], -row[3]))
        expected[f"success-{dataset}"] = winner[:2]
    median = float(np.median([row[2] for row in rows]))
    partial = min(rows, key=lambda row: (abs(row[2] - median), row[0], row[1]))
    expected["partial"] = partial[:2]
    expected["failure"] = min(invalid)
    actual = {item["role"]: (item["dataset"], item["item_id"]) for item in manifest["items"]}
    if actual != expected:
        raise ValueError(
            f"committed showcase selection does not match its rule: {actual} != {expected}"
        )


def _resolve_items(manifest: dict[str, Any]) -> tuple[ShowcaseItem, ...]:
    source = manifest["source"]
    pilot = ROOT / source["pilot_root"]
    mesh_root = ROOT / source["mesh_root"]
    _validate_selection_rule(manifest, pilot)
    resolved: list[ShowcaseItem] = []
    for entry in manifest["items"]:
        dataset = entry["dataset"]
        item_id = entry["item_id"]
        row = entry["candidate_row"]
        result_path = _result_path(pilot, dataset, row, item_id)
        valid, chamfer, iou, candidate, invalid_reason = _numeric_record(result_path)
        if valid is not bool(entry["expected_valid"]):
            raise ValueError(f"validity drift for {dataset}/{item_id}")
        for label, actual, expected_value in (
            ("Chamfer", chamfer, entry["expected_chamfer_x1000"]),
            ("IoU", iou, entry["expected_iou_percent"]),
        ):
            if actual is None or expected_value is None:
                if actual is not expected_value:
                    raise ValueError(f"{label} nullability drift for {dataset}/{item_id}")
            elif not math.isclose(actual, float(expected_value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{label} drift for {dataset}/{item_id}: {actual} != {expected_value}"
                )
        if candidate != entry["selected_candidate"]:
            raise ValueError(f"selected candidate drift for {dataset}/{item_id}")
        expected_reason = entry.get("expected_invalid_reason")
        if expected_reason is not None and invalid_reason != expected_reason:
            raise ValueError(f"invalid reason drift for {dataset}/{item_id}")

        view_root = pilot / "view_subsets" / dataset / item_id / "n08"
        views = tuple(sorted(view_root.glob("view_*.png")))
        if len(views) != manifest["view_count"]:
            raise ValueError(f"expected 8 views for {dataset}/{item_id}; got {len(views)}")
        cloud_path = _only(
            sorted(
                (pilot / "geometry" / dataset / item_id / "n08").glob("*/artefacts/fused_cloud.npz")
            ),
            f"fused cloud for {dataset}/{item_id}",
        )
        gt_path = mesh_root / dataset / f"{item_id}.stl"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing showcase GT mesh: {gt_path}")
        model_path: Path | None = None
        program_path: Path | None = None
        if candidate is not None:
            decode_root = pilot / "decode" / dataset / item_id / "n08"
            candidate_dir = _only(
                sorted(decode_root.glob(f"*/candidate_{candidate:02d}")),
                f"candidate {candidate} for {dataset}/{item_id}",
            )
            model_path = candidate_dir / "model.stl"
            program_path = candidate_dir / "model.py"
            if not model_path.is_file() or not program_path.is_file():
                raise FileNotFoundError(f"incomplete selected candidate: {candidate_dir}")
        resolved.append(
            ShowcaseItem(
                role=entry["role"],
                dataset=dataset,
                item_id=item_id,
                candidate_row=row,
                selected_candidate=candidate,
                valid=valid,
                chamfer=chamfer,
                iou=iou,
                invalid_reason=invalid_reason,
                result_path=result_path,
                views=views,
                cloud_path=cloud_path,
                gt_path=gt_path,
                model_path=model_path,
                program_path=program_path,
            )
        )
    return tuple(resolved)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill: str = PANEL) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=LINE, width=2)


def _label(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, color: str = MUTED
) -> None:
    draw.text(xy, text.upper(), font=_font(17, bold=True), fill=color)


def _wrap(
    draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.FreeTypeFont
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str = INK,
    spacing: int = 7,
) -> int:
    lines = _wrap(draw, text, width, font)
    x, y = xy
    line_height = font.size + spacing
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, font=font, fill=fill)
    return y + len(lines) * line_height


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(rgb, size, method=Image.Resampling.LANCZOS)


def _draw_views(
    image: Image.Image,
    box: tuple[int, int, int, int],
    views: tuple[Path, ...],
    *,
    count: int = 4,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    gap = 8
    cols = 2
    rows = 2 if count > 2 else 1
    width = (x1 - x0 - gap) // cols
    height = (y1 - y0 - gap * (rows - 1)) // rows
    indices = np.linspace(0, len(views) - 1, count, dtype=int)
    for slot, index in enumerate(indices):
        col, row = slot % cols, slot // cols
        left = x0 + col * (width + gap)
        top = y0 + row * (height + gap)
        tile = _fit_image(views[int(index)], (width, height))
        image.paste(tile, (left, top))
        draw.rounded_rectangle(
            (left, top, left + width, top + height), radius=8, outline="#FFFFFF", width=2
        )
        draw.rounded_rectangle((left + 8, top + 8, left + 78, top + 35), radius=8, fill="#172033CC")
        draw.text(
            (left + 17, top + 10),
            f"view {int(index):02d}",
            font=_font(14, bold=True),
            fill="#FFFFFF",
        )


def _normalize(vertices: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    points = np.asarray(vertices, dtype=np.float64)
    low = points.min(axis=0)
    high = points.max(axis=0)
    extent = float(np.max(high - low))
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("cannot normalize degenerate geometry")
    return (points - (low + high) / 2.0) / extent


def _rotation() -> npt.NDArray[np.float64]:
    yaw, pitch = math.radians(35.0), math.radians(-25.0)
    ry = np.asarray(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return rx @ ry


def _project(
    vertices: npt.NDArray[np.floating[Any]], box: tuple[int, int, int, int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rotated = np.asarray(vertices, dtype=np.float64) @ _rotation().T
    x0, y0, x1, y1 = box
    span = min(x1 - x0, y1 - y0) * 0.80
    screen = np.empty((len(rotated), 2), dtype=np.float64)
    screen[:, 0] = (x0 + x1) / 2.0 + rotated[:, 0] * span
    screen[:, 1] = (y0 + y1) / 2.0 - rotated[:, 1] * span
    return screen, rotated[:, 2]


def _mesh_layer(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    mesh: trimesh.Trimesh,
    *,
    color: tuple[int, int, int],
    alpha: int = 230,
    wire_only: bool = False,
) -> Image.Image:
    vertices = _normalize(np.asarray(mesh.vertices))
    screen, depth = _project(vertices, box)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ _rotation().T
    order = np.argsort(depth[faces].mean(axis=1))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    light = np.asarray([-0.25, 0.55, 0.80], dtype=np.float64)
    light /= np.linalg.norm(light)
    for face_index in order:
        face = faces[int(face_index)]
        polygon = [tuple(screen[int(vertex)]) for vertex in face]
        if wire_only:
            draw.line([*polygon, polygon[0]], fill=(*color, alpha), width=3, joint="curve")
            continue
        intensity = 0.48 + 0.52 * abs(float(np.dot(normals[int(face_index)], light)))
        fill = tuple(int(channel * intensity) for channel in color)
        draw.polygon(polygon, fill=(*fill, alpha), outline=(*color, min(255, alpha + 15)))
    return layer


def _draw_mesh(
    image: Image.Image,
    box: tuple[int, int, int, int],
    mesh: trimesh.Trimesh,
    *,
    color: tuple[int, int, int] = (22, 160, 133),
) -> None:
    image.alpha_composite(_mesh_layer(image.size, box, mesh, color=color))


def _draw_cloud(image: Image.Image, box: tuple[int, int, int, int], path: Path) -> None:
    payload = np.load(path)
    points = np.asarray(payload["points"], dtype=np.float64)
    colors = np.asarray(payload["colors"], dtype=np.uint8)
    maximum = 7_500
    indices = np.linspace(0, len(points) - 1, min(maximum, len(points)), dtype=int)
    points = _normalize(points[indices])
    colors = colors[indices]
    screen, depth = _project(points, box)
    order = np.argsort(depth)
    draw = ImageDraw.Draw(image)
    for index in order:
        x, y = screen[int(index)]
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            color = tuple(int(value) for value in colors[int(index), :3])
            draw.ellipse((x - 1.3, y - 1.3, x + 1.3, y + 1.3), fill=color)


def _parameter_rows(item: ShowcaseItem, maximum: int = 7) -> list[tuple[str, float]]:
    if item.program_path is None:
        return []
    parameters = extract_parameters(item.program_path.read_text(encoding="utf-8"))
    prefixes = ("box_", "extrude_", "circle_", "radius", "diameter", "thickness")
    ranked = sorted(
        parameters.items(), key=lambda pair: (not pair[0].startswith(prefixes), pair[0])
    )
    return ranked[:maximum]


def _draw_parameters(
    image: Image.Image,
    box: tuple[int, int, int, int],
    item: ShowcaseItem,
    *,
    compact: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, _ = box
    rows = _parameter_rows(item, maximum=5 if compact else 7)
    if not rows:
        draw.text(
            (x0 + 16, y0 + 16), "No recovered parameters", font=_font(19, bold=True), fill=RED
        )
        _draw_wrapped(
            draw,
            (x0 + 16, y0 + 52),
            item.invalid_reason or "No valid candidate",
            x1 - x0 - 32,
            font=_font(15),
            fill=MUTED,
        )
        return
    draw.text((x0 + 12, y0 + 8), "Implementation literals", font=_font(17, bold=True), fill=INK)
    draw.text((x0 + 12, y0 + 35), "not inferred design intent", font=_font(13), fill=ORANGE)
    top = y0 + 69
    row_height = 31 if compact else 36
    for index, (name, value) in enumerate(rows):
        y = top + index * row_height
        if index % 2 == 0:
            draw.rounded_rectangle(
                (x0 + 8, y - 3, x1 - 8, y + row_height - 4), radius=6, fill="#F2F5F9"
            )
        shown = name if len(name) <= 25 else f"{name[:22]}…"
        draw.text((x0 + 16, y), shown, font=_font(13), fill=INK)
        value_text = f"{value:g}"
        value_width = draw.textlength(value_text, font=_font(13, bold=True))
        draw.text((x1 - 17 - value_width, y), value_text, font=_font(13, bold=True), fill=TEAL_DARK)


def _footer(draw: ImageDraw.ImageDraw, manifest: dict[str, Any], *, y: int, width: int) -> None:
    source = manifest["source"]
    text = (
        f"Rendered benchmark · N=8 · seed {manifest['global_seed']} · "
        f"DA3-L {source['da3_checkpoint'].split('@')[1][:7]} · "
        f"Cadrille-RL {source['cadrille_checkpoint'].split('@')[1][:7]} · "
        f"run commit {source['repository_commit'][:7]}"
    )
    tw = draw.textlength(text, font=_font(14))
    draw.text(((width - tw) / 2, y), text, font=_font(14), fill=MUTED)


def _hero_frame(item: ShowcaseItem, manifest: dict[str, Any], active: int) -> Image.Image:
    width, height = 1500, 520
    image = Image.new("RGBA", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (48, 28), "Photographs to an inspectable CAD candidate", font=_font(32, bold=True), fill=INK
    )
    draw.text(
        (48, 72),
        "Every stage remains visible — including the limitations.",
        font=_font(18),
        fill=MUTED,
    )
    boxes = [
        (42, 124, 408, 430),
        (436, 124, 747, 430),
        (775, 124, 1086, 430),
        (1114, 124, 1458, 430),
    ]
    labels = ("1 · INPUT VIEWS", "2 · FUSED CLOUD", "3 · VALID SOLID", "4 · PARAMETERS")
    for index, box in enumerate(boxes):
        _panel(draw, box, fill="#FFFFFF" if index != active else "#EEF9F6")
        if index == active:
            draw.rounded_rectangle(box, radius=18, outline=TEAL, width=5)
        _label(
            draw,
            (box[0] + 16, box[1] + 15),
            labels[index],
            color=TEAL_DARK if index == active else MUTED,
        )
    _draw_views(image, (58, 167, 392, 412), item.views)
    _draw_cloud(image, (452, 168, 731, 411), item.cloud_path)
    if item.model_path is None:
        raise ValueError("hero item must have a valid solid")
    _draw_mesh(image, (791, 168, 1070, 411), _mesh(item.model_path))
    _draw_parameters(image, (1128, 160, 1444, 414), item, compact=True)
    for x in (420, 759, 1098):
        draw.line((x - 7, 277, x + 7, 277), fill=TEAL, width=4)
        draw.polygon(((x + 7, 277), (x - 2, 269), (x - 2, 285)), fill=TEAL)
    draw.rounded_rectangle((1130, 389, 1439, 417), radius=8, fill="#FFF4E8")
    draw.text(
        (1141, 394), "AST-lifted; no semantic feature tree", font=_font(12, bold=True), fill=ORANGE
    )
    _footer(draw, manifest, y=476, width=width)
    return image


def _save_hero(item: ShowcaseItem, manifest: dict[str, Any], path: Path) -> None:
    frames = [_hero_frame(item, manifest, active) for active in range(4)]
    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames
    ]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=(900, 900, 900, 1300),
        loop=0,
        optimize=False,
        disposal=2,
    )


def _metric_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    value: str,
    *,
    color: str,
) -> int:
    x, y = xy
    text = f"{label}  {value}"
    width = int(draw.textlength(text, font=_font(18, bold=True))) + 30
    draw.rounded_rectangle((x, y, x + width, y + 42), radius=13, fill=color)
    draw.text((x + 15, y + 9), text, font=_font(18, bold=True), fill="#FFFFFF")
    return x + width


def _draw_overlay(
    image: Image.Image,
    box: tuple[int, int, int, int],
    gt: trimesh.Trimesh,
    recovered: trimesh.Trimesh | None,
) -> None:
    image.alpha_composite(_mesh_layer(image.size, box, gt, color=(59, 130, 196), alpha=145))
    if recovered is not None:
        image.alpha_composite(
            _mesh_layer(image.size, box, recovered, color=(230, 126, 34), alpha=235, wire_only=True)
        )
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 14, y1 - 46, x0 + 118, y1 - 14), radius=8, fill="#E8F2FB")
    draw.text((x0 + 25, y1 - 40), "GT mesh", font=_font(14, bold=True), fill=CYAN)
    if recovered is None:
        draw.rounded_rectangle((x0 + 127, y1 - 46, x0 + 342, y1 - 14), radius=8, fill="#FBECEC")
        draw.text(
            (x0 + 138, y1 - 40), "Recovered: no valid solid", font=_font(14, bold=True), fill=RED
        )
        draw.line((x0 + 85, y0 + 75, x1 - 85, y1 - 75), fill=RED, width=5)
        draw.line((x1 - 85, y0 + 75, x0 + 85, y1 - 75), fill=RED, width=5)
    else:
        draw.rounded_rectangle((x0 + 127, y1 - 46, x0 + 300, y1 - 14), radius=8, fill="#FFF0E1")
        draw.text((x0 + 138, y1 - 40), "Recovered solid", font=_font(14, bold=True), fill=ORANGE)


def _showcase_figure(item: ShowcaseItem, manifest: dict[str, Any]) -> Image.Image:
    width, height = 1600, 980
    image = Image.new("RGBA", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    role_title = {
        "success-deepcad": "Success selected by the frozen rule · DeepCAD",
        "success-fusion360": "Success selected by the frozen rule · Fusion 360",
        "partial": "Partial recovery · pooled-median example",
        "failure": "Explicit failure · fixed single decode",
    }[item.role]
    draw.text((46, 30), role_title, font=_font(31, bold=True), fill=INK)
    draw.text(
        (48, 76),
        f"{item.dataset} / {item.item_id} · {item.candidate_row}",
        font=_font(18),
        fill=MUTED,
    )
    x = 1120
    if item.valid:
        if item.chamfer is None or item.iou is None:
            raise ValueError("valid showcase item has no metrics")
        x = _metric_badge(draw, (x, 36), "CD×10³", f"{item.chamfer:.3f}", color=PURPLE) + 12
        _metric_badge(draw, (x, 36), "IoU", f"{item.iou:.2f}%", color=TEAL_DARK)
    else:
        _metric_badge(draw, (x, 36), "CD / IoU", "N/A — invalid", color=RED)

    top_boxes = [
        (42, 128, 478, 545),
        (500, 128, 842, 545),
        (864, 128, 1206, 545),
        (1228, 128, 1558, 545),
    ]
    for box in top_boxes:
        _panel(draw, box)
    for box, title in zip(
        top_boxes, ("INPUT VIEWS", "RECOVERED CLOUD", "RECOVERED SOLID", "PARAMETERS"), strict=True
    ):
        _label(draw, (box[0] + 16, box[1] + 15), title)
    _draw_views(image, (58, 171, 462, 526), item.views)
    _draw_cloud(image, (516, 173, 826, 525), item.cloud_path)
    recovered = _mesh(item.model_path) if item.model_path is not None else None
    if recovered is None:
        draw.line((914, 250, 1156, 425), fill=RED, width=6)
        draw.line((1156, 250, 914, 425), fill=RED, width=6)
        draw.text((936, 453), "No valid solid", font=_font(21, bold=True), fill=RED)
    else:
        _draw_mesh(image, (880, 173, 1190, 525), recovered)
    _draw_parameters(image, (1242, 171, 1544, 525), item)

    overlay_panel = (42, 571, 882, 907)
    details_panel = (904, 571, 1558, 907)
    _panel(draw, overlay_panel)
    _panel(draw, details_panel)
    _label(draw, (60, 590), "GT VS RECOVERED · EVALUATOR FRAME")
    _draw_overlay(image, (66, 628, 858, 888), _mesh(item.gt_path), recovered)
    _label(draw, (924, 590), "WHAT THIS CASE SAYS")
    if item.role.startswith("success"):
        statement = (
            "Highest valid N=8 best-of-10 IoU in this dataset, selected only after the "
            "population and rule were frozen. It is a relative success, not production accuracy."
        )
    elif item.role == "partial":
        statement = (
            "Closest valid record to the pooled median IoU. A solid is emitted, but the GT overlay "
            "shows that topology and proportions are not recovered reliably."
        )
    else:
        statement = (
            "The fixed candidate zero failed AST geometry-equivalence validation. CD and IoU stay "
            "N/A; invalid outputs are retained rather than silently removed."
        )
    bottom = _draw_wrapped(draw, (926, 630), statement, 600, font=_font(19), spacing=9)
    facts = [
        f"Status: {'valid solid' if item.valid else 'invalid — no solid'}",
        f"Selector: {item.candidate_row}",
        f"Candidate: {item.selected_candidate if item.selected_candidate is not None else 'none'}",
        "Alignment: independent bbox-centre + isotropic scale; no ICP",
    ]
    for index, fact in enumerate(facts):
        y = bottom + 20 + index * 39
        draw.ellipse((928, y + 5, 939, y + 16), fill=TEAL if item.valid else RED)
        draw.text((951, y), fact, font=_font(16), fill=INK)
    _footer(draw, manifest, y=944, width=width)
    return image


def _facts_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {fact["id"]: fact for fact in payload["facts"]}


def _decomposition_figure(release_facts: Path) -> Image.Image:
    facts = _facts_by_id(release_facts)
    left_ids = (
        "bottleneck-uncalibrated",
        "bottleneck-exact-cameras",
        "bottleneck-exact-cameras-per-view-gt-affine",
    )
    left = [facts[item_id] for item_id in left_ids]
    segmentation = facts["tless-segmentation-control-n8-best-of-10-input-CD"]
    values = [float(fact["metrics"]["precision_at_0.05"]) * 100.0 for fact in left]
    metrics = segmentation["metrics"]
    auto = float(metrics["automatic_mean_iou_percent"])
    oracle = float(metrics["oracle_mean_iou_percent"])
    gain = float(metrics["mean_iou_gain_percentage_points"])
    threshold = float(metrics["material_gain_threshold_percentage_points"])

    width, height = 1600, 850
    image = Image.new("RGBA", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (52, 32),
        "Three measured levers behind the photo → CAD gap",
        font=_font(34, bold=True),
        fill=INK,
    )
    draw.text(
        (54, 80),
        "Two controlled domains; the percentages are diagnostic, not additive.",
        font=_font(19),
        fill=MUTED,
    )
    left_box, right_box = (42, 132, 1018, 750), (1040, 132, 1558, 750)
    _panel(draw, left_box)
    _panel(draw, right_box)
    _label(draw, (68, 158), "RENDERED N=8 · PRECISION@.05 · 20 OBJECTS")
    draw.text(
        (68, 196),
        "Camera pose is the largest single measured gain",
        font=_font(23, bold=True),
        fill=INK,
    )
    names = ("Unposed DA3", "Exact cameras\nGT oracle", "+ per-view depth affine\nGT oracle")
    colors = (PURPLE, CYAN, TEAL_DARK)
    base_y = 620
    bar_width = 224
    max_height = 320
    x_positions = (104, 393, 682)
    for index, (name, value, color, x) in enumerate(
        zip(names, values, colors, x_positions, strict=True)
    ):
        height_px = int(max_height * value / 70.0)
        draw.rounded_rectangle(
            (x, base_y - height_px, x + bar_width, base_y), radius=14, fill=color
        )
        label = f"{value:.1f}%"
        label_width = draw.textlength(label, font=_font(28, bold=True))
        draw.text(
            (x + (bar_width - label_width) / 2, base_y - height_px + 18),
            label,
            font=_font(28, bold=True),
            fill="#FFFFFF",
        )
        for line_index, line in enumerate(name.splitlines()):
            line_width = draw.textlength(line, font=_font(16, bold=True))
            draw.text(
                (x + (bar_width - line_width) / 2, base_y + 17 + line_index * 24),
                line,
                font=_font(16, bold=True),
                fill=INK,
            )
        if index:
            previous = values[index - 1]
            delta = value - previous
            arrow_x = x - 48
            draw.line((arrow_x - 18, 390, arrow_x + 18, 390), fill=TEAL, width=4)
            draw.polygon(((arrow_x + 18, 390), (arrow_x + 7, 381), (arrow_x + 7, 399)), fill=TEAL)
            delta_text = f"+{delta:.1f} pp"
            delta_width = draw.textlength(delta_text, font=_font(15, bold=True))
            draw.text(
                (arrow_x - delta_width / 2, 350),
                delta_text,
                font=_font(15, bold=True),
                fill=TEAL_DARK,
            )
    draw.text(
        (68, 704),
        "seed 20260810 · DA3-L c54c26b · commits b82d0bf / 71d7dee",
        font=_font(14),
        fill=MUTED,
    )

    _label(draw, (1066, 158), "T-LESS N=8 · MEAN MESH IOU · 30 OBJECTS")
    draw.text(
        (1066, 196), "Segmentation is broken, but secondary", font=_font(22, bold=True), fill=INK
    )
    bar_x = 1090
    max_bar = 410
    for index, (name, value, color) in enumerate(
        (("Automatic mask", auto, RED), ("Official GT-mask oracle", oracle, TEAL_DARK))
    ):
        y = 286 + index * 150
        draw.text((bar_x, y), name, font=_font(17, bold=True), fill=INK)
        draw.rounded_rectangle((bar_x, y + 36, bar_x + max_bar, y + 85), radius=10, fill="#E9EEF5")
        shown = max(1, int(max_bar * value / 10.0))
        draw.rounded_rectangle((bar_x, y + 36, bar_x + shown, y + 85), radius=10, fill=color)
        draw.text((bar_x + 13, y + 47), f"{value:.2f}%", font=_font(19, bold=True), fill="#FFFFFF")
    draw.rounded_rectangle(
        (1082, 594, 1518, 670), radius=12, fill="#FFF4E8", outline="#F4C78E", width=2
    )
    draw.text(
        (1102, 609),
        f"+{gain:.2f} pp  <  preregistered +{threshold:.0f} pp gate",
        font=_font(18, bold=True),
        fill=ORANGE,
    )
    draw.text(
        (1102, 640), "Mask precision 4.42% → 100%; dominant gap remains.", font=_font(14), fill=INK
    )
    draw.text(
        (1066, 704),
        "seed 20260810 · DA3-L c54c26b · Cadrille-RL 712489b",
        font=_font(14),
        fill=MUTED,
    )
    draw.text(
        (52, 791),
        "Left: exact-camera and depth interventions use GT and are upper bounds. "
        "Right: paired real-camera segmentation control. "
        "See benchmarks/release_facts.json.",
        font=_font(15),
        fill=MUTED,
    )
    return image


def _viewer_metadata(item: ShowcaseItem) -> dict[str, Any]:
    rows = _parameter_rows(item, maximum=12)
    return {
        "schema_version": "2.0",
        "backend": "cadrille-rl",
        "units": "decoder-training-unit",
        "parameter_semantics": {"status": "implementation-literals-only"},
        "primary_parameters": [],
        "implementation_parameters": [
            {
                "name": name,
                "value": value,
                "unit": "decoder unit",
                "semantic_role": "AST-lifted literal",
            }
            for name, value in rows
        ],
        "warnings": ["No primary engineering semantics were inferred from Cadrille output."],
    }


def _viewer_screenshot(item: ShowcaseItem, output: Path) -> None:
    if item.model_path is None or item.program_path is None:
        raise ValueError("viewer screenshot requires a valid showcase candidate")
    snap_firefox = Path("/snap/firefox/current/usr/lib/firefox/firefox")
    firefox = str(snap_firefox) if snap_firefox.is_file() else shutil.which("firefox")
    if firefox is None:
        raise RuntimeError("Firefox is required to generate the real viewer screenshot")
    xephyr = shutil.which("Xephyr")
    if xephyr is None:
        raise RuntimeError("Xephyr is required to isolate the real viewer screenshot")
    with tempfile.TemporaryDirectory(prefix="da3-cad-viewer-") as temporary:
        run = Path(temporary) / "showcase-run"
        artefacts = run / "artefacts"
        artefacts.mkdir(parents=True)
        shutil.copy2(item.program_path, run / "model.py")
        shutil.copy2(item.model_path, run / "model.stl")
        step_path = item.model_path.with_suffix(".step")
        if step_path.is_file():
            shutil.copy2(step_path, run / "model.step")
        cloud_ply = item.cloud_path.with_suffix(".ply")
        if not cloud_ply.is_file():
            raise FileNotFoundError(f"viewer cloud PLY is missing: {cloud_ply}")
        shutil.copy2(cloud_ply, artefacts / "fused_cloud.ply")
        (run / "parameters.json").write_text(
            json.dumps(_viewer_metadata(item), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run / "quality.json").write_text(
            json.dumps(
                {"status": "valid", "backend": "cadrille-rl", "fallback_used": False},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "run_id": f"showcase-{item.dataset}-{item.item_id}-n08",
                    "command": "frozen Phase-D pilot",
                    "warnings": ["Rendered benchmark example; dimensions are not millimetres."],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        viewer = build_viewer(run, images_dir=Path(item.views[0]).parent)
        html = viewer.read_text(encoding="utf-8")
        display_name = f"frozen-showcase/{item.dataset}/{item.item_id}/n08"
        absolute_run = str(run.resolve())
        if absolute_run not in html:
            raise RuntimeError("viewer payload does not contain its absolute run path")
        html = html.replace(absolute_run, display_name)
        pointer_css = "canvas{width"
        if pointer_css not in html:
            raise RuntimeError("viewer template canvas style changed")
        html = html.replace(pointer_css, "canvas{pointer-events:none;width", 1)
        viewer.write_text(html, encoding="utf-8")
        profile = Path(temporary) / "firefox-profile"
        profile.mkdir()
        display_number = next(
            number for number in range(90, 100) if not Path(f"/tmp/.X11-unix/X{number}").exists()
        )
        display = f":{display_number}"
        xephyr_process = subprocess.Popen(
            [xephyr, display, "-screen", "1440x900", "-br", "-noreset", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        firefox_process: subprocess.Popen[bytes] | None = None
        try:
            socket = Path(f"/tmp/.X11-unix/X{display_number}")
            for _ in range(50):
                if socket.exists():
                    break
                if xephyr_process.poll() is not None:
                    raise RuntimeError("Xephyr exited before creating its display socket")
                time.sleep(0.1)
            else:
                raise RuntimeError("Xephyr did not create its display socket within five seconds")
            environment = os.environ.copy()
            environment["DISPLAY"] = display
            firefox_process = subprocess.Popen(
                [
                    firefox,
                    "--no-remote",
                    "--profile",
                    str(profile),
                    "--kiosk",
                    viewer.as_uri(),
                ],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(8.0)
            if firefox_process.poll() is not None:
                raise RuntimeError("Firefox exited before the viewer screenshot was captured")
            screenshot = ImageGrab.grab(xdisplay=display, all_screens=True).convert("RGB")
            pixels = np.asarray(screenshot)
            occupied = np.any(pixels != 0, axis=2)
            rows, columns = np.where(occupied)
            if len(rows) == 0 or len(columns) == 0:
                raise RuntimeError("viewer screenshot is an empty framebuffer")
            screenshot = screenshot.crop(
                (
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()) + 1,
                    int(rows.max()) + 1,
                )
            )
            screenshot.convert("RGB").save(output, format="PNG", optimize=True)
        finally:
            if firefox_process is not None and firefox_process.poll() is None:
                firefox_process.terminate()
                try:
                    firefox_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    firefox_process.kill()
                    firefox_process.wait(timeout=5.0)
            if xephyr_process.poll() is None:
                xephyr_process.terminate()
                try:
                    xephyr_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    xephyr_process.kill()
                    xephyr_process.wait(timeout=5.0)


def _dimensions(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        frames = int(getattr(image, "n_frames", 1))
        return image.width, image.height, frames


def _write_asset_manifest(
    path: Path,
    assets: tuple[Path, ...],
    source_manifest: Path,
    release_facts: Path,
) -> None:
    payload = {
        "schema_version": "1.0",
        "generator": "scripts/generate_readme_figures.py",
        "source_manifest": {
            "path": source_manifest.relative_to(ROOT).as_posix(),
            "sha256": _sha256(source_manifest),
        },
        "release_facts": {
            "path": release_facts.relative_to(ROOT).as_posix(),
            "sha256": _sha256(release_facts),
        },
        "assets": [
            {
                "path": asset.relative_to(ROOT).as_posix(),
                "sha256": _sha256(asset),
                "bytes": asset.stat().st_size,
                "width": _dimensions(asset)[0],
                "height": _dimensions(asset)[1],
                "frames": _dimensions(asset)[2],
            }
            for asset in assets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_assets(asset_manifest: Path) -> None:
    payload = _load_json(asset_manifest)
    source = ROOT / payload["source_manifest"]["path"]
    facts = ROOT / payload["release_facts"]["path"]
    if _sha256(source) != payload["source_manifest"]["sha256"]:
        raise ValueError("showcase source manifest changed; regenerate README figures")
    if _sha256(facts) != payload["release_facts"]["sha256"]:
        raise ValueError("release facts changed; regenerate README figures")
    for record in payload["assets"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"committed README asset is missing: {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"README asset SHA-256 mismatch: {path}")
        width, height, frames = _dimensions(path)
        observed = (path.stat().st_size, width, height, frames)
        expected = (record["bytes"], record["width"], record["height"], record["frames"])
        if observed != expected:
            raise ValueError(f"README asset metadata drift: {path}: {observed} != {expected}")


def generate(
    manifest_path: Path,
    release_facts: Path,
    output_dir: Path,
    asset_manifest: Path,
) -> tuple[Path, ...]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported showcase manifest schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = _resolve_items(manifest)
    by_role = {item.role: item for item in items}
    hero = by_role["success-fusion360"]
    assets: list[Path] = []
    hero_path = output_dir / "hero.gif"
    _save_hero(hero, manifest, hero_path)
    assets.append(hero_path)
    for item in items:
        path = output_dir / f"showcase_{item.role.replace('-', '_')}.png"
        _showcase_figure(item, manifest).convert("RGB").save(path, format="PNG", optimize=True)
        assets.append(path)
    decomposition = output_dir / "bottleneck_decomposition.png"
    _decomposition_figure(release_facts).convert("RGB").save(
        decomposition, format="PNG", optimize=True
    )
    assets.append(decomposition)
    viewer = output_dir / "viewer.png"
    _viewer_screenshot(hero, viewer)
    assets.append(viewer)
    resolved = tuple(assets)
    _write_asset_manifest(asset_manifest, resolved, manifest_path, release_facts)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--release-facts", type=Path, default=DEFAULT_RELEASE_FACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify committed assets and their source hashes without runtime benchmark data",
    )
    args = parser.parse_args()
    if args.verify:
        _verify_assets(args.asset_manifest.resolve())
        print(f"Verified README assets: {args.asset_manifest}")
        return 0
    assets = generate(
        args.manifest.resolve(),
        args.release_facts.resolve(),
        args.output_dir.resolve(),
        args.asset_manifest.resolve(),
    )
    for asset in assets:
        print(asset.relative_to(ROOT))
    print(args.asset_manifest.resolve().relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
