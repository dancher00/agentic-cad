#!/usr/bin/env python3
"""Run the bundled CPU smoke fixture and build its README animation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "sample_data" / "plate" / "views"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "release" / "cpu_smoke.gif"
WIDTH = 1200
HEIGHT = 540
BACKGROUND = (247, 244, 236)
CARD = (255, 254, 250)
INK = (24, 27, 32)
MUTED = (103, 115, 137)
BORDER = (211, 208, 199)
ORANGE = (242, 103, 49)
GREEN = (39, 145, 105)
BLUE = (63, 91, 142)


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(images: list[Image.Image], *, cell: int = 102) -> Image.Image:
    gap = 8
    output = Image.new("RGB", (cell * 2 + gap, cell * 2 + gap), CARD)
    for index, image in enumerate(images[:4]):
        thumbnail = image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        x = (index % 2) * (cell + gap)
        y = (index // 2) * (cell + gap)
        output.paste(thumbnail, (x, y))
    return output


def _depth_images(npz_path: Path) -> list[Image.Image]:
    with np.load(npz_path) as archive:
        depth = np.asarray(archive["depth"], dtype=np.float32)
    finite = np.isfinite(depth)
    values = depth[finite]
    if values.size == 0:
        raise ValueError("CPU smoke depth artifact contains no finite values")
    low = float(np.min(values))
    high = float(np.max(values))
    scale = max(high - low, 1e-6)
    result: list[Image.Image] = []
    for plane, valid in zip(depth, finite, strict=True):
        t = np.clip((plane - low) / scale, 0.0, 1.0)
        rgb = np.full((*plane.shape, 3), 250, dtype=np.uint8)
        rgb[..., 0][valid] = np.asarray(45 + 208 * t[valid], dtype=np.uint8)
        rgb[..., 1][valid] = np.asarray(
            82 + 91 * (1.0 - np.abs(2.0 * t[valid] - 1.0)), dtype=np.uint8
        )
        rgb[..., 2][valid] = np.asarray(180 - 120 * t[valid], dtype=np.uint8)
        result.append(Image.fromarray(rgb, mode="RGB"))
    return result


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    rz = np.asarray(
        (
            (math.cos(azimuth), -math.sin(azimuth), 0.0),
            (math.sin(azimuth), math.cos(azimuth), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    rx = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(elevation), -math.sin(elevation)),
            (0.0, math.sin(elevation), math.cos(elevation)),
        ),
        dtype=np.float64,
    )
    return rx @ rz


def _mesh_preview(stl_path: Path, *, size: int = 226) -> Image.Image:
    loaded = trimesh.load(stl_path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected one mesh: {stl_path}")

    supersample = 3
    canvas_size = size * supersample
    rotation = _rotation(32.0, 20.0)
    vertices = np.asarray(loaded.vertices, dtype=np.float64) @ rotation.T
    faces = np.asarray(loaded.faces, dtype=np.int64)
    projected = vertices[:, :2]
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = max(float(np.max(maximum - minimum)), 1e-9)
    pixels = (projected - center) * ((canvas_size * 0.78) / extent) + canvas_size / 2.0
    pixels[:, 1] = canvas_size - pixels[:, 1]

    image = Image.new("RGB", (canvas_size, canvas_size), CARD)
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (
            canvas_size * 0.16,
            canvas_size * 0.76,
            canvas_size * 0.84,
            canvas_size * 0.88,
        ),
        fill=(226, 222, 212),
    )
    triangles = vertices[faces]
    for face_index in np.argsort(triangles[..., 2].mean(axis=1)):
        triangle = triangles[face_index]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        facing = float(normal[2] / norm) if norm > 1e-12 else 0.0
        if facing <= 1e-6:
            continue
        shade = int(143 + 91 * facing)
        polygon = pixels[faces[face_index]].copy()
        centroid = polygon.mean(axis=0)
        polygon = centroid + (polygon - centroid) * 1.008
        coordinates = [tuple(float(value) for value in point) for point in polygon]
        draw.polygon(
            coordinates,
            fill=(min(shade + 15, 246), max(shade - 45, 70), max(shade - 76, 45)),
        )
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _parameters(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["name"]): float(item["value"]) for item in payload["primary_parameters"]}


def _draw_arrow(draw: ImageDraw.ImageDraw, left: int, right: int, y: int, *, active: bool) -> None:
    color = ORANGE if active else BORDER
    draw.line((left, y, right - 7, y), fill=color, width=4)
    draw.polygon(((right - 7, y - 7), (right, y), (right - 7, y + 7)), fill=color)


def _stage_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    number: str,
    title: str,
    active: bool,
    complete: bool,
) -> None:
    color = ORANGE if active else GREEN if complete else BORDER
    draw.rounded_rectangle(box, radius=14, fill=CARD, outline=color, width=4 if active else 2)
    x0, y0, x1, _ = box
    draw.text(
        (x0 + 18, y0 + 15),
        number,
        fill=ORANGE if complete or active else MUTED,
        font=_font(14, bold=True),
    )
    draw.text(
        (x0 + 18, y0 + 37),
        title,
        fill=INK if complete or active else MUTED,
        font=_font(19, bold=True),
    )
    if not complete and not active:
        draw.text((x0 + 18, y0 + 92), "waiting...", fill=BORDER, font=_font(16))
    if complete and not active:
        draw.ellipse((x1 - 36, y0 + 18, x1 - 18, y0 + 36), fill=GREEN)
        draw.text((x1 - 33, y0 + 16), "✓", fill=(255, 255, 255), font=_font(14, bold=True))


def _render_frame(
    *,
    revealed: int,
    inputs: Image.Image,
    depths: Image.Image,
    preview: Image.Image,
    parameters: dict[str, float],
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "CPU SMOKE: 4 PNG  →  model.step", fill=INK, font=_font(34, bold=True))
    draw.text(
        (30, 66),
        "offline · deterministic · no GPU · no weights · no network",
        fill=MUTED,
        font=_font(17),
    )
    draw.rounded_rectangle((1010, 24, 1170, 65), radius=20, fill=(231, 246, 238))
    draw.text((1033, 34), "< 60 s contract", fill=GREEN, font=_font(16, bold=True))

    margin = 28
    gap = 18
    card_width = 273
    top = 112
    bottom = 444
    boxes = [
        (
            margin + index * (card_width + gap),
            top,
            margin + index * (card_width + gap) + card_width,
            bottom,
        )
        for index in range(4)
    ]
    titles = ("RGB FIXTURE", "PIXEL STUB", "CAD PROGRAM", "VALID B-REP")
    for index, (box, title) in enumerate(zip(boxes, titles, strict=True), start=1):
        _stage_card(
            draw,
            box,
            number=f"0{index}",
            title=title,
            active=index == revealed,
            complete=index <= revealed,
        )
        if index < 4:
            _draw_arrow(
                draw,
                box[2] + 3,
                boxes[index][0] - 3,
                (top + bottom) // 2,
                active=index < revealed,
            )

    if revealed >= 1:
        image.paste(inputs, (boxes[0][0] + 28, top + 77))
        draw.text(
            (boxes[0][0] + 20, bottom - 28),
            "sample_data/plate/views",
            fill=MUTED,
            font=_font(13, mono=True),
        )
    if revealed >= 2:
        image.paste(depths, (boxes[1][0] + 28, top + 77))
        draw.text(
            (boxes[1][0] + 20, bottom - 46), "image-derived test signal", fill=MUTED, font=_font(13)
        )
        draw.text(
            (boxes[1][0] + 20, bottom - 27),
            "NOT DEPTH ANYTHING 3",
            fill=ORANGE,
            font=_font(13, bold=True),
        )
    if revealed >= 3:
        x = boxes[2][0] + 20
        y = top + 85
        draw.rounded_rectangle((x, y, boxes[2][2] - 20, y + 184), radius=9, fill=(244, 242, 236))
        code = (
            "rectangle(width, height)",
            "  + extrude(thickness)",
            "  - circle(hole_diameter)",
            "",
            f"width      {parameters['plate_width']:7.2f}",
            f"height     {parameters['plate_height']:7.2f}",
            f"thickness  {parameters['plate_thickness']:7.2f}",
            f"hole dia   {parameters['hole_diameter']:7.2f}",
        )
        for line_index, line in enumerate(code):
            draw.text(
                (x + 12, y + 10 + line_index * 21), line, fill=BLUE, font=_font(13, mono=True)
            )
        draw.text((x, bottom - 28), "editable model.py", fill=MUTED, font=_font(13, mono=True))
    if revealed >= 4:
        image.paste(preview, (boxes[3][0] + 24, top + 70))
        badge = (boxes[3][0] + 35, bottom - 48, boxes[3][2] - 35, bottom - 16)
        draw.rounded_rectangle(badge, radius=15, fill=GREEN)
        draw.text(
            (badge[0] + 23, badge[1] + 7),
            "1 VALID SOLID · STEP",
            fill=(255, 255, 255),
            font=_font(13, bold=True),
        )

    footer = (
        "SMOKE ONLY — verifies image I/O, program generation and OpenCascade export; "
        "not DA3 accuracy."
    )
    draw.text((30, 486), footer, fill=INK, font=_font(16, bold=True))
    draw.text(
        (30, 513),
        "The evaluator-only sample_data/plate/gt.stl is never read by this command.",
        fill=MUTED,
        font=_font(14),
    )
    return image


def build_gif(output: Path) -> float:
    with TemporaryDirectory(prefix="da3-cad-cpu-smoke-") as temporary:
        run_dir = Path(temporary) / "run"
        started = time.monotonic()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "da3_cad",
                "cpu-smoke",
                "--output",
                str(run_dir),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError("CPU smoke command failed:\n" + completed.stdout + completed.stderr)
        quality = json.loads((run_dir / "quality.json").read_text(encoding="utf-8"))
        if quality["status"] != "valid" or not (run_dir / "model.step").is_file():
            raise RuntimeError("CPU smoke command did not create a valid STEP")
        if elapsed >= 60.0:
            raise RuntimeError(f"CPU smoke exceeded its 60 second contract: {elapsed:.2f}s")

        inputs = _contact_sheet(
            [Image.open(path).convert("RGB") for path in sorted(DEFAULT_FIXTURE.glob("*.png"))]
        )
        depths = _contact_sheet(_depth_images(run_dir / "artefacts" / "stub_depth.npz"))
        preview = _mesh_preview(run_dir / "model.stl")
        parameters = _parameters(run_dir / "parameters.json")
        frames = [
            _render_frame(
                revealed=index,
                inputs=inputs,
                depths=depths,
                preview=preview,
                parameters=parameters,
            )
            for index in range(1, 5)
        ]
        palette_source = Image.new("RGB", (WIDTH, HEIGHT * len(frames)))
        for index, frame in enumerate(frames):
            palette_source.paste(frame, (0, index * HEIGHT))
        palette = palette_source.quantize(colors=128)
        quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        output.parent.mkdir(parents=True, exist_ok=True)
        quantized[0].save(
            output,
            save_all=True,
            append_images=quantized[1:],
            duration=(950, 1050, 1250, 2600),
            loop=0,
            optimize=True,
            disposal=2,
        )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    elapsed = build_gif(args.output.resolve())
    print(f"wrote {args.output} from a valid CPU smoke run in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
