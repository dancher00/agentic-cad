"""Build an offline HTML viewer without adding a web-framework dependency."""

from __future__ import annotations

import base64
import io
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageOps

from da3_cad.evaluation.mesh import TessellationConfig, load_mesh
from da3_cad.observations import IMAGE_EXTENSIONS
from da3_cad.pipeline import inspect_run

MAX_CLOUD_POINTS = 25_000
MAX_SOLID_FACES = 7_500
MAX_INPUT_PREVIEWS = 16


def _indices(length: int, maximum: int) -> npt.NDArray[np.int64]:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _cloud_path(run_dir: Path) -> Path | None:
    candidates = (
        run_dir / "artefacts/fused_cloud.ply",
        run_dir / "artefacts/geometry/artefacts/fused_cloud.ply",
    )
    return next((path for path in candidates if path.is_file()), None)


def _cloud_payload(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.points.PointCloud):
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        raw_colors = loaded.colors
    elif isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        raw_colors = getattr(loaded.visual, "vertex_colors", None)
    else:
        raise ValueError(f"viewer cloud is not a point/triangle payload: {path}")
    indices = _indices(len(vertices), MAX_CLOUD_POINTS)
    selected = vertices[indices]
    colors: list[list[int]] | None = None
    visual_colors = np.asarray(raw_colors) if raw_colors is not None else np.empty((0, 0))
    if visual_colors.ndim == 2 and len(visual_colors) == len(vertices):
        colors = visual_colors[indices, :3].astype(np.uint8).tolist()
    return {
        "source": path.name,
        "original_points": len(vertices),
        "displayed_points": len(selected),
        "points": selected.tolist(),
        "colors": colors,
    }


def _solid_payload(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    mesh = load_mesh(path, TessellationConfig())
    faces = np.asarray(mesh.faces, dtype=np.int64)
    chosen = faces[_indices(len(faces), MAX_SOLID_FACES)]
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[chosen]
    return {
        "source": path.name,
        "original_faces": len(faces),
        "displayed_faces": len(triangles),
        "triangles": triangles.reshape(-1, 9).tolist(),
    }


def _image_previews(images_dir: Path | None) -> list[dict[str, str]]:
    if images_dir is None:
        return []
    if not images_dir.is_dir():
        raise ValueError(f"viewer image directory does not exist: {images_dir}")
    paths = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )[:MAX_INPUT_PREVIEWS]
    previews: list[dict[str, str]] = []
    for path in paths:
        with Image.open(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((320, 240), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            preview.save(stream, format="JPEG", quality=80, optimize=True)
        previews.append(
            {
                "name": path.name,
                "data_url": "data:image/jpeg;base64,"
                + base64.b64encode(stream.getvalue()).decode("ascii"),
            }
        )
    return previews


def _relative_download(output: Path, path: Path) -> str | None:
    if not path.is_file():
        return None
    relative = Path(os.path.relpath(path, start=output.parent)).as_posix()
    return "/".join(quote(part) for part in relative.split("/"))


def viewer_payload(
    run_dir: Path,
    *,
    output: Path,
    images_dir: Path | None = None,
) -> dict[str, object]:
    inspection = inspect_run(run_dir)
    solid_path = next(
        (path for path in (run_dir / "model.stl", run_dir / "model.step") if path.is_file()),
        None,
    )
    metadata = cast(dict[str, Any], inspection["parameter_metadata"])
    return {
        "schema_version": "1.0",
        "run": str(run_dir.resolve()),
        "quality": inspection["quality"],
        "provenance": inspection["provenance"],
        "parameters": {
            "primary": metadata.get("primary_parameters", []),
            "implementation": metadata.get("implementation_parameters", []),
            "units": metadata.get("units"),
            "warnings": metadata.get("warnings", []),
        },
        "cloud": _cloud_payload(_cloud_path(run_dir)),
        "solid": _solid_payload(solid_path),
        "images": _image_previews(images_dir),
        "downloads": {
            "step": _relative_download(output, run_dir / "model.step"),
            "stl": _relative_download(output, run_dir / "model.stl"),
            "parameters": _relative_download(output, run_dir / "parameters.json"),
            "provenance": _relative_download(output, run_dir / "provenance.json"),
        },
    }


def build_viewer(
    run_dir: Path,
    *,
    output: Path | None = None,
    images_dir: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    destination = (output if output is not None else run_dir / "viewer.html").resolve()
    if destination.exists():
        raise ValueError(f"viewer output already exists: {destination}")
    payload = viewer_payload(run_dir, output=destination, images_dir=images_dir)
    template = files("da3_cad.viewer").joinpath("template.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__DA3_CAD_VIEWER_PAYLOAD__", encoded)
    if "__DA3_CAD_VIEWER_PAYLOAD__" in html:
        raise RuntimeError("viewer template payload placeholder was not replaced")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination
