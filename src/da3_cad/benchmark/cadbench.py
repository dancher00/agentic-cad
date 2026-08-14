"""Input-only adapter for the official CADBench image modalities.

Reference geometry deliberately does not appear in this module's API.  The
reconstruction phase consumes only the directory returned by
``prepare_image_case``; a separate evaluator may open the benchmark STL after
reconstruction has completed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

CADBENCH_REPOSITORY = "https://github.com/anniedoris/CADBench"
CADBENCH_DATASET = "DeCoDELab/CADBench"
CADBENCH_COMMIT = "99e41a2eeb351f04611e83980f9f23cb0ca216c7"

CADBenchModality = Literal["singleview", "multiview", "pbr"]

_IMAGE_SIZE = 1200
_MULTIVIEW_GUTTER = 10
_MULTIVIEW_SIZE = 2 * _IMAGE_SIZE + _MULTIVIEW_GUTTER
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class PreparedCADBenchCase:
    """A reference-free reconstruction input prepared from CADBench images."""

    file_id: str
    split: str
    modality: CADBenchModality
    input_dir: Path
    images: tuple[Path, ...]
    manifest_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_component(value: str, label: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe CADBench {label}: {value!r}")


def _source_image(
    dataset_root: Path,
    split: str,
    file_id: str,
    modality: CADBenchModality,
) -> Path:
    _validate_component(split, "split")
    _validate_component(file_id, "file_id")
    source = dataset_root / modality / split / f"{file_id}.png"
    if not source.is_file():
        raise FileNotFoundError(f"CADBench image not found: {source}")
    return source


def _write_multiview_tiles(source: Path, input_dir: Path) -> tuple[Path, ...]:
    with Image.open(source) as montage:
        if montage.size != (_MULTIVIEW_SIZE, _MULTIVIEW_SIZE):
            raise ValueError(
                "official CADBench multiview montage must be 2410x2410 "
                f"(four 1200x1200 tiles with a 10 px gutter), got {montage.size}"
            )
        rgb = montage.convert("RGB")
        offsets = (
            (0, 0),
            (_IMAGE_SIZE + _MULTIVIEW_GUTTER, 0),
            (0, _IMAGE_SIZE + _MULTIVIEW_GUTTER),
            (_IMAGE_SIZE + _MULTIVIEW_GUTTER, _IMAGE_SIZE + _MULTIVIEW_GUTTER),
        )
        paths: list[Path] = []
        for index, (left, top) in enumerate(offsets):
            path = input_dir / f"view_{index:02d}.png"
            rgb.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE)).save(path)
            paths.append(path)
    return tuple(paths)


def _write_single_image(source: Path, input_dir: Path) -> tuple[Path, ...]:
    path = input_dir / "view_00.png"
    with Image.open(source) as image:
        if image.size != (_IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError(
                f"official CADBench single-view image must be 1200x1200, got {image.size}"
            )
        image.convert("RGB").save(path)
    return (path,)


def prepare_image_case(
    dataset_root: Path,
    split: str,
    file_id: str,
    destination: Path,
    *,
    modality: CADBenchModality = "multiview",
) -> PreparedCADBenchCase:
    """Create a new input-only case without copying or naming reference CAD."""

    if destination.exists():
        raise FileExistsError(f"CADBench case destination already exists: {destination}")
    source = _source_image(dataset_root, split, file_id, modality)
    input_dir = destination / "images"
    input_dir.mkdir(parents=True)
    images = (
        _write_multiview_tiles(source, input_dir)
        if modality == "multiview"
        else _write_single_image(source, input_dir)
    )
    image_records: list[dict[str, str | int]] = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
            "width": _IMAGE_SIZE,
            "height": _IMAGE_SIZE,
        }
        for path in images
    ]
    manifest = {
        "schema_version": "da3-cad-cadbench-input-v1",
        "benchmark": {
            "repository": CADBENCH_REPOSITORY,
            "dataset": CADBENCH_DATASET,
            "source_commit": CADBENCH_COMMIT,
            "split": split,
            "file_id": file_id,
            "modality": modality,
        },
        "protocol": {
            "reconstruction_inputs": "RGB images only",
            "multiview_layout": (
                "2x2 row-major 1200px tiles separated by a 10px gutter"
                if modality == "multiview"
                else None
            ),
            "camera_intrinsics_or_extrinsics_supplied": False,
            "dataset_masks_supplied": False,
            "reference_geometry_supplied": False,
        },
        "images": image_records,
        "input_digest": hashlib.sha256(
            "".join(str(record["sha256"]) for record in image_records).encode()
        ).hexdigest(),
        "claim_boundary": {
            "reference_cad_available_to_reconstruction": False,
            "reference_geometry_opened_after_reconstruction_only": True,
            "small_slice_is_leaderboard_comparable": False,
        },
    }
    manifest_path = destination / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return PreparedCADBenchCase(
        file_id=file_id,
        split=split,
        modality=modality,
        input_dir=input_dir,
        images=images,
        manifest_path=manifest_path,
    )


def cadquery_submission_row(file_id: str, run_dir: Path) -> dict[str, str]:
    """Load a successful DA3-CAD run as one official CADBench JSONL row."""

    _validate_component(file_id, "file_id")
    program_path = run_dir / "model.py"
    step_path = run_dir / "model.step"
    if not program_path.is_file() or not step_path.is_file():
        raise FileNotFoundError(f"successful DA3-CAD artifacts not found in {run_dir}")
    generated = program_path.read_text(encoding="utf-8")
    tree = ast.parse(generated, filename=str(program_path))
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    if "r" not in assigned_names:
        raise ValueError("DA3-CAD model.py does not assign the expected final solid 'r'")
    return {"file_id": file_id, "generated": generated}


def write_submission(rows: list[dict[str, str]], path: Path) -> None:
    """Write deterministic official-format CADBench JSONL."""

    ids = [row["file_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate CADBench file_id in submission")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in sorted(rows, key=lambda r: r["file_id"])
    )
    path.write_text(payload, encoding="utf-8")
