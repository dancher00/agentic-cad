"""Deterministic four-view image inputs matching Cadrille's released protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from da3_cad.models import BoolArray, UInt8Array

CADRILLE_IMAGE_TILE_SIZE = 128
CADRILLE_IMAGE_BORDER = 3
CADRILLE_IMAGE_VIEW_COUNT = 4


@dataclass(frozen=True, slots=True)
class CadrilleImageInput:
    collage: UInt8Array
    view_indices: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        expected_side = 2 * (CADRILLE_IMAGE_TILE_SIZE + 2 * CADRILLE_IMAGE_BORDER)
        if self.collage.shape != (expected_side, expected_side, 3):
            raise ValueError(
                f"Cadrille image collage must have shape ({expected_side},{expected_side},3)"
            )
        if self.collage.dtype != np.uint8:
            raise ValueError("Cadrille image collage must use uint8 RGB")
        if len(set(self.view_indices)) != CADRILLE_IMAGE_VIEW_COUNT:
            raise ValueError("Cadrille image input requires four distinct source views")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.collage.tobytes(order="C")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "modality": "image-four-view-collage",
            "view_indices": list(self.view_indices),
            "shape": list(self.collage.shape),
            "dtype": "uint8",
            "sha256": self.sha256,
            "background": "white outside reconstruction input mask",
            "layout": (
                "four 128x128 white-background crops with 3px black borders "
                "in a 2x2 collage"
            ),
        }


def _masked_tile(image: UInt8Array, mask: BoolArray) -> Image.Image:
    pixels = np.asarray(image, dtype=np.uint8)
    selected = np.asarray(mask, dtype=np.bool_)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or selected.shape != pixels.shape[:2]:
        raise ValueError("Cadrille image tile requires matching HxWx3 RGB and HxW mask")
    ys, xs = np.nonzero(selected)
    if len(xs) == 0:
        raise ValueError("Cadrille image tile mask is empty")
    object_extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    margin = max(4, int(round(0.08 * object_extent)))
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(pixels.shape[1], int(xs.max()) + 1 + margin)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(pixels.shape[0], int(ys.max()) + 1 + margin)

    local_mask = selected[y0:y1, x0:x1]
    # The released Cadrille image dataset renders CAD geometry over white and
    # adds only the explicit 3 px tile border in black. Keeping the masked
    # background white is therefore part of the decoder input contract, not a
    # cosmetic choice.
    crop = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
    crop[local_mask] = pixels[y0:y1, x0:x1][local_mask]
    tile = Image.fromarray(crop, mode="RGB")
    side = max(tile.size)
    square = Image.new("RGB", (side, side), "white")
    square.paste(tile, ((side - tile.width) // 2, (side - tile.height) // 2))
    resized = square.resize(
        (CADRILLE_IMAGE_TILE_SIZE, CADRILLE_IMAGE_TILE_SIZE),
        Image.Resampling.LANCZOS,
    )
    return ImageOps.expand(resized, border=CADRILLE_IMAGE_BORDER, fill="black")


def _view_indices(view_count: int, candidate_index: int, candidate_count: int) -> tuple[int, ...]:
    if view_count < CADRILLE_IMAGE_VIEW_COUNT:
        raise ValueError("Cadrille image candidates require at least four input views")
    offset = int(np.floor(candidate_index * view_count / (4 * candidate_count)))
    values = tuple(
        int(np.floor(offset + slot * view_count / CADRILLE_IMAGE_VIEW_COUNT)) % view_count
        for slot in range(CADRILLE_IMAGE_VIEW_COUNT)
    )
    if len(set(values)) != CADRILLE_IMAGE_VIEW_COUNT:
        raise ValueError("Cadrille image view schedule did not produce four distinct views")
    return values


def build_cadrille_image_inputs(
    images: tuple[UInt8Array, ...],
    masks: BoolArray,
    *,
    candidate_count: int,
) -> tuple[CadrilleImageInput, ...]:
    """Build 2x2 masked collages from temporally offset, evenly spaced views."""

    if not 1 <= candidate_count <= 4:
        raise ValueError("Cadrille image candidate count must be in [1,4]")
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.ndim != 3 or len(images) != mask_values.shape[0]:
        raise ValueError("Cadrille image inputs require one mask per RGB view")
    outputs: list[CadrilleImageInput] = []
    for candidate_index in range(candidate_count):
        indices = _view_indices(len(images), candidate_index, candidate_count)
        tiles = [_masked_tile(images[index], mask_values[index]) for index in indices]
        width, height = tiles[0].size
        collage = Image.new("RGB", (2 * width, 2 * height), "black")
        positions = ((0, 0), (width, 0), (0, height), (width, height))
        for tile, position in zip(tiles, positions, strict=True):
            collage.paste(tile, position)
        outputs.append(
            CadrilleImageInput(
                collage=np.asarray(collage, dtype=np.uint8).copy(),
                view_indices=(indices[0], indices[1], indices[2], indices[3]),
            )
        )
    return tuple(outputs)


def write_cadrille_image_inputs(
    output_dir: Path,
    inputs: tuple[CadrilleImageInput, ...],
) -> None:
    """Persist exact collages and their source-view provenance."""

    if not inputs:
        raise ValueError("cannot write an empty Cadrille image input set")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Cadrille image input directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(inputs):
        Image.fromarray(item.collage, mode="RGB").save(output_dir / f"candidate_{index:02d}.png")
    payload = {
        "schema_version": "1.0",
        "ground_truth_access": False,
        "inputs": [item.as_dict() for item in inputs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
