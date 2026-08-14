"""Prepare an explicitly selected object before DA3 inference.

The target mask is the geometric contract.  It determines a target-centred
crop and is preserved for the later masked-fusion stage.  Every view uses the
same crop dimensions, so DA3 does not silently centre-crop a mixed-shape
batch.  Cropping never changes camera extrinsics; supplied intrinsics are
translated into the crop's pixel coordinate system.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
from da3_cad.models import BoolArray, FloatArray, ImageObservation, UInt8Array
from da3_cad.observations import load_observations, load_rgb

TargetSelectionSource = Literal[
    "user-mask",
    "robot-mask",
    "dataset-mask-oracle",
    "sam2-box",
    "sam2-point",
    "sam2-track",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CropWindow:
    """Rectangular crop in EXIF-corrected source pixel coordinates.

    Coordinates are half-open.  Negative origins or endpoints beyond the
    source image denote neutral padding, not invented object support.
    """

    x0: int
    y0: int
    width: int
    height: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("crop and source dimensions must be positive")

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)

    @property
    def source_to_crop(self) -> FloatArray:
        return np.asarray(
            ((1.0, 0.0, -float(self.x0)), (0.0, 1.0, -float(self.y0)), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )

    @property
    def padding(self) -> tuple[int, int, int, int]:
        x1 = self.x0 + self.width
        y1 = self.y0 + self.height
        return (
            max(0, -self.x0),
            max(0, -self.y0),
            max(0, x1 - self.source_width),
            max(0, y1 - self.source_height),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "xyxy": list(self.xyxy),
            "width": self.width,
            "height": self.height,
            "padding_left_top_right_bottom": list(self.padding),
            "source_to_crop": self.source_to_crop.tolist(),
        }


@dataclass(frozen=True, slots=True)
class TargetPreparationResult:
    root: Path
    images_dir: Path
    masks_dir: Path
    overlays_dir: Path
    manifest_path: Path
    camera_bundle_path: Path | None
    report: dict[str, object]


def match_explicit_masks(
    observations: tuple[ImageObservation, ...],
    mask_dir: Path,
) -> tuple[Path, ...]:
    """Match one PNG mask to every input image by an unambiguous stem."""

    if not mask_dir.is_dir():
        raise ValueError(f"mask directory does not exist: {mask_dir}")
    stems = [Path(item.relative_path).stem for item in observations]
    if len(set(stems)) != len(stems):
        raise ValueError("input image stems must be unique before matching explicit masks")
    available: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*.png"), key=lambda item: item.name.casefold()):
        key = path.stem
        if key in available:
            raise ValueError(f"duplicate explicit-mask stem: {key}")
        available[key] = path
    missing = [stem for stem in stems if stem not in available]
    if missing:
        raise ValueError(
            f"explicit masks must use the input image stems with PNG extension; missing={missing}"
        )
    return tuple(available[stem] for stem in stems)


def load_source_mask(path: Path, *, width: int, height: int) -> BoolArray:
    """Load a mask defined in the EXIF-corrected source image coordinate space."""

    with Image.open(path) as image:
        corrected = ImageOps.exif_transpose(image).convert("L")
        if corrected.size != (width, height):
            raise ValueError(
                f"mask resolution must exactly match the EXIF-corrected source image: "
                f"{path} is {corrected.width}x{corrected.height}, expected {width}x{height}"
            )
        mask = np.asarray(corrected, dtype=np.uint8) > 0
    if not np.any(mask):
        raise ValueError(f"target mask is empty: {path}")
    return mask


def mask_bbox_xyxy(mask: BoolArray) -> tuple[int, int, int, int]:
    """Return the tight half-open XYXY box containing every selected component."""

    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("target mask must be a non-empty 2D array")
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _axis_origin(center: float, side: int, source_extent: int) -> int:
    proposed = int(math.floor(center - side / 2.0))
    if side <= source_extent:
        return min(max(proposed, 0), source_extent - side)
    # Keep the complete source in view when a square necessarily needs padding.
    return min(max(proposed, source_extent - side), 0)


def square_crop_for_mask(mask: BoolArray, *, margin_fraction: float = 0.15) -> CropWindow:
    """Find a square crop containing one complete mask.

    Kept as a small public geometry primitive.  Dataset preparation uses
    :func:`common_crop_windows_for_masks` because independent square crops can
    create large artificial borders around tall or wide objects.
    """

    if not 0.0 <= margin_fraction <= 1.0:
        raise ValueError("crop margin fraction must be in [0,1]")
    height, width = mask.shape
    x0, y0, x1, y1 = mask_bbox_xyxy(mask)
    object_width = x1 - x0
    object_height = y1 - y0
    side = max(1, int(math.ceil(max(object_width, object_height) * (1.0 + 2.0 * margin_fraction))))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    crop_x0 = _axis_origin(center_x, side, width)
    crop_y0 = _axis_origin(center_y, side, height)
    crop = CropWindow(crop_x0, crop_y0, side, side, width, height)
    crop_x1, crop_y1 = crop_x0 + side, crop_y0 + side
    if crop_x0 > x0 or crop_y0 > y0 or crop_x1 < x1 or crop_y1 < y1:
        raise RuntimeError("computed crop does not contain the complete target mask")
    return crop


def common_crop_windows_for_masks(
    masks: list[BoolArray], *, margin_fraction: float = 0.15
) -> tuple[CropWindow, ...]:
    """Build target-centred crops with one shared pixel shape for every view.

    The global width and height are the largest expanded mask extents in the
    sequence.  This retains genuine projective scale differences between
    views, avoids per-image anisotropic resizing, and prevents DA3's batch
    unifier from discarding different image borders.
    """

    if not masks:
        raise ValueError("at least one target mask is required")
    if not 0.0 <= margin_fraction <= 1.0:
        raise ValueError("crop margin fraction must be in [0,1]")
    bboxes = [mask_bbox_xyxy(mask) for mask in masks]
    scale = 1.0 + 2.0 * margin_fraction
    required_width = max(x1 - x0 for x0, _, x1, _ in bboxes)
    required_height = max(y1 - y0 for _, y0, _, y1 in bboxes)
    desired_width = int(math.ceil(required_width * scale))
    desired_height = int(math.ceil(required_height * scale))
    minimum_source_width = min(mask.shape[1] for mask in masks)
    minimum_source_height = min(mask.shape[0] for mask in masks)
    # Margin is context, not geometry.  Reduce it before inventing pixels when
    # every target already fits inside the smallest source frame.
    common_width = min(desired_width, minimum_source_width)
    common_height = min(desired_height, minimum_source_height)
    if required_width > common_width:
        common_width = required_width
    if required_height > common_height:
        common_height = required_height
    # DA3 resizes the longest side to 504 px.  Extremely thin crops destroy
    # resolution on the short axis and remove context used by its pose head.
    # Expand context (never the mask) until the crop is at least 3:4 in either
    # orientation, bounded by pixels that genuinely exist in every source.
    minimum_short_to_long_ratio = 0.75
    if common_width < common_height * minimum_short_to_long_ratio:
        common_width = min(
            minimum_source_width,
            int(math.ceil(common_height * minimum_short_to_long_ratio)),
        )
    if common_height < common_width * minimum_short_to_long_ratio:
        common_height = min(
            minimum_source_height,
            int(math.ceil(common_width * minimum_short_to_long_ratio)),
        )

    crops: list[CropWindow] = []
    for mask, (x0, y0, x1, y1) in zip(masks, bboxes, strict=True):
        source_height, source_width = mask.shape
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        crop_x0 = _axis_origin(center_x, common_width, source_width)
        crop_y0 = _axis_origin(center_y, common_height, source_height)
        crop = CropWindow(
            crop_x0,
            crop_y0,
            common_width,
            common_height,
            source_width,
            source_height,
        )
        crop_x1, crop_y1 = crop.xyxy[2:]
        if crop_x0 > x0 or crop_y0 > y0 or crop_x1 < x1 or crop_y1 < y1:
            raise RuntimeError("computed common crop does not contain the complete target mask")
        crops.append(crop)
    return tuple(crops)


def _crop_with_padding(
    values: npt.NDArray[np.generic],
    crop: CropWindow,
    *,
    mask: bool,
) -> npt.NDArray[np.generic]:
    x0, y0, x1, y1 = crop.xyxy
    source = values[
        max(y0, 0) : min(y1, crop.source_height), max(x0, 0) : min(x1, crop.source_width)
    ]
    left, top, right, bottom = crop.padding
    pad_width: tuple[tuple[int, int], ...]
    if values.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
    if mask:
        result = np.pad(source, pad_width, mode="constant", constant_values=False)
    else:
        # Repeating edge pixels creates strong fake planes in DA3 depth.  A
        # flat mid-grey border is explicit missing context and carries no
        # duplicated texture or false silhouette support.
        result = np.pad(source, pad_width, mode="constant", constant_values=127)
    if result.shape[:2] != (crop.height, crop.width):
        raise RuntimeError(f"crop produced unexpected shape: {result.shape}")
    return result


def crop_rgb(rgb: UInt8Array, crop: CropWindow) -> UInt8Array:
    if rgb.shape != (crop.source_height, crop.source_width, 3):
        raise ValueError("RGB dimensions do not match the crop source dimensions")
    return np.asarray(_crop_with_padding(rgb, crop, mask=False), dtype=np.uint8)


def crop_mask(mask: BoolArray, crop: CropWindow) -> BoolArray:
    if mask.shape != (crop.source_height, crop.source_width):
        raise ValueError("mask dimensions do not match the crop source dimensions")
    return np.asarray(_crop_with_padding(mask, crop, mask=True), dtype=np.bool_)


def adjust_intrinsics_for_crop(intrinsics: FloatArray, crop: CropWindow) -> FloatArray:
    """Translate K into crop pixels; DA3 performs any later resize itself."""

    matrix = np.asarray(intrinsics, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera intrinsics must be one finite 3x3 matrix")
    adjusted = crop.source_to_crop @ matrix
    return np.asarray(adjusted, dtype=np.float32)


def _mask_warnings(mask: BoolArray) -> list[str]:
    warnings: list[str] = []
    area_fraction = float(mask.mean())
    if area_fraction < 0.001:
        warnings.append("target mask covers less than 0.1% of the source image")
    if area_fraction > 0.98:
        warnings.append("target mask covers more than 98% of the source image")
    if bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
        warnings.append("target mask touches a source boundary; visible geometry may be truncated")
    return warnings


def _overlay(rgb: UInt8Array, mask: BoolArray) -> UInt8Array:
    result = rgb.astype(np.float32)
    tint = np.zeros_like(result)
    tint[..., 0] = 255.0
    result[mask] = 0.58 * result[mask] + 0.42 * tint[mask]
    return np.asarray(np.clip(result, 0.0, 255.0), dtype=np.uint8)


def prepare_target_from_masks(
    input_dir: Path,
    output_dir: Path,
    mask_dir: Path,
    *,
    camera_bundle_path: Path | None = None,
    margin_fraction: float = 0.15,
    selection_source: TargetSelectionSource = "user-mask",
) -> TargetPreparationResult:
    """Write lossless target crops, matching masks, optional cameras and evidence."""

    allowed_sources = {
        "user-mask",
        "robot-mask",
        "dataset-mask-oracle",
        "sam2-box",
        "sam2-point",
        "sam2-track",
    }
    if selection_source not in allowed_sources:
        raise ValueError(f"unsupported target selection source: {selection_source}")
    if not 0.0 <= margin_fraction <= 1.0:
        raise ValueError("crop margin fraction must be in [0,1]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"target output directory is not empty: {output_dir}")

    observations = load_observations(input_dir)
    mask_paths = match_explicit_masks(observations.images, mask_dir)
    masks: list[BoolArray] = []
    output_names: list[str] = []
    for observation, mask_path in zip(observations.images, mask_paths, strict=True):
        mask = load_source_mask(
            mask_path,
            width=observation.width,
            height=observation.height,
        )
        masks.append(mask)
        output_names.append(f"{Path(observation.relative_path).stem}.png")
    crops = list(common_crop_windows_for_masks(masks, margin_fraction=margin_fraction))
    if len(set(output_names)) != len(output_names):
        raise ValueError("lossless target PNG names would collide")

    camera_bundle = (
        load_camera_bundle(camera_bundle_path, observations)
        if camera_bundle_path is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    images_dir.mkdir(exist_ok=False)
    masks_dir.mkdir(exist_ok=False)
    overlays_dir.mkdir(exist_ok=False)

    view_reports: list[dict[str, object]] = []
    for observation, source_mask_path, source_mask, crop, output_name in zip(
        observations.images,
        mask_paths,
        masks,
        crops,
        output_names,
        strict=True,
    ):
        rgb = load_rgb(observation.path)
        prepared_rgb = crop_rgb(rgb, crop)
        prepared_mask = crop_mask(source_mask, crop)
        image_output = images_dir / output_name
        mask_output = masks_dir / output_name
        overlay_output = overlays_dir / output_name
        Image.fromarray(prepared_rgb, mode="RGB").save(image_output)
        Image.fromarray(np.where(prepared_mask, 255, 0).astype(np.uint8), mode="L").save(
            mask_output
        )
        Image.fromarray(_overlay(prepared_rgb, prepared_mask), mode="RGB").save(overlay_output)
        warnings = _mask_warnings(source_mask)
        if any(crop.padding):
            warnings.append("common crop uses neutral padding outside the source image")
        view_reports.append(
            {
                "source_image": observation.relative_path,
                "source_image_sha256": observation.sha256,
                "source_resolution": [observation.width, observation.height],
                "source_mask": source_mask_path.name,
                "source_mask_sha256": _sha256(source_mask_path),
                "output_image": output_name,
                "output_image_sha256": _sha256(image_output),
                "output_mask_sha256": _sha256(mask_output),
                "mask_bbox_xyxy": list(mask_bbox_xyxy(source_mask)),
                "mask_area_pixels": int(source_mask.sum()),
                "mask_area_fraction": float(source_mask.mean()),
                "crop": crop.as_dict(),
                "warnings": warnings,
            }
        )

    adjusted_camera_path: Path | None = None
    camera_report: dict[str, object]
    if camera_bundle is None:
        camera_report = {
            "status": "not-supplied",
            "contract": "DA3 estimates intrinsics/extrinsics directly on prepared crops",
        }
    else:
        assert camera_bundle_path is not None
        adjusted_intrinsics = np.stack(
            [
                adjust_intrinsics_for_crop(camera_bundle.intrinsics[index], crop)
                for index, crop in enumerate(crops)
            ],
            axis=0,
        )
        adjusted_bundle = CameraBundle(
            image_names=tuple(output_names),
            intrinsics=adjusted_intrinsics,
            extrinsics=camera_bundle.extrinsics.astype(np.float32, copy=True),
            source=f"{camera_bundle.source}+target-crop-v1",
            scale_status=camera_bundle.scale_status,
            world_units=camera_bundle.world_units,
            world_units_to_mm=camera_bundle.world_units_to_mm,
            details={
                "parent": camera_bundle.details or {},
                "operation": "K_crop = source_to_crop @ K_source; extrinsics unchanged",
                "source_camera_bundle": str(camera_bundle_path),
            },
        )
        adjusted_camera_path = output_dir / "cameras.npz"
        adjusted_bundle.save(adjusted_camera_path)
        camera_report = {
            "status": "adjusted",
            "path": adjusted_camera_path.name,
            "sha256": _sha256(adjusted_camera_path),
            "source": str(camera_bundle_path),
            "source_sha256": _sha256(camera_bundle_path),
            "intrinsics_transform": "K_crop = source_to_crop @ K_source",
            "extrinsics_transform": "unchanged",
        }

    report: dict[str, object] = {
        "schema_version": "da3-cad-target-preparation-v2",
        "selection_source": selection_source,
        "input_root": str(input_dir.resolve()),
        "input_digest": observations.digest,
        "view_count": len(observations.images),
        "coordinate_space": "EXIF-corrected source pixels",
        "crop_policy": {
            "shape": "shared-width-height-target-centred",
            "output_width": crops[0].width,
            "output_height": crops[0].height,
            "margin_fraction_per_side_requested": margin_fraction,
            "margin_policy": "best-effort-clamped-before-padding",
            "minimum_short_to_long_ratio": 0.75,
            "context_contract": "expand crop with real source pixels for DA3 pose "
            "and short-axis resolution",
            "rgb_padding": "constant-127-only-when-unavoidable",
            "mask_padding": "zero",
            "resize": None,
            "projective_scale_between_views": "preserved",
        },
        "mask_contract": (
            "all nonzero pixels are selected; every selected component is preserved; "
            "the same transformed mask must be passed to reconstruction"
        ),
        "cameras": camera_report,
        "views": view_reports,
        "next_command": (
            "da3-cad reconstruct images/ --masks masks/ "
            + ("--cameras cameras.npz " if adjusted_camera_path is not None else "")
            + "--config configs/internet_photo_masked.yaml ..."
        ),
    }
    manifest_path = output_dir / "target.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TargetPreparationResult(
        root=output_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        overlays_dir=overlays_dir,
        manifest_path=manifest_path,
        camera_bundle_path=adjusted_camera_path,
        report=report,
    )
