"""External camera bundles and deterministic COLMAP recovery for video capture."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from da3_cad.geometry.fusion import ScaleChannel
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import FloatArray, ObservationSet

CameraScaleStatus = Literal["unresolved", "known"]


@dataclass(frozen=True, slots=True)
class CameraBundle:
    image_names: tuple[str, ...]
    intrinsics: FloatArray
    extrinsics: FloatArray
    source: str
    scale_status: CameraScaleStatus = "unresolved"
    world_units: str = "arbitrary-colmap-unit"
    world_units_to_mm: float | None = None
    details: dict[str, object] | None = None

    def __post_init__(self) -> None:
        count = len(self.image_names)
        intrinsics = np.asarray(self.intrinsics)
        extrinsics = np.asarray(self.extrinsics)
        if self.scale_status not in {"unresolved", "known"}:
            raise ValueError(f"unsupported camera scale status: {self.scale_status!r}")
        if not self.source.strip() or not self.world_units.strip():
            raise ValueError("camera source and world units must be non-empty")
        if count < 3 or len(set(self.image_names)) != count:
            raise ValueError("camera bundle requires at least three uniquely named images")
        if intrinsics.shape != (count, 3, 3):
            raise ValueError(f"camera intrinsics must have shape ({count},3,3)")
        if extrinsics.shape != (count, 4, 4):
            raise ValueError(f"camera extrinsics must have shape ({count},4,4)")
        if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
            raise ValueError("camera bundle arrays must be finite")
        for matrix in extrinsics:
            homogeneous = as_homogeneous_extrinsic(matrix)
            rotation = homogeneous[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
                raise ValueError("camera extrinsic rotation blocks must be orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
                raise ValueError("camera extrinsic rotations must have determinant +1")
        if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(
            intrinsics[:, 2, :],
            np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), (count, 3)),
            atol=1e-6,
        ):
            raise ValueError("camera intrinsics must have homogeneous bottom row [0,0,1]")
        if self.scale_status == "known":
            if self.world_units_to_mm is None or self.world_units_to_mm <= 0.0:
                raise ValueError("known camera scale requires positive world_units_to_mm")
        elif self.world_units_to_mm is not None:
            raise ValueError("unresolved camera scale cannot carry world_units_to_mm")

    @property
    def scale(self) -> ScaleChannel:
        return ScaleChannel(
            status=self.scale_status,
            units=self.world_units,
            world_units_to_mm=self.world_units_to_mm,
            source=self.source,
            evidence=self.details or {},
        )

    def as_dict(self) -> dict[str, object]:
        rotations = self.extrinsics[:, :3, :3]
        translations = self.extrinsics[:, :3, 3]
        centers = -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)
        centered = centers - centers.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        return {
            "schema_version": "1.0",
            "source": self.source,
            "image_names": list(self.image_names),
            "count": len(self.image_names),
            "intrinsics_shape": list(self.intrinsics.shape),
            "extrinsics_shape": list(self.extrinsics.shape),
            "extrinsic_convention": "world_to_camera",
            "camera_center_rank": int(np.linalg.matrix_rank(centered, tol=1e-8)),
            "camera_center_singular_values": singular.tolist(),
            "scale": {
                "status": self.scale_status,
                "world_units": self.world_units,
                "world_units_to_mm": self.world_units_to_mm,
            },
            "details": self.details or {},
        }

    def reordered(self, names: tuple[str, ...]) -> CameraBundle:
        lookup = {name: index for index, name in enumerate(self.image_names)}
        missing = [name for name in names if name not in lookup]
        extra = [name for name in self.image_names if name not in set(names)]
        if missing or extra:
            raise ValueError(
                "camera bundle and observation names differ: "
                f"missing={missing or 'none'}, extra={extra or 'none'}"
            )
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
        return CameraBundle(
            image_names=names,
            intrinsics=self.intrinsics[indices].astype(np.float32, copy=True),
            extrinsics=self.extrinsics[indices].astype(np.float32, copy=True),
            source=self.source,
            scale_status=self.scale_status,
            world_units=self.world_units,
            world_units_to_mm=self.world_units_to_mm,
            details=self.details,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.as_dict()
        np.savez_compressed(
            path,
            image_names=np.asarray(self.image_names, dtype=np.str_),
            intrinsics=np.asarray(self.intrinsics, dtype=np.float32),
            extrinsics=np.asarray(self.extrinsics, dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_camera_bundle(path: Path, observations: ObservationSet) -> CameraBundle:
    """Load a non-pickled NPZ bundle and order it exactly like the observations."""

    with np.load(path, allow_pickle=False) as payload:
        required = {"image_names", "intrinsics", "extrinsics", "metadata"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"camera bundle is missing arrays: {sorted(missing)}")
        metadata_value = np.asarray(payload["metadata"])
        if metadata_value.shape != ():
            raise ValueError("camera bundle metadata must be a scalar JSON string")
        metadata = json.loads(str(metadata_value.item()))
        if not isinstance(metadata, dict):
            raise ValueError("camera bundle metadata root must be an object")
        scale = metadata.get("scale", {})
        if not isinstance(scale, dict):
            raise ValueError("camera bundle scale metadata must be an object")
        status = scale.get("status", "unresolved")
        if status not in {"unresolved", "known"}:
            raise ValueError(f"unsupported camera scale status: {status!r}")
        names = tuple(str(value) for value in np.asarray(payload["image_names"]).tolist())
        bundle = CameraBundle(
            image_names=names,
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            source=str(metadata.get("source", "external-camera-bundle")),
            scale_status=status,
            world_units=str(scale.get("world_units", "arbitrary-external-unit")),
            world_units_to_mm=(
                float(scale["world_units_to_mm"])
                if scale.get("world_units_to_mm") is not None
                else None
            ),
            details=(metadata.get("details") if isinstance(metadata.get("details"), dict) else {}),
        )
    observation_names = tuple(image.relative_path for image in observations.images)
    return bundle.reordered(observation_names)


@dataclass(frozen=True, slots=True)
class ColmapCameraResult:
    registered_frames_dir: Path
    camera_bundle_path: Path
    report_path: Path
    bundle: CameraBundle
    report: dict[str, object]


def _pycolmap() -> Any:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "camera recovery requires pycolmap; install the video/SfM dependencies"
        ) from error
    return pycolmap


def _registered_images(reconstruction: Any) -> list[Any]:
    images = [reconstruction.images[image_id] for image_id in reconstruction.reg_image_ids()]
    return sorted(images, key=lambda image: str(image.name).casefold())


def recover_colmap_cameras(
    frames_dir: Path,
    output_dir: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
) -> ColmapCameraResult:
    """Recover arbitrary-scale cameras and undistort registered video frames."""

    frames_dir = frames_dir.resolve()
    if not frames_dir.is_dir():
        raise ValueError(f"frames directory does not exist: {frames_dir}")
    image_names = tuple(
        path.name
        for path in sorted(frames_dir.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_names) < 3:
        raise ValueError("COLMAP recovery requires at least three frames")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"COLMAP output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pycolmap = _pycolmap()
    database = output_dir / "database.db"
    sparse_root = output_dir / "sparse"
    sparse_root.mkdir()

    cpu_threads = min(8, os.cpu_count() or 1)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = cpu_threads
    extraction.use_gpu = False
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(frames_dir),
        image_names=list(image_names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = cpu_threads
    matching.use_gpu = False
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = min(10, len(image_names) - 1)
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    pycolmap.match_sequential(
        database_path=str(database),
        matching_options=matching,
        pairing_options=pairing,
        device=pycolmap.Device.cpu,
    )
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(frames_dir),
        output_path=str(sparse_root),
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP did not recover a model; add texture/background detail and "
            "slower overlapping views"
        )
    reconstruction = max(
        reconstructions.values(),
        key=lambda model: (int(model.num_reg_images()), int(model.num_points3D())),
    )
    registered = _registered_images(reconstruction)
    if len(registered) < 3:
        raise RuntimeError(
            f"COLMAP registered only {len(registered)} frame(s); at least 3 are required"
        )

    model_dir = output_dir / "selected_model"
    model_dir.mkdir()
    reconstruction.write(str(model_dir))
    registered_dir = output_dir / "registered_frames"
    registered_dir.mkdir()
    intrinsics: list[FloatArray] = []
    extrinsics: list[FloatArray] = []
    names: list[str] = []
    undistort_options = pycolmap.UndistortCameraOptions()
    for image in registered:
        source = frames_dir / str(image.name)
        bitmap = pycolmap.Bitmap.read(str(source), True)
        if bitmap is None:
            raise RuntimeError(f"pycolmap could not read registered frame: {source}")
        camera = reconstruction.cameras[image.camera_id]
        undistorted_bitmap, undistorted_camera = pycolmap.undistort_image(
            undistort_options,
            bitmap,
            camera,
        )
        destination = registered_dir / str(image.name)
        if not undistorted_bitmap.write(str(destination)):
            raise RuntimeError(f"failed to write undistorted frame: {destination}")
        matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :4] = matrix
        intrinsics.append(np.asarray(undistorted_camera.calibration_matrix(), dtype=np.float64))
        extrinsics.append(homogeneous)
        names.append(str(image.name))

    rotations = np.stack(extrinsics)[:, :3, :3]
    translations = np.stack(extrinsics)[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)
    camera_rank = int(np.linalg.matrix_rank(centers - centers.mean(axis=0), tol=1e-8))
    if camera_rank < 2:
        raise RuntimeError(
            "COLMAP camera centres are collinear/collapsed; pose-conditioned DA3 scale alignment "
            "requires at least three non-collinear camera positions"
        )

    missing_names = sorted(set(image_names).difference(names))
    details: dict[str, object] = {
        "camera_model_requested": camera_model,
        "input_frames": len(image_names),
        "registered_frames": len(names),
        "missing_frames": missing_names,
        "points3d": int(reconstruction.num_points3D()),
        "mean_reprojection_error_pixels": float(reconstruction.compute_mean_reprojection_error()),
        "mean_observations_per_registered_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "camera_center_rank": camera_rank,
        "matching": "sequential-overlap-10-quadratic-no-loop-detection",
        "features": f"SIFT CPU ({cpu_threads} threads maximum)",
        "undistorted_for_da3": True,
    }
    bundle = CameraBundle(
        image_names=tuple(names),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        extrinsics=np.stack(extrinsics).astype(np.float32),
        source="pycolmap-video-sfm",
        scale_status="unresolved",
        world_units="arbitrary-colmap-unit",
        details=details,
    )
    bundle_path = output_dir / "cameras.npz"
    bundle.save(bundle_path)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "usable",
        "camera_bundle": bundle.as_dict(),
        "input_frames_dir": str(frames_dir),
        "registered_frames_dir": str(registered_dir),
        "warnings": [
            "COLMAP world scale is arbitrary; use a fiducial or a named known "
            "dimension for millimetres",
            "the object must remain stationary while the camera moves; a turntable "
            "violates this model",
        ],
    }
    report_path = output_dir / "camera_recovery.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Database and sparse files remain as provenance; remove no evidence on success.
    if not model_dir.is_dir():  # pragma: no cover - defensive postcondition
        shutil.rmtree(output_dir, ignore_errors=True)
        raise RuntimeError("COLMAP did not persist the selected sparse model")
    return ColmapCameraResult(
        registered_frames_dir=registered_dir,
        camera_bundle_path=bundle_path,
        report_path=report_path,
        bundle=bundle,
        report=report,
    )
