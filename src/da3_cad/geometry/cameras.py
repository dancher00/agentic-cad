"""External camera bundles and deterministic COLMAP recovery for image captures."""

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

ColmapPairing = Literal["sequential", "exhaustive"]
ColmapDevice = Literal["auto", "cpu", "cuda"]
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


def _resolve_colmap_device(
    pycolmap: Any,
    requested: ColmapDevice,
) -> tuple[Any, Literal["cpu", "cuda"], bool]:
    """Resolve auto without silently accepting an unavailable CUDA build."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported COLMAP device: {requested!r}")
    raw_has_cuda = getattr(pycolmap, "has_cuda", False)
    has_cuda = bool(raw_has_cuda() if callable(raw_has_cuda) else raw_has_cuda)
    selected: Literal["cpu", "cuda"] = (
        ("cuda" if has_cuda else "cpu") if requested == "auto" else requested
    )
    if selected == "cuda" and not has_cuda:
        raise RuntimeError(
            "COLMAP CUDA was requested, but this pycolmap build has no CUDA support; "
            "install a CUDA-enabled COLMAP/pycolmap build or use --device cpu"
        )
    return getattr(pycolmap.Device, selected), selected, has_cuda


def _match_colmap_features(
    pycolmap: Any,
    *,
    database: Path,
    matching_options: Any,
    pairing: ColmapPairing,
    image_count: int,
    device: Any,
) -> str:
    """Run the requested pairing policy and return an auditable description."""

    if pairing == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = min(10, image_count - 1)
        pairing_options.quadratic_overlap = True
        pairing_options.loop_detection = False
        pycolmap.match_sequential(
            database_path=str(database),
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return f"sequential-overlap-{pairing_options.overlap}-quadratic-no-loop-detection"
    if pairing == "exhaustive":
        pairing_options = pycolmap.ExhaustivePairingOptions()
        pycolmap.match_exhaustive(
            database_path=str(database),
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return f"exhaustive-all-pairs-block-{pairing_options.block_size}"
    raise ValueError(f"unsupported COLMAP pairing: {pairing!r}")


def _write_camera_recovery_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recover_colmap_cameras(
    frames_dir: Path,
    output_dir: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    pairing: ColmapPairing = "sequential",
    device: ColmapDevice = "auto",
    minimum_registered_fraction: float = 0.8,
) -> ColmapCameraResult:
    """Recover cameras and undistort registered images from one static scene."""

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
    if pairing not in {"sequential", "exhaustive"}:
        raise ValueError(f"unsupported COLMAP pairing: {pairing!r}")
    if not 0.0 < minimum_registered_fraction <= 1.0:
        raise ValueError("minimum_registered_fraction must be in (0, 1]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"COLMAP output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pycolmap = _pycolmap()
    database = output_dir / "database.db"
    sparse_root = output_dir / "sparse"
    sparse_root.mkdir()
    report_path = output_dir / "camera_recovery.json"

    cpu_threads = min(8, os.cpu_count() or 1)
    colmap_device, selected_device, has_cuda = _resolve_colmap_device(pycolmap, device)
    run_configuration: dict[str, object] = {
        "camera_model_requested": camera_model,
        "camera_mode": "single-shared-intrinsics",
        "input_frames": len(image_names),
        "pairing": pairing,
        "device_requested": device,
        "device_selected": selected_device,
        "pycolmap_has_cuda": has_cuda,
        "minimum_registered_fraction": minimum_registered_fraction,
    }
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = cpu_threads
    extraction.use_gpu = selected_device == "cuda"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(frames_dir),
        image_names=list(image_names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=colmap_device,
    )
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = cpu_threads
    matching.use_gpu = selected_device == "cuda"
    matching_description = _match_colmap_features(
        pycolmap,
        database=database,
        matching_options=matching,
        pairing=pairing,
        image_count=len(image_names),
        device=colmap_device,
    )
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(frames_dir),
        output_path=str(sparse_root),
    )
    if not reconstructions:
        _write_camera_recovery_report(
            report_path,
            {
                "schema_version": "1.1",
                "status": "abstained",
                "reason": "COLMAP recovered no sparse model",
                "input_frames_dir": str(frames_dir),
                "configuration": run_configuration,
            },
        )
        raise RuntimeError(
            "COLMAP did not recover a model; add texture/background detail and "
            f"slower overlapping views; diagnostics: {report_path}"
        )
    reconstruction = max(
        reconstructions.values(),
        key=lambda model: (int(model.num_reg_images()), int(model.num_points3D())),
    )
    registered = _registered_images(reconstruction)
    if len(registered) < 3:
        _write_camera_recovery_report(
            report_path,
            {
                "schema_version": "1.1",
                "status": "abstained",
                "reason": "fewer than three images registered",
                "input_frames_dir": str(frames_dir),
                "configuration": run_configuration,
                "registered_frames": len(registered),
            },
        )
        raise RuntimeError(
            f"COLMAP registered only {len(registered)} frame(s); at least 3 are required"
        )
    registered_fraction = len(registered) / len(image_names)
    if registered_fraction < minimum_registered_fraction:
        registered_names = {str(image.name) for image in registered}
        _write_camera_recovery_report(
            report_path,
            {
                "schema_version": "1.1",
                "status": "abstained",
                "reason": "registered-image fraction is below the acceptance threshold",
                "input_frames_dir": str(frames_dir),
                "configuration": run_configuration,
                "registered_frames": len(registered),
                "registered_fraction": registered_fraction,
                "missing_frames": sorted(set(image_names).difference(registered_names)),
            },
        )
        raise RuntimeError(
            "COLMAP registered only "
            f"{len(registered)}/{len(image_names)} images ({registered_fraction:.1%}); "
            f"required {minimum_registered_fraction:.1%}; diagnostics: {report_path}"
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
        _write_camera_recovery_report(
            report_path,
            {
                "schema_version": "1.1",
                "status": "abstained",
                "reason": "camera centres are collinear or collapsed",
                "input_frames_dir": str(frames_dir),
                "configuration": run_configuration,
                "registered_frames": len(names),
                "registered_fraction": registered_fraction,
                "camera_center_rank": camera_rank,
            },
        )
        raise RuntimeError(
            "COLMAP camera centres are collinear/collapsed; pose-conditioned DA3 scale alignment "
            "requires at least three non-collinear camera positions"
        )

    missing_names = sorted(set(image_names).difference(names))
    details: dict[str, object] = {
        "camera_model_requested": camera_model,
        "input_frames": len(image_names),
        "registered_frames": len(names),
        "registered_fraction": registered_fraction,
        "missing_frames": missing_names,
        "points3d": int(reconstruction.num_points3D()),
        "mean_reprojection_error_pixels": float(reconstruction.compute_mean_reprojection_error()),
        "mean_observations_per_registered_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "camera_center_rank": camera_rank,
        "matching": matching_description,
        "pairing": pairing,
        "device_requested": device,
        "device_selected": selected_device,
        "pycolmap_has_cuda": has_cuda,
        "features": f"SIFT {selected_device.upper()} ({cpu_threads} CPU threads maximum)",
        "acceptance": {
            "minimum_registered_fraction": minimum_registered_fraction,
            "registered_fraction_passed": True,
            "minimum_camera_center_rank": 2,
            "camera_center_rank_passed": True,
        },
        "undistorted_for_da3": True,
    }
    bundle = CameraBundle(
        image_names=tuple(names),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        extrinsics=np.stack(extrinsics).astype(np.float32),
        source=("pycolmap-video-sfm" if pairing == "sequential" else "pycolmap-photo-sfm"),
        scale_status="unresolved",
        world_units="arbitrary-colmap-unit",
        details=details,
    )
    bundle_path = output_dir / "cameras.npz"
    bundle.save(bundle_path)
    report: dict[str, object] = {
        "schema_version": "1.1",
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
    _write_camera_recovery_report(report_path, report)
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
