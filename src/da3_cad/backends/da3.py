"""Pinned Depth Anything 3 adapter with explicit licenses and contracts."""

from __future__ import annotations

import hashlib
import importlib
import os
import random
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.model_manager import ModelLifecycleReport, StagedModelManager
from da3_cad.models import DepthPrediction, FloatArray, ObservationSet, UInt8Array

DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_SOURCE_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3"


@dataclass(frozen=True, slots=True)
class Da3ModelSpec:
    key: Literal["base", "large"]
    model_id: str
    revision: str
    license: str
    noncommercial: bool
    parameter_scale: str
    weight_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "noncommercial": self.noncommercial,
            "parameter_scale": self.parameter_scale,
            "weight_sha256": self.weight_sha256,
            "source_url": DA3_SOURCE_URL,
            "source_revision": DA3_SOURCE_REVISION,
        }


DA3_MODELS: dict[str, Da3ModelSpec] = {
    "base": Da3ModelSpec(
        key="base",
        model_id="depth-anything/DA3-BASE",
        revision="f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        license="Apache-2.0",
        noncommercial=False,
        parameter_scale="0.12B (upstream model card)",
        weight_sha256="e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
    ),
    "large": Da3ModelSpec(
        key="large",
        model_id="depth-anything/DA3-LARGE",
        revision="c54c26b16ec04d218e8d584ecf4bce082a9fcc20",
        license="CC BY-NC 4.0",
        noncommercial=True,
        parameter_scale="0.35B (upstream model card)",
        weight_sha256="eaf2ae06df55889ad23eb245c82e2dd2a30c0cbf7e3d873a118fa5ed27a3e421",
    ),
}


def get_da3_model_spec(key: str) -> Da3ModelSpec:
    try:
        return DA3_MODELS[key]
    except KeyError as error:
        raise ValueError(f"unsupported DA3 checkpoint profile: {key}") from error


def da3_license_notice(spec: Da3ModelSpec) -> str:
    scope = "non-commercial research only" if spec.noncommercial else "permissive weights"
    return (
        f"{spec.model_id}@{spec.revision} — {spec.license} ({scope}); "
        f"terms/model card: https://huggingface.co/{spec.model_id}"
    )


def require_weight_terms(spec: Da3ModelSpec, *, accepted_noncommercial: bool) -> None:
    if spec.noncommercial and not accepted_noncommercial:
        raise ValueError(
            f"{da3_license_notice(spec)}. Re-run with the explicit "
            "--accept-noncommercial-weights flag after reviewing those terms."
        )


def verify_da3_source(source_dir: Path) -> str:
    """Require the exact audited source checkout; never silently use another revision."""

    if not (source_dir / ".git").is_dir():
        raise FileNotFoundError(
            f"pinned DA3 source checkout is missing at {source_dir}; "
            "run scripts/fetch_da3_source.py"
        )
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != DA3_SOURCE_REVISION:
        raise RuntimeError(
            f"DA3 source revision mismatch: expected {DA3_SOURCE_REVISION}, got {revision}"
        )
    return revision


def _disabled_export(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("DA3-CAD does not use the optional upstream export stack")


def _import_da3_model_class(source_dir: Path) -> type[Any]:
    verify_da3_source(source_dir)
    source_python = source_dir / "src"
    if not source_python.is_dir():
        raise FileNotFoundError(f"DA3 Python source directory is missing: {source_python}")
    source_text = str(source_python.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    # Upstream api.py eagerly imports every optional exporter (3DGS, COLMAP,
    # movie rendering) even when export_dir=None. Our inference path never calls
    # export, so provide a fail-closed stub instead of requiring those unrelated
    # dependencies. The model, input processor and output processor remain the
    # unmodified pinned upstream implementation.
    if "depth_anything_3.api" not in sys.modules:
        export_module = types.ModuleType("depth_anything_3.utils.export")
        export_module.export = _disabled_export  # type: ignore[attr-defined]
        sys.modules["depth_anything_3.utils.export"] = export_module

    module = importlib.import_module("depth_anything_3.api")
    module_path = Path(str(module.__file__)).resolve()
    if not module_path.is_relative_to(source_python.resolve()):
        raise RuntimeError(f"imported DA3 from unexpected path: {module_path}")
    model_class: type[Any] = module.DepthAnything3
    return model_class


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_da3_checkpoint(
    spec: Da3ModelSpec,
    cache_dir: Path,
    *,
    local_files_only: bool,
) -> dict[str, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("DA3 inference requires huggingface-hub") from error
    started = time.perf_counter()
    path = Path(
        hf_hub_download(
            repo_id=spec.model_id,
            filename="model.safetensors",
            revision=spec.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    actual_sha256 = _sha256(path)
    if actual_sha256 != spec.weight_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {spec.model_id}: "
            f"expected {spec.weight_sha256}, got {actual_sha256}"
        )
    return {
        "filename": "model.safetensors",
        "bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "sha256_verified": True,
        "acquisition_and_hash_seconds": time.perf_counter() - started,
    }


def _array_statistics(values: FloatArray) -> dict[str, object]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {"finite_fraction": 0.0, "min": None, "median": None, "max": None}
    return {
        "finite_fraction": float(finite.mean()),
        "positive_fraction": float((finite_values > 0.0).mean()),
        "min": float(finite_values.min()),
        "median": float(np.median(finite_values)),
        "max": float(finite_values.max()),
    }


def adapt_da3_prediction(raw: Any, spec: Da3ModelSpec) -> DepthPrediction:
    """Validate the upstream object before admitting it to our geometry core."""

    missing = [
        name
        for name in ("depth", "conf", "intrinsics", "extrinsics", "processed_images")
        if getattr(raw, name, None) is None
    ]
    if missing:
        raise ValueError(f"DA3 prediction is missing required fields: {', '.join(missing)}")

    depth = np.asarray(raw.depth, dtype=np.float32)
    confidence = np.asarray(raw.conf, dtype=np.float32)
    intrinsics = np.asarray(raw.intrinsics, dtype=np.float32)
    extrinsics = np.asarray(raw.extrinsics, dtype=np.float32)
    processed_array = np.asarray(raw.processed_images, dtype=np.uint8)
    processed: tuple[UInt8Array, ...] = tuple(processed_array[index] for index in range(len(depth)))
    prediction = DepthPrediction(
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=processed,
        backend=f"da3-{spec.key}@{spec.revision[:12]}",
        warnings=(
            "DA3 any-view depth has unresolved global scale unless metric evidence is supplied",
            "DA3 extrinsics are interpreted as world-to-camera per pinned source/exporter",
        ),
    )

    if not np.any(np.isfinite(depth) & (depth > 0.0)):
        raise ValueError("DA3 returned no finite positive depths")
    if not np.any(np.isfinite(confidence)):
        raise ValueError("DA3 returned no finite confidence values")
    for index, intrinsic in enumerate(intrinsics):
        if not np.isfinite(intrinsic).all() or abs(float(np.linalg.det(intrinsic))) < 1e-10:
            raise ValueError(f"DA3 intrinsic matrix {index} is non-finite or singular")
    for extrinsic in extrinsics:
        as_homogeneous_extrinsic(extrinsic)
    return prediction


class Da3Backend:
    """Real multi-view DA3 inference; each call releases its model from CUDA."""

    name = "depth-anything-3"

    def __init__(
        self,
        *,
        checkpoint: Literal["base", "large"],
        source_dir: Path,
        cache_dir: Path,
        process_resolution: int = 504,
        process_resolution_method: str = "upper_bound_resize",
        local_files_only: bool = False,
        accepted_noncommercial: bool = False,
        model_class_loader: Callable[[Path], type[Any]] = _import_da3_model_class,
    ) -> None:
        self.spec = get_da3_model_spec(checkpoint)
        require_weight_terms(
            self.spec,
            accepted_noncommercial=accepted_noncommercial,
        )
        self.source_dir = source_dir
        self.cache_dir = cache_dir
        self.process_resolution = process_resolution
        self.process_resolution_method = process_resolution_method
        self.local_files_only = local_files_only
        self._model_class_loader = model_class_loader
        self.last_lifecycle: ModelLifecycleReport | None = None
        self.last_runtime_report: dict[str, object] | None = None

    def predict(self, observations: ObservationSet, *, device: str, seed: int) -> DepthPrediction:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DA3 inference requires the pinned GPU dependencies") from error
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = verified_da3_checkpoint(
            self.spec,
            self.cache_dir,
            local_files_only=self.local_files_only,
        )
        model_class = self._model_class_loader(self.source_dir)

        def load_model() -> Any:
            return model_class.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

        image_paths = [str(observation.path) for observation in observations.images]

        def infer(model: Any) -> Any:
            return model.inference(
                image_paths,
                process_res=self.process_resolution,
                process_res_method=self.process_resolution_method,
                export_dir=None,
                use_ray_pose=False,
            )

        raw, lifecycle = StagedModelManager(device).execute(load_model, infer)
        prediction = adapt_da3_prediction(raw, self.spec)
        self.last_lifecycle = lifecycle
        confidence = prediction.confidence
        if confidence is None:
            raise RuntimeError("validated DA3 prediction unexpectedly lost confidence")
        self.last_runtime_report = {
            "model": self.spec.as_dict(),
            "source_revision_verified": DA3_SOURCE_REVISION,
            "checkpoint_file": checkpoint,
            "process_resolution": self.process_resolution,
            "process_resolution_method": self.process_resolution_method,
            "use_ray_pose": False,
            "input_views": len(observations.images),
            "depth": _array_statistics(prediction.depth),
            "confidence": _array_statistics(confidence),
            "is_metric": bool(getattr(raw, "is_metric", 0)),
            "output_shapes": {
                "depth": list(prediction.depth.shape),
                "confidence": list(confidence.shape),
                "intrinsics": list(prediction.intrinsics.shape),
                "extrinsics": list(prediction.extrinsics.shape),
                "processed_images": [list(image.shape) for image in prediction.processed_images],
            },
            "lifecycle": lifecycle.as_dict(),
        }
        return prediction


def default_da3_source_dir() -> Path:
    configured = os.environ.get("DA3_CAD_DA3_SOURCE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "data" / "upstream" / "Depth-Anything-3"
