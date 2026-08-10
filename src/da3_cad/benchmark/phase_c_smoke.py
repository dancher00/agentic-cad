"""Strict stop-point summary for the real Phase C integration smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _decoder_sha256(value: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def _require_gpu_lifecycle(lifecycle: dict[str, Any], label: str) -> None:
    if lifecycle["compute_capability"] != [12, 0]:
        raise ValueError(f"{label} did not run on sm_120")
    if "sm_120" not in lifecycle["compiled_architectures"]:
        raise ValueError(f"{label} torch build does not contain sm_120")
    if lifecycle["model_tensors_off_cuda"] is not True:
        raise ValueError(f"{label} retained model tensors on CUDA")
    if lifecycle["cuda_parameters_after_cpu_transfer"] != 0:
        raise ValueError(f"{label} retained CUDA parameters")
    if lifecycle["cuda_buffers_after_cpu_transfer"] != 0:
        raise ValueError(f"{label} retained CUDA buffers")


def build_phase_c_smoke_summary(
    *,
    reconstruction_dir: Path,
    checkpoint_report_path: Path,
    repository_commit: str,
) -> dict[str, object]:
    """Validate ignored real outputs and build compact committed evidence."""

    reconstruction = _load_json(reconstruction_dir / "artefacts" / "reconstruction_report.json")
    geometry = _load_json(reconstruction_dir / "artefacts" / "geometry" / "geometry_report.json")
    decoder = _load_json(reconstruction_dir / "artefacts" / "decoder_report.json")
    quality = _load_json(reconstruction_dir / "quality.json")
    parameters = _load_json(reconstruction_dir / "parameters.json")
    provenance = _load_json(reconstruction_dir / "provenance.json")
    checkpoints = _load_json(checkpoint_report_path)

    required_files = (
        reconstruction_dir / "model.py",
        reconstruction_dir / "model.step",
        reconstruction_dir / "model.stl",
        reconstruction_dir / "report.md",
        reconstruction_dir / "artefacts" / "raw_decoder_output.txt",
        reconstruction_dir / "artefacts" / "raw_decoder_output.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"full reconstruction is missing artifacts: {missing}")
    if reconstruction["status"] != "valid" or quality["status"] != "valid":
        raise ValueError("full RL reconstruction did not produce a valid solid")
    if reconstruction["profile"] != "research":
        raise ValueError("full reconstruction did not use the research profile")
    if reconstruction["depth_backend"] != "da3-large":
        raise ValueError("full reconstruction did not use DA3-LARGE")
    if reconstruction["cad_backend"] != "cadrille-rl":
        raise ValueError("full reconstruction did not use Cadrille RL")
    if reconstruction["input_views"] != 8:
        raise ValueError("full reconstruction must use the selected eight-view flatness fixture")
    if reconstruction["fallback_used"] is not False or quality["fallback_used"] is not False:
        raise ValueError("a silent fallback was used")
    source_control = reconstruction["source_control"]
    if source_control["commit"] != repository_commit:
        raise ValueError("full reconstruction was produced by a different commit")
    if source_control["working_tree_clean"] is not True:
        raise ValueError("full reconstruction did not start from a clean working tree")

    canonical = reconstruction["canonicalizer"]
    if canonical["decoder_contract"] != {
        "shape": [1, 256, 3],
        "dtype": "float32",
        "channels": ["x", "y", "z"],
        "coordinate_space": "[-1,1]^3 isotropic bbox",
        "finite": True,
    }:
        raise ValueError("canonical decoder contract differs from the verified contract")
    orientation = canonical["orientation"]
    if orientation["method"] != "planar-dominance-symmetry":
        raise ValueError("thin-cloud full run did not take the planar orientation branch")
    if not orientation["planar_extent_ratio"] < orientation["planar_threshold"]:
        raise ValueError("planar orientation provenance has an inconsistent threshold decision")
    if abs(float(orientation["determinant"]) - 1.0) > 1e-9:
        raise ValueError("canonical orientation is not right-handed")

    decoder_input = np.load(
        reconstruction_dir / "artefacts" / "canonicalizer" / "decoder_input.npy",
        allow_pickle=False,
    )
    if decoder_input.shape != (1, 256, 3) or decoder_input.dtype != np.float32:
        raise ValueError("saved decoder tensor has wrong shape or dtype")
    if not np.isfinite(decoder_input).all():
        raise ValueError("saved decoder tensor is non-finite")
    if float(decoder_input.min()) < -1.00001 or float(decoder_input.max()) > 1.00001:
        raise ValueError("saved decoder tensor is outside [-1,1]^3")
    input_sha256 = _decoder_sha256(decoder_input)

    geometry_runtime = geometry["da3"]
    if geometry_runtime["model"]["key"] != "large":
        raise ValueError("geometry report mislabeled the DA3 model")
    if geometry_runtime["checkpoint_file"]["sha256_verified"] is not True:
        raise ValueError("DA3-LARGE checkpoint bytes were not verified")
    _require_gpu_lifecycle(geometry_runtime["lifecycle"], "DA3-LARGE")
    fusion_views = geometry["fusion"]["views"]
    if len(fusion_views) != 8 or not all(view["fused"] > 0 for view in fusion_views):
        raise ValueError("not every input view contributed fused points")

    decoder_runtime = decoder["runtime"]
    if decoder_runtime["model"]["key"] != "rl":
        raise ValueError("decoder report mislabeled the Cadrille model")
    if decoder_runtime["checkpoint_file"]["sha256_verified"] is not True:
        raise ValueError("Cadrille RL checkpoint bytes were not verified")
    if decoder_runtime["input"]["sha256"] != input_sha256:
        raise ValueError("Cadrille did not consume the saved canonical tensor")
    if decoder_runtime["runtime_model_contract"]["attention_implementation"] != "sdpa":
        raise ValueError("full Cadrille run did not use SDPA")
    if decoder_runtime["runtime_model_contract"]["point_encoder_dtype"] != "torch.float32":
        raise ValueError("Cadrille point encoder did not retain upstream float32")
    if decoder_runtime["runtime_model_contract"]["input_embedding_dtype"] != "torch.bfloat16":
        raise ValueError("Cadrille language embedding did not use bfloat16")
    _require_gpu_lifecycle(decoder_runtime["lifecycle"], "Cadrille RL full run")
    parameterization = decoder["parameterization_validation"]
    if parameterization["status"] != "equivalent" or parameterization["equivalent"] is not True:
        raise ValueError("raw and parameterized full-run geometry are not equivalent")

    if parameters["units"] != "normalized-cad-training-units":
        raise ValueError("full run silently invented metric scale")
    if parameters["backend"] != "cadrille-point-cloud-rl":
        raise ValueError("parameter metadata lost the neural backend")
    if not parameters["parameters"]:
        raise ValueError("full run did not expose editable parameters")
    expected_stages = [
        "depth-unprojection-fusion",
        "canonicalization",
        "cad-generation",
        "program-validation",
    ]
    if [stage["name"] for stage in provenance["stages"]] != expected_stages:
        raise ValueError("full run provenance is missing a required stage")

    if checkpoints["status"] != "real-checkpoint-compatibility-not-quality-benchmark":
        raise ValueError("checkpoint report has the wrong status")
    if checkpoints["repository_commit"] != repository_commit:
        raise ValueError("checkpoint smoke used a different commit")
    if checkpoints["repository_clean_before_report"] is not True:
        raise ValueError("checkpoint smoke did not start clean")
    if checkpoints["input"]["sha256"] != input_sha256:
        raise ValueError("checkpoint smoke used a different decoder tensor")
    runs = checkpoints["runs"]
    if set(runs) != {"sft", "rl"}:
        raise ValueError("checkpoint smoke must contain both SFT and RL")
    if runs["sft"]["status"] != "invalid-solid":
        raise ValueError("SFT outcome changed; inspect rather than hiding the result")
    if runs["rl"]["status"] != "valid-solid":
        raise ValueError("RL checkpoint did not produce a valid solid")
    for key in ("sft", "rl"):
        run = runs[key]
        if run["fallback_used"] is not False:
            raise ValueError(f"{key} smoke used a fallback")
        runtime = run["runtime"]
        if runtime["model"]["key"] != key:
            raise ValueError(f"{key} run has swapped model metadata")
        if runtime["checkpoint_file"]["sha256_verified"] is not True:
            raise ValueError(f"{key} checkpoint hash was not verified")
        if runtime["runtime_model_contract"]["attention_implementation"] != "sdpa":
            raise ValueError(f"{key} run did not use SDPA")
        _require_gpu_lifecycle(runtime["lifecycle"], f"Cadrille {key.upper()}")

    rl_repeat_exact = (
        runs["rl"]["runtime"]["generation"]["raw_text_sha256"]
        == decoder_runtime["generation"]["raw_text_sha256"]
        and runs["rl"]["runtime"]["generation"]["source_sha256"]
        == decoder_runtime["generation"]["source_sha256"]
        and runs["rl"]["parameterized_validation"]["volume"]
        == reconstruction["validation"]["volume"]
        and runs["rl"]["parameterized_validation"]["bbox"] == reconstruction["validation"]["bbox"]
    )
    if not rl_repeat_exact:
        raise ValueError("full-run and standalone RL greedy outputs are not exact repeats")

    return {
        "schema_version": "1.0",
        "status": "real-phase-c-integration-not-quality-benchmark",
        "repository_commit": repository_commit,
        "full_reconstruction": {
            "input_views": reconstruction["input_views"],
            "input_digest": reconstruction["input_digest"],
            "profile": reconstruction["profile"],
            "fallback_used": False,
            "source_control": source_control,
            "da3": {
                "model": geometry_runtime["model"],
                "checkpoint_file": geometry_runtime["checkpoint_file"],
                "lifecycle": geometry_runtime["lifecycle"],
                "fusion": geometry["fusion"],
            },
            "canonicalizer": {
                "decoder_contract": canonical["decoder_contract"],
                "decoder_input_sha256": input_sha256,
                "decoder_input_minimum": float(decoder_input.min()),
                "decoder_input_maximum": float(decoder_input.max()),
                "orientation": orientation,
                "scale": canonical["scale"],
            },
            "decoder": {
                "model": decoder_runtime["model"],
                "checkpoint_file": decoder_runtime["checkpoint_file"],
                "generation": decoder_runtime["generation"],
                "runtime_model_contract": decoder_runtime["runtime_model_contract"],
                "lifecycle": decoder_runtime["lifecycle"],
                "parameterization_validation": parameterization,
            },
            "output": {
                "backend": quality["backend"],
                "template_id": quality["template_id"],
                "units": parameters["units"],
                "parameter_count": len(parameters["parameters"]),
                "validation": reconstruction["validation"],
                "step": "model.step",
                "stl": "model.stl",
            },
            "provenance_stages": expected_stages,
        },
        "checkpoint_smoke": checkpoints,
        "rl_greedy_repeat_exact": rl_repeat_exact,
        "claims": [
            "one eight-view DA3-LARGE to Cadrille-RL reconstruction produced "
            "a valid editable solid",
            "raw and AST-parameterized RL geometry matched within 1e-9",
            "SFT and RL checkpoints loaded and ran on torch 2.13 sm_120 with SDPA",
            "both Cadrille models transferred every parameter and buffer off CUDA",
        ],
        "not_claimed": [
            "CAD reconstruction quality or benchmark accuracy",
            "generalization beyond this single synthetic eight-view part",
            "metric scale without external evidence",
            "validity of the SFT program for this input",
        ],
    }
