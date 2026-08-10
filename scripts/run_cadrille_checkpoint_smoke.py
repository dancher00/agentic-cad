"""Run real SFT/RL Cadrille compatibility checks on one canonical DA3 cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.backends.cadrille import (
    CADRILLE_MODELS,
    CadrilleBackend,
    cadrille_license_notice,
    require_cadrille_terms,
)
from da3_cad.cad.equivalence import compare_validation_geometry
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import CadrilleConfig, SandboxConfig
from da3_cad.models import FloatArray


@dataclass(frozen=True, slots=True)
class _DecoderInput:
    decoder_points: FloatArray


def _sha256_bytes(value: FloatArray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_decoder_input(path: Path) -> FloatArray:
    value = np.load(path, allow_pickle=False)
    if value.shape == (1, 256, 3):
        value = value[0]
    if value.shape != (256, 3):
        raise ValueError(f"decoder input must have shape (256,3) or (1,256,3), got {value.shape}")
    if value.dtype != np.float32:
        raise ValueError(f"decoder input must be float32, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError("decoder input contains non-finite values")
    if float(value.min()) < -1.00001 or float(value.max()) > 1.00001:
        raise ValueError("decoder input is outside the verified [-1,1]^3 contract")
    return np.asarray(value, dtype=np.float32)


def _run_profile(
    *,
    profile: str,
    decoder_input: _DecoderInput,
    output_dir: Path,
    cache_dir: Path,
    local_files_only: bool,
    max_new_tokens: int,
    seed: int,
    accepted_license: str | None,
    sandbox: SandboxConfig,
) -> dict[str, object]:
    config = CadrilleConfig(
        checkpoint=profile,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        max_new_tokens=max_new_tokens,
        attn_implementation="sdpa",
        use_cache=True,
    )
    backend = CadrilleBackend(
        config,
        accepted_license=accepted_license,
        device="cuda",
    )
    started = time.perf_counter()
    program = backend.generate(decoder_input, seed=seed)
    total_seconds = time.perf_counter() - started
    if (
        backend.last_runtime_report is None
        or backend.last_raw_text is None
        or backend.last_clean_source is None
        or backend.last_parameterization_report is None
    ):
        raise RuntimeError(f"{profile} backend omitted required runtime evidence")

    output_dir.mkdir(parents=True)
    (output_dir / "raw_transport.txt").write_text(backend.last_raw_text, encoding="utf-8")
    (output_dir / "raw_clean.py").write_text(backend.last_clean_source, encoding="utf-8")
    (output_dir / "parameterized.py").write_text(program.source, encoding="utf-8")
    raw_validation = validate_and_export(
        backend.last_clean_source,
        output_dir / "raw",
        sandbox,
    )
    parameterized_validation = validate_and_export(
        program.source,
        output_dir / "parameterized",
        sandbox,
    )
    equivalence = compare_validation_geometry(raw_validation, parameterized_validation)
    if equivalence["equivalent"] is False:
        raise RuntimeError(f"{profile} AST parameterization changed geometry or validity")

    runtime = backend.last_runtime_report
    lifecycle = runtime["lifecycle"]
    model_contract = runtime["runtime_model_contract"]
    checkpoint_file = runtime["checkpoint_file"]
    if not isinstance(lifecycle, dict) or not isinstance(model_contract, dict):
        raise RuntimeError(f"{profile} runtime evidence has invalid structure")
    if not isinstance(checkpoint_file, dict):
        raise RuntimeError(f"{profile} checkpoint evidence has invalid structure")
    if checkpoint_file.get("sha256_verified") is not True:
        raise RuntimeError(f"{profile} checkpoint hash was not verified")
    if model_contract.get("attention_implementation") != "sdpa":
        raise RuntimeError(f"{profile} did not execute with SDPA")
    if lifecycle.get("compute_capability") != [12, 0]:
        raise RuntimeError(f"{profile} did not execute on sm_120")
    architectures = lifecycle.get("compiled_architectures")
    if not isinstance(architectures, list) or "sm_120" not in architectures:
        raise RuntimeError(f"{profile} torch build does not contain sm_120")
    if lifecycle.get("model_tensors_off_cuda") is not True:
        raise RuntimeError(f"{profile} retained model tensors on CUDA")
    if lifecycle.get("cuda_parameters_after_cpu_transfer") != 0:
        raise RuntimeError(f"{profile} retained CUDA parameters")
    if lifecycle.get("cuda_buffers_after_cpu_transfer") != 0:
        raise RuntimeError(f"{profile} retained CUDA buffers")

    return {
        "status": "valid-solid" if parameterized_validation.valid else "invalid-solid",
        "is_benchmark_result": False,
        "fallback_used": False,
        "backend": program.backend,
        "template_id": program.template_id,
        "parameter_count": len(program.parameters),
        "warnings": list(program.warnings),
        "runtime": runtime,
        "raw_validation": raw_validation.as_dict(),
        "parameterized_validation": parameterized_validation.as_dict(),
        "parameterization_geometry": equivalence,
        "artifacts": {
            "raw_transport": str(output_dir / "raw_transport.txt"),
            "raw_clean_source": str(output_dir / "raw_clean.py"),
            "editable_source": str(output_dir / "parameterized.py"),
            "raw_exports": str(output_dir / "raw"),
            "parameterized_exports": str(output_dir / "parameterized"),
        },
        "total_generate_seconds": total_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-input", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cadrille_checkpoint_smoke"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/cadrille_smoke/checkpoints.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--profile", choices=("sft", "rl", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--accept-license")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    keys = tuple(CADRILLE_MODELS) if args.profile == "all" else (args.profile,)
    print("Cadrille checkpoint terms:")
    for key in keys:
        spec = CADRILLE_MODELS[key]
        print(f"- {cadrille_license_notice(spec)}")
        require_cadrille_terms(spec, accepted_license=args.accept_license)
    if args.output_root.exists():
        raise ValueError(f"output root already exists: {args.output_root}")

    repository_commit = _git_output("rev-parse", "HEAD")
    dirty_before_report = _git_output("status", "--porcelain")
    if dirty_before_report:
        raise RuntimeError(
            "checkpoint evidence must start from a clean commit; commit or move changes first"
        )

    points = _load_decoder_input(args.decoder_input)
    decoder_input = _DecoderInput(points)
    sandbox = SandboxConfig()
    runs: dict[str, Any] = {}
    for key in keys:
        print(f"running {key}...")
        runs[key] = _run_profile(
            profile=key,
            decoder_input=decoder_input,
            output_dir=args.output_root / key,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            accepted_license=args.accept_license,
            sandbox=sandbox,
        )
        print(f"{key}: {runs[key]['status']}")

    payload = {
        "schema_version": "1.0",
        "status": "real-checkpoint-compatibility-not-quality-benchmark",
        "repository_commit": repository_commit,
        "repository_clean_before_report": True,
        "python": sys.version,
        "input": {
            "source": str(args.decoder_input),
            "shape": [1, 256, 3],
            "dtype": "float32",
            "coordinate_space": "[-1,1]^3 isotropic bbox",
            "minimum": float(points.min()),
            "maximum": float(points.max()),
            "sha256": _sha256_bytes(points),
        },
        "license": {
            "accepted": args.accept_license,
            "weights_redistributed": False,
        },
        "seed": args.seed,
        "runs": runs,
        "notes": [
            "single-cloud compatibility smoke; not reconstruction-quality evidence",
            "greedy single-candidate generation only",
            "invalid model output is reported without a geometric fallback",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()
