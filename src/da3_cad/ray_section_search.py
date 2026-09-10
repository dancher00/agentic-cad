"""Executable section search: enumerate interval budgets before CAD admission.

The discrete best partition may be impossible to compile as one solid. Search
the best partition at each budget as well, rather than abandoning the frame.
This is a bounded search, not an exact optimizer over kernel-valid programs.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.ray_sections import (
    RayBundle,
    _bounded_compile,
    carve,
    evidence_frames,
    observed_points,
    partition,
)


def reconstruct_sections(
    bundle: RayBundle,
    output: Path,
    *,
    resolution: int = 72,
    maximum_sections: int = 8,
    penalty: float = 0.03,
    device: str = "cuda",
    mode: str = "adaptive",
    use_depth: bool = True,
    frame_modes: tuple[str, ...] = ("world", "pca", "planes"),
    compile_timeout: float = 20.0,
) -> dict[str, Any]:
    import torch

    bundle.validate()
    if not 16 <= resolution <= 256 or not 1 <= maximum_sections <= 32:
        raise ValueError("resolution must be 16..256 and maximum_sections 1..32")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if not np.isfinite(compile_timeout) or compile_timeout <= 0:
        raise ValueError("compile timeout must be finite and positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    points = observed_points(bundle)
    candidates = []
    frames_report = []
    for name, frame in evidence_frames(points):
        if name not in frame_modes:
            continue
        volume, lower, step, carving = carve(
            bundle, points, frame, resolution=resolution, device=device, use_depth=use_depth
        )
        frames_report.append({"frame": name, **carving})
        for axis in range(3):
            budgets = range(1, maximum_sections + 1) if mode == "adaptive" else [maximum_sections]
            seen = set()
            for budget in budgets:
                sections, objective = partition(
                    volume, axis, maximum_sections=budget, penalty=penalty, device=device, mode=mode
                )
                boundaries = tuple((a, b) for a, b, _ in sections)
                if boundaries in seen:
                    continue
                seen.add(boundaries)
                candidates.append((objective, name, axis, sections, frame, lower, step))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    failures = []
    for objective, name, axis, sections, frame, lower, step in candidates:
        try:
            compiled = _bounded_compile(
                (sections, axis, frame, lower, step), output, compile_timeout
            )
        except Exception as error:
            failures.append(
                {
                    "frame": name,
                    "axis": axis,
                    "sections": len(sections),
                    "objective": objective,
                    "error": str(error),
                }
            )
            continue
        report = {
            "schema": "executable-ray-section-cad-v2",
            "status": "CANDIDATE",
            "kernel_valid": True,
            "decision_policy": "requires independent source-view assessment",
            "reference_access": False,
            "views": list(bundle.names),
            "device": device,
            "resolution": resolution,
            "maximum_sections": maximum_sections,
            "penalty": penalty,
            "mode": mode,
            "use_depth": use_depth,
            "frame": name,
            "axis": axis,
            "frame_matrix": frame.tolist(),
            "lower": lower.tolist(),
            "voxel_size": step,
            "objective": objective,
            "sections": len(sections),
            "candidate_partitions": len(candidates),
            **compiled,
            "compile_timeout_seconds": compile_timeout,
            "elapsed_seconds": time.perf_counter() - started,
            "failed_compilations": failures,
            "carving": frames_report,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(
                Path(__file__).with_name("ray_sections.py").read_bytes()
            ).hexdigest(),
            "step_sha256": hashlib.sha256((output / "candidate.step").read_bytes()).hexdigest(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    report = {
        "status": "ABSTAIN",
        "kernel_valid": False,
        "failed_compilations": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
