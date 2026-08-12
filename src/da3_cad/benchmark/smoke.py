"""Offline stub smoke harness; it deliberately emits no quality metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

from da3_cad.config import AppConfig
from da3_cad.observations import discover_images
from da3_cad.pipeline import reconstruct


def discover_cases(root: Path) -> tuple[tuple[str, Path], ...]:
    try:
        discover_images(root)
    except ValueError:
        cases: list[tuple[str, Path]] = []
        for child in sorted(
            (item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name
        ):
            try:
                discover_images(child)
            except ValueError:
                continue
            cases.append((child.name, child))
        if not cases:
            raise ValueError(f"no image case directories found under: {root}") from None
        return tuple(cases)
    return ((root.name, root),)


def run_smoke_benchmark(root: Path, output_dir: Path, config: AppConfig) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for case_id, case_path in discover_cases(root):
        case_started = time.monotonic()
        validation = reconstruct(case_path, output_dir / case_id, config)
        rows.append(
            {
                "case_id": case_id,
                "valid": validation.valid,
                "seconds": time.monotonic() - case_started,
                "metrics": None,
                "metrics_reason": "offline stub smoke does not implement benchmark metrics",
            }
        )
    payload: dict[str, object] = {
        "protocol": "offline-stub-smoke-v2",
        "is_benchmark_result": False,
        "case_count": len(rows),
        "seconds": time.monotonic() - started,
        "rows": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
