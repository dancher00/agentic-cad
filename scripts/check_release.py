#!/usr/bin/env python3
"""Validate the product release: documentation, fixtures and aggregate results."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "sample_data/public_benchmark_v2"
MAX_PUBLIC_FILE_BYTES = 8 * 1024 * 1024
REQUIRED = (
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/PHOTO_CAD.md",
    "docs/RAY_SECTIONS.md",
    "docs/LICENSES.md",
    "docs/BENCHMARKS.md",
    "docs/benchmarks.json",
    "docs/assets/quickstart/viewer.png",
    "docs/assets/brand/agentic-cad-mark.svg",
    "sample_data/public_benchmark_v2/manifest.json",
    "configs/public_benchmark_v2.yaml",
)
FORBIDDEN_PREFIXES = (
    "data/",
    "weights/",
    "cache/",
    "outputs/",
    "captures/",
    "legacy/",
    "paper/",
    "docs/research/",
    "docs/experiments/",
    "docs/results/",
)
FORBIDDEN_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".mp4", ".mov", ".avi", ".mkv")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def _public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return list(dict.fromkeys(ROOT / name for name in result.stdout.split("\0") if name))


def _check_fixtures(errors: list[str]) -> tuple[int, int]:
    manifest = _read_json(FIXTURES / "manifest.json")
    cases = manifest.get("cases", [])
    if len(cases) != 10 or not str(manifest.get("license", "")).startswith("Apache-2.0"):
        errors.append("expected ten Apache-2.0 benchmark fixtures")
    total = 0
    for case in cases:
        directory = FIXTURES / case["id"]
        views = sorted((directory / "views").glob("*.png"))
        masks = sorted((directory / "masks").glob("*.png"))
        total += len(views)
        if len(views) != case["input_views"] or [p.name for p in views] != [p.name for p in masks]:
            errors.append(f"incomplete view/mask set: {case['id']}")
        for name in ("cameras.npz", "gt.step", "gt.stl", "manifest.json"):
            if not (directory / name).is_file():
                errors.append(f"missing fixture file: {case['id']}/{name}")
        if case.get("reference_available_to_reconstruction") is not False:
            errors.append(f"reference leakage contract missing: {case['id']}")
    return len(cases), total


def _check_benchmarks(errors: list[str]) -> None:
    path = ROOT / "docs/benchmarks.json"
    if not path.exists():
        return
    payload = _read_json(path)
    for suite in payload.get("suites", []):
        total = suite.get("total", 0)
        outcomes = suite.get("outcomes", {})
        if total <= 0 or sum(outcomes.values()) != total:
            errors.append(f"benchmark denominator mismatch: {suite.get('name')}")
        if not suite.get("protocol") or not suite.get("command"):
            errors.append(f"benchmark protocol missing: {suite.get('name')}")
    if len(payload.get("suites", [])) != 3:
        errors.append("expected three complete benchmark summaries")
    if "/home/" in path.read_text():
        errors.append("benchmark summary contains local machine paths")


def main() -> None:
    errors: list[str] = []
    files = _public_files()
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required product file: {relative}")
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            continue
        if name.startswith(FORBIDDEN_PREFIXES) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"runtime/research artifact in product release: {name}")
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            errors.append(f"public file exceeds 8 MiB: {name}")
        if path.suffix == ".md":
            for raw in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
                target = raw.strip().strip("<>").split(maxsplit=1)[0]
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f"broken link in {name}: {target}")
    cases, views = _check_fixtures(errors)
    _check_benchmarks(errors)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if f"version: {project['version']}" not in (ROOT / "CITATION.cff").read_text():
        errors.append("package/citation version mismatch")
    preview = ROOT / "docs/assets/quickstart/viewer.png"
    if preview.is_file():
        with Image.open(preview) as image:
            if image.width < 1000 or image.height < 600:
                errors.append("workspace screenshot is too small")
            image.verify()
    if errors:
        print("release check failed:\n" + "\n".join(f"- {e}" for e in errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"release check passed: {len(files)} files · {cases} fixtures · {views} RGB views")


if __name__ == "__main__":
    main()
