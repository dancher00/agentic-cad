#!/usr/bin/env python3
"""Fail when the public repository contains private or inconsistent artifacts."""

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
FIXTURES = ROOT / "sample_data" / "public_benchmark_v2"
LEDGER = ROOT / "docs" / "results" / "public-benchmark-v2.json"
MAX_PUBLIC_FILE_BYTES = 8 * 1024 * 1024

REQUIRED = (
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/PUBLIC_BENCHMARK.md",
    "docs/DA3-CAD_public_benchmark_v2.pdf",
    "docs/assets/release/teaser.png",
    "docs/assets/release/cpu_smoke.gif",
    "docs/assets/release/public_benchmark_v2.png",
    "docs/results/public-benchmark-v2.json",
    "sample_data/public_benchmark_v2/README.md",
    "sample_data/public_benchmark_v2/manifest.json",
    "configs/public_benchmark_v2.yaml",
    "scripts/build_cpu_smoke_gif.py",
)
FORBIDDEN_PREFIXES = ("data/", "weights/", "cache/", "outputs/", "captures/", "legacy/")
FORBIDDEN_SUFFIXES = (".safetensors", ".ckpt", ".pth", ".pt", ".mp4", ".mov", ".avi", ".mkv")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _public_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / name for name in completed.stdout.split("\0") if name]


def _check_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required release file: {relative}")


def _check_repository_surface(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if not path.exists():
            continue
        if any(relative.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            errors.append(f"forbidden public path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"model/video artifact must not be public: {relative}")
        if path.name.lower().startswith("mug") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            errors.append(f"private capture must remain ignored: {relative}")
        if path.is_file() and path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            errors.append(
                f"public file exceeds {MAX_PUBLIC_FILE_BYTES // (1024 * 1024)} MiB: {relative}"
            )


def _check_readme_links(errors: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(readme):
        target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", maxsplit=1)[0]
        if target and not (ROOT / target).exists():
            errors.append(f"README link target does not exist: {target}")


def _check_fixture_and_ledger(errors: list[str]) -> tuple[int, int]:
    manifest = _read_json(FIXTURES / "manifest.json")
    ledger = _read_json(LEDGER)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        errors.append("fixture manifest cases must be a list")
        return 0, 0
    if len(cases) < 10:
        errors.append(f"public benchmark is too small: {len(cases)} cases")
    if not str(manifest.get("license", "")).startswith("Apache-2.0"):
        errors.append("fixture manifest must declare Apache-2.0")

    total_views = 0
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            errors.append("fixture case must be an object")
            continue
        case_id = str(case.get("id"))
        case_ids.append(case_id)
        case_root = FIXTURES / case_id
        expected = int(case.get("input_views", 0))
        views = sorted((case_root / "views").glob("*.png"))
        masks = sorted((case_root / "masks").glob("*.png"))
        total_views += len(views)
        if len(views) != expected or len(masks) != expected:
            errors.append(
                f"{case_id}: expected {expected} views/masks, found {len(views)}/{len(masks)}"
            )
        if [path.name for path in views] != [path.name for path in masks]:
            errors.append(f"{case_id}: view and mask names differ")
        for name in ("cameras.npz", "gt.step", "gt.stl", "manifest.json"):
            if not (case_root / name).is_file():
                errors.append(f"{case_id}: missing {name}")
        if case.get("reference_available_to_reconstruction") is not False:
            errors.append(f"{case_id}: reference leakage contract is not false")

    ledger_cases = ledger.get("cases")
    if not isinstance(ledger_cases, list):
        errors.append("release ledger cases must be a list")
        return len(cases), total_views
    ledger_ids = [str(case.get("id")) for case in ledger_cases if isinstance(case, dict)]
    if ledger_ids != case_ids:
        errors.append("fixture and ledger case order/identity differ")
    summary = ledger.get("summary", {})
    decisions = [case.get("product_decision") for case in ledger_cases if isinstance(case, dict)]
    valid_steps = sum(
        bool(case.get("valid_step")) for case in ledger_cases if isinstance(case, dict)
    )
    expected_summary = {
        "cases": len(cases),
        "rgb_views": total_views,
        "valid_steps": valid_steps,
        "product_accepts": decisions.count("accept"),
        "provenance_rejects": decisions.count("reject-provenance"),
        "abstentions": decisions.count("abstain"),
    }
    for key, expected in expected_summary.items():
        if not isinstance(summary, dict) or summary.get(key) != expected:
            errors.append(f"ledger summary {key!r} is not {expected!r}")
    if (
        ledger.get("claim_boundary", {}).get("reference_cad_available_to_reconstruction")
        is not False
    ):
        errors.append("ledger reference-CAD leakage contract is not false")
    ledger_text = LEDGER.read_text(encoding="utf-8")
    if "/home/" in ledger_text or '"executable"' in ledger_text:
        errors.append("ledger contains machine-local runtime paths")
    return len(cases), total_views


def _check_metadata(errors: list[str]) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if f"version: {project['version']}" not in citation:
        errors.append("pyproject and CITATION.cff versions differ")
    if not (ROOT / "docs" / "DA3-CAD_public_benchmark_v2.pdf").read_bytes().startswith(b"%PDF"):
        errors.append("public benchmark PDF is invalid")
    for relative in (
        "docs/assets/release/teaser.png",
        "docs/assets/release/public_benchmark_v2.png",
    ):
        if not (ROOT / relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"invalid PNG signature: {relative}")
    animation_path = ROOT / "docs/assets/release/cpu_smoke.gif"
    if not animation_path.read_bytes().startswith((b"GIF87a", b"GIF89a")):
        errors.append("invalid GIF signature: docs/assets/release/cpu_smoke.gif")
    else:
        with Image.open(animation_path) as animation:
            if animation.size != (1200, 540) or animation.n_frames < 4:
                errors.append("CPU smoke GIF must be 1200x540 with at least four frames")


def main() -> None:
    errors: list[str] = []
    files = _public_files()
    _check_required(errors)
    _check_repository_surface(files, errors)
    _check_readme_links(errors)
    cases, views = _check_fixture_and_ledger(errors)
    _check_metadata(errors)
    if errors:
        print("release check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"release check passed: {len(files)} files · {cases} cases · {views} RGB views")


if __name__ == "__main__":
    main()
