#!/usr/bin/env python3
"""Package the report and synthetic research artifacts, excluding real imagery."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def package() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper/revival"
    dist = root / "dist/raysection"
    dist.mkdir(parents=True, exist_ok=True)
    if not (paper / "main.pdf").exists():
        raise FileNotFoundError("Build paper/revival/main.pdf before packaging")
    shutil.copyfile(paper / "main.pdf", root / "docs/RaySection_technical_report.pdf")
    shutil.copyfile(paper / "main.pdf", dist / "RaySection_technical_report.pdf")
    with zipfile.ZipFile(
        dist / "raysection-paper-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(paper.glob("*.tex")):
            text = path.read_text().replace("../../docs/assets/ray_sections/", "figures/")
            archive.writestr(path.name, text)
        archive.write(paper / "references.bib", "references.bib")
        archive.write(paper / "main.bbl", "main.bbl")
        for name in ("quantitative.pdf", "qualitative.pdf"):
            archive.write(root / "docs/assets/ray_sections" / name, f"figures/{name}")
        archive.writestr(
            "README.txt",
            "Build: pdflatex main; bibtex main; pdflatex main; pdflatex main\n"
            "Technical report. Author metadata required before external submission.\n",
        )
    with zipfile.ZipFile(
        dist / "raysection-synthetic-artifacts.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ("ray-section-eval-data", "ray-section-eval-v1", "ray-section-eval-search-v2"):
            for path in sorted((root / "outputs" / name).rglob("*")):
                if path.is_file() and path.suffix in {".json", ".npz", ".py", ".step", ".ply"}:
                    archive.write(path, str(path.relative_to(root / "outputs")))
        archive.write(root / "LICENSE", "LICENSE")
        archive.writestr(
            "README.txt",
            "Project-generated synthetic data and measurements only.\n"
            "No T-LESS imagery or third-party weights.\n"
            "See repository docs/RAY_SECTIONS.md for commands and limitations.\n",
        )
    files = sorted(
        path for path in dist.iterdir() if path.is_file() and path.name != "SHA256SUMS.json"
    )
    manifest = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    }
    (dist / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    package()
