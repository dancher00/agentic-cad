from __future__ import annotations

import json
import tomllib
from pathlib import Path


def test_product_documentation_and_entry_points() -> None:
    readme = Path("README.md").read_text()
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert readme.startswith("# Agentic CAD")
    assert project["scripts"]["agentic-cad"] == project["scripts"]["da3-cad"]
    assert f"version: {project['version']}" in Path("CITATION.cff").read_text()
    assert "docs/PHOTO_CAD.md" in readme
    assert "docs/BENCHMARKS.md" in readme
    assert "RESEARCH_HISTORY" not in readme


def test_benchmark_summary_counts_every_attempt() -> None:
    payload = json.loads(Path("docs/benchmarks.json").read_text())
    assert len(payload["suites"]) == 3
    for suite in payload["suites"]:
        assert suite["total"] > 0
        assert sum(suite["outcomes"].values()) == suite["total"]
        assert suite["reference_access_during_generation"] is False
        assert suite["protocol"] and suite["command"]
        assert "cases" not in suite
