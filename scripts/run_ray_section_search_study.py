#!/usr/bin/env python3
"""Run the executable-budget search variant with the same observation/evaluator code."""

import hashlib
import json
import sys
from pathlib import Path

import run_ray_section_real
import run_ray_section_study

from da3_cad.ray_section_search import reconstruct_sections

if __name__ == "__main__":
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        "src/da3_cad/ray_sections.py",
        "src/da3_cad/ray_section_search.py",
        "scripts/run_ray_section_study.py",
        "scripts/run_ray_section_real.py",
    ]
    config = {
        "method": "executable-budget-search-v2",
        "sources": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
    }
    manifest = output / "search-config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("search implementation changed; use a new output directory")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    if "--real" in sys.argv:
        sys.argv.remove("--real")
        run_ray_section_real.reconstruct_sections = reconstruct_sections
        run_ray_section_real.main()
    else:
        run_ray_section_study.reconstruct_sections = reconstruct_sections
        run_ray_section_study.main()
