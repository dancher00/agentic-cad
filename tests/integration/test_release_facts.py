from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_release_facts_are_derived_with_complete_provenance(tmp_path: Path) -> None:
    output = tmp_path / "release_facts.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_release_facts.py",
            "--allow-missing-tless",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    facts = {item["id"]: item for item in payload["facts"]}

    assert facts["decoder-control"]["metrics"]["mean_iou_percent"] == 92.06083857517498
    assert facts["bottleneck-uncalibrated"]["metrics"]["precision_at_0.05"] == 0.21875
    assert facts["bottleneck-exact-cameras"]["metrics"]["precision_at_0.05"] == 0.4296875
    assert (
        facts["bottleneck-exact-cameras-per-view-gt-affine"]["metrics"][
            "precision_at_0.05"
        ]
        == 0.626953125
    )
    assert facts["view-saturation-n16"]["metrics"]["precision_at_0.05"] == 0.75
    assert (
        facts["gt-blind-n32-projected"]["metrics"][
            "oracle_median_within_object_max_min_scale"
        ]
        == 13.891397050194614
    )
    assert facts["oracle-scale-spread-n8"]["metrics"][
        "median_within_object_max_min_scale"
    ] == 1.496859774861361
    assert len([key for key in facts if key.startswith("hypothesis-")]) == 6
    for fact in facts.values():
        assert fact["objects"] > 0
        assert fact["records"] > 0
        assert fact["seed"]
        assert fact["checkpoint"]
        assert len(fact["commit"]) == 40
        assert len(fact["source"]["sha256"]) == 64
