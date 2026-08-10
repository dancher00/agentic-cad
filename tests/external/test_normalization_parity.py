from __future__ import annotations

import json
from pathlib import Path

import pytest

from da3_cad.geometry.normalization_audit import audit_manifest

MANIFEST = Path("benchmarks/normalization/mesh_samples.json")
REPORT = Path("benchmarks/normalization/parity_report.json")
DATA_ROOT = Path("data/normalization_audit")


@pytest.mark.external_data
def test_real_deepcad_and_fusion360_mesh_normalization_parity() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip("external normalization meshes were not downloaded with accepted terms")

    actual = audit_manifest(MANIFEST, data_root=DATA_ROOT)
    expected = json.loads(REPORT.read_text(encoding="utf-8"))

    assert actual == expected
    assert actual["sample_count"] == 10
    assert actual["conclusion"] == "isotropic-largest-extent-with-centered-short-axes"
    assert actual["hypotheses"] == {
        "isotropic_largest_extent_centered": True,
        "per_axis_scaling": False,
        "corner_anchored": False,
    }
