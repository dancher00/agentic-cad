from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from da3_cad.geometry.normalization import normalize_bbox_for_decoder
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


@pytest.mark.external_data
def test_production_transform_matches_upstream_on_real_mesh_vertices() -> None:
    if not DATA_ROOT.is_dir():
        pytest.skip("external normalization meshes were not downloaded with accepted terms")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checked = 0
    for dataset in manifest["datasets"]:
        for sample in dataset["samples"]:
            path = DATA_ROOT / dataset["name"] / sample["path"]
            mesh = trimesh.load_mesh(path, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            unit, decoder, transform = normalize_bbox_for_decoder(vertices)
            upstream = (vertices - 0.5) * 2.0
            np.testing.assert_allclose(decoder, upstream, atol=1e-3)
            np.testing.assert_allclose(unit.min(axis=0) + unit.max(axis=0), np.ones(3), atol=1e-6)
            assert transform.largest_extent == pytest.approx(1.0, abs=5e-4)
            checked += 1
    assert checked == 10
