from __future__ import annotations

import json
from pathlib import Path

import trimesh

from da3_cad.viewer import build_viewer


def test_build_viewer_embeds_geometry_parameters_and_downloads(tmp_path: Path) -> None:
    run = tmp_path / "run"
    artefacts = run / "artefacts"
    artefacts.mkdir(parents=True)
    box = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    box.export(run / "model.stl")
    (run / "model.step").write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;\n")
    cloud = trimesh.points.PointCloud(box.vertices)
    cloud.export(artefacts / "fused_cloud.ply")
    (run / "model.py").write_text(
        "import cadquery as cq\n"
        "PARAMETERS = {'width': 2.0}\n"
        "r = cq.Workplane('XY').box(PARAMETERS['width'], 1.0, 0.5)\n",
        encoding="utf-8",
    )
    (run / "parameters.json").write_text(
        json.dumps(
            {
                "units": "canonical-model-unit",
                "primary_parameters": [
                    {"name": "width", "value": 2.0, "semantic_role": "overall width"}
                ],
                "implementation_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "quality.json").write_text(
        json.dumps({"status": "valid", "backend": "test"}), encoding="utf-8"
    )
    (run / "provenance.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id": "fixture"}), encoding="utf-8"
    )

    output = build_viewer(run)
    html = output.read_text(encoding="utf-8")

    assert output == run / "viewer.html"
    assert "DA3-CAD Viewer" in html
    assert '"name":"width"' in html
    assert '"original_points":8' in html
    assert "model.step" in html
    assert "triangle edges hidden" in html
    assert "ctx.stroke()" not in html
    assert "__DA3_CAD_VIEWER_PAYLOAD__" not in html
