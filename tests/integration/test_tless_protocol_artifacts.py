from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_tless_split_covers_all_objects_with_nested_sixteen_views() -> None:
    split = json.loads(Path("benchmarks/splits/tless_primesense.json").read_text())
    assert split["object_count"] == 30
    assert split["view_counts"] == [1, 2, 4, 8, 16]
    assert [item["object_id"] for item in split["objects"]] == list(range(1, 31))
    for item in split["objects"]:
        assert len(item["views"]) == 16
        assert [view["rank"] for view in item["views"]] == list(range(16))
        assert len({view["frame_id"] for view in item["views"]}) == 16
        assert {view["scene_id"] for view in item["views"]} == {item["scene_id"]}
        assert {view["gt_index"] for view in item["views"]} == {item["gt_index"]}
        assert min(view["visible_fraction"] for view in item["views"]) >= item[
            "visibility_threshold"
        ]
    encoded = json.dumps(
        split["objects"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == split["objects_sha256"]


def test_tless_gt_repair_audit_retains_all_objects() -> None:
    audit = json.loads(
        Path("benchmarks/tless_primesense/gt_mesh_audit.json").read_text()
    )
    assert audit["object_count"] == 30
    assert audit["repair_count"] == 3
    assert {
        record["object_id"]
        for record in audit["records"]
        if record["repair_applied"]
    } == {5, 7, 11}
    assert sum(
        record["source_sha256"] == record["output_sha256"]
        for record in audit["records"]
    ) == 27
