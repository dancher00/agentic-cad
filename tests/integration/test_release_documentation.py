from __future__ import annotations

import json
from pathlib import Path


def test_release_documents_match_generated_evidence() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    tless_doc = Path("docs/TLESS_RESULTS.md").read_text(encoding="utf-8")
    ledger = json.loads(Path("benchmarks/release_facts.json").read_text(encoding="utf-8"))
    facts = {item["id"]: item for item in ledger["facts"]}

    assert ledger["tless_status"] == "complete"
    assert ledger["tless_oracle_status"] == "complete"
    assert "<!-- TLESS_" not in readme
    assert "<!-- TLESS_" not in tless_doc
    assert "arbitrary phone photos" in readme
    assert "Six closed hypotheses" in readme
    assert "camera pose and depth scale between views" in " ".join(readme.split())

    normalized_documents = (readme + "\n" + tless_doc).replace("**", "")
    tless_facts = [fact for fact in facts.values() if fact["id"].startswith("tless-n")]
    assert len(tless_facts) == 10
    for fact in tless_facts:
        metrics = fact["metrics"]
        selector = (
            "single"
            if metrics["candidate_row"] == "single-decode"
            else "best-of-10 input-CD"
        )
        row_prefix = (
            f"| {metrics['views']} | {selector} | "
            f"{metrics['mean_iou_percent']:.2f}% | "
            f"{metrics['median_chamfer_x1000']:.3f} | "
            f"{metrics['ir_percent']:.2f}% | {metrics['valid']}/30 | 30 / 30 |"
        )
        assert normalized_documents.count(row_prefix) == 2
        assert fact["objects"] == 30
        assert fact["records"] == 30
        assert fact["seed"] == "20260810"
        assert "c54c26b16ec04d218e8d584ecf4bce082a9fcc20" in fact["checkpoint"]
        assert "712489b5890a0ce81b18cf441e14b2ed2eadc02a" in fact["checkpoint"]
        assert fact["commit"] == "cc7e3e5583d2b99f5cb4cd8040f11a27fa4ec359"

    oracle_facts = [
        fact
        for fact in facts.values()
        if fact["id"].startswith("tless-gt-mask-oracle-n8-")
    ]
    assert len(oracle_facts) == 2
    for fact in oracle_facts:
        metrics = fact["metrics"]
        selector = (
            "single"
            if metrics["candidate_row"] == "single-decode"
            else "best-of-10 input-CD"
        )
        row_prefix = (
            f"| 8 | {selector} | "
            f"{metrics['mean_iou_percent']:.2f}% | "
            f"{metrics['median_chamfer_x1000']:.3f} | "
            f"{metrics['ir_percent']:.2f}% | {metrics['valid']}/30 | 30 / 30 |"
        )
        assert normalized_documents.count(row_prefix) == 2
        assert fact["objects"] == 30
        assert fact["records"] == 30
        assert fact["seed"] == "20260810"
        assert fact["commit"] == "aa793b926279f9436c23a54527bf7ed1638736b5"

    segmentation = facts["tless-segmentation-control-n8-best-of-10-input-CD"]
    assert segmentation["metrics"]["automatic_mean_iou_percent"] == 6.2933392177150465
    assert segmentation["metrics"]["oracle_mean_iou_percent"] == 8.913444189145197
    assert segmentation["metrics"]["material_gain"] is False
    assert "2.620" in readme
    assert "third lever" in (readme + tless_doc)
    assert "REPOSITORY_URL" in Path("docs/AWESOME_PR.md").read_text(encoding="utf-8")

    assert facts["bottleneck-uncalibrated"]["metrics"]["precision_at_0.05"] == 0.21875
    assert facts["bottleneck-exact-cameras"]["metrics"]["precision_at_0.05"] == 0.4296875
    assert (
        facts["bottleneck-exact-cameras-per-view-gt-affine"]["metrics"][
            "precision_at_0.05"
        ]
        == 0.626953125
    )
    assert facts["view-saturation-n16"]["metrics"]["precision_at_0.05"] == 0.75
    assert facts["decoder-control"]["metrics"]["mean_iou_percent"] == 92.06083857517498
