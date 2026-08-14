from __future__ import annotations

from da3_cad.benchmark.pose_errors import run_pose_error_benchmark


def test_pose_error_controls_cover_recovery_noop_and_safe_rejection() -> None:
    ledger, _evaluations = run_pose_error_benchmark()
    summary = ledger["summary"]

    assert summary["passed"] == 7
    assert summary["recoveries_passed"] == 3
    assert summary["unsafe_failures_rejected"] == 3
    assert summary["false_corrections"] == 0
    assert summary["depth_and_intrinsics_preserved"] is True
    controls = {case["id"]: case for case in ledger["cases"]}
    assert controls["bounded_translation"]["selected_method"] == "translation"
    assert controls["bounded_rotation"]["selected_method"] == "se3"
    for case in ledger["cases"]:
        assert case["invariants"]["rejected_pose_unchanged"] is True
        optimization = set(case["audit"]["optimization_views"])
        held_out = set(case["audit"]["held_out_views"])
        assert not optimization.intersection(held_out)
