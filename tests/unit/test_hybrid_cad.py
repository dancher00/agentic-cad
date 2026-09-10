from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from da3_cad.cad.sandbox import validate_and_export
from da3_cad.cli import app
from da3_cad.config import SandboxConfig
from da3_cad.gpt_cad import CADResponse, GPTConfig, Parameter, run_gpt_cad
from da3_cad.hybrid_evidence import HybridConfig
from da3_cad.hybrid_fit import rasterize, silhouette_iou


def test_projection_preserves_holes_and_rejects_behind_camera() -> None:
    # Four rectangular strips form a square aperture at z=10.
    vertices, faces = [], []
    for x0, y0, x1, y1 in [(1, 1, 9, 3), (1, 7, 9, 9), (1, 3, 3, 7), (7, 3, 9, 7)]:
        offset = len(vertices)
        vertices.extend([(x0, y0, 10), (x1, y0, 10), (x1, y1, 10), (x0, y1, 10)])
        faces.extend([(offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)])
    points, triangles = np.asarray(vertices, dtype=float), np.asarray(faces)
    k = np.diag([10.0, 10.0, 1.0])
    mask = rasterize(points, triangles, k, np.eye(4), (12, 12))
    assert mask[2, 5] and mask[5, 2]
    assert not mask[5, 5] and not mask[0, 0]
    assert silhouette_iou(mask, mask) == 1.0
    assert silhouette_iou(mask, np.zeros_like(mask)) == 0.0
    points[:, 2] = -10
    assert not rasterize(points, triangles, k, np.eye(4), (12, 12)).any()


@pytest.mark.parametrize("subtract, expected", [(False, False), (True, True)])
def test_cavity_check_detects_attachment_intrusion(
    tmp_path: Path, subtract: bool, expected: bool
) -> None:
    # A handle-like attachment crosses a hollow wall; the B-rep itself is valid.
    source = """import cadquery as cq
NON_PENETRATION_CAVITY = cq.Workplane('XY').circle(8).extrude(18).translate((0,0,2))
body = cq.Workplane('XY').circle(10).extrude(20).cut(NON_PENETRATION_CAVITY)
attachment = cq.Workplane('XY').box(8,4,4).translate((8,0,12))
r = body.union(attachment)
"""
    if subtract:
        source += "r = r.cut(NON_PENETRATION_CAVITY)\n"
    result = validate_and_export(source, tmp_path, SandboxConfig())
    assert result.valid is expected
    if not expected:
        assert "penetrates NON_PENETRATION_CAVITY" in str(result.error)
    else:
        assert result.details["topology_invariants"]["non_penetration_cavity"]["passed"]


def test_empty_clearance_cannot_bypass_check(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\nNON_PENETRATION_CAVITY=cq.Workplane('XY')\n"
        "r=cq.Workplane('XY').box(10,10,10)"
    )
    assert not validate_and_export(source, tmp_path, SandboxConfig()).valid


def test_hybrid_dry_run_is_side_effect_free(tmp_path: Path) -> None:
    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    output = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "generate",
            str(photo),
            "--prompt",
            "mug",
            "--reconstruction",
            "hybrid",
            "--dry-run",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hybrid"]["fit_parameters"] == 4
    assert payload["config"]["max_output_tokens"] == 32768
    assert payload["config"]["timeout_seconds"] == 900
    assert payload["api_calls"] == 0
    assert not output.exists()


@pytest.mark.parametrize("last_result", ["worse", "invalid", "incomplete", "empty", "api_error"])
def test_geometric_retry_keeps_better_candidate_and_original_photos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_result: str,
) -> None:
    from types import SimpleNamespace

    import da3_cad.hybrid_evidence as evidence
    import da3_cad.hybrid_fit as fitting

    def prepare(paths: Any, prompt: str, output: Path, *args: Any) -> dict[str, Any]:
        output.mkdir()
        return {
            "contract": {"has_cavity": False, "has_handle_aperture": False, "visible_features": []},
            "views": [],
            "panels": [],
        }

    scores = iter([0.7, 0.6])

    def fit(candidate: Any, folder: Path, *args: Any) -> Any:
        iou = next(scores)
        report = {"after": {"loss": 1 - iou, "mean_silhouette_iou": iou}}
        (folder / "geometry-review.json").write_text(json.dumps(report))
        Image.new("RGB", (32, 32)).save(folder / "comparison-00.png")
        return candidate, None, report

    monkeypatch.setattr(evidence, "prepare_evidence", prepare)
    monkeypatch.setattr(fitting, "fit_candidate", fit)
    calls = []

    def parse(**kwargs: Any) -> Any:
        calls.append(kwargs)
        length = 20 + len(calls)
        candidate = CADResponse(
            name="block",
            code=f'import cadquery as cq\nlength={length}\nr=cq.Workplane("XY").box(length,10,5)',
            parameters=[Parameter(name="length", value=length, unit="mm", source="estimated")],
            assumptions=[],
        )
        if last_result == "invalid" and len(calls) == 2:
            candidate = candidate.model_copy(
                update={"code": f'import cadquery as cq\nlength={length}\nr=cq.Workplane("XY")'}
            )
        status = "completed"
        parsed = candidate
        if len(calls) == 2:
            if last_result == "api_error":
                from openai import OpenAIError

                raise OpenAIError("provider-secret-must-not-appear")
            if last_result == "incomplete":
                status = "incomplete"
            if last_result == "empty":
                parsed = None
        return SimpleNamespace(
            id="test", model="test-model", status=status, usage=None, output_parsed=parsed
        )

    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    output = tmp_path / "run"
    result = run_gpt_cad(
        "block",
        output,
        images=[photo],
        hybrid=HybridConfig(feature_review=False),
        config=GPTConfig(max_repairs=1),
        create_viewer=False,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    assert result["selected_attempt"] == 1
    assert result["geometry"]["after"]["mean_silhouette_iou"] == 0.7
    assert not result["observation_target_met"]
    assert (
        json.loads((output / "parameters.json").read_text())["primary_parameters"][0]["value"] == 21
    )
    assert len(calls) == 2
    inputs = calls[1]["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in inputs) == 2
    assert all("hybrid_feedback" not in item for item in inputs)
    if last_result != "worse":
        assert result["attempts"][-1]["status"] != "valid"
    assert "provider-secret-must-not-appear" not in (output / "report.json").read_text()


def test_multiview_parameter_fit_improves_width_without_changing_known_height(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from da3_cad.gpt_cad import parameterize
    from da3_cad.hybrid_fit import GeometryObjective, fit_candidate

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    size = 96
    k = np.array([[150.0, 0, 47.5], [0, 150.0, 47.5], [0, 0, 1.0]])
    yy, xx = np.mgrid[:size, :size]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(k).T
    depths, masks, extrinsics = [], [], []
    for camera_x in (-10.0, 10.0):
        origin = np.array([camera_x, 0.0, 0.0])
        bounds = np.array([[-12.0, -10.0, 55.0], [12.0, 10.0, 65.0]])
        times = (bounds[:, None, None, :] - origin) / rays[None]
        entry = np.min(times, axis=0).max(axis=-1)
        leave = np.max(times, axis=0).min(axis=-1)
        mask = (entry <= leave) & (entry > 0)
        depths.append(np.where(mask, entry, 0))
        masks.append(mask)
        ext = np.eye(4)
        ext[:3, 3] = -origin
        extrinsics.append(ext)
    np.savez(
        evidence / "geometry.npz",
        depth=depths,
        masks=masks,
        intrinsics=np.stack([k, k]),
        extrinsics=extrinsics,
    )
    candidate = CADResponse(
        name="box",
        code=(
            'import cadquery as cq\nwidth=20\nheight=20\nr=cq.Workplane("XY").box(width,height,10)'
        ),
        parameters=[
            Parameter(name="width", value=20, unit="mm", source="estimated"),
            Parameter(name="height", value=20, unit="mm", source="specified"),
        ],
        assumptions=[],
    )
    folder = tmp_path / "attempt"
    folder.mkdir()
    source = parameterize(candidate)
    (folder / "model.py").write_text(source)
    settings = GPTConfig()
    assert validate_and_export(source, folder, settings.sandbox).valid

    def known_registration(self: GeometryObjective) -> dict[str, Any]:
        # Isolate parameter fitting from registration: cameras and pose are exact here.
        pose = np.r_[
            np.zeros(3),
            np.log(self.mesh_radius / self.radius),
            (np.array([0.0, 0.0, 60.0]) - self.center) / self.radius,
        ]
        return self.evaluate(pose)

    monkeypatch.setattr(GeometryObjective, "register", known_registration)
    fitted, validation, review = fit_candidate(candidate, folder, evidence, settings, 2)
    assert validation is not None and validation.valid
    assert review["after"]["mean_silhouette_iou"] > review["before"]["mean_silhouette_iou"]
    assert fitted.parameters[0].value > 20
    assert fitted.parameters[1].value == 20
    assert len(review["after"]["view_ious"]) == 2
    assert all(row["parameter"] == "width" for row in review["trials"])


def test_material_chords_recover_thin_plate_thickness() -> None:
    import trimesh

    from da3_cad.hybrid_checks import material_chords

    result = material_chords(trimesh.creation.box(extents=[20, 10, 2]))
    assert result["hit_samples"] == 96
    assert result["p10_p50_p90_mm"][0] == pytest.approx(2)
    assert result["p10_p50_p90_mm"][1] == pytest.approx(2)
    assert result["certified_minimum_wall_thickness"] is False


def test_height_check_uses_solid_extent_not_unchanged_parameter() -> None:
    from types import SimpleNamespace

    from da3_cad.hybrid_checks import check_overall_height

    candidate = SimpleNamespace(
        parameters=[Parameter(name="target_height", value=100, unit="mm", source="specified")]
    )
    with pytest.raises(ValueError, match="actual Z extent is 110"):
        check_overall_height(candidate, SimpleNamespace(valid=True, bbox=(0, 0, -10, 50, 50, 100)))
    check_overall_height(candidate, SimpleNamespace(valid=True, bbox=(0, 0, -10, 50, 50, 90)))


def test_mixed_aspect_masks_follow_da3_crop_instead_of_stretching() -> None:
    from da3_cad.hybrid_evidence import align_mask_to_da3

    mask = np.zeros((224, 336), dtype=np.uint8)
    mask[80:140, 20:40] = 255  # Outside the shared center crop.
    mask[80:140, 150:180] = 255
    aligned = align_mask_to_da3(Image.fromarray(mask), (224, 224))
    assert aligned.shape == (224, 224)
    assert aligned.sum() == 60 * 30
    assert aligned[100, 100]
    assert not aligned[:, :40].any()


def test_cached_evidence_rejects_a_different_photo(tmp_path: Path) -> None:
    from da3_cad.hybrid_evidence import prepare_evidence

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "evidence.json").write_text(
        json.dumps({"input_identity": {"prompt": "mug", "sha256": ["wrong-photo-hash"]}})
    )
    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="exact prompt and ordered photo hashes"):
        prepare_evidence([photo], "mug", output, None, None, HybridConfig(evidence_cache=cache))
    assert not output.exists()


def test_scalar_fit_preserves_body_profile_and_equal_radii() -> None:
    from types import SimpleNamespace

    from da3_cad.hybrid_checks import check_profile_order

    def profile(top: float, upper: float, bottom: float) -> Any:
        return SimpleNamespace(
            parameters=[
                Parameter(name=name, value=value, unit="mm", source="estimated")
                for name, value in [
                    ("top_body_diameter", top),
                    ("upper_body_diameter", upper),
                    ("body_base_diameter", bottom),
                ]
            ]
        )

    reference = profile(84, 81, 65)
    with pytest.raises(ValueError, match="reverse body profile"):
        check_profile_order(reference, profile(77.28, 81, 65))
    check_profile_order(reference, profile(84, 80, 66))
    with pytest.raises(ValueError):
        check_profile_order(profile(80, 80, 65), profile(84, 80, 65))


def test_fit_cannot_trade_away_a_view_or_accept_depth_only_improvement() -> None:
    from da3_cad.hybrid_checks import fit_improves, observation_target_met

    initial = {
        "loss": 0.2,
        "mean_silhouette_iou": 0.82,
        "view_ious": [0.84, 0.8],
        "relative_depth_surface_residual": 0.2,
    }
    assert not observation_target_met({"mean_silhouette_iou": 0.9, "view_ious": [0.99, 0.81]}, 0.85)
    assert not fit_improves(
        initial, {**initial, "loss": 0.15, "relative_depth_surface_residual": 0.1}
    )
    assert not fit_improves(
        initial, {**initial, "loss": 0.15, "mean_silhouette_iou": 0.88, "view_ious": [0.98, 0.78]}
    )
    assert fit_improves(
        initial, {**initial, "loss": 0.15, "mean_silhouette_iou": 0.86, "view_ious": [0.87, 0.85]}
    )


def test_candidate_selection_prefers_all_view_target_over_better_average() -> None:
    from da3_cad.hybrid_checks import candidate_rank

    misleading_average = {"loss": 0.09, "mean_silhouette_iou": 0.91, "view_ious": [0.99, 0.83]}
    all_views = {"loss": 0.12, "mean_silhouette_iou": 0.88, "view_ious": [0.88, 0.88]}
    assert candidate_rank(all_views, 0.85) < candidate_rank(misleading_average, 0.85)
