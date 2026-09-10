import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from da3_cad.gpt_cad import CADResponse, GPTConfig, Parameter, run_gpt_cad
from da3_cad.hybrid_evidence import HybridConfig


def test_wrong_local_feature_forces_retry_despite_high_global_iou(tmp_path, monkeypatch):
    import da3_cad.feature_review as review
    import da3_cad.hybrid_evidence as evidence
    import da3_cad.hybrid_fit as fitting

    def prepare(paths, prompt, output, *args):
        output.mkdir()
        return {
            "contract": {
                "has_cavity": False,
                "has_handle_aperture": False,
                "visible_features": ["base"],
            },
            "views": [],
            "panels": [],
        }

    def fit(candidate, folder, *args):
        score = 0.99 if candidate.parameters[0].value == 20 else 0.95
        result = {"after": {"loss": 1 - score, "mean_silhouette_iou": score}}
        (folder / "geometry-review.json").write_text(json.dumps(result))
        return candidate, None, result

    reviews = iter([2, 0])

    def inspect(*args):
        return {
            "findings": [
                {
                    "feature": "base",
                    "severity": next(reviews),
                    "photo_evidence": "narrow foot",
                    "cad_difference": "too wide",
                    "correction": "narrow the foot",
                }
            ],
            "camera_caveats": "",
        }

    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        size = 20 + len(calls) - 1
        candidate = CADResponse(
            name="block",
            code=f'import cadquery as cq\nwidth={size}\nr=cq.Workplane("XY").box(width,10,5)',
            parameters=[Parameter(name="width", value=size, unit="mm", source="estimated")],
            assumptions=[],
        )
        return SimpleNamespace(
            id="fake", model="fake", status="completed", usage=None, output_parsed=candidate
        )

    monkeypatch.setattr(evidence, "prepare_evidence", prepare)
    monkeypatch.setattr(fitting, "fit_candidate", fit)
    monkeypatch.setattr(review, "review_features", inspect)
    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    result = run_gpt_cad(
        "block",
        tmp_path / "run",
        images=[photo],
        hybrid=HybridConfig(),
        config=GPTConfig(max_repairs=1),
        create_viewer=False,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    assert len(calls) == 2
    assert result["selected_attempt"] == 2
    assert result["feature_review_passed"]
    assert "narrow the foot" in str(calls[1]["input"])
    previous = tmp_path / "run"
    rejected = {
        "protocol_version": review.REVIEW_PROTOCOL_VERSION,
        "findings": [{"feature": "base", "severity": 2, "correction": "narrow the foot"}],
    }
    (previous / "attempts/02/feature-review.json").write_text(json.dumps(rejected))
    monkeypatch.setattr(
        review,
        "review_features",
        lambda *args: {
            "findings": [{"feature": "base", "severity": 0}],
            "camera_caveats": "",
        },
    )
    resumed = run_gpt_cad(
        "block",
        tmp_path / "continued",
        images=[photo],
        hybrid=HybridConfig(),
        config=GPTConfig(max_repairs=0),
        create_viewer=False,
        resume_from=previous,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    assert len(calls) == 3  # Generate from saved critique without reviewing the old mesh again.
    assert "narrow the foot" in str(calls[2]["input"])
    assert resumed["resume"]["uses_saved_feature_feedback"]
    rejected["protocol_version"] = 1
    (previous / "attempts/02/feature-review.json").write_text(json.dumps(rejected))
    refreshed = run_gpt_cad(
        "block",
        tmp_path / "refreshed",
        images=[photo],
        hybrid=HybridConfig(),
        config=GPTConfig(max_repairs=0),
        create_viewer=False,
        resume_from=previous,
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    assert (
        len(calls) == 3
    )  # Old review protocol must be reassessed, not sent as repair instructions.
    assert not refreshed["resume"]["uses_saved_feature_feedback"]


def test_render_preserves_vertical_axis_and_aspect(tmp_path: Path):
    import numpy as np
    import trimesh

    from da3_cad.feature_review import render_views

    mesh = trimesh.creation.box(extents=(1, 1, 3))
    mesh.export(tmp_path / "model.stl")
    image = Image.open(render_views(tmp_path / "model.stl", tmp_path))
    pixels = np.asarray(image)[:500, :500]
    ys, xs = np.where(np.all(pixels < 230, axis=2) & (np.indices(pixels.shape[:2])[0] > 80))
    assert 2.8 < np.ptp(ys) / np.ptp(xs) < 3.2


def test_disconnected_spline_error_explains_cadquery_continuity(tmp_path):
    from da3_cad.cad.sandbox import validate_and_export
    from da3_cad.config import SandboxConfig

    result = validate_and_export(
        "import cadquery as cq\n"
        'r = cq.Workplane("XZ").moveTo(0,0).lineTo(5,0)'
        ".spline([(6,3),(5,6)]).lineTo(0,6).close().revolve(360,(0,0),(0,1))",
        tmp_path,
        SandboxConfig(),
    )
    assert not result.valid
    assert "includeCurrent=True" in result.error


def test_resume_reuses_saved_program_without_provider_call_and_rejects_changed_input(tmp_path):
    import pytest

    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="saved",
            model="fake",
            status="completed",
            usage=None,
            output_parsed=CADResponse(
                name="box",
                code='import cadquery as cq\nwidth=2\nr=cq.Workplane("XY").box(width,3,4)',
                parameters=[Parameter(name="width", value=2, unit="mm", source="estimated")],
                assumptions=[],
            ),
        )

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    previous = tmp_path / "previous"
    run_gpt_cad("box", previous, client=client, create_viewer=False)
    result = run_gpt_cad(
        "box", tmp_path / "resumed", client=client, create_viewer=False, resume_from=previous
    )
    assert len(calls) == 1
    assert result["attempts"][0]["reused_from"] == str(previous)
    assert (previous / "model.py").read_bytes() == (tmp_path / "resumed/model.py").read_bytes()
    with pytest.raises(ValueError, match="exact same prompt"):
        run_gpt_cad("different", tmp_path / "bad", client=client, resume_from=previous)
    assert not (tmp_path / "bad").exists()


def test_thin_shell_front_surface_occludes_inner_triangles(tmp_path):
    import numpy as np
    import trimesh

    from da3_cad.feature_review import render_views

    mesh = trimesh.creation.annulus(r_min=0.99, r_max=1.0, height=4, sections=32)
    mesh.export(tmp_path / "model.stl")
    pixels = np.asarray(Image.open(render_views(tmp_path / "model.stl", tmp_path)))
    # A straight front wall has the same illumination along its height;
    # inner/back triangles must not show through the thin shell.
    stripe = pixels[100:430, 240:260, 0]
    assert np.max(np.ptp(stripe.astype(int), axis=0)) <= 1


def test_disconnected_component_feedback_locates_the_detached_part(tmp_path):
    from da3_cad.cad.sandbox import validate_and_export
    from da3_cad.config import SandboxConfig

    result = validate_and_export(
        'import cadquery as cq\na=cq.Workplane("XY").box(2,2,2)\n'
        'b=cq.Workplane("XY").box(1,1,1).translate((0,0,10))\nr=a.union(b)',
        tmp_path,
        SandboxConfig(),
    )
    assert not result.valid
    assert "2 disconnected solids" in result.error
    assert "bbox_mm" in result.error and "10.5" in result.error


def test_reviewer_receives_measured_sections_and_rejects_incomplete_response(tmp_path):
    import pytest
    import trimesh

    from da3_cad.feature_review import FeatureFinding, FeatureReview, review_features

    mesh = trimesh.creation.box(extents=(2, 3, 4))
    mesh.export(tmp_path / "model.stl")
    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            id="review",
            model="fake",
            usage=None,
            output_parsed=FeatureReview(
                findings=[
                    FeatureFinding(
                        feature="body",
                        severity=0,
                        photo_evidence="box",
                        cad_difference="none",
                        correction="retain",
                    )
                ],
                camera_caveats="",
            ),
        )

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result = review_features(client, GPTConfig(), [photo], tmp_path, "box", ["body"])
    assert result["cad_measurements"]["extents_xyz_mm"] == pytest.approx([2, 3, 4])
    assert result["cad_measurements"]["bottom_contact"]["span_xy_mm"] == pytest.approx([2, 3])
    assert all(
        s["span_xy_mm"] == pytest.approx([2, 3])
        for s in result["cad_measurements"]["horizontal_sections"]
    )
    assert "Measured CAD geometry" in str(calls[0]["input"])
    tapered = trimesh.creation.revolve([[0, 0], [1, 0], [2, 1], [2, 4], [0, 4]])
    tapered.export(tmp_path / "model.stl")
    measured = review_features(client, GPTConfig(), [photo], tmp_path, "foot", ["base"])
    cad = measured["cad_measurements"]
    assert cad["bottom_contact"]["span_xy_mm"] == pytest.approx([2, 2])
    assert (
        next(s for s in cad["horizontal_sections"] if s["height_fraction"] == 0.02)["span_xy_mm"][0]
        > 2.1
    )
    client.responses.parse = lambda **kwargs: SimpleNamespace(
        status="incomplete", output_parsed=None
    )
    with pytest.raises(RuntimeError, match="Feature review incomplete"):
        review_features(client, GPTConfig(), [photo], tmp_path, "box", ["body"])


def test_indentation_repair_preserves_parameters_and_does_not_relax_policy():
    import pytest

    from da3_cad.cad.ast_policy import AstPolicyError
    from da3_cad.gpt_cad import parameterize, repair_program_indentation

    candidate = CADResponse(
        name="box",
        code='import cadquery as cq\n width = 2.5\nr=cq.Workplane("XY").box(width,3,4)\n',
        parameters=[Parameter(name="width", value=2.5, unit="mm", source="specified")],
        assumptions=[],
    )
    repaired = repair_program_indentation(candidate)
    assert repaired.parameters == candidate.parameters
    assert "2.5" in parameterize(repaired)
    with pytest.raises(AstPolicyError):
        repair_program_indentation(
            candidate.model_copy(update={"code": " import os\nwidth=2.5\nr=width\n"})
        )


def test_cpu_budget_failure_is_reported_as_resource_limit(tmp_path, monkeypatch):
    import signal

    import da3_cad.cad.sandbox as sandbox
    from da3_cad.config import SandboxConfig

    monkeypatch.setattr(
        sandbox.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=-signal.SIGXCPU, communicate=lambda **kw: ("", "")
        ),
    )
    result = sandbox.validate_and_export(
        'import cadquery as cq\nr=cq.Workplane("XY").box(1,1,1)',
        tmp_path,
        SandboxConfig(cpu_seconds=5),
    )
    assert not result.valid
    assert "CPU budget of 5s" in result.error


def test_photo_mask_profiles_use_object_bounds_and_preserve_view_order(tmp_path):
    import numpy as np

    from da3_cad.feature_review import photo_mask_profiles

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    masks = np.zeros((2, 100, 120), dtype=bool)
    masks[0, 10:90, 30:90] = True
    masks[0, 70:90, 30:45] = False
    masks[0, 70:90, 75:90] = False
    masks[1, 20:80, 20:100] = True
    np.savez(evidence / "geometry.npz", masks=masks)
    profiles = photo_mask_profiles(tmp_path / "attempts/01")
    assert [p["view"] for p in profiles] == [1, 2]
    assert profiles[0]["bbox_width_px"] == 60
    assert profiles[0]["bands"][0]["projected_width_px"] == 30
    assert profiles[1]["bands"][0]["projected_width_px"] == 80
