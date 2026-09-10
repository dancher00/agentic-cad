from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from da3_cad.photo_cad import run_photo_cad
from da3_cad.segmentation.query_context import parse_target_description
from da3_cad.segmentation.text_object import select_instance


def test_selector_rejects_two_competing_objects_but_accepts_duplicate_boxes() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        select_instance([[0, 0, 10, 10], [20, 0, 30, 10]], [0.9, 0.87])
    assert select_instance([[0, 0, 10, 10], [1, 1, 10, 10]], [0.9, 0.89]) == 0
    with pytest.raises(ValueError, match="not found"):
        select_instance([], [])


@pytest.mark.parametrize(
    "response",
    [
        '{"present":false,"ambiguous":false,"description":"bottle"}',
        '{"present":true,"ambiguous":true,"description":"bottle"}',
        '{"present":"true","ambiguous":false,"description":"bottle"}',
        '__import__("os").system("false")',
    ],
)
def test_vlm_response_is_strict_data_and_requires_unique_visible_target(response: str) -> None:
    with pytest.raises(ValueError):
        parse_target_description(response)


def test_vlm_description_accepts_json() -> None:
    assert (
        parse_target_description(
            '```json\n{"present":true,"ambiguous":false,"description":"red soda can"}\n```'
        )
        == "red soda can"
    )


def _images(tmp_path: Path, count: int = 3) -> Path:
    source = tmp_path / "images"
    source.mkdir()
    for i in range(count):
        Image.new("RGB", (32, 32), (40 * i, 80, 100)).save(source / f"{i}.png")
    return source


def test_single_image_is_rejected_before_models_or_output_creation(tmp_path: Path) -> None:
    source = _images(tmp_path, 1)
    with pytest.raises(ValueError, match="at least three"):
        run_photo_cad(source, tmp_path / "result", "bottle")
    assert not (tmp_path / "result").exists()


def test_draft_cannot_claim_accept_and_vlm_query_reaches_segmenter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _images(tmp_path)
    output = tmp_path / "result"
    seen: list[str] = []

    def segment(images: Path, directory: Path, query: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(query)
        (directory / "masks").mkdir(parents=True)
        return {"status": "masked"}

    def command(args: list[str], **kwargs: Any) -> SimpleNamespace:
        cad = Path(args[args.index("--output") + 1])
        cad.mkdir()
        for suffix in ("step", "py", "stl"):
            (cad / f"model.{suffix}").write_text("test artifact")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("da3_cad.segmentation.text_object.segment_text_object", segment)
    monkeypatch.setattr(
        "da3_cad.segmentation.query_context.resolve_query",
        lambda *a, **k: {"description": "red bottle"},
    )
    monkeypatch.setattr("da3_cad.photo_cad.subprocess.run", command)
    report = run_photo_cad(source, output, "красная бутылка", geometry="da3")
    assert seen == ["red bottle"]
    assert report["status"] == "CANDIDATE"
    assert (output / "candidate.step").exists()
    assert not (output / "model.step").exists()
    assert json.loads((output / "report.json").read_text())["status"] == "CANDIDATE"


def test_selection_failure_is_preserved_and_no_geometry_is_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _images(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("ambiguous target")

    monkeypatch.setattr("da3_cad.segmentation.text_object.segment_text_object", fail)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="ambiguous"):
        run_photo_cad(source, output, "bottle", use_vlm=False)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED"
    assert not report["stages"]


def test_mvs_preserves_interpreter_symlink_and_abstention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _images(tmp_path)
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to("/usr/bin/python3")
    seen: list[list[str]] = []

    def segment(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "masked"}

    def command(args: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(args)
        if "fit-cad" in args:
            directory = Path(args[args.index("--output") + 1])
            directory.mkdir()
            (directory / "candidate.step").write_text("test candidate")
            return SimpleNamespace(returncode=3)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("da3_cad.segmentation.text_object.segment_text_object", segment)
    monkeypatch.setattr("da3_cad.photo_cad.subprocess.run", command)
    monkeypatch.setattr("da3_cad.photo_cad.prepare_stereo_resolution", lambda *a: None)
    report = run_photo_cad(
        source, tmp_path / "result", "book", use_vlm=False, mvs_python=executable
    )
    dense = next(args for args in seen if "dense-surface" in args)
    assert dense[dense.index("--mvs-python") + 1] == str(executable.absolute())
    assert report["status"] == "ABSTAIN"
    assert report["step"] == "candidate.step"


def test_stereo_resize_preserves_projected_rays_and_binary_masks(tmp_path: Path) -> None:
    import numpy as np

    from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
    from da3_cad.observations import load_observations
    from da3_cad.photo_cad import prepare_stereo_resolution

    target = tmp_path / "target"
    target.mkdir()
    images = _images(target)
    masks = target / "masks"
    masks.mkdir()
    for image in images.glob("*.png"):
        Image.new("L", (32, 32), 255).save(masks / image.name)
    k = np.tile(np.array([[20.0, 0, 16], [0, 20, 16], [0, 0, 1]]), (3, 1, 1))
    e = np.tile(np.eye(4), (3, 1, 1))
    CameraBundle(("0.png", "1.png", "2.png"), k, e, "test").save(target / "cameras.npz")
    output = tmp_path / "resized"
    prepare_stereo_resolution(target, output, maximum=16)
    bundle = load_camera_bundle(output / "cameras.npz", load_observations(output / "images"))
    point = np.array([0.3, -0.2, 2.0])
    original = k[0] @ point
    projected = bundle.intrinsics[0] @ point
    np.testing.assert_allclose(projected[:2] / projected[2], original[:2] / original[2] / 2)
    np.testing.assert_array_equal(bundle.extrinsics, e)
    with Image.open(output / "masks/0.png") as mask:
        assert mask.size == (16, 16)
        assert set(np.unique(mask)) <= {0, 255}
