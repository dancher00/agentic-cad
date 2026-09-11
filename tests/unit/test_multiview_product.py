from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from da3_cad.camera_render import render_camera
from da3_cad.cli import app
from da3_cad.multiview_evidence import camera_consistency


@pytest.mark.parametrize("count,mode", [(0, "gpt"), (1, "gpt"), (2, "hybrid")])
def test_auto_mode_selects_geometric_reconstruction_for_multiple_photos(
    tmp_path: Path, count: int, mode: str
) -> None:
    args = [
        "reconstruct",
        "--prompt",
        "one stationary part",
        "--dry-run",
        "--output",
        str(tmp_path / "output"),
    ]
    for i in range(count):
        photo = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 32), (i * 100, 0, 0)).save(photo)
        args.extend(["--image", str(photo)])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reconstruction"] == mode
    assert (payload["hybrid"] is not None) == (mode == "hybrid")
    assert not (tmp_path / "output").exists()
    explicit = CliRunner().invoke(app, [*args, "--reconstruction", "gpt"])
    assert explicit.exit_code == 0, explicit.output
    assert json.loads(explicit.output)["hybrid"] is None


def test_camera_render_uses_depth_occlusion_not_face_order() -> None:
    # Same projected triangle, different depths and orientations/shading.
    near = np.array([[-1.0, -1.0, 2.0], [0.0, 1.0, 2.0], [1.0, -1.0, 2.0]])
    far = near * 2
    vertices = np.concatenate([near, far])
    faces = np.array([[0, 1, 2], [5, 4, 3]])
    k = np.array([[20.0, 0, 20], [0, 20, 20], [0, 0, 1]])
    expected = np.asarray(render_camera(near, np.array([[0, 1, 2]]), k, np.eye(4), (40, 40)))
    for order in (faces, faces[::-1]):
        actual = np.asarray(render_camera(vertices, order, k, np.eye(4), (40, 40)))
        np.testing.assert_array_equal(actual, expected)
    assert np.any(expected != 238)
    moved = np.eye(4)
    moved[0, 3] = 1
    assert not np.array_equal(
        expected, np.asarray(render_camera(near, faces[:1], k, moved, (40, 40)))
    )


def test_cross_view_diagnostics_do_not_count_occlusion_as_missing_geometry() -> None:
    depth = np.full((2, 32, 32), 4.0)
    masks = np.zeros((2, 32, 32), dtype=bool)
    masks[:, 8:24, 8:24] = True
    cameras = np.repeat(np.eye(4)[None, :3], 2, axis=0)
    k = np.repeat(np.array([[[20.0, 0, 16], [0, 20, 16], [0, 0, 1]]]), 2, axis=0)
    good = camera_consistency(depth, masks, k, cameras)
    assert all(p["mask_containment"] == 1 for p in good["pairs"])
    assert good["estimated_max_view_separation_degrees"] == pytest.approx(0, abs=1e-5)
    # A closer occluder covers most of the second view; retain visible object support.
    depth[1, :, :20] = 2
    masks[1, :, :20] = False
    occluded = camera_consistency(depth, masks, k, cameras)
    forward = occluded["pairs"][0]
    assert forward["visible_samples"] == 64
    assert forward["mask_containment"] == 1
    # Incorrect target mask is a mismatch where the predicted surface is visible.
    masks[1] = False
    masks[1, :4, :4] = True
    mismatch = camera_consistency(depth, masks, k, cameras)
    assert mismatch["pairs"][0]["mask_containment"] == 0


def test_camera_diagnostics_reject_nonrigid_camera() -> None:
    cameras = np.eye(4)[None, :3]
    cameras[0, 0, 0] = 2
    with pytest.raises(ValueError, match="proper rotation"):
        camera_consistency(
            np.ones((1, 8, 8)), np.ones((1, 8, 8), dtype=bool), np.eye(3)[None], cameras
        )


def test_small_object_keeps_sampling_resolution_and_full_frame(tmp_path: Path) -> None:
    from da3_cad.hybrid_fit import GeometryObjective

    masks = np.zeros((2, 336, 252), dtype=bool)
    masks[:, 120:170, 100:140] = True
    depth = np.ones(masks.shape)
    k = np.repeat(np.array([[[200.0, 0, 126], [0, 200, 168], [0, 0, 1]]]), 2, axis=0)
    extrinsics = np.repeat(np.eye(4)[None, :3], 2, axis=0)
    np.savez(
        tmp_path / "geometry.npz", depth=depth, masks=masks, intrinsics=k, extrinsics=extrinsics
    )
    objective = GeometryObjective(tmp_path, resolution=192)
    target = objective.targets[0]
    ys, _ = np.where(target)
    assert np.ptp(ys) > 130  # Previously about 28 px when resizing the entire frame.
    assert target.shape[0] > 192  # Background retained, so excess CAD is penalized.
    assert objective.intrinsics[0, 0, 2] == pytest.approx(126 * target.shape[1] / 252)
    assert objective.intrinsics[0, 1, 2] == pytest.approx(168 * target.shape[0] / 336)


def test_shared_shape_score_penalizes_one_bad_view(tmp_path: Path, monkeypatch) -> None:
    import trimesh

    import da3_cad.hybrid_fit as fitting

    masks = np.ones((2, 8, 8), dtype=bool)
    np.savez(
        tmp_path / "geometry.npz",
        depth=np.ones(masks.shape),
        masks=masks,
        intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
        extrinsics=np.repeat(np.eye(4)[None, :3], 2, axis=0),
    )
    objective = fitting.GeometryObjective(tmp_path, resolution=8)
    mesh_path = tmp_path / "model.stl"
    trimesh.creation.box().export(mesh_path)
    objective.load_mesh(mesh_path)

    # Same mean overlap: balanced .75/.75 must beat front-only 1.0/.5.
    def score(counts):
        predictions = []
        for count in counts:
            mask = np.zeros((8, 8), dtype=bool)
            mask.flat[:count] = True
            predictions.append(mask)
        iterator = iter(predictions)
        monkeypatch.setattr(fitting, "rasterize", lambda *args: next(iterator))
        return objective.evaluate(np.zeros(7))

    balanced, unbalanced = score([48, 48]), score([64, 32])
    assert balanced["mean_silhouette_iou"] == unbalanced["mean_silhouette_iou"]
    assert balanced["loss"] < unbalanced["loss"]
    assert unbalanced["worst_silhouette_iou"] == 0.5


def test_perspective_depth_interpolation_handles_slanted_surfaces() -> None:
    k = np.array([[20.0, 0, 20], [0, 20, 20], [0, 0, 1]])
    screen = np.array([[10.0, 10, 1], [30.0, 10, 1], [20.0, 30, 1]])
    rays = screen @ np.linalg.inv(k).T
    slanted = rays * np.array([1.0, 10.0, 10.0])[:, None]
    flat = rays * 3
    camera = np.eye(4)
    first = np.array([[0, 1, 2]])
    only_slanted = np.asarray(render_camera(slanted, first, k, camera, (40, 40)))
    only_flat = np.asarray(render_camera(flat, first[:, ::-1], k, camera, (40, 40)))
    combined = np.asarray(
        render_camera(
            np.concatenate([slanted, flat]), np.array([[0, 1, 2], [5, 4, 3]]), k, camera, (40, 40)
        )
    )
    # At this pixel the slanted face is closer under reciprocal-depth interpolation,
    # but would incorrectly disappear behind z=3 with linear camera-Z interpolation.
    assert not np.array_equal(only_slanted[16, 20], only_flat[16, 20])
    np.testing.assert_array_equal(combined[16, 20], only_slanted[16, 20])


def test_camera_recovery_keeps_all_photos_and_orders_undistorted_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from da3_cad.geometry import cameras
    from da3_cad.multiview_evidence import recover_photo_cameras

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    names = ("02.png", "00.png", "01.png")
    for name in names:
        Image.new("RGB", (32, 32)).save(images / name)
    extrinsics = np.repeat(np.eye(4)[None], 3, axis=0)
    extrinsics[:, 0, 3] = [2, 0, 1]
    bundle = cameras.CameraBundle(
        names,
        np.repeat(np.eye(3)[None], 3, axis=0),
        extrinsics,
        "test-sfm",
        details={"mean_reprojection_error_pixels": 0.5},
    )
    calls = []

    def recover(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            bundle=bundle,
            registered_frames_dir=tmp_path / "rectified",
            registered_masks_dir=tmp_path / "rectified-masks",
        )

    monkeypatch.setattr(cameras, "recover_colmap_cameras", recover)
    frames, aligned_masks, selected, report = recover_photo_cameras(
        images, masks, tmp_path / "sfm", "auto"
    )
    assert calls[0]["minimum_registered_fraction"] == 1
    assert calls[0]["pairing"] == "exhaustive"
    assert calls[0]["masks_dir"] == masks
    assert frames.name == "rectified" and aligned_masks.name == "rectified-masks"
    assert selected.image_names == ("00.png", "01.png", "02.png")
    np.testing.assert_array_equal(selected.extrinsics[:, 0, 3], [0, 1, 2])
    assert report["source"] == "colmap"
    assert report["mean_reprojection_error_pixels"] == 0.5
    bundle.details["mean_reprojection_error_pixels"] = 3.0
    _, _, selected, rejected = recover_photo_cameras(images, masks, tmp_path / "sfm", "auto")
    assert selected is None and "2 pixels" in rejected["reason"]
    with pytest.raises(RuntimeError, match="2 pixels"):
        recover_photo_cameras(images, masks, tmp_path / "sfm", "colmap")


@pytest.mark.parametrize("reason", ["incomplete registration", "missing pycolmap"])
def test_camera_fallback_is_reported_and_explicit_colmap_cannot_silently_fallback(
    tmp_path, monkeypatch, reason
):
    from da3_cad.geometry import cameras
    from da3_cad.multiview_evidence import recover_photo_cameras

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32)).save(images / f"{i:02d}.png")

    def fail(*args, **kwargs):
        raise RuntimeError(reason)

    monkeypatch.setattr(cameras, "recover_colmap_cameras", fail)
    frames, aligned_masks, selected, report = recover_photo_cameras(
        images, masks, tmp_path / "sfm", "auto"
    )
    assert frames == images and aligned_masks == masks and selected is None
    assert report["source"] == "da3" and report["reason"] == reason
    with pytest.raises(RuntimeError, match="COLMAP camera recovery failed"):
        recover_photo_cameras(images, masks, tmp_path / "sfm", "colmap")


def test_explicit_camera_source_cannot_reuse_incompatible_evidence(tmp_path):
    import hashlib

    from da3_cad.hybrid_evidence import HybridConfig, prepare_evidence

    photo = tmp_path / "photo.png"
    Image.new("RGB", (32, 32)).save(photo)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "evidence.json").write_text(
        json.dumps(
            {
                "input_identity": {
                    "prompt": "part",
                    "sha256": [hashlib.sha256(photo.read_bytes()).hexdigest()],
                },
                "cameras": {"source": "da3"},
            }
        )
    )
    with pytest.raises(ValueError, match="different camera source"):
        prepare_evidence(
            [photo],
            "part",
            tmp_path / "output",
            None,
            None,
            HybridConfig(evidence_cache=cache, cameras="colmap"),
        )
    assert not (tmp_path / "output").exists()


def test_search_proxy_preserves_visible_aperture_and_verification_uses_export(tmp_path):
    import hashlib

    import trimesh

    from da3_cad.hybrid_fit import GeometryObjective, rasterize, search_mesh, silhouette_iou

    mesh = trimesh.creation.annulus(r_min=0.6, r_max=1, height=0.5, sections=1024)
    vertices, faces = search_mesh(mesh.vertices, mesh.faces, 96)
    assert len(faces) < len(mesh.faces) / 2
    k = np.array([[100.0, 0, 60], [0, 100, 60], [0, 0, 1]])
    camera = np.eye(4)
    camera[2, 3] = 3
    coarse = rasterize(vertices, faces, k, camera, (120, 120))
    exact = rasterize(mesh.vertices, mesh.faces, k, camera, (120, 120))
    assert not coarse[60, 60] and not exact[60, 60]
    assert silhouette_iou(coarse, exact) > 0.95
    path = tmp_path / "model.stl"
    mesh.export(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez(
        tmp_path / "geometry.npz",
        depth=np.ones((1, 120, 120)),
        masks=exact[None],
        intrinsics=k[None],
        extrinsics=camera[None],
    )
    search, verification = GeometryObjective(tmp_path, 96), GeometryObjective(tmp_path, 192)
    search.load_mesh(path)
    verification.load_mesh(path)
    assert len(search.faces) < len(verification.faces)
    assert len(verification.faces) == verification.export_face_count == len(mesh.faces)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_cached_mask_upgrade_keeps_cameras_depth_and_original_cache(tmp_path, monkeypatch):
    import hashlib

    from da3_cad.hybrid_evidence import HybridConfig, prepare_evidence
    from da3_cad.segmentation import sam2_box

    cache = tmp_path / "cache"
    (cache / "images").mkdir(parents=True)
    (cache / "masks").mkdir()
    paths = []
    for i in range(2):
        photo = cache / "images" / f"{i:02d}.png"
        Image.new("RGB", (32, 32), (i * 100, 10, 10)).save(photo)
        paths.append(photo)
    report = {
        "input_identity": {
            "prompt": "part",
            "sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        },
        "segmentation": "masks/segmentation.json",
        "panels": ["evidence-00.png", "evidence-01.png"],
        "views": [{}, {}],
    }
    (cache / "evidence.json").write_text(json.dumps(report))
    (cache / "boxes.json").write_text("{}")
    (cache / "masks/segmentation.json").write_text(json.dumps({"schema_version": "v2"}))
    arrays = dict(
        depth=np.ones((2, 336, 336)),
        masks=np.ones((2, 336, 336), dtype=bool),
        intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
        extrinsics=np.repeat(np.eye(4)[None, :3], 2, axis=0),
    )
    np.savez(cache / "geometry.npz", **arrays)
    before = (cache / "geometry.npz").read_bytes()

    def segment(images, boxes, output, **kwargs):
        output.mkdir()
        for i in range(2):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            Image.fromarray(mask).save(output / f"{i:02d}.png")
        (output / "segmentation.json").write_text(
            json.dumps({"schema_version": "da3-cad-sam2-box-segmentation-v3"})
        )

    monkeypatch.setattr(sam2_box, "segment_box_prompts_sam2", segment)
    result = prepare_evidence(
        paths, "part", tmp_path / "new", None, None, HybridConfig(evidence_cache=cache)
    )
    assert result["mask_refresh"]["cad_used"] is False
    assert (cache / "geometry.npz").read_bytes() == before
    with np.load(tmp_path / "new/geometry.npz") as updated:
        for key in ("depth", "intrinsics", "extrinsics"):
            np.testing.assert_array_equal(updated[key], arrays[key])
        assert not np.array_equal(updated["masks"], arrays["masks"])
    assert (tmp_path / "new/evidence-01.png").is_file()
