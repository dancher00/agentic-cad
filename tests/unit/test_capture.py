from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from da3_cad.capture import _enable_display_orientation, extract_video_keyframes

cv2 = pytest.importorskip("cv2")


def _write_test_video(path: Path, *, frames: int = 48, fps: float = 12.0) -> None:
    width, height = 160, 120
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer")
    try:
        yy, xx = np.indices((height, width))
        checker = (((xx // 10 + yy // 10) % 2) * 35).astype(np.uint8)
        for index in range(frames):
            frame = np.stack(
                (
                    (checker + index * 3) % 255,
                    (checker + 60 + index * 5) % 255,
                    (checker + 120 + index * 7) % 255,
                ),
                axis=2,
            ).astype(np.uint8)
            center_x = 20 + (index * 3) % 120
            center_y = 60 + int(25 * np.sin(index / 5.0))
            cv2.circle(frame, (center_x, center_y), 15, (255, 255, 255), 2)
            cv2.line(
                frame,
                (index % width, 0),
                ((index * 4) % width, height - 1),
                (0, 0, 0),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV did not persist the synthetic test video")


def test_video_keyframes_are_diverse_ordered_and_repeatable(tmp_path: Path) -> None:
    video = tmp_path / "orbit.avi"
    _write_test_video(video)

    first = extract_video_keyframes(
        video,
        tmp_path / "first",
        views=8,
        candidate_multiplier=3,
    )
    second = extract_video_keyframes(
        video,
        tmp_path / "second",
        views=8,
        candidate_multiplier=3,
    )

    assert [record.output_name for record in first.selected] == [
        f"view_{index:03d}.png" for index in range(8)
    ]
    indices = [record.frame_index for record in first.selected]
    assert indices == sorted(indices)
    assert len(set(indices)) == 8
    assert indices == [record.frame_index for record in second.selected]
    assert [record.sha256 for record in first.selected] == [
        record.sha256 for record in second.selected
    ]
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_contract"]["acquisition_mode"] == ("stationary-object-moving-camera")
    assert manifest["capture_contract"]["turntable_capture_supported"] is False
    assert manifest["capture_contract"]["center_crop_fraction"] == 1.0
    assert manifest["capture_contract"]["output_resolution"] == [160, 120]
    assert manifest["source"]["display_orientation_degrees"] == 0.0
    assert manifest["source"]["display_orientation_applied"] is False
    assert len(list(first.frames_dir.glob("*.png"))) == 8


def test_video_keyframes_can_be_center_cropped(tmp_path: Path) -> None:
    video = tmp_path / "orbit.avi"
    _write_test_video(video)

    result = extract_video_keyframes(
        video,
        tmp_path / "cropped",
        views=6,
        candidate_multiplier=2,
        center_crop_fraction=0.5,
    )

    frame = cv2.imread(str(result.frames_dir / "view_000.png"))
    assert frame.shape[:2] == (60, 80)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_contract"]["center_crop_fraction"] == 0.5
    assert manifest["capture_contract"]["output_resolution"] == [80, 60]


def test_nonzero_video_orientation_is_applied_or_fails_closed() -> None:
    class FakeCv2:
        CAP_PROP_ORIENTATION_META = 48
        CAP_PROP_ORIENTATION_AUTO = 49

    class FakeCapture:
        auto = 0.0

        def get(self, key: int) -> float:
            return 90.0 if key == 48 else self.auto

        def set(self, key: int, value: float) -> bool:
            if key != 49:
                return False
            self.auto = value
            return True

    degrees, applied = _enable_display_orientation(FakeCv2(), FakeCapture())

    assert degrees == 90.0
    assert applied is True
