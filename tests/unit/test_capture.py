from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from da3_cad.capture import extract_video_keyframes

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
    assert len(list(first.frames_dir.glob("*.png"))) == 8
