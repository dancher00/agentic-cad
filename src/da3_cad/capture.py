"""Deterministic video key-frame extraction for object-centric capture."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "video capture requires OpenCV; install the DA3/video dependencies"
        ) from error
    return cv2


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _center_crop(
    bgr: npt.NDArray[np.uint8],
    fraction: float,
) -> npt.NDArray[np.uint8]:
    if not 0.25 <= fraction <= 1.0:
        raise ValueError("center_crop_fraction must be in [0.25,1.0]")
    if fraction == 1.0:
        return bgr
    height, width = bgr.shape[:2]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    return np.asarray(bgr[y0 : y0 + crop_height, x0 : x0 + crop_width], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class VideoFrameCandidate:
    frame_index: int
    timestamp_seconds: float
    bgr: npt.NDArray[np.uint8]
    descriptor: npt.NDArray[np.float32]
    blur_score: float
    exposure_score: float
    quality_score: float


@dataclass(frozen=True, slots=True)
class VideoFrameRecord:
    output_name: str
    frame_index: int
    timestamp_seconds: float
    blur_score: float
    exposure_score: float
    quality_score: float
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "output_name": self.output_name,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "blur_score": self.blur_score,
            "exposure_score": self.exposure_score,
            "quality_score": self.quality_score,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class VideoCaptureResult:
    frames_dir: Path
    manifest_path: Path
    selected: tuple[VideoFrameRecord, ...]
    report: dict[str, object]


def _frame_descriptor(
    bgr: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.float32], float, float, float]:
    cv2 = _cv2()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur = float(np.var(laplacian))
    mean_luma = float(np.mean(gray))
    exposure = float(np.clip(1.0 - abs(mean_luma - 127.5) / 127.5, 0.0, 1.0))
    quality = float(0.72 * np.tanh(np.log1p(max(blur, 0.0)) / 6.0) + 0.28 * exposure)
    small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    scale = float(np.linalg.norm(small))
    if scale > 1e-8:
        small /= scale
    return small.reshape(-1), blur, exposure, quality


def _candidate_indices(
    total_frames: int,
    *,
    start_frame: int,
    end_frame: int,
    count: int,
) -> npt.NDArray[np.int64]:
    available = end_frame - start_frame + 1
    if available <= 0:
        raise ValueError("video trim range contains no frames")
    actual = min(available, count)
    rounded = np.rint(np.linspace(start_frame, end_frame, num=actual)).astype(np.int64)
    return cast(npt.NDArray[np.int64], np.unique(rounded))


def _select_diverse(
    candidates: list[VideoFrameCandidate],
    *,
    count: int,
) -> list[VideoFrameCandidate]:
    if len(candidates) < count:
        raise ValueError(
            f"video yielded only {len(candidates)} decodable candidates for {count} views"
        )
    descriptors = np.stack([candidate.descriptor for candidate in candidates])
    qualities = np.asarray([candidate.quality_score for candidate in candidates], dtype=np.float64)
    times = np.asarray([candidate.timestamp_seconds for candidate in candidates], dtype=np.float64)
    time_span = max(float(np.ptp(times)), 1e-9)
    times = (times - times.min()) / time_span

    # Seed at a high-quality early frame, then greedily fill visual and temporal gaps.
    early_limit = max(1, math.ceil(len(candidates) * 0.15))
    selected = [int(np.argmax(qualities[:early_limit]))]
    while len(selected) < count:
        remaining = np.asarray(
            [index for index in range(len(candidates)) if index not in selected],
            dtype=np.int64,
        )
        visual = np.min(
            np.linalg.norm(
                descriptors[remaining, None, :] - descriptors[np.asarray(selected)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        temporal = np.min(
            np.abs(times[remaining, None] - times[np.asarray(selected)][None, :]),
            axis=1,
        )
        visual /= max(float(visual.max()), 1e-9)
        temporal /= max(float(temporal.max()), 1e-9)
        score = 0.55 * visual + 0.30 * temporal + 0.15 * qualities[remaining]
        selected.append(int(remaining[int(np.argmax(score))]))
    return sorted((candidates[index] for index in selected), key=lambda item: item.frame_index)


def _enable_display_orientation(cv2: Any, capture: Any) -> tuple[float, bool]:
    """Make decoded pixels match the video display orientation metadata."""

    meta_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if meta_property is None:
        return 0.0, False
    degrees = float(capture.get(meta_property))
    if not np.isfinite(degrees):
        degrees = 0.0
    requires_transform = not math.isclose(degrees % 360.0, 0.0, abs_tol=0.5)
    if not requires_transform:
        return degrees, False
    if auto_property is None or not bool(capture.set(auto_property, 1.0)):
        raise RuntimeError(
            f"video declares {degrees:g} degree display rotation, but this OpenCV "
            "backend cannot apply it safely"
        )
    applied = float(capture.get(auto_property)) >= 0.5
    if not applied:
        raise RuntimeError(
            f"video declares {degrees:g} degree display rotation, but OpenCV left "
            "orientation auto-application disabled"
        )
    return degrees, True


def extract_video_keyframes(
    video_path: Path,
    output_dir: Path,
    *,
    views: int = 16,
    candidate_multiplier: int = 5,
    center_crop_fraction: float = 1.0,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> VideoCaptureResult:
    """Extract a deterministic, diverse set of frames from a moving-camera video."""

    if views < 3 or views > 64:
        raise ValueError("video key-frame count must be in [3,64]")
    if candidate_multiplier < 1 or candidate_multiplier > 20:
        raise ValueError("candidate_multiplier must be in [1,20]")
    if not 0.25 <= center_crop_fraction <= 1.0:
        raise ValueError("center_crop_fraction must be in [0.25,1.0]")
    if start_seconds < 0.0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise ValueError("video trim times are invalid")
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise ValueError(f"video file does not exist: {video_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"video output directory is not empty: {output_dir}")

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")
    try:
        orientation_degrees, orientation_applied = _enable_display_orientation(cv2, capture)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not np.isfinite(fps) or fps <= 0.0 or total_frames <= 0:
            raise ValueError("video does not expose a finite FPS/frame-count index")
        duration = total_frames / fps
        if start_seconds >= duration:
            raise ValueError("video trim start is outside the video duration")
        effective_end = min(end_seconds if end_seconds is not None else duration, duration)
        start_frame = min(int(math.floor(start_seconds * fps)), total_frames - 1)
        end_frame = min(int(math.ceil(effective_end * fps)) - 1, total_frames - 1)
        indices = _candidate_indices(
            total_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            count=views * candidate_multiplier,
        )
        targets = set(int(index) for index in indices)
        candidates: list[VideoFrameCandidate] = []
        frame_index = 0
        last_target = int(indices[-1])
        while frame_index <= last_target:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in targets:
                cropped = _center_crop(
                    np.asarray(frame, dtype=np.uint8),
                    center_crop_fraction,
                )
                descriptor, blur, exposure, quality = _frame_descriptor(cropped)
                candidates.append(
                    VideoFrameCandidate(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        bgr=cropped,
                        descriptor=descriptor,
                        blur_score=blur,
                        exposure_score=exposure,
                        quality_score=quality,
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    selected = _select_diverse(candidates, count=views)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    records: list[VideoFrameRecord] = []
    for output_index, candidate in enumerate(selected):
        name = f"view_{output_index:03d}.png"
        frame_path = frames_dir / name
        if not cv2.imwrite(str(frame_path), candidate.bgr):
            raise RuntimeError(f"failed to write extracted frame: {name}")
        records.append(
            VideoFrameRecord(
                output_name=name,
                frame_index=candidate.frame_index,
                timestamp_seconds=candidate.timestamp_seconds,
                blur_score=candidate.blur_score,
                exposure_score=candidate.exposure_score,
                quality_score=candidate.quality_score,
                sha256=_file_sha256(frame_path),
            )
        )

    descriptors = np.stack([candidate.descriptor for candidate in selected])
    pairwise = np.linalg.norm(descriptors[:, None, :] - descriptors[None, :, :], axis=2)
    upper = pairwise[np.triu_indices(len(selected), k=1)]
    median_diversity = float(np.median(upper)) if len(upper) else 0.0
    warnings: list[str] = []
    if median_diversity < 0.12:
        warnings.append(
            "selected frames have low appearance diversity; the camera may not have completed "
            "a useful orbit"
        )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "source": {
            "path": str(video_path),
            "sha256": _file_sha256(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
            "resolution": [width, height],
            "display_orientation_degrees": orientation_degrees,
            "display_orientation_applied": orientation_applied,
        },
        "capture_contract": {
            "acquisition_mode": "stationary-object-moving-camera",
            "turntable_capture_supported": False,
            "requested_views": views,
            "candidate_multiplier": candidate_multiplier,
            "center_crop_fraction": center_crop_fraction,
            "output_resolution": [
                int(round(width * center_crop_fraction)),
                int(round(height * center_crop_fraction)),
            ],
            "trim_seconds": [start_seconds, effective_end],
            "selection": "quality-seeded greedy visual-plus-temporal farthest point",
        },
        "candidate_count": len(candidates),
        "selected": [record.as_dict() for record in records],
        "median_pairwise_descriptor_distance": median_diversity,
        "warnings": warnings,
    }
    manifest_path = output_dir / "capture.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return VideoCaptureResult(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        selected=tuple(records),
        report=report,
    )
