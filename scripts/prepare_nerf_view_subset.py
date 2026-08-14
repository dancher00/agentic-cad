#!/usr/bin/env python3
"""Create a symlinked NeRF view subset from a DA3 prior manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from da3_cad.integrations.gaussian_depth_prior import DepthPriorBundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("transforms", type=Path)
    parser.add_argument("output_dir", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prior", type=Path, help="use selected_indices from a prior NPZ")
    group.add_argument("--indices", nargs="+", type=int)
    return parser


def _image_source(root: Path, relative_without_suffix: str) -> Path:
    candidate = (root / relative_without_suffix).resolve()
    if candidate.suffix and candidate.is_file():
        return candidate
    for suffix in (".png", ".jpg", ".jpeg"):
        resolved = candidate.with_suffix(suffix)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"view asset is missing: {candidate}")


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination)


def main() -> None:
    args = _parser().parse_args()
    transforms = args.transforms.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing subset: {output}")
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("NeRF transforms JSON contains no frames")
    if args.prior is not None:
        prior = DepthPriorBundle.load(args.prior)
        raw_indices = prior.metadata.get("selected_indices")
        if not isinstance(raw_indices, list):
            raise ValueError("depth prior has no selected_indices metadata")
        indices = tuple(int(item) for item in raw_indices)
    else:
        indices = tuple(args.indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("view-subset indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise ValueError("view-subset index is out of range")

    output.mkdir(parents=True)
    selected_frames = [frames[index] for index in indices]
    source_root = transforms.parent
    companion_directories = ("corner_img", "edge_img", "mask_img", "line_mask_img")
    for frame in selected_frames:
        relative = str(frame["file_path"]).removeprefix("./")
        image = _image_source(source_root, relative)
        _symlink(image, (output / relative).with_suffix(image.suffix))
        if relative.startswith("train_img/"):
            for directory in companion_directories:
                companion_relative = relative.replace("train_img/", f"{directory}/", 1)
                companion = _image_source(source_root, companion_relative)
                _symlink(companion, (output / companion_relative).with_suffix(companion.suffix))

    subset_payload = dict(payload)
    subset_payload["frames"] = selected_frames
    (output / transforms.name).write_text(
        json.dumps(subset_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    points = source_root / "points3d.ply"
    if points.is_file():
        _symlink(points, output / points.name)
    print(
        json.dumps(
            {
                "output": str(output),
                "view_count": len(indices),
                "selected_indices": list(indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
