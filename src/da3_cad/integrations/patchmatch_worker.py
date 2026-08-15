"""Isolated CUDA PatchMatch worker for the dedicated MVS environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--max-image-size", type=int, default=800)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gpu-index", default="0")
    args = parser.parse_args()
    if args.max_image_size < 64 or args.iterations < 1:
        raise ValueError("PatchMatch image size and iteration count must be positive")
    report_path = args.workspace / "workspace_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    depth_min = float(report["depth_min"])
    depth_max = float(report["depth_max"])
    if not 0.0 < depth_min < depth_max:
        raise ValueError("workspace contains an invalid explicit depth range")

    import pycolmap

    if _version_tuple(pycolmap.__version__) < (4, 1, 1):
        raise RuntimeError("Blackwell-safe dense MVS requires PyCOLMAP >= 4.1.1")
    if not pycolmap.has_cuda:
        raise RuntimeError("PatchMatch worker requires a CUDA-enabled PyCOLMAP build")
    existing = tuple((args.workspace / "stereo" / "depth_maps").glob("*.bin"))
    if existing:
        raise ValueError(f"refusing to overwrite {len(existing)} existing depth maps")

    options = pycolmap.PatchMatchOptions()
    options.max_image_size = args.max_image_size
    options.gpu_index = args.gpu_index
    options.num_iterations = args.iterations
    options.geom_consistency = True
    options.depth_min = depth_min
    options.depth_max = depth_max
    started = time.monotonic()
    pycolmap.patch_match_stereo(args.workspace, options=options)
    elapsed = time.monotonic() - started
    geometric = tuple((args.workspace / "stereo" / "depth_maps").glob("*.geometric.bin"))
    if len(geometric) != int(report["images"]):
        raise RuntimeError(
            f"PatchMatch wrote {len(geometric)} geometric maps for {report['images']} images"
        )
    result = {
        "schema_version": "da3-cad-patchmatch-worker-v1",
        "pycolmap_version": pycolmap.__version__,
        "pycolmap_cuda": True,
        "maximum_image_size": args.max_image_size,
        "iterations": args.iterations,
        "gpu_index": args.gpu_index,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "geometric_depth_maps": len(geometric),
        "elapsed_seconds": elapsed,
        "reference_geometry_access": False,
    }
    output = args.workspace / "patchmatch_report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
