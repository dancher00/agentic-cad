"""Convert accepted Objectron annotations into explicit target box prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from da3_cad.objectron import objectron_boxes_for_capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    parser.add_argument("capture_manifest", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=0)
    args = parser.parse_args()
    payload = objectron_boxes_for_capture(
        args.annotation,
        args.capture_manifest,
        args.images,
        args.output,
        object_id=args.object_id,
    )
    print(
        f"wrote {len(payload['views'])} Objectron localization boxes to {args.output}; "
        "these boxes are prompts, not masks or CAD ground truth"
    )


if __name__ == "__main__":
    main()
