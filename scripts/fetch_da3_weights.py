"""Acquire pinned DA3 checkpoints with displayed terms and complete SHA verification."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from da3_cad.backends.da3 import (
    DA3_MODELS,
    da3_license_notice,
    require_weight_terms,
    verified_da3_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("base", "large", "metric-large", "all"),
        default="all",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = tuple(DA3_MODELS) if args.profile == "all" else (args.profile,)
    print("DA3 checkpoints are not redistributed by DA3-CAD:")
    for key in keys:
        print(f"- {da3_license_notice(DA3_MODELS[key])}")
    print(f"- target: {args.cache_dir} (ignored runtime cache)")
    if any(DA3_MODELS[key].noncommercial for key in keys):
        print("- required acknowledgement: --accept-noncommercial-weights")
    if args.dry_run:
        return
    for key in keys:
        require_weight_terms(
            DA3_MODELS[key],
            accepted_noncommercial=args.accept_noncommercial_weights,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("install the DA3 dependencies before fetching checkpoints") from error

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for key in keys:
        spec = DA3_MODELS[key]
        hf_hub_download(
            repo_id=spec.model_id,
            filename="config.json",
            revision=spec.revision,
            cache_dir=args.cache_dir,
        )
        checkpoint = verified_da3_checkpoint(
            spec,
            args.cache_dir,
            local_files_only=False,
        )
        reports.append({"model": spec.as_dict(), "checkpoint_file": checkpoint})
        print(f"verified {spec.model_id}: {checkpoint['sha256']}")

    receipt_dir = args.cache_dir / "da3-cad-license-receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "accepted_at": datetime.now(UTC).isoformat(),
        "accepted_noncommercial_weights": args.accept_noncommercial_weights,
        "artifacts": reports,
        "redistributed": False,
    }
    receipt_path = receipt_dir / f"da3-{args.profile}.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote local acquisition receipt: {receipt_path}")


if __name__ == "__main__":
    main()
