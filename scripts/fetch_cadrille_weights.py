"""Acquire pinned CC BY-NC 4.0 Cadrille weights after explicit opt-in."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from da3_cad.backends.cadrille import (
    CADRILLE_LICENSE_ACCEPTANCE,
    CADRILLE_MODELS,
    CADRILLE_PROCESSOR_ID,
    CADRILLE_PROCESSOR_REVISION,
    cadrille_license_notice,
    require_cadrille_terms,
    verified_cadrille_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("sft", "rl", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument("--accept-license")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = tuple(CADRILLE_MODELS) if args.profile == "all" else (args.profile,)
    print("Cadrille checkpoints are not redistributed by DA3-CAD:")
    for key in keys:
        print(f"- {cadrille_license_notice(CADRILLE_MODELS[key])}")
    print(
        f"- processor/tokenizer: {CADRILLE_PROCESSOR_ID}@{CADRILLE_PROCESSOR_REVISION} "
        "(Apache-2.0)"
    )
    print(f"- target: {args.cache_dir} (ignored runtime cache)")
    print(f"- required acknowledgement: --accept-license {CADRILLE_LICENSE_ACCEPTANCE}")
    if args.dry_run:
        return
    for key in keys:
        require_cadrille_terms(
            CADRILLE_MODELS[key],
            accepted_license=args.accept_license,
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the 'cadrille' optional dependencies first") from error

    reports: list[dict[str, object]] = []
    for key in keys:
        spec = CADRILLE_MODELS[key]
        for filename in ("config.json", "generation_config.json", "README.md"):
            hf_hub_download(
                repo_id=spec.model_id,
                filename=filename,
                revision=spec.revision,
                cache_dir=args.cache_dir,
            )
        report = verified_cadrille_checkpoint(
            spec,
            args.cache_dir,
            local_files_only=False,
        )
        reports.append({"model": spec.as_dict(), "checkpoint_file": report})
        print(f"verified {spec.model_id}: {report['sha256']}")

    AutoTokenizer.from_pretrained(
        CADRILLE_PROCESSOR_ID,
        revision=CADRILLE_PROCESSOR_REVISION,
        cache_dir=args.cache_dir,
        padding_side="left",
    )
    receipt_dir = args.cache_dir / "da3-cad-license-receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "accepted_at": datetime.now(UTC).isoformat(),
        "accepted_license": CADRILLE_LICENSE_ACCEPTANCE,
        "artifacts": reports,
        "processor": {
            "model_id": CADRILLE_PROCESSOR_ID,
            "revision": CADRILLE_PROCESSOR_REVISION,
            "license": "Apache-2.0",
        },
        "redistributed": False,
    }
    receipt_path = receipt_dir / f"cadrille-{args.profile}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote local acceptance receipt: {receipt_path}")


if __name__ == "__main__":
    main()
