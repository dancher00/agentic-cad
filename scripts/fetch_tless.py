#!/usr/bin/env python3
"""Fetch the pinned, non-redistributed T-LESS Primesense BOP release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da3_cad.benchmark.tless import (
    download_archive,
    load_tless_manifest,
    safe_extract_zip,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/tless.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/tless"))
    parser.add_argument("--accept-license", default=None)
    args = parser.parse_args()

    source = load_tless_manifest(args.manifest)
    print("T-LESS data are external and are not redistributed by DA3-CAD:")
    print(f"- dataset: https://huggingface.co/datasets/{source.repo_id}")
    print(f"- immutable revision: {source.revision}")
    print(f"- terms: {source.license} ({source.terms_url})")
    print(f"- original dataset and attribution: {source.original_url}")
    for archive in source.archives:
        print(f"- {archive.filename}: {archive.bytes} bytes, sha256={archive.sha256}")
    if args.accept_license != "cc-by-4.0":
        parser.error("pass --accept-license cc-by-4.0 only after reviewing the terms above")

    archive_root = args.data_root / "archives"
    extracted_root = args.data_root / "extracted"
    receipts_root = args.data_root / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    for archive in source.archives:
        archive_path = archive_root / archive.filename
        download_archive(source, archive, archive_path)
        print(f"verified {archive_path}")
        receipt_path = receipts_root / f"{archive.filename}.json"
        expected_receipt = {
            "archive": archive.filename,
            "archive_bytes": archive.bytes,
            "archive_sha256": archive.sha256,
            "dataset_revision": source.revision,
            "license": source.license,
            "redistributed": False,
        }
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing == expected_receipt:
                print(f"verified prior extraction receipt {receipt_path}")
                continue
        extracted = safe_extract_zip(archive_path, extracted_root)
        if not extracted:
            raise ValueError(f"archive contains no regular files: {archive_path}")
        receipt_path.write_text(
            json.dumps(expected_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"extracted {len(extracted)} files from {archive.filename}")
        if sha256_file(archive_path) != archive.sha256:
            raise RuntimeError(f"archive changed during extraction: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
