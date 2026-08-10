"""Pinned T-LESS acquisition metadata and safe archive extraction."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TlessArchive:
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TlessSource:
    repo_id: str
    revision: str
    license: str
    terms_url: str
    original_url: str
    archives: tuple[TlessArchive, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tless_manifest(path: Path) -> TlessSource:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported T-LESS source manifest schema")
    archives = tuple(
        TlessArchive(
            filename=str(item["filename"]),
            bytes=int(item["bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload["archives"]
    )
    if not archives or len({item.filename for item in archives}) != len(archives):
        raise ValueError("T-LESS archive filenames must be unique")
    for archive in archives:
        if archive.bytes <= 0:
            raise ValueError(f"invalid byte size for {archive.filename}")
        if len(archive.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in archive.sha256
        ):
            raise ValueError(f"invalid SHA-256 for {archive.filename}")
    return TlessSource(
        repo_id=str(payload["repo_id"]),
        revision=str(payload["revision"]),
        license=str(payload["license"]),
        terms_url=str(payload["terms_url"]),
        original_url=str(payload["original_url"]),
        archives=archives,
    )


def download_archive(source: TlessSource, archive: TlessArchive, destination: Path) -> None:
    """Download one pinned archive atomically and verify size plus SHA-256."""

    if (
        destination.is_file()
        and destination.stat().st_size == archive.bytes
        and sha256_file(destination) == archive.sha256
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(archive.filename)
    url = (
        f"https://huggingface.co/datasets/{source.repo_id}/resolve/"
        f"{source.revision}/{encoded}"
    )
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
    actual_bytes = partial.stat().st_size
    actual_sha256 = sha256_file(partial)
    if actual_bytes != archive.bytes or actual_sha256 != archive.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"download verification failed for {archive.filename}: expected "
            f"{archive.bytes} bytes {archive.sha256}, got {actual_bytes} {actual_sha256}"
        )
    os.replace(partial, destination)


def safe_extract_zip(archive: Path, destination: Path) -> tuple[str, ...]:
    """Extract a ZIP while rejecting traversal, links and duplicate members."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name in seen:
                raise ValueError(f"invalid or duplicate ZIP member: {name!r}")
            seen.add(name)
            target = (destination / name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe ZIP member path: {name}")
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise ValueError(f"symbolic links are forbidden in T-LESS ZIP: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source_stream, target.open("wb") as output_stream:
                while chunk := source_stream.read(8 * 1024 * 1024):
                    output_stream.write(chunk)
            extracted.append(name)
    return tuple(extracted)
