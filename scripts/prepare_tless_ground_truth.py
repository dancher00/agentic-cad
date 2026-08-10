#!/usr/bin/env python3
"""Materialize evaluator-safe T-LESS CAD GT with an explicit repair audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh

from da3_cad.benchmark.tless import sha256_file
from da3_cad.benchmark.tless_protocol import repair_official_cad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tless/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("data/tless/gt_cad"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/tless_primesense/gt_mesh_audit.json"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for object_id in range(1, 31):
        source = args.data_root / f"models_cad/obj_{object_id:06d}.ply"
        destination = args.output_root / f"obj_{object_id:06d}.ply"
        mesh = trimesh.load_mesh(source, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"official T-LESS CAD is not one mesh: {source}")
        repaired, repair = repair_official_cad(mesh)
        repaired.export(destination, file_type="ply", encoding="binary_little_endian")
        records.append(
            {
                "object_id": object_id,
                "source_path": source.relative_to(args.data_root).as_posix(),
                "source_sha256": sha256_file(source),
                "output_path": destination.as_posix(),
                "output_sha256": sha256_file(destination),
                **repair,
            }
        )
    report = {
        "schema_version": "1.0",
        "protocol": "da3-cad-tless-cad-gt-repair-v1",
        "source": "official T-LESS models_cad",
        "object_count": len(records),
        "repair_count": sum(bool(item["repair_applied"]) for item in records),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "objects": len(records),
                "repairs": report["repair_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
