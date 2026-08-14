"""Internal worker for limited CadQuery execution; not a public CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

from da3_cad.cad.ast_policy import ALLOWED_BUILTINS, validate_source


def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    del globals_, locals_, fromlist, level
    if name != "cadquery":
        raise ImportError(f"sandbox import is not allowed: {name}")
    import cadquery

    return cadquery


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(source_path: Path, output_dir: Path) -> int:
    manifest = output_dir / "validation.json"
    try:
        import builtins

        import cadquery as cq

        source = source_path.read_text(encoding="utf-8")
        tree = validate_source(source)
        safe_builtins = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}
        safe_builtins["__import__"] = _safe_import
        namespace: dict[str, Any] = {
            "__builtins__": MappingProxyType(safe_builtins),
            "cq": cq,
        }
        exec(compile(tree, str(source_path), "exec"), namespace, namespace)
        result = namespace.get("r")
        if result is None:
            raise ValueError("program did not produce 'r'")
        shape = result.val() if hasattr(result, "val") else result
        if shape is None or not hasattr(shape, "isValid") or not shape.isValid():
            raise ValueError("program did not produce a valid CadQuery solid")
        solids = shape.Solids()
        solid_count = len(solids)
        if solid_count != 1:
            raise ValueError(
                f"program produced {solid_count} disconnected solids; "
                "single-part CAD requires exactly one"
            )
        volume = float(shape.Volume())
        if not volume > 0.0:
            raise ValueError("program produced a non-positive-volume shape")

        topology_invariants: dict[str, object] = {}
        non_penetration_shapes = (
            ("NON_PENETRATION_CAVITY", "non_penetration_cavity"),
            ("NON_PENETRATION_HANDLE_APERTURE", "non_penetration_handle_aperture"),
        )
        for namespace_name, report_name in non_penetration_shapes:
            non_penetration_shape = namespace.get(namespace_name)
            if non_penetration_shape is None:
                continue
            if not hasattr(result, "intersect"):
                raise ValueError("non-penetration invariant requires a CadQuery workplane")
            overlap = result.intersect(non_penetration_shape)
            overlap_shape = overlap.val() if hasattr(overlap, "val") else overlap
            intrusion_volume = 0.0 if overlap_shape is None else float(overlap_shape.Volume())
            tolerance = max(1e-12, 1e-9 * volume)
            if intrusion_volume > tolerance:
                raise ValueError(
                    f"generated solid penetrates {namespace_name}: "
                    f"{intrusion_volume:.12g} > {tolerance:.12g}"
                )
            topology_invariants[report_name] = {
                "intrusion_volume": intrusion_volume,
                "tolerance": tolerance,
                "passed": True,
            }
        required_union_overlap = namespace.get("REQUIRED_UNION_OVERLAP")
        if required_union_overlap is not None:
            overlap_shape = (
                required_union_overlap.val()
                if hasattr(required_union_overlap, "val")
                else required_union_overlap
            )
            overlap_volume = 0.0 if overlap_shape is None else float(overlap_shape.Volume())
            tolerance = max(1e-12, 1e-9 * volume)
            if overlap_volume <= tolerance:
                raise ValueError(
                    "generated features do not satisfy REQUIRED_UNION_OVERLAP: "
                    f"{overlap_volume:.12g} <= {tolerance:.12g}"
                )
            topology_invariants["required_union_overlap"] = {
                "overlap_volume": overlap_volume,
                "tolerance": tolerance,
                "passed": True,
            }

        bbox = shape.BoundingBox()
        bbox_values = [bbox.xmin, bbox.ymin, bbox.zmin, bbox.xmax, bbox.ymax, bbox.zmax]
        if not all(float("-inf") < float(value) < float("inf") for value in bbox_values):
            raise ValueError("program produced non-finite bounds")
        cq.exporters.export(result, str(output_dir / "model.step"))
        cq.exporters.export(
            result,
            str(output_dir / "model.stl"),
            tolerance=0.01,
            angularTolerance=0.1,
        )
        _write(
            manifest,
            {
                "valid": True,
                "volume": volume,
                "bbox": [float(value) for value in bbox_values],
                "solid_count": solid_count,
                "topology_invariants": topology_invariants,
            },
        )
        return 0
    except BaseException as error:  # worker must turn every candidate failure into a manifest
        _write(
            manifest,
            {
                "valid": False,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        return 2


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m da3_cad.cad.sandbox_worker SOURCE OUTPUT_DIR", file=sys.stderr)
        return 64
    return run(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
