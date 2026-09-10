"""Internal worker for limited CadQuery execution; not a public CLI."""

from __future__ import annotations

import ast
import json
import sys
import traceback
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
            components = []
            for solid in solids[:8]:
                bounds = solid.BoundingBox()
                components.append(
                    {
                        "volume_mm3": round(float(solid.Volume()), 6),
                        "bbox_mm": [
                            round(float(v), 6)
                            for v in (
                                bounds.xmin,
                                bounds.ymin,
                                bounds.zmin,
                                bounds.xmax,
                                bounds.ymax,
                                bounds.zmax,
                            )
                        ],
                    }
                )
            raise ValueError(
                f"program produced {solid_count} disconnected solids; "
                "single-part CAD requires exactly one. Component locations: "
                + json.dumps(components)
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
            clearance = (
                non_penetration_shape.val()
                if hasattr(non_penetration_shape, "val")
                else non_penetration_shape
            )
            if (
                clearance is None
                or not hasattr(clearance, "Volume")
                or not clearance.isValid()
                or clearance.Volume() <= 1e-9
            ):
                raise ValueError(f"{namespace_name} must be a nonempty valid clearance solid")
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
        # CadQuery's generic STL exporter uses relative deflection. Use an
        # absolute millimeter tolerance so large faces keep curved outlines.
        shape.exportStl(
            str(output_dir / "model.stl"),
            tolerance=0.05,
            angularTolerance=0.2,
            relative=False,
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
        message = f"{type(error).__name__}: {error}"
        frames = traceback.extract_tb(error.__traceback__)
        program_frames = [frame for frame in frames if frame.filename == str(source_path)]
        if program_frames:
            frame = program_frames[-1]
            message += f"; generated program line {frame.lineno}: {(frame.line or '')[:240]}"
        invalid_intermediates = []
        for name, value in locals().get("namespace", {}).items():
            if name.startswith("_") or not hasattr(value, "val"):
                continue
            try:
                intermediate = value.val()
                if (
                    hasattr(intermediate, "Solids")
                    and intermediate.Solids()
                    and not intermediate.isValid()
                ):
                    invalid_intermediates.append(name)
            except Exception:
                continue
        if invalid_intermediates:
            message += "; invalid intermediate solids: " + ", ".join(invalid_intermediates[:12])
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "spline"
            and not any(keyword.arg == "includeCurrent" for keyword in node.keywords)
            for node in ast.walk(locals().get("tree", ast.Module(body=[], type_ignores=[])))
        ):
            message += (
                "; check spline continuity: CadQuery 2.4 excludes the current point by default. "
                "When extending a profile, use includeCurrent=True and omit the current point "
                "from the spline point list. Otherwise the wire has a gap and cannot revolve."
            )
        _write(
            manifest,
            {
                "valid": False,
                "error": message,
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
