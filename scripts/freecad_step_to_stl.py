#!/usr/bin/env python3
"""FreeCAD-side deterministic STEP/STP tessellation helper.

Run this script through ``FreeCADCmd``/``freecadcmd``, not ordinary Python.
The caller normalizes FreeCAD's STL output and performs exact face chunking;
this helper performs no decimation or mesh simplification.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--linear-deflection-mm", required=True, type=float)
    parser.add_argument("--angular-deflection-deg", required=True, type=float)
    args = parser.parse_args(argv)

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if source.suffix.lower() not in (".step", ".stp"):
        parser.error(f"input must be STEP/STP, got {source}")
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    if not math.isfinite(args.linear_deflection_mm) or args.linear_deflection_mm <= 0:
        parser.error("--linear-deflection-mm must be finite and positive")
    if (not math.isfinite(args.angular_deflection_deg)
            or not 0 < args.angular_deflection_deg < 180):
        parser.error("--angular-deflection-deg must lie in (0, 180)")

    try:
        import FreeCAD  # type: ignore
        import Part  # type: ignore
        import MeshPart  # type: ignore
    except ImportError as error:
        print(
            "This helper must run inside FreeCAD's Python environment. Invoke "
            "FreeCADCmd/freecadcmd or use scripts/prepare_project_cad.py.",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    document = FreeCAD.newDocument("handoff_step_conversion")
    try:
        Part.insert(str(source), document.Name)
        document.recompute()
        shapes = [obj.Shape for obj in document.Objects
                  if hasattr(obj, "Shape") and not obj.Shape.isNull()]
        if not shapes:
            raise RuntimeError(f"FreeCAD imported no solid/surface shapes from {source}")
        shape = shapes[0] if len(shapes) == 1 else Part.makeCompound(shapes)
        mesh = MeshPart.meshFromShape(
            Shape=shape,
            LinearDeflection=args.linear_deflection_mm,
            AngularDeflection=math.radians(args.angular_deflection_deg),
            Relative=False,
        )
        if mesh.CountFacets <= 0:
            raise RuntimeError(f"FreeCAD generated no triangles from {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(str(destination))
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"FreeCAD did not write a valid STL to {destination}")
        print(f"FreeCAD tessellated {source.name}: {mesh.CountFacets} faces -> {destination}")
    finally:
        FreeCAD.closeDocument(document.Name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
