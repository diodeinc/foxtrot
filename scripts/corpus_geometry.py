#!/usr/bin/env python3
"""Coarse geometry oracle for Foxtrot's corpus tests.

The metrics deliberately do not establish mesh equivalence.  They can catch
gross changes in extents or surface area, but not changed topology, holes,
self-intersections, triangle winding, or local geometric errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from typing import Any

_TRIANGLE = struct.Struct("<12fH")


def mesh_metrics(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Return JSON-safe metrics and validation results for a binary STL.

    Facet normals and attributes are ignored.  Signed volume depends on
    winding and is meaningful only for consistently oriented closed meshes.
    """
    with open(path, "rb") as stream:
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError(f"{path}: truncated binary STL header")
        triangle_count = struct.unpack_from("<I", header, 80)[0]
        expected_size = 84 + triangle_count * _TRIANGLE.size
        actual_size = os.fstat(stream.fileno()).st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path}: binary STL size is {actual_size}, expected {expected_size} "
                f"for {triangle_count} triangles"
            )

        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        area = 0.0
        volume = 0.0
        finite = True
        degenerate_count = 0

        for _ in range(triangle_count):
            record = stream.read(_TRIANGLE.size)
            values = _TRIANGLE.unpack(record)
            # Validate normals too, although they are not trusted for metrics.
            if not all(math.isfinite(value) for value in values[:12]):
                finite = False
                continue
            a, b, c = values[3:6], values[6:9], values[9:12]
            for vertex in (a, b, c):
                for axis, coordinate in enumerate(vertex):
                    minimum[axis] = min(minimum[axis], coordinate)
                    maximum[axis] = max(maximum[axis], coordinate)
            ab = tuple(b[i] - a[i] for i in range(3))
            ac = tuple(c[i] - a[i] for i in range(3))
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            double_area = math.sqrt(sum(component * component for component in cross))
            if double_area == 0.0:
                degenerate_count += 1
            area += 0.5 * double_area
            volume += (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            ) / 6.0

    valid = finite and triangle_count > 0 and degenerate_count == 0
    measurable = finite and triangle_count > 0
    return {
        "triangle_count": triangle_count,
        "bounds": {"min": minimum, "max": maximum} if measurable else None,
        "surface_area": area if measurable else None,
        "signed_volume": volume if measurable else None,
        "validation": {
            "valid": valid,
            "finite": finite,
            "degenerate_triangle_count": degenerate_count,
            "empty": triangle_count == 0,
        },
    }


def compare_meshes(
    actual_path: os.PathLike[str] | str,
    reference_path: os.PathLike[str] | str,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Coarsely compare STL bounds and area; volume is diagnostic only."""
    if any(
        not math.isfinite(t) or t < 0 for t in (relative_tolerance, absolute_tolerance)
    ):
        raise ValueError("tolerances must be finite and non-negative")
    actual = mesh_metrics(actual_path)
    reference = mesh_metrics(reference_path)
    valid = actual["validation"]["valid"] and reference["validation"]["valid"]

    if actual["bounds"] is not None and reference["bounds"] is not None:
        ref_min, ref_max = reference["bounds"]["min"], reference["bounds"]["max"]
        diagonal = math.sqrt(sum((ref_max[i] - ref_min[i]) ** 2 for i in range(3)))
        bounds_tolerance = absolute_tolerance + relative_tolerance * diagonal
        coordinate_differences = [
            abs(actual["bounds"][side][i] - reference["bounds"][side][i])
            for side in ("min", "max")
            for i in range(3)
        ]
        bounds_passed = all(
            value <= bounds_tolerance for value in coordinate_differences
        )
        area_difference = abs(actual["surface_area"] - reference["surface_area"])
        area_tolerance = absolute_tolerance**2 + relative_tolerance * abs(
            reference["surface_area"]
        )
        area_passed = area_difference <= area_tolerance
        volume_difference = abs(actual["signed_volume"] - reference["signed_volume"])
    else:
        diagonal = bounds_tolerance = area_difference = area_tolerance = (
            volume_difference
        ) = None
        coordinate_differences = None
        bounds_passed = area_passed = False

    return {
        "passed": bool(valid and bounds_passed and area_passed),
        "actual": actual,
        "reference": reference,
        "differences": {
            "bounds": {
                "coordinate_absolute": coordinate_differences,
                "reference_diagonal": diagonal,
                "tolerance": bounds_tolerance,
                "passed": bounds_passed,
            },
            "surface_area": {
                "absolute": area_difference,
                "tolerance": area_tolerance,
                "passed": area_passed,
            },
            "signed_volume": {"absolute": volume_difference, "diagnostic_only": True},
        },
        "limitations": (
            "Coarse bounds/area check only; it does not prove mesh equivalence or "
            "check topology, local shape, manifoldness, or winding. Volume is diagnostic."
        ),
    }


def step_to_stl(
    input_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str
) -> None:
    """Convert STEP to binary STL with OCCT, scaling STEP units to millimeters."""
    try:
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Interface import Interface_Static
        from OCP.STEPControl import STEPControl_Reader
        from OCP.StlAPI import StlAPI_Writer
    except ImportError as error:
        raise RuntimeError(
            "STEP conversion requires the optional 'cadquery-ocp' package "
            "(install with: python -m pip install cadquery-ocp)"
        ) from error

    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    reader = STEPControl_Reader()
    if reader.ReadFile(os.fspath(input_path)) != IFSelect_RetDone:
        raise RuntimeError(f"OCCT could not read STEP file: {input_path}")
    transferred = reader.TransferRoots()
    if transferred == 0 or reader.NbShapes() == 0:
        raise RuntimeError(f"STEP file contained no transferable shapes: {input_path}")
    shape = reader.OneShape()
    # Absolute 0.1 mm linear deflection, 0.1 rad angular deflection.
    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.1, True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not writer.Write(shape, os.fspath(output_path)):
        raise RuntimeError(f"OCCT failed to write STL file: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a STEP file to binary STL in millimeters"
    )
    parser.add_argument("input", help="input .step/.stp path")
    parser.add_argument("output", help="output .stl path")
    args = parser.parse_args(argv)
    try:
        step_to_stl(args.input, args.output)
        metrics = mesh_metrics(args.output)
        if not metrics["validation"]["valid"]:
            raise RuntimeError(
                f"OCCT produced an invalid mesh: {json.dumps(metrics['validation'])}"
            )
        return 0
    except Exception as error:
        print(f"corpus_geometry: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
