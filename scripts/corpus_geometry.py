#!/usr/bin/env python3
"""OCCT geometry oracle: aggregate metrics and sampled surface distances.

Agreement is evidence of geometric similarity, not proof of mesh equivalence,
topology, watertightness, winding, or shading correctness.
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
        "comparable": measurable and area > 0,
        "validation": {
            "valid": valid,
            "finite": finite,
            "degenerate_triangle_count": degenerate_count,
            "empty": triangle_count == 0,
        },
    }


def surface_distances(actual_path, reference_path, samples, tolerance):
    """Seeded area samples plus bounded face probes, queried against triangles.

    Never align, normalize, repair, or rewrite the meshes. Zero-area STL facets
    have no surface to sample; their counts remain in the aggregate diagnostics.
    """
    try:
        import numpy as np
        import rtree  # noqa: F401 -- required by trimesh's triangle index
        import scipy  # noqa: F401 -- required by trimesh proximity queries
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "surface comparison requires numpy, trimesh, scipy and rtree; "
            "install with: python -m pip install trimesh scipy rtree"
        ) from error

    def load(path):
        mesh = trimesh.load_mesh(path, file_type="stl", process=False)
        mesh.update_faces(mesh.area_faces > 0)
        return mesh

    def directed(source, target):
        rng = np.random.default_rng(0)
        faces = rng.choice(
            len(source.faces), samples, p=source.area_faces / source.area
        )
        triangles = source.triangles[faces]
        root = np.sqrt(rng.random(samples))
        v = rng.random(samples)
        weights = np.column_stack((1 - root, root * (1 - v), root * v))
        points = np.einsum("ij,ijk->ik", weights, triangles)
        # Small faces can disappear in area sampling; probe their centroids too.
        probes = np.linspace(
            0, len(source.faces) - 1, min(samples, len(source.faces)), dtype=int
        )
        points = np.vstack((points, source.triangles_center[probes]))
        distances, closest = [], []
        for start in range(0, len(points), 256):
            nearest, distance, _ = target.nearest.on_surface(
                points[start : start + 256]
            )
            distances.extend(distance)
            closest.extend(nearest)
        distances = np.asarray(distances)
        if not np.isfinite(distances).all():
            raise ValueError("nonfinite surface distance")
        worst = int(np.argmax(distances))
        area_distances = distances[:samples]
        return {
            "area_samples": samples,
            "face_probes": len(probes),
            "p50_mm": float(np.percentile(area_distances, 50)),
            "p95_mm": float(np.percentile(area_distances, 95)),
            "p99_mm": float(np.percentile(area_distances, 99)),
            "rms_mm": float(np.sqrt(np.mean(area_distances**2))),
            "max_sampled_mm": float(distances[worst]),
            "area_fraction_outside_tolerance": float(
                np.mean(area_distances > tolerance)
            ),
            "worst_source_mm": points[worst].tolist(),
            "worst_target_mm": closest[worst].tolist(),
            "passed": bool(distances.max() <= tolerance),
        }

    actual, reference = load(actual_path), load(reference_path)
    forward, reverse = directed(actual, reference), directed(reference, actual)
    return {
        "method": "bidirectional_area_samples_and_face_centroids_to_triangles",
        "seed": 0,
        "tolerance_mm": tolerance,
        "actual_to_reference": forward,
        "reference_to_actual": reverse,
        "passed": forward["passed"] and reverse["passed"],
    }


def compare_meshes(
    actual_path: os.PathLike[str] | str,
    reference_path: os.PathLike[str] | str,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.01,
    *,
    surface_samples: int = 0,
    surface_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Compare bounds/area and optionally local shape; volume is diagnostic."""
    if any(
        not math.isfinite(t) or t < 0
        for t in (relative_tolerance, absolute_tolerance, surface_tolerance)
    ):
        raise ValueError("tolerances must be finite and non-negative")
    if type(surface_samples) is not int or surface_samples < 0:
        raise ValueError("surface_samples must be a non-negative integer")
    actual = mesh_metrics(actual_path)
    reference = mesh_metrics(reference_path)
    valid = actual["comparable"] and reference["comparable"]

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

    distances = (
        surface_distances(
            actual_path, reference_path, surface_samples, surface_tolerance
        )
        if valid and surface_samples
        else None
    )
    return {
        "passed": bool(
            valid
            and bounds_passed
            and area_passed
            and (distances is None or distances["passed"])
        ),
        "surface_distance": distances,
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
            "Distances sample surfaces, not a certified Hausdorff bound; unsampled "
            "defects can be missed. No topology, manifoldness, winding or shading "
            "guarantee. World-coordinate STL uses f32. OCCT is a reference, not "
            "ground truth. Volume is diagnostic."
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
    if (
        transferred == 0
        or transferred != reader.NbRootsForTransfer()
        or reader.NbShapes() == 0
    ):
        raise RuntimeError(f"OCCT did not transfer every STEP root: {input_path}")
    shape = reader.OneShape()
    # Reference chords are finer than the default 0.1 mm comparison tolerance.
    BRepMesh_IncrementalMesh(shape, 0.01, False, 0.1, True)
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
    parser.add_argument("--compare", help="Foxtrot STL to compare with the OCCT mesh")
    parser.add_argument("--report", help="JSON comparison output")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--surface-tolerance", type=float, default=0.1)
    parser.add_argument("--surface-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    if bool(args.compare) != bool(args.report):
        parser.error("--compare and --report must be supplied together")
    if args.surface_samples < 1:
        parser.error("--surface-samples must be positive")
    try:
        step_to_stl(args.input, args.output)
        metrics = mesh_metrics(args.output)
        if not metrics["comparable"]:
            raise RuntimeError(
                f"OCCT produced an invalid mesh: {json.dumps(metrics['validation'])}"
            )
        if args.compare:
            report = compare_meshes(
                args.compare,
                args.output,
                args.relative_tolerance,
                args.absolute_tolerance,
                surface_samples=args.surface_samples,
                surface_tolerance=args.surface_tolerance,
            )
            with open(args.report, "w") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
        # A completed comparison is not a process failure, even on mismatch.
        return 0
    except Exception as error:
        print(f"corpus_geometry: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
