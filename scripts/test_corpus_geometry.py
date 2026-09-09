import builtins
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import corpus_geometry
except ModuleNotFoundError:  # Supports ``python -m unittest scripts/test_...``.
    from scripts import corpus_geometry


def write_stl(path, triangles, normal=(0.0, 0.0, 0.0)):
    with open(path, "wb") as output:
        output.write(b"test fixture".ljust(80, b"\0"))
        output.write(struct.pack("<I", len(triangles)))
        for vertices in triangles:
            output.write(struct.pack("<12fH", *(normal + sum(vertices, ())), 0))


TETRAHEDRON = [
    ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
]


class MeshMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "mesh.stl"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_tetrahedron_metrics_are_json_safe(self):
        write_stl(self.path, TETRAHEDRON)
        result = corpus_geometry.mesh_metrics(self.path)
        self.assertEqual(
            result["bounds"], {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        )
        self.assertAlmostEqual(result["surface_area"], 1.5 + math.sqrt(3) / 2)
        self.assertAlmostEqual(result["signed_volume"], 1 / 6)
        self.assertTrue(result["validation"]["valid"])
        json.dumps(result, allow_nan=False)

    def test_degenerate_and_nonfinite_facets_are_invalid(self):
        write_stl(self.path, [((0, 0, 0), (0, 0, 0), (1, 0, 0))])
        result = corpus_geometry.mesh_metrics(self.path)
        self.assertFalse(result["validation"]["valid"])
        self.assertEqual(result["validation"]["degenerate_triangle_count"], 1)

        write_stl(self.path, [((0, 0, 0), (1, 0, 0), (0, 1, float("nan")))])
        result = corpus_geometry.mesh_metrics(self.path)
        self.assertFalse(result["validation"]["finite"])
        self.assertIsNone(result["surface_area"])

    def test_rejects_truncated_or_trailing_binary_stl(self):
        self.path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "truncated"):
            corpus_geometry.mesh_metrics(self.path)
        write_stl(self.path, TETRAHEDRON)
        with self.path.open("ab") as output:
            output.write(b"x")
        with self.assertRaisesRegex(ValueError, "size"):
            corpus_geometry.mesh_metrics(self.path)

    def test_compare_uses_bounds_and_area_not_volume(self):
        reference = Path(self.tempdir.name) / "reference.stl"
        write_stl(reference, TETRAHEDRON)
        # Reversing every triangle changes volume sign, but not bounds or area.
        write_stl(self.path, [(a, c, b) for a, b, c in TETRAHEDRON])
        result = corpus_geometry.compare_meshes(self.path, reference, 0.0, 0.0)
        self.assertTrue(result["passed"])
        self.assertGreater(result["differences"]["signed_volume"]["absolute"], 0)

    def test_compare_detects_scale_change(self):
        reference = Path(self.tempdir.name) / "reference.stl"
        write_stl(reference, TETRAHEDRON)
        write_stl(
            self.path,
            [
                tuple(tuple(2 * x for x in vertex) for vertex in tri)
                for tri in TETRAHEDRON
            ],
        )
        self.assertFalse(corpus_geometry.compare_meshes(self.path, reference)["passed"])

    def test_optional_ocp_dependency_has_actionable_error(self):
        real_import = builtins.__import__

        def missing_ocp(name, *args, **kwargs):
            if name == "OCP" or name.startswith("OCP."):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing_ocp):
            with self.assertRaisesRegex(RuntimeError, "cadquery-ocp"):
                corpus_geometry.step_to_stl("input.step", self.path)


@unittest.skipUnless(importlib.util.find_spec("OCP"), "optional OCCT dependency not installed")
class ReferenceCompletenessTests(unittest.TestCase):
    def test_invalid_transferred_shape_is_rejected_before_meshing(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

        wire = BRepBuilderAPI_MakePolygon()
        for x, y in [(0, 0), (2, 2), (2, 0), (0, 2)]:
            wire.Add(gp_Pnt(x, y, 0))
        wire.Close()
        # A genuine self-intersecting face; exercise OCCT's validity checker.
        face = BRepBuilderAPI_MakeFace(
            gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), wire.Wire(), True
        ).Face()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("OCP.STEPControl.STEPControl_Reader") as factory, \
                mock.patch("OCP.BRepMesh.BRepMesh_IncrementalMesh") as mesher:
            reader = factory.return_value
            reader.ReadFile.return_value = IFSelect_RetDone
            reader.TransferRoots.return_value = reader.NbRootsForTransfer.return_value = 1
            reader.NbShapes.return_value = 1
            reader.OneShape.return_value = face
            output = Path(directory) / "invalid.stl"
            with self.assertRaisesRegex(RuntimeError, "transferred shape as invalid"):
                corpus_geometry.step_to_stl("invalid.step", output)
            mesher.assert_not_called()
            self.assertFalse(output.exists())

    def test_reference_requires_every_source_face_to_be_meshed(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "box.step"
            output = Path(directory) / "box.stl"
            writer = STEPControl_Writer()
            writer.Transfer(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(), STEPControl_AsIs)
            self.assertEqual(writer.Write(str(source)), IFSelect_RetDone)
            corpus_geometry.step_to_stl(source, output)
            self.assertAlmostEqual(corpus_geometry.mesh_metrics(output)["surface_area"], 22.0)
            # Simulate meshing failure on genuine transferred topology, not a
            # fabricated mesh-metric result. A writable STL is not sufficient.
            with mock.patch("OCP.BRepMesh.BRepMesh_IncrementalMesh"):
                with self.assertRaisesRegex(RuntimeError, "faces unmeshed"):
                    corpus_geometry.step_to_stl(source, output)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("numpy", "igl")),
    "optional surface oracle dependencies not installed",
)
class SurfaceDistanceTests(unittest.TestCase):
    def compare(self, actual, reference, tolerance=1e-6):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "actual.stl", Path(directory) / "reference.stl"
            write_stl(a, actual)
            write_stl(b, reference)
            return corpus_geometry.compare_meshes(
                a, b, surface_samples=500, surface_tolerance=tolerance
            )

    def test_different_triangulations_agree_without_vertex_correspondence(self):
        a, b, c, d = (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)
        result = self.compare([(a, b, c), (a, c, d)], [(a, b, d), (b, c, d)])
        self.assertTrue(result["passed"])
        self.assertLess(
            result["surface_distance"]["actual_to_reference"]["max_sampled_mm"], 1e-6
        )

    def test_nearest_surface_is_preserved_across_single_triangle_batches(self):
        with mock.patch.object(corpus_geometry, "_MESH_BLOCK_TRIANGLES", 1):
            result = self.compare(TETRAHEDRON, TETRAHEDRON[::-1])
        self.assertTrue(result["passed"])
        for direction in ("actual_to_reference", "reference_to_actual"):
            self.assertLess(result["surface_distance"][direction]["max_sampled_mm"], 1e-12)

    def test_displaced_interior_surface_evades_aggregate_checks(self):
        def panel(z):
            return ((0.2, 0.2, z), (0.3, 0.2, z), (0.2, 0.3, z))

        result = self.compare(TETRAHEDRON + [panel(0.2)], TETRAHEDRON + [panel(0.4)])
        self.assertTrue(result["differences"]["bounds"]["passed"])
        self.assertTrue(result["differences"]["surface_area"]["passed"])
        self.assertFalse(result["passed"])
        self.assertGreater(
            result["surface_distance"]["actual_to_reference"]["max_sampled_mm"], 0.1
        )

    def test_reverse_direction_detects_missing_face(self):
        result = self.compare(TETRAHEDRON[:3], TETRAHEDRON)
        distance = result["surface_distance"]
        self.assertTrue(distance["actual_to_reference"]["passed"])
        self.assertFalse(distance["reference_to_actual"]["passed"])
        self.assertGreater(
            distance["reference_to_actual"]["area_fraction_outside_tolerance"], 0.1
        )

    def test_zero_area_facets_are_reported_without_blocking_surface_comparison(self):
        result = self.compare(TETRAHEDRON + [((0, 0, 0),) * 3], TETRAHEDRON)
        self.assertTrue(result["passed"])
        self.assertEqual(result["actual"]["validation"]["degenerate_triangle_count"], 1)


if __name__ == "__main__":
    unittest.main()
