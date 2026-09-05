import builtins
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


if __name__ == "__main__":
    unittest.main()
