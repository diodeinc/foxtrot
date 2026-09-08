import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import board_benchmark as bench


def metrics():
    return dict(
        schema=1,
        status="ok",
        total_ms=12,
        stages=[],
        serialized_bytes=100,
        timings=dict(component_tessellation_secs=0.005, board_geometry_secs=0.002),
        counts=dict(
            unique_models=2,
            placeholder_instances=0,
            resolved_instances=3,
            total_instances=3,
            triangles=7,
        ),
        foxtrot=dict(
            models=2,
            faces=5,
            errors=0,
            panics=0,
            failed_instances=0,
            skipped_instances=0,
        ),
        validation=dict(
            nonfinite_vertices=0,
            nonfinite_instances=0,
            invalid_indices=0,
            empty_batches=0,
        ),
    )


class BoardBenchmarkTests(unittest.TestCase):
    def test_fetch_keeps_auth_failure_and_inventories_later_repository(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "board-corpus.json").write_text(
                json.dumps([["demo/b/blocked", "Blocked"], ["demo/b/open", "Open"]])
            )
            corpus = root / "corpus"
            (corpus / "Open").mkdir(parents=True)
            board = corpus / "Open" / "main.kicad_pcb"
            board.write_text("(kicad_pcb)")

            def git(command, **unused):
                return {
                    "origin": "https://code.diode.computer/demo/b/open.git",
                    "HEAD": "revision",
                    "--porcelain": "",
                }[command[-1]]

            with (
                patch.object(bench, "HERE", root),
                patch.object(bench, "execute", return_value=dict(status="crash")),
                patch.object(bench.subprocess, "check_output", side_effect=git),
            ):
                self.assertEqual(bench.fetch(corpus), 1)
            repos = json.loads((corpus / "inventory.json").read_text())["repositories"]
            self.assertEqual([r["acquisition_status"] for r in repos], ["error", "ok"])
            self.assertEqual(repos[1]["boards"][0]["sha256"], bench.digest(board))

    def test_returned_scene_is_not_proof_of_complete_tessellation(self):
        m = metrics()
        self.assertEqual(bench.classify(m), "ok")
        m["foxtrot"]["errors"] = 1
        self.assertEqual(bench.classify(m), "partial")
        m["foxtrot"]["errors"] = 0
        m["foxtrot"]["failed_instances"] = 1
        self.assertEqual(bench.classify(m), "partial")
        m["foxtrot"]["failed_instances"] = 0
        m["foxtrot"]["models"] = 1
        self.assertEqual(bench.classify(m), "missing_diagnostics")

    def test_placeholders_and_invalid_instance_transforms_are_visible(self):
        m = metrics()
        m["counts"]["placeholder_instances"] = 2
        self.assertEqual(bench.classify(m), "missing_models")
        m["validation"]["nonfinite_instances"] = 1
        self.assertEqual(bench.classify(m), "invalid_mesh")
        m["total_ms"] = float("nan")
        with self.assertRaises(ValueError):
            bench.classify(m)

    def test_comparison_rejects_changed_inputs_and_instrumentation(self):
        report = dict(
            config=dict(repeat=2),
            build=dict(
                diode="revision",
                source_sha256="worker",
                instrumentation_patch_sha256="patch",
            ),
            results=[dict(path="a", sha256="content")],
        )
        bench.compatible(report, copy.deepcopy(report))
        changed = copy.deepcopy(report)
        changed["results"][0]["sha256"] = "other content"
        with self.assertRaises(ValueError):
            bench.compatible(report, changed)
        changed = copy.deepcopy(report)
        changed["build"]["source_sha256"] = "other worker"
        with self.assertRaises(ValueError):
            bench.compatible(report, changed)

    def run_fixture(self, root, output):
        board = root / "test.kicad_pcb"
        board.write_text("(kicad_pcb)")
        worker = root / "worker"
        worker.write_text("placeholder")
        (root / "build.json").write_text(
            json.dumps(dict(worker=str(worker), worker_sha256=bench.digest(worker)))
        )
        inventory = dict(
            repositories=[
                dict(
                    alias="test",
                    acquisition_status="ok",
                    boards=[dict(path=board.name, sha256=bench.digest(board))],
                )
            ]
        )
        (root / "inventory.json").write_text(json.dumps(inventory))
        return SimpleNamespace(
            root=root,
            output=output,
            build=root / "build.json",
            compare=None,
            repeat=2,
            timeout=10,
            budget=1,
        )

    def test_deadline_accounts_for_every_unfinished_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.run_fixture(root, root / "out")
            with (
                patch.object(bench.time, "monotonic", side_effect=[0, 2, 2, 2, 2, 2]),
                patch.object(bench, "execute") as run,
            ):
                self.assertEqual(bench.bench(args), 1)
            run.assert_not_called()
            report = json.loads((args.output / "results.json").read_text())
            self.assertEqual(
                [s["status"] for s in report["results"][0]["samples"]],
                ["budget_exhausted"] * 2,
            )

    def test_bad_worker_metrics_do_not_destroy_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = self.run_fixture(root, root / "out")

            def run(command, *unused):
                Path(command[-1]).write_text('{"schema":1,"status":"ok"}')
                return dict(status="ok", returncode=0, wall_ms=1)

            with patch.object(bench, "execute", side_effect=run):
                self.assertEqual(bench.bench(args), 1)
            report = json.loads((args.output / "results.json").read_text())
            self.assertTrue(
                all(
                    s["status"] == "invalid_metrics"
                    for s in report["results"][0]["samples"]
                )
            )
            self.assertTrue((args.output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
