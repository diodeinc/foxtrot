import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus

# Exercise the subprocess protocol without compiling Rust or requiring CAD data.
WORKER = """#!/usr/bin/env python3
import json, os, pathlib, struct, subprocess, sys, time
source, metrics, mesh = map(pathlib.Path, sys.argv[1:])
mode = source.read_text()
if mode == 'descendant':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    source.with_suffix('.pids').write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(30)
if mode == 'timeout': time.sleep(30)
if mode == 'crash': sys.exit(7)
if mode == 'protocol': sys.exit(0)
data = dict(read_ms=1, parse_ms=2, triangulate_ms=3, export_ms=4,
            triangles=1, vertices=3, faces=1, shells=1, errors=0,
            panics=0, log_warn=0, log_error=0)
if mode == 'partial': data['errors'] = 1
if mode == 'nonfinite': data['parse_ms'] = float('nan')
if mode == 'counts': data['triangles'] = 2
if mode == 'f64_invalid': data['degenerate_f64'] = 1
if mode == 'f64_nonfinite': data['degenerate_f64'] = float('nan')
if mode.startswith('browser_'):
    data.update(triangles=2, browser_nonfinite=0, browser_triangles=2, browser_degenerate=0,
                browser_zero_normals=0)
if mode == 'browser_some_collapsed': data.update(browser_degenerate=1, browser_zero_normals=3)
if mode == 'browser_all_collapsed': data['browser_degenerate'] = 2
if mode == 'browser_nonfinite_components': data['browser_nonfinite'] = 1
if mode == 'browser_missing': del data['browser_zero_normals']
if mode == 'browser_f64_invalid': data['degenerate_f64'] = 1
if mode == 'browser_bad_number': data['browser_degenerate'] = float('nan')
marker = metrics.with_suffix('.count')
count = int(marker.read_text()) if marker.exists() else 0
marker.write_text(str(count + 1))
if mode == 'vary': data['faces'] += count
if mode == 'browser_vary': data['browser_zero_normals'] += count
metrics.write_text(json.dumps(data))
if mode == 'stale' and count: sys.exit(0)
if mode == 'invalid':
    mesh.write_bytes(bytes(80) + struct.pack('<I', 0))
    data['triangles'] = 0
    metrics.write_text(json.dumps(data))
    sys.exit(0)
facets = 2 if mode.startswith('browser_') else 1
mesh.write_bytes(bytes(80) + struct.pack('<I', facets) +
                 struct.pack('<12fH', 0,0,0, 0,0,0, 1,0,0, 0,1,0, 0) * facets)
if mode == 'browser_no_stl' or str(mesh) == '-': mesh.unlink()
print('worker diagnostic')
"""


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "inputs"
        self.root.mkdir()
        self.worker = self.base / "worker"
        self.worker.write_text(WORKER)
        self.worker.chmod(0o755)

    def model(self, name, content="ok"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def run_harness(self, name, *args):
        output = self.base / name
        code = corpus.main(
            [
                str(self.root),
                "--worker",
                str(self.worker),
                "--output",
                str(output),
                *args,
            ]
        )
        report = (
            json.loads((output / "results.json").read_text())
            if (output / "results.json").exists()
            else None
        )
        return code, report, output

    def test_discovery_sampling_hashes_and_duplicate_basenames(self):
        for name in [
            "a/same.step",
            "b/same.StP",
            'quo"te.step',
            "模型.STEP",
            "ignore.txt",
        ]:
            self.model(name)
        entries = corpus.discover(self.root, ["*"], [], "fixed", None)
        self.assertEqual(len(entries), 4)
        self.assertEqual(
            corpus.discover(self.root, ["*"], [], "fixed", 2),
            corpus.discover(self.root, ["*"], [], "fixed", 2),
        )
        self.assertEqual(len(corpus.discover(self.root, ["a/*"], [], "0", None)), 1)
        self.assertEqual(len(corpus.discover(self.root, ["*"], ["a/*"], "0", None)), 3)
        manifest = self.base / "manifest.json"
        corpus.write_json(manifest, {"schema": 1, "files": entries})
        self.assertEqual(corpus.load_manifest(manifest, self.root), entries)
        self.model("a/same.step", "changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            corpus.load_manifest(manifest, self.root)

    def test_manifest_rejects_escape_and_duplicate(self):
        p = self.model("a.step")
        manifest = self.base / "manifest.json"
        entry = {"path": "../worker", "sha256": corpus.digest(self.worker)}
        corpus.write_json(manifest, {"schema": 1, "files": [entry]})
        with self.assertRaisesRegex(ValueError, "unsafe"):
            corpus.load_manifest(manifest, self.root)
        entry = {"path": "a.step", "sha256": corpus.digest(p)}
        corpus.write_json(manifest, {"schema": 1, "files": [entry, entry]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            corpus.load_manifest(manifest, self.root)

    def test_run_benchmark_replay_and_compare(self):
        self.model("nested/模型.step")
        code, report, output = self.run_harness(
            "first", "--repeat", "3", "--warmup", "1", "--meshes", "all"
        )
        self.assertEqual(code, 0)
        result = report["results"][0]
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual(result["timing"]["process_ms"]["median"], 5)
        self.assertTrue((output / result["artifacts"] / "mesh.stl").exists())
        code, second, _ = self.run_harness(
            "second",
            "--manifest",
            str(output / "manifest.json"),
            "--compare",
            str(output / "results.json"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(second["changes"], [])
        self.assertEqual(self.run_harness("first")[0], 2)  # Never clobber artifacts.

    def test_failures_are_isolated_reported_and_rerunnable(self):
        for mode in ("ok", "crash", "timeout", "partial", "protocol"):
            self.model(mode + ".step", mode)
        code, report, output = self.run_harness(
            "failures", "--timeout", "0.3", "--jobs", "2"
        )
        self.assertEqual(code, 1)
        states = {r["path"]: r["status"] for r in report["results"]}
        self.assertEqual(
            states,
            {
                "ok.step": "ok",
                "crash.step": "crash",
                "timeout.step": "timeout",
                "partial.step": "tessellation_error",
                "protocol.step": "harness_error",
            },
        )
        code, rerun, _ = self.run_harness(
            "rerun", "--rerun", str(output / "results.json"), "--timeout", "0.1"
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(rerun["results"]), 4)
        self.assertTrue(all(r["path"] != "ok.step" for r in rerun["results"]))
        self.assertEqual(len((output / "progress.jsonl").read_text().splitlines()), 5)

    def test_comparison_requires_same_corpus_content_and_settings(self):
        self.model("one.step")
        _, report, _ = self.run_harness("baseline")
        changed = copy.deepcopy(report["results"])
        changed[0]["sha256"] = "changed"
        self.assertIn(
            "input content changed",
            corpus.compare(changed, report, report["config"], None)[0],
        )
        self.assertIn("missing", corpus.compare([], report, report["config"], None)[0])
        changed = copy.deepcopy(report["results"])
        changed[0]["timing"]["process_ms"]["median"] = 10
        self.assertTrue(corpus.compare(changed, report, report["config"], 0.1))
        config = dict(report["config"], threads=8)
        self.assertTrue(corpus.compare(report["results"], report, config, 0.1))

    def test_empty_selection_fails(self):
        self.assertEqual(self.run_harness("empty")[0], 2)

    def test_bad_reports_fail_before_launching_workers(self):
        self.model("one.step")
        _, report, _ = self.run_harness("report-source")
        duplicate = copy.deepcopy(report)
        duplicate["results"] *= 2
        bad_timing = copy.deepcopy(report)
        bad_timing["results"][0]["timing"] = []
        bad_metrics = copy.deepcopy(report)
        bad_metrics["results"][0]["metrics"]["errors"] = "oops"
        for i, value in enumerate(
            [[], None, {"schema": 1}, duplicate, bad_timing, bad_metrics]
        ):
            path = self.base / f"bad-{i}.json"
            corpus.write_json(path, value)
            code, _, output = self.run_harness(f"bad-run-{i}", "--compare", str(path))
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
        path = self.base / "bad-manifest.json"
        path.write_text("[]")
        self.assertEqual(
            self.run_harness("bad-manifest", "--manifest", str(path))[0], 2
        )

    def test_worker_output_validation_and_nondeterminism(self):
        expected = {
            "nonfinite": "harness_error",
            "counts": "harness_error",
            "invalid": "invalid_mesh",
            "f64_invalid": "invalid_mesh",
            "f64_nonfinite": "harness_error",
            "vary": "nondeterministic",
            "stale": "harness_error",
        }
        for mode, status in expected.items():
            self.model(mode + ".step", mode)
            code, report, _ = self.run_harness(
                mode, "--include", mode + ".step", "--repeat", "2"
            )
            self.assertEqual(code, int(status != "ok"))
            self.assertEqual(report["results"][0]["status"], status)

    def test_browser_acceptance_and_quality_diagnostics(self):
        expected = {
            "browser_some_collapsed": "ok",
            "browser_all_collapsed": "invalid_mesh",
            "browser_nonfinite_components": "invalid_mesh",
            "browser_missing": "harness_error",
            "browser_bad_number": "harness_error",
            "browser_f64_invalid": "ok",
            "browser_no_stl": "ok",
            "browser_vary": "nondeterministic",
        }
        for mode, status in expected.items():
            with self.subTest(mode=mode):
                self.model(mode + ".step", mode)
                code, report, _ = self.run_harness(
                    "case-" + mode,
                    "--include",
                    mode + ".step",
                    "--repeat",
                    "2" if mode == "browser_vary" else "1",
                )
                result = report["results"][0]
                self.assertEqual(code, int(status != "ok"))
                self.assertEqual(result["status"], status)
                if status != "harness_error":
                    self.assertEqual(
                        result["classification"]["basis"],
                        "browser_f32_triangle_buffer",
                    )
        some = json.loads(
            (self.base / "case-browser_some_collapsed" / "results.json").read_text()
        )["results"][0]
        self.assertEqual(some["quality_diagnostics"]["browser_degenerate_triangles"], 1)
        self.assertEqual(some["quality_diagnostics"]["browser_zero_normal_vertices"], 3)
        f64 = json.loads(
            (self.base / "case-browser_f64_invalid" / "results.json").read_text()
        )["results"][0]
        self.assertEqual(f64["quality_diagnostics"]["degenerate_f64_review"], 1)

    def test_no_stl_scan_keeps_browser_diagnostics(self):
        self.model("one.step", "browser_some_collapsed")
        code, report, output = self.run_harness("fast", "--meshes", "none")
        self.assertEqual(code, 0)
        result = report["results"][0]
        self.assertNotIn("geometry", result)
        self.assertEqual(result["quality_diagnostics"]["browser_degenerate_triangles"], 1)
        self.assertFalse(list(output.rglob("*.stl")))
        self.assertIn("browser_degenerate_triangles: 1", (output / "report.md").read_text())
        review = json.loads((output / "review-manifest.json").read_text())
        self.assertEqual([e["path"] for e in review["files"]], ["one.step"])
        self.assertEqual(review["files"][0]["sha256"], result["sha256"])

    def test_timing_gate_and_retained_comparison_meshes(self):
        self.model("one.step")
        _, report, output = self.run_harness("timing-before", "--repeat", "2")
        report["results"][0]["timing"]["process_ms"]["median"] = 1
        corpus.write_json(output / "results.json", report)
        code, after, output = self.run_harness(
            "timing-after",
            "--repeat",
            "2",
            "--compare",
            str(output / "results.json"),
            "--timing-threshold",
            "0.1",
        )
        self.assertEqual(code, 1)
        self.assertTrue(any("processing median" in c for c in after["changes"]))
        self.assertTrue(
            (output / after["results"][0]["artifacts"] / "mesh.stl").exists()
        )

    @unittest.skipUnless(sys.platform == "linux", "Linux process inspection")
    def test_real_sigint_terminates_worker_and_descendant(self):
        source = self.model("one.step", "descendant")
        for i in range(10):
            self.model(f"queued-{i}.step", "timeout")
        output = self.base / "sigint"
        # Noninteractive runners can inherit SIG_IGN. Simulate an interactive
        # Python launch without changing the harness's inherited signal policy.
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import runpy, signal, sys; "
                "signal.signal(signal.SIGINT, signal.default_int_handler); "
                "sys.argv = sys.argv[1:]; "
                "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent)); "
                "runpy.run_path(sys.argv[0], run_name='__main__')",
                str(corpus.REPO / "scripts/corpus.py"),
                str(self.root),
                "--worker",
                str(self.worker),
                "--output",
                str(output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pids = []
        try:
            deadline = time.monotonic() + 5
            while (
                not source.with_suffix(".pids").exists() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            pids = json.loads(source.with_suffix(".pids").read_text())
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertFalse((output / "results.json").exists())
            self.assertTrue((output / "manifest.json").exists())
            for pid in pids:
                stat = Path(f"/proc/{pid}/stat")
                self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            if pids:
                try:
                    os.killpg(pids[0], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_interruption_keeps_manifest_but_not_complete_baseline(self):
        self.model("one.step", "timeout")
        with patch("corpus.as_completed", side_effect=KeyboardInterrupt):
            code, report, output = self.run_harness("interrupted")
        self.assertEqual(code, 130)
        self.assertIsNone(report)
        self.assertTrue((output / "manifest.json").exists())

    def test_cancellation_reaps_worker(self):
        event = corpus.Event()
        event.set()
        result = corpus.execute(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            self.base / "cancel.log",
            30,
            os.environ.copy(),
            event,
        )
        self.assertEqual(result["status"], "cancelled")
        self.assertLess(result["wall_ms"], 3000)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_timeout_reaps_worker(self):
        result = corpus.execute(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            self.base / "timeout.log",
            0.05,
            os.environ.copy(),
        )
        self.assertEqual(result["status"], "timeout")
        self.assertLess(result["returncode"], 0)
        self.assertLess(result["wall_ms"], 3000)


if __name__ == "__main__":
    unittest.main()
