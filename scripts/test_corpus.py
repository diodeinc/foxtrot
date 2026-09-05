import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import corpus

# Exercise the subprocess protocol without compiling Rust or requiring CAD data.
WORKER = """#!/usr/bin/env python3
import json, pathlib, struct, sys, time
source, metrics, mesh = map(pathlib.Path, sys.argv[1:])
mode = source.read_text()
if mode == 'timeout': time.sleep(30)
if mode == 'crash': sys.exit(7)
if mode == 'protocol': sys.exit(0)
data = dict(read_ms=1, parse_ms=2, triangulate_ms=3, export_ms=4,
            triangles=1, vertices=3, faces=1, shells=1, errors=0,
            panics=0, log_warn=0, log_error=0)
if mode == 'partial': data['errors'] = 1
metrics.write_text(json.dumps(data))
mesh.write_bytes(bytes(80) + struct.pack('<I', 1) +
                 struct.pack('<12fH', 0,0,0, 0,0,0, 1,0,0, 0,1,0, 0))
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
