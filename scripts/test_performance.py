import json
import sys
import tempfile
import unittest
from pathlib import Path

import performance
from corpus import digest, write_json

WORKER = '''#!/usr/bin/env python3
import json, pathlib, sys, time
mode = pathlib.Path(sys.argv[1]).read_text()
if mode == "slow": time.sleep(.3)
m={"schema":1,"stages":[{"name":"all","wall_ms":1,"cpu_ms":1,"peak_rss_bytes":1024}],"phases":[{"name":"nested","seconds":.001,"calls":1}],"mesh":{"vertices":3,"triangles":1,"faces":1,"shells":1,"completion":"complete","failures":[]},"storage":{"input_bytes":1,"flattened_bytes":2,"entities":1,"mesh_bytes":3,"mesh_capacity_bytes":4,"browser_bytes":5},"total_ms":1}
if mode == "nan": m["total_ms"]=float("nan")
if mode == "partial": m["mesh"].update(completion="partial",failures=["face"])
pathlib.Path(sys.argv[2]).write_text(json.dumps(m))
'''

class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.root = self.base / "root"; self.root.mkdir()
        self.worker = self.base / "worker"; self.worker.write_text(WORKER); self.worker.chmod(0o755)

    def capture(self, files, *extra):
        entries=[]
        for name, content in files:
            p=self.root/name; p.write_text(content); entries.append({"path":name,"sha256":digest(p),"bytes":p.stat().st_size})
        manifest=self.base/("manifest"+str(len(list(self.base.glob('out*'))))+".json"); write_json(manifest,{"schema":1,"files":entries})
        out=self.base/("out"+str(len(list(self.base.glob('out*')))))
        code=performance.main([str(self.root),"--manifest",str(manifest),"--worker",str(self.worker),"--output",str(out),*extra])
        return code, json.loads((out/"results.json").read_text()), out

    def test_deadline_has_explicit_skips(self):
        code, report, _ = self.capture([("a.step","slow"),("b.step","ok")],"--budget","0.1","--timeout","1")
        self.assertEqual(code,1); self.assertEqual(len(report["results"]),2)
        self.assertEqual(report["results"][1]["samples"][0]["status"],"skipped")

    def test_malformed_nonfinite_rejected(self):
        code, report, _ = self.capture([("a.step","nan")])
        self.assertEqual(code,1); self.assertEqual(report["results"][0]["status"],"crash")

    def test_partial_is_not_green(self):
        code, report, _ = self.capture([("a.step","partial")])
        self.assertEqual(code,1); self.assertEqual(report["results"][0]["status"],"partial")
        self.assertEqual(report["coverage"]["completed_samples"],0)

    def test_input_mismatch_comparison_rejected(self):
        _, prior, _ = self.capture([("a.step","ok")])
        _, current, _ = self.capture([("a.step","ok")])
        current["results"][0]["sha256"]="different"
        with self.assertRaisesRegex(ValueError,"input hash differs"):
            performance.compare_reports(current,prior)

    def test_compares_optimized_binary_time_and_memory(self):
        _, prior, _ = self.capture([("a.step","ok")])
        _, current, _ = self.capture([("a.step","ok")])
        current["worker"]["sha256"] = "new-build"
        for sample in current["results"][0]["samples"]:
            sample["metrics"]["total_ms"] = 0.5
            sample["metrics"]["stages"][0]["peak_rss_bytes"] = 512
        comparison = performance.compare_reports(current, prior)[0]
        self.assertEqual(comparison["ratio"], 0.5)
        self.assertEqual(comparison["rss_ratio"], 0.5)

if __name__ == "__main__": unittest.main()
