#!/usr/bin/env python3
"""Capture a small, reproducible native Foxtrot performance profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from corpus import REPO, digest, execute, load_manifest, write_json

SCHEMA = 1
THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _number(value, name, integer=False, nullable=False):
    if nullable and value is None:
        return
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")


def validate_metrics(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported worker schema")
    _number(value.get("total_ms"), "total_ms")
    for collection in ("stages", "phases"):
        if not isinstance(value.get(collection), list):
            raise ValueError(f"invalid {collection}")
        for item in value[collection]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError(f"invalid {collection} entry")
            keys = ("wall_ms", "cpu_ms", "peak_rss_bytes") if collection == "stages" else ("seconds", "calls")
            for key in keys:
                _number(item.get(key), f"{collection}.{key}", integer=(key == "calls"), nullable=(key == "peak_rss_bytes"))
    mesh = value.get("mesh")
    storage = value.get("storage")
    if not isinstance(mesh, dict) or mesh.get("completion") not in ("complete", "partial") or not isinstance(mesh.get("failures"), list):
        raise ValueError("invalid mesh")
    for key in ("vertices", "triangles", "faces", "shells"):
        _number(mesh.get(key), f"mesh.{key}", integer=True)
    if (mesh["completion"] == "complete") != (len(mesh["failures"]) == 0):
        raise ValueError("mesh completion disagrees with failures")
    if not isinstance(storage, dict):
        raise ValueError("invalid storage")
    for key in ("input_bytes", "flattened_bytes", "entities", "mesh_bytes", "mesh_capacity_bytes", "browser_bytes"):
        _number(storage.get(key), f"storage.{key}", integer=True)
    return value


def compare_reports(current, prior):
    if prior.get("schema") != SCHEMA:
        raise ValueError("comparison rejected: incompatible schema")
    for key in ("repeat", "threads"):
        if prior.get("config", {}).get(key) != current["config"][key]:
            raise ValueError(f"comparison rejected: {key} differs")
    # Different worker hashes are expected when measuring an optimization.
    # Preserve both hashes as provenance; compatibility is the measurement
    # protocol, input bytes and execution configuration, not binary identity.
    old = {r["path"]: r for r in prior.get("results", [])}
    if set(old) != {r["path"] for r in current["results"]}:
        raise ValueError("comparison rejected: file set differs")
    comparisons = []
    for result in current["results"]:
        before = old[result["path"]]
        if before.get("sha256") != result["sha256"]:
            raise ValueError(f"comparison rejected: input hash differs for {result['path']}")
        if before.get("status") != "ok" or result["status"] != "ok":
            comparisons.append({"path": result["path"], "status": "not_comparable"})
            continue
        a = statistics.median(s["metrics"]["total_ms"] for s in before["samples"])
        b = statistics.median(s["metrics"]["total_ms"] for s in result["samples"])
        old_rss = [peak_rss(s["metrics"]) for s in before["samples"]]
        new_rss = [peak_rss(s["metrics"]) for s in result["samples"]]
        old_peak = max(old_rss) if all(x is not None for x in old_rss) else None
        new_peak = max(new_rss) if all(x is not None for x in new_rss) else None
        comparisons.append({"path": result["path"], "status": "ok", "prior_median_ms": a, "median_ms": b,
                            "ratio": b / a if a else None, "delta_ms": b - a,
                            "prior_peak_rss_bytes": old_peak, "peak_rss_bytes": new_peak,
                            "rss_ratio": new_peak / old_peak if old_peak and new_peak is not None else None})
    return comparisons


def peak_rss(metrics):
    return max((s["peak_rss_bytes"] for s in metrics["stages"] if s["peak_rss_bytes"] is not None), default=None)


def markdown(report):
    lines = ["# Native performance capture", "", f"Total capture: {report['capture_wall_seconds']:.3f} s; coverage: {report['coverage']['completed_samples']}/{report['coverage']['requested_samples']} samples.", "",
             "| Case | Status | total_ms median / min / max | peak RSS MiB |", "| --- | --- | ---: | ---: |"]
    for case in report["results"]:
        good = [s["metrics"] for s in case["samples"] if s["status"] == "ok"]
        totals = [x["total_ms"] for x in good]
        rss = [st["peak_rss_bytes"] for x in good for st in x["stages"] if st["peak_rss_bytes"] is not None]
        timing = f"{statistics.median(totals):.3f} / {min(totals):.3f} / {max(totals):.3f}" if totals else "—"
        peak = f"{max(rss) / 1048576:.2f}" if rss else "—"
        lines.append(f"| {case['path'].replace('|', chr(92)+'|')} | {case['status']} | {timing} | {peak} |")
    for case in report["results"]:
        lines += ["", f"## {case['path']}", ""]
        for index, sample in enumerate(case["samples"], 1):
            lines += [f"### Sample {index}: {sample['status']}", ""]
            if "metrics" not in sample:
                continue
            m = sample["metrics"]
            lines += ["| Stage | wall ms | CPU ms | peak RSS bytes |", "| --- | ---: | ---: | ---: |"]
            lines += [f"| {x['name']} | {x['wall_ms']} | {x['cpu_ms']} | {x['peak_rss_bytes'] if x['peak_rss_bytes'] is not None else 'unsupported'} |" for x in m["stages"]]
            lines += ["", "| Inclusive phase | seconds | calls |", "| --- | ---: | ---: |"]
            lines += [f"| {x['name']} | {x['seconds']} | {x['calls']} |" for x in m["phases"]]
            lines += ["", "Mesh: " + ", ".join(f"{k}={v}" for k, v in m["mesh"].items()), "", "Storage: " + ", ".join(f"{k}={v}" for k, v in m["storage"].items())]
    if report.get("comparisons"):
        lines += ["", "## Comparison", "", "Ratios are current / baseline; smaller is better.", "",
                  "| Case | time ratio | delta ms | peak RSS ratio |", "| --- | ---: | ---: | ---: |"]
        for c in report["comparisons"]:
            lines.append(f"| {c['path']} | {c.get('ratio', '—')} | {c.get('delta_ms', '—')} | {c.get('rss_ratio', '—')} |")
    lines += ["", "## Scope", "", "This capture excludes builds, validation, STL, and OCCT. It has no warmup and is not a cold-cache benchmark. Processes are serial and native math/Rayon threads are limited to one. Stage RSS is cumulative process high-water memory (or null when unsupported), not live memory and must not be summed. Phase timings are nested and inclusive and must never be summed. `total_ms` ends when the browser buffer has been generated.", ""]
    return "\n".join(lines)


def _git_metadata():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    revision = git("rev-parse", "HEAD")
    return {"revision": revision or None, "dirty": bool(git("status", "--porcelain"))}


def main(argv=None):
    capture_start = time.perf_counter()  # Deliberately includes argument/manifest validation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=REPO / "scripts/performance-sample.json")
    parser.add_argument("--worker", type=Path, default=REPO / "target/release/examples/profile")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", default=2, type=positive_int)
    parser.add_argument("--budget", default=240, type=positive)
    parser.add_argument("--timeout", default=60, type=positive)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    if args.budget > 270:
        parser.error("--budget must not exceed 270 seconds")
    args.root, args.worker, args.output = args.root.resolve(), args.worker.resolve(), args.output.resolve()
    try:
        if not args.root.is_dir() or not args.worker.is_file():
            raise ValueError("root directory and native profile worker must exist")
        entries = load_manifest(args.manifest.resolve(), args.root)
        args.output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, **THREAD_ENV)
        results = []
        for entry in entries:
            case = dict(entry, status="ok", samples=[])
            for index in range(args.repeat):
                remaining = args.budget - (time.perf_counter() - capture_start)
                if remaining <= 0:
                    case["samples"].append({"status": "skipped", "reason": "capture budget exhausted"})
                    case["status"] = "skipped" if index == 0 else "partial"
                    continue
                metrics_path = args.output / f"sample-{len(results)}-{index}.json"
                process = execute([str(args.worker), str(args.root / entry["path"]), str(metrics_path)],
                                  args.output / f"sample-{len(results)}-{index}.log", min(remaining, args.timeout), env)
                sample = dict(process)
                if process["status"] == "ok":
                    try:
                        sample["metrics"] = validate_metrics(json.loads(metrics_path.read_text()))
                        sample["status"] = "ok" if sample["metrics"]["mesh"]["completion"] == "complete" else "partial"
                    except Exception as error:
                        sample.update(status="crash", error=str(error))
                case["samples"].append(sample)
                if sample["status"] != "ok":
                    case["status"] = "partial" if any(s["status"] == "ok" for s in case["samples"]) else sample["status"]
            results.append(case)
        completed = sum(s["status"] == "ok" for r in results for s in r["samples"])
        report = {"schema": SCHEMA, "config": {"repeat": args.repeat, "threads": 1, "budget_seconds": args.budget, "timeout_seconds": args.timeout},
                  "worker": {"path": str(args.worker), "sha256": digest(args.worker)}, "git": _git_metadata(),
                  "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
                  "capture_wall_seconds": time.perf_counter() - capture_start,
                  "coverage": {"files": len(results), "requested_samples": len(entries) * args.repeat, "completed_samples": completed}, "results": results}
        if args.compare:
            prior = json.loads(args.compare.read_text())
            report["baseline_worker"] = prior["worker"]
            report["comparisons"] = compare_reports(report, prior)
        report["capture_wall_seconds"] = time.perf_counter() - capture_start
        write_json(args.output / "results.json", report)
        (args.output / "report.md").write_text(markdown(report))
        print(f"Report: {args.output / 'report.md'} ({report['capture_wall_seconds']:.1f}s; {completed}/{len(entries) * args.repeat} samples)")
        return 0 if completed == len(entries) * args.repeat else 1
    except Exception as error:
        print(f"performance capture error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
