#!/usr/bin/env python3
"""Reproducible, process-isolated STEP testing and benchmarking (Python 3.10+)."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event

from corpus_geometry import compare_meshes, mesh_metrics

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def discover(root, includes, excludes, seed, sample):
    paths = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".step", ".stp")
        and p.resolve().is_relative_to(root)
    )
    paths = [
        p
        for p in paths
        if any(fnmatch.fnmatchcase(p, x) for x in includes)
        and not any(fnmatch.fnmatchcase(p, x) for x in excludes)
    ]
    # Hash ranking is reproducible across Python versions and directory traversal order.
    if sample is not None:
        paths = sorted(
            paths, key=lambda p: hashlib.sha256(f"{seed}\0{p}".encode()).digest()
        )[:sample]
    return [
        {"path": p, "sha256": digest(root / p), "bytes": (root / p).stat().st_size}
        for p in sorted(paths)
    ]


def load_manifest(path, root):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported manifest schema")
    entries = manifest["files"]
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError("manifest files must be a list of objects")
    seen = set()
    for entry in entries:
        name = entry["path"]
        target = (root / name).resolve()
        if name in seen or Path(name).is_absolute() or not target.is_relative_to(root):
            raise ValueError(f"unsafe or duplicate manifest path: {name}")
        seen.add(name)
        if digest(target) != entry["sha256"]:
            raise ValueError(f"input changed since manifest was recorded: {name}")
    return entries


def load_report(path):
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("unsupported report schema; record a new baseline")
    if not isinstance(report.get("config"), dict) or not isinstance(
        report.get("results"), list
    ):
        raise ValueError("report must contain config and results")
    for key in (
        "occt",
        "relative_tolerance",
        "absolute_tolerance",
        "threads",
        "jobs",
        "repeat",
        "warmup",
        "rust_log",
    ):
        if key not in report["config"]:
            raise ValueError(f"report configuration missing {key}")
    seen = set()
    for result in report["results"]:
        if not isinstance(result, dict) or any(
            not isinstance(result.get(k), str) for k in ("path", "sha256", "status")
        ):
            raise ValueError("report results require path, sha256, and status strings")
        if result["path"] in seen:
            raise ValueError(f"duplicate report path: {result['path']}")
        seen.add(result["path"])
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in metrics.values()
        ):
            raise ValueError(f"invalid metrics for {result['path']}")
        if result["status"] == "ok":
            timing = result.get("timing")
            if not isinstance(timing, dict) or not isinstance(
                timing.get("process_ms"), dict
            ):
                raise ValueError(f"missing processing timing for {result['path']}")
            median = timing["process_ms"].get("median")
            if (
                not isinstance(median, (int, float))
                or not math.isfinite(median)
                or median < 0
            ):
                raise ValueError(f"invalid processing timing for {result['path']}")
    return report


def execute(command, log, timeout, env, cancel=None):
    start = time.perf_counter()
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        while process.poll() is None:
            if time.perf_counter() - start >= timeout or (cancel and cancel.is_set()):
                break
            try:
                process.wait(timeout=min(0.1, timeout))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            status = "cancelled" if cancel and cancel.is_set() else "timeout"
        else:
            status = "ok" if process.returncode == 0 else "crash"
        code = process.returncode
    return {
        "status": status,
        "returncode": code,
        "wall_ms": (time.perf_counter() - start) * 1000,
    }


def run_file(entry, args):
    if args.cancel.is_set():
        return dict(entry, status="cancelled")
    directory = (
        args.output / "cases" / hashlib.sha256(entry["path"].encode()).hexdigest()[:20]
    )
    directory.mkdir(parents=True)
    result = dict(
        entry,
        status="ok",
        samples=[],
        artifacts=directory.relative_to(args.output).as_posix(),
    )
    source = args.root / entry["path"]
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), RUST_BACKTRACE="1")
    command = [
        str(args.worker),
        str(source),
        str(directory / "metrics.json"),
        "-" if args.meshes == "none" else str(directory / "mesh.stl"),
    ]
    result["reproduce"] = shlex.join(
        ["env", f"RAYON_NUM_THREADS={args.threads}", "RUST_BACKTRACE=1"] + command
    )
    try:
        for index in range(args.warmup + args.repeat):
            metrics_path = directory / "metrics.json"
            metrics_path.unlink(missing_ok=True)
            (directory / "mesh.stl").unlink(missing_ok=True)
            sample = execute(
                command, directory / f"run-{index}.log", args.timeout, env, args.cancel
            )
            if sample["status"] != "ok":
                result["status"] = sample["status"]
                result["failure"] = sample
                break
            metrics = json.loads(metrics_path.read_text())
            required = (
                "read_ms",
                "parse_ms",
                "triangulate_ms",
                "export_ms",
                "triangles",
                "vertices",
                "faces",
                "shells",
                "errors",
                "panics",
                "log_warn",
                "log_error",
            )
            browser_required = (
                "browser_nonfinite",
                "browser_triangles",
                "browser_degenerate",
                "browser_zero_normals",
            )
            browser_supplied = [k for k in browser_required if k in metrics]
            if browser_supplied and len(browser_supplied) != len(
                browser_required
            ):
                raise ValueError("incomplete browser worker metrics")
            # Older workers do not report this diagnostic. It remains review
            # evidence rather than an acceptance gate when present.
            degenerate_f64 = metrics.get("degenerate_f64", 0)
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in [metrics.get(k) for k in required]
                + [metrics[k] for k in browser_supplied]
                + [degenerate_f64]
            ):
                raise ValueError("invalid worker metrics")
            browser_metrics = "browser_triangles" in metrics
            if browser_metrics and (
                metrics["browser_triangles"] != metrics["triangles"]
                or metrics["browser_degenerate"] > metrics["browser_triangles"]
                or metrics["browser_zero_normals"] > metrics["browser_triangles"] * 3
            ):
                raise ValueError("inconsistent browser worker counts")
            mesh_path = directory / "mesh.stl"
            geometry = mesh_metrics(mesh_path) if mesh_path.exists() else None
            if not browser_metrics and geometry is None:
                raise ValueError("legacy worker did not export a mesh")
            if geometry is not None and geometry["triangle_count"] != metrics["triangles"]:
                raise ValueError("worker triangle count does not match exported mesh")
            sample.update(metrics)
            sample["process_ms"] = metrics["parse_ms"] + metrics["triangulate_ms"]
            deterministic = [
                "triangles",
                "faces",
                "errors",
                "panics",
                "log_warn",
                "log_error",
                "degenerate_f64",
            ]
            if browser_metrics:
                deterministic += [
                    "browser_nonfinite",
                    "browser_triangles",
                    "browser_degenerate",
                    "browser_zero_normals",
                ]
            if "metrics" in result and any(
                result["metrics"].get(k) != metrics.get(k)
                for k in deterministic
            ):
                result["status"] = "nondeterministic"
            result["metrics"] = metrics
            if geometry is not None:
                result["geometry"] = geometry
            result["classification"] = {
                "basis": (
                    "browser_f32_triangle_buffer" if browser_metrics else "legacy_stl"
                ),
                "visual_verification": False,
            }
            result["quality_diagnostics"] = {
                "browser_degenerate_triangles": metrics.get("browser_degenerate", 0),
                "browser_zero_normal_vertices": metrics.get("browser_zero_normals", 0),
                "degenerate_f64_review": degenerate_f64,
            }
            if metrics["errors"] or metrics["panics"] or metrics["log_error"]:
                result["status"] = "tessellation_error"
            elif browser_metrics and (
                metrics["browser_nonfinite"]
                or not metrics["browser_triangles"]
                or metrics["browser_degenerate"] >= metrics["browser_triangles"]
            ):
                result["status"] = "invalid_mesh"
            elif not browser_metrics and (
                degenerate_f64 or not geometry["validation"]["valid"]
            ):
                result["status"] = "invalid_mesh"
            if index >= args.warmup:
                result["samples"].append(sample)
            if result["status"] != "ok":
                break
        if result["samples"]:
            result["timing"] = {}
            for key in (
                "read_ms",
                "parse_ms",
                "triangulate_ms",
                "export_ms",
                "process_ms",
                "wall_ms",
            ):
                values = [s[key] for s in result["samples"]]
                result["timing"][key] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "samples": values,
                }
        if args.occt and result["status"] == "ok":
            oracle = execute(
                [
                    sys.executable,
                    str(REPO / "scripts/corpus_geometry.py"),
                    str(source),
                    str(directory / "occt.stl"),
                ],
                directory / "occt.log",
                args.timeout,
                env,
                args.cancel,
            )
            result["oracle_process"] = oracle
            if oracle["status"] != "ok":
                result["status"] = "oracle_" + oracle["status"]
                if oracle["returncode"] > 0:
                    result["status"] = "oracle_error"
                    if (directory / "occt.stl").exists():
                        result["oracle_geometry"] = mesh_metrics(directory / "occt.stl")
                        if not result["oracle_geometry"]["validation"]["valid"]:
                            result["status"] = "oracle_invalid_mesh"
            else:
                result["oracle"] = compare_meshes(
                    directory / "mesh.stl",
                    directory / "occt.stl",
                    args.relative_tolerance,
                    args.absolute_tolerance,
                )
                if not result["oracle"]["passed"]:
                    result["status"] = "oracle_mismatch"
        if digest(source) != entry["sha256"]:
            result["status"] = "input_changed"
        # Baseline regressions are determined later; retain their evidence too.
        if args.meshes == "failures" and result["status"] == "ok" and not args.compare:
            (directory / "mesh.stl").unlink(missing_ok=True)
            (directory / "occt.stl").unlink(missing_ok=True)
    except Exception as error:
        result["status"] = "harness_error"
        result["message"] = str(error)
    write_json(directory / "result.json", result)
    return result


def compare(results, baseline, config, threshold):
    if baseline.get("schema") != SCHEMA:
        raise ValueError(
            "unsupported baseline schema (legacy baselines must be re-recorded)"
        )
    changes = []
    old = {r["path"]: r for r in baseline["results"]}
    current = {r["path"]: r for r in results}
    for key in ("occt", "relative_tolerance", "absolute_tolerance"):
        if baseline["config"][key] != config[key]:
            changes.append(f"validation configuration changed: {key}")
    for name in sorted(old.keys() ^ current.keys()):
        changes.append(f"{name}: missing from current run or baseline")
    if threshold is not None:
        for key in ("threads", "jobs", "repeat", "warmup", "rust_log"):
            if baseline["config"][key] != config[key]:
                changes.append(f"benchmark configuration changed: {key}")
    for name in sorted(old.keys() & current.keys()):
        before, after = old[name], current[name]
        if before["sha256"] != after["sha256"]:
            changes.append(f"{name}: input content changed; not comparable")
            continue
        if before["status"] == "ok" and after["status"] != "ok":
            changes.append(f"{name}: ok -> {after['status']}")
        for key in ("errors", "panics", "log_warn", "log_error"):
            if after.get("metrics", {}).get(key, 0) > before.get("metrics", {}).get(
                key, 0
            ):
                changes.append(f"{name}: {key} increased")
        for key in ("triangles", "faces"):
            if before.get("metrics", {}).get(key) != after.get("metrics", {}).get(key):
                changes.append(f"{name}: {key} changed (inspect mesh)")
        if threshold is not None and before["status"] == after["status"] == "ok":
            a = before["timing"]["process_ms"]["median"]
            b = after["timing"]["process_ms"]["median"]
            if b > a * (1 + threshold):
                changes.append(f"{name}: processing median {a:.3f} -> {b:.3f} ms")
    return changes


def markdown(report):
    lines = [
        "# Foxtrot corpus report",
        "",
        f"Status counts: {dict(Counter(r['status'] for r in report['results']))}",
        "",
        "| Model | Status | Median parse + mesh (ms) | Triangles | Quality review | Diagnostics |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for r in sorted(
        report["results"],
        key=lambda r: (
            r["status"] == "ok",
            not any(r.get("quality_diagnostics", {}).values()),
            -r.get("timing", {}).get("process_ms", {}).get("median", 0),
        ),
    ):
        name = r["path"].replace("|", "\\|").replace("\n", " ")
        median = r.get("timing", {}).get("process_ms", {}).get("median", 0)
        quality = ", ".join(f"{k}: {v}" for k, v in r.get("quality_diagnostics", {}).items() if v)
        lines.append(
            f"| {name} | {r['status']} | {median:.3f} | {r.get('metrics', {}).get('triangles', '—')} | {quality or '—'} | [artifacts]({r['artifacts']}/result.json) |"
        )
    lines += ["", "## Baseline changes", ""] + [f"- {c}" for c in report["changes"]]
    lines += [
        "",
        "See manifest.json for exact inputs, results.json for raw samples and configuration,",
        "and cases/*/run-*.log for diagnostics/backtraces. Timing excludes STL export and process startup.",
        "Each result records whether acceptance used browser f32 triangle-buffer metrics or the legacy strict STL gate; these classifications are not interchangeable fixes or visual verification.",
        "Browser isolated degenerates, zero normals, and f64 degenerates are named review diagnostics, not strict failures. STL is optional diagnostic/world-unit transport for browser-metric workers.",
        "OCCT checks bounds and area only, not topology or mesh equivalence.",
        "",
    ]
    return "\n".join(lines)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("TEST_ASSETS_DIR", "test_assets")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new output directory; never overwrites a run",
    )
    parser.add_argument(
        "--worker", type=Path, default=REPO / "target/release/examples/corpus_worker"
    )
    parser.add_argument(
        "--include", action="append", help="relative-path glob; repeatable"
    )
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--sample", type=positive)
    parser.add_argument("--seed", default="0")
    parser.add_argument(
        "--manifest", type=Path, help="replay an exact manifest; verifies input hashes"
    )
    parser.add_argument(
        "--rerun", type=Path, help="rerun non-ok cases from results.json"
    )
    parser.add_argument("--repeat", type=positive, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--jobs", type=positive, default=1)
    parser.add_argument(
        "--threads", type=positive, default=1, help="Rayon threads per worker"
    )
    parser.add_argument(
        "--timeout",
        type=nonnegative,
        default=60,
        help="seconds per invocation, including OCCT",
    )
    parser.add_argument(
        "--meshes",
        choices=["none", "all", "failures"],
        default="failures",
        help="none skips STL export and validation; requires a browser-metric worker",
    )
    parser.add_argument("--occt", action="store_true")
    parser.add_argument("--relative-tolerance", type=nonnegative, default=0.05)
    parser.add_argument("--absolute-tolerance", type=nonnegative, default=0.01)
    parser.add_argument(
        "--compare", type=Path, help="previous results.json (exact corpus required)"
    )
    parser.add_argument(
        "--timing-threshold",
        type=nonnegative,
        help="opt-in per-model fractional timing regression gate",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.timeout <= 0:
        parser.error("warmup must be nonnegative and timeout must be positive")
    if (args.manifest or args.rerun) and (
        args.include or args.exclude or args.sample or (args.manifest and args.rerun)
    ):
        parser.error(
            "manifest/rerun selection cannot be combined with each other or filters/sampling"
        )
    if args.timing_threshold is not None and not args.compare:
        parser.error("--timing-threshold requires --compare")
    if args.meshes == "none" and args.occt:
        parser.error("--occt requires STL export; use --meshes failures or all")
    args.root, args.output, args.worker = (
        args.root.resolve(),
        args.output.resolve(),
        args.worker.resolve(),
    )
    try:
        if not args.root.is_dir():
            raise ValueError(f"corpus directory does not exist: {args.root}")
        if not args.worker.is_file():
            raise ValueError(
                "build worker first: cargo build --release -p triangulate --example corpus_worker"
            )
        if args.manifest:
            entries = load_manifest(args.manifest, args.root)
        elif args.rerun:
            previous = load_report(args.rerun)
            names = {r["path"] for r in previous["results"] if r["status"] != "ok"}
            entries = [
                e
                for e in load_manifest(args.rerun.parent / "manifest.json", args.root)
                if e["path"] in names
            ]
        else:
            entries = discover(
                args.root, args.include or ["*"], args.exclude, args.seed, args.sample
            )
        if not entries:
            raise ValueError("no STEP files selected")
        baseline = load_report(args.compare) if args.compare else None
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output / "manifest.json",
            {"schema": SCHEMA, "seed": args.seed, "files": entries},
        )
        config = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        config["rust_log"] = os.environ.get("RUST_LOG", "warn")
        report = {
            "schema": SCHEMA,
            "config": config,
            "platform": platform.platform(),
            "python": sys.version,
            "worker_sha256": digest(args.worker),
            "results": [],
            "changes": [],
        }
        try:
            report["cadquery_ocp"] = version("cadquery-ocp") if args.occt else None
        except PackageNotFoundError:
            report["cadquery_ocp"] = None
        # Provenance is informational: the worker hash is authoritative, not checkout HEAD.
        for key, command in [
            ("git_head", ["git", "rev-parse", "HEAD"]),
            ("git_status", ["git", "status", "--porcelain"]),
            ("rustc", ["rustc", "--version"]),
        ]:
            try:
                report[key] = subprocess.check_output(
                    command, cwd=REPO, stderr=subprocess.DEVNULL, text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                report[key] = None
        args.cancel = Event()
        pool = ThreadPoolExecutor(max_workers=args.jobs)
        try:
            with (args.output / "progress.jsonl").open("w") as progress:
                futures = [pool.submit(run_file, entry, args) for entry in entries]
                for future in as_completed(futures):
                    result = future.result()
                    report["results"].append(result)
                    progress.write(json.dumps(result, allow_nan=False) + "\n")
                    progress.flush()
                    print(f"{result['status']:22} {result['path']}", flush=True)
        except KeyboardInterrupt:
            print(
                "Interrupted; replay manifest.json to complete this selection.",
                file=sys.stderr,
            )
            return 130
        finally:
            args.cancel.set()
            pool.shutdown(wait=True, cancel_futures=True)
        report["results"].sort(key=lambda r: r["path"])
        if baseline is not None:
            report["changes"] = compare(
                report["results"], baseline, config, args.timing_threshold
            )
        write_json(args.output / "results.json", report)
        review = {
            r["path"]
            for r in report["results"]
            if r["status"] != "ok"
            or any(r.get("quality_diagnostics", {}).values())
        }
        write_json(
            args.output / "review-manifest.json",
            {"schema": SCHEMA, "files": [e for e in entries if e["path"] in review]},
        )
        (args.output / "report.md").write_text(markdown(report))
        print(f"Report: {args.output / 'report.md'}")
        return int(
            bool(report["changes"])
            or any(r["status"] != "ok" for r in report["results"])
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"corpus: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
