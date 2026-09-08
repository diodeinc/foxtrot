#!/usr/bin/env python3
"""Serial, process-isolated benchmarks of Diode's actual board-to-scene pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from corpus import digest, execute, write_json
from performance import THREAD_ENV, positive, positive_int

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    repos = []
    for source, alias in json.loads((HERE / "board-corpus.json").read_text()):
        url = f"https://code.diode.computer/{source}.git"
        checkout = root / alias
        entry = dict(
            url=url, alias=alias, revision=None, acquisition_status="error", boards=[]
        )
        try:
            if not checkout.exists():
                result = execute(
                    ["git", "clone", "--depth", "1", url, str(checkout)],
                    root / f"{alias}.clone.log",
                    120,
                    dict(os.environ, GIT_TERMINAL_PROMPT="0"),
                )
                if result["status"] != "ok":
                    raise ValueError(
                        f"clone {result['status']}; see {alias}.clone.log (authentication may be required)"
                    )

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(checkout), *args], text=True
                ).strip()

            if git("remote", "get-url", "origin") != url:
                raise ValueError("existing checkout has a different origin")
            entry["revision"] = git("rev-parse", "HEAD")
            entry["dirty"] = bool(git("status", "--porcelain"))
            for board in sorted(checkout.rglob("*.kicad_pcb")):
                if board.is_symlink():
                    raise ValueError(f"refusing symlink board: {board}")
                with board.open("rb") as stream:
                    lfs = stream.read(128).startswith(
                        b"version https://git-lfs.github.com/spec/v1"
                    )
                entry["boards"].append(
                    dict(
                        path=board.relative_to(root).as_posix(),
                        size_bytes=board.stat().st_size,
                        sha256=digest(board),
                        lfs_placeholder=lfs,
                    )
                )
            entry["acquisition_status"] = "ok"
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            entry["error"] = str(exc)
        repos.append(entry)
        write_json(root / "inventory.json", dict(schema_version=1, repositories=repos))
        print(
            f"{alias}: {entry['acquisition_status']}, {len(entry['boards'])} boards",
            flush=True,
        )
    return int(any(r["acquisition_status"] != "ok" for r in repos))


def classify(metrics):
    if metrics.get("schema") != 1:
        raise ValueError("unsupported worker schema")
    if metrics.get("status") != "ok":
        return "processing_error"
    # Missing telemetry must never turn a worker mismatch into a successful run.
    for key in (
        "total_ms",
        "stages",
        "counts",
        "foxtrot",
        "validation",
        "serialized_bytes",
        "timings",
    ):
        if key not in metrics:
            raise ValueError(f"missing worker field {key}")
    from performance import _number

    _number(metrics["total_ms"], "total_ms")
    _number(metrics["serialized_bytes"], "serialized_bytes", integer=True)
    for name, value in metrics["timings"].items():
        _number(value, name)
    for key in ("counts", "foxtrot", "validation"):
        for name, value in metrics[key].items():
            _number(value, name, integer=True)
    for stage in metrics["stages"]:
        _number(stage["wall_ms"], "stage wall_ms")
        _number(stage["cpu_ms"], "stage cpu_ms")
        _number(stage["peak_rss_bytes"], "RSS", integer=True, nullable=True)
    if any(
        metrics["validation"][k]
        for k in (
            "nonfinite_vertices",
            "nonfinite_instances",
            "invalid_indices",
            "empty_batches",
        )
    ):
        return "invalid_mesh"
    fox = metrics["foxtrot"]
    if fox["errors"] or fox["panics"] or fox["failed_instances"]:
        return "partial"
    if fox["models"] != metrics["counts"]["unique_models"]:
        return "missing_diagnostics"
    if fox["skipped_instances"] or metrics["counts"]["placeholder_instances"]:
        return "missing_models"
    return "ok"


def compatible(current, prior):
    if current["config"] != prior["config"]:
        raise ValueError(
            "comparison requires matching repeat, thread, timeout and budget settings"
        )
    for key in ("diode", "source_sha256", "instrumentation_patch_sha256"):
        if current["build"][key] != prior["build"][key]:
            raise ValueError(
                f"comparison changes Diode or benchmark implementation: {key}"
            )
    identity = lambda report: [(x["path"], x["sha256"]) for x in report["results"]]
    if identity(current) != identity(prior):
        raise ValueError("comparison board paths or hashes differ")


def render(report):
    blocked = [r for r in report["repositories"] if r["acquisition_status"] != "ok"]
    samples = [s for r in report["results"] for s in r["samples"]]
    finished = sum(
        "metrics" in s and s["metrics"].get("status") == "ok" for s in samples
    )
    lines = [
        "# Diode board-to-scene benchmark",
        "",
        f"Capture: {report['capture_seconds']:.2f}s. Repositories: {len(report['repositories']) - len(blocked)}/{len(report['repositories'])} accessible. "
        f"Boards: {len(report['results'])}. Generated-scene samples: {finished}/{len(samples)}.",
        "",
        "Native execution of the web worker's parse → DNP-inclusive placements → full scene preparation → serialization. "
        "Includes PCB geometry and embedded STEP model tessellation, once per unique model name per board. "
        "No cross-board cache, GPU, browser startup, download or JS/WASM overhead. Validation is outside total_ms. "
        "Foxtrot measures only tessellate_step_bytes (flatten, parse, mesh, color grouping). "
        "RSS is cumulative process peak, not live memory. Failed/incomplete samples are not discarded.",
        "",
        "| Board | Status | Total median s | Foxtrot s | Component s | Board geometry s | RSS MiB | Models | Resolved / placements | Triangles (stored) | Serialized MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["results"]:
        good = [
            s["metrics"]
            for s in case["samples"]
            if s.get("metrics", {}).get("status") == "ok"
        ]
        statuses = ", ".join(sorted({s["status"] for s in case["samples"]}))
        if good:
            m = good[0]
            rss = [
                s["peak_rss_bytes"]
                for v in good
                for s in v["stages"]
                if s["peak_rss_bytes"] is not None
            ]
            c = m["counts"]
            lines.append(
                f"| {case['path']} | {statuses} | {statistics.median(v['total_ms'] for v in good) / 1000:.3f} | "
                f"{statistics.median(v['timings']['foxtrot_call_secs'] for v in good):.3f} | "
                f"{statistics.median(v['timings']['component_tessellation_secs'] for v in good):.3f} | "
                f"{statistics.median(v['timings']['board_geometry_secs'] for v in good):.3f} | "
                f"{max(rss, default=0) / 2**20:.1f} | {c['unique_models']} | {c['resolved_instances']} / {c['total_instances']} | "
                f"{c['triangles']} | {m['serialized_bytes'] / 2**20:.2f} |"
            )
        else:
            lines.append(
                f"| {case['path']} | {statuses} | — | — | — | — | — | — | — | — | — |"
            )
    lines += [
        "",
        "Stored triangles count indexed geometry once, not once per rendered instance. "
        "`missing_models` includes unresolved/unsupported/absent footprint models; placeholders are not successful STEP tessellations. "
        "`partial` includes Foxtrot face failures or failed model instances even if Diode returns a scene. "
        "Zero unique models means the case does not exercise Foxtrot.",
    ]
    if report.get("comparisons"):
        lines += [
            "",
            "## Comparison",
            "",
            "Ratios use Foxtrot direct API time only, current / baseline; smaller is faster.",
        ]
        for c in report["comparisons"]:
            lines.append(f"- {c['path']}: {c['time_ratio']:.3f}×")
    lines += ["", "## Unavailable repositories", ""]
    lines += [f"- {r['alias']}: {r.get('error', 'unavailable')}" for r in blocked]
    lines += ["", "## Acquired repositories without boards", ""]
    lines += [
        f"- {r['alias']}"
        for r in report["repositories"]
        if r["acquisition_status"] == "ok" and not r["boards"]
    ]
    return "\n".join(lines) + "\n"


def bench(args):
    inventory = json.loads((args.root / "inventory.json").read_text())
    build = json.loads(args.build.read_text())
    worker = Path(build["worker"])
    if digest(worker) != build["worker_sha256"]:
        raise ValueError("worker differs from build metadata; rebuild")
    cases = [
        dict(b, repository=r["alias"])
        for r in inventory["repositories"]
        if r["acquisition_status"] == "ok"
        for b in r["boards"]
    ]
    cases.sort(key=lambda c: c["path"])
    for case in cases:
        path = (args.root / case["path"]).resolve()
        if not path.is_relative_to(args.root) or digest(path) != case["sha256"]:
            raise ValueError(
                f"board changed or escapes corpus: {case['path']}; fetch inventory again"
            )
        case["samples"] = []
    report = dict(
        schema=1,
        config=dict(
            repeat=args.repeat, threads=1, timeout=args.timeout, budget=args.budget
        ),
        build=build,
        platform=platform.platform(),
        repositories=inventory["repositories"],
        results=cases,
    )
    prior = json.loads(args.compare.read_text()) if args.compare else None
    if prior:
        compatible(report, prior)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = dict(os.environ, **THREAD_ENV)
    for case in cases:
        directory = (
            args.output
            / "cases"
            / hashlib.sha256(case["path"].encode()).hexdigest()[:20]
        )
        directory.mkdir(parents=True)
        for index in range(args.repeat):
            remaining = args.budget - (time.monotonic() - started)
            metrics_path = directory / f"{index}.json"
            if case.get("lfs_placeholder"):
                sample = dict(status="lfs_placeholder")
            elif remaining <= 0:
                sample = dict(status="budget_exhausted")
            else:
                sample = execute(
                    [str(worker), str(args.root / case["path"]), str(metrics_path)],
                    directory / f"{index}.log",
                    min(args.timeout, remaining),
                    env,
                )
                if metrics_path.exists():
                    try:
                        sample["metrics"] = json.loads(metrics_path.read_text())
                        classification = classify(sample["metrics"])
                        if sample["status"] != "timeout":
                            sample["status"] = (
                                classification
                                if sample["returncode"] == 0
                                else "processing_error"
                            )
                    except (ValueError, KeyError, TypeError) as exc:
                        sample["status"] = "invalid_metrics"
                        sample["error"] = str(exc)
                        sample.pop("metrics", None)
                elif sample["status"] == "ok":
                    sample["status"] = "missing_metrics"
            sample["artifacts"] = str(directory.relative_to(args.output))
            case["samples"].append(sample)
            report["capture_seconds"] = time.monotonic() - started
            write_json(args.output / "results.json", report)
            print(
                f"{sample['status']:20} {case['path']} [{index + 1}/{args.repeat}]",
                flush=True,
            )
    if prior:
        report["comparisons"] = []
        for current, old in zip(cases, prior["results"]):
            # Missing authored models are comparable only when coverage stays
            # identical; never award a speedup for doing less component work.
            signatures = [
                (
                    s["status"],
                    s.get("metrics", {}).get("counts"),
                    s.get("metrics", {}).get("foxtrot"),
                )
                for c in (current, old)
                for s in c["samples"]
            ]
            if (
                signatures
                and all(x == signatures[0] for x in signatures)
                and signatures[0][0] in ("ok", "missing_models")
            ):
                before = statistics.median(
                    s["metrics"]["timings"]["foxtrot_call_secs"] for s in old["samples"]
                )
                after = statistics.median(
                    s["metrics"]["timings"]["foxtrot_call_secs"] for s in current["samples"]
                )
                if before > 0:
                    report["comparisons"].append(
                        dict(path=current["path"], time_ratio=after / before)
                    )
    report["capture_seconds"] = time.monotonic() - started
    write_json(args.output / "results.json", report)
    (args.output / "report.md").write_text(render(report))
    print(f"Report: {args.output / 'report.md'} ({report['capture_seconds']:.1f}s)")
    return int(
        not cases
        or any(r["acquisition_status"] != "ok" for r in inventory["repositories"])
        or any(s["status"] != "ok" for c in cases for s in c["samples"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "run"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=ROOT / "local/board-corpus")
        if command == "run":
            p.add_argument(
                "--build",
                type=Path,
                default=ROOT / "local/board-worker-build/build.json",
            )
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--repeat", type=positive_int, default=2)
            p.add_argument("--timeout", type=positive, default=120)
            p.add_argument("--budget", type=positive, default=240)
            p.add_argument("--compare", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    try:
        return fetch(args.root) if args.command == "fetch" else bench(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
