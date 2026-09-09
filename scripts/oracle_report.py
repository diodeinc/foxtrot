#!/usr/bin/env python3
"""Build a self-contained three-pane review from retained corpus oracle meshes."""

import argparse
import json
from pathlib import Path
import shutil
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Corpus output directory")
    parser.add_argument("--after", required=True, type=Path, help="Current Foxtrot corpus output")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notes", type=Path, help="JSON mapping input paths to investigation notes")
    args = parser.parse_args()
    notes = json.loads(args.notes.read_text()) if args.notes else {}
    before = json.loads((args.before / "results.json").read_text())
    after = {
        row["path"]: row for row in json.loads((args.after / "results.json").read_text())["results"]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    def copy_mesh(source, name):
        shutil.copyfile(source, args.output / name)
        return name

    def distances(row):
        d = row.get("oracle", {}).get("surface_distance")
        if d is None:
            return None
        return [d[k]["max_sampled_mm"] for k in ("actual_to_reference", "reference_to_actual")]

    for index, row in enumerate(before["results"]):
        if row["status"] not in ("oracle_mismatch", "tessellation_error"):
            continue
        case = args.before / row["artifacts"]
        old = copy_mesh(case / "mesh.stl", f"{index}-before.stl")
        item = {"path": row["path"], "before": old,
                "reference": copy_mesh(case / "occt.stl", f"{index}-oracle.stl"),
                "beforeDistances": distances(row),
                "beforeCompletion": row.get("metrics", {}).get("completion", "unknown"),
                "note": notes.get(row["path"], "")}
        fixed = after[row["path"]]
        if fixed["sha256"] != row["sha256"]:
            raise ValueError(f"Input changed: {row['path']}")
        mesh = args.after / fixed["artifacts"] / "mesh.stl"
        if not mesh.exists():
            raise ValueError(f"After mesh missing: {row['path']}")
        item["after"] = copy_mesh(mesh, f"{index}-after.stl")
        item["afterStatus"] = fixed["status"]
        item["afterCompletion"] = fixed.get("metrics", {}).get("completion", "unknown")
        item["afterDistances"] = distances(fixed)
        rows.append(item)
    (args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("oracle_report.html"), args.output / "index.html")
    # Pin and vendor the viewer so the resulting report needs no external requests.
    for path in ["build/three.module.js", "build/three.core.js",
                 "examples/jsm/controls/OrbitControls.js", "examples/jsm/loaders/STLLoader.js"]:
        target = args.output / Path(path).name
        if not target.exists():
            urllib.request.urlretrieve(f"https://cdn.jsdelivr.net/npm/three@0.180.0/{path}", target)
    urllib.request.urlretrieve("https://cdn.jsdelivr.net/npm/three@0.180.0/LICENSE", args.output / "THREE-LICENSE.txt")
    print(f"Generated {len(rows)} three-pane comparisons in {args.output}")


if __name__ == "__main__":
    main()
