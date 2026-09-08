#!/usr/bin/env python3
"""Build the native PCB3D benchmark worker against an explicit Diode checkout."""

import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = pathlib.Path(__file__).with_name("board_worker.rs")
OLD = "        Ok(Ok((mesh, _diag))) => Ok(mesh),"
NEW = """        Ok(Ok((mesh, diag))) => {
            log::info!("foxtrot_bench_diagnostics faces={} errors={} panics={}", diag.num_faces, diag.num_errors(), diag.num_panics());
            Ok(mesh)
        }"""


def run(*args, cwd=None, env=None):
    return subprocess.run(
        args, cwd=cwd, env=env, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_info(path):
    revision = run("git", "rev-parse", "HEAD", cwd=path)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {"revision": revision, "dirty": bool(diff), "dirty_diff_sha256": sha(diff)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diode", type=pathlib.Path, default=ROOT / "local/diode-benchmark")
    p.add_argument(
        "--output", type=pathlib.Path, default=ROOT / "local/board-worker-build"
    )
    a = p.parse_args()
    diode = a.diode.resolve()
    out = a.output.resolve()
    scene = diode / "projects/editor/crates/pcb3d-scene/src/lib.rs"
    if not (diode / "Cargo.toml").is_file() or not scene.is_file():
        sys.exit(f"invalid Diode checkout: {diode}")
    content = scene.read_text()
    if OLD in content:
        if content.count(OLD) != 1:
            sys.exit("refusing instrumentation: expected exactly one source match")
        scene.write_text(content.replace(OLD, NEW))
    elif NEW not in content or content.count(NEW) != 1:
        sys.exit("refusing instrumentation: scene source has unknown drift")
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy2(SOURCE, out / "src/main.rs")
    rel = lambda x: os.path.relpath(x, out)
    manifest = f"""[workspace]\n\n[package]\nname="foxtrot-board-worker"\nversion="0.1.0"\nedition="2024"\n\n[dependencies]\nanyhow="1"\nlibc="0.2"\nlog="0.4"\nserde_json="1"\neditor-kicad={{path={json.dumps(rel(diode / "projects/editor/crates/kicad"))}}}\neditor-pcb3d-scene={{path={json.dumps(rel(diode / "projects/editor/crates/pcb3d-scene"))},features=["scene-prep"]}}\n\n[patch."https://github.com/diodeinc/foxtrot.git"]\ntriangulate={{path={json.dumps(str(ROOT / "triangulate"))}}}\nstep={{path={json.dumps(str(ROOT / "step"))}}}\n\n[profile.release]\nincremental=false\n"""
    (out / "Cargo.toml").write_text(manifest)
    metadata = json.loads(run("cargo", "metadata", "--format-version=1", cwd=out))
    tri = [x for x in metadata["packages"] if x["name"] == "triangulate"]
    expected = (ROOT / "triangulate/Cargo.toml").resolve()
    if len(tri) != 1 or pathlib.Path(tri[0]["manifest_path"]).resolve() != expected:
        sys.exit(f"triangulate did not resolve to current Foxtrot: {tri}")
    env = os.environ.copy()
    env["CARGO_INCREMENTAL"] = "0"
    run("cargo", "build", "--release", "--jobs", "1", "--locked", cwd=out, env=env)
    worker = out / "target/release/foxtrot-board-worker"
    lock = json.loads(
        run("cargo", "metadata", "--locked", "--format-version=1", cwd=out)
    )
    packages = sorted(
        {
            (
                p["name"],
                p["version"],
                p.get("source") or pathlib.Path(p["manifest_path"]).parent.as_posix(),
            )
            for p in lock["packages"]
        }
    )
    instrument = hashlib.sha256(NEW.encode()).hexdigest()
    info = {
        "schema": 1,
        "diode": git_info(diode),
        "foxtrot": git_info(ROOT),
        "compiler": run("rustc", "-Vv"),
        "source_sha256": sha(SOURCE.read_bytes()),
        "worker_sha256": sha(worker.read_bytes()),
        "instrumentation_patch_sha256": instrument,
        "cargo_lock_sha256": sha((out / "Cargo.lock").read_bytes()),
        "resolved_packages": [
            {"name": n, "version": v, "source": s} for n, v, s in packages
        ],
        "triangulate_manifest": str(expected),
        "worker": str(worker),
    }
    (out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(worker)


if __name__ == "__main__":
    main()
