# Foxtrot

Foxtrot is a **fast** viewer for
[STEP files](https://en.wikipedia.org/wiki/ISO_10303-21),
a standard interchange format for mechanical [CAD](https://en.wikipedia.org/wiki/Computer-aided_design).
It is an _experimental_ project built from the ground up,
including new libraries for parsing and triangulation.

This repository includes a simple native GUI:

![Motherboard example](https://mattkeeter.com/projects/foxtrot/rpi.png)  
([demo model source](https://grabcad.com/library/raspberry-pi-3-reference-design-model-b-rpi-raspberrypi-raspberry-pi-1))

In addition, the same code can run in a browser (click to [see the demo](https://mattkeeter.com/projects/foxtrot/demo)):

[![Browser example](https://www.mattkeeter.com/projects/foxtrot/foxtrot365.png)](https://mattkeeter.com/projects/foxtrot/demo)  
([demo model source](https://grabcad.com/library/6-dof-mechanical-arm-claw-kit-1))

For more background on the project, check out [this writeup](https://mattkeeter.com/projects/foxtrot)
by one of the main authors.

## Quick start
(Prerequisite: [install Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html), and clone this repository)
```sh
cargo run --release -- examples/cube_hole.step
```

## WebAssembly demo
(Prerequisite: [install `wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) and clone this repository)
```sh
cd wasm
wasm-pack build --target no-modules
python3 -m http.server --directory deploy # or the simple server of your choice
```
Then, open the local server's URL (typically `127.0.0.1:8000`)
and select a sample file from the list.

The deploy directory links to the generated `pkg/wasm.js` and
`pkg/wasm_bg.wasm`; rebuild them after Rust changes. The worker transfers a
`Float32Array` plus completion and source-identified failure records, rather
than treating a partial mesh as a successful conversion. Partial-load warnings
remain visible when the camera moves.

## Subsystems
- `cdt`: Constrained Delaunay triangulation (standalone)
- `express`: Parser for EXPRESS schemas files and a matching code generation
  system
- `step`: Auto-generated STEP file parser.  This take a _very_ long time to
  compile, so it is isolated into this crate.
- `triangulate`: Converts a file loaded by `step` into a triangle mesh, using
  `cdt` as its core
- `nurbs`: A handful of NURBS / B-spline algorithms used by `triangulate`
- `gui`: GUI for rendering STEP files, using WebGPU
- `wasm`: Scaffolding to run in the browser using WebAssembly

## Code generation
`step/src/ap214.rs` is automatically generated from
`10303-214e3-aim-long.exp`, which is available via [CVS](https://en.wikipedia.org/wiki/Concurrent_Versions_System) [here](http://www.steptools.com/stds/help/cvshowto.html)
(check out the `APs` folder).

To regenerate, run
```
cargo run --release --example gen_exp -- path/to/APs/10303-214e3-aim-long.exp step/src/ap214.rs
```

## Regression testing and benchmarking
STEP models are not vendored into this repository. Two public libraries
provide a large, varied corpus of real-world component models for testing
and benchmarking the parser and tessellator:

- [Würth Elektronik KiCad Library](https://github.com/WurthElektronik/KiCad-Library)
  — several thousand vendor-authored models under `3dmodels/`, exported from
  Solid Edge, FreeCAD, and other tools, spanning AP203, AP214, and AP242.
- [KiCad packages3D](https://gitlab.com/kicad/libraries/kicad-packages3D/)
  — the official KiCad 3D model library, mostly FreeCAD/OCCT-generated
  generic packages (`CC-BY-SA-4.0 WITH KiCad-libraries-exception`).

Clone either library outside the repository (or under ignored `local/`).
Keep the library's license and revision with your corpus; the harness does not
download or redistribute models. Start with the checked-in examples:
```sh
cargo build --release -p triangulate --example corpus_worker
python3 scripts/corpus.py examples --output local/smoke --meshes all
```

The new harness requires Python 3.10+ and uses only the standard library unless
you enable OCCT. It recursively discovers `.step`/`.stp` files, regardless of
extension case. Each model runs in a separate process with a hard timeout;
on POSIX the harness kills the worker process group and reaps the worker.
Partial tessellation, panics, crashes, nonfinite browser attributes, empty or
entirely collapsed browser meshes, and oracle failures produce a nonzero exit.
The worker evaluates the same centered f32 position/normal/color buffer as
Three.js, produced from f64 geometry by `Mesh::to_triangle_buffer`.

Isolated collapsed triangles, zero normals, and f64 degenerates remain explicit
`quality_diagnostics`, not automatic whole-model failures. `ok` means processing
and browser-buffer checks passed, **not** that visual quality is established.
Review flagged models at useful viewing scales for holes, silhouette errors,
and shading defects. No triangles are deleted or vertices merged to pass checks.
STL remains a world-coordinate diagnostic for coarse measurements and optional
OCCT comparison; its f32 precision does not gate browser output. Reports identify
their classification basis. Worker protocol schema 2 is required; old aggregate
counter and STL-fallback workers are intentionally unsupported.

For fast browser-only iteration, use `--meshes none`: the worker computes the
same browser diagnostics but skips STL serialization, disk writes and Python
geometry validation. Use `--meshes all` on a focused selection when you need
world-coordinate geometry or `--occt`. Reports list failures and quality-review
cases before unflagged successes. `review-manifest.json` selects failures and
quality-flagged models with their source hashes; replay it with `--manifest`
to investigate only that set, without another full scan.

To capture the exact browser buffer for a Three.js reproduction:
```sh
target/release/examples/corpus_worker model.step metrics.json - browser.bin
```
`browser.bin` contains little-endian f32 values with stride nine: position,
normal, color, expanded three vertices per triangle. No STEP reparsing or
alternate mesh conversion is required on the JavaScript side.

### Select, benchmark, compare

```sh
# Stable hash-ranked sampling; globs match relative paths and may be repeated.
python3 scripts/corpus.py path/to/KiCad-Library/3dmodels \
  --sample 100 --seed 42 --exclude '*Connector*' \
  --repeat 5 --warmup 1 --output local/before

# Rebuild after your change, then use precisely the same inputs.
cargo build --release -p triangulate --example corpus_worker
python3 scripts/corpus.py path/to/KiCad-Library/3dmodels \
  --manifest local/before/manifest.json --compare local/before/results.json \
  --repeat 5 --warmup 1 --timing-threshold 0.15 --output local/after
```

`TEST_ASSETS_DIR` supplies the root if the positional root is omitted.
`--jobs` controls model concurrency and `--threads` controls Rayon threads per
worker (both default to 1). Use concurrency for functional sweeps; benchmark
serially on the same idle machine with identical settings. Warmups are discarded
fresh-process runs, warming OS caches, not an in-process allocator or thread pool.
Reports retain raw samples, median/min/max phase timings, worker SHA-256, checkout
revision/status, and configuration. The timing gate uses median parse +
triangulation time, excluding reads, STL export, validation, OCCT, and startup;
those worker I/O phases and total worker wall time are recorded separately.
Timing gates are opt-in because short runs and shared machines are noisy.

The manifest records relative paths, sizes, and SHA-256 hashes, so duplicate
basenames cannot collide and replay rejects changed inputs. A fixed seed repeats
selection for an unchanged corpus; use the manifest to freeze it as the library
grows. Comparisons require the same selected paths and contents, report changed
face/triangle counts and increased tessellation failures, and reject incompatible
benchmark settings when timing is gated. Counts changing is a review signal,
not necessarily a bug. Any current functional failure still fails the run even
if it existed in the baseline. The old thread-based `regression_test` executable
is removed; use this process-isolated harness and record a fresh baseline instead
of importing its basename-keyed JSON.

### Debug and close the loop

Each new output directory contains:

- `report.md`: failures and quality-review cases first, with links to diagnostics.
- `results.json`: machine-readable report and baseline, including geometry metrics.
- `manifest.json`: exact replayable selection, with input hashes.
- `review-manifest.json`: replayable failures and quality-review cases.
- `progress.jsonl`: flushed completed results; per-case results also survive interruption.
- `cases/<path-hash>/`: per-invocation logs/backtraces, worker metrics, a reproduction
  command in `result.json`, and any requested binary STL meshes. Failed meshes are
  retained by default; `--meshes none` writes no STL, while `--meshes all` or
  `--compare` retains successful meshes too. Inputs are not copied.

```sh
# Recheck just failures, with verbose Rust diagnostics and saved meshes.
RUST_LOG=debug python3 scripts/corpus.py path/to/KiCad-Library/3dmodels \
  --rerun local/after/results.json --timeout 120 --meshes all --output local/recheck

# Or isolate one relative path, then open its mesh in an STL viewer.
python3 scripts/corpus.py examples --include 'cube_hole.step' \
  --meshes all --output local/one-model
```

The output directory must not already exist. Exit codes are 0 for success,
1 for model failures or baseline changes, and 2 for setup/selection/report errors.
Ctrl-C cancels queued work, terminates active workers, and exits with code 130.
An interrupted run retains individual case results but is not a complete baseline;
replay its manifest into a new directory. Timeouts default to 60 seconds per
invocation, independently for Foxtrot and OCCT. There is no memory limit: reduce
concurrency or use OS/container limits for untrusted or very large inputs.

### Optional OCCT oracle

```sh
python3 -m venv local/occt-env
local/occt-env/bin/pip install cadquery-ocp trimesh scipy rtree
local/occt-env/bin/python scripts/corpus.py examples --occt \
  --meshes all --output local/oracle
```

OCCT independently reads STEP and emits a millimeter-scale STL with 0.01 mm
linear and 0.1 rad angular deflection. In addition to bounds and area, the harness
compares **bidirectional sampled point-to-triangle distances** using Trimesh's
spatial index. Foxtrot → OCCT detects extra/displaced surfaces; OCCT → Foxtrot
detects missing surfaces. There is no alignment, rescaling, mesh repair or vertex
correspondence: different triangulations can represent the same surface.

Each direction uses 10,000 seeded, area-weighted samples plus up to 10,000 face
centroids to probe smaller faces. `--surface-samples` controls that budget.
`--surface-tolerance` defaults to **0.1 mm**; the worst sampled distance in both
directions must be within it. Choose tolerances appropriate to the part's smallest
important features; this default is not an accuracy guarantee. `oracle.json`
records p50/p95/p99, RMS, the worst sampled distance and its coordinates, and the
area-sampled fraction outside tolerance. Percentiles/fractions use only the
area-weighted samples; the maximum also includes the face probes.

Defaults are 5% relative tolerance and 0.01 mm absolute bounds tolerance;
adjust `--relative-tolerance` and `--absolute-tolerance` for your models.
Bounds allow absolute tolerance + relative tolerance × reference diagonal;
area allows absolute tolerance squared + relative tolerance × reference area.
Signed volume is diagnostic only because STEP can contain open shells or
inconsistent winding. Zero-area STL facets are counted but excluded from the
distance query; they do not block comparison of the remaining surface. Neither
exported mesh is modified. Empty/nonfinite/zero-area meshes are not comparable.

**Agreement is evidence, not proof:** sampling can miss tiny holes; it does not
certify Hausdorff distance, topology, watertightness, winding or shading. OCCT can
also be wrong. Both STL transports have world-coordinate f32 rounding, so extreme
coordinate offsets or very tight tolerances need a higher-precision follow-up.
Both meshes and detailed deltas are kept on a mismatch. Conversion and distance
queries run together in the timeout-isolated oracle subprocess. Missing optional
dependencies or failed conversion is an explicit failure, never a skipped check.
Reports record comparison settings and dependency versions. Re-record old OCCT
baselines: aggregate-only acceptance is not equivalent to surface agreement.

### Visual oracle review

Generate a three-pane browser report for every `oracle_mismatch` in a retained
corpus run. The panes share camera, scale and lighting; rotation, zoom and
wireframe controls support inspection beyond the initial view.

```sh
python3 scripts/oracle_report.py local/before --after local/after \
  --output local/visual-report
python3 -m http.server 8090 --directory local/visual-report
```

Both corpus runs must retain meshes and oracle measurements for the same input
hashes. `--after` is required: every comparison shows genuine baseline, current
Foxtrot, and OCCT meshes rather than substituting the baseline for a missing
output. The generator downloads pinned Three.js 0.180.0 modules and its license once into the output;
viewing needs no external assets. Geometry uses neutral flat shading, not STEP
colors or the application's vertex-normal pipeline. Serve only the generated
directory, not the repository or corpus roots. In an orb, use a supervised
service with a portal to share the report.

Harness tests (no Rust build or corpus download required):
```sh
python3 -m unittest discover -s scripts -p 'test_corpus*.py'
```
Run the same command with `local/occt-env/bin/python` to include the optional
surface-distance tests (different triangulations, displaced/missing faces and
zero-area facets).

To extract STEP models embedded in a KiCad board or footprint for local
testing, use `scripts/extract_steps.py` (requires `pip install zstandard`).

## License
© 2021 [Formlabs](https://formlabs.com)

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Disclaimer
Foxtrot is a proof-of-concept demo, not an industrial-strength CAD kernel.
It may not work for your models!
Even in the screenshots above,
there are a handful of surfaces that it fails to triangulate;
look in the console for details.

This isn't an official Formlabs project (experimental or otherwise),
it is just code that happens to be owned by Formlabs.

No one at Formlabs is paid to maintain this,
so set your expectations for support accordingly.

## References
[STEP Integrated Definitions](https://www.steptools.com/stds/stp_expg/aim.html)
