# Würth STEP corpus worklog

## Scope and acceptance

Run every `.step`/`.stp` file in the public Würth Elektronik KiCad library with
the repository's `scripts/corpus.py` harness. Record the corpus revision and
hash manifest, retain failure diagnostics, and provide a root-cause assessment
for every failing input. This is an assessment, not a promise to repair every
unsupported geometry. No processing code is changed for the baseline.

A pass requires the existing harness's complete tessellation and finite,
nonempty, nondegenerate mesh checks. The optional OCCT comparison is not enabled
for the initial sweep; a harness pass is not proof of geometric equivalence.

## 2026-09-05 — Setup

- Read the harness, worker, and documented acceptance criteria.
- Initial checkout has no modifications and no downloaded Würth corpus.
- Orb resources: 16 CPUs, 31 GiB RAM, approximately 60 GiB free disk.
- Plan: build release worker and test harness; acquire the upstream library
  under ignored `local/`, retaining its license; run every discovered model;
  group diagnostics by mechanism and inspect the input and source for each
  failure. Recheck ambiguous or timeout cases with focused diagnostics.

## Baseline execution

- `cargo build --release -p triangulate --example corpus_worker` succeeds
  (52.67 s; existing parser lifetime warnings).
- `python3 -m unittest discover -s scripts -p 'test_corpus*.py'`: 19 tests pass.
- `python3 scripts/corpus.py examples --output local/wurth-smoke`: 3/3 pass.
- Cloned https://github.com/WurthElektronik/KiCad-Library into `local/wurth`.
  Corpus revision: `40adcee44afdab5f4ed8038699f78bd784e0f594`.
  Upstream license and disclaimer PDFs remain in that checkout.
- Discovered **7,328 files**, all under `3dmodels/`, totaling 9,257,894,601 bytes.
- Full sweep started with no sampling or exclusions:

  ```sh
  RUST_LOG=triangulate=debug,cdt=warn,step=warn python3 scripts/corpus.py \
    local/wurth/3dmodels --jobs 8 --threads 1 --timeout 60 \
    --output local/wurth-baseline > local/wurth-baseline.log 2>&1
  ```

- Debug logging for triangulation retains individual face failure reasons.
  Per-file logs, backtraces, failure meshes, metrics, and reproduction commands
  live in `local/wurth-baseline/cases/`; the manifest freezes every input hash.

## Investigation and corrected sweep

- Found a false-pass bug: `open_shell` and `closed_shell` log early
  `advanced_face` errors but do not increment `Stats::num_errors`. For example,
  `A_Wurth_WA-BCPH_79578211.step` loses a `DEGENERATE_TOROIDAL_SURFACE` face yet
  the original worker reports `ok`. Added the missing increments and a test
  covering unsupported surfaces in both shell types. No geometry algorithms
  are changed.
- `cargo test -p triangulate --lib`: 10 tests pass, including the new test.
- Initial sweep bottleneck is Python STL validation: one harness process uses
  approximately one CPU despite `--jobs 8` (threaded Python geometry loop).
  Started a corrected **full** sweep as eight separate harness processes with
  disjoint round-robin slices (`files[i::8]`) of the original manifest. Each
  process uses `--jobs 1 --threads 1 --timeout 60`; all 7,328 inputs are included
  exactly once. No timing comparison is made across these configurations.
- Corrected worker built separately with
  `CARGO_TARGET_DIR=local/wurth-corrected-target cargo build --release -p triangulate --example corpus_worker`
  so the running original sweep's binary is unchanged. Final shard artifacts
  and manifest selections are in `local/wurth-final/`.
- Confirmed initial failure families: collapsed/retraced UV contours, CDT
  constraint-walk failures, unsupported degenerate toroidal surfaces, and
  degenerate triangles. A focused f64/f32 probe found both already-degenerate
  triangles and tiny f64 triangles that collapse during binary STL rounding.
  Exact upstream causes remain qualified where only a diagnostic site is known.

## Follow-up evidence

- `cargo test -p triangulate`: 10 unit tests and the checked-in-model integration
  test pass. The latter processes all three example STEP files.
- Confirmed a separate input-format failure:
  `Inductor_THT_Wurth.3dshapes/L_Wurth_WE-CMANC-M_7848031002.step` starts with
  `**PARASOLID`, not `ISO-10303-21`. Its `.step` extension is misleading.
  `StepFile::into_blocks` panics at `step/src/step_file.rs:97` when scanning
  delimiter-free trailing Parasolid data. A separate harness run with a
  120-second timeout reproduces the same immediate crash; this is not a timeout
  or resource failure. Evidence: `local/wurth-parasolid-recheck/`.
- Isolated CDT replay locates the SMSI failure at the 1000-iteration collinear
  constraint guard, the representative wedge failure at a missing buddy edge,
  and the representative spherical `InvalidEdge` at a missing seed remapping.
  All-collinear seed selection retains default index 0 and overwrites a valid
  original index; the error is not an invalid STEP reference.
- Completed-case meshes are losslessly compressed as `mesh.stl.gz` to avoid
  exhausting the orb's disk. Logs, metrics, and result JSON are unchanged.
  Use `gzip -dc path/to/mesh.stl.gz > /tmp/model.stl` to inspect one. Reproduction
  commands still emit uncompressed STL. No input model is modified.

## Final outcome

**Foxtrot cannot currently process the entire Würth corpus successfully.**
The corrected sweep completed all **7,328** manifest entries, once each:

| Harness status | Files |
| --- | ---: |
| `ok` | 3,830 |
| `tessellation_error` | 2,831 |
| `invalid_mesh` | 665 |
| `crash` | 2 |
| Timeout / harness error / nonfinite or empty mesh | 0 |

All eight harness shards exit 1 because of model failures, not setup errors.
The merged report verifies identical worker hashes across shards, no duplicate
paths, and exact equality of the path/hash set with the original manifest.
The original, superseded sweep was stopped after 3,543 completed cases; it is
**not** a complete baseline. Its SIGINT was ignored, so SIGTERM stopped it after
the corrected full sweep finished. Comparing those 3,543 results against the
corrected worker finds **159 false passes corrected**, five `invalid_mesh`
statuses upgraded to `tessellation_error`, and **zero triangle-count changes**.

### Review artifacts

- [Per-file RCA CSV](.amp/in/artifacts/wurth/failures.csv): all **3,498 failed files**, with categories, failing face/surface IDs, precision counts, diagnostic links, and input hashes.
- [Detailed per-file RCA JSON](.amp/in/artifacts/wurth/failures.json): exact diagnostics and log line numbers, STEP surface types, cause explanations, metrics, and reproduction commands.
- [CDT investigation](.amp/in/artifacts/wurth/cdt-rca.md): representative UV dumps, source-level mechanisms, instrumented replay findings, and uncertainties.
- [Mesh investigation](.amp/in/artifacts/wurth/mesh-rca.md): initial precision investigation and concrete triangle coordinates. The full-corpus precision counts below supersede its explicitly marked snapshot.
- [Full harness report](local/wurth-final/report.md), [results](local/wurth-final/results.json), and [replay manifest](local/wurth-final/manifest.json).
- [Passing-model warnings](.amp/in/artifacts/wurth/passing-warnings.json): nine passing files warn about legacy `DESIGN_CONTEXT`/`MECHANICAL_CONTEXT` metadata. A tenth, `T_Wurth_WE-PLN-ER19.step`, drops three wire curves: `COMPOSITE_CURVE` references #112/#114/#116 use `.U.`, but `Logical` parsing accepts `.UNKNOWN.` instead; representation #68 (a geometric set of those curves plus placement) is also unsupported. Its solid mesh passes, but its wire content is not supported. No passing file logs a dropped face.

Generated reports, logs, corpus, and meshes remain in ignored orb directories;
this worklog and the accounting regression test are source changes. No models
are vendored, no geometry repair is claimed, and nothing is pushed.

### RCA coverage and mechanisms

Categories overlap: one file can have several distinct failures. Every counted
face error and caught panic reconciles with its log entry: **23,975 face errors
and 24 caught panics**. No failed file is left without an identified diagnostic
family. A family assignment does **not** establish the deepest geometric cause
for every member; limitations are recorded per file rather than guessed.

| Mechanism | Affected files | Finding |
| --- | ---: | --- |
| `CrossingFixedEdge` | 1,447 | Constraint insertion collides with a locked edge. A thin spline example has near-collinear UV noise; not proof of bad vendor topology. |
| `PointOnFixedEdge` | 1,262 | Collinear constraint splitting stalls. SMSI replay confirms the 1,000-step guard on a retraced, zero-area periodic UV contour, even without Steiner points. |
| `HalfEdgeInvariant` | 648 | Checked CDT topology invariant fails. Exact invariant and upstream trigger remain unresolved per file. |
| Unsupported surfaces | 588 | `get_surface` lacks `DEGENERATE_TOROIDAL_SURFACE`, `SURFACE_OF_LINEAR_EXTRUSION`, and `SURFACE_OF_REVOLUTION` implementations. |
| `WedgeEscape` | 207 | Constraint search leaves its expected triangle wedge. Representative planar replay confirms an absent buddy edge. |
| `InvalidEdge` | 65 | Representative spherical UV collapse triggers the constructor's missing-remapping guard through failed collinear seed selection. Other occurrences retain branch uncertainty. |
| `TooFewPoints` | 48 | Contour cannot supply three points to seed CDT; the investigated planar face has only two UV points. |
| Surface inversion failure | 20 | Spline point-to-UV Newton solve returns no solution: singular Jacobian outside its fallback or exhausted 256 iterations. Exact solver branch remains qualified. |
| Caught CDT panic | 18 | All 24 backtraces reach `half.rs:199`: flood erase indexes an invalid `next`/`prev` sentinel (4,294,967,295) before checking it. Upstream topology damage remains unresolved. |
| Missing surface reference | 1 | `L_Wurth_WE-RFI-0402.step`, face #707, explicitly references nonexistent surface #0. |
| Mislabeled Parasolid | 2 | `L_Wurth_WE-CMANC-M_7848031002.step` and `L_Wurth_WE-CMBNC-TypeM_7448031002.step` have identical hashes and Parasolid contents; STEP block splitting panics. |
| f32 STL precision collapse | 1,124 | Independently reprocessed triangles are nonzero in f64 but exactly zero-area after STL rounding. |
| Already degenerate in f64 | 286 | Independently reprocessed STL-degenerate triangles are already exactly zero-area before serialization. |

All **1,159** files with degenerate triangles (665 `invalid_mesh` plus 494 that
also have tessellation errors) were reprocessed by a focused f64/f32 probe.
Its triangle counts and STL-degenerate counts match the harness for every file.
Of **31,119** degenerate STL triangles, **7,578** are already degenerate in f64
and **23,541** collapse at f32 export. Mesh construction appends CDT triangles
without checking their 3D area, and STL export casts coordinates without an
area check. This establishes where exact degeneracy appears, not the precise
upstream face-generation defect for every triangle.

### Verification and replay

```sh
cargo test -p triangulate
# 10 unit tests + 1 integration test pass
python3 -m unittest discover -s scripts -p 'test_corpus*.py'
# 19 tests pass
git diff --check
# no whitespace errors

# Serial replay of all exact inputs with the corrected worker:
RUST_LOG=triangulate=debug,cdt=warn,step=warn python3 scripts/corpus.py \
  local/wurth/3dmodels --manifest local/wurth-final/manifest.json \
  --worker local/wurth-corrected-target/release/examples/corpus_worker \
  --jobs 1 --threads 1 --timeout 60 --output local/wurth-replay
```

The output directory must be new. For faster functional replay, run each of
`local/wurth-final/selection-0.json` through `selection-7.json` in its own
harness process. This is a functional assessment, not a benchmark or OCCT
equivalence test. A harness pass does not prove a watertight or faithful model.

## 2026-09-05 — Continue with the full KiCad corpus

- User requests all Würth files followed by all KiCad STEP files, without
  stopping at a partial run. Rechecked the Würth filesystem against the final
  manifest/results: **7,328 discovered = 7,328 completed**, exact path match.
- Cloned the official repository
  `https://gitlab.com/kicad/libraries/kicad-packages3D.git` into
  `local/kicad-packages3D`, revision
  `e62ed1fc7862da83f789bd562671b5e4b82afcdf`.
  `LICENSE.md` remains with the corpus (CC-BY-SA 4.0 with KiCad exception).
- Discovered **7,251 STEP files**, totaling **3,367,745,818 bytes**. No sampling,
  exclusions, filename deduplication, or extension-case filtering is used.
- Started all files through `scripts/corpus.py`, as eight disjoint manifest
  shards under `local/kicad-final/`, each with `--jobs 1 --threads 1 --timeout 60`.
  Logging and the corrected worker are identical to the Würth run; worker
  SHA-256 is `7e92685ff97e668a8d0c994c01c750027ca77645ee110d2786dd101bbb4de25c`.
- Completion requires a result for every discovered path/hash and per-file
  diagnostic assessment for every failure. Timing gates and the optional OCCT
  oracle are not enabled; acceptance remains the documented mesh checks.

## KiCad completion — every file processed and every failure rechecked

All **7,251 / 7,251** KiCad files completed. Each of the eight harness processes
exits 1 for model failures; there are no setup errors, crashes, timeouts, caught
panics, empty meshes, or nonfinite meshes.

| Corpus | Discovered and completed | Pass | Tessellation error | Invalid mesh | Crash |
| --- | ---: | ---: | ---: | ---: | ---: |
| Würth | 7,328 | 3,830 | 2,831 | 665 | 2 |
| KiCad packages3D | 7,251 | 6,169 | 1,060 | 22 | 0 |
| **Total** | **14,579** | **9,999** | **3,891** | **687** | **2** |

Both corpora have full path/hash manifests and a result for every discovered
STEP file. No sample, exclusions, or basename deduplication is used. Completion
means the assessment is complete, **not that every model converts correctly**.

### KiCad RCA artifacts

- [Per-file RCA CSV](.amp/in/artifacts/kicad/failures.csv): all **1,082 failed files**.
- [Detailed RCA JSON](.amp/in/artifacts/kicad/failures.json): face/surface IDs, exact diagnostic sites and explanations, source entity types, precision evidence, original and diagnostic logs, hashes, and reproduction commands.
- [Full harness report](local/kicad-final/report.md), [results](local/kicad-final/results.json), and [replay manifest](local/kicad-final/manifest.json).
- [Passing-model warning audit](.amp/in/artifacts/kicad/passing-warnings.json): 822 passing models have warnings. Of these, 821 have metadata parse warnings only; `Display.3dshapes/NHD-0420H1Z.step` also omits geometric curve sets. The mesh pass criterion does not establish complete wire-content support. No passing model logs a dropped face.

Every failure was reprocessed with an isolated, diagnostic-only CDT build.
For all **1,082** models, triangle/vertex/face/shell counts, errors, panics, and
warning/error log counts match the ordinary worker. The copy only adds
return-site diagnostics; it does not repair or replace geometry algorithms.
Diagnostic worker hashes and rerun provenance are retained under
`local/kicad-probes/`. Seventeen models received an additional rerun to locate
23 previously untagged optional-return failures.

All **9,977 face failures** reconcile with their diagnostic logs:
**5,093 CDT errors**, **4,789 unsupported-surface failures**, and **95
unsupported-curve failures**. Every CDT error now has its exact return site
recorded, rather than only an enum name.

| KiCad failure mechanism | Affected files (overlapping) | Confirmed finding |
| --- | ---: | --- |
| Unsupported surface | 563 | Missing `SURFACE_OF_LINEAR_EXTRUSION` and `SURFACE_OF_REVOLUTION` implementations: 2,845 and 1,944 omitted faces respectively. |
| Half-edge invariant | 320 | Missing/erased hull entries, stale hull edges, incomplete contour reconstruction, or inconsistent edge endpoints; the exact condition is recorded per face. |
| Crossing fixed edge | 320 | Constraint insertion encounters an already locked edge; each of the three traversal return sites is identified. |
| Collinear fixed-edge failure | 123 | Non-progressing split or 1,000-iteration guard; precise guard recorded per face. |
| Wedge escape | 30 | All 45 face failures reach the missing-buddy guard at `cdt/src/triangulate.rs:919`. |
| Unsupported curve | 16 | `curve()` lacks `HYPERBOLA` and `PARABOLA`: 87 and 8 failed boundaries respectively. |
| Too few points | 2 | All eight failing faces have exactly two projected points. |
| Invalid edge | 1 | Both failing faces reach the missing point-remapping guard at `cdt/src/triangulate.rs:312`. |
| STL precision collapse | 216 | Nonzero f64 facets become exactly degenerate after f32 serialization. |
| Already degenerate in f64 | 17 | Zero-area facets already exist in the in-memory mesh. |

The dominant invariant sites are `hull.rs:205` (point is not mapped to a hull
entry), `hull.rs:208` (hull links are erased), and `triangulate.rs:561` (insertion
selects an erased edge). All 242 occurrences of the latter explicitly confirm
empty `next` and `prev`, not duplicate endpoints or an existing buddy. These
checks identify the immediate failure, but not the exact earlier mutation or
geometric trigger for every model; those deeper causes remain qualified.

All **222** models containing degenerate STL triangles were independently
checked in f64 and after f32 rounding (22 `invalid_mesh`, 200 also containing
tessellation failures). Probe triangle counts and degenerate counts match the
harness in every case. Of **4,086** degenerate STL triangles, **154** were
already degenerate in f64 and **3,932** collapsed at export.

The expanded instrumentation also resolves the immediate invariant in the
earlier Würth representative `J_Wurth_WR-BTB_658105303064.step`, surface #55272:
it selects an erased hull edge at `cdt/src/triangulate.rs:561`. The new evidence
is `local/wurth-probes/half-complete.log`; the upstream mutation remains unknown.

### Final coverage verification and replay

```sh
# Replay every exact KiCad input through the ordinary corrected worker:
RUST_LOG=triangulate=debug,cdt=warn,step=warn python3 scripts/corpus.py \
  local/kicad-packages3D --manifest local/kicad-final/manifest.json \
  --worker local/wurth-corrected-target/release/examples/corpus_worker \
  --jobs 1 --threads 1 --timeout 60 --output local/kicad-replay
```

As for Würth, the eight `selection-N.json` manifests can instead run in separate
harness processes for faster functional replay. Output paths must be new.
Final checks compare filesystem discovery, manifest path/hashes, and result
path/hashes for both corpora, verify the 3,498 + 1,082 per-file RCA entries,
and validate every diagnostic link. Temporary scripts and copied diagnostic
source are removed after use; original inputs, run reports, meshes, and evidence
remain. No further production source changes are needed for the KiCad audit.

Final verification succeeds: **7,328/7,328 Würth** and **7,251/7,251 KiCad**,
all input hashes rechecked, **4,580** RCA rows with valid diagnostic links, and
all **5,093** KiCad CDT failures tied to exact return sites. The machine-readable
[coverage record](.amp/in/artifacts/corpus-coverage.json) retains these checks.
`git diff --check` reports no whitespace errors. Temporary probe source and
orchestration scripts are removed; no shared state is changed or code pushed.

## Fundamental repair phase — 2026-09-05 (in progress)

The user now requests repairs, separate local commits for each logical change,
and full verification. The assessment above remains the immutable baseline;
this section does **not** claim that all failures are fixed.

Implemented and committed locally:

- `18a48c3`: count failed faces in both shell types (10 unit tests and the
  checked-in-model integration test pass; 19 harness tests pass).
- `337b396`: preserve the complete baseline worklog.
- `3dc446d`: parse ISO logical unknown as `.U.`, not `.UNKNOWN.`.
- `2cfbfb9`: fallible STEP lexical/structural parsing. Preserve whitespace and
  delimiters inside literals, handle doubled apostrophes and comments, reject
  missing sections/unterminated records instead of panicking. Callers propagate
  errors; out-of-range typed entity lookup returns `None`. Seven STEP tests
  pass, and the supported workspace tests pass. Borrowed strings retain escaped
  apostrophes, and legacy non-ASCII byte replacement remains a limitation.
  Both Parasolid inputs now return a parse error without panic; they are still
  invalid STEP inputs, not successful geometry conversions.
- `50be439`: exact hyperbola and parabola edges with endpoint-derived direction
  and adaptive tangent-angle sampling. All 16 affected KiCad models replayed:
  all 95 unsupported-conic errors eliminated; five unrelated face errors remain
  (one crossing constraint and four revolutions). Thirteen models have no face
  errors at this checkpoint; `/dev/null` STL replay is not mesh verification.
- `5ef65d8`: use exact orientation for CDT seed selection and reject a missing
  noncollinear seed instead of reusing index zero and corrupting the remap.
- `8d55632`: sort radial sweep distances around the actual seed center, not the
  old bounding-box center. Remove the invalid seed comparator and its repair
  branches by keeping seed indices outside the sort. Fourteen CDT unit tests
  and two documentation tests pass.
- `0d0fa20`, `ebcab4b`: represent linear extrusions and revolutions as exact
  homogeneous tensor-product NURBS, reusing existing surface evaluation rather
  than adding projection/normal special cases. Constructor point/normal tests
  pass. All 79 affected Würth and 563 affected KiCad models replayed through a
  dedicated worker: all 414 + 187 and 2,845 + 1,944 unsupported swept-surface
  errors eliminated. Other tessellation errors remain; meshes were not checked
  in this fast support-only replay.

The conic equations follow ISO 10303-42 geometry_schema §§4.5.28–29 as published
by STEP Tools. Sweeps use the homogeneous affine extrusion and exact rational
quadratic circle product, not fitted geometry or per-model substitutions.

The seven-model strict CDT checkpoint is `local/repair-cdt-order/report.md`.
One previously failed model (`97730256332R`) becomes valid under the existing
STL checks; others still fail, and `79527141` increases from two to three face
errors. This is evidence that sorting alone is not a complete CDT fix.

OCCT is installed in `local/occt-venv` for independent checks. The first check
of `97730256332R` is **inconclusive**, not a pass:
`local/repair-order-occt/report.md` reports `oracle_invalid_mesh` because OCCT's
STL contains one degenerate facet. Foxtrot has no degenerate facets in this run,
but its surface area is 51.2314 versus OCCT's 50.3344 and signed volume is
19.1981 versus 19.4880. These discrepancies still need investigation. No mesh
validation rules or oracle failures have been suppressed.

Next ownership issue identified in CDT: constraint walking assumes every
intermediate collinear vertex belongs to the exterior hull. Interior vertices
do not have hull slots; the walk must follow incident triangle edges instead.
Zero-area surface charts, precision collapse, and malformed geometry references
also remain open. Full repaired-corpus verification is still pending.

### Second repair checkpoint (in progress)

The bespoke radial-sweep CDT lost information in its hull ordering and required
multiple interacting repair paths. It is replaced by Spade 2.15.1 constrained
Delaunay triangulation, preserving original vertex provenance and even/odd trim
parity. Crossing constraints are rejected rather than silently split by CDT.
The obsolete Steiner-point-dropping retry is removed. Empty per-face output now
counts as a failed STEP face, even when parity cancellation is valid standalone
CDT input. Flat face traversal replaces per-face allocations and searches.

Other separate logical commits correct revolution parameter orientation, close
rational circle control nets exactly, and represent both spindle-torus branches
through signed major radii. Spherical charts now choose an exterior projection
pole from oriented boundary clearance, with signed solid-angle verification;
hemispheres, large caps, bands, and reversed face sense have regression coverage.
Sphere normalization is independent of length units (radii 1e-9, 1, and 1e9).
The supported workspace tests pass after these changes.

The first full repair checkpoint exhausted disk space while retaining failure
meshes. `local/wurth-repair-pass1` is incomplete and is NOT coverage evidence;
KiCad did not start in that checkpoint. Completed meshes are losslessly gzip
compressed, preserving logs and metrics. No baseline results are overwritten.

A new full run uses frozen `local/repair-pass2-worker`, first all 7,328 Würth
inputs, then all 7,251 KiCad inputs. Eight disjoint manifest shards each use one
worker and one Rayon thread. The supervisor compresses retained meshes only
after atomic per-case results confirm validation is finished. Final merging
checks complete path/hash coverage and an identical worker hash across shards.
Results will be in `local/{wurth,kicad}-repair-pass2`. These runs are pending,
not yet proof that the remaining geometry and precision issues are resolved.

### Complete second checkpoint and inverse-projection repair

Pass 2 finished with exact manifest/hash coverage, Würth before KiCad:

| Corpus | Completed | ok | tessellation_error | invalid_mesh | worker error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Würth | 7,328 | 4,655 | 1,852 | 819 | 2 |
| KiCad | 7,251 | 6,403 | 550 | 298 | 0 |

The two worker errors are the already identified non-STEP Parasolid inputs
(the harness calls nonzero worker exit `crash`; these are returned parse errors,
not Rust panics). All retained failure meshes have been validated before gzip
compression. These totals show progress, NOT completion of the repair task.
In particular, newly supported surfaces and stricter empty-face accounting
expose remaining failures rather than hiding them.

The strict seven-model Spade checkpoint passes four models. The sphere case
242117113 and old CDT panic case 649008221732 now pass; 97730256332R fails because
its geometry #250 projects to a retraced, empty region. The former apparent
success omitted this face. This is why empty-face validation must remain.

Separate subsequent changes normalize arbitrary periodic Newton steps, reject
undefined explicit STEP references (including #707 -> #0 in WE-RFI-0402), and
correct the spherical hole regression's winding. Reference validation uses the
existing dense entity table, distinguishes `$` from explicit #0, and checks
unknown records while ignoring quoted literal hashes.

The inverse solver now uses derivative-scaled, damped Gauss–Newton with a
line search, replacing second-derivative Hessian inversion and its incompatible
absolute-distance/small-step/singular-inverse acceptance rules. Convergence is
componentwise first-order stationarity, including active domain boundaries.
Periodic steps retain their actual travel direction across stored UV seams.
The objective comparison accounts for floating-point evaluation uncertainty;
this is not an exact-arithmetic monotonicity certificate. Ten NURBS tests cover
curved and periodic surfaces, normal offsets, boundaries, length/domain scales,
singular derivatives, and a resolvable 1e-16-thick patch. Supported workspace
tests and all 19 Python harness tests pass.

The strict seven-model inverse-projection checkpoint passes five models:
97730256332R now includes all nine faces without tessellation errors or degenerate
STL facets. 79527141 still has one face error and 615032243321 still has eight.
Results: `local/repair-cdt-projection`. Independent OCCT BRep area/volume checks
are being recorded separately, without relying on a possibly degenerate oracle
STL. A third complete Würth-then-KiCad run is underway with frozen
`local/repair-pass3-worker`; its added `degenerate_f64` metric distinguishes
in-memory defects from binary-STL quantization.

### Complete third checkpoint and chart ownership repairs

Pass 3 completed every manifest entry, Würth before KiCad, with identical frozen
worker hashes across shards and no timeout or harness error:

| Corpus | Completed | ok | tessellation_error | invalid_mesh | worker error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Würth | 7,328 | 4,631 | 1,968 | 726 | 3 |
| KiCad | 7,251 | 6,465 | 487 | 299 | 0 |

The third returned worker error is the missing-reference WE-RFI-0402 input.
In-memory zero-area facets total 9,553 in 362 Würth files and two in one KiCad
file. This checkpoint is not all-green and predates the chart repairs below.
Reports are `local/{wurth,kicad}-repair-pass3`. To make space for continued
verification, 2,334 generated scratch meshes from the obsolete, interrupted
pass 1 are removed (4,728,717,556 bytes). Its logs, metrics, manifests and frozen
worker remain. Complete baseline and pass 2/3 results are not deleted.

Independent OCCT BRep checks use surface/volume integration, not an oracle STL.
All seven reference BReps are valid. Results are retained in
`.amp/in/artifacts/repair-occt-properties.json`. They reveal defects invisible
to the coarse mesh-validity harness: SMSI surface area was 2.216% too large,
and D-SUB signed volume was 7.31% too large despite near-matching total area.

Confirmed chart defect: `straighten_periodic_runs` linearly redistributed
nonuniform intrinsic parameters without moving the corresponding 3D boundary
vertices. Removing it deletes 80 production lines and restores the invariant
that unwrapping changes parameters only by whole periods. On SMSI geometry
#195, measured area falls from 6.21101 to 4.97683 versus OCCT 4.98665; geometry
#222 falls from 1.59864 to 1.25651 versus OCCT 1.26267. This is a geometry fix,
not a validation adjustment.

A separate representation-only refactor lifts polynomial spline controls to
homogeneous coordinates and removes the duplicate spline Surface variant.
The next change gives singly-periodic splines one continuous polar chart:
cylinder-like surfaces map to annuli, and one collapsed radial end maps to
the origin. Both collapsed ends and doubly-periodic surfaces still use their
existing Cartesian handling. A shared chart owns lowering, raising, normal and
Steiner conversions. Boundary collapse uses endpoint basis functions, including
nonclamped knots, and compares represented Cartesian controls without a
geometric epsilon. Roundtrip domain checks account for floating arithmetic.
Tests cover both periodic axes, both pole ends, full conical disks and cylindrical
bands, positive orientation, area, and absence of f64-degenerate triangles.

All seven targeted files pass the strict harness in `local/repair-cdt-polar`.
This is NOT geometry equivalence: SMSI area is now 50.12845 versus OCCT 50.34325,
but WR-MJ 615032243321 area increases to 49,071.5 versus OCCT 31,526.7. That
discrepancy remains under active investigation rather than being hidden by the
successful worker status.

Another proven numerical defect: plane #422 in antenna 7488918022 produces a
UV coordinate -3.941e-46 through affine inversion, outside Spade's safe exponent
range, although all input coordinates are finite. Planes now copy the two world
coordinates orthogonal to the largest normal component, with orientation
preserved and distortion bounded by sqrt(3). This avoids creating rounding-only
coordinates and preserves coordinate predicates. The eight-model checkpoint
`local/repair-plane-check` passes every file. Workspace tests pass (25 triangulate
tests plus integration); broad verification of these latest repairs is pending.

The knot-span sampling experiment is not accepted: even after correcting seam
coordinates, WR-MJ area is 40,681.58 versus OCCT 31,526.66. Its temporary sampler
and face-area instrumentation are removed. Frozen experimental workers and
measurements remain under `local/`. The experiment independently exposes an
exact seam defect: evaluating sin(2π) gives a different chart coordinate from
sin(0), creating 30,792 f64-degenerate facets. Reducing the intrinsic parameter
modulo its period before trigonometric evaluation eliminates all of those
facets (456,154 triangles, zero worker errors/panics/f64 degenerates), without
snapping coordinates or deleting triangles. This seam normalization is retained
as a separate fix; 25 triangulate unit tests pass, including exact seam equality
for both periodic axes, both radial directions and positive/negative periods.

Full pass 4 uses the frozen plane-fix worker, not experimental source. It runs
every Würth input first, then every KiCad input, with hash-checked coverage.

Pass 4 completes all 7,328 Würth files (5,563 ok, 987 tessellation errors,
775 invalid meshes, three parser-return failures) and all 7,251 KiCad files
(6,458 ok, 496 tessellation errors, 297 invalid meshes). No timeouts or harness
errors occur. Full per-file qualified RCA exports are in
`.amp/in/artifacts/repair-pass4/{wurth,kicad}/`; coverage and hashes are checked
against the complete reports. Early face projection failures are distinguished
from CDT failures and parser exits. Mesh validity still does not imply accuracy.

WR-FPC 68610414422 surface #10973 concretely reproduces the remaining exponent
problem on a spline: finite projected x=1.7655773007981676e-46, y=3..11, is below
Spade's 2^-142 minimum. The CDT now applies one exact positive power-of-two
similarity to fit all nonzero components into Spade's accepted range. It does
not translate, weld, discard or independently rescale coordinates. Inputs whose
dynamic range cannot fit remain errors. Public point queries retain input units.
Seven CDT unit tests and two doctests pass, including identical indexed topology
from smallest-subnormal to 2^1000 scales and preservation of tiny components.
The FPC representative now strictly passes: 185,250 triangles, zero face errors,
panics or f64-degenerate facets. All 547 Würth and 90 KiCad pass-4 InvalidInput
files are selected for targeted rerun; other failure classes remain actionable.

Disk maintenance removes 3,519 obsolete generated pass-2 compressed scratch
meshes (4,727,878,200 bytes), preserving its reports/logs/metrics/manifests and
frozen worker. Baseline and newer complete pass-3/pass-4 evidence remain.

Curve projection has the same independent fixed-distance defect previously
removed from surface projection: its signed cosine test accepts antiparallel
residuals, and the 0.01 world-unit tolerance can return the nearest cached
parameter without solving a short edge at all. Replace its heuristic extra
iteration/stall state with derivative-scaled, line-search Gauss--Newton and
first-order stationarity. Failure now propagates rather than returning an
unconverged best guess. The scalar solver uses the same descent/roundoff contract
as surface inversion and removes 58 production lines from the solver.
Regression tests resolve distinct nearby parameters across 1e-12..1e6 geometry
scales, with normal offsets, endpoint minima, curved projections and different
knot units. NURBS/triangulate tests pass; all eight strict model checkpoints pass
in `local/repair-curve-check`. This does not fix WR-MJ surface undersampling or
WR-TBL's 112 floating-cross-zero facets; those are explicitly still reproduced.

The complete targeted exponent rerun finishes: all 547 Würth and 90 KiCad
InvalidInput files lose that diagnostic. Würth outcomes are 407 ok, 108 invalid
meshes, 32 other tessellation errors; KiCad outcomes are 50 ok, 17 invalid meshes,
23 other tessellation errors. These are selected subsets, not a new full sweep.
Reports: `local/{wurth,kicad}-scale-check`. Retained meshes are compressed after
validation. Separate A/B reproduction of RPSMA 63012242121508 establishes that
the curve solver change resolves two omitted faces: scale-only worker has two
errors, curve-fixed worker has zero (232 faces, 115,316 triangles).

Degree-one spline spans now emit their trim endpoints and intervening knots,
not eight redundant points along each straight segment. Rational degree-one
spans are also straight; changing parameter speed does not require curvature
samples. A regression keeps a piecewise-linear corner and checks reversed trims.
The eight strict checkpoints pass in `local/repair-linear-check`. This reduction
alone does not resolve the planar spline chart defect: WR-TBL's floating-cross-
zero count changes from 112 to 224 as the CDT changes. The next independently
verified geometry reduction addresses that root cause rather than keeping
redundant samples to accidentally influence the triangulator.

Concrete f64-degeneracy RCA: WR-TBL 691313710008 contains 16 affected planar
bilinear spline faces. On surface #1089, boundary inversion returns nominally
zero u as values ranging from -1.3e-15 to +2.14e-13. Curvature samples at u=0
then form positive-area chart slivers whose raised world points all have
x=18 and y=3.665: they are exactly collinear. The diagnostic worker and logs
are retained as `local/repair-degen-probe-worker`, `local/tbl-degen.log` and
`local/tbl-uv.log`; temporary production instrumentation is removed.

Convex, constant-weight bilinear patches now reduce to the existing Plane
representation when exact robust predicates establish coplanarity and strict
convexity. This bypasses both iterative inversion and unnecessary curvature
sampling. One-ulp warped, folded, variable-weight and unsafe-predicate-range
inputs do not take this reduction. Tests cover rotated planes, homogeneous
weight scaling, one-ulp rejection, and a trimmed world-coordinate rectangle.
All eight checkpoints pass in `local/repair-bilinear-check`. WR-TBL now has
53,396 triangles (down from 228,454), zero face errors/panics, zero f64 or f32
degenerates, and passes the unchanged strict mesh validator. A full sweep of
the combined fixes is still required; known OCCT discrepancies remain open.

Pass 5 completes every file: Würth 6,816 ok, 392 tessellation errors, 117 invalid
meshes, three parser-return failures; KiCad 6,507 ok, 423 tessellation errors,
321 invalid meshes. No InvalidInput diagnostics remain. Full reports are under
`local/{wurth,kicad}-repair-pass5`; qualified per-file RCA exports under
`.amp/in/artifacts/repair-pass5/{wurth,kicad}/` cover all 512/744 non-ok files
with hash-checked coverage. Obsolete generated pass-3 compressed meshes are
removed (3,480 files, 4,846,619,593 bytes), preserving their reports, diagnostics,
metrics, manifests and frozen worker. Baseline and complete pass-4/5 meshes remain.

D-SUB 242117113's independent volume defect is now isolated to torus chart
handedness. Switching the angular/radial roles from major-polar to minor-polar
reverses orientation; lowering previously omitted the corresponding reflection.
A differential-orientation assertion fails on the old minor-polar chart while
the major-polar chart passes. Reflecting its second coordinate, consistently
in lowering and raising, fixes the assertion and all eight strict checkpoints.
Surface area stays 6,512.680575. Signed volume changes from 2,526.741065 to
2,350.363572 versus OCCT 2,354.698870 (0.184% low rather than 7.31% high).
The binary mesh has zero directed-edge imbalance, down from 2,704 unbalanced
edges; its summed area normal is within 6e-13 of zero. This verifies winding
closure independently of the unchanged mesh-validity harness. The fix is not
in the frozen pass-5 worker. WR-MJ's large area discrepancy remains unresolved.

Curve #5447 in WR-TBL 691337500008 exposes a separate representability limit:
its parameter interval is approximately [-41.36699,-41.31699], so adjacent
parameters differ by 7.1054e-15. After one valid step the remaining tangential
residual is 2.8935e-15, and the full proposed step rounds back to the current
parameter. The solver now recognizes that discrete limit before line search.
It deliberately does NOT accept a backtracked no-op as convergence. A regression
fails before the fix and verifies that the returned parameter is better than
both adjacent representable parameters. Of the 20 selected Würth curve-failure
files, 17 now strictly pass, one has only mesh degeneracy, and two retain other
projection nonconvergence. Both selected KiCad files remain nonconvergent and
require a separate RCA. Reports: `local/{wurth,kicad}-quantization-check`.

The remaining FLYLT EP7 projection cycle is traced below the optimizer. All
seven curve controls have exactly the same z=8.58999999999999, but direct sums
of derivative basis coefficients times translated controls produce a nonzero
z derivative. Its normal-offset target amplifies this false tangential signal.
Curve and tensor-product surface evaluation now use differences from one local
control, adding that control back only to the position. Partition of unity
makes this algebraically identical while preserving constant coordinates and
their zero derivatives exactly, without tolerances or extra optimizer states.
Both constant-coordinate regression tests fail before the fix and pass after.
Workspace tests and all eight strict checkpoints pass (`local/repair-partition-check`).
FLYLT now processes all 518 faces with zero errors/panics/f64 degenerates;
`local/flylt-partition.log` has no projection nonconvergence. Other remaining
curves still need separate positive-curvature/knot-side treatment.

The remaining positive-curvature cases require the actual squared-distance
Hessian: in the BMS 74942302 cubic and two KiCad Bourns degree-five curves it
is 87–600 times the Gauss–Newton approximation. Scalar inversion now uses
positive finite distance curvature, retaining Gauss–Newton only when curvature
does not define descent. Steps remain line-searched and are bounded to one
parameter domain. The BMS regression fails before this change and passes after;
all 930 BMS faces process without errors, panics or f64 degenerates. Neither
KiCad regression retains curve-projection diagnostics (the L34.3 model still
has invalid triangles; L19.3 has independent face triangulation errors).
Thus no knot-side special case is currently justified by these files.
`cargo test -p nurbs -p triangulate` and all eight strict checkpoint files pass.
Evidence: `local/repair-newton-check`, `local/kicad-newton-check`, and
`local/bms-newton-metrics.json`; frozen worker `local/repair-newton-worker`.

Before using representation uncertainty for polar topology, consolidate the
length-unit resolver so uncertainty can be converted to each representation's
native coordinates. The old resolver only handled SI units, while output-scale
detection separately guessed conversion units from labels and defaulted missing
base units to metres. The shared resolver now follows declared conversion-factor
chains, rejects cycles/non-length bases, and covers all STEP SI prefixes.
Structured scale detection delegates to it, removing more code than is added.
Regression coverage includes arbitrarily named inch/foot chains, a misleading
label, decimetres, angle units and a cycle. Workspace tests and all eight strict
checkpoints pass (`local/repair-units-check`). Legacy fallback detection for files
without parsed unit contexts is unchanged; this is not a claim of per-instance
unit scaling for mixed-unit assemblies.

Polar pole classification now consumes the owning representation's declared
length uncertainty in native coordinates. Shape data carries it through shell,
face and surface construction; absent precision remains zero, and shared shapes
use the strictest context. A positive-weight rational boundary is bounded by its
Cartesian control hull, with hypot-based distances that do not underflow on tiny
features. There is no global epsilon or changed geometric control point. Tests
cover distinct mm/metre contexts, zero precision, exact nonclamped boundaries,
scales from 1e-200 to 1e200, near-axis revolution poles and holes above precision.
Workspace tests and eight strict checkpoints pass.

BatteryClip_Keystone_54_D16-19mm now processes all 143 faces, 4,572 triangles,
without errors/panics/f64 or f32 degenerates (`local/battery-uncertainty-check`).
Its area is 1,322.201946 and volume 217.099895; OCCT's exported mesh reports
1,323.608742 and 217.882307 (0.106%/0.359% differences), but OCCT itself emits
four degenerate f32 triangles, so the oracle harness correctly reports
`oracle_invalid_mesh`, not a passing equivalence check.

Rechecked every pass-5 failure, Würth first then KiCad, with eight disjoint shards
and exact path/hash coverage (`local/{wurth,kicad}-uncertainty-check`). Of 512
Würth inputs: 187 now pass, 195 have face errors, 127 invalid meshes and three
unchanged source-format/reference failures. Of 744 KiCad inputs: 17 now pass,
405 have face errors and 322 invalid meshes. These are selected-failure results,
not new full-corpus totals. Remaining CrossingFixedEdge diagnostics occur in
91 Würth/405 KiCad files. The smallest KiCad example is a five-face test-point
bridge whose torus seam uses the same EDGE_CURVE forward and backward. The
current implementation resamples it separately in opposite parameter directions;
the next investigation is exact consistency of those two discretizations.

Confirmed: the forward/reversed samples of the same test-point circular edge
are not identical. A regression for trimmed and closed edges, both same_sense
values, fails before the fix. EDGE_CURVE now owns a canonical discretization;
ORIENTED_EDGE only reverses its point order. This removes orientation from curve
construction and eliminates direction-dependent floating-point seam slivers
without snapping or special-case curve types. Workspace tests and the eight
strict checkpoints pass. TestPoint_Bridge_Pitch2.54mm_Drill1.0mm now meshes all
five faces with 334 triangles and no errors or degenerates. Its independent
OCCT comparison still fails: the torus interior uses only two radial sample
rings even over a half revolution, missing the bridge apex (z=1.84625 versus
the analytic 2.07). That under-resolution requires a separate meshing change.
Evidence: `local/testpoint-canonical-{check,occt}`, `local/repair-canonical-check`.
All 512/744 pass-5 failures are being rechecked with `local/repair-canonical-worker`.

Canonical-edge selected sweep completes: Würth remains 187 ok / 195 face errors /
127 invalid meshes / 3 source failures; KiCad improves to 384 ok / 244 face errors /
116 invalid meshes. Hash-complete reports: `local/{wurth,kicad}-canonical-check`.

Increasing torus radial density alone exposes another chart defect: bridge area
gets worse, 35.13314 to 44.23667 rather than OCCT 28.11727. Chart dumps show boundary
radii up to 8.379645 instead of 4.389823: a half revolution is spuriously expanded
to a full revolution. `rem_euclid` rounds a tiny negative angle to exactly 2π;
the interval finder then normalizes that selected endpoint a second time to 0,
while point lowering keeps 2π. The fix copies the selected endpoint representative
from the sorted data instead of normalizing it again. A regression fails before
and passes after; workspace and eight checkpoints pass. Without any density
change, bridge area becomes 27.10426 and the coarse OCCT gate passes. Apex/volume
still expose the independent two-ring resolution defect. Evidence:
`local/testpoint-arc-occt`, `local/repair-arc-check`, frozen `local/repair-arc-worker`.

Source-level examination of all 50 TooFewPoints files (101 faces) is retained in
`local/two-point-face-rca.json`: 45 files / 88 faces have two opposing LINE edges
with only two shared vertices; two KiCad files / eight faces have short curved
spline trims; three Würth files / five toroidal faces use VERTEX_LOOP bounds.
The latter two groups need meshing investigation, not blanket rejection. OCCT
accepts representative whole shapes from all three groups, but that does not
prove raw STEP conformance: ISO 10303-42:2021 §5.5.20 IP2 explicitly requires
nonzero face_surface extent, and §5.5.1 IP1 forbids overlapping distinct edge
domains. Do not manufacture triangles for provably collapsed planar line loops.
Authoritative text read in full:
https://www.steptools.com/stds/smrl/data/resource_docs/geometric_and_topological_representation/sys/5_schema.htm

With the interval defect fixed, torus radial sampling now resolves the second
intrinsic angle at the same 32-segments-per-revolution rate as the polar angle,
instead of always using two rings. The dimensionless rule works for either
chart orientation and across length units. Its angular-gap regression fails
before the change; workspace and eight strict checkpoints pass after it.
The test-point bridge now has 1,296 triangles, no degenerates, area 28.098084
versus OCCT 28.117268 (0.068% difference), volume 5.375025 versus 5.420478
(0.839%), and apex z=2.062299 versus analytic 2.07. This is an actual geometric
improvement, not just a successful process return. Evidence:
`local/testpoint-torus-density-arc-occt`, `local/repair-torus-density-arc-check`.
For the next full sweep, removed only superseded generated pass-4 `.stl.gz`
meshes (2,555 files, 4,451,650,165 bytes); reports, source hashes, logs, metrics,
manifests and frozen workers are preserved, as are baseline/pass-5 meshes.

Short spline trims now sample the intersection of each knot span with the trim,
rather than filtering a grid over the untrimmed span. A curved interval shorter
than one grid cell previously became a straight chord with no interior samples.
The new regression fails before this change (2 points instead of 9); workspace
tests and all eight checkpoints pass afterward. All eight TooFewPoints faces
in the KiCad L39.4/L41.9 Bourns models now triangulate; those files retain separate
crossing-constraint defects. Evidence: `local/kicad-trim-check` and
`local/repair-trim-check`. The full pass-6 sweep uses an earlier frozen worker
and deliberately does not contain this change.

### Pass 6 reconciliation and endpoint evaluation repair

Full pass 6 covers all 7,328 Würth files (7,000 ok, 198 tessellation errors,
127 invalid meshes, 3 source parse/reference crashes), then all 7,251 KiCad
files (6,901 ok, 233 tessellation errors, 117 invalid meshes). Exact path/hash
coverage and all eight shards reconcile for each corpus. Per-file evidence
and regression tables are in `.amp/in/artifacts/repair-pass6/`. Comparing to
pass 5 reveals 13 Würth and 9 KiCad formerly-ok regressions; improved aggregate
counts do not establish regression freedom. These remain tracked individually.

Confirmed an evaluator defect independently of the inverse solver: using the
first active control as a translation anchor erases small endpoint coordinates
when subtracting/re-adding a distant control. Curve and surface regressions
return zero instead of 1e-30 before the repair. Anchor selection now follows
the largest basis coefficient (the product of the largest axis coefficients
on surfaces), consistently for positions and derivatives. This preserves exact
endpoint interpolation and constant-coordinate derivatives without special
endpoint branches or relaxed convergence tolerances. Both new regressions and
`cargo test --workspace` pass; all eight strict checkpoint files pass.
The Würth WL-SMCW-0603dome, previously failing surface inversion, now passes
the strict harness. Evidence: `local/repair-anchor-led`,
`local/repair-anchor-check`; frozen worker `local/repair-anchor-worker`.
The complete pass-6 non-ok cohort is being rerun, Würth first then KiCad;
this repair is not yet certified against every previously passing file.
