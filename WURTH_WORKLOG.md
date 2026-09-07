# Würth and KiCad STEP repair worklog

## Oracle cohort 2 — 2026-09-07

### Apply the OCCT-supported corpus policy

The user explicitly excludes models OCCT rejects or cannot fully tessellate
from the repair corpus. Add `BRepCheck_Analyzer(shape).IsValid()` immediately
after STEP transfer, before reference meshing. Preserve oracle errors as errors;
exclude their inputs through the active replay manifest, never turn them into
native passes. Tool-installation failures and resource timeouts alone are not
evidence that a model is unsupported. This supersedes the earlier requirement
to investigate native discrepancies against incomplete references.

Recheck all 17 current sources sequentially, verifying their SHA-256s. Reuse
the retained completeness diagnostics rather than retessellating references.
The eligibility audit takes 20.81 s and peaks at 440,164 KiB RSS. Eight inputs
leave the active cohort:

- Invalid transferred OCCT shape: Bourns3299X, CMB-XS744821120, RSTV471006268143.
- Incomplete OCCT tessellation: Murata1400, RSTV471NS03268640,
  RSTV471NS04268540, LANMX749600000.
- Previously proved original STEP schema violation: CMBNC7448052502. Its
  transferred OCCT shape passes validity; do not misreport it as OCCT-invalid.

CMB-XS previously passed the numerical oracle and is excluded consistently too.
No original STEP files or historical reports are deleted or rewritten.
`local/cohort2-eligibility/eligibility.json` records each decision, hash and OCCT
package version; `excluded-manifest.json` preserves the eight excluded inputs.
The active `local/cohort2-eligibility/manifest.json` contains nine inputs:
five existing passes (PD3-TypeL, HCFT, all three FI parts) and four remaining
mismatches (both AIG8 capacitors, TBL691308330002, CHK EI48). All nine have valid
OCCT shapes and retained complete references. No active native processing
failure remains; the four geometric mismatches still require investigation.
Use this active manifest for subsequent cohort replays instead of the frozen
17-case baseline manifest. The baseline remains available for before views.

`PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
local/occt-venv/bin/python -m unittest scripts/test_corpus_geometry.py
scripts/test_corpus.py`: 28 tests pass. The new regression uses a genuine
self-intersecting face and OCCT's real analyzer, and verifies rejection before
meshing or STL output. The existing valid-box/completeness test still passes.
Review export: `.amp/in/artifacts/cohort2-eligibility.json`.

### Reject the CMBNC strip experiment; confirm a source schema violation

A face-coverage-selected strip cutter clears both CMBNC processing errors, but
the new complete OCCT reference rejects its geometry: 0.8300 mm forward and
0.7482 mm reverse sampled error. The original partial mesh has 0.0137 mm forward
and 1.6910 mm reverse error. Added geometry is not necessarily correct geometry.
Reject the prototype, including its synthetic test, despite 148 passing library
tests and complete native processing. The first synthetic assertion mistakenly
tested spatial accuracy before interior refinement; its corrected assertion
checks exact parameter-domain coverage. Neither substitutes for the real-part
oracle. Production Rust sources are restored, not committed as a partial fix.
Retain `local/cohort2-strip-prototype.patch`, `local/cohort2-strip-worker`,
`local/cohort2-strip-cmbnc/`, and `local/cohort2-cmbnc-partial-comparison.json`.

The original CMBNC source has two FACE_OUTER_BOUNDs on each failing face:
#3039 references #6922 and #6923; #3187 references #7215 and #7216. Both pairs
are explicitly FACE_OUTER_BOUND entities in the input, not inferred OCCT types.
[ISO 10303-42:2021 topology_schema, face.WR2](https://www.steptools.com/stds/smrl/data/resource_docs/geometric_and_topological_representation/sys/5_schema.htm)
requires `SIZEOF(QUERY(temp <* bounds | 'TOPOLOGY_SCHEMA.FACE_OUTER_BOUND' IN TYPEOF(temp))) <= 1`.
This proves source nonconformance independently of healing, source/surface gaps,
or native failure. Per the user's invalid-input policy, do not add recovery for
this input. This schema violation does not itself explain every numerical
discrepancy: many passing FI faces also violate it. It does not discredit the
general chart-conditioning fix or justify treating other unresolved inputs as
invalid. The broader doubly-periodic trimming limitation remains unsupported,
not claimed fixed by this rejected experiment.

### Resource checkpoint and complete 17-case replay

The bounded-oracle replay finishes all 17 cases with one job and one native
thread: six pass, four oracle mismatches, two native processing failures and
five incomplete-reference errors. All three FI models now pass. The three
previous reference timeouts (PD3-TypeL, CMB-XS, HCFT) finish and pass. No omitted
reference face is reclassified as an invalid source or a native fix.
Evidence: `local/cohort2-bounded-oracle/{results.json,report.md}`. Wall time is
615.38 s, peak child RSS 3,053,132 KiB; this includes OCCT reference generation,
not just the bounded comparison. After completion, system used memory is about
1.2 GiB and available memory about 30 GiB.

OCCT's internal parallel mesher bypassed the harness's native thread setting.
Disable that nested meshing pool; corpus jobs own concurrency. Serial HCFT
reference generation takes 286 s and 1,574,528 KiB peak RSS, and `cmp` confirms
the 4,354,716-triangle STL is byte-identical to the retained parallel reference.
This intentionally trades wall time for predictable single-job resource use:
the earlier parallel conversion plus comparison took 183.6 s. It is not a
measurement of parallel conversion alone or a claim of a universal memory cap.
The 27 Python tests pass. Reuse complete retained reference meshes for native
fix trials; do not regenerate OCCT unnecessarily. Evidence logs:
`local/cohort2-resource-{serial-hcft,tests}.log`.

Remove 1.8 GiB of regenerable `target/debug/incremental` cache with no Cargo
build running; disk free space rises to 9.6 GiB. Inputs, baseline meshes,
reports and frozen workers remain. Future builds use one build job and disable
incremental output where appropriate.

A targeted native CMBNC replay peaks at 120,712 KiB and takes 20 s. Its boundary
failures occur with only 177/170 points, not at the million-point resource cap.
The endpoints are spatially adjacent at roundoff, but their chart coordinates
differ by exactly one full v period on a doubly periodic revolution surface.
The existing singly-periodic cutter skips these faces; subdivision cannot
repair that topological seam. This is an RCA lead, not a completed fix.
Temporary instrumentation is removed. Evidence:
`local/cohort2-resource-bound-diagnostic.{json,log}`.

### Bound proximity-query memory and replace the Python spatial index

Replace Trimesh/rtree proximity with libigl's native float64 point-to-triangle
AABB queries. Read STL transport through a memory map; compute areas in bounded
blocks and build at most 262,144 target triangles into a BVH. Take the minimum
over every target block. This is the same nearest-surface query, not a sampled
target, simplified reference, or looser acceptance gate. libigl 2.6.3 is an
optional offline dependency; Rust/browser dependencies are unchanged.

On the retained large CMB-XS comparison, `/usr/bin/time -v` measures 8.82 s and
660,016 KiB peak RSS (645 MiB). The old process was still running after six
minutes at roughly 3.6 GiB RSS. An intermediate whole-mesh libigl BVH took
10.22 s / 2,526,160 KiB; bounded BVHs remove that remaining memory spike.
Blocked and whole-mesh libigl summaries agree within 1e-12 mm. All twelve
retained cohort mismatch comparisons keep the same distance pass/fail result
as Trimesh; maximum summary difference is 7.25e-9 mm. The first deliberately
tight 1e-10 cross-backend assertion exposed tiny RMS differences, not changed
acceptance or missing geometry; full parity evidence records the differences.

27 Python tests pass, including a one-triangle-per-block regression that must
find nearest points across all blocks. Evidence: `local/cohort2-bvh-*.json` and
`/tmp/cohort2-bvh-{blocked,parity,tests}.log`. Documentation and dependency-version
provenance now match the new backend. Continue large-case iteration with one
worker; reference generation can still have its own memory cost.

### Condition spline charts in Cartesian distance per knot unit

FI7447054 source face #206/surface #512 has a v range of only 0.00975204.
The old chart scale compared homogeneous control-polygon lengths without
dividing by parameter ranges. Its mapped v width was 0.00673677 against a full
u period of 1, despite comparable physical extents. The resulting long, skinny
chords fold on the surface. UV triangles remain positively oriented and cover
one parameter rectangle; checking only distance to the supporting surface misses
the excess area. OCCT evaluation of all native UV samples agrees to roundoff,
so this is chart conditioning, not a surface-evaluation discrepancy.

Estimate Cartesian travel per parameter unit instead. Dehomogenize controls
before measuring lengths; otherwise weights and translations also distort the
chart. Replace the old generic aspect-ratio method with the rational-surface
scale actually needed by its sole consumer. No new subdivision modes or
model-specific conditions. A translated, non-unit-weight, unequal-knot-range
regression checks the units. All 147 library tests and 53 processing controls
pass. FI7447054 now passes the oracle: 524.260408 mm² vs 525.485950 mm², compared
with 634.765290 before; 66,943 triangles, roughly 1.1 s native meshing.

The rejected three-probe interior-chord experiment took 31.4 s, emitted 528,408
triangles and still had 583.107782 mm² area. It is not retained; evidence is
`local/cohort2-chord-*`. Accepted scale evidence: `local/cohort2-scale-*`.
The original 56-case controls now have 51 pass / five oracle errors: the new
reference-completeness check correctly rejects all five Coilcraft partial OCCT
meshes. Those are not new native processing failures and their previous sampled
agreements against partial references are not complete-reference guarantees.

At the user's memory checkpoint, three concurrent large OCCT comparisons used
about 1.7/3.7/3.4 GiB RSS. Stop that run's workload scope and retain its manifest,
two completed case results and partial diagnostic outputs; it is not a completed
17-case replay. Machine used memory drops from about 9.9 to 1.2 GiB. Subsequent
large comparisons will run singly while replacing the costly proximity path.

Freeze the preceding committed candidate as `local/cohort2-before-worker`
(SHA-256 `5b10d053c883ac4960cbe27ac85ce7b64ea5a1ad62167b6e5197bab024d0a479`).
Discovery tests 240 Würth and 240 previously untested KiCad models, in two
deterministic 120-file batches each. Würth: 226 pass, nine oracle mismatches,
two processing failures, three oracle timeouts. KiCad: 237 pass, three oracle
mismatches. Stop collection at these 14 geometry/processing failures; investigate
the three reference timeouts separately, not as proven native mesh defects.
All source hashes, manifests, baseline meshes and reference meshes are retained
under `local/cohort2-discovery-{wurth,wurth-batch2,kicad}`. Thresholds remain
0.1 mm sampled distance, 5% aggregate tolerance, and 0.01 mm OCCT deflection.

Initial failure families: FI7447037/7447054/7447070 excess mesh area;
AIG8 D22L30/D30L40 displacement; two RSTV switch mismatches and one RSTV
processing failure; TBL691308330002 and LANMX749600000 displacement;
CMBNC7448052502 processing failure; KiCad Murata1400, Bourns3299X and CHK EI48.
These are hypotheses pending per-face evidence, not claims that every cause is
already known. Native face localization must include instance transforms and
unit scaling; initial untransformed localization was wrong for several Würth
models. `local/cohort2-rca/localization.json` records witness match residuals.

### Reject incomplete OCCT reference tessellations

Murata1400's 4.474 mm discrepancy is an oracle defect: OCCT exports no triangles
for planar source face #111, despite a successful STL write and valid BRep.
That face has exact area 405.901625 mm². Full exact area 1846.994855 mm² agrees
with native 1846.232154, not the incomplete OCCT mesh's 1440.814250 mm².
The oracle now requires a nonempty triangulation for every transferred face.
It preserves a partial STL for diagnosis but rejects it as a reference rather
than asking Foxtrot to imitate missing geometry. The Murata run now explicitly
reports unmeshed face index 4. No native geometry changes for this case.

`PYTHONPATH=scripts local/occt-venv/bin/python -m unittest
scripts/test_corpus_geometry.py scripts/test_corpus.py`: 26 tests pass, including
a real six-face STEP box round-trip and rejection with meshing disabled.
The first invocation without PYTHONPATH failed to import the existing harness
test module; correcting the invocation resolves it. OCCT source/area evidence
is retained in `local/cohort2-occt-rca/`.

### Continue trim branches without disturbing good global projections

HCF2920roundwire's independent closest-point projections jump between nearby
regions of its swept spline. One closed contour acquires the wrong winding,
leaving three unmatched ports on each periodic cut. The source has no PCURVE
for these edges. OCCT's curve projector uses local continuation with global
fallback, rather than independent nearest points for every sample.

Walk source contours with fixed shared anchors. Keep the global projection
unless its connecting native surface path leaves the chord/source-uncertainty
budget; accept a local Newton branch only when both its residual and connecting
path fit that budget. Unconditional warm starts regressed a capacitor and CRD;
path qualification preserves both. No source vertex is moved and no triangle
is discarded. The two-sheet test checks continuity, rejection outside the
source budget, and preservation of the starting anchor.

The combined candidate passes all 146 workspace library tests and all 53
processing regressions. The original oracle replay remains 55 pass / one
invalid-source mismatch (`local/cohort-trace6-*`). Independent OCCT checks of
the two newly repaired MJ connectors and SMA connector all pass. HCF processing
completes with 442,505 triangles; its reference contains roughly 2.5M triangles,
so comparison exceeded the harness's 120 s limit. Comparing the retained meshes
without repeating reference generation completes in 288 s and passes: maximum
sampled distances 0.0402001 / 0.0163380 mm at the unchanged 0.1 mm threshold,
10,000 area samples plus 10,000 face probes per direction. Bounds agree exactly;
area differs by 0.0974%. Evidence: `local/cohort-hcf-oracle.json`. The original
four-case harness report retains its timeout as historical evidence; the
standalone comparison resolves it. All four additional models now pass sampled
comparison. Neither that result nor the report certifies topology or shading.

CMANC7848040382 is a separate source inconsistency, not a failed inverse solve.
OCCT independently confirms source curve #10238 lies 0.0531–0.0650 mm away from
surface #9520 at tested points, versus declared uncertainty 0.001 mm. Those
points lie on the original curve to about 2e-15 mm. The transferred OCCT shape
passes IsValid only with raised edge tolerances (up to 0.0537 mm on this face).
The stalled chord midpoint's 0.0717886 mm nearest distance agrees in OCCT and
Foxtrot. Evidence: `local/cohort-cmanc-{source-consistency,occt-gap}.json`.
No tolerance is inflated in Foxtrot to hide that inconsistency. OCCT IsValid
after transfer alone is not proof of a consistent original STEP representation.

The regenerated original-cohort report uses `cohort-trace6-oracle`, not a stale
candidate. All 48 meshes exist, all 16 Before/After hashes differ, all input
hashes match, and all 16 three-pane renders plus a wireframe state are inspected
with no browser errors. Coilcraft161 remains explicitly unresolved. The report
does not claim that the broader corpus is now free of accuracy failures.

Content-addressed hardlink deduplication of 672 frozen oracle meshes preserves
all paths and verifies every SHA-256 afterward. It links 529 duplicate files
and recovers 735.33 MiB (`local/cohort-mesh-dedup.{log,sha256}`). No frozen worker
duplicates are found. Approximately 12 GiB remains free.

### Require geometric extent before inferring periodicity

SMA60312102114506 declares 0.2 mm uncertainty. Its 0.05 mm rational fillets
have unequal interior weights; the control-weight test incorrectly inferred
a second periodic direction and wrapped the open fillet ends together.
Adjacent f64 chart values then lifted to positions separated by 0.05 mm.
No amount of mesh subdivision can resolve that discontinuity.

Closure inference now measures actual sampled displacement from the starting
iso-curve over every knot span. Weight changes alone are not physical extent.
The new rational-fillet regression preserves the existing resolved-loop and
weighted-loop tests. All 145 workspace library tests pass. The representative
SMA completes; the expanded 41-case replay has 33 passes, six other face-error
cases and two timeouts (`local/cohort-extent-regressions`). The unchanged
oracle replay remains 55 pass / one invalid-source mismatch.

Independent OCCT transfer/validity checks cover all 272 newly regressed inputs:
246 valid, 26 invalid, zero unknown; every input hash and root transfer matches.
HCF2920roundwire, LQS5020, CMANC7848040382, MJ615024143921 and RPSMA63012242124506
are all valid after OCCT transfer. Their remaining failures are not dismissed
as invalid sources. Evidence: `local/cohort-new-source-validity.{json,jsonl}`.

### Give bounded two-pole splines a lens chart

MJ615016137621 surfaces 250/251/253/255 have two collapsed ends. One is exact;
the other differs by approximately 1e-14 in decimal STEP controls. The polar
chart stretches that second pole around its rim, so refinement keeps bisecting
an already sub-ULP spatial chord while the chart still spans a large angle.

A convex lens chart collapses both ends and stays invertible in the interior.
Classification admits coordinate roundoff at the opposite end of an established
pole, not the potentially much larger source uncertainty. Original spatial
boundary vertices remain unchanged. The analytic regression exercises both
parameter orientations, interior inverse maps, and the near-collapsed trim.
The valid connector now completes (`local/cohort-lens-mj`). All 144 workspace
library tests pass; the combined 56-file oracle replay remains 55 pass / one
invalid-source mismatch (`local/cohort-lens-oracle`).

The complete pre-fix KiCad replay is also finished: 7,245 ok / 6 tessellation
errors out of 7,251. Both obsolete experimental full sweeps are stopped through
their systemd workload scopes; their completed per-case evidence is retained.
Current replay resources are reserved for the accepted fixes and open failures.

### Full replay exposes refinement regressions; preserve red owners

The committed `cohort-clean-worker` completes all 7,328 Würth inputs with
7,046 ok / 164 tessellation errors / 115 timeouts / 3 input errors. Relative
to `architecture-wurth`, 272 previously passing cases regress. The original
55/56 oracle result is not a claim of corpus-wide reliability. Exact replay
inputs are retained in `local/cohort-new-regressions-manifest.json`.

RCA: permanent green completion triangles can retain their diameter while
their altitude vanishes under repeated neighbor-driven splits. Surface 1145
of MJ615004141121 reaches a chart altitude/diameter ratio of 2.10e-8 at round
30; surface 1141 reaches 6.84e-8. The midpoint table partitions correctly;
ownership, not the table or inverse projection, is wrong.

Retain red leaves and rebuild temporary green completion. A failing child
promotes its owner; a second hanging midpoint level promotes the coarser
neighbor. No nested tree or adjacency layer is needed. If one tiny edge has
no representable midpoint, refine the other edges without moving either
endpoint; exhaustion of all usable edges remains an explicit error.

MJ615004141121 now completes in 31.22 s with 191,366 triangles. All 53 targeted
processing cases and the unchanged 56-file OCCT replay pass as before
(55 ok / one invalid-source mismatch). Evidence: `local/cohort-owner2-*`.
The broader 272-case replay is still running and already has remaining
failures. No global all-clear is claimed.

Rejected exact-global and distance-budget inverse queries do not repair the
refinement invariant. Longest-edge closure with incremental adjacency times
out on the same part; batch closure completes but produces 1.72M triangles.
Their patches/workers remain diagnostic evidence, not implementation.

Housekeeping removes a stale 422,211,072-byte Cargo cache at
`local/wurth-corrected-target`, 19,591,168 bytes of Python bytecode caches and
9,216 bytes of Ruff cache. Inputs, workers, meshes, reports and diagnostics
remain intact. Total reclaimed: 441,811,456 bytes.

### Refine curved face interiors with conforming subdivision

Use one CDT, then shared edge midpoints and an eight-entry subdivision table.
Split every free edge of a failing triangle rather than ranking edges in a
distorted metric. The latter left long skinny children unresolved after 32
rounds; subdividing all free edges clears the remaining CIRCM12 cap and large
capacitor regressions. Affine chart interpolation preserves parent partitions;
native-polar averaging did not. Local surface projection measures geometric
distance without counting tangential parameter distortion, and keeps the
sample itself as a feasible distance bound. Source offsets remain separate.

The hemisphere regression checks actual interior deflection, not triangle
counts. All 143 workspace library tests and all 53 targeted processing cases
pass. OCCT remains 55 pass / one invalid-source mismatch out of 56; all eleven
original OCCT-valid failures and four invalid Coilcraft variants now agree at
the unchanged comparison thresholds. This is sampled agreement, not a proof
of topology or exact geometry. Full sweep and report bookkeeping follows.

### Resolve trim chart curvature without moving source chords

Subdivide a trim chord when its lifted chart segment departs from the lifted
endpoint chord beyond the physical budget. Reproject the spatial midpoint;
keep source curve/surface offsets separate from chart distortion so refinement
does not try to erase STEP tolerances. This repairs trim connectivity that
interior Steiner points cannot repair. Exhaustion remains an explicit face
error. Combined validation: 143 library tests, 53 targeted processing passes,
and the 55/56 OCCT result recorded below.

### Cut regular periodic patches in native parameters

The polar annulus chart couples circumferential chord error to the other
parameter; on the 46-turn CRD spring it jumps across many polynomial spans.
Regular singly periodic patches now use a native strip. Clip trim segments
at a cut inside the largest vertex-free gap, preserve original endpoints,
pair odd-degree seam ports, and sample constructed seams by knot span.
Only genuine collapsed poles retain polar charts. Curvature seeds use native
knot spans and periodic copies inside the face's strip, not a fixed chart grid.

The rejected internal lattice constraints, native-polar midpoint averaging,
chart-Jacobian edge ranking, and unconditional Cartesian seam experiment are
not retained. Native averaging moved pole spokes off their parent edges;
the internal grid also introduced unnecessary trim intersections. The final
combined candidate passes all 53 targeted processing regressions and all 143
workspace library tests. The unchanged 56-file OCCT replay is 55 pass / one
mismatch (Coilcraft 2222SQ-161, invalid source; not claimed fixed).
Evidence: `local/cohort-clean-{regressions,oracle}`. Full processing sweeps
run Würth then KiCad with the frozen `local/cohort-clean-worker`.

### Keep torus cuts away from trim vertices

Place the radial angular cut in the middle of the unused angular gap rather
than on a trim vertex, where projection roundoff can cross it. Lowering now
reuses the same angle extraction as chart preparation, removing the duplicate
matrix path. This resolves the USB torus seam regression. The 143 workspace
tests and the targeted USB replay pass (`local/cohort-clean-regressions`).

### Condition cylindrical sliver charts by physical scale

Use the larger of axial extent and radius as the cylinder chart's axial scale.
Sub-tolerance axial slivers can have coplanar edge chords without making the
underlying cylinder singular. Store that scale explicitly and provide its
inverse map for curvature refinement. The scale check no longer compares a
length against an absolute machine epsilon. All 143 workspace library tests
pass (`/tmp/cohort-clean-tests.log`).

### Do not infer periods from thin extrusions

The last capacitor regression, WCAP-AI3H-P10D25L51 surface 10902, was assigned
two periods although STEP declares a linear open u direction. Its endpoint
iso-curves coincide within uncertainty because the whole extrusion is thin,
not because it closes. This produced a full-period chart edge between almost
identical XYZ points, which no boundary subdivision could approximate.
Require resolved interior variation (including rational basis variation)
before inferring closure. The control-net regression and 110 NURBS/triangulation
tests pass; the capacitor now completes. Evidence:
`local/cohort-regular-capacitor`, `/tmp/cohort-regular-tests.log`.

### Preserve small curved trims

Keep degree + 1 samples per polynomial trim span and at least three samples
on conics, in addition to the physical chord bound. A chord budget alone can
collapse two distinct sub-tolerance curved edges onto one segment and erase
a valid small face. The capacitor regression replay confirms the previously
collapsed small faces are retained. NURBS 53 tests and triangulation 56 tests
pass. Evidence: `/tmp/cohort-curve-tests.log`,
`/tmp/cohort-partition-tests.log`, `local/cohort-partition-regressions`.
This is independent of the still-experimental interior refinement below.

### Reuse feasible inverse upper bounds

Subdivision samples are themselves feasible solutions. Keep improving sample
positions as upper bounds and run Newton only to accelerate those improvements,
instead of repeatedly solving a knot cell from already-worse samples. This
retains global control-hull search and its precision. NURBS 53 / triangulation
56 tests pass; the CIRCM12 probe drops from 20.28 s to 5.17 s and remains
complete. Evidence: `/tmp/incumbent-{tests.log,circm12.json}`.

### Balanced inverse subdivision

The CIRCM12-643210100404 timeout localizes before surface 19948 reaches CDT.
Instrumented inverse queues retain almost the full v range while u shrinks
below 1e-5; the finite incumbent is not the issue. Curvature-driven splitting
does not measure uncertainty in the distance bound. Replace that heuristic
with balanced normalized parameter widths. Add a supporting-plane bound in
the incumbent residual direction, preserving correlations lost by boxes.
The exact oblique-extrusion endpoint regression and all 53 NURBS tests pass
in 0.27 s; CIRCM12 now completes in 20.28 s rather than exceeding 120 s.
No inverse tolerance or oracle acceptance limit changes. The connector replay
also includes the in-progress boundary/interior refinement. Evidence:
`/tmp/{trace-circm12.log,balanced-tests.log,balanced-circm12.json}`.

### Finite-patch inverse bounds and performance regression

The wider replay caught a regression in the curvature-directed inverse search:
roundoff-sized second differences kept splitting one effectively straight
direction while leaving the other unresolved. A normal slab bounds an infinite
plane, not the finite patch, so its lower bound could not prune those cells.
The `A_Wurth_WA-BCMC_79573131` probe exceeded 120 s. Instrumentation captured
subcells with one parameter width around 2.4e-7 and the other still 0.125.

Use an orthonormal control-hull box (normal and both tangent directions), and
fall back to balanced parameter subdivision below coordinate roundoff. The
bound remains geometric; no model names or relaxed acceptance limits enter the
algorithm. Cache extracted Bézier cells lazily, after their coarse hull passes
the distance bound. NURBS: 52 tests pass in 0.18 s. The reproduced Würth model
meshes in 725.75 ms with the same 3,287 triangles and complete status. Evidence:
`/tmp/cohort-box-{tests.log,json}`. Earlier `cohort-final` and `cohort-full`
sweeps precede this correction and are not final regression evidence.

## Implementing the oracle cohort fixes — 2026-09-07 (in progress)

The before baseline remains frozen in `local/oracle-rca-kicad`. Accepted changes
so far select spherical chart candidates by whole oriented-boundary clearance,
search subdivided homogeneous Bézier control hulls beyond local inverse minima,
and replace fixed edge sampling counts with a 0.01 mm chord budget converted to
native file units. Spline chord bounds use restricted control hulls; conics use
second-derivative bounds. Resolved curve/topological-vertex offsets are retained
so two distinct edges with shared endpoints do not collapse together.

The inverse-search implementation includes a same-knot-cell folded-cubic
regression. A focused Coilcraft131 probe confirms that original vertex 555 now
uses the correct parameter region, but the model still has a large area mismatch:
this corrects the demonstrated inverse defect, not all of its trim problems.
Splitting the inverse search by curvature instead of extrusion length reduces
the complete 52-test NURBS suite from 91.07 s to 0.70 s. Translation roundoff and
geometric size have separate terms in the search resolution.

The initial delegated inverse change contained formatting and an unused import,
not the reported search algorithm. Parent diff review caught this; the actual
control-hull search is implemented and verified in the later commits. Concurrent
worker commits also caused formatting churn across the two worker commits; no
history is rewritten. Do not treat their initial summaries as verification.

Completed cohort stages (56 identical input hashes, unchanged oracle settings):

| Stage | Result | Evidence |
| --- | --- | --- |
| Whole-boundary sphere chart | 43 ok / 13 mismatch | `local/cohort-chart` |
| Adaptive edge chords + inverse search | 45 ok / 11 mismatch | `local/cohort-edge` |
| Experimental triangle-plane interior criterion | 51 ok / 4 mismatch / 1 face error | `local/cohort-interior2` |

All eleven initially failing OCCT-valid sources and all forty passing controls
pass the triangle-plane experiment, but it does not adequately measure the
remaining Coilcraft chord errors. A geometric point-to-surface interior criterion
is now being checked instead; it is not yet accepted. Its analytic hemisphere
regression and all 56 triangulation tests pass. Two rejected criteria illustrate
why parameter-space correspondence error (over-refines nonlinear planar charts)
and triangle-plane distance (misses in-plane chord errors) are insufficient.
No rejected experimental report is presented as the finished After report.

`local/cohort-interior3` is the current oracle replay; full processing sweeps
run Würth then KiCad in `local/cohort-full-{wurth,kicad}`, with no STL exports to
avoid disk bloat. Results remain pending. Old intermediate chart/edge STLs are
losslessly compressed and SHA-256 checked; `local/cohort-cleanup.json` records
the operations. The original before meshes, current evidence, STEP inputs and
reports are retained. Approximately 18 GiB is free at this checkpoint.

## Three-pane visual report — 2026-09-07

Added `scripts/oracle_report.py` and its HTML viewer. The generated report in
`local/oracle-visual-report` covers all 16 oracle mismatches with synchronized
Before / After / OCCT panes, shared bounds/cameras, neutral flat shading,
wireframe, triangle counts and directional distance metrics. No accepted fix
exists yet, so After explicitly duplicates Before. A separate middle-pane
selector shows the rejected sphere-mean / spline64 experiments. Future runs
can supply `--after` with a new corpus output; input hashes must match.

Generated the report successfully and loaded all 16 cases in both modes
(32 browser states), with three canvases and no browser errors. Inspected
the complete 16-model screenshot overview, baseline state, Disc16 improvement,
Sharp regression, wireframe, and linked rotation after dragging the left pane.
Captures are in `.amp/in/artifacts/oracle-comparisons` and
`.amp/in/artifacts/oracle-review-*.png`. This verifies the report rendering,
not geometry correctness. Python compilation and `git diff --check` pass.
The `oracle-review` supervised service serves only the generated directory on
port 8090. The report vendors pinned Three.js modules and license; no external
requests are needed when viewing. It occupies 167 MiB; obsolete blank capture
removed, current RCA evidence retained, 19 GiB disk space remains free.

## OCCT failure batch and root causes — 2026-09-07

The oracle implementation and initial samples are committed. The next targeted
KiCad cohort completes 56 files: **40 ok, 16 oracle_mismatch**, with no processing
failures. Stop collection here and investigate the batch, as requested. This
cohort combines earlier quality cases and a sample; its failure fraction is not
an estimate of the complete corpus. No production geometry fix is made here.

Reproduction:

```sh
RUST_LOG=error local/occt-venv/bin/python scripts/corpus.py local/kicad-packages3D \
  --manifest local/cleanup24-kicad-manifest.json --occt \
  --worker local/architecture-worker --meshes all --jobs 2 --threads 1 \
  --timeout 120 --output local/oracle-rca-kicad
```

Distances below are maximum sampled mm, Foxtrot→OCCT / OCCT→Foxtrot. The
baseline uses 10,000 area samples per direction, additional face centroids,
0.1 mm acceptance, and OCCT deflection 0.01 mm / 0.1 rad. Face/surface numbers
are STEP entity IDs, not OCCT explorer indices. All worst-point assignments
match the native per-face mesh within 4e-15 mm. In the reverse direction,
localization uses the nearest Foxtrot point, not the OCCT source point.

| Model (basename without `.step`) | Max mm, forward / reverse | Localized cause / face IDs |
| --- | ---: | --- |
| CP_Elec_5x5.3 | .18246 / .17802 | Cylinder chord sag, face 973 / surface 1020 |
| CP_Axial_L30.0mm_D15.0mm_P35.00mm_Horizontal | .43681 / .43742 | Cylinder chord sag, 1102 / 1115 |
| C_Disc_D16.0mm_W5.0mm_P7.50mm | 5.34288 / 2.69460 | Distorted sphere trim chart, 298 / 376 and 183 / 293 |
| SMA_Molex_73251-2200_Horizontal | .11011 / .09501 | Spline approximation, 7981 / 8076; reverse witness cylinder 10381 / 10437 |
| L_Coilcraft_2222SQ-111 | 1.72070 / .77029 | Wrong inverse-projection minimum, 17 / 479 |
| L_Coilcraft_2222SQ-131 | 1.72162 / .80428 | Wrong inverse-projection minimum, 17 / 493 |
| L_Coilcraft_2222SQ-161 | 1.76063 / 1.05528 | Wrong inverse-projection minimum, 17 / 729 |
| L_Coilcraft_2222SQ-181 | 1.70173 / .91650 | Wrong inverse-projection minimum, 17 / 601 |
| L_Coilcraft_2222SQ-221 | 1.72593 / 1.07558 | Wrong inverse-projection minimum, 17 / 616 |
| L_Vishay_IHSM-7832 | .10762 / .11604 | Distorted sphere trims, 510 / 513 and 166 / 173 |
| L_Wuerth_XHMI-8080 | .11656 / .11662 | Distorted sphere trims, 378 / 381 and 202 / 211 |
| Bourns_8100, L38.1mm W20.3mm Px15.24mm Py22.86mm | .22007 / .23539 | Sphere charts, 6074 / 6080 and 803 / 815; residual cylinder approximation |
| Bourns_8100, L39.4mm W20.3mm Px15.24mm Py22.86mm | .17473 / .18744 | Sphere charts, 49980 / 49986 and 47834 / 47846; residual cylinder approximation |
| Bourns_8100, L41.9mm W20.3mm Px15.24mm Py22.86mm | .17069 / .18724 | Cylinder 18983 / 19191 forward; sphere 47403 / 47409 reverse |
| Sharp_IS485 | .28968 / .29190 | Sphere chart conditioning/interior approximation, 1113 / 1118 |
| RV_Disc_D12mm_W6.3mm_P7.5mm | .10055 / .09949 | Revolution approximation, 50 / 151 and 168 / 271 |

### Evidence and rejected explanations

**Inverse projection, not spline evaluation.** All five Coilcraft main faces
contain original boundary vertices projected onto a wrong local minimum.
For 131, vertex 555 projects to native UV (3.45717, 14.19925), 0.06310 mm from
its input position. OCCT finds (5.92636, approximately 0), residual 4.21e-15 mm.
Adjacent boundary vertices remain near v=0: this creates a large trim-domain
spike across a 0.096 mm spatial edge. Other variants reproduce residuals
0.051–0.064 mm versus approximately 4e-15 mm. Native forward evaluation agrees
with OCCT at all audited inserted points on 131 within 2.7e-14 mm. Its main-face
area is about 1036 mm² versus the underlying complete surface's 216.65 mm²;
raising the interior grid from 16 to 64 increases the native area to about
6488 mm². More samples cannot repair the wrong domain.

`nurbs/src/sampled_surface.rs::uv_from_point` bounds candidate knot cells but
tries only one nearest seed per cell. A cell's position bound does not prove
that one local Newton solve finds its global closest point. Local stationarity
is not an on-surface residual contract.

**Distorted charts, not a basic spherical inverse failure.** The sphere chart
places its antipode only half a boundary-edge clearance outside the trim.
Joining independently projected sparse vertices with straight chart chords
can introduce crossings absent from the true surface boundary. Crossing
resolution then interpolates spatial chords to manufacture new vertices.
Vishay's 21 original boundary vertices round-trip within 2.7e-15 mm; its two
constructed crossings deviate by up to 0.5002 mm. Disc16's 161 originals are
within 1.14e-7 mm, but three constructed crossings reach 9.082 mm.

A diagnostic mean-centered chart makes Vishay and Wuerth pass (.00915 and
.00517 mm maxima), and reduces Disc16's maximum from 5.343 to .1057 mm. Bourns
variants improve but retain cylinder deviations around .151–.171 mm. However,
Sharp's reverse error worsens from .292 to 1.052 mm: a blind center change can
select the wrong/complementary region. This experiment is rejected. Sharp's
original vertices already round-trip within 1e-15 mm; centroid deviation is
.8214 mm, implicating chart conditioning and interior approximation instead.

**Approximation needs a physical tolerance, not larger fixed grids.** CP_Elec
and CP_Axial worst points lie on cylinders, not their nearby spline faces.
Their radius-2.5/radius-7.5 boundaries have chord sag .1913/.4394 mm while
vertex radial residuals are only about 1e-5 mm. Densifying spline interiors
does not change these failures. Current policies use 8 samples per spline
knot span, 32 per ellipse revolution, sphere grid 6×6, and spline grid 16×16;
these do not bound physical error. The varistor improves to .03074 mm with
grid 64, demonstrating approximation sensitivity, not endorsing that fix.
Finer OCCT references (.001 mm/.03 rad, 100,000 samples) still fail SMA
(.11065/.09582 mm) and the varistor (.10347/.10292 mm). SMA's exact split of
edge versus interior error remains unresolved; the failing spline face is
localized. These mismatches are not explained by baseline OCCT coarseness.

### Source validity and limits

Eleven failing models pass OCCT BRep validation. All five Coilcraft complete
shapes fail it; on 131, two small planar caps report SelfIntersectingWire /
UnorientableShape, while the main spline face is valid. This does not establish
STEP spec invalidity by itself. Keep cap/source issues separate from the
independently demonstrated inverse-projection defect; do not add source repair
heuristics to accommodate invalid inputs. Some meshes contain zero-area facets;
the oracle compares positive-area surfaces and does not prove manifoldness.

### Fundamental fix strategy (not implemented)

1. **Make inverse projection residual-checked.** Subdivide bounded spline
   patches where the lower bound still permits a better solution; retain
   Newton as a local accelerator, not a completeness test. Include parameter
   boundaries in minimization. Use knot/Bézier-aware bounds, with explicit
   handling of rational-weight assumptions. Distinguish closest-point queries
   from inversion of a trim point known to belong to the surface. Return a
   diagnostic when the required geometric residual cannot be met. Neighbor
   continuity may propose candidates, never establish correctness.
2. **Preserve the oriented trim domain in well-conditioned charts.** Choose
   charts using the complete oriented boundary, not a centroid heuristic.
   Preserve outer/inner loop meaning and face sense; split patches when one
   chart cannot represent the domain safely. Discretize the actual charted
   trim, rather than treating long projected chords as exact. Do not repair
   invented intersections by snapping vertices onto the surface.
3. **Replace fixed sampling counts with one physical error budget.** Own
   canonical 3D edge samples once and share them between incident faces.
   Refine boundary curves and face interiors together until their deviation
   budgets are met. Use flat arrays of samples, patch bounds and a work queue;
   keep surface evaluators responsible for geometry, not separate meshing
   policies. High-degree splines require bounds/subdivision rather than a
   midpoint-only test that can miss oscillations. Keep geometry in f64 and
   convert only at the browser output boundary.

Implement each logical change in its own commit. First add focused analytic
regressions for the demonstrated invariant (multiple minima within one cell,
oriented spherical trims, cylinder sag/shared edges), then replay these 16
failures plus the 40 passing controls at unchanged oracle tolerances. Follow
with full Würth/KiCad processing checks and broader oracle coverage. Track
triangle counts and timings to prevent solving error by indiscriminate density.
Do not claim all 16 fixed until that replay succeeds; sampled agreement still
is not a certified maximum-distance bound or a topology guarantee.

Evidence retained in `local/oracle-rca-kicad`, `local/oracle-face-localization.json`,
`local/oracle-cylinder-sag.json`, `local/oracle-coilcraft-{projection,evaluation}.json`,
and `local/oracle-counterfactuals`. Replay scripts and instrumented workers remain
under `local/oracle-*`; diagnostic source changes are removed from production.
The initial `local/oracle-moderate-rca/REPORT.md` contains superseded face guesses
and conflates constructed crossings with original vertices; this worklog and
the exact localization JSON supersede those conclusions.

**Post-cohort housekeeping is now required:** preserve inputs, manifests,
reports and active failure evidence; archive superseded cohorts with verified
contents, remove disposable build/cache outputs, and check available disk space
before starting the next cohort. Avoid accumulating another set of full meshes
when failure-only output or a frozen existing baseline is sufficient.

This checkpoint archives Würth/KiCad passes 19–21 as `local/*-repair-pass*.tar.gz`.
All 132,237 regular files pass SHA-256 comparison against archive contents before
their expanded directories are removed, recovering 8,864,024,213 allocated bytes.
`local/oracle-rca-cleanup.json` records the inventory. `cargo clean` removes stale
build outputs; the release corpus worker is then rebuilt successfully and is
SHA-256 identical to `local/architecture-worker`. `df -h .` reports 20 GiB free
after rebuilding, up from 6.3 GiB. STEP sources, reports, and current oracle
evidence remain available. The compact user-facing evidence export is
`.amp/in/artifacts/oracle-rca-batch.json`. Its 56 results, 16 failing paths and
all face assignments are checked against the original reports. `git diff
--check` passes; only this worklog changes after removing the experiments.

## Sampled OCCT surface oracle — 2026-09-07

Processing acceptance is not geometric correctness. Extended the existing
optional bounds/area oracle with bidirectional point-to-triangle distances,
using Trimesh, NumPy, SciPy and Rtree rather than a custom spatial index.
The native tessellator and browser pipeline are unchanged.

`--occt` now uses 10,000 seeded area samples per direction plus up to 10,000
face-centroid probes, with a configurable 0.1 mm maximum sampled-distance
tolerance. OCCT uses 0.01 mm linear / 0.1 rad angular deflection, in millimeters,
and must transfer every STEP root. Conversion and comparison share the existing
timeout-isolated subprocess. Reports retain percentiles, RMS, directional
out-of-tolerance area fractions, worst-point coordinates and dependency versions.
No alignment, rescaling or mesh repair is applied. Zero-area STL facets remain
reported but do not prevent comparison of the positive-area surfaces; source
exports are untouched. This avoids making STL degeneracy a browser acceptance
criterion again.

Executed samples with frozen `local/architecture-worker`, OCCT 7.9.3.1,
Trimesh 4.12.2, NumPy 2.4.6, SciPy 1.17.1 and Rtree 1.4.1:

| Part | Foxtrot → OCCT max (mm) | OCCT → Foxtrot max (mm) | Result |
| --- | ---: | ---: | --- |
| examples/cube_hole.step | 0.032535 | 0.032464 | agreement within tolerance |
| DSUB-15 socket, 14.56 mm edge offset / 15.98 mm mounting offset | 0.035962 | 0.039898 | agreement within tolerance |
| Coilcraft 2222SQ-131 | 1.721622 | 0.804278 | oracle mismatch |

The first two also pass bounds and area checks, with no area samples outside
0.1 mm. Coilcraft's Foxtrot area is 1095.004 mm² versus OCCT's 275.003 mm²;
77.30% of Foxtrot area samples and 40.07% of OCCT area samples exceed tolerance.
This is strong evidence of a geometry discrepancy, not merely different mesh
density or normals. The discrepancy is not repaired in this tooling task.
Next investigation should localize the worst points to STEP faces and compare
their trims/sampling; do not loosen the oracle to make the part pass.

Evidence: `local/surface-oracle-{examples,parts}/results.json`, each case's
`oracle.json`, and both retained STLs. Reproduction uses `scripts/corpus.py`
with `--occt --meshes all --timeout 120`, selecting the exact model paths from
these manifests. Optional tolerances/budgets are documented in README.

Verification: all 25 Python tests pass in the oracle environment. Added
regressions show that different triangulations agree, displaced surfaces can
evade bounds/area but fail the distance check, the reverse direction detects
missing geometry, and zero-area facets remain diagnosed. No Rust code changes.

Limitations: sampled agreement is not a certified Hausdorff bound, topological
proof, or shading check. Very small unsampled defects can escape; OCCT itself
can be wrong. STL world coordinates round to f32, so extreme coordinate offsets
or much tighter tolerances require higher-precision follow-up. These three
samples are not a corpus-wide accuracy claim.

## Architecture refactors — 2026-09-06

Implemented the approved sequence: face-local construction, structured outcomes,
then immutable surfaces with explicit face-chart preparation. Each logic change
has its own local commit on `wurth-kicad-step-repairs`; no compatibility shims,
geometry fallback, tolerance relaxation, or new tessellation dependency is added.

- Faces build into a reusable local mesh and append only after success. Failed
  faces cannot leak vertices or affect the following face's indices. Removed
  obsolete global offsets and the whole-torus combine/truncate dance.
- Every final vertex receives its face normal and color after topology finishes,
  including constructed constraint intersections. Boundary positions do not
  move. A crossing-face regression checks color, normals and reversed winding;
  failed-before/after-success cases check atomic publication.
- `Stats.failures` is the source of truth for completion and derived counts.
  Records identify the STEP entity, surface when available, category and reason;
  deterministic sorting survives parallel shape traversal. Native worker schema
  2 distinguishes rejected input from process crashes and partial tessellation.
  The harness no longer counts log lines or accepts legacy STL-only workers.
- Native, GUI and browser consumers receive the structured diagnostics. Browser
  geometry remains a transferred `Float32Array`, not JSON numbers. Partial
  warnings survive camera movement; error text uses `textContent`; diagnostic
  STL conversion refuses to silently export partial results.
- `PreparedSurface` borrows immutable `Surface` geometry and privately owns its
  face chart. Preparation takes trims, orientation, source uncertainty and seam
  evidence; projection, sampling and normal evaluation use that prepared chart.
  Removed mutable preparation state, premature attribute writes and test-only
  compatibility methods. Tests prepare actual surfaces, including independent
  opposite spherical-face charts over shared geometry.

Integration verification:
- `cargo test --release --workspace`: 141 tests pass, including doc tests.
  The final local-offset cleanup also passes all 54 triangulate library tests;
  `cargo test --release -p triangulate --example corpus_worker` passes its test.
- Python harness and geometry tests: 21 pass. Release native worker build and
  actual `wasm32-unknown-unknown` release build pass. Matching wasm-bindgen
  0.2.128 generates the demo wrapper and binary; no generated binary is committed.
- Real browser testing finds and fixes an initialization race: install the
  message handler immediately and await one initialization promise per request.
  The binary URL is explicit and the deploy symlink matches its generated name.
  An immediate request to a new worker returns schema 2, a real Float32Array,
  and the connector's precise face #13017 / surface #25 diagnostic.
- Complete, partial, rejected-input and recovery loads execute in Three.js.
  Mouse-drag orbit preserves the partial warning; both selectors are enabled
  after rejection. Screenshots are inspected under `.amp/in/artifacts/architecture-*`.
  Coilcraft before/after inspection confirms coarse faceting and dark patches
  already exist in the baseline; this is not a claim of visually correct meshes.
- Against the frozen pre-chart `local/outcomes-worker`, final oriented triangle
  records are byte-identical for Coilcraft 2222SQ-131, a DSUB-15 socket and the
  unresolved Würth connector. Comparison preserves winding, normals, colors and
  multiplicity while ignoring nondeterministic triangle order. Evidence:
  `local/architecture-buffer-comparison.json`.

The final full replay uses frozen `local/architecture-worker`, the exact scan-22
manifests, four processes, one Rayon thread per process, a 60-second per-file
timeout and `--meshes none`. Würth completes before KiCad starts. Reports and
per-file diagnostics live in `local/architecture-{wurth,kicad}/`.

Full replay result: all 7,328 Würth and all 7,251 KiCad inputs complete.
Würth has 7,318 accepted, seven partial tessellations and three input rejections;
KiCad has 7,251 accepted. There are no new processing failures. The three former
parser "crashes" now correctly report `input_error`. Exact input path/hash sets,
triangle/face/shell counts, f64 degenerates and browser degenerates/nonfinite
counts match scan 22. Finalized attributes reduce zero-normal vertex uses on
1,232 Würth and 832 KiCad models, with no increases. Only the seven partial
Würth meshes change vertex counts: failed-face scratch vertices are no longer
published (two connector vertices and 28 per invalid transformer). Detailed
comparison: `local/architecture-comparison.json`.

Outstanding processing scope is unchanged: nine previously confirmed invalid
sources need no support; WR-TBL 691404910001B face #13017 / surface #25 still
has a completely cancelling boundary and remains unresolved, not proven invalid.
The existing local geometry/shading quality issues remain separate from
processing acceptance. No claim of watertightness or complete visual correctness
is made by this full replay.

## Branch cleanup review — 2026-09-06

Reviewed the complete branch diff against `origin/master` (this repository has
no `main` branch), covering CDT replacement, STEP parsing and call sites,
NURBS evaluation/projection, surface charts and face construction, browser
conversion, corpus tooling, tests and documentation. Invalid STEP inputs do
not need compatibility handling, per the user's latest decision. The nine
confirmed invalid sources are not repair work; the unresolved connector trim
and previously documented shading/geometry defects are not claimed fixed.

Cleanup changes, committed independently:
- Share the factored squared-distance comparison used by curve/surface
  projection, preserving operation order and convergence safeguards.
- Remove the unused browser-area metric and its validation requirement;
  retain all acceptance and quality diagnostics and legacy report reading.
- Correct README descriptions of buffer-only runs and review manifests.
- Remove CDT's unused logging dependency, no-op `long-indexes` feature, and
  never-emitted legacy `PointOnFixedEdge`/`WedgeEscape` errors. These removed
  public names have no repository consumers. Correct predicate-range errors.
- Reuse curve knot conversion and surface construction for STEP splines;
  surface-construction diagnostics now name surface types, not curve types.
- Derive periodic unwrapping directly from Cartesian chart data, reusing its
  stored scale instead of recomputing it and separately excluding polar charts.
- Remove the stale member-level `cdt/Cargo.lock`: Cargo resolves this member
  through the root workspace, and repository policy ignores generated locks.

Verification: `cargo test --release --workspace` passes all 137 tests;
`python3 -m unittest discover -s scripts -p 'test_corpus*.py'` passes 21;
release worker build and WASM host `cargo check --release` pass. The worker's
targeted unit test also passes. No numerical tolerances or mesh acceptance
rules are loosened. No new dependency or geometry fallback is added.

Replay 24 covers 70 Würth files followed by 56 KiCad files: a hash-selected
sample plus every remaining failure, every f64-degenerate case and high
zero-normal-count models. Status and all retained geometry/quality counts
match scan 22 exactly (60 Würth ok, 10 unchanged failures; all 56 KiCad ok).
Evidence: `local/cleanup24-{wurth,kicad}/results.json` and
`local/cleanup24-comparison.json`; frozen worker `local/cleanup-worker24`.
Three difficult representative browser buffers have byte-identical oriented
triangle records after sorting triangles and cyclically rotating their vertex
records, retaining winding, normals, colors and multiplicity. Raw buffer order
varies even between unchanged-worker runs; it is not an equality guarantee.
Evidence: `local/cleanup24-buffer-comparison.json`. This is a targeted cleanup
replay, not a new full-corpus scan or visual certification. Changes remain local.

## Remaining processing failures — replay 23, 2026-09-06

Replayed every non-ok input from browser scan 22 with the unchanged
`local/browser-fast-worker`, `--meshes none --jobs 4 --threads 1 --timeout 60`
and `RUST_LOG=triangulate=debug`. Command:

```
python3 scripts/corpus.py local/wurth/3dmodels --worker local/browser-fast-worker --rerun local/wurth-repair-browser22/results.json --meshes none --jobs 4 --threads 1 --timeout 60 --output local/remaining-processing23
```

Result: **all ten failures reproduce**, seven `tessellation_error` and three
`crash` labels. The latter are controlled `StepParseError` returns, not process
panics. No geometry or acceptance logic changes in this investigation.
`local/remaining-processing23/results.json` preserves hashes, reproduction
commands and per-case logs. KiCad has no remaining processing failures in the
complete scan 22; it is not needlessly rerun for this read-only investigation.

Root causes and disposition:

- WE-CMANC-M 7848031002 and WE-CMBNC-TypeM 7448031002 contain Parasolid, not
  STEP. They require correctly exported STEP sources or a separate Parasolid
  importer; loosening the STEP parser cannot recover their geometry.
- WE-RFI-0402 references undefined surface #0 from face #707. The replay
  confirms that exact reference error. There is no supplied surface to mesh.
- The six EE13 transformer variants listed in browser22/wurth.json each fail
  on surface #36. Their equal 0.127 radii violate
  [DEGENERATE_TOROIDAL_SURFACE WR1](https://www.steptools.com/stds/stp_aim/html/t_degenerate_toroidal_surface.html):
  `major_radius < minor_radius`. Supporting their horn-torus limit would be an
  explicit nonconforming-input repair policy, not a spec-correctness fix.
- WR-TBL 691404910001B face #13017, surface #25, bound #12195 remains
  unresolved, **not proved invalid**. Rechecked its curve sampling, bounded
  inverses, projection convergence and exact-coordinate retrace cancellation
  against the existing high-precision source analysis. Curve #827's bounded
  nearest-point trim is zero; #828 moves only 8.13e-14 mm. The topological
  vertices are 1.99e-8 mm apart and about 7e-7 mm off the surface, within the
  source's 0.005 mm uncertainty. The tiny nonzero high-precision UV area is
  that of our endpoint-replaced polyline, not independent evidence of the
  intended source trim. More precision or samples do not resolve that
  ambiguity. Source uncertainty alone does not authorize deleting topology.

No solver contract violation has been established for the last case. Keep
the cancellation error rather than dropping the face or manufacturing a
triangle. Remaining decisions require corrected sources or explicit agreement
to best-effort, visibly diagnosed nonconforming/under-resolved geometry
handling. Such handling must not count omissions as fully successful meshes.

## Browser-first checkpoint — 2026-09-06

The user clarified the product target: good-looking Three.js meshes, not
lossless CAD topology or world-coordinate STL. No glTF export is needed.
Acceptance now exercises the actual centered f32 position/normal/color buffer;
f64 geometry remains the computational source of truth. Nothing is deleted,
merged or perturbed to make diagnostics pass.

| Full browser scan 22 | Würth | KiCad |
| --- | ---: | ---: |
| Inputs processed | 7,328 | 7,251 |
| Browser-buffer acceptance | 7,318 | 7,251 |
| Processing/input failures | 10 | 0 |
| Models flagged for quality review | 1,562 | 849 |
| Models with collapsed browser triangles | 289 | 21 |
| Models with zero-normal vertices | 1,417 | 843 |
| Models with f64 degenerate triangles | 4 | 2 |

Quality categories overlap. These are diagnostic counts, not counts of visible
bugs. The 350 previous invalid-mesh results are **reclassified**, not fixed.
All source path/hash sets and the frozen worker digest are verified; triangle,
face, error, panic and f64-degenerate counts match pass 21 exactly. All Würth
files finish before KiCad starts. Reported tessellation errors, nonfinite
attribute buffers, empty meshes and entirely collapsed meshes fail acceptance;
unreported omissions still require geometric or visual review.

**Current evidence:** `.amp/in/artifacts/browser22/{summary,wurth,kicad}.json`.
The corpus-specific `*-review-manifest.json` files preserve source hashes for
focused replay. `ok` is not visual certification. Screenshots cover DSUB,
axial and air-core inductors using the production Three.js scene module and
the exact shared browser buffer generated natively. The WASM wrapper passes
host `cargo check`; its deployed prebuilt wasm file is not rebuilt in this orb.

**Outstanding real work:**
- Nine confirmed invalid Würth inputs remain correctly rejected; the terminal
  block WR-TBL691404910001B's canceled trim remains unresolved.
- Crossing-created vertices have zero normals and are never assigned normals
  in face finalization. This is a concrete shading bug. The 2,260 zero-normal
  model flags also include singular-surface fallbacks; not all are attributed
  to that one path. Coilcraft 2222SQ-131 has 639 zero-normal vertex uses out of
  8,592 and visibly poor faceting/dark patches; a comparative repair is needed
  to separate geometry defects from normal defects.
- Six models retain f64 degeneracies (DSUB61803729321, CMB-XS744821110,
  CMBHC-S, CMBNiZn-S, Bourns L39.4/W20.3 and L41.9/W20.3). Chart crossings and
  inadequate interior sampling remain genuine geometric risks, not STL issues.
- Browser collapse alone is now review evidence. The normal-sized axial
  Fastron inductor looks intact from the side with both bent leads visible;
  another viewing angle hides one lead, illustrating why one screenshot
  cannot establish missing geometry. DSUB has suspicious dark slivers that
  remain review items. There is no corpus-wide visual or area-equivalence claim.

**Simplification and iteration tooling:** shared `Mesh::to_triangle_buffer`
removes duplicate browser centering code and fixes the z-bounds bug. Bounds use
referenced vertices, avoiding unused tessellation samples changing framing.
The obsolete 352-line thread-based regression runner and unused `glob` dev
dependency are removed. One process-isolated harness remains; old frozen
workers retain explicitly labeled legacy acceptance for reproducibility.
`--meshes none` avoids all STL export/readback while preserving browser
diagnostics. Every new report emits a replayable `review-manifest.json` and
sorts quality-flagged results ahead of unflagged successes. The worker can
optionally emit the exact browser binary buffer for direct Three.js repros.

**Verification:** 137 workspace tests and 21 Python harness/geometry tests pass.
Browser buffer capture is byte-identical to the rendered native probe. The
no-STL path preserves counts and area within summation roundoff. On one DSUB
file, three invocations take 4.62 seconds with STL versus 2.49 seconds without
(1.86x wall-speedup); this is not a corpus-wide benchmark. Evidence:
`.amp/in/artifacts/browser22/iteration-benchmark.json`. Full scan 22 uses the
earlier frozen browser worker with STL diagnostics, unchanged during its run;
later no-STL tooling does not alter mesh conversion or acceptance.

Completed scan logs larger than 16 KiB are compressed and their decompressed
SHA-256 values verified before removing raw copies. The 12,200 operations
recover 2,648,179,043 bytes; inventory: `local/browser22-log-compression.json`.
Those log paths now end in `.log.gz`. Approximately 11 GiB is free. Temporary
browser-probe source and preview service are removed/stopped; the production
worker's optional buffer capture replaces the probe. Review screenshots remain.

Changes are committed separately on `wurth-kicad-step-repairs`; this round is
not pushed. The previously pushed checkpoint remains unchanged remotely.

## Historical strict-STL checkpoint — 2026-09-05

This section records the old strict-STL methodology; it is not current acceptance.
The task has expanded from assessment to fundamental repairs of both corpora,
with one local commit per logic change. No push or merge is authorized.

| Verification level | Würth | KiCad |
| --- | --- | --- |
| Last completed full sweep (pass 21) | 6,991 / 7,328 pass | 7,228 / 7,251 pass |
| Improvement over pass 20 | 4 more pass | 11 more pass |
| Inverse-cell cohort coverage | 452 files | 156 files |

- **Committed repair:** knot-cell-bounded inverse projection; 136 workspace
  tests pass, no cohort status regressions or increased f64 degenerate counts.
  Frozen worker: `local/repair-inverse-cells-worker`. Full pass 21 completes
  Würth first, then KiCad, confirms all 15 improvements and has no status
  regressions. There are 360 failures remaining, including invalid inputs.
- **Experiment shelved:** chart-boundary refinement plus local surface samples
  clears all f64/f32 degenerates on WR-DSUB61803729321 and passes 137 workspace
  tests, but broader cohorts expose regressions. Its changes are removed from
  the source tree and saved in `local/chart-centroid-prototype.patch`; frozen
  workers and diagnostics remain. The OCCT comparison is not a pass: OCCT's
  exported reference itself contains two degenerate triangles, and the
  surface-area difference remains about 1.87%.
- **Canonical completed reports:**
  `.amp/in/artifacts/repair-pass21/{wurth,kicad}.{json,csv}` and
  `.amp/in/artifacts/repair-pass21/regressions.json`. Exact source path/hash
  coverage and frozen worker hashes are verified. A harness pass establishes
  neither manifold topology nor geometric equivalence to the STEP model.
- **Remaining work:** chart topology and interior sampling, f32 quantization
  versus avoidable slivers, seven Würth face-error cases, and per-file RCA
  that is still qualified where the geometric cause is not established.
- **Disk policy:** keep inputs, manifests, reports, logs, reproductions and
  frozen workers; remove superseded reproducible meshes and losslessly
  compress retained ones. Latest inventories: `local/bookkeeping-mesh-cleanup.json`
  and `local/chart-mesh-compression.json`. They record another 4,042,129,173
  bytes recovered. Full passes 20/21 remain untouched; completed experimental
  meshes are retained as verified gzip streams. Earlier inventories are
  `local/superseded-mesh-cleanup.json` and `local/disk-cleanup-followup.json`.
- **Next checkpoint:** review every chart-cohort status regression before
  attempting another geometry change. Never promote a better failure
  count or an invalid OCCT reference into a geometric-correctness claim.

## Original baseline scope and acceptance (historical)

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

The anchor-cohort rerun completed: of 328 formerly non-ok Würth files, 6 now
pass (188 tessellation errors, 131 invalid meshes, 3 source crashes remain).
Of 350 KiCad files, 6 now pass (228 tessellation errors, 116 invalid meshes).
These selected counts must not be extrapolated into full-corpus totals.

### Cancel retraced chart boundaries before intersection construction

KiCad Texas DSBGA-8 0.9x1.9 face #238 has one circular trim plus EDGE_CURVE
#242 traversed forward and backward. The seam contributes no region boundary.
However, splitting its projected crossings before parity cancellation creates
slivers: exact rational predicates on the dumped coordinates confirm interior
crossings at parameters 4.28e-17 and 1.85e-15, which the epsilon-based splitter
then ignores. The CDT sees a genuinely inconsistent polygon manufactured by
the preprocessing stage, despite identical forward/backward samples.

Canonicalize exact chart-coordinate identities and cancel even segment
multiplicities before intersection construction. This implements the existing
even-odd region semantics earlier, with no tolerance, retry, or model-specific
path; distinct periodic cut representatives remain distinct. Retain all samples
as vertices. Reject a nonempty boundary that cancels completely rather than
accidentally filling its convex hull. Unit coverage includes signed zero,
duplicate identities, complete cancellation, and one-ulp-separated coordinates.
Workspace tests and eight checkpoints pass. The 1,295-ball BGA now passes with
all 1,295 former CrossingFixedEdge faces resolved, as do both tested Texas
DSBGA-8 variants. Evidence: `local/repair-seam-parity-{check,bga}`.

OCCT comparison for Texas 0.9x1.9 is **not an oracle pass**: OCCT itself exports
eight degenerate triangles. Foxtrot exports 1,204 finite nondegenerate triangles;
bounds match exactly, area 5.937683 versus OCCT 6.038595 (1.67% difference),
signed volume 0.592487 versus 0.600214 (1.29%). These are diagnostic geometric
checks, not a proof of topology. Evidence: `local/seam-parity-bga-occt`.

Separate source evidence: Würth WE-KI-0603 face #1739 references closed spline
#32 whose five controls alternate between only two positions. It retraces a
straight segment; it must not be repaired as though it were a regular hole.
This finding does not by itself establish whole-file conformance or explain
the independent failing cylindrical face #79.

Freed disk by deleting only superseded generated mesh files from pass 5
(1,253 files, 493,630,272 bytes) and old `final`/pass-1 outputs (4,578 files,
5,409,207,138 bytes). Source corpora, manifests, reports, metrics, logs and
frozen workers remain; current pass-6 and repair-check evidence is retained.

### Full pass 7 and rational-curve quotient recurrence

Completed all Würth files then all KiCad files with frozen seam-parity worker.
Würth: 6,896 ok, 234 invalid meshes, 195 tessellation errors, 3 source crashes.
KiCad: 7,102 ok, 129 invalid meshes, 20 tessellation errors. Exact path/hash
coverage and shard-worker consistency verified; per-file exports are under
`.amp/in/artifacts/repair-pass7/`. There are 114 new Würth failures relative to
pass 6 (101 invalid meshes, 13 tessellation errors) and six new KiCad failures
(one invalid mesh, five tessellation errors). Rerunning every one with the
pre-seam `repair-anchor-worker` reproduces the same status. Thus the newly
failing files predate seam cancellation; the changed spline arithmetic/trim
path remains under investigation. Evidence: `local/*-pass7-regression-anchor`.

Found an independent algebraic error in the rational curve quotient recurrence:
the i-th weight derivative was multiplied by C[k-1] instead of C[k-i]. For
x(u)=u/(1+u²), the old second derivative at zero is -2 instead of 0. The analytic
regression fails before and passes after the one-index repair; workspace tests
and all eight checkpoints pass. No tolerance or fallback changes.
Evidence: `local/repair-quotient-check`, worker `local/repair-quotient-worker`.

Remaining dome inverse failure is now localized to rational derivative
roundoff: near the pole the analytic constant z coordinate acquires an apparent
u tangent of order 1e-16. Normalizing this tiny tangent mixes a 2.9e-11 normal
offset into the stationarity test. A Cartesian-relative homogeneous evaluation
is being investigated rather than weakening convergence. Diagnostic probes
were removed after capture (`/tmp/dome-inverse-probe.log`).

WR-MJ area error now has direct triangle evidence, not only grid-density
inference: surface #171 alone has area 983.719 versus OCCT 62.408. Its largest
triangle contributes 27.149, spans radial parameters 0.484448–0.816114, and
bridges the narrow bend cluster near 0.66. The five largest triangles contribute
128.123, already twice the correct whole-face area. Retained raw chart points,
triangle indices and XYZ positions: `local/wrmj-triangle-probe.log`; quantified
triangles: `local/wrmj-triangle-quantification.json` (angular coordinate is in
radians; divide by 2π for this surface's raw u). This defect remains unfixed.

Preparatory evaluator refactor separates basis accumulation of local control
differences from restoring the origin. The same four accumulation kernels
continue to own curve/surface position/derivative evaluation; no numerical
behavior changes yet. This lets rational callers provide a Cartesian-relative
homogeneous difference without copying knot/basis logic. Workspace tests and
eight checkpoints pass. WR-MJ's 50,254 binary STL facets are byte-identical as
a multiset before/after; aggregate summation order alone differs between runs.
Evidence: `local/repair-relative-refactor-check` and frozen worker.

Rational positions and derivatives now accumulate homogeneous controls relative
to the Cartesian anchor and restore translation only after quotient evaluation.
This prevents large constant Cartesian coordinates from leaking through weight
derivatives into tiny tangents. Both new constant-coordinate regressions fail
before the change (surface derivative z=1.45e-16 instead of zero) and pass after;
workspace tests and eight checkpoints pass. Both WL-SMRW-1206 dome variants
now pass. Evidence: `local/repair-rational-relative-{check,dome}`.

Rerunning all pass-7 regressions with this worker resolves one Würth and one
KiCad file; 100 Würth invalid meshes and 13 tessellation errors remain, as do
one KiCad invalid mesh and four tessellation errors. The crystal IQXC-26 has
zero measured f64 degenerates but 28 f32 STL degenerates. This is not resolved
by rational arithmetic. Do not blindly delete exported facets to declare the
models fixed: exact coordinate quantization can collapse a closed tetrahedron
into opposite coincident triangles that pass the current strict STL gate while
losing its volume. Source correctness, output representability and topology
must remain separate findings. Oracle consultation confirmed that an export-only
quotient would require an explicitly lossy triangle-soup contract, not a claim
of STEP-solid preservation. No filtering or relaxed harness gate was added.
Removed generated `target/debug/incremental` caches to free about 3 GiB;
retained source and corpus evidence.

### Pass 8 and exact polyline reduction

Full pass 8 (before polyline reduction): Würth 6,891 ok, 238 invalid meshes,
196 tessellation errors, 3 source crashes; KiCad 7,094 ok, 129 invalid meshes,
28 tessellation errors. Exact 7,328/7,251 coverage and shard worker identities
verified. Per-file exports: `.amp/in/artifacts/repair-pass8/`. Compared to pass 7,
seven new Würth lowering errors and two invalid meshes appear, plus ten new
KiCad crossing failures. Rational arithmetic improves the two SMRW domes but
regresses the SMCW dome and other pole cases; this is still unresolved.
Deleted only superseded pass-6 generated meshes (675 files, 408,821,773 bytes).

Direct f64/f32 triangle probes on IQXC-26 show the short-trim repair introduced
eight redundant samples on exactly straight microscopic spline segments. The
28 exported degenerate facets lie on four planar faces (#1683/#1139,
#1687/#1143, #1691/#1147, #1717/#1161). These are not necessary curved samples:
the redundant collinear points divide one straight edge into unrepresentable
f32 intervals. Evidence: `local/crystal-quantization-probe.log`.

Reduce the sampled polyline in place by removing only exactly collinear points
between their neighbours, with robust predicates in all three projections.
Corners, reversals and any nonzero curvature remain; outside the predicate
exponent envelope samples remain unchanged. This preserves the represented
polyline as a point set and traversal, unlike snapping or dropping output
triangles. The straight-cubic short-trim regression fails before and passes
after; curved short-trim, reversal and tiny-curvature tests pass, as do workspace
tests and eight checkpoints. IQXC-26 now passes with no exported degenerates.
Evidence: `local/repair-straight-{check,crystal}`.

On the 114 pass-7 Würth regressions, exact reduction brings 27 to ok, leaving
69 invalid meshes and 18 tessellation errors; KiCad's six-case cohort has one
ok, one invalid mesh and four tessellation errors. Error-stage shifts are not
equivalent to fixes. Full-corpus verification remains necessary after this
change. Evidence: `local/{wurth,kicad}-straight-regressions`.

The remaining pole failure was a contract mismatch: chart construction already
identifies a collapsed iso-boundary under declared representation uncertainty,
but lowering still asks Newton to recover a unique angular coordinate there.
Such a coordinate does not exist at a pole. Preserve the identified pole and
its declared uncertainty as chart-associated data; matching points map directly
to the polar chart origin. All other points still use the unchanged inverse.
This is neither a solver fallback nor a relaxed global convergence threshold.
The near-pole regression fails before the change, and tests retain a non-pole
point away from the uncertainty region. Workspace tests, eight checkpoints and
all 12 selected dome/infrared cases pass, including every pass-8 LED pole
regression. Evidence: `local/repair-pole-{check,dome}`.

KiCad CP_Elec_5x5.4 crossing RCA: bounded degree-(2,13) surface #116 has endpoint
iso-curves matching within its declared 2e-6 length uncertainty, but both spline
closure flags are false. The face repeats EDGE_CURVE #90 as its seam. Independent
inverse projection maps both occurrences to the same endpoint of the bounded
domain, collapsing the rectangular trim into two retraced lines. Floating-point
arithmetic changes only decide which seam side wins; they are not the cause.

After a separately verified iso-boundary extraction refactor, chart selection
also recognizes matching geometric endpoint boundaries. A positive-weight
control-hull test with matching normalized rational weights bounds separation
along the complete iso-curves, not a sparse sample test. This only selects the
continuous chart; it does not rewrite closedness flags, knot periodicity or
bounded Newton domains. Tests reject resolvable gaps and different rational
bases, and distinguish zero from declared uncertainty. Workspace tests and
eight checkpoints pass; the CP_Elec crossing is resolved.

The capacitor is **not geometrically certified**: OCCT comparison reports
area 169.176084 versus 160.459846 (5.43% difference), with matching bounds;
signed volume 103.280538 versus 106.651604. Retain this approximation defect
alongside WR-MJ rather than enlarging the oracle tolerance. Evidence:
`local/repair-closed-{check,capacitor,capacitor-occt}`.

### Pass 9 and topological seam qualification

Full pass 9 covers all 7,328 Würth files (6,758 ok, 203 invalid meshes,
364 tessellation errors, 3 source crashes) and all 7,251 KiCad files
(7,115 ok, 125 invalid meshes, 11 tessellation errors). Per-file observations,
face IDs and coverage evidence: `.amp/in/artifacts/repair-pass9/`.
KiCad disk recovery retains 6,461 original atomic ok results and reruns the
remaining 790 with the identical frozen worker; the final path/hash coverage
is exact. The original eight-shard union is not asserted for this recovery.

Geometric endpoint proximity alone incorrectly closes thin open spline patches.
Require a repeated, oppositely oriented EDGE_CURVE in the source boundary before
inferring closure for an otherwise bounded spline. Explicit closure flags are
unchanged. This recovers 99 of the 164 pass-9 Würth regressions; 63 tessellation
errors and two invalid meshes remain in that cohort. It does not establish a
complete seam-axis interpretation, and the remaining regressions need further
RCA. Workspace tests, eight checkpoints, two resistors and the capacitor pass.
Evidence: `local/wurth-topological-seam-regressions` and
`local/repair-topological-seam-{check,resistor,capacitor}`.

Disk maintenance removes superseded generated meshes only: 1,167 baseline
meshes (about 1.58 GB), 429 pass-7 Würth meshes (446 MB), 149 pass-7 KiCad
meshes (38 MB), and about 661 MB of incremental compiler caches. 1,105 targeted
STLs are losslessly compressed (3.78 GB original). Older baseline/final/pass-1
through pass-6 worker logs are losslessly gzip-compressed: append `.gz` to old
log links or decompress them. Sources, manifests, results, metrics and frozen
workers remain available; pass-7 and newer logs remain plain text.

### Exact crossing classification

Murata L_Radial_D24.4 plane #154 contains a crossing at segment parameters
about 0.979 and 2.7e-14. Tangent circular trims generate nearly collinear
polygon chords; rounded coordinates cross, but the old 1e-10 endpoint epsilon
and 1e-12 determinant cutoff hide the crossing from preprocessing. CDT's exact
predicates then correctly reject the unprocessed constraint arrangement.

Replace these cutoffs with adaptive exact orientation signs. Signed triangle
areas also supply the intersection parameters without subtracting two nearly
equal products in an ordinary determinant. Scale, near-endpoint, near-parallel,
shared-endpoint and disjoint-segment tests pass, as do workspace tests and eight
checkpoints. All eleven pass-9 KiCad crossing cases now have zero face errors.
All eleven still fail mesh validation: nine have only f32-degenerate facets;
the two Bourns models also contain six f64-degenerate facets each. This repairs
crossing classification, not the final meshes. Evidence:
`local/repair-predicate-{check,crossings}`. The pre-existing splitter iteration
limit and representable intersection construction remain separate concerns.

### Polyline ownership refactor

Move exact polyline reduction from the NURBS sampler into the sole STEP edge
consumer, retaining reduction before endpoint replacement for this refactor.
The sampler now supplies samples and the edge builder owns their reduction;
there is no new wrapper or configuration branch. Move reduction tests with
the responsibility. Workspace tests and eight checkpoints pass with unchanged
vertex/triangle/error counts. WR-MJ's complete STL facet multiset is byte-identical
between frozen workers (digest `ba285c587e894f0118b64ee0118bd1107d578ccd165af774825e86712e6be4b9`).
Tiny area-sum changes in three checkpoint reports arise from ordering, not new
geometry. Evidence: `local/repair-polyline-owner-check`.

The 63 residual tessellation regressions all resolve to planar two-edge faces
(120 failed faces), not spline-chart construction. WCAP-FTXH plane #406 is a
concrete example: its cubic #46 is a straight line at y=-5.995, while its shared
topological endpoints lie at y=-5.9975. Reducing the sampled spline before
installing these endpoints discards the interior curve. Endpoint replacement
then turns it into the same segment as the other bound edge, cancelling the
whole boundary. Distinct control sequences alone are not proof of a valid lens;
the source offsets plus the actual operation order establish this defect.

Install topological endpoints before exact reduction. Remove the degree-one
sampling bypass so all spline degrees follow the same sample/endpoint/reduce
pipeline. The regression fails before and passes after for degrees one and
three, in both directions. Straight on-curve trims still reduce to endpoints;
no coordinate tolerance or model-specific branch is added. Workspace tests and
eight checkpoints pass. The 164-file regression cohort is now 163 ok, one invalid
mesh: all 63 planar tessellation failures clear. DSUB 61803729321 still has one
f64 and ten f32 degenerate facets; its upstream cause is not established here.
Evidence: `local/wurth-endpoint-regressions`, `local/repair-endpoint-check`, and
`.amp/in/artifacts/topological-seam-remaining/` (per-file source observations and
before/after results; the two original invalid meshes remain RCA-unresolved).

### Pass 10 and the surface distance Hessian

Full pass 10 uses the frozen endpoint worker for both corpora, Würth first:
7,328 Würth files (6,922 ok, 276 invalid meshes, 127 tessellation errors,
3 source crashes), then 7,251 KiCad files (7,115 ok, 136 invalid meshes,
zero tessellation errors). Exact path/hash coverage, complete eight-shard unions
and common worker digests are verified. Per-file observations and reproduction
commands: `.amp/in/artifacts/repair-pass10/`. Failure STLs are losslessly gzipped
after their atomic result files are complete.

The surface inverse uses a Gauss–Newton approximation which omits the residual
times second-derivative terms in the squared-distance Hessian. Normal offsets
can then make first-order convergence arbitrarily slow. An analytic extruded
parabola near its curvature center exhausts the old iteration limit. Include
the full derivative-normalized Hessian, with a scaled eigenvalue shift to retain
a positive definite descent model when curvature is negative or singular.
The analytic regression and all 28 NURBS tests pass; workspace tests and eight
checkpoints pass. No convergence tolerance or iteration limit is increased.

Of the 54 pass-10 Würth files with lowering errors, 11 now pass and two reach
mesh validation; 41 retain tessellation errors. WPCC-RX 760308102210 has one
additional face error and needs further investigation. These are not all
resolved by the Hessian: DSUB 216612013 still fails at a closed-domain seam.
Temporary probes are removed; retained evidence is in
`local/repair-hessian-{lowering,check}` and `local/dsub-{inverse,hessian}-probe.*`.

Source analysis of the 137 canceled faces (61 files) identifies 49 faces in
33 files with distinct parallel LINE offsets, 47 faces in 14 files with identical
geometric curves, and 41 faces in 19 files not resolved by exact pair comparison.
File groups overlap. Evidence: `.amp/in/artifacts/pass10-canceled-faces/`.
Unlike the earlier endpoint-only hypothesis, this analysis examines infinite
LINE origins/directions and uses exact decimal predicates. It does not infer
zero area from two vertices or curvature merely from distinct spline controls.

### Bounded projection and seam representatives

DSUB 216612013 surface #192234 fails at (3.967504898759, 5.42,
-2.484510124102), seeded at (0,0). The source is geometrically closed, but
wrapping the local minimizer across its rounded knot endpoints prevents it from
accepting an endpoint minimum. STEP closedness is not periodicity. OCCT's
`StepToGeom::MakeBSplineSurface` infers periodic representation from multiplicity
structure, not UClosed/VClosed; `GeomAPI_ProjectPointOnSurf` passes finite bounds
to `Extrema_GenExtPS`, whose local root search is box-constrained and whose
candidate minima include domain edges/corners. Reference repository:
https://github.com/Open-Cascade-SAS/OCCT, files `StepToGeom.cxx`,
`GeomAPI_ProjectPointOnSurf.cxx`, `Extrema_GenExtPS.cxx`.

Use bounded local projection for curves and surfaces alike. When the nearest
sample lies at a closed seam, enumerate its alternate endpoint representatives
and select the bounded result with least squared distance. This is a defined
candidate set, not a failure-triggered alternate solver; all candidates use the
same minimizer. The initial bounded-only experiment correctly retained endpoint
minima but failed the opposite-side circle test because the geometric sample
tie picked the wrong seam representative. Enumerating both fixes that defect.
Curve seed samples also now exclude intervals outside the active knot domain.

The synthetic rounded-seam regression fails before and passes after; tests
cover both sides of a closed circle, domain endpoints and restricted seed
domains. Workspace tests and eight checkpoints pass. Of the 54 lowering-error
files, 28 now pass, ten reach mesh validation, and 16 retain tessellation errors
(14 files with 17 lowering failures, plus two with canceled boundaries).
No distance tolerance changes or coordinate snapping are involved. Evidence:
`local/repair-bounded-all-{lowering,check}`. Intermediate surface-only evidence
is retained separately as `local/repair-bounded-{lowering,check}`.

### Directed closed-curve trims

WR-SMB 61612202121306 face #486 uses EDGE_CURVEs #490 and #504 on identical
closed splines #495/#505. The edges have opposite endpoint order but both declare
same_sense=true. They therefore cover complementary directed arcs. The previous
sampler ignored same_sense for non-loop spline edges and sampled the same arc
twice; cancellation correctly exposed the earlier wrong curve discretization.
Whole-loop edges also started at the first knot rather than their actual vertex,
omitting part of the curve when that vertex was away from the knot seam.

Represent the directed trim as one or two bounded parameter intervals, joining
at the closed seam once. Both partial arcs and whole loops use the actual
projected start vertex. Square-curve regressions exercise complementary arcs,
both directions, and whole loops starting midway along an edge; they fail before
and pass after. Workspace tests and eight checkpoints pass. Four of the 14
identical-curve cohort files now reach mesh validation; ten still have face
errors. WR-SMB's six canceled faces clear but two other faces now expose lowering
failures. Remaining CIRCM12 cancellations are torus seams, not this curve bug.
All eleven KiCad crossing-cohort files still reach mesh validation with no face
errors. Evidence: `local/repair-closed-trim-{identical,smb,check,kicad}`.

### LINE geometry and shared topology

The 49 affected faces in 33 files use distinct parallel LINE origins but shared
topological endpoints. The old LINE branch ignored origin and direction and
replaced both curves by the same chord. Project the trim endpoints onto the
infinite line and represent the result as a degree-one spline, using the
existing sample/shared-endpoint/reduction pipeline. Remove the endpoint-only
Curve variant. No proximity threshold or model-specific branch is introduced.

The offset-line regression fails before and passes after. Workspace tests and
eight checkpoints pass. All 33 files clear their face errors: 32 pass; WE-TPC
8012 retains invalid mesh facets. All eleven KiCad crossing-cohort models still
reach mesh validation but remain invalid meshes, not complete repairs. Evidence:
`local/repair-line-{offsets,check,kicad}` and `/tmp/line-tests.log`.

### Complete torus topology

CIRCM12 models encode complete ring tori using either a VERTEX_LOOP or opposing
uses of the same EDGE_CURVEs. These are seams, not physical trims. Planar contour
cancellation cannot represent a compact surface without boundary. Accumulate
oriented source edge uses across all face bounds, respecting bound orientation;
when no physical boundary remains, tessellate the complete ring torus on a
periodic grid with shared wrapped indices. This decision precedes projection
and is not a fallback after a triangulation failure. Other surface domains keep
their existing trimmed path; spindle/lemon radius validation is unchanged.

Regressions exercise vertex-only bounds, opposite seam uses within one loop and
across two bounds, both face senses, outward winding, two oppositely directed
uses of every mesh edge, Euler characteristic zero, and area within 1% of the
analytic torus. Workspace tests, the new STEP integration test, and eight
checkpoints pass. All eleven previously failing CIRCM12 models clear face
errors: three pass and eight still have invalid mesh facets elsewhere.
Evidence: `local/repair-complete-torus-{models,check}`.

### Surface parameter representability

UMRF 636101111001/112001 and all six RSTV switches converge to a parameter whose
full Newton correction is less than half an f64 ULP. The line search then cannot
move and reports a lowering failure. For UMRF, u=4.709888977781001 and the
correction is 3.8586e-16. Apply the existing curve solver's representability
termination to surfaces too: test the full Newton correction before line
search, never a repeatedly halved trial. No tolerance is relaxed.

An extruded short-line regression fails before and passes after, checks that a
resolvable first step occurs, and compares both adjacent representable
parameters. Workspace tests and eight checkpoints pass. The 54-file lowering
cohort changes from 25 ok / 14 invalid / 15 face errors with the current torus
worker to 27 ok / 20 invalid / 7 face errors. Both UMRF models pass; all six RSTV
models clear lowering errors but retain invalid facets. Temporary instrumentation
is removed. Logs: `local/inverse-probe-lowering`; results:
`local/repair-surface-ulp-{lowering,check}`.

### Bounded surface trial steps

CMANC-XL 7848053201, CMBNC-TypeXL 7448053201, and FIRM 1770105311 fail
immediately when negative distance curvature makes the shifted Hessian request
parameter steps of order 1e8–1e9. Forty halvings cannot reach the local descent
region. Start line search with travel limited to one normalized parameter
domain, retaining its direction and the existing convergence tests. An analytic
high-curvature extruded parabola reproduces the old failure and verifies the
new solution's stationarity. Workspace tests and eight checkpoints pass.
All three named models now pass. The 54-file lowering cohort is 30 ok, 20
invalid meshes and four files with face errors. SMB, WPCC-RX/TX and OLRM remain.
Evidence: `local/repair-surface-step-{lowering,check}`.

### Pass 11 regression investigation — do not treat targeted passes as completion

Full pass 11 (torus worker, before the two later surface solver fixes) processes
all 7,328 Würth inputs, then all 7,251 KiCad inputs. Würth: 6,664 ok, 572 invalid
meshes, 89 face errors, three source crashes. KiCad: 7,109 ok, 140 invalid meshes,
two face errors. Exact path/hash coverage and common worker digests are verified;
failure STLs are compressed after result publication.

There are 347 previously passing Würth regressions (296 invalid meshes, 51 face
errors). The initial suspicion that LINE projection caused most is contradicted
by archived-worker bisection: 229/230 examined failures already occur before
the LINE change, mostly after directed closed-curve trimming. CNSA-1210 has a
projected loop start 3.7279626446548087e-7 from its smooth parameter seam. Eight
samples in that tiny clipped span yield five adjacent f32 collisions and 15
exported degenerate facets. Its shared vertex is 1.8428e-5 off-curve, so the
endpoint-induced bend must not be discarded with those redundant samples.
Evidence: `.amp/in/artifacts/closed-trim-regressions/`. This sampling defect is
not yet fixed. Uncertainty-based LINE snapping is rejected: 25/33 known lens
models have real resolved offsets smaller than their declared uncertainty.

Move directed-interval sampling and concatenation into SampledCurve so the
sampler owns the complete trim, not isolated fragments. This is a behavior-only
preserving refactor before changing sample allocation. Workspace tests pass;
CNSA-1210's STL is byte-identical before/after with one worker thread. Updated
full pass-11 per-file diagnostic exports: `.amp/in/artifacts/repair-pass11/`.

### Sampling density belongs to the complete trim

Allocate samples proportionally to the clipped fraction of the original knot
span, with the complete directed trim length as the upper bound on that span.
A genuinely short curved edge retains eight subdivisions, while a tiny seam
fragment of a long loop receives one. Preserve every knot/corner and both trim
endpoints; no coordinate snapping or facet deletion occurs. The periodic-cut
regression fails before and passes after; the short-parabolic-trim and corner
tests continue to pass. Workspace tests and eight checkpoints pass.

CNSA-1210 now has zero face errors and zero f32 degenerates. Of the 347 pass-11
Würth regressions, 110 now pass and 237 retain invalid facets; all face errors
clear in this cohort. Two of six KiCad regressions pass, four retain invalid
facets. Evidence: `local/repair-curve-density-{regressions,check,kicad}`.

The remaining WCAP-AI3H-P10D22L30 facets expose a distinct sampling artifact:
the shared endpoint and a smooth periodic seam sample differ by about 1e-12
and become identical in f32, even without redundant subdivisions. Source
curve #4785 has simple non-clamped knots and three exactly repeated closing
controls, not a sharp seam. Faces #4778/#5927 are affected. Temporary probes
are removed; retained trace: `local/cap-density-probe.log`. This is unresolved.

### Smooth periodic cut sample placement

Separate parameter scheduling from curve evaluation first (commit 5a67935).
Workspace tests pass and WCAP-AI3H-P10D22L30's STL is byte-identical across that
refactor. Then identify a smooth periodic cut from repeated controls, simple
cut-end knots and the translated knot sequence, not from the closed flag alone.
Balance the cut sample between its parameter neighbors and evaluate that point
on the curve. Keep the topological endpoints and all genuine corner samples.
This changes sampling placement, not source coordinates or exported facets.

Tests cover both travel directions, starts near either cut end, f32 sample
separation, and rejection of mismatched controls, knots and polygon corners.
Workspace tests and eight checkpoints pass. The 347-file regression cohort is
now 200 ok / 147 invalid meshes, with no regressions from the preceding density
fix. All face errors remain clear. The known LINE-offset group stays 32 ok /
one invalid; the 14 closed-curve models are two ok / ten invalid / two with face
errors. KiCad's six-file regression group remains two ok / four invalid.
Evidence: `local/repair-curve-seam-{regressions,check,lines,closed,kicad}`.

Projection research continues separately. An 80-digit analysis distinguishes
the original Cartesian-plus-weight STEP surface from its rounded homogeneous
control net: WPCC's apparent derivative jumps are dominated by homogenization
and evaluation error, not a macroscopic source crease. OLRM is at a clamped
endpoint/pole, not an interior repeated knot; instrumented iterations leave the
pole, activate a tiny second derivative direction, and cycle back. Do not
interpret the initial f64-only analyzer as a proof of smoothness or a complete
RCA. Corrected evidence: `.amp/in/artifacts/knot-projection-rca/`, including two
80-digit evaluations, and `local/olrm-iterations.log`. No projection repair is
claimed for these four remaining files.

### Active knots define a periodic seam

Exclude the two exterior knots from the translated-knot test: neither enters
an active basis function. Some exporters repeat the adjacent exterior knot
instead of extending the periodic sequence there. The regression changes those
two values and verifies identical points and first/second derivatives throughout
the active domain, while still rejecting a changed active knot. The targeted
test and `cargo test --workspace` pass (`/tmp/active-knots-workspace.log`).
This is a structural correction, not a relaxed geometric tolerance.

The separate query-relative surface-evaluation experiment does not yet repair
the remaining four lowering models. A like-for-like six-model comparison keeps
UMRF 111001/112001 passing only with mixed per-axis convergence; OLRM, WPCC-RX/TX
and SMB remain failures (`local/repair-{seam,relative-mixed}-remaining`). These
experimental changes are not yet committed or counted as recovered files.

### Query-relative surface derivative jets

Add a derivative evaluation whose zeroth entry is the residual relative to a
query, before restoring world coordinates. Nonrational surfaces retain their
anchored sum; rational surfaces translate homogeneous controls with fused
multiply-add before rational evaluation. Ordinary derivatives use the same
path with a zero reference. This avoids losing resolvable residuals when a
small surface is far from the origin, without a second evaluation algorithm.
The translated-plane regression verifies residuals below a world-coordinate
ULP for both surface types. Workspace tests pass. This commit exposes the
evaluation primitive only; solver integration is a separate logic change.

### One-sided surface derivatives at knot boundaries

Allow derivative evaluation in an explicit knot cell, reusing the existing
basis-derivative algorithm. Ordinary evaluation still selects its original
cell. A two-plane crease regression verifies coincident positions, distinct
left/right tangents, and unchanged default right-side selection. Workspace
tests pass (`/tmp/cell-derivatives-tests.log`). Solver use is separate.

This is needed by the remaining WPCC failures: `local/box-probe-remaining`
records iterates alternating between u=0.5 and u=0.49999999999999106. The
one-sided unit gradients have opposite signs, but the solver assumes one
smooth quadratic across the knot. This observation is about the represented
surface, not proof that the original decimal surface has a macroscopic crease.

### Bounded distance quadratics preserve poles

OLRM surface #114 loses a tangent direction at its clamped pole. Independently
normalizing that vanishing Jacobian column makes the angular step discontinuous
and unbounded. The 100-digit source evaluation locates an interior stationary
point at u=0.00112246588893905226, v=0.51055254769788338, 1.51367736e-13 inside
the pole (`local/olrm_high_precision.py`, `local/olrm-high-precision.out`).

Replace the normalized/damped Hessian with the actual distance quadratic in
fixed unit-domain coordinates. In two dimensions its box-constrained minimum
is among the stationary interior point and four edge minima/corners. Compare
candidate quadratic values in factored form so a thin direction is not erased
by another direction's common contribution. A shrinking box trust region
handles negative curvature without a spectral shift or model-specific cases.
Use query-relative residuals consistently and apply convergence per direction.
Only the full-domain step, never a shortened trust step, permits representability
termination. The intermediate line-search version is rejected: a direction
chosen for negative curvature need not have a negative linear slope, and that
version introduces 20 face errors in the regression cohort.

Workspace tests pass, including a vanishing-tangent projection, a quadratic
with a 1e-32 thin direction, and an indefinite quadratic. Eight checkpoints
pass. The 54-file lowering cohort is 41 ok / 11 invalid / two face errors:
OLRM now passes, SMB has zero face errors and zero f64 degenerates but retains
f32-degenerate facets, and WPCC-RX/TX remain unresolved. The 347-file cohort is
205 ok / 142 invalid, exactly the active-knot worker's statuses and no face
errors (`local/repair-box-trust-{lowering,check,regressions}`).

OLRM's bounds match OCCT; areas are 3041.8511 versus 3031.1182 (about 0.35%).
Foxtrot's exported mesh has no degenerates, while OCCT's has 64. Therefore the
reference run reports `oracle_invalid_mesh`, not a clean equivalence pass:
`local/repair-box-trust-olrm-reference`. No bad facets are discarded.

Separate distance-model construction from trust-region iteration before
introducing knot-cell traversal. This removes duplicated direction handling
and keeps the model's residual, quadratic, bounds and convergence state
together. Workspace tests pass before the new knot-projection regression is
added. OLRM's STL triangle-record multiset is exactly identical to the trust
worker (only ordering differs); all six focused model statuses are unchanged.
Evidence: `local/repair-distance-model-remaining`, `/tmp/distance-model-tests.log`.
KiCad's six regression cases remain two ok / four invalid with the trust worker.

### Surface projection respects knot cells

Bound each distance model to its actual knot cell. At an interior knot, build
the same model for every incident nonempty cell and require all one-sided
stationarity conditions before convergence. Preserve an exact knot when a
trial reaches its bound instead of relying on a normalization roundtrip.
This replaces the invalid smooth-across-all-knots assumption; there is no
WPCC-specific tolerance, snapping or fallback.

The two-plane-crease projection test fails before this change and passes after.
It checks both a minimum on the crease and travel through the crease, starting
on either side and directly at the knot. Additional coverage excludes repeated
zero-width and inactive exterior intervals. Workspace tests and eight corpus
checkpoints pass. All 54 lowering cases now have zero face errors: 43 pass and
11 retain invalid exported meshes. Both WPCC-RX 760308102210 and WPCC-TX
760308101103 pass. The 347 Würth and six KiCad regression statuses are exactly
unchanged from the trust worker (205/142 and 2/4 ok/invalid respectively).
Evidence: `local/repair-knot-cell-{lowering,check,regressions,kicad}` and
`/tmp/knot-projection-before.log`. All 142 invalid meshes in this Würth
regression cohort have zero measured f64 degenerates; their failures occur
after f32 export. This does not yet establish whether each is avoidable
sampling or an intrinsic output-representation limit.

### Complete pass 12 — still not complete repair

Run all 7,328 Würth files, then all 7,251 KiCad files with the frozen
`repair-knot-cell-worker` (source commit 9e7db4a; SHA-256
`0370db3bd880bac3d60ff74df5003fa23d1f9a86f7ac8b6c011448bc165ce862`).
Eight disjoint shards per corpus verify every path/hash, the exact merged
manifest set, counts and common worker digest. Failure meshes are compressed
only after atomic case-result publication. Results:

| Corpus | Pass | Invalid mesh | Face errors | Source worker errors |
| --- | ---: | ---: | ---: | ---: |
| Würth | 6,958 | 355 | 12 | 3 |
| KiCad | 7,096 | 155 | 0 | 0 |

Of invalid meshes, 16 Würth and 15 KiCad files already contain f64 degenerate
triangles; the other 339 and 140 fail only after f32 export. Compared with
pass 11, eight previously passing Würth files and 17 KiCad files now fail the
exported-mesh gate. These regressions are not waived. Fresh per-file diagnostics,
provenance and reproduction commands: `.amp/in/artifacts/repair-pass12/`.
Full logs: `local/{wurth,kicad}-repair-pass12`; orchestration log:
`local/repair-pass12.log`.

Six of the remaining Würth face-error files are confirmed invalid source
geometry, not a radius-boundary implementation omission. Both OLLT EE13_6_6
variants and OLSTM EE13_7_6 variants 750370423, 750810014, 7508110341 and
7508110351 declare surface #36 as DEGENERATE_TOROIDAL_SURFACE with equal radii
0.127/0.127. Its formal WR1 requires **major_radius < minor_radius**, not <=.
The current strict rejection is correct; do not silently reinterpret this
entity as an ordinary horn torus. Reference:
https://www.steptools.com/stds/stp_aim/html/t_degenerate_toroidal_surface.html.
Together with the two mislabeled Parasolid files and the missing #0 reference,
there are nine confirmed invalid-source inputs. The six other face-error files
have two boundary-cancellation cases and four CrossingFixedEdge cases; their
upstream causes remain under investigation.

The independent WR-MJ geometry defect persists despite passing the mesh gate:
surface #171 area is now 983.7192077441117 versus OCCT 62.408. New trace:
`local/wrmj-pass12-probe.log`. Earlier raw-knot samples alone yield 535.624,
so merely adding unconstrained points is insufficient.

Before adding internal knot constraints, investigated Spade's
`add_constraint_and_split`. It does not propagate user edge parity on splits
and may reroute through an existing vertex. Oracle tracing disproves a proposed
"internal first, reject original boundary crossings" invariant: restarted legs
can cross a parity-true edge that the original segment did not cross, corrupting
classification. Do not adopt that API here. Keep vertex-free `try_add_constraint`
and explicitly split tagged input constraints; internal constraints must not
toggle boundary parity. Do not skip a rejected internal edge to manufacture a
passing face. This design is not implemented yet.

### Remove uncertainty-radius pole snapping

Direct tracing disproves the earlier hypothesis that SMA 60312202114511 loses
its caps through a two-pole chart or missing degenerate edges. Surface #1229
already selects the correct single-pole polar chart. Its declared uncertainty
is 0.5 mm; the special pole-lowering branch maps every one of its 82 boundary
samples to (0,0), including real points more than 0.4 mm from the pole.
The same branch destroys caps #1268, #1307 and #1347.

Remove the separate pole position/uncertainty state and proximity branch.
All points now use the distance projection followed by the chart map. This
reduces code and does not introduce another snapping threshold. The revised
pole regression checks actual on-surface points inside the source uncertainty
and fails before the fix. Workspace tests pass afterward; all eight corpus
checkpoints pass. In the 36-file Würth investigation cohort, only SMA changes
status: all four face errors disappear, with zero f64 degenerate triangles.
It still has eight f32-degenerate exported triangles, so remains invalid_mesh.
Evidence: `local/wurth-repair-pass12/sma-pre-cancel.log`,
`local/repair-project-pole-{investigate,check}`, `/tmp/pole-before.log`,
`/tmp/pole-tests.log`; frozen worker `local/repair-project-pole-worker`.

TBL 691404910001B is not proven invalid by the earlier endpoint comparison.
Enumerating stationary points of each cubic's squared-distance polynomial
places both declared vertices #7509/#7510 within 1.99e-8 mm of both curves
#827/#828, versus source uncertainty 0.005 mm. EDGE_CURVE may trim an interior
portion of its underlying curve. OCCT face #13017 (imported face 216) reports
UnorientableShape and its wire reports NotClosed in the face, but both edges
report NoError. OCCT uses the full spline ranges and inflates vertex tolerance
to approximately 0.00655 mm. This is evidence of an ill-conditioned imported
wire, not proof that the source incidence violates its tolerance. Keep its
canceled-boundary implementation limitation open. Reproducible incidence and
BRepCheck evidence: `local/wurth-repair-pass12/tbl691404910001b-*`.

### Distinguish refinement constraints from trim boundaries

The CDT now accepts `(start, end, boundary)` records through
`new_with_constraints`; existing boundary-only constructors use the same path.
Internal constraints lock triangulation edges without changing region parity.
This is the required representation for spline knot-cell refinement, not a
second triangulation implementation. No caller generates the grid yet.
Tests cover a hole, an internal edge inside the hole, overlapping boundary and
internal constraints, subdivision by an existing collinear point, both insertion
orders, and an internal-only triangulation. `cargo test -p cdt`: eight unit tests
and two doc tests pass (`/tmp/internal-constraints-tests.log`).

### Resolve all constraint crossings, not the first hundred

Replace repeated first-crossing searches and the arbitrary 100-intersection
cutoff with a bounding-box sweep, per-edge sorted split records, and shared
intersection vertices. Repeat only to resolve crossings introduced by rounded
construction. Preserve boundary tags on every child; a boundary owns the 3D
interpolation when crossed by an internal constraint. Exact coordinate reuse
avoids manufacturing duplicate intersection vertices. No edge is skipped on
failure, and no triangle is discarded.

All four CrossingFixedEdge files now pass the complete exported-mesh gate:
SMA 60312872112545, TNC 67011042241505, WE-TI-1014 and WE-TIHV-1014. The former
resolver always inserted exactly 100 crossing vertices on the failing faces;
the batch resolver inserts 310, 156, 186 and 186 respectively. The face logs
record the corresponding increased constraint counts. These are real uncapped
arrangements, not error suppression.

Workspace tests pass, including 121 grid crossings and boundary/interior
intersections in both insertion orders. Eight checkpoints pass. The Würth
36-file cohort is now 6 ok / 23 invalid_mesh / 7 tessellation_error, versus
2 / 23 / 11 before this change. The 19 KiCad investigation files retain their
invalid_mesh status. Evidence: `local/repair-cross-batch-{investigate,check,kicad}`,
`/tmp/cross-batch-tests.log`; frozen worker `local/repair-cross-batch-worker`.
The remaining genuine face-processing investigation is TBL 691404910001B;
the other six face-error inputs violate the degenerate-torus formal rule.
Output degeneracy and the independent WR-MJ surface-area defect remain open.

### Complete pass 13 and disk cleanup

Run all 7,328 Würth inputs, followed by all 7,251 KiCad inputs with frozen
`repair-cross-batch-worker`, source 7ae3e88, SHA-256
`236ad84eb3b30404c93ddd5708a00384d9d24066a96fed37746ae3f9a19d624f`.
Exact manifest path/hash coverage and eight disjoint shards are verified.

| Corpus | Pass | Invalid mesh | Face errors | Source worker errors |
| --- | ---: | ---: | ---: | ---: |
| Würth | 6,963 | 352 | 10 | 3 |
| KiCad | 7,096 | 155 | 0 | 0 |

There are 16 Würth and 15 KiCad files with f64 degenerates. All KiCad statuses
are unchanged. Three previously passing Würth files now fail: RCIS 7847225100
has an invalid mesh, and both WL-SMRW-1206dome variants fail inverse projection
on surface #3310. WL-SIQW-3535 changes from invalid_mesh to a projection error
on #2796. These new pole-projection failures must be repaired; the broad
uncertainty-radius snapping must not be restored. Thus the focused-cohort
statement above is superseded: TBL is not the only open processing defect.
Current per-file observations, qualified diagnoses and reproductions for all
520 failures are in `.amp/in/artifacts/repair-pass13/`; generation script:
`local/export_repair_rca.py`. Full evidence: `local/{wurth,kicad}-repair-pass13`.

At the user's request, remove only rebuildable `target/debug/examples` and
`target/debug/incremental`, and losslessly compress 73,042 historical run logs
in both corpora's passes 7–11. Every compressed log passes `gzip -t`; their
paths now end in `.log.gz`. Manifests, case JSON, RCA reports, corpora, frozen
workers, current probes and passes 12–13 remain intact. Free space increases
from 1.5 GiB to about 16 GiB despite the new completed sweep. Use
`CARGO_INCREMENTAL=0` for subsequent diagnostic builds to limit cache growth.

A knot-grid prototype is saved separately as `local/knot-grid-prototype.patch`,
not installed or committed. Internal cell constraints reduce WR-MJ surface
#171 area from 983.719 to 62.285528 versus OCCT 62.408166, but introduce seven
f64 degenerate triangles in the model and fail the existing periodic-band
test. The test exposes identical XYZ with different UV: inverse-projected
boundary coordinates differ by about 1e-14 from exact grid coordinates.
Evidence: `local/wrmj-knot-grid-probe.log`, frozen
`local/repair-knot-grid-probe-worker`, `/tmp/knot-grid-disk.log`.
Do not discard these triangles or waive the test. Reconcile boundary/grid
representatives before CDT while preserving intentional periodic cuts and
exact knot barriers. An oracle proposes projection-error bands in raw parameter
space; its global-position-norm estimate is not yet accepted because it could
erase thin directions. Any such bound must respect the existing componentwise
projection error model and singularities. Prioritize the pass-13 regressions
before resuming this prototype.

### Compare quadratic projection candidates after parameter rounding

WL-SIQW-3535 exposed a coupled quadratic step whose radial component is less
than half a parameter ULP. Rounding removes that component but retains angular
motion in the wrong direction. Compare candidates after mapping them to actual
representable parameters; include the two axis stationary candidates so the
remaining direction can descend independently. Share exact knot restoration
between the convergence model and trust-region trials. This does not relax
stationarity or accept shortened-step stagnation.

The SIQW direct reproduction now has no face errors or f64 degenerates. The
dome reproduction still reaches the iteration limit and remains open. A
synthetic coupled-quadratic regression checks that rounding changes ascent
into descent only when the representable candidates are compared. Workspace
tests pass; the 54-file lowering cohort has 43 ok / 11 invalid_mesh and no
face errors; all eight checkpoints pass. Evidence:
`/tmp/representable-step-workspace-tests.log`,
`local/repair-representable-step-{lowering,check}` and frozen
`local/repair-representable-step-worker`. These are focused checks, not a new
full-corpus result.

### Trust-region radius follows model fit, not just step acceptance

The dome iteration trace shows a two-cycle between u=2.094395102393 and
u=4.188790204786 at fixed v. Each model predicts a distance improvement near
1.9e-31, but actual changes are opposite-signed 8.2e-44. The normal offset
produces a 3e-24 acceptance allowance, so both steps pass while the radius
remains one forever. This is a trust-region update defect, not evidence for
changing the source control net or reinstating pole-radius snapping.

Shrink the radius when actual descent is less than one quarter of predicted
descent, including accepted roundoff-sized steps. Retain the existing
expansion threshold and convergence conditions. A synthetic 120-degree arc
with a large constrained offset reproduces the cycle: the test fails with the
old update and passes with this change from both endpoints. Workspace tests
pass. The combined 83-file lowering/checkpoint/pass-12/pass-13 change cohort
has 61 ok / 22 invalid_mesh, no face errors, and no status regressions versus
pass 13. Both dome variants now pass the complete exported-mesh gate. SIQW
reaches mesh validation but retains an f32 defect.

Evidence: `local/dome-cycle.log`, `/tmp/trust-radius-old-regression.log`,
`/tmp/trust-radius-regression.log`, `/tmp/trust-fit-workspace-tests.log`,
`local/repair-trust-fit-cohort`; frozen worker `local/repair-trust-fit-worker`.

### Share the surface tensor-product accumulator before changing arithmetic

Point and derivative evaluation now use one tensor-product accumulator. Keep
the summation order, anchor choice and coordinate translation unchanged.
Workspace tests pass and the HTAH-D10L10 STL is byte-identical to the preceding
frozen worker with one Rayon thread. This is only a refactor; the HTAH f64
degenerate remains. Evidence: `/tmp/tensor-refactor-tests.log`,
`local/repair-tensor-refactor-worker`, `/tmp/htah-{dump,refactor}.stl`.

The HTAH diagnostic identifies face #4324 / surface #84, an ordinary cubic
curve extruded in z. Boundary points with identical z get v coordinates that
differ by a few ULPs because summing the u basis perturbs the v-only coordinate.
Overlapping collinear trims become a tiny crossing. The inserted intersection
and an original vertex have identical XYZ but different UV, creating the f64
degenerate. Trace: `local/htah-face84.log`. Correct per-axis partition-of-unity
evaluation next; do not weld arbitrary XYZ aliases or discard the facet.

### Complete pass 14 and quantify the remaining f32 defects

All 7,328 Würth files, then all 7,251 KiCad files complete with frozen
`repair-trust-fit-worker`, source 54f5270, SHA-256
`4dc2ea4e9b4159e73af6674215170f05046f4b71839fdd251743d21995842185`.
Exact manifest/hash coverage and disjoint shard unions are verified.

| Corpus | Pass | Invalid mesh | Face errors | Source worker errors |
| --- | ---: | ---: | ---: | ---: |
| Würth | 6,965 | 353 | 7 | 3 |
| KiCad | 7,096 | 155 | 0 | 0 |

Both dome variants pass, SIQW changes from a projection error to invalid_mesh,
and every other status is unchanged. Per-file RCA/reproduction exports for
all 518 failures are in `.amp/in/artifacts/repair-pass14/`. This sweep precedes
the tensor-product refactor and arithmetic work.

A separate frozen diagnostic probes all 478 pass-13 invalid_mesh files with
zero f64 degenerates (338 Würth, 140 KiCad). All hashes match. It finds 5,664
local pre-instance degenerate facets: 4,629 have equal f32 vertices; 1,035 have
distinct but f32-collinear vertices. 4,739 are boundary-only and 925 include
interior samples. Seventy-four Würth files have no local degenerates, so their
remaining placement/export behavior needs investigation. Confirm the current
transformed verdict before attributing each to an instance transform.

The research report calls near-coincident samples redundant, but proximity
and UV separation alone do not establish this. Its proposed f32-based sample
suppression is not adopted. Trace endpoint/knot ownership and actual geometry
before changing sampling; never erase true topology or delete bad facets to
satisfy the gate. Machine-readable per-facet source IDs, coordinates and hashes:
`.amp/in/artifacts/repair-pass14/f32-diagnostics.json`. Original probe logs,
scripts and analysis: `local/f32-rca/`. No diagnostic instrumentation remains
in production code.

### Preserve partition of unity on both tensor axes

Accumulate differences from a local anchor separately in each axis, restoring
the anchor only for derivative order zero. Constant coordinates along an axis
then contribute exactly zero to its derivatives, including mixed derivatives.
The shared point/jet accumulator applies this identity without extrusion or
surface-type branches. A nonrational/unit-weight-rational extrusion test in
both axis orders fails before the change and passes after, checking residual
height independence and exact derivative zeros over 65 parameter samples.

Workspace tests pass. The 95-file Würth cohort has 64 ok / 31 invalid_mesh,
no face errors and no status regressions. HTAH-D10L10 and HTG5-D10L10 both
lose their f64 degenerates and pass the complete exported-mesh gate; TBL
691378100020 also changes from invalid_mesh to ok. USB 692221030100 loses one
of three f64 degenerates. All 15 KiCad f64-defect cohort files still have
invalid exported meshes, but five lose one or more f64 degenerates; three now
fail only after f32 export. Evidence: `/tmp/extrusion-partition-before.log`,
`/tmp/tensor-partition-workspace-tests.log`,
`local/repair-tensor-partition-{wurth,kicad}` and frozen
`local/repair-tensor-partition-worker`.

### Complete pass 15; keep its KiCad regression open

All 7,328 Würth files followed by all 7,251 KiCad files complete with the
frozen tensor-partition worker (source 320b7dd); manifest sets, shards and
worker hash are verified by the sweep and RCA exporter. Würth: 6,968 ok /
350 invalid_mesh / 7 tessellation_error / 3 source crashes. KiCad: 7,095 ok /
156 invalid_mesh. The three focused Würth recoveries are the only Würth status
changes. KiCad Fastron 77A L26.0mm/D10.0mm/P30.48mm regresses from ok to
invalid_mesh and must be repaired. Reports for all 516 current failures:
`.amp/in/artifacts/repair-pass15/`.

Losslessly compress historical pass-12 through pass-14 run logs after their
reports are published; all compressed files pass `gzip -t`. Their paths now
end in `.log.gz`. Current pass-15 logs, original STEP files, frozen workers,
metrics and investigation evidence remain intact. Free space returns from
6.8 GiB to approximately 15 GiB.

### Rational boundary investigation and rejected shortcut

CIRCM12 643250100405 face #3824 / surface #4036 has three exactly collinear
XYZ vertices but a spurious u=8e-26 on one. Its nonbinary homogeneous weight
introduces a tiny z warp into an extrusion. An oracle's arithmetic replica
agrees with the Rust first step within 2%; retaining Cartesian controls would
remove that preweighting error, but quotient arithmetic still perturbs mixed
derivatives. Storage reform alone is not a complete projection fix.

An uncommitted experiment compares a quadratic trial with actual residuals
at nearby cell bounds, within the trust region, without a distance epsilon.
It removes all f64 defects from the five 6432501004/6xx variants and makes
13 KiCad axial-inductor cases pass, including the pass-15 regression. However,
it initially introduces 13 Würth projection errors: selecting an unchanged
bound discards a valid descent trial. The oracle's suggested early success on
that unchanged point is rejected because a shortened-step tie cannot prove
stationarity. Require predicted descent for a bound candidate instead; its
workspace/cohort validation is in progress, not a claimed completed repair.
Initial experiment: `local/repair-knot-candidate-{wurth,kicad}`.

### TBL 691404910001B high-precision follow-up

The exact source curves #827/#828 are not identical. Bounded closest-point
trimming gives a zero interval on #827 and a 1.24147e-11 interval on #828,
whose underlying curve moves only 8.13107e-14 mm. The two topological vertices
are 1.98852e-8 mm apart. High-precision surface projections of the endpoint-
replaced Foxtrot polyline enclose signed normalized-UV area -2.06496e-23;
the implementation currently cancels the projected boundary. This tiny area
primarily comes from connecting off-curve topology vertices, not from a proved
nondegenerate source trim. Neither validity nor invalidity of the source face
is established. Do not infer an exact duplicate trim, or relax the projection
termination rule based only on this experiment. Evidence:
`local/tbl-rca/high_precision_geometry.{py,txt}`.

### Evaluate descending knot candidates without a snapping tolerance

Compare each quadratic trial with its nearest cell-bound variants using
factored actual residual differences. A bound is eligible only inside the
trust region and with negative predicted distance change. Prefer it on an
actual-distance tie. The original trial remains available when the bound
cannot descend. No unchanged/shortened trial establishes convergence, and
stationarity thresholds stay unchanged.

The six CIRCM12 models now have zero f64-degenerate triangles; their f32
failures remain. All 13 axial-inductor cases pass the full gate, including
the pass-15 KiCad regression. The 95-file Würth cohort has 63 ok / 32 invalid
meshes and no face errors; the 16-file KiCad cohort has 13 ok / 3 invalid
meshes. TBL 691378100020 reverts from ok to an f32-only invalid mesh; that
regression remains open. Workspace tests pass, including the nonbinary-weight
extrusion regression, which fails before this change. Evidence:
`/tmp/knot-candidate-before.log`, `/tmp/knot-descent-workspace-tests.log`,
`local/repair-knot-descent-{wurth,kicad}`; frozen
`local/repair-knot-descent-worker`.

### Retain both intersection endpoint weights

Compute both barycentric weights directly from the oriented areas instead of
recovering the small weight as one minus the large one. Interpolate UV and
XYZ relative to the nearer endpoint. Use both weights to order split records
when the large weights round equal. This preserves constant coordinates and
representable endpoint offsets in either edge direction, without an epsilon.
The regression fails before the change: a reversed intersection 1e-20 from
an endpoint is collapsed into the endpoint. It passes after; workspace tests,
the 121-crossing arrangement, boundary ownership checks and eight checkpoints
pass. Frozen worker: `local/repair-intersection-relative-worker`; evidence:
`/tmp/intersection-relative-{before,workspace-tests}.log` and
`local/repair-intersection-relative-{investigate,check,kicad}`.

The 36-file investigation cohort is 8 ok / 20 invalid_mesh / 8 face errors;
the 19 KiCad investigations are 16 ok / 3 invalid_mesh. These figures include
the preceding knot-candidate changes, not just interpolation. A newly exposed
CMB-XS 744821110 projection error on surface #849 reproduces with the earlier
frozen knot-descent worker and is not caused by this arithmetic change.

Accurate interpolation still permits a generated intersection to round to an
existing endpoint's XYZ while retaining a distinct UV. KI-0603 has two such
intersections (#77/#78 both equal endpoint #3), now exposing three f64
degenerates rather than one; Bourns L39.4/W20.3 similarly goes from six to
eight. These are not repaired. Their representable construction identity must
be reconciled before CDT, without welding intentional seams/poles. Traces:
`local/{ki,cmanc}-endpoint.log`; the temporary instrumentation is removed.

### Reuse an unambiguous incident endpoint after rounded construction

When a newly interpolated intersection equals exactly one of its owner's
endpoints in XYZ, reuse that endpoint's existing UV/vertex identity. The
otherwise generated child edge has no representable 3D length. Matching both
endpoints is deliberately not enough: those chart representatives may encode
a seam or pole. This affects only newly constructed intersections, not global
vertex welding, existing topology, tolerance snapping or facet deletion.

The focused regression fails before and passes after, covers both unique and
ambiguous endpoint identity, and checks a nonzero face offset. Workspace tests
and all eight checkpoints pass. KI-0603 loses all three f64 degenerates; CMANC
7848050219 and CMBNC 7448050219 each lose their remaining f64 degenerate. Their
f32 failures persist. The 36-file and 19-file investigation status counts
remain unchanged (8/20/8 and 16/3 respectively), with no new status failures.
Evidence: `/tmp/intersection-endpoint-{before,workspace-tests}.log`,
`local/repair-intersection-endpoint-{investigate,check,kicad}` and frozen
`local/repair-intersection-endpoint-worker`.

### Complete pass 16 and correct overbroad knot candidates

The complete frozen knot-descent worker processes all 7,328 Würth files,
then all 7,251 KiCad files. Exact manifest path/hash sets, shard unions and
worker hashes match. Würth: 6,941 ok / 331 invalid_mesh / 53 tessellation_error
/ 3 crash. KiCad: 7,112 ok / 139 invalid_mesh. Per-file evidence is in
`.amp/in/artifacts/repair-pass16/{wurth,kicad}.{json,csv}`. Seventeen KiCad
files recover versus pass 15, but 46 Würth files gain projection errors.
TBL 691378100020 remains an ok-to-invalid regression; CIRCM12643220100404
recovers. These results are not completion of the repair request.

The CMB-XS 744821110 trace establishes the projection regression: the nearby-
knot comparison repeatedly resets an unconverged coordinate and destroys the
trust radius (3.26e-55 after 130 iterations), despite a remaining nonstationary
gradient. Evidence: `local/cmb-projection-probe.log`, including surface #849.
Restrict the comparison to retaining a stationary, already active bound.
Interior/unconverged parameters must follow the minimization model, not jump
to a nearby knot. No tolerance or convergence criterion is relaxed.

Workspace tests pass (`/tmp/active-bound-workspace-tests.log`). The 151-file
Würth cohort is 92 ok / 52 invalid_mesh / 7 tessellation_error: all 46 new
projection errors disappear, with no status regressions against pass 16.
Against pass 15, TBL still regresses and CIRCM still recovers. The 20-file
KiCad cohort is 7 ok / 13 invalid_mesh: seven recover against pass 15, but
11 pass-16 successes return to invalid meshes and one other file recovers.
Those mesh defects remain open; the broad knot rule was not a sound repair.
Evidence: `local/repair-active-bound-{wurth,kicad}`, frozen worker
`local/repair-active-bound-worker`.

### Additional disk cleanup

Removed rebuildable `target/debug` and obsolete `local/swept-target` after
checking that no Cargo build or corpus worker was running. Reclaimed about
5 GB; free space rises from 8.9 to 14 GB. STEP inputs, frozen workers, RCA
reports, manifests and diagnostic evidence are retained. Use nonincremental
builds and avoid regenerating debug build caches unnecessarily.

### Complete pass 17

All 7,328 Würth files complete before all 7,251 KiCad files with the frozen
active-bound worker. Coverage and worker hashes verify. Würth is 6,968 ok /
350 invalid_mesh / 7 tessellation_error / 3 controlled parser rejections
(the harness calls these `crash`). KiCad is 7,102 ok / 149 invalid_mesh.
The focused transitions above exactly match the full sweep. Per-file exports:
`.amp/in/artifacts/repair-pass17/{wurth,kicad}.{json,csv}`.

The f64 diagnostic flags eight facets in six Würth files and 17 facets in
seven KiCad files. Two of those Würth files pass the f32 gate: WR-USB
632723130112 and CMB-XS 744821110. Keep these upstream findings open even
though the exported STL gate passes. This diagnostic currently uses a rounded
cross product; exact collinearity needs checking before promoting every flag
to a proved degenerate geometric facet.

### Carry spline sampling density continuously across smooth spans

The previous sampler restarts its phase at every knot and emits that knot
as a vertex. Near a trim endpoint this constructs a tiny edge even at a
differentiable knot with no second topological vertex. The representative
source study is in `local/sampling-provenance/{REPORT.md,representatives.json}`;
it distinguishes generated samples from real endpoints and deliberately
does not classify the entire corpus by proximity.

Replace per-span sampling resets and the special periodic-cut balancing rule
with ordered cells carrying parameter endpoints, sampling measure and a
mandatory-corner flag. Integrate the existing per-span density over each
continuous run and distribute samples in that measure. Preserve topological
trim endpoints and every knot whose multiplicity can permit a tangent break.
Narrow full spans still receive their density; short trims retain their full
budget. No f32-dependent sampling, coordinate merging or facet removal.

`cargo test --release --workspace`: all 130 tests pass. New regressions cover
smooth knots immediately beside either endpoint in both directions, plus a
1e-9-wide span which must retain its sampling density. Existing corner,
periodic-cut and short-trim tests pass. The 452-file Würth cohort is 111 ok /
331 invalid_mesh / 7 tessellation_error / 3 parser rejections; the 156-file
KiCad cohort is 122 ok / 34 invalid_mesh. Relative to pass 17, 19 Würth and
115 KiCad files recover with no status regressions in these cohorts. One
Bourns f64 diagnostic disappears; the other flagged f64 cases remain.
Evidence: `/tmp/curve-measure-workspace-tests.log`,
`local/repair-curve-measure-{wurth,kicad}`, frozen
`local/repair-curve-measure-worker`.

All eight checkpoints pass the ordinary mesh gate. With OCCT enabled, four
pass the coarse comparison, three OCCT exports themselves have invalid
meshes, and WR-MJ 615032243321 still has the previously investigated geometry
mismatch. CP_Radial_D10.0mm_P3.80mm now has zero f64/f32 diagnostic degenerates,
but its area is 505.0375 mm² versus OCCT 566.2080 mm². The earlier area was
506.8602 mm²: this material mismatch predates the sampling change and remains
open. Evidence: `local/repair-curve-measure-occt` and
`local/curve-measure-cp`. Do not equate passing the ordinary gate with complete
geometry repair.

Compressed historical pass-15 per-case logs and verified every gzip stream.
Their paths now end in `.log.gz`; reports and frozen workers remain intact.

### Validate f64 collinearity with adaptive predicates

A rounded cross product is not a reliable collinearity predicate: the triangle
with vertices (0,0), (1,1+epsilon), (1-epsilon,1) has a nonzero determinant
epsilon², but the old diagnostic rounds it to zero. Use the triangulator's
adaptive orientation predicates on the three coordinate projections instead.
The focused worker test distinguishes this case from duplicate vertices and
true collinear triples. `cargo test --release -p triangulate --example
corpus_worker` passes. Rechecking all 13 previously flagged models preserves
their counts (eight Würth and 16 KiCad facets after the sampling change), so
none of the remaining flags can be dismissed as this diagnostic cancellation.
No mesh generation or gate changes. Evidence: `/tmp/collinear-tests.log`,
`local/repair-collinear-{wurth,kicad}`, frozen `local/repair-collinear-worker`.

### Share open/closed shell traversal before adding cavity shells

Consolidate the duplicated open-shell and closed-shell face loops into `shell`.
This removes two wrappers and 63 net lines while retaining the existing
meshing path. All 46 triangulate library tests pass; the CP_Radial reproduction
exports byte-identical STL before and after. Evidence:
`/tmp/shell-refactor-tests.log`, `local/curve-measure-cp/refactor*`, frozen
`local/repair-shell-refactor-worker`.

Direct source inspection resolves the capacitor's missing-face question:
BREP_WITH_VOIDS #15 has outer shell #16 (43 faces) and void shell #1963,
whose underlying shell #1964 contains five more faces. The shape dispatcher
explicitly skips `voids`. These are #1965, #1998, #2030, #2047 and #2074,
not the five complex spline faces tentatively discussed in the research
report. The STEP file and OCCT each contain 48 faces; OCCT did not split a
43-face source. Fixing void traversal is the next behavior change.

### Complete pass 18

The continuous-sampling worker completes all 7,328 Würth inputs before all
7,251 KiCad inputs, with exact coverage/hash verification. Würth: 6,987 ok /
331 invalid_mesh / 7 tessellation_error / 3 parser rejections. KiCad: 7,217 ok
/ 34 invalid_mesh. The full sweep confirms all 134 cohort recoveries and no
status regressions versus pass 17. Per-file exports and transitions:
`.amp/in/artifacts/repair-pass18/`. The 13 flagged f64 cases remain open.

### Include oriented cavity shells

Mesh the outer shell and every BREP_WITH_VOIDS cavity through the same face-set
traversal. Resolve oriented open/closed shells to their concrete face set and
orientation, rejecting a nested oriented element rather than recursing (WR1).
False orientation reverses triangle winding and render normals, not the order
of faces. The source definitions are
https://www.steptools.com/docs/stp_aim/html/t_brep_with_voids.html and
https://www.steptools.com/docs/stp_aim/html/t_oriented_closed_shell.html.

All 131 workspace tests pass, including a toroidal cavity test against analytic
solid-minus-void volume and normal/winding signs. CP_Radial now processes all
48 faces in two shells, with zero face errors or f64/f32 degenerates. Its area
is 583.6274 mm² versus OCCT's STL 566.2080 mm² (coarse comparison now passes).
The cavity's exact OCCT area is 79.1681 mm² and the old outer mesh already
overestimates its reference area: the remaining roughly 3% total difference
is not resolved by counting the five restored faces. No full equivalence claim.
Evidence: `local/curve-measure-cp/void-{metrics,comparison}.json` and
`/tmp/void-shell-workspace-tests.log`.

Scanned both full corpora for BREP_WITH_VOIDS/oriented-shell declarations and
tested all matching inputs: 468 Würth (432 ok / 36 invalid_mesh), then 211
KiCad (209 ok / 1 invalid_mesh / 1 tessellation_error). No formerly passing
file fails. Bourns L34.3/W20.3 was already invalid and now exposes two curve-
projection errors on newly visited cavity faces (surfaces #46027/#46037).
Those underlying projection failures remain open; skipping the cavity is not
a fix. Evidence: `local/repair-void-shell-{wurth,kicad}`, frozen
`local/repair-void-shell-worker`. A temporary mesh/edge-provenance probe is
frozen as `local/void-mesh-probe-worker`; instrumentation is removed.

Compressed and gzip-verified historical pass-16/pass-17 per-case logs and
completed continuous-sampling cohort STLs. These now use `.log.gz` and
`.stl.gz`; reports, metrics, inputs and workers remain. This restores about
7 GB while preserving evidence; free space is approximately 12–13 GB.

### Gate exact f64 degenerates as well as exported STL degenerates

The worker's adaptive f64 collinearity diagnostic is now an acceptance check,
not just a report field. Float32 rounding can move an exactly collinear f64
triangle off its line, so an apparently valid STL cannot override a defective
upstream mesh. Validate the optional metric and compare it across repeated
runs; legacy workers which omit it retain their existing behavior.

All 19 harness tests pass, including invalid/nonfinite f64 metrics. Rechecking
the six flagged Würth models rejects all six, including two that previously
passed the STL-only gate. Evidence: `/tmp/f64-gate-tests.log` and
`local/repair-f64-gate-wurth`. Pass 18 predates this strengthened gate and
cavity-shell traversal; its totals are not current all-corpus certification.

### Store spline-chart transforms as coordinate data

Replace seven separately named polar-chart fields with the angular axis,
two-coordinate origin/scale and parameter bounds. Remove the axis-dependent
construction branches without changing the mapping or acceptance domain.
This is a behavior-preserving prerequisite for representing bounded pole
sectors with the same chart as periodic disks. All 47 triangulate library
tests pass; the capacitor's exported STL is byte-identical to the previous
worker. Evidence: `/tmp/chart-data-tests.log`, `local/chart-data-check`, frozen
`local/repair-chart-data-worker`.

### Use the pole chart for bounded spline sectors

WR-USB 632723130112 face #19963/surface #19973 and WR-USB 692221030100
face #5642/surface #5659 have exact collapsed boundaries on bounded rational
conical patches. The rectangular chart represents the pole at different UVs,
allowing noncollinear UV facets whose raised vertices lie on one generator.
Choose the existing polar chart for a single collapsed boundary even without
a periodic axis. A bounded angular interval occupies a half disk, keeping its
ends distinct; inverse mapping enforces both parameter bounds. No triangle
deletion, proximity welding or change to source knots/control geometry.

All 132 workspace tests pass, including both angular axes and both pole ends,
out-of-domain rejection, pole identification and a triangulated quarter-cone
area check. Both USB models now have zero f64 and f32 degenerate facets and
pass the strengthened gate. The other four flagged Würth models remain
invalid. Evidence: `/tmp/bounded-pole-workspace-tests.log`,
`local/repair-bounded-pole-wurth`, frozen `local/repair-bounded-pole-worker`.

### Resolve curve projection at nonsmooth knot junctions

Directly loading Bourns L34.3/W20.3 curve #45959 disproves the initial external
diagnosis. The OCCT extraction accidentally selected curve #45994, which
shares the endpoints but has different knots and controls. There is no
downstream near-incidence rejection: the failing solver itself returns None.
The exact source curve's nearest seed is its C0 knot u=0.566534474389, with
one-sided unit gradients -3.76297e-9 and +5.49289e-9 and residual 4.18252e-8.
It is a one-sided minimum, but the old unrestricted smooth model oscillates
across the corner for 256 iterations.

Evaluate derivatives in explicitly selected knot intervals, constrain each
Newton step to its interval, and require stationarity on every incident side.
Interiors, corners and domain ends share this path; no source tolerance or
nearest-sample fallback is introduced. All 133 workspace tests pass, including
polynomial/rational corners and traversal across a nonstationary knot. The
exact reproduction now returns the knot; the whole Bourns model meshes all
468 faces/four shells with zero face errors and zero f64 degenerates. Its
f32 export remains invalid. Evidence: `local/bourns-cavity-projection` (with
the mistaken research explicitly corrected), `/tmp/curve-cells-workspace-tests.log`,
frozen `local/repair-curve-cells-worker`. The diagnostic example is removed
from the source tree after preserving its reproduction under `local/`.

### Complete pass 19 and bound disk use

All 7,328 Würth inputs complete before all 7,251 KiCad inputs, with frozen
worker/hash-set verification. This run includes cavity shells, bounded pole
charts and the f64 acceptance gate, but predates the curve-junction repair.
Würth: 6,984 ok / 334 invalid_mesh / 7 tessellation_error / 3 parser rejections.
KiCad: 7,217 ok / 33 invalid_mesh / 1 tessellation_error. Exports:
`.amp/in/artifacts/repair-pass19/`. Three previously passing Würth capacitors
regress because nonzero short boundaries were classified as bounded poles
using source uncertainty; those regressions are not acceptable and are being
repaired. The newly rejected CMB-XS has a pre-existing f64 degenerate that the
old f32-only gate missed.

The subsequent curve-junction worker processes the 452-file Würth and 156-file
KiCad cohorts. No status regressions versus pass 19; Bourns L34.3/W20.3 loses
its two curve errors and reaches mesh validation. Cohort totals: Würth
111 ok / 331 invalid / 7 face errors / 3 parser rejections; KiCad 122 ok /
34 invalid. Evidence: `local/repair-curve-cells-{wurth,kicad}`.

Removed 5,873 superseded generated mesh exports from passes 1–17, recovering
5,093,604,499 bytes. Their reports, metrics, diagnostic logs, manifests and
frozen workers remain, as do current failure meshes and all corpus inputs.
Deletion inventory: `local/superseded-mesh-cleanup.json`. Compressed all
pass-18 per-case logs and verified their gzip streams; those paths now end
in `.log.gz`. Removed stale temporary probe STLs from `/tmp/`. Approximately
14 GB is free after the new sweep. No source or input data is removed.

### Distinguish bounded poles from short nonzero edges

The three capacitor regressions select boundary curves shorter than the
declared 5e-6 uncertainty, not actual collapsed boundaries. Require coincidence
for bounded pole charts instead of using source uncertainty to erase an edge.
Keep the existing periodic-pole convention. Measure rational control offsets
with the evaluator's homogeneous translation before division: weighting one
Cartesian point and dividing it back can otherwise create unequal rounded
quotients (WR-USB 692221030100's shared z=-15.57 is an example).

All 135 workspace tests pass, including preservation of a 1e-8 nonzero edge
under 1e-6 source uncertainty, recognition of weighted Cartesian constants,
and rejection of a one-ULP weighted displacement. All three capacitor models
recover and both USB models keep zero f64/f32 degenerates. The other four
flagged models remain invalid. Evidence: `/tmp/pole-identity-workspace-tests.log`,
`local/repair-pole-identity-wurth`, `local/bounded-pole-regressions`, frozen
`local/repair-pole-identity-worker`. The intermediate exact-quotient worker
misses one USB pole and is retained only as diagnostic evidence.

### Additional disk cleanup and inverse-sheet investigation

Removed another 3,651 obsolete mesh exports (5,959,055,057 bytes), retaining
metadata and requiring each input path/hash to be covered by the newer full
pass-19 sweep. Inventory: `local/disk-cleanup-followup.json`. Input corpora,
OCCT, frozen workers, diagnostic reproductions, pass 19 and active pass 20 are
preserved. About 19 GB is free after cleanup.

The Fastron horizontal axial inductor's surface 2557 has two neighboring
boundary samples whose nearest-grid Newton start reaches the wrong surface
sheet, with 0.105686 mm residual. None of the 64 starts in that selected cell
reaches the correct sheet. A start in another knot cell reaches 7.535e-11 mm.
The exact source vertex 2541 itself succeeds; the failure concerns its two
neighboring samples. Reproduction and full numerical tables are preserved in
`local/inverse-seed-rca/`.

An initial control-hull-pruned multi-start experiment passes 136 workspace
tests but does not fix the actual model: residual remains 0.093348 mm. A hull
bounds the surface cell, not an unconstrained Newton basin which may leave
that cell. The repair under investigation therefore restricts each candidate
search to its knot cell, using the same bounded distance model rather than
adding a revolution-specific antipodal seed. No successful corpus outcome is
claimed for this work yet.

### Complete pass 20

The frozen pole-identity worker completes all 7,328 Würth inputs and then all
7,251 KiCad inputs. Exact manifest path/hash sets and worker digest are checked.
Würth: 6,987 ok / 331 invalid_mesh / 7 tessellation_error / 3 parser rejections.
KiCad: 7,217 ok / 34 invalid_mesh / 0 tessellation_error. Compared with pass 19,
the three capacitor regressions recover; Bourns loses its curve errors but
still fails mesh validation. No passing file regresses. Per-file evidence:
`.amp/in/artifacts/repair-pass20/{wurth,kicad}.{json,csv}` and `regressions.json`.
These checks do not establish manifold topology or OCCT geometry equivalence.

The subsequent cell-bounded inverse experiment now recovers both failing
Fastron neighbors to 7.535e-11 mm residual. All 136 workspace tests pass.
The 452-file Würth and 156-file KiCad repair cohorts complete against the
frozen `local/repair-inverse-cells-worker`; this change is not in pass 20.
Würth: 115 ok / 327 invalid / 7 face errors / 3 parser rejections. KiCad:
133 ok / 23 invalid. Four Würth inductors (HCI-1890 and three RCIS parts) and
eleven KiCad axial inductors recover. No status regression or increased f64
degenerate count occurs in either cohort. Median triangulation-time ratios
versus the prior full sweep are 1.63x/1.48x; these concurrent runs are not an
isolated benchmark. The repair adds cached per-knot-cell control hulls and
search domains to the existing bounded Newton model, with no conic-specific
branch, neighboring-vertex hint, or distance-tolerance escape. The analytic
folded-strip regression covers polynomial and rational surfaces. Evidence:
`local/repair-inverse-cells-{wurth,kicad}` and
`/tmp/inverse-cells-workspace-tests.log`. A new full sweep is still required.

### Chart-boundary refinement experiment (not accepted)

The straight spans on WR-DSUB61803729321 face 53394/surface 31 are cubic
B-splines (#9624/#9628), not STEP LINE entities. Their collinear 3D samples
are simplified, but their polar-chart images are curved. Long chart chords
cross the separate inner loop; manufactured crossing vertices are collinear
in space and corrupt trim parity. Refining existing 3D chords until chart
midpoint sag is at most 1/32 of chord length removes this face's degenerates.
However, the whole model regresses from one to 46 f64 degenerates.

Per-triangle instrumentation attributes all 46 new f64 degenerates to four
other NURBS surfaces (22, 24, 27, 29), not the torus surfaces speculated about
in the advisory review. This is direct evidence for the feared failure:
subdivision adds collinear XYZ points whose bent chart image lets CDT form
zero-area lifted ears where the uniform interior grid is too coarse. Example:
face 53888, indices [40,39,62], has vertex 62 at the exact spatial midpoint
but off the chart chord. Surface 31 now has no f64 or f32 degenerates.
Evidence: `local/chart-refinement-dsub-audit.{json,log}` and frozen
`local/chart-refinement-audit-worker`; uninstrumented prototype saved as
`local/chart-refinement-prototype.patch`.

A second experiment adds an actual surface sample at each subdivision ear's
chart centroid (inside its circumdisc), without moving boundary geometry or
discarding triangles. It clears all f64/f32 degenerates on the complete DSUB
model and passes 137 workspace tests, including a triangular-prism trim/area
regression. An additional 50-test library check passes after removing the
temporary diagnostic instrumentation and duplicate polar-period guard.

The complete 452-file Würth cohort rejects this experiment: 94 ok / 348 invalid
/ 7 face errors / 3 parser rejections. Four files improve, but 25 previously
passing files regress (23 f32-only, two with f64 degenerates). The 156-file
KiCad cohort has no status changes: 133 ok / 23 invalid. Source path/hash sets
and worker digest are checked. Per-file transitions and counts are exported
to `.amp/in/artifacts/chart-centroid-evaluation.json`; underlying causes of
the new degeneracies remain open. All experimental source changes are
removed, with a replayable patch at `local/chart-centroid-prototype.patch`.
Instrumented evidence is retained in `local/chart-centroid-audit.patch` and
the frozen `local/chart-centroid-audit-worker`.

The optional harness OCCT comparison for DSUB reports `oracle_invalid_mesh`:
our experimental STL has zero degenerates; the OCCT reference has two. Their
bounds match, but surface areas are 15174.720246 versus 14896.001290 mm²
(about 1.87% difference), so no equivalence is claimed. The prior baseline's
area is 14679.618859 mm². Evidence: `local/chart-dsub-reference`.

### Complete pass 21 and bookkeeping checkpoint

The committed inverse-cell worker completes all 7,328 Würth inputs, then all
7,251 KiCad inputs. Würth: 6,991 ok / 327 invalid_mesh / 7 tessellation_error /
3 parser rejections. KiCad: 7,228 ok / 23 invalid_mesh. This confirms all four
Würth and eleven KiCad cohort improvements, with zero status regressions.
Exact manifest/hash coverage and worker digest are verified. Canonical
per-file exports are `.amp/in/artifacts/repair-pass21/`. Neither rejected
chart experiment is part of this run.

Bookkeeping removes superseded pass-18/19 meshes covered by pass 20 and
losslessly compresses inverse-cell cohort meshes, verifying decompressed
SHA-256 values before removing raw copies. The 1,104 recorded operations
recover 2,385,852,945 bytes: `local/bookkeeping-mesh-cleanup.json`. Inputs,
reports, logs, metrics, manifests, reference data, workers and current full
sweep meshes remain. Temporary probe STLs are removed; `uv cache prune`
removes another 2.1 MiB of unused cache. The release worker is rebuilt after
shelving the experiment and is byte-identical to the frozen pass-21 worker.

After the chart cohorts finish, their 378 retained mesh exports are also
losslessly compressed and SHA-256 verified, recovering 1,656,276,228 bytes.
Inventory: `local/chart-mesh-compression.json`. Together, this bookkeeping
pass recovers 4,042,129,173 bytes; approximately 13 GiB is free. All inventory
destinations and deleted raw paths are checked after completion.

The user permits high-quality crates when they reduce complexity. Existing
core dependencies already include Spade 2.15.1 for CDT, `robust` predicates,
nalgebra and `thiserror`; no new dependency is added just for bookkeeping.
