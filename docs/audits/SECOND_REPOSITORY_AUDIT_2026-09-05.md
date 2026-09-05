# Second repository audit: consistency, shared kernels and truthful certificates

Date: 2026-09-05. Repository version: `0.0.3.5`.

Subsequent execution, integrator, manifest and reporting findings are recorded
in the [third audit](THIRD_REPOSITORY_AUDIT_2026-09-05.md). The measurements below
remain the historical result of this second pass.

This pass starts from the completed [first audit](REPOSITORY_AUDIT_2026-09-05.md),
whose uncommitted changes remain in the working tree. The baseline is the
first audit's corrected source, not Git HEAD alone. The first full test run
for this pass reproduced **2,519 passed, 12 skipped, 91 warnings** in 99.03 s.
`manual/` was excluded. No commit, publication or deployment was performed.

The investigation followed four independent tracks: operator grammar/history,
field kernels/precision, geometric certificates, and transport/spectral
readouts. Independent cross-review then tested the fixes against counterexamples
outside the original fixtures. This is a deep, bounded audit of those shared
paths, not a claim that every research module or theorem is correct.

## Implemented findings

| ID | Priority | Reproduced problem | Correction and evidence |
| --- | --- | --- | --- |
| B01 | P1 | Neutral operations and history eviction erase outstanding U2 debt; earlier stabilization can prepay future destabilization | One nonnegative causal debt kernel, persisted on nodes; batch prefixes, incremental selection and execution share it. Stabilizers discharge one outstanding unit; neutral operations preserve debt. |
| B02 | P1 | Incremental Mutation validation forgets prior Coherence outside its recent window, while batch validation accepts it | Lifetime prior-IL state is tracked separately from recent destabilizer context and shared with execution preconditions/readiness. Snapshot restoration and deliberate trace replacement handle both counters. |
| B03 | P2 | Selector records an already-recorded glyph; depth and serialized-name checks diverge | Record successful operations once, centralize name/classification handling, validate actual Recursivity depth, reject noninteger/nonfinite declarations. |
| B04 | P1 | Automatic landmark potential above 500 nodes changes U6 telemetry drastically | Exact shortest-path potential is the default for all sizes. On the 600-node path reproducer, RMAE falls from 4.507402 to 0 and maximum potential from 10.810648 to 0.328320145468. |
| B05 | P1 | Landmark distance bounds, directed legs, disconnected sentinels and sampled error estimates produce misleading potential validation | Explicit landmarks use actual paths through landmarks; validation checks every node and falls back to exact values. Nonfinite reference values cannot receive an error-free certificate. |
| B06 | P2 | Competing potential/readout implementations disagree; a path integral includes the target's outgoing edge | Telemetry and weighted nodal classification call the canonical potential kernel; path integrals stop at the target and a zero-length path returns zero. |
| B07 | P1 | Mean frequency replaces heterogeneous nodal capacity in decay rates and thresholds | Spectrum/dispersion use the actual generator `A = diag(nu_f) L_rw`. The homogeneous Fiedler certificate explicitly checks its applicability. |
| B08 | P1 | A diffusion certificate samples unit-capacity flow, demands global uniformity, and labels degree-weighted conservation universal | Sample `e' = -A e` at the requested step; check all left-nullspace invariants and final stationarity. Keep legacy degree/global-uniformity outputs as separate observations. |
| B09 | P2 | Parallel-edge adjacency is misread; copied graph caches bypass directedness checks; returned mode arrays can corrupt later readouts | One weighted adjacency convention; topology signatures include graph kind; symmetric eigenbases reject asymmetric adjacency; public mode arrays are independent copies. |
| B10 | P1 | Disconnected transport receives finite resistance; commute times include unreachable components; currents ignore conductance | Absorbing zero-strength transitions, component-wise resistance/commute calculation, weighted current, and reachable Ohm checks with an actual equation residual. |
| B11 | P1 | Single-state products certify maps as symplectic and reject valid rotations | Snapshot-only results are inconclusive. An optional Jacobian is checked against the shared symplectic pullback identity. |
| B12 | P1 | Zero-energy reduction is certified as a regular positive-dimensional quotient | The singular zero level is a point; positive levels use their actual horizontal tangent space, including zero-action coordinate cases. |
| B13 | P2 | Nonzero quadratic thresholds are called critical points, and model-specific identities are asserted universally | Test actual stationarity of the quadratic potential, retain proximity separately, label heuristic grammar diagnostics, and state the graph-wave/substrate distinction. |
| B14 | P2 | Precision-mode switches reuse stale field caches, and research/dense sums erase signed cancellation residuals | Precision-aware dependencies invalidate canonical field/aggregate caches; exact dense and streamed paths share compensated summation, accounting for platforms where longdouble equals float64. |

Technical detail and scoped test evidence:

- [Grammar, depth and history](SECOND_AUDIT_GRAMMAR_2026-09-05.md).
- [Structural potential, numerical accuracy and readouts](SECOND_AUDIT_FIELDS_2026-09-05.md).
- [Symplectic and variational certificates](SECOND_AUDIT_CERTIFICATES_2026-09-05.md).
- [Random walks, resistance and current](SECOND_AUDIT_TRANSPORT_2026-09-05.md).

## Nodal transport: independent analytical checks

On a three-node path with capacity `[1, 3, 5]`, the exact generator is

```text
A = [[1, -1, 0], [-1.5, 3, -1.5], [0, -5, 5]]
det(x I - A) = x (x - 2) (x - 7)
```

The corrected decay rates are `[0, 2, 7]`; scalar reaction `r=0.5` gives
growth rates `[0.5, -1.5, -6.5]`. The previous mean-frequency calculation
returned `[0, 3, 6]`. The old Fiedler certificate also reported threshold 3,
although the first actual nonuniform rate is 2. Its geometric Fiedler claim
now requires symmetric connected support and a common positive capacity;
heterogeneous rates remain available from the general spectrum API.

For symmetric positive-frequency networks, the conserved linear functional
has weights `d_i/nu_i`. On one edge with capacities `[1, 3]` and EPI `[1, 0]`,
the long-time EPI is `[0.75, 0.75]`: the degree-weighted sum changes from 1
to 1.5, while the true invariant stays fixed. The certificate now reports
both facts. For general fixed generators, an SVD finds all left-nullspace
coordinates, including disconnected components and zero-capacity nodes.

Sampling now respects capacity and physical step size. On one edge with
capacity 2, EPI `[1, 0]` and `dt=0.1`, one step gives `[0.8, 0.2]`, standard
deviation 0.3. Explicit steps must remain nonnegative transitions. Invalid
frequency/weight values and nonfinite EPI cannot be silently sanitized into
successful certificates.

Two disconnected edges with EPI `[0, 0, 1, 1]` are stationary without global
uniformity. All-zero frequency similarly freezes a nonuniform field even if
pressure remains nonzero. These distinguish nodal stationarity from pressure
equilibrium. The SDK also retains extra zero modes: a disconnected graph's
gap is zero and its `1/sqrt(lambda_2)` proxy is infinite.

Directed damping uses real parts of the actual nonsymmetric generator's
eigenvalues. A directed three-cycle has rates `[0, 1.5, 1.5]`; feeding its
matrix to a symmetric eigensolver is rejected. These rates do not describe
oscillation frequencies, nonnormal transient growth or a canonical U2 metric.

Regression files: [generator consistency](../../tests/physics/test_diffusion_generator_consistency.py),
[transport consistency](../../tests/physics/test_random_walk_consistency.py),
and [SDK readouts](../../tests/sdk/test_simple_advanced.py).

## Centralization and optimization scope

The changes remove competing definitions at shared points: U2 accounting,
lifetime Mutation context, glyph classification, exact potential, weighted
adjacency, symplectic pullbacks, and component resistance geometry. Geometry
eigendecompositions remain reusable across capacity changes, while nodal
rates recompute the capacity-dependent part. Public mode copies prevent
callers from poisoning the shared decomposition.

Exact potential now streams shortest paths instead of constructing a dense
all-pairs distance matrix for sparse graphs. Source-confirmed cold-cache
medians during the exact-kernel phase, before the precision addendum, were
**0.198 s** at 600 path nodes and **0.773 s** at 1,200 nodes, with RMAE 0.
Those are measured timings, not a general speedup
claim. Sparse all-source work remains quadratic; dense and very large graphs
need separate measurements. Explicit 50-landmark estimates still had RMAE
approximately 0.898 and 0.943 on those paths. Their limitations are documented;
unvalidated approximations are unsuitable for U6 decisions.

Numerical precision selection is part of the readout state. Cross-review
additionally checks cancellation and cache behavior when switching precision
modes; final results are recorded in the field report.

## Compatibility and mathematical limits

Signatures are retained where possible, but incorrect results intentionally
change. Snapshot-only `is_canonical` becomes `None`; callers must handle
inconclusive evidence. Symmetric geometry/resistance APIs reject asymmetric
inputs instead of returning unsupported results. The Fiedler certificate
rejects heterogeneous, frozen and disconnected cases; general nodal spectra
and diffusion diagnostics handle those cases within their stated scope.

Existing diffusion certificate construction remains compatible through
appended optional fields. Its degree-conservation and global-uniformity
fields retain literal meanings. Legacy random-walk certificate names remain,
with component volume specified for disconnected networks. Deliberately
replacing a history requires resetting its grammar bookkeeping; complete
snapshots should retain the persisted state. Lost legacy history cannot be
reconstructed. Read-only incremental/shadow validation rejects one-shot
iterators before consumption; replayable lists, tuples and deques remain valid.

The first audit's assertion that real Recursivity instances did not retain
depth was incorrect and has been corrected there. Tests now use actual
instances. U5 still checks declared depth and stabilizer context; it does not
measure the full parent/child coherence inequality on an evolving nested EPI.

The mathematical issues T02, T07, T08, T10 and the completeness part of T11
in the [first theory audit](THEORY_CONTRADICTIONS_2026-09-05.md) remain open.
The tetrad potential is not proved to generate the full nodal pressure;
the implemented isotropic Hamiltonian is not the graph-wave model used by
the damping limit; phase wrapping alone does not prove universal potential
confinement. Grammar debt is discrete accounting, not a proof of continuous
integrability or an exact perturbation relaxation window. No operator or
open mathematical program receives a new universal certificate from this work.

## Integrated validation

The final full default suite passed **2,686 tests**, with **13 skips** and
**91 warnings**, in **101.97 s**. This adds 167 passing cases to the second-pass
baseline. The warning count is unchanged. The skips comprise seven unavailable
JAX cases, three unavailable scikit-learn modules, two existing manifest/CLI
format skips, and one new test requiring a genuinely extended longdouble
mantissa unavailable on this Windows runtime. None of those skipped paths is
claimed as validated. Focused totals in the linked reports overlap and must
not be added together.

Both the wheel and source distribution built successfully. Their **448 Python
modules** match the final working source byte for byte, including the new shared
grammar module. Package name/version, Python/dependency requirements and console
entry points match the first-audit wheel. These are local build artifacts.

All 448 modules parse; `git diff --check` passes; 40 local links across eight
audit documents resolve. Final logs are retained locally in
`tmp/second-audit-final-tests.log` and `tmp/second-audit-build.log`; the pre-change
baseline is `tmp/second-audit-baseline.log`. Package artifacts are in
`tmp/second-audit-dist/`. The local artifact verifier is
`tmp/verify_second_audit_artifacts.py`.

Reproduction from the repository root:

```powershell
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
.venv312/Scripts/python.exe -m build --outdir tmp/second-audit-dist
```

For standalone experiments, set `PYTHONPATH=src` or insert the absolute source
directory into `sys.path`, and record the imported module path. The existing
environment also contains an installed release; pytest's configured source
path does not automatically apply to ordinary Python invocations.
