# Third repository audit: execution, numerical evolution and reproducible reporting

Date: 2026-09-05. Repository version: `0.0.3.5`.

Subsequent ownership, configuration, replay and SDK findings are recorded in
the [fourth audit](FOURTH_REPOSITORY_AUDIT_2026-09-05.md). The measurements below
remain the historical result of the third pass.

This audit begins from the completed [second audit](SECOND_REPOSITORY_AUDIT_2026-09-05.md).
Its baseline full suite passed **2,686 tests, 13 skipped, 91 warnings**, in
99.78 seconds. The first two audits remain uncommitted in the working tree;
their changes are preserved. The second-audit wheel is an immutable source
reference for distinguishing this pass from those earlier changes. The
untracked `manual/` directory is outside the audit scope. No commit, publication
or deployment was performed.

Four parallel investigations examined numerical integration, operator execution,
manifest transport, and shared utilities/SDK reporting. Independent cross-review
then exercised execution context, numeric scalar compatibility, undefined phase
means and exception handling. Findings below are supported by reproducible
counterexamples and regression tests. This is a bounded audit of these shared
paths; it does not certify every research module or theorem in the repository.

## Implemented findings

| ID | Priority | Reproduced failure | Correction |
| --- | --- | --- | --- |
| C01 | P1 | Invalid trace arguments could reject an operator after EPI mutation; an invalid later batch target left earlier targets changed | Validate glyph, window, replayable history and the complete target collection before the first application. Preserve tuple/frozenset node identity. |
| C02 | P1 | An IL fallback still executed the requested THOL/OZ wrapper, creating nesting or propagation inconsistent with its history | Centralize selection before subclass execution; run the actual selected operator's workflow, metrics and metadata. |
| C03 | P1 | A Dissonance contract probe could certify substituted Coherence | Prepare actual prerequisites, verify executed history and measure the direct post-operator pressure change. |
| C04 | P1 | The valid word `AL, OZ, IL, SHA` executed as `AL, IL, IL, SHA`; SDK validation failures could be ignored | Share immutable validated-word context for future U4a handlers, retain live U2/U3/U4b checks, honor validation results and reject blocked steps explicitly. |
| C05 | P1 | The Yoshida routine advanced the wrong interval and failed fourth-order convergence | Compose three synchronized Verlet steps using the symmetric fourth-order coefficients; remove the incomplete drift/kick implementation. |
| C06 | P1 | Clipping changed stationary nodes and zero-duration calls; graph-default invalid timesteps bypassed validation | Preserve zero-update identity and centralize timestep and clipping policies, including finite NumPy real scalars. |
| C07 | P1 | Extended Euler concealed invalid field calls with synthetic values and depended on node insertion order | Call canonical field functions once per substep and stage derivatives before writing state; use configured bounds and reject unsupported RK4 explicitly. |
| C08 | P2 | Flux divergence changed at 101 nodes, reinterpreted weights and assigned flux divergence to isolates | Use one unique-neighbor discretization at every graph size and remove redundant sparse/index construction. |
| C09 | P2 | CPU integration imported an optional GPU subsystem; explicit zero parameters selected defaults | Isolate optional imports, distinguish zero from omission and share the exact frozen-derivative update between accepted method names. |
| C10 | P1 | Alias priority depended on previous reads and stale mapping identity; temporary mappings accumulated cached entries | Remove the unsafe per-mapping cache and resolve current aliases in order. Share collector setup and materialize alias iterators once. |
| C11 | P1 | Optional JSON backend availability changed options, numeric values and output bytes | Use a single standard-library encoder contract, preserving all requested options and explicit byte conversion. |
| C12 | P1 | Failed JSON export could truncate an existing file; atomic append could replace rather than append | Serialize before opening the destination, use the shared atomic writer, reject nonreplacement atomic modes and preserve structured exceptions. |
| C13 | P2 | SDK exports reused stale coherence; arithmetic phase means crossed the wrap incorrectly; missing averages crashed summaries | Measure current state, use circular phase averaging, report an undefined balanced mean as `None`, and distinguish measured zero from missing values. |
| C14 | P1 | Exporters advertised runner compatibility without entries or reconstructible graphs; telemetry imports silently failed | Introduce a shared finite-JSON graph schema, entries index, canonical telemetry readout and generic runner alongside legacy Paley input. |
| C15 | P1 | A dry run reported archived coherence drift as an optimization gain | Define deltas as actual after-minus-before snapshots, label archive drift separately and disallow promotion from dry runs or nonpositive/nonfinite changes. |
| C16 | P2 | Partition size limits, hash-order reproducibility, zero-valued aliases and directed/empty diagnostics diverged | Enforce capacity, retain stable ordering and scalar identity, share alias/diagnostic kernels and clarify that recorded seed labels do not configure execution RNGs. |

Detailed evidence and reproduction cases:

- [Operator execution and validated grammar](THIRD_AUDIT_EXECUTION_2026-09-05.md).
- [Numerical integration and local timings](THIRD_AUDIT_INTEGRATORS_2026-09-05.md).
- [Manifest production, loading and optimization reporting](THIRD_AUDIT_MANIFESTS_2026-09-05.md).
- [Alias regressions](../../tests/test_alias_resolution_consistency.py),
  [JSON/write regressions](../../tests/test_json_backend_consistency.py) and
  [SDK reporting regressions](../../tests/sdk/test_fluent_reporting_consistency.py).

## Utility and reporting evidence

### Alias precedence and redundant state

The former [AliasAccessor](../../src/tnfr/alias.py) remembered an alias using
`(id(mapping), aliases)` and the mapping's length. Replacing a padding entry
with a higher-priority key preserved the length, so reads and writes continued
to select the old fallback. Repairing an invalid primary value, changing the
requested conversion, or enabling strict conversion also failed to invalidate
that decision. A preceding permissive read could even determine which key a
later write changed.

Resolution now examines the current mapping in declared alias order for every
access. Reads preserve conversion/fallback rules; writes update the first
existing alias, or the first declared alias when none exists. Legacy-only
mappings therefore keep their existing write behavior. Canonical graph setters
retain their cache-invalidation hooks. Validated alias-tuple caching remains;
the mapping-identity cache and its lock are removed. Duplicate node-collection
setup and a redundant dataclass decoration were also removed.

For example, after reading `{'fallback': 2, 'padding': None}`, replacing
`padding` with `primary: 1` now makes the next read return 1. A write then
changes `primary` while leaving `fallback` intact. Supplying an alias iterator
to a multi-node collector no longer consumes it for the first node only.

A local comparison loaded the previous implementation from the second-audit
wheel and the current implementation from the working `src`. Five batches of
100,000 operations used six aliases and a float converter; the table reports
median elapsed seconds. Inputs have the value at either the first or last alias.

| Operation | Second audit | Third audit | Measured reduction |
| --- | ---: | ---: | ---: |
| First-alias reads | 0.101185 | 0.065111 | 35.7% |
| Last-alias reads | 0.101765 | 0.077019 | 24.3% |
| First-alias writes | 0.090829 | 0.050036 | 44.9% |
| Last-alias writes | 0.096258 | 0.060147 | 37.5% |

After accessing 10,000 distinct retained mappings, identity-cache entries were
10,000 before and zero after. These are local timing and retained-state
measurements, not an engine-wide speed or memory guarantee. The reproducible
benchmark is retained locally as `tmp/third_audit_alias_benchmark.py`.

### Deterministic JSON and preservation of existing files

[utils/io.py](../../src/tnfr/utils/io.py) previously dispatched to an optional
encoder with incompatible defaults. Options including `ensure_ascii`, `indent`,
`separators`, `cls` and `allow_nan` could be ignored. With real orjson installed,
NaN/infinity became null, large integers or non-string keys could fail, and
serialized bytes could differ solely because an optional package was present.
The single encoder now follows the existing standard-library fallback contract.
Callers requesting strict finite JSON must still pass `allow_nan=False`.

This intentionally favors a stable public encoding contract. No JSON throughput
improvement is claimed. Existing optional-backend-specific output can change;
persisted hashes derived from that output may need regeneration. The public
parameter container and warning-reset compatibility entry point remain.

[SDK JSON export](../../src/tnfr/sdk/utils.py) now finishes serialization before
delegating to `safe_write`. Both an unsupported object and an invalid UTF-8
surrogate preserve the previous destination byte content, with no temporary
file left behind. The original `UnicodeEncodeError` and its structured details
are retained; reconstructing an exception from only a message previously masked
that failure with `TypeError`.

Atomic writing supports replacement modes based on `w`. Atomic `a`, `ab`, `x`
and `r+` now fail before writing rather than silently changing their meaning.
An explicitly non-atomic append retains its original append behavior. This
is a per-file replacement guarantee, not a multi-file transaction or a power-loss
durability guarantee.

### Current coherence and circular phase

[Fluent reporting](../../src/tnfr/sdk/fluent.py) now remeasures before exporting.
In the regression, measuring an equilibrium graph and then setting pressure to
10 used to export the cached coherence 1; the current graph's coherence is
`1/(1+10)=1/11`, which is now reported. The export is documented as network
metadata and current metrics, not a complete restorable graph snapshot.

Phases `[0.1, 2*pi-0.1]` formerly averaged to pi. The mean now uses the shared
circular kernel, with the zero direction represented modulo `2*pi`. For balanced
antipodal pairs the resultant is numerically zero and there is no reportable
direction. Their mean is `None`, including rotated pairs. The zero-resultant
guard uses floating-point roundoff, not a new TNFR structural threshold.
Summaries accept unmeasured/undefined means and identify zero as a computed value.

The three utility/reporting modules passed **46 focused cases**. These tests
exercise public outputs, exception details, alias mutations and retained files.

## Numerical and structural effect

For a force-free mechanical node with `q=0`, `q_dot=1`, `dt=0.1`, the former
Yoshida routine returned `q=0.13243964040201714`; the corrected result is 0.1.
For `q''=-q` to `T=1`, halving the step gives measured orders **4.00157 and
4.00039**, versus **2.00105 and 2.00026** for Verlet. The separable autonomous
Hamiltonian assumptions and force representation are stated in the numerical
report; this does not establish arbitrary-callback symplecticity.

Zero-duration ordinary and extended integrations preserve state. With zero
external forcing, zero nodal capacity or pressure preserves EPI during a
nonzero step. Active soft clipping now follows the existing scalar boundary
policy in both backends. The extended step reads one graph state, computes
canonical fields and stages derivatives, so its regression no longer changes
when node insertion order is reversed.

For unweighted paths, the unified flux-divergence routine agrees with the old
scalar formula within `1e-14`. Its local median at 1,000 nodes decreased from
12.929 ms to 1.229 ms. Weighted, parallel and directed behavior now follows
that explicit unique-neighbor definition at every size; graph padding no longer
switches the discretization. These changes establish numerical consistency,
not a universal increase of C(t), Si or stability margin.

Execution uses the existing operator catalog. Future U4a handlers are available
only through a canonically validated word; live U2 debt, U3 phase compatibility
and U4b historical prerequisites remain independent. All 15 predefined fluent
words pass both current validation layers. The compatibility validator retains
additional restrictions, including its consecutive-Expansion rejection; this
audit documents that distinction without removing a public validation contract.

## Compatibility and unresolved scope

Incorrect behavior intentionally changes: rejected operator requests no longer
run their subclass effects; blocked validated steps raise instead of silently
substituting; invalid trace sizes and timesteps fail earlier; extended `rk4`
raises instead of secretly executing Euler. The constant-field CPU wrapper
accepts Euler and RK4 names for the same exact frozen-derivative calculation;
its convergence flag is a step-change diagnostic, not structural equilibrium.
The GPU execution branch remains unvalidated in this environment.

Operators and sequences are not transactions. A runtime error, hook failure or
failed future handler can leave already completed nodes/prefixes changed. The
new sequence context stores no continuing permission on the graph. The retained
U5 validation checks declared depth and stabilizer context; it does not prove
the full evolving parent/child coherence inequality.

The manifest graph schema preserves built-in finite JSON state and scalar node
IDs across all four NetworkX graph kinds. Unsupported arrays, callbacks, RNGs,
tuple IDs and other runtime objects are rejected. A bundle is not a multi-file
transaction. Repository SDK orchestration still depends on factorization-lab
helpers; generic pattern/fractal operation validation remains pending. Seed
labels are recorded metadata, not a seeded-execution promise. Successful graph
loading or a dry run cannot stand in for a certified optimization gain.

The mathematical issues T02, T07, T08, T10 and the completeness part of T11 in
the [first theory audit](THEORY_CONTRADICTIONS_2026-09-05.md) remain open. The
variational/substrate bridge, universal potential confinement, exact relaxation
window, continuous integrability and tetrad completeness do not acquire proofs
from these implementation fixes. No open research problem is claimed solved.

## Integrated validation

The final full default suite passed **2,880 tests, 11 skipped, 96 warnings**, in
109.22 seconds. This is 194 more passing cases than the third-audit baseline;
two previous manifest/CLI integration skips now pass with actual success
assertions. The remaining skips are seven unavailable JAX cases, three
unavailable scikit-learn modules, and one extended-longdouble precision test
unsupported on this Windows runtime. Warnings concern repeated Coherence and
unavailable optional JAX; five additional optional-import warnings accompany
the newly exercised manifest paths. None of those skipped paths is certified.

The default configuration selects `tests/` and excludes the `slow` marker.
The focused manifest run additionally covers the modified factorization-lab
helper tests outside that default directory. Focused counts in the linked
reports overlap and must not be added together. Runtime: Python 3.12.10,
NumPy 2.3.3, NetworkX 3.5, pytest 9.0.2. The installed optional orjson path was
included in regressions.

Both wheel and source distribution built successfully. All **450 Python
modules** parse and match both archives byte for byte, including the two new
shared execution and manifest modules. Package name/version, Python/dependency
requirements and console entry points match the second-audit wheel. Whitespace
validation passed; all 67 local links across 12 audit documents resolve.

Independent execution cross-review reran all 50 new boundary/context cases
and executed all 15 predefined fluent words for two cycles on a two-node path
with EPI 0.5, frequency 1, pressure 0.2, phase 0 and fresh history/debt. Every
node's retained history matched its requested word.
Independent numerical review caught and corrected NumPy real-scalar timestep
compatibility; its final focused run passed 78 cases. The utility cross-review
supplied the antipodal-mean and structured-encoding-error counterexamples now
covered by the final 46 utility/reporting cases. No continuous-time convergence
or engine-wide performance certificate follows from these test totals.

Local evidence files are `tmp/third-audit-baseline.log`,
`tmp/third-audit-final-tests.log`, `tmp/third-audit-build.log`, and
`tmp/verify_third_audit_artifacts.py`. Wheel/source archives are under
`tmp/third-audit-dist/`.

Reproduction from the repository root:

```powershell
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
.venv312/Scripts/python.exe -m build --outdir tmp/third-audit-dist
.venv312/Scripts/python.exe tmp/verify_third_audit_artifacts.py
```

Standalone experiments must insert the working `src` directory or set
`PYTHONPATH=src`; the environment also contains an installed package, and
pytest's source-path configuration does not apply to ordinary Python invocations.
