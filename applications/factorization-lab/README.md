# TNFR spectral factorization lab — archived

**Reference only; not maintained or part of the current application workflow.**
The source and historical outputs remain at their original paths to preserve
uncommitted changes and recorded path associations. The
[retirement record](RETIREMENT.json) identifies retained source, recovery
revisions, retired integration tests and concrete counterexamples.

This pipeline combines supplied Jacobi-residue graphs, spectral heuristics,
arithmetic statistics and partition workflows. Its acceptance and completeness
labels are not reliable factor certificates. It establishes neither autonomous
factor recovery from the nodal equation nor a complexity improvement or a
cryptographic capability. This page records the archived implementation's scope;
it is not an installation or execution recommendation.

The maintained [arithmetic utility](../primality-test/README.md) and
[number-theory owner](../../theory/TNFR_NUMBER_THEORY.md) retain exact
arithmetic and independently proved residue-graph results. The Paley, CRT and
cyclotomy constructions have their own hypotheses; sharing a graph operator
does not validate this application's candidate pipeline.

## Results and verification

The retirement is based on implementation counterexamples, not only a change
of research priority:

- The structural verifier can accept candidate `6` for target `35` while its
  own `support_divisible` field is false.
- An integer `size_hint` can enable an exception to the stabilization condition.
- A composite signature for `35` with candidates `[5, 7, 11]` can report
  completeness although their product is `385`.
- Both modes compute arithmetic telemetry by factoring the input. Integer-size
  candidates, divisibility checks and trial-division fallback remain involved;
  a spectral result does not isolate the source of factor information.

| Archived result | Actual boundary |
| --- | --- |
| `candidate_factors` | Heuristic seeds, arithmetic refinements and partition hints; neither complete nor necessarily prime |
| `tnfr_certified_factors`, `tnfr_verification` | Configured acceptance scores with a separate divisibility observation; the acceptance rule need not require divisibility |
| `factor_signature`, `composite_signature` | Serialized records; hashes check recorded content, not the truth of powers or completeness |
| `certificate_path`, partition paths | Analysis/workflow records; a proposed word is not an executed operator trace |

Independent proper-divisor, primality, product and multiplicity checks are
necessary before interpreting any historical factor list. Passing them would
not show that recovery came from a TNFR mechanism.

## Setup and public interfaces

The engine no longer adds this directory to its import path. The deprecated
`tnfr.factorization.factorize`, `TNFR.factorize` and `Network.factorize`
interfaces remain compatibility bridges for a separately supplied legacy
`tnfr_factorization` package. Missing or archived-checkout implementations
produce retirement guidance; an external implementation produces a deprecation
warning. Keeping these names does not maintain or certify the archived lab.
See the [API migration](../../docs/API_CONTRACTS.md#arithmetic-application-boundaries).

Generic manifest preparation, dry-run assessment and report validation remain
engine responsibilities. They do not bootstrap this lab, reconstruct a residue
graph from a modulus, or apply a proposed operator word implicitly.

The retained source owners are the [lab API](tnfr_factorization/api.py),
[candidate pipeline](tnfr_factorization/spectral_paley.py),
[partition planner](tnfr_factorization/partitioning.py),
[seed handling](seed_management.py) and [snapshot records](snapshot_system.py).
Their original CLI and helper files are historical source, not supported
entry points. The lab has no separate `setup.py` or `pyproject.toml`.

## Candidate policy and configuration

Historical `pure=True` changes a process-wide environment policy for the call.
It skips arithmetic-statistic hints and gcd refinement in the initial candidate
stage, but does not make the complete pipeline arithmetic-free:

- `_compute_arithmetic_telemetry` factors the input to obtain `Omega`, `tau`
  and `sigma` in both modes.
- Even-number hints, integer-size seeds and later partition-size divisibility
  checks remain.
- Empty initial candidates reach trial division without a pure-mode guard;
  its default bound is `isqrt(n)`.
- `TNFR_PURE_MODE_VERIFY_DIVISIBILITY` filters one initial stage, not every
  later candidate or final acceptance label.

Automatic modulus selection rounds upward to at least `n` and `5`, with
residue `1 mod 4`, without proving primality. Distinct inputs can therefore
share a graph (`35` and `37` both select modulus `37`). Explicit overrides
enforce only the lower bound `5`, while prime-modulus Paley identities require
their own domain. Support is supplied arithmetically, not formed by a nodal law.

The low-level constructor defaults to a node cap of `4097`; `None` removes it.
The historical high-level wrapper and CLI pass `None` when omitted.
CLI zero also removes the cap, unlike the constructor. These differences
matter when interpreting retained records, as do cache, backend and mode.

Archived JSON projections encode nonfinite diagnostics as `null` with
`numeric_availability` reasons. When projection changes the payload,
`numeric_projection.retained_hash_scope` distinguishes internal hash inputs
from exported bytes. This is neither provenance authentication nor a guarantee
that the underlying diagnostic has physical meaning.

## Telemetry and mechanism boundaries

The lab's field names differ from canonical `tnfr.physics.fields`:

| Lab field | Computation or source |
| --- | --- |
| `phi_s` | Normalized edge density, without nodewise pressure or path distance |
| `phase_gradient` | Selected spectral gap divided by node count |
| `phase_curvature` | Largest supplied eigenvalue divided by node count |
| `coherence_length` | Positive backend value, otherwise inverse selected gap |
| Arithmetic EPI, capacity, pressure | Static factor/divisor readouts without phase or autonomous evolution |
| Energy/Noether proxies | Algebraic lab features, without an established balance or conserved charge |

On regular unit-weight graphs the fallback combinatorial spectrum is
`d * spectrum(L_rw)`. Its inverse selected gap is not the normalized diffusion
timescale `1 / (nu * lambda_2(L_rw))` without degree/time conversion. The
selection discards eigenvalues at or below `1e-9`, so it need not select the
true second eigenvalue.

Partition pressure reduction includes a fixed attenuation surrogate. It is
not operator execution or a convergence theorem. Grammar-valid proposals,
configured thresholds and finite score changes do not prove U5, autonomous
selection, conservation or a factorization mechanism.

<a id="verification-and-historical-evidence"></a>
## Verification and evidence

Archived tests and benchmark source are retained for interpretation and
recovery, outside the maintained test selection and automatic validation maps.
Their finite passing cases do not override the counterexamples above.
Do not rerun frozen producers to update these records.

Historical JSON, notebook records and captured paths retain their original
bytes. Partition hashes cover selected members, telemetry and metadata in
supplied order, excluding other state; they are not complete checkpoints.
Snapshot demonstrations use synthetic persistence records.

The current [testing policy](../../TESTING.md) covers independent core
arithmetic controls, the basic primality utility and compatibility admission.
The root [LICENSE.md](../../LICENSE.md) remains authoritative.
