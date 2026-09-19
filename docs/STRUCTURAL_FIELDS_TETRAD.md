# TNFR Structural Field Tetrad

**Status:** Canonical read-only telemetry. This guide defines the fields,
current API behavior, and numerical safety policies. The distinction between
kinematic bounds, selected thresholds, and open reconstruction claims is
centralized in
[Mathematical Scope of Structural Diagnostics and Grammar](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

The four channels are structural potential Φ_s, phase-gradient magnitude
|∇φ|, circular curvature K_φ, and coherence length ξ_C. They complement the
structural triad and global C(t)/Si measurements. They do not by themselves
reconstruct every graph state or determine its evolution.

Exact closure of a coarse nodal evolution does not automatically preserve
these field formulas on the coarse graph. In particular, aggregating fine
potential can retain sources inside each coarse node that the self-excluded
macro formula removes. Preserve the observation map and metric separately
from the dynamical quotient; the
[prism inheritance test](../theory/NODAL_PARAMETER_FOUNDATIONS.md#13-faithful-macro-state-and-tetrad-inheritance-on-the-retained-prism)
gives an exact counterexample and the correct inherited kernel.

A separate [restricted prism identity](../theory/NODAL_PARAMETER_FOUNDATIONS.md#183-the-same-oriented-area-is-readable-through-the-tetrad)
recovers relative phase/form oriented area from projected potential and
curvature, using known channel coefficients and fresh pressure. It requires
the repeated unit-distance support and retains a pressure-defect correction;
it neither reconstructs absolute common phase/EPI nor introduces a feedback
rule or a fifth canonical field.

## 1. Physics basis

The nodal equation relates structural change to capacity and pressure:

    ∂EPI/∂t = ν_f ΔNFR.

Its finite-interval integral gives accumulated form change. Infinite-horizon
boundedness, convergence, and absolute integrability require distinct
hypotheses. Grammar U2 prescribes stabilizers and debt accounting; it does not
by itself prove convergence of every admissible trajectory.

These field functions read graph attributes and do not mutate EPI. Operator
applications remain subject to their contracts and
[U1–U6](../theory/UNIFIED_GRAMMAR_RULES.md). Frequency is measured in Hz_str.

### Readout ownership and checkpoints

Readouts require the graph's topology, attributes and configuration to remain
fixed for the duration of the call. Capture telemetry between completed
operator applications. Composite field suites read each required base field
once, detach its map and reuse that local collection; later graph evolution
does not rewrite the returned maps.

Public potential, phase, current and pressure-flux dictionaries are detached
from internal cache entries. Editing a returned map does not alter a later
direct read or another composite suite. Cache reuse remains internal; public
dictionary identity is not an API guarantee.

The words "capture" and "snapshot" describe the returned data, not an atomic
transaction with concurrent writers. If multiple threads share a graph, its
owner must serialize both evolution and the complete readout with the same
lock, or supply an independently owned graph copied while writers are stopped.
An internal reader-only lock cannot protect direct NetworkX attribute writes.
The engine does not install such a lock or silently copy the graph per metric.
These rules apply to the unified and variational suites and to conservation
and Lagrangian snapshots as well as individual field functions.

## 2. Canonical field definitions

Public imports are available from
[tnfr.physics.fields](../src/tnfr/physics/fields.py); the implementations are in
[canonical.py](../src/tnfr/physics/canonical.py). Initialize the canonical
phase and ΔNFR attributes; the shared alias resolver supports the engine's
alternate names.

The first present alias is authoritative. Field readers validate its raw
phase/pressure value as a finite binary64 real before consulting their caches;
booleans, numeric strings and nonfinite values are not silently coerced into
valid data. A nonzero source that cannot survive binary64 representation also
raises. Missing channels retain the existing zero convention, and a valid
primary alias is not overridden by a malformed secondary one. These are
source-admission rules, not a guarantee that every derived sum is representable.
Independent phase readouts do not require a valid pressure channel.

### 2.1 Structural potential Φ_s

For reachable nodes at positive distance,

    Φ_s(i) = Σ_(j ≠ i) ΔNFR_j / d(i,j)^α,    α = 2 by default.

The default evaluates exact shortest-path distances at every graph size.
“Exact” means the graph distance computation with floating-point sums.
Distances follow outgoing arcs for a directed graph. The explicit `length`
edge attribute defines geometry when present. For backward compatibility, an
edge without `length` uses its `weight`; an edge with neither defaults to 1.
The `weight` attribute remains the diffusion conductance, so experiments in
which conductance and distance differ must set both attributes. Unreachable and
zero-distance pairs contribute zero under the current compatibility behavior.
Parallel edges use minimum effective path lengths.

For a fixed distance kernel in exact arithmetic, Φ_s = B_G ΔNFR is linear in
pressure. Therefore

    ||Φ_s||∞ ≤ ||B_G||∞ ||ΔNFR||∞.

The implemented floating-point path sums, powers and source accumulation retain
rounding; this identity does not assert exact linearity of runtime arithmetic.

A change in graph topology or edge length changes B_G as well as any pressure
response. Uniform pressure scaling generally produces nonzero potential drift;
it must not be treated as a cache-invariance property.

The per-node π/4 and U6 drift π/2 values are selected safety policies.
They are not consequences of phase wrapping: on K₄, unit pressure and zero
phase give Φ_s = 3 at every vertex. Establishing a stricter bound requires
additional assumptions on pressure, geometry, normalization, and evolution.

The inverse-square exponent is the canonical choice. Chain summability does
not uniquely derive α = 2; all α > 1 give finite absolute chain sums. Preserve
the canonical exponent unless a study explicitly specifies an alternate
kernel and its interpretation.

### 2.2 Phase-gradient magnitude |∇φ|

    |∇φ|(i) = mean_(j in N(i)) |wrap(φ_j − φ_i)|.

This is the mean magnitude of wrapped neighbor phase differences. Its exact
kinematic bound is **|∇φ| ≤ π**. The canonical warning level **π/16 ≈ 0.19635**
is a selected threshold, not a universal synchronization transition.
Experimental onsets depend on the frequency spread, graph, and protocol.

The magnitude discards signs and local ordering. It can identify the location
of phase stress that a single global C(t) does not report, but it is not a
complete reconstruction of the phase field.

<a id="phase-curvature"></a>
### 2.3 Circular phase curvature K_φ

    K_φ(i) = wrap(φ_i − Arg(Σ_(j in N(i)) exp(i φ_j))).

For a defined circular neighbor direction this is its wrapped deviation.
The shared read-out materializes binary64 trigonometric components and sums
those components exactly. A nonzero represented resultant uses its numerical
direction without the old `1e-9` arithmetic-angle fallback. The exact wrapped
bound on defined values is **|K_φ| ≤ π**; the warning **0.9π ≈ 2.82743** is an
operational margin, not a conditioning certificate.

`observe_phase_curvature(G)`, exported through `tnfr.physics.fields`, returns
detached `PhaseCurvatureObservation` evidence with per-node neighbors,
gradient, optional curvature, status and represented resultant. At exact joint
zero of the materialized components, curvature is `None` with status
`undefined_represented_resultant`. Numeric `compute_phase_curvature` and
full-field telemetry instead raise `UndefinedPhaseCurvatureError`; they do
not silently report zero or a valid safety decision. `compute_phase_gradient`
still works on that domain. Isolates explicitly use
`isolated_zero_convention`, distinct from cancellation of a nonempty neighborhood.

Exact reduction does not certify exact sine/cosine values or a uniform angle
error. A zero represented sum can differ from the exact-real trigonometric
sum at those input angles. The evidence records requested precision separately
from the binary64 component and approximate-angle semantics. No supplied
precision label turns ill-conditioned data into a certified direction.
Malformed authoritative phase aliases fail before a cached result is reused;
missing phase retains its established zero default. This is a diagnostic
correction, with no replacement of the pressure or phase-evolution kernels.
The immutable observation may be reused from cache. All public numeric field
maps are detached; full structural telemetry returns a fresh container and
copies every nested field map, so caller edits cannot overwrite cached evidence.

`TelemetryEmitter(safe=True)` collects the tetrad fields independently even
when a requested composite suite fails. It retains the available fields,
omits an unavailable value, and records
`metrics["field_errors"][field]={"type": ..., "message": ...}`. Undefined
curvature therefore does not hide valid potential, gradient or correlation
readouts. Strict mode propagates the error. `include_extended=False` skips
the composite suites and collects only the core metrics and individual tetrad.
Successful unified collection reuses its extended block instead of calculating
that suite twice. If it fails, safe mode can still collect the extended fields
independently.
JSON emission supports NumPy arrays/scalars through the shared JSON writer;
unsupported objects still raise, and serialization completes before opening
the output file. Safe field collection does not silently discard export errors.
The optional human mirror displays unavailable core metrics explicitly and is
formatted before either output file is opened. This prevents formatting errors
from leaving an appended JSON batch queued for retry; it is not an atomic
transaction across the two files or a guarantee against filesystem failures.

For small phase spread on a consistent branch, and matching neighbor-weight
conventions, the circular mean approaches the arithmetic mean and K_φ
approaches L_rw φ. This is a linearization; it is not an exact identity for
arbitrary wrapped configurations.

The multiscale utilities fit curvature-variance decay under a specified
coarse-graining protocol. A fitted exponent or the utility's default exponent
is not a topology-independent law.

<a id="coherence-length"></a>
### 2.4 Coherence length ξ_C

The current estimator forms pressure-derived local coherence

    c_i = 1/(1+|ΔNFR_i|),

then fits distance-binned products to an exponential profile:

    mean_(d(i,j)=r) c_i c_j ≈ A exp(−r/ξ_C).

This is the canonical coherence kernel evaluated with dEPI = 0 and an
**uncentered static product fit**, not connected covariance or the full runtime
coherence. The scalar and vector paths share
[_coherence_fit.py](../src/tnfr/physics/_coherence_fit.py): shortest-path distances
use explicit edge `length`, else compatibility `weight`, else one; parallel
lengths combine by minimum. Undirected pairs are counted once and directed
pairs follow outgoing reachable paths. Distinct-node zero distances are
omitted, so zero-length edges describe a pseudometric rather than a separating
undirected metric. Directed distances can be asymmetric.

The declared fit policy requires ten positive-distance pairs, two pairs per
exact represented distance bin, and three bins with mean product above `1e-9`.
Only a finite negative log-linear slope giving a finite positive length is
accepted; this is not a goodness-of-fit test. Below 1000 nodes every pair is
used. Larger graphs use the same deterministic insertion-order source sample
in both backends. Node/source order is part of the fit cache key; the outer
telemetry cache also binds neighbor order and the numerical path. Pressure,
both edge channels and precision mode participate in cache invalidation.

Negative or nonfinite effective edge lengths and overflowing reachable path
distances now raise `ValueError`; invalid geometry is not a failed fit and
does not trigger spectral fallback. The vector helper's optional distance
matrix must satisfy its shape, numeric-domain, diagonal and undirected-symmetry
contract. Such a matrix remains caller-declared data, not authenticated shortest
paths. Positive infinity marks omitted pairs, while NaN and negative sentinels
are rejected intentionally.

When valid inputs supply no usable fit, the separate fallback is
`1/sqrt(lambda_positive)` from the normalized graph-Laplacian eigenvalues.
The implementation selects the smallest eigenvalue above `1e-9`, a numerical
mode-selection policy. On an admitted connected symmetric graph with a
resolved positive gap this corresponds to λ₂. Disconnected graphs do not
thereby acquire cross-component correlation, directed graphs cannot use this
symmetric fallback, and unavailable estimates remain NaN. A fitted ξ_C has
the declared path-length units; the normalized-generator fallback is
dimensionless and does not scale with an independent explicit length.

Use `estimate_coherence_length_with_provenance(G)` to retain the method,
distance/units, source selection, fit policy and graph regime.
`estimate_coherence_length(G)` retains its scalar return for compatibility;
neither function accepts a `coherence_key` argument. A large estimate, flat
profile or failed fit alone proves no critical transition. The weighted-star
controls in [the distance-contract tests](../tests/physics/test_coherence_distance_contract.py)
check a known exponential product and its distance scaling without a trajectory
or a physical correlation claim.

### Observation completeness across scale

Preserving averaged EPI and potential does not necessarily preserve this
nonlinear coherence estimator. The exact P5 reflection example has identical
three-coordinate EPI/potential observations but distinct successful product
fits. The [scale bridge](../theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md)
owns the pressure-factorization criterion, numerical field provenance and
counterexamples; [P5 reflection invariants](../theory/DERIVED_EPI_MEMORY.md)
retain the hidden shape up to a specified reflection without reducing its
generic continuous dimension. These are observation results for held fine
models. Primitive phase, capacity, support and lengths remain declared inputs,
and the four diagnostic names do not certify full-state reconstruction.

## 3. Contracts, units, and edge cases

- Telemetry does not mutate EPI or replace the operator execution path.
- ν_f remains in Hz_str; phase comparisons use wrapped angular differences.
- Coupling and Resonance require actual phase verification under U3.
- Sequence and scale obligations remain those of U1–U5.
- Isolated vertices return zero local gradient and curvature. Only reachable
  positive-distance sources contribute to potential.
- A nonempty neighborhood with exact represented phasor cancellation has
  unavailable curvature; it does not satisfy a curvature safety check by default.
- Initialize required attributes explicitly for reproducible studies instead
  of relying on missing-value fallbacks.

The fields have different topology conventions. Potential uses effective
shortest-path lengths (`length`, then the legacy `weight` fallback), while local
phase readouts average graph neighbors and diffusion uses `weight` as
conductance. A study comparing geometric and spectral quantities must declare
both edge channels when their physical meanings differ.

## 4. API summary

| Public call | Purpose |
|-------------|---------|
| compute_structural_potential(G, alpha=2.0, ...) | Full-source numerical potential by default; optional landmark approximation |
| compute_phase_gradient(G) | Per-node wrapped mismatch magnitude |
| compute_phase_curvature(G) | Per-node circular curvature; raises on undefined represented direction |
| observe_phase_curvature(G) | Immutable per-node resultant evidence, availability and independent gradient |
| estimate_coherence_length(G) | Scalar correlation estimate with spectral fallback |
| compute_k_phi_multiscale_variance(G, scales) | Research utility for scale-dependent curvature variance |
| fit_k_phi_asymptotic_alpha(var_by_scale) | Fit a variance-decay exponent |
| k_phi_multiscale_safety(G, ...) | Configured multiscale diagnostic |
| fit_correlation_length_exponent(Is, xi_vals, I_c, ...) | Fit a critical exponent under a supplied critical-point model |
| measure_phase_symmetry(G) | Phase-symmetry readout |
| path_integrated_gradient(G, source, target) | Gradient along a graph path |
| compute_phase_winding(G, cycle_nodes) | Winding on a supplied cycle |

Use function docstrings for complete signatures and backend details. Extended
phase and pressure currents remain separate readouts available from the same
public module.

For structural potential, landmark_ratio explicitly opts into approximation.
With signed pressure, landmark path lengths do not bound relative potential
error. The validate option compares against the full-source numerical field,
refines the approximation and falls back to that evaluation when required.
Here “exact” distinguishes full-source evaluation from landmark sampling; it
does not remove path-sum, inverse-power or accumulation rounding. Use the
full-source default for U6 decisions, with its numerical scope retained.

## 5. Validation and safety thresholds

Current values are centralized in
[constants/canonical.py](../src/tnfr/constants/canonical.py).

| Quantity | Value | Interpretation |
|----------|-------|----------------|
| Per-node potential warning | π/4 | Selected potential policy |
| U6 potential drift limit | π/2 | Selected before/after policy |
| Phase-gradient maximum | π | Exact wrapped-angle bound |
| Phase-gradient warning | π/16 | Selected early-warning level |
| Curvature maximum | π | Exact wrapped-angle bound |
| Curvature warning | 0.9π | Selected margin below π |
| Correlation-length comparisons | Diameter and configured mean-distance ratios | Diagnostic policy, not a proof of criticality |

Record the baseline, node aggregation, pressure scale, graph geometry, and
operator sequence for potential drift. Crossing a threshold flags a policy
condition; it does not independently prove fragmentation. U1–U5 word
acceptance does not guarantee that every later field reading will pass U6.

Operator verification follows the specific
[API contracts](API_CONTRACTS.md), including their local/event scope and
state-dependent preconditions. Field verification should test definitions and
edge cases, including pressure scaling and wrapped phases. Reproducibility
requires declared inputs, execution order and backend as well as a seed; no
universal monotonicity or post-Silence freezing follows from an operator name.

## 6. Minimal read-only example

~~~python
import networkx as nx
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

G = nx.watts_strogatz_graph(60, k=4, p=0.2, seed=42)
for node in G:
    G.nodes[node]["theta"] = 0.1 * node / 59.0
    G.nodes[node]["delta_nfr"] = 0.1

potential = compute_structural_potential(G)
gradient = compute_phase_gradient(G)
curvature = compute_phase_curvature(G)
coherence_length = estimate_coherence_length(G)
~~~

This initializes a deterministic field fixture; it is not an operator
trajectory or a claim that the resulting state passes every safety threshold.

## 7. Interpretation and research tools

Canonical C(t) is a global scalar and is amplitude sensitive: at dEPI = 0,
uniform |ΔNFR| = 1 gives C = 1/2 and doubling it gives C = 1/3. The separate
normalized dispersion statistic is invariant under positive pressure scaling
when its denominator is nonzero. Neither scalar identifies local phase-stress
locations.

Operator responses depend on initial graph state, gains, and context. An
operator-tetrad fingerprint measured in one experiment is a property of that
protocol, not an injective identification of every operator on every state.
Retain the initial state and full operator trace when comparing responses.

The research tools include
[the integrated field study](../benchmarks/integrated_force_regime_study.py),
[operator-tetrad experiments](../examples/02_physics_regimes/37_operator_tetrad_synergy.py),
and [nodal-channel decomposition](../examples/02_physics_regimes/39_nodal_equation_decomposition.py).
Numerical findings must be interpreted with their recorded seeds, topology,
parameters, and measurement definitions.

## Appendix: Topological winding

On a supplied closed cycle,

    Q = round((1/(2π)) Σ wrap(φ_(i+1) − φ_i)).

This complementary telemetry can distinguish phase winding under the
implementation's cycle and branch conventions. It does not belong to the
selected tetrad, and its usefulness illustrates why a canonical four-channel
readout should not be mistaken for a theorem excluding every additional
observable.

## References

- [AGENTS.md](../AGENTS.md): invariants and canonical synthesis.
- [Unified grammar](../theory/UNIFIED_GRAMMAR_RULES.md).
- [Tetrad reconstruction scope](../theory/MINIMAL_STRUCTURAL_DEGREES.md).
- [Mathematical hypotheses and witnesses](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
- [Field implementation](../src/tnfr/physics/fields.py).
