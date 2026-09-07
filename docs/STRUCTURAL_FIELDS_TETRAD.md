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

On a fixed graph, Φ_s = B_G ΔNFR is exactly linear in pressure. Therefore

    ||Φ_s||∞ ≤ ||B_G||∞ ||ΔNFR||∞.

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

### 2.3 Circular phase curvature K_φ

    K_φ(i) = wrap(φ_i − Arg(Σ_(j in N(i)) exp(i φ_j))).

The circular neighbor mean respects phase periodicity. For nondegenerate
neighbor resultants this is the wrapped deviation from their mean angle;
the implementation retains its defined fallback for ambiguous means.
The exact bound is **|K_φ| ≤ π**. The warning threshold **0.9π ≈ 2.82743**
is an operational margin.

For small phase spread on a consistent branch, and matching neighbor-weight
conventions, the circular mean approaches the arithmetic mean and K_φ
approaches L_rw φ. This is a linearization; it is not an exact identity for
arbitrary wrapped configurations.

The multiscale utilities fit curvature-variance decay under a specified
coarse-graining protocol. A fitted exponent or the utility's default exponent
is not a topology-independent law.

### 2.4 Coherence length ξ_C

The current estimator forms pressure-derived local coherence

    c_i = 1/(1+|ΔNFR_i|),

then fits distance-binned products to an exponential profile:

    mean_(d(i,j)=r) c_i c_j ≈ A exp(−r/ξ_C).

This pressure-only local quantity is the canonical coherence kernel evaluated
with dEPI = 0. It is a correlation readout; the full global coherence
C(t) also includes mean|dEPI|.

When the fit cannot supply a usable positive decay length, the implementation
uses a graph-spectral fallback 1/√λ_gap. The fallback currently selects the
smallest positive eigenvalue returned by structural_eigenmodes; on a connected
undirected graph this is λ₂. On disconnected graphs it does not measure
correlation across components, and with no positive modes the result may be
NaN. A fitted length and a spectral fallback are distinct measurements.

A large estimate, a flat correlation profile, or a failed fit does not alone
prove a critical transition. Report the graph regime and fitting/fallback
method when interpreting the value. The current public function accepts G
only; it does not accept a coherence_key argument.

## 3. Contracts, units, and edge cases

- Telemetry does not mutate EPI or replace the operator execution path.
- ν_f remains in Hz_str; phase comparisons use wrapped angular differences.
- Coupling and Resonance require actual phase verification under U3.
- Sequence and scale obligations remain those of U1–U5.
- Isolated vertices return zero local gradient and curvature. Only reachable
  positive-distance sources contribute to potential.
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
| compute_structural_potential(G, alpha=2.0, ...) | Exact potential by default; optional landmark approximation |
| compute_phase_gradient(G) | Per-node wrapped mismatch magnitude |
| compute_phase_curvature(G) | Per-node circular curvature |
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
error. The validate option compares against the full exact field, refines the
approximation and falls back to exact evaluation when required. Use the exact
default for U6 decisions.

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

Operator verification retains Coherence monotonicity, controlled bifurcation,
Resonance propagation, Silence latency, Mutation threshold behavior, nested
identity, and same-seed reproducibility. Field verification should test exact
definitions and edge cases, including pressure scaling and wrapped phases.

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
