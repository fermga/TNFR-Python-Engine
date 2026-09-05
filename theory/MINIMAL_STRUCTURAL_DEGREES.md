# Structural Tetrad: Diagnostic Degrees and Reconstruction Scope

**Status:** Four canonical diagnostic channels. Universal minimality and
complete state reconstruction are open questions with explicit hypotheses
required. Definitions are centralized in
[the field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md); proofs and counterexamples
are centralized in [the mathematical scope note](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

## 1. Statement

The structural tetrad (Φ_s, |∇φ|, K_φ, ξ_C) is the engine's selected readout of
pressure accumulation, local phase stress, local circular curvature, and
spatial correlation. All four fields retain canonical status.

The selection is a diagnostic design. It does not establish that every scalar
observable is a function of four reported quantities, or that the tetrad
determines a full graph state and its evolution under

    ∂EPI/∂t = ν_f · ΔNFR.

A completeness claim must specify a state space, admissible observables,
gauge identifications, and what information is retained by the readout. A
minimality claim must additionally prove that no selected channel is redundant
for that specified task. These stronger claims are not established by naming
four structural questions or by computing the fields on a collection of graphs.

## 2. Four diagnostic questions

| Question | Field | Construction |
|----------|-------|--------------|
| How much pressure accumulates from other nodes? | Φ_s | Distance-weighted aggregation of ΔNFR |
| How misaligned is a node with its neighbors? | \|∇φ\| | Mean magnitude of wrapped phase differences |
| How does its phase differ from the neighborhood mean? | K_φ | Wrapped deviation from the circular mean |
| At what spatial scale does coherence correlate? | ξ_C | Correlation-decay fit with spectral fallback |

This organizes local phase information and non-local aggregation/correlation.
Φ_s is a per-node value with global dependence; it is not a purely local
quantity. ξ_C compresses a spatial correlation profile into a length. The table
does not exhaust every possible graph observable.

## 3. The operator-derivative tower

### 3.1 Construction

    ΔNFR → distance-kernel aggregation → Φ_s
    phase → wrapped neighbor differences → |∇φ|
    phase → circular-neighborhood deviation → K_φ
    local coherence → spatial correlation → ξ_C

The first two phase operations are related to first and second derivatives.
For an undirected graph, an oriented incidence matrix gives an edge gradient,
and composing it with its weighted adjoint gives the graph Laplacian.
For small phase spread within a consistent branch and matching neighbor
weights, circular curvature approaches L_rw φ. The equality is a linearized
description; the exact implemented field uses circular means and wrapping.

### 3.2 Algebraic generation does not terminate information at second order

Higher Laplacian powers are compositions of the same operator. This fact does
not mean their actions are linearly dependent or recoverable from lossy
pointwise summaries. If L has at least four distinct eigenvalues, then I, L,
L², and L³ are linearly independent. Otherwise a polynomial of degree at most
three would vanish at four distinct eigenvalues.

The finite-dimensional cutoff is controlled by L's minimal polynomial. It
depends on the graph and is not universally second order. In addition, |∇φ|
discards signs and ξ_C discards most of the correlation profile. A generating
set of operators and a sufficient set of state statistics are different
mathematical objects.

### 3.3 Dynamical non-reconstruction

At fixed graph, phase, EPI, and pure-EPI pressure, a uniform rescaling of ν_f
leaves the tetrad unchanged while scaling ∂EPI/∂t = ν_f ΔNFR. Consequently the
tetrad alone does not determine the full dynamical state. The structural triad,
frequency, pressure law, and graph remain part of the model.

## 4. Field scales and selected thresholds

The exact phase-wrap maximum π must be distinguished from policy values
expressed as fractions of π. The numerical values below remain unchanged in
[canonical.py](../src/tnfr/constants/canonical.py).

| Field | Exact or conditional mathematical scale | Canonical telemetry policy |
|-------|----------------------------------------|----------------------------|
| Φ_s | ‖Φ_s‖∞ ≤ ‖B_G‖∞ ‖ΔNFR‖∞ for a fixed distance kernel | Per-node π/4; U6 drift π/2 |
| \|∇φ\| | At most π by wrapped-angle definition | Warning level π/16 |
| K_φ | Absolute value at most π by wrapping | Warning level 0.9π |
| ξ_C | Spectral reference 1/√λ_gap when a positive graph gap is used | Configured diameter and mean-distance comparisons |

### 4.1 Structural potential

The canonical exponent is α = 2 in B_ij = d(i,j)⁻². The resulting field is
linear in ΔNFR and unbounded under pressure rescaling. Even at zero phase,
unit pressure on K₄ gives Φ_s = 3. No phase-wrap argument can enforce π/4 or
π/2 for this linear field without an additional pressure/geometry constraint.

The chain sums Σ d⁻² = ζ(2) and Σ d⁻⁴ = ζ(4) are correct particular values.
They do not uniquely select exponent 2: absolute chain sums converge for all
α > 1 and independent unit-variance pressure sums for all α > 1/2. Graph
families with growing distance shells need their own summability assumptions.
The inverse-square kernel is the canonical modeling choice, not a uniqueness
theorem from those series.

### 4.2 Phase gradient

The mean of absolute wrapped neighbor differences is bounded by π. The
warning value π/16 is selected policy; a measured synchronization onset depends
on the experiment's frequency spread, topology, and coupling protocol.

Canonical aggregate coherence is

    C(t) = 1 / (1 + mean|ΔNFR| + mean|dEPI|).

It does not identify the spatial location of phase stress. It also is not
scale invariant: with dEPI = 0 and uniform |ΔNFR| = 1, C(t) = 1/2; doubling
pressure gives 1/3. Only the separate normalized dispersion statistic
C_disp = 1 − σ_ΔNFR/ΔNFR_max is invariant under proportional positive pressure
scaling when its denominator is nonzero.

### 4.3 Circular curvature

K_φ = wrap(φ_i − circular_mean(neighbors)) gives |K_φ| ≤ π by definition.
The 0.9π threshold is an operational margin inside that exact bound.
Near-antiphase configurations also require care with the circular mean when
the neighbor resultant is small. A hotspot is a diagnostic flag, not a proof
of a singularity in an underlying continuous manifold.

### 4.4 Correlation length

The current estimator first fits decay of a pressure-derived coherence
correlation against graph distance. If that fit is unsuitable, it uses a
spectral reference from the graph. On a connected undirected graph this is
1/√λ₂. A fitted correlation length need not equal that reference for every
state; exponential spatial decay is itself a fitting assumption.

On disconnected graphs, a second-smallest eigenvalue and a smallest positive
eigenvalue need not coincide. Reports must identify the estimator/fallback and
graph regime. A failed fit or large estimate alone does not prove a phase
transition.

## 5. Diagnostic removal studies

Each field addresses a distinct diagnostic task: accumulated pressure, local
mismatch magnitude, local curvature, or correlation range. This motivates
retaining all four channels. A formal non-redundancy study should construct
explicit pairs of states with identical retained readouts and a differing
omitted readout, or specify a detection problem and prove the omitted channel
changes the decision.

Descriptions such as “a blind spot appears” require the actual fixture,
readout resolution, and detection threshold. Passing many implementation
tests does not prove universal irreducibility. Nor does usefulness of each
channel prove that no additional observable can carry independent information.

The strong claim of a minimal complete representation therefore remains open
until its target state space and equivalence relation are specified and the
appropriate injectivity and non-redundancy results are proved.

## 6. Energy, currents, and conservation

The tetrad can participate in energy-like diagnostics with phase and pressure
currents:

    V = ½ Σ_i (Φ_s(i)² + |∇φ|(i)² + K_φ(i)²),
    T = ½ Σ_i (J_φ(i)² + J_ΔNFR(i)²).

These are nonnegative functionals of measured fields. Their existence does not
establish state reconstruction, does not make ΔNFR equal to −∂V/∂EPI, and
does not turn all implemented operators into Hamiltonian transformations.
Conjugate coordinates belong to a separately specified substrate model.

Likewise a continuity residual or measured energy descent is evidence about
the tested trajectory. Universal Noether conservation and asymptotic stability
require the stated symmetry, flow, and Lyapunov hypotheses. The mathematical
scope is developed in [the variational note](TNFR_VARIATIONAL_PRINCIPLE.md) and
[the conservation note](STRUCTURAL_CONSERVATION_THEOREM.md); the tetrad's
canonical diagnostic status does not supply those proofs.

## 7. Symmetry-restricted research

Symmetry can restrict which sectors a graph readout detects. A statement about
the tetrad on a particular prime-ladder graph with uniform parameters must
name that graph, its action, and the parameter restrictions. Invariance under
that action constrains reach; it does not imply that the tetrad spans every
vector in the invariant sector.

The REMESH-∞ and Riemann programs use separate projection and observability
constructions. Their particular graph certificates do not establish a
universal complete “smooth structural sector” on arbitrary networks.
See [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) and
[TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) for the precise
program assumptions and open conjectures.

## 8. Verification scope

| Claim | Supported scope |
|-------|-----------------|
| Four canonical field APIs exist | Definitions and implementation contracts |
| Phase magnitudes are at most π | Exact wrapped-angle bounds |
| Potential is linear in pressure | Exact fixed-kernel identity |
| Potential has a uniform π-fraction maximum | False without additional pressure/geometry assumptions |
| Three operations bound every diffusion relaxation time | False; the 21-node path is a counterexample |
| Laplacian compositions prove tetrad completeness | False inference; operator generation is not reconstruction |
| Tetrad is a minimal sufficient statistic for a specified state quotient | Open; quotient and observable class must first be specified |

Operator gains, discretization steps, numerical clamps, and safety thresholds
remain configured policies unless an explicit derivation with its assumptions
is supplied. A formula using π alone is not sufficient evidence of a physical
necessity.

## References

- [AGENTS.md](../AGENTS.md): canonical synthesis.
- [Field definitions and APIs](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- [Mathematical scope and exact witnesses](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
- [Unified grammar](UNIFIED_GRAMMAR_RULES.md).
- [Fundamental theory](FUNDAMENTAL_THEORY.md): broader theory context.
- [Finite-graph witnesses and scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
