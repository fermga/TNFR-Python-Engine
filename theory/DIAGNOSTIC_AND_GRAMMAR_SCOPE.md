# Mathematical Scope of Structural Diagnostics and Grammar

**Status:** Exact identities under stated hypotheses, canonical engine policies,
and open sufficiency questions. This note fixes the distinction between these
three categories without changing the operator catalog or numerical policies.

The nodal equation is

    x'(t) = ν_f(t) p(t),    x = EPI,    p = ΔNFR.

It specifies the relation between structural change, capacity, and pressure.
A theorem about its solutions also needs a pressure law, graph assumptions,
initial conditions, and, for a discrete evolution, an integrator and step size.
Operator contracts supply additional constraints; they do not follow from the
scalar identity without those premises.

## 1. Existence, boundedness, and convergence

If f(t) = ν_f(t)p(t) is locally integrable on a finite-dimensional structural
coordinate space, then

    x(t) = x(t₀) + ∫[t₀,t] f(s) ds

defines an absolutely continuous trajectory on each finite interval, with the
nodal equation holding almost everywhere. Continuity of f gives the usual
pointwise derivative. In particular, x = 0 is not a singularity of this equation:
ν_f = 1 and p = 1 give x' = 1 at zero. U1's generator requirement is an
initialization contract for the operator language, including its supported
pre-existing-structure context, rather than a claim that this derivative is
undefined.

On an infinite horizon the following properties are different:

| Property | Exact statement |
|----------|-----------------|
| Bounded trajectory | The partial integrals of f remain bounded. |
| Convergence to a finite limit | The vector improper integral of f converges. |
| Absolute integrability | The integral of the norm of f is finite. This is sufficient for convergence and gives finite total variation. |

Neither boundedness nor p(t) tending to zero establishes the last two properties:

- x(t) = sin(t), ν_f = 1, p(t) = cos(t) is bounded and satisfies the nodal
  equation, but x(t) has no limit.
- p(t) = 1/(1+t), ν_f = 1 gives x(t) = x(0) + log(1+t). Its pressure is bounded
  and tends to zero, while its accumulated displacement diverges.
- On a finite connected undirected graph, fixed homogeneous ν_f > 0 and the
  pure EPI law p = −L_rw x give autonomous negative feedback. Relaxation does
  not mathematically require an additional named operator during that flow.

An example of a sufficient analytic assumption is

    ||ν_f(t)p(t)|| ≤ M exp(−a(t−t₀)),    M < ∞, a > 0.

Then the tail displacement is at most (M/a) exp(−a(t−t₀)). For repeated operator
applications, proving such a bound requires control of their gains, elapsed
times, and intervening feedback. U2's prescribed stabilizers and debt accounting
remain mandatory engine policy; grammar acceptance by itself is not an
infinite-horizon convergence certificate.

## 2. Modal relaxation and the grammar calibration

On a fixed undirected graph with positive edge weights, homogeneous frequency,
and pure EPI diffusion, each nonstationary mode of an explicit Euler step has
multiplier

    q_k = 1 − ν_f dt λ_k.

Its amplitude after n steps is q_kⁿ times the initial amplitude. Decay requires
|q_k| < 1; a negative multiplier alternates sign and need not decay. The
continuous solution instead has multiplier exp(−ν_f λ_k t). These are different
evolution laws at finite dt.

For a loopless graph without isolated vertices, trace(L_rw)/N = 1. This is the
mean of all eigenvalues, including stationary modes, not the relaxation rate
of each mode. Isolates and self-loops require different trace accounting.

The functions derive_bifurcation_window_from_physics and
derive_u2_debt_capacity_from_physics retain their public names and formulas.
They use the fixed surrogate rate ρ = 1 and q = 1 − ν_f dt ρ. At the canonical
ν_f = 1 and dt = 0.5:

- The first n with qⁿ < 1/(π+1) is **3**, the U4b recency policy.
- The floor of 1/(1−q) is **2**, the U2 debt-capacity policy.

The geometric sum describes a scalar recurrence with unit forcing and
0 ≤ q < 1. Its finite steady state is not a maximum physical pressure or a
criterion for integrability of a sustained forcing. Mapping its floor to a
count of destabilizing operators is the engine's calibration choice.

The actual functions also retain their finite fallbacks: a nonpositive q returns
a one-operation window, and the search stops at 64; nonpositive ν_f dt returns
zero debt capacity. These are compatibility policies, not stability results for
the corresponding Euler modes.

**Finite-graph witness.** On the 21-node unweighted path,

    λ₂ = 1 − cos(π/20) = 0.012311659404862...
    q₂³ = (1 − 0.5 λ₂)³ = 0.981645960340198...
    1/(π+1) = 0.241453007005224...

The Fiedler amplitude needs 231 steps to cross that target. The same graph
still has mean eigenvalue 1. Thus the three-operation grammar window is not a
mode-uniform relaxation theorem. A trajectory bound must use the relevant
spectrum, norm, Euler stability region, stationary-mode treatment, and
operator-induced gains. Directed non-normal and heterogeneous-frequency
generators require their own analysis.

## 3. Structural potential and topology-dependent bounds

For a fixed distance convention and positive finite distances, define

    B_ij = d(i,j)⁻² for reachable j ≠ i; otherwise 0,
    Φ_s = B p.

The field is linear in pressure. The triangle inequality gives the exact bound

    |Φ_s(i)| ≤ Σ_j B_ij |p_j|,
    ||Φ_s||∞ ≤ b_G ||p||∞,    b_G = max_i Σ_j B_ij.

For two pressure fields on the same graph,

    ||Φ_s(t₁) − Φ_s(t₀)||∞ ≤ b_G ||p(t₁) − p(t₀)||∞.

If the graph changes, an additional term is necessary. Writing B₀ and B₁ for
the two kernels gives

    Φ₁ − Φ₀ = B₁(p₁−p₀) + (B₁−B₀)p₀.

These bounds explicitly depend on pressure, graph geometry, and normalization.
There is no phase angle in Bp. Setting every phase to zero and p_j = 1 on K₄
gives Φ_s(i) = 3 at every vertex. On K_n it gives n−1, and multiplying p by a
multiplies Φ_s by a. A transition from p = 0 to p = 1 has the same n−1 drift.

Consequently the canonical **π/4 per-node** and **π/2 drift** thresholds are
selected safety policies. A state can exceed them and trigger telemetry; phase
wrapping does not enforce them. Establishing them as trajectory bounds would
require explicit pressure and topology hypotheses implying the inequalities
above. U6 remains a read-only monitor, separate from the word grammar.

The inverse-square exponent α = 2 is the canonical kernel choice. Chain
summability does not uniquely select it: Σ d⁻α converges for every α > 1,
and independent unit-variance pressure gives variance Σ d⁻²α, convergent for
every α > 1/2. On a graph family with shell counts bounded by C r^(d−1),
α > d is sufficient for absolute accumulation and 2α > d for independent
unit-variance accumulation. These are sufficient shell-growth conditions;
they need not hold on every graph family. Particular ζ(2) and ζ(4) values do
not establish universal π-fraction confinement.

In contrast, the phase definitions give exact kinematic bounds:

    |∇φ| ≤ π,    |K_φ| ≤ π.

The curvature warning level 0.9π is a selected margin inside that bound. The
phase-gradient warning level π/16 is also a policy value.

## 4. Diagnostic channels and state reconstruction

The tetrad selects four useful readouts: source accumulation Φ_s, local phase
mismatch |∇φ|, local circular curvature K_φ, and correlation length ξ_C. These
are canonical telemetry channels. Canonical status does not establish that
their values reconstruct all state variables or every independent observable.

On a graph, edge derivatives and Laplacian powers can be generated by composing
a gradient and a Laplacian. Algebraic generation does not imply that successive
operators, or lossy scalar summaries of their outputs, are linearly dependent.
If L has at least four distinct eigenvalues, I, L, L², and L³ are linearly
independent: a dependence would give a polynomial of degree at most three
vanishing at four distinct points. More generally, independence is controlled
by the minimal polynomial, not a universal second-order cutoff.

An independent obstruction comes from the nodal equation. At fixed graph,
phase, and pure-EPI pressure, rescaling homogeneous ν_f leaves the tetrad
unchanged but rescales x' = ν_f p. Therefore the tetrad does not determine the
full dynamical state. Canonical coherence also depends on pressure amplitude:
with dEPI = 0 and uniform |p| = 1 it equals 1/2; doubling pressure gives 1/3.
Only the separate normalized dispersion statistic is scale invariant.

A reconstruction or minimality theorem must specify the state space, gauge
identifications, full fields versus node/global summaries, allowed observables,
and the desired equivalence relation. A diagnostic removal study instead needs
explicit fixtures and detection thresholds. Neither a test count nor an
operator-composition identity supplies those missing hypotheses. Completeness
and minimality in this stronger sense remain open.

## 5. Authoritative entry points

- [Unified grammar](UNIFIED_GRAMMAR_RULES.md): current U1–U6 requirements.
- [Tetrad scope](MINIMAL_STRUCTURAL_DEGREES.md): interpretation and limitations.
- [Field definitions and API](../docs/STRUCTURAL_FIELDS_TETRAD.md).
- [Canonical calibration functions](../src/tnfr/config/physics_derivation.py).
- [Threshold values](../src/tnfr/constants/canonical.py).
- The deterministic finite-graph witnesses used by this scope statement are
  recorded directly in the sections above and covered by the linked tests.
