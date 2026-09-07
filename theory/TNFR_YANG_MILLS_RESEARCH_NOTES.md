# TNFR–Yang–Mills Structural Gap Research Notes

**Status**: Pre-registered research programme; Y1–Y5 diagnostics implemented; finite result remains pre-branch and the Clay-strength obstruction is classified as Branch B
**Date**: 2026-09-06 (scope audit; implementation milestones dated 2026-05-31)
**Scope**: auxiliary TNFR field-coordinate diagnostics and a finite structural spectral matrix; not a proof of the Clay Yang–Mills and Mass Gap problem
**Primary anchors**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, canonical operators, grammar U1–U6, structural field tetrad `(Φ_s, |∇φ|, K_φ, ξ_C)`, complex geometric field `Ψ = K_φ + i·J_φ`

---

## 0. Terminology Discipline

This programme must be formulated in TNFR language only.

TNFR does **not** introduce an independent entity called "quantum mechanics" or a separate microscopic ontology.  The same nodal equation,

$$
\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t),
$$

admits different coherence regimes:

- smooth-trajectory regimes, externally comparable to classical mechanics;
- discrete-mode / high-dissonance regimes, externally comparable to quantum-mechanical behaviour;
- auxiliary field-coordinate rotations of `Ψ = K_φ + i·J_φ`, without a derived independent gauge degree of freedom.

Therefore, references to Yang–Mills and mass gap are treated as **external comparison targets**. The TNFR object implemented here is a finite nodal spectral diagnostic with an auxiliary pure-gauge twist: construct the declared matrix, measure its spectral separation, and test associations with TNFR telemetry. Y1–Y4 neither execute an operator history nor compare two structural-potential snapshots, so they cannot infer that U1–U6 enforce the measured gap.

No claim in this document should be read as a solution of the Clay Millennium Problem.  The Clay problem concerns rigorous four-dimensional non-Abelian Yang–Mills existence and a positive mass gap in the continuum.  The current TNFR codebase contains an auxiliary nodewise **U(1)** rotation model with the pure-gauge connection `A=d(arg Ψ)`. It does not contain an independently dynamical gauge sector. Any such Abelian or non-Abelian extension must be derived from the nodal equation and grammar before being called canonical.

---

## 1. Existing field and diagnostic base in the repository

The programme starts from already-shipped TNFR machinery:

| Component | Existing source | Role |
| --- | --- | --- |
| Complex geometric field | `src/tnfr/physics/fields.py`, `src/tnfr/physics/unified.py` | `Ψ = K_φ + i·J_φ`, geometric-transport sector |
| Auxiliary U(1) action | `src/tnfr/physics/gauge.py` | nodewise coordinate rotation `Ψ(i) → e^{iα(i)}Ψ(i)`; not an engine operator |
| Gauge connection | `compute_gauge_connection()` | exact vertex-phase difference `A_ij = arg(Ψ_j) − arg(Ψ_i)` on edges |
| Gauge curvature | `compute_gauge_curvature()` | cycle holonomy `F_C = Σ A_ij`; zero in exact arithmetic for this exact one-form, with only floating residuals in the present implementation |
| Covariant derivative | `compute_covariant_derivative()` | `D_ijΨ = Ψ(j) − e^{iA_ij}Ψ(i)` |
| Yang–Mills-named action | `compute_yang_mills_action()` | `S_YM = 1/2 Σ_C F_C^2`; squared floating closure residual only |
| Equation-named diagnostic | `compute_yang_mills_equations()` | pure-gauge snapshot consistency residual; not a TNFR equation of motion |
| Conservation-gauge comparison | `src/tnfr/physics/conservation_gauge_unification.py` | auxiliary and finite-snapshot compatibility diagnostics |

This means the first TNFR–Yang–Mills step is only a spectral diagnostic built from existing fields. A genuine gauge-dynamics step would require a newly derived independent edge degree of freedom and its relation to canonical operators; the present API does not supply either.

---

## 2. TNFR-Native Reformulation of the Mass Gap Question

The external Clay statement asks for a rigorous four-dimensional Yang–Mills theory with compact simple gauge group and positive mass gap.  TNFR reframes the first attack surface as follows.

Given a declared family of finite nodal graphs `G`, construct a self-adjoint structural diagnostic matrix

$$
H_{\mathrm{structural}}(G)
$$

from declared TNFR telemetry and auxiliary transforms:

$$
(\Phi_s, |\nabla\phi|, K_\phi, \xi_C, J_\phi, J_{\Delta\mathrm{NFR}}, A_{ij}, F_C, D_{ij}\Psi).
$$

Define the finite-graph structural gap

$$
\Delta_{\mathrm{TNFR}}(G) = \lambda_1\!\left(H_{\mathrm{structural}}(G)\right) - \lambda_0\!\left(H_{\mathrm{structural}}(G)\right).
$$

The first TNFR question is not the full Clay theorem, but the discrete structural precursor:

> **YMG-1**: Does the declared pure-gauge-covariant finite matrix built from `Ψ` have a reproducible positive separation between its two lowest distinct modes?

The continuum-strength question is deferred:

> **YMG-5**: Does `liminf_{a→0, L→∞} Δ_TNFR(a,L) > 0` hold under a canonically specified scaling regime?

YMG-5 is the Clay-hard boundary and is not assumed.

---

## 3. Structural Interpretation of the Gap

TNFR does not treat "particle mass" as primitive. In this finite programme,
"mass gap" is only an external analogy for the separation between the two
lowest distinct eigenvalues of a selected structural matrix. Stability and
dynamical admissibility require separate proofs.

| External term | TNFR structural object |
| --- | --- |
| Vacuum | lowest mode of the selected finite structural matrix |
| Excitation | next distinct eigenmode of the selected finite matrix |
| Mass gap | external analogy for its lowest distinct-mode separation; no dynamical admissibility theorem is supplied |
| Confinement | Future hypothesis requiring an independently derived non-flat edge connection; absent from the current pure-gauge surface |
| Gauge field | Not implemented independently; the current `A` is only `d(arg Ψ)` |

The working hypothesis is:

$$
\Delta_{\mathrm{TNFR}} > 0
\quad\Longleftrightarrow\quad
\text{U6 drift control + independently derived curvature + declared evolution hypotheses exclude zero-cost non-trivial modes.}
$$

This is an open TNFR working hypothesis about nodal dynamics, not an established
consequence of U6 and not an ontological statement about a separate quantum layer.
The present Y1–Y4 surface cannot test the hypothesis: it supplies neither a U6
reference trajectory nor an independent edge connection with nonzero holonomy.
Its positive finite gaps can already arise from finite graph connectivity and
the selected single-snapshot potential penalty.

---

## 4. Candidate Operator Surface

The initial finite-graph matrix combines canonical graph read-outs with the
auxiliary pure-gauge connection. A minimal candidate family is:

$$
H_{\mathrm{structural}} = L_A + V_{F} + V_{\Phi},
$$

where:

- `L_A` is a gauge-covariant graph Laplacian derived from `D_ijΨ`;
- `V_F` is a formal curvature potential derived from cycle holonomies `F_C`; for the current exact phase-difference connection it vanishes analytically, so computed values are floating-point residuals;
- `V_Φ` is the single-snapshot magnitude penalty `Φ_s²/(π/2)²`. Borrowing the U6 drift threshold as a denominator is a legacy normalization choice, not a U6 drift test and not the separate `π/4` per-node magnitude warning.

The implementation exposes non-negative weights for sensitivity studies and uses unit defaults. A result with non-unit weights is conditional on those declared diagnostic coefficients; the nodal equation has not derived them as canonical constants.

---

## 5. Gap Ledger

| Gap | Question | Status |
| --- | --- | --- |
| **YMG-0** | TNFR-native terminology and scope discipline | **CLOSED by pre-registration** |
| **YMG-1** | Finite-graph TNFR gauge gap diagnostic | **IMPLEMENTED** by `tnfr.yang_mills.compute_structural_gauge_gap()`; finite graph only |
| **YMG-2** | Auxiliary U(1) covariance of the finite gap operator | **PURE-GAUGE COVARIANCE SUPPORTED** by seeded spectral-invariance checks; no Ward identity for engine dynamics follows |
| **YMG-3** | Relate a genuine two-snapshot U6 drift condition to a finite gap lower bound | **OPEN**; compatibility-named `run_u6_confinement_sweep()` measures only a single-snapshot `Φ_s` magnitude coordinate |
| **YMG-4** | Independent/non-Abelian gauge derivability audit from the nodal equation | **AUDITED: OPEN_DERIVABILITY_GAP**; current implementation is only an auxiliary U(1) pure-gauge construction |
| **YMG-5** | Continuum + thermodynamic scaling `liminf Δ > 0` | **FINITE SCALING DIAGNOSTIC IMPLEMENTED** by Y4; continuum limit remains **OPEN / Clay-hard** |
| **YMG-6** | Closure / obstruction classification | **CLASSIFIED: BRANCH_B_OBSTRUCTION_CLASSIFIED** by Y5 |

The key honesty constraint is YMG-4: classical Yang–Mills mass gap is non-Abelian.  A multi-channel or non-Abelian TNFR gauge sector cannot be assumed merely because external Yang–Mills uses it.  It must be derived as a structural consequence of the nodal equation, tetrad, operators, and U1–U6.

---

## 6. Pre-Registered Milestones

### Y1 — Finite Structural Gauge Gap Diagnostic

**Implementation status (2026-05-31)**: `DIAGNOSTIC_SURFACE_CREATED`.

Implemented in:

- `src/tnfr/yang_mills/__init__.py`
- `src/tnfr/yang_mills/structural_gap.py`
- `tests/physics/test_yang_mills_structural_gap.py`

The implemented operator is:

$$
H_{\mathrm{structural}} = L_A + V_F + V_{\Phi},
$$

where `L_A` is the gauge-covariant graph Laplacian assembled from `A_ij`, `V_F` is the cycle-closure penalty normalised by `π²`, and `V_Φ` is a structural-potential magnitude penalty normalised by `(π/2)²`. Because the bundled `A_ij` is the coboundary of a vertex phase, every cycle sum telescopes to zero modulo `2π`; `V_F` is therefore zero in exact arithmetic in Y1. The code records its maximum floating residual and rejects a supplied connection outside the same pure-gauge orbit. The diagnostic reports `(λ0, λ1, Δ)`, self-adjointness, seeded auxiliary-U(1) spectral covariance, both the actual `π/4` magnitude-warning ratio and legacy `π/2` normalization metadata, legacy squared-closure/coupling statistics, and the subset of snapshot checks that is actually applicable. It is read-only with respect to EPI and phase attributes. Compatibility metadata keys containing `u6` denote the historical magnitude proxy and explicitly report `u6_drift_assessed=False`, the mean-absolute nodewise aggregation, and a strict threshold comparison.

Implement a finite-graph diagnostic that:

1. constructs a phase-compatible finite gauge graph without claiming full word-grammar validation;
2. computes `Ψ`, `A_ij`, the analytically flat `F_C`, `D_ijΨ`, and `S_YM` residuals;
3. assembles the self-adjoint finite matrix `H_structural`;
4. reports `(λ0, λ1, Δ)` with seed, graph, and threshold metadata;
5. verifies the declared algebraic invariants and spectral covariance before and after auxiliary local `U(1)` rotations.

First verdict: **DIAGNOSTIC_SURFACE_CREATED**, not closure. The gap reported by Y1 is a finite-graph structural spectral gap only. With the current pure-gauge connection, it cannot be attributed to nonzero curvature; it also does not address non-Abelian derivability or the continuum / thermodynamic limit.

### Y2 — legacy U6-named potential-magnitude sweep

**Implementation status (2026-05-31)**: `EMPIRICAL_FINITE_GRAPH_ONLY`.

Implemented in:

- `src/tnfr/yang_mills/u6_sweep.py`
- `tests/physics/test_yang_mills_u6_sweep.py`

The Y2 sweep wraps the Y1 operator across finite graph families and target ratios

$$
\rho_{\mathrm{U6}} = \frac{\max_i |\Phi_s(i)|}{\pi/2},
$$

Despite the historical symbol and public API name, this is a single-snapshot magnitude coordinate. `ρ_U6 < 1` says only that `max|Φ_s|` is below the numerical scale borrowed from the U6 threshold; it does not say that U6 passed or that the separate `π/4` magnitude warning passed. A canonical U6 decision requires aligned reference and observed fields and evaluates `mean_i |Φ_s,after(i)-Φ_s,before(i)| < π/2`. The sweep reports the `π/4` magnitude-warning ratio separately, along with gap statistics, pure-gauge consistency residuals under legacy Yang–Mills names, applicable snapshot checks, self-adjointness, and seeded auxiliary-U(1) spectral covariance.

Sweep graph families across the declared magnitude coordinate. Test whether the finite gap correlates with:

- sampled `Φ_s` magnitude relative to both declared scales;
- the expected numerical flatness residual of `F_C`;
- low pure-gauge consistency residual exposed under the legacy equation name;
- the currently applicable snapshot diagnostics. Operator-history grammar is not assessed.

First verdict: **EMPIRICAL_FINITE_GRAPH_ONLY**. This creates a finite potential-magnitude surface, but it does not yet test YMG-3's two-snapshot U6 premise and does not address YMG-4/YMG-5.

### Y3 — Non-Abelian Derivability Audit

**Implementation status (2026-05-31)**: `OPEN_DERIVABILITY_GAP`.

Implemented in:

- `src/tnfr/yang_mills/derivability.py`
- `tests/physics/test_yang_mills_derivability.py`

The Y3 audit evaluates whether any candidate route supplies all of the following without external input: a TNFR-native multiplet, a canonical connection mixing multiplet components, non-commuting generators derived from nodal dynamics, and U1–U6 compatibility.  The implemented candidate routes are:

| Route | Result | Obstruction |
| --- | --- | --- |
| `u5_nested_epi_multiplet` | `OPEN_MULTIPLET_WITHOUT_CANONICAL_CONNECTION` if nested EPI data are present; otherwise `FAILED_NO_TNFR_MULTIPLET` | Nested EPI can provide components, but no canonical component-mixing connection or non-commuting generator algebra is derived |
| `thol_remesh_internal_space` | `OPEN_HISTORY_WITHOUT_CANONICAL_GENERATORS` if operator history exists; otherwise `FAILED_NO_OPERATOR_INTERNAL_SPACE` | THOL/REMESH history does not expose a derived non-commuting generator algebra |
| `cycle_basis_bundle` | `FAILED_BASIS_DEPENDENT_EXTERNAL_SELECTION` when enough cycles exist | Cycle-basis generator selection depends on non-canonical basis/orientation choices |

Net verdict: the repository has an auxiliary local `U(1)` coordinate action and pure-gauge derived connection, not a canonical dynamical gauge sector. No independent Abelian or non-Abelian TNFR gauge sector is promoted by Y3.

Attempt to derive multi-component gauge structure from TNFR-internal data only.  Candidate routes:

- multi-channel `Ψ` multiplets from nested EPI levels (U5);
- operator-induced internal state spaces from THOL/REMESH;
- graph-cycle basis bundles from canonical topology.

Acceptance requires a nodal-equation derivation.  If the construction needs external group labels or hand-selected generators, it is non-canonical.

First verdict: **OPEN_DERIVABILITY_GAP**.

### Y4 — Scaling Study

**Implementation status (2026-05-31)**: `FINITE_SCALING_EVIDENCE` or `GAP_COLLAPSE_OBSERVED`, depending on the sampled finite family.

Implemented in:

- `src/tnfr/yang_mills/scaling.py`
- `tests/physics/test_yang_mills_scaling.py`

The Y4 diagnostic evaluates graph-size surrogates using the realized node count `n` and fixed values of the legacy `ρ_U6` magnitude coordinate. This distinction matters for the grid builder, which realizes the largest square graph not exceeding the requested size. For each `(topology, ρ_U6)` family it records mean gap by realized size and a finite log-log slope of gap versus `n`. The slope is a diagnostic of sampled finite behaviour only; it is not a continuum exponent, does not define a thermodynamic limit, and does not assess U6 drift.

The report classifies finite samples as:

| Verdict | Meaning |
| --- | --- |
| `FINITE_SCALING_EVIDENCE` | all sampled finite points are self-adjoint, auxiliary-U(1) spectrally covariant, and have positive gap above tolerance |
| `GAP_COLLAPSE_OBSERVED` | at least one sampled finite point has gap at or below tolerance |
| `SCALING_FAILED_NON_SELF_ADJOINT` | a sampled operator violates Hermiticity |
| `SCALING_FAILED_GAUGE_VARIANCE` | a sampled spectrum is not invariant under seeded local `U(1)` rotation |

If Y1–Y3 provide a stable finite diagnostic surface while keeping YMG-4 explicitly open, evaluate `Δ(a,L)` across graph spacing / size surrogates.  This is still not the Clay theorem; it is a TNFR scaling diagnostic.

First verdict: **FINITE_SCALING_EVIDENCE** for stable finite samples, or **GAP_COLLAPSE_OBSERVED** for sampled collapse.  Both verdicts remain finite-diagnostic only.

### Y5 — Closure / Obstruction Classification

**Implementation status (2026-05-31)**: `BRANCH_B_OBSTRUCTION_CLASSIFIED`.

Implemented in:

- `src/tnfr/yang_mills/closure.py`
- `tests/physics/test_yang_mills_closure.py`

Y5 separates two logically different questions:

1. **Finite field-derived diagnostic** — Y1–Y4 support a finite, self-adjoint, seeded auxiliary-U(1)-covariant matrix surface built from `Ψ`, the pure-gauge `A_ij`, its cycle-closure residual, and `Φ_s`; this is pre-branch evidence rather than a dynamical gap result.
2. **Clay-strength result** — not closed, because Y3 leaves non-Abelian derivability open and Y4 does not prove a continuum / thermodynamic lower-bound theorem.

Current classification:

| Layer | Verdict | Meaning |
| --- | --- | --- |
| Finite diagnostic layer | `A_FINITE_U1_DIAGNOSTIC_SURFACE` | legacy API verdict name for a pre-branch auxiliary-U(1), pure-gauge matrix diagnostic; it does not satisfy Branch A |
| Clay-strength layer | `B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION` | a Clay-strength route requires a new TNFR-native non-Abelian derivation plus a continuum lower-bound theorem |
| Programme verdict | `BRANCH_B_OBSTRUCTION_CLASSIFIED` | the obstruction is now localized, not removed |

Y5 therefore does **not** claim the Yang–Mills Millennium Problem is solved.  It closes the first TNFR programme pass by identifying the exact boundary: finite pure-gauge structural diagnostics are available; an independently derived gauge sector, non-Abelian derivability, and continuum lower bounds remain open requirements.

Classify the programme into one of three branches:

| Branch | Meaning |
| --- | --- |
| **A** | a finite TNFR structural gap is derived from declared dynamics plus a genuine U6 reference-state condition |
| **B** | finite gap requires a new canonical derivation, likely non-Abelian/multiplet |
| **C** | no TNFR-internal mass-gap analogue survives canonical constraints |

Current Y5 result: **a pre-branch finite diagnostic surface plus the Branch-B
obstruction at Clay-strength scope**. Branch A remains unestablished: Y1–Y4 do
not evolve an independent connection, apply a genuine two-state U6 condition,
or derive their positive finite gap from canonical nodal dynamics.

---

## 7. Acceptance Criteria

Any claimed TNFR–Yang–Mills result must satisfy:

1. **Nodal derivability** — every term traces to `∂EPI/∂t = νf · ΔNFR(t)` or canonical tetrad telemetry.
2. **Operator discipline** — no EPI mutation outside the 13 canonical operators.
3. **Grammar compliance** — U1–U6 constraints are checked or explicitly scoped.
4. **Auxiliary U(1) covariance** — the reported finite spectrum must be invariant under the paired `Ψ → e^{iα}Ψ`, `A → A+dα` analysis transformation, or any dependence must be classified as a diagnostic failure; this check is not an engine symmetry theorem.
5. **Declared coefficients** — every diagnostic weight and normalization is recorded; promotion to a canonical constant requires a nodal-dynamics derivation.
6. **Reproducibility** — graph family, seed, spectra, thresholds, and residuals are recorded.
7. **Scope honesty** — finite TNFR gap results are not called a Clay proof unless non-Abelian existence and continuum scaling are addressed rigorously.

---

## 8. Immediate Next Step

The next research target is **Y6 / Branch-B derivation search**: first derive an independent TNFR edge degree of freedom that can carry nonzero cycle holonomy, then determine whether it supports a TNFR-native non-Abelian connection and non-commuting generator algebra from the nodal equation, nested EPI structure, and canonical operator histories. If such a derivation cannot be found without external group labels, the Yang–Mills programme should remain paused at Branch B rather than extending finite diagnostics indefinitely.

The programme begins from TNFR's own structural dynamics.  External Yang–Mills terminology is used only to name the comparison problem and to define the mass-gap target surface.
