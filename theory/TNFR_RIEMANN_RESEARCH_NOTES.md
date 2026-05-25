# TNFR–Riemann Program Memo

**Status**: Exploratory research (non-canonical)
**Version**: 0.5.0 (March 2026)
**Owner**: `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`

---

This memo defines the minimum structure required to evaluate TNFR claims about the Riemann Hypothesis (RH). It scopes the computational program, prescribes telemetry, and records open work items so contributors can extend the investigation without rewriting the physics or the SDK contracts. All historical notes remain in the appendix for context.

## 1. Purpose and Scope

- Translate RH questions into TNFR constructs: nodal operators, structural partition functions, and confinement criteria derived from Φ_s, |∇φ|, K_φ, and ξ_C.
- Maintain reproducible sandboxes (finite prime graphs, spectral benchmarks, telemetry artifacts) that connect theoretical conjectures to code in `src/tnfr/riemann/` and `examples/16_riemann_operator_demo.py`.
- Document how canonical operators (AL, UM, RA, OZ, IL, THOL) compose to form the discrete TNFR Riemann operator used in experiments.

## 2. Program Objectives

### 2.1 Partition Function Mapping

- Show that the TNFR structural partition function $Z_{TNFR}(s)$ converges to ζ(s) or ξ(s) by enforcing the identification $e^{-\beta E_p(s)} \leftrightarrow p^{-s}$ for prime-labeled resonant modes.
- Specify how ν_f and ΔNFR sources enter the effective energy $E_p(s)$ so the mapping respects U2 (convergence) and U3 (resonant coupling).

### 2.2 Operator Construction

- Construct $\mathcal{H}_{TNFR}$ as a Laplacian-plus-structural-potential on prime path graphs, ensuring self-adjointness with respect to the TNFR inner product.
- Demonstrate numerically that eigenvalues migrate toward the critical line as graph size increases (σ_c^{(k)} \to 1/2) and record telemetry in `results/riemann_program/`.

### 2.3 Critical-Line Confinement

- Formulate a Lyapunov-style functional $\mathcal{L}_{RH}(s)$ derived from TNFR invariants so that σ = 1/2 is the only stable attractor.
- Quantify escapes (σ ≠ 1/2) via Φ_s drift and |∇φ| spikes to test whether confinement behaves like U6 in the complex-s domain.

## 3. Workflow Expectations

1. **Model definition** – Choose $G_k$ (prime path graph) size, seeds, and operator sequences; record configs in `results/riemann_program/configs/*.json`.
2. **Operator execution** – Use SDK helpers (`TNFRRiemannOperator`) to generate spectra while logging ν_f, ΔNFR, Φ_s, |∇φ|, and effective σ(t) trajectories.
3. **Spectral analysis** – Compute eigenvalue ladders, determinant surrogates, and compare against ζ/ξ predictions. Scripts belong in `scripts/riemann/` or notebooks under `notebooks/Riemann/` with nbconvert support.
4. **Benchmark enforcement** – Run `python benchmarks/riemann_program.py` (invoked automatically via `make test`/CI) to regress σ_c^{(k)} estimates across graph sizes and emit telemetry in `results/riemann_program/`.
5. **Validation** – Run targeted tests (e.g., `examples/16_riemann_operator_demo.py`, new `tests/test_riemann_operator.py`) to ensure deterministic seeds and grammar compliance (U1–U6).

## 4. Telemetry & Reproducibility

- Log Φ_s, |∇φ|, K_φ, ξ_C, ν_f, ΔNFR, and σ estimates at every operator step; store as Parquet/CSV in `results/riemann_program/telemetry/` with metadata (graph size, seed, operator stack). The helper dataclass `tnfr.riemann.telemetry.RiemannTelemetryRecord` now carries aggregate Φ_s/|∇φ|/K_φ statistics plus ξ_C computed via `tnfr.riemann.telemetry.compute_field_aggregates` so tetrad coverage is explicit.
- Publish spectra, determinant traces, and Lyapunov metrics in `results/riemann_program/plots/` along with scripts used to generate them.
- Capture environment details (Python version, tnfr package hash) inside each artifact manifest to satisfy invariants #5 (Structural Metrology) and #6 (Reproducible Dynamics).

## 5. Outstanding Work

1. **Lyapunov functional derivation** – Formalize $\mathcal{L}_{RH}(s)$ using existing field invariants and document stability proofs in `docs/STRUCTURAL_FIELDS_TETRAD.md` or a dedicated theory note.
2. **Spectral determinant prototype** – Produce a working determinant or trace formula implementation and compare against numerical ζ(s) evaluations over multiple σ bands.
3. **Telemetry-field linkage** – Extend `tnfr.riemann.telemetry` so Φ_s, |∇φ|, K_φ, and ξ_C aggregates from live runs attach automatically to each record (current benchmark logs spectral data only).

## 6. Cross-References

### Implementation Modules (`src/tnfr/riemann/`)

Discrete operator and spectral framework:
- `operator.py` — discrete TNFR-Riemann operator $H^{(k)}(\sigma) = L_k + V_\sigma$ and prime graph builders.
- `spectral_proof.py` — four-line spectral convergence framework ($\sigma_c^{(k)} \to 1/2$).
- `topology.py` — alternative graph topologies and cross-topology convergence (P2).
- `eigenmode_fields.py` — per-eigenmode structural field tetrad on the prime path model (P3).
- `complex_extension.py` — complex-$s$ non-Hermitian extension (P4).
- `spectral_zeta.py` — discrete spectral zeta and heat kernel; original Conjecture 10.1 affine bridge (P5, **negative**; superseded by P12–P15 via §7.8).
- `random_ensemble.py` — random prime-graph ensembles / RMT universality (P6).
- `spectral_conservation.py` — conservation laws and grammar compliance at criticality (P7).
- `analytical_convergence.py` — analytical proof of $\sigma_c \to 1/2$ via PNT + telescoping (P8).
- `functional_equation.py` — TNFR-side $s \leftrightarrow 1-s$ reflection check (P9).
- `convergence_proof.py` — end-to-end formal $\sigma_c \to 1/2$ certificate (P10).
- `zeta_bridge.py` — affine bridge prototype $\zeta_H \approx C \cdot \zeta_R$ (P11, tested **negative**, see §7).
- `telemetry.py` — Riemann telemetry records and field aggregate helpers.

Prime-ladder / von Mangoldt construction (closes G1, G2, G3 operationally; G5 superseded):
- `von_mangoldt.py` — TNFR prime-ladder spectrum reproducing $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n)\, n^{-s}$ on $\operatorname{Re}(s) > 1$ (P12, §8).
- `analytic_continuation.py` — continuation of the prime-ladder vM zeta to $\mathbb{C}$; Riemann zeros as resonance poles on $\operatorname{Re}(s) = 1/2$ (P13, §9).
- `prime_ladder_hamiltonian.py` — self-adjoint Hamiltonian whose weighted spectral trace reproduces P12 (P14, §10; **closes G1**).
- `weil_explicit_formula.py` — numerical Weil–Guinand identity using P14 on the prime side; residual $\le 5 \times 10^{-12}$ (P15, §11; **closes G3**).
- `li_keiper.py` — Li–Keiper positivity criterion from the TNFR resonance spectrum (P16, §12; **RH-equivalent diagnostic**, not proof).

TNFR-native G4 attack surface (research; does **not** close G4 = RH):
- `weil_positivity.py` — Weil–TNFR positivity bridge $\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma]$ (P17, §14).
- `alpha_sweep.py` — admissibility / gauge sweep of $\alpha(\sigma)$ across Gaussian width × gauge family (P18, §15).
- `admissible_family_sweep.py` — extends P18 beyond Gaussian (Gaussian mixture, Hermite2-Gaussian admissible families) (P19/P21, §16/§18).
- `nodeaware_gauge_sweep.py` — node-aware gauge extension parameterised by local $\nu_f$ and node-weight channels (P20, §17).
- `coercivity_uniform.py` — empirical uniform-coercivity certificate over $\sigma$ intervals, plus adaptive $\sigma$ refinement near the coercivity bottleneck (P22 / P23 / P24, §13 / §13bis).
- `paley_gap_coercivity.py` — Paley-gap coercivity diagnostic (Martínez Gamo, Zenodo 10.5281/zenodo.17665853 v2) (P25, §13ter).
- `lyapunov_spectral_positivity.py` — Lyapunov-spectral positivity certificate for the P14 Hamiltonian (P26, §13quater).
- `hilbert_polya.py` — Hilbert–Pólya scaffold $T_{\mathrm{HP}} = \operatorname{diag}(\gamma_n)$ populated by `mpmath.zetazero` (diagnostic only) (P27, §13quinquies).
- `structural_zero_density.py` — structural derivation of the smooth Riemann zero density via the Riemann–Siegel $\theta$ function (P28, §13sexies; **closes smooth half of G4 at density level**).
- `spectral_emergence.py` — spectral universality emergence under canonical UM+RA inter-prime couplings; KS-distance to the GUE Wigner surmise (P29, §13octies.3).
- `admissible_rescaling.py` — operator-level admissible spectral-rescaling lift of P28 (P30, §13nonies; **closes smooth half of T-HP at operator level**).

Conjectural reformulation (does **not** close G4):
- §13septies — Tetrad-Hilbert–Pólya reformulation of G4 (Conjecture T-HP).
- §13octies — Assembled-argument audit (links L1–L7 closed, L8 = T-HP open).

### Examples

End-to-end pipeline demos (`examples/`):
- `16_riemann_operator_demo.py` — discrete TNFR-Riemann eigenvalues at varying $\sigma$.
- `18_riemann_convergence_proof.py` — spectral convergence proof ($\sigma_c \to 1/2$).
- `19_topology_comparison.py` — cross-topology critical parameter comparison.
- `20_eigenmode_tetrad.py` — eigenmode-based tetrad field analysis.
- `21_complex_extension_demo.py` — non-Hermitian operator on complex $s$.
- `22_spectral_zeta_demo.py` — discrete spectral zeta, heat kernel, Mellin bridge.
- `23_random_ensemble_rmt_demo.py` — random matrix ensembles (GOE/GUE/Poisson).
- `24_spectral_conservation_demo.py` — spectral conservation law at criticality.
- `25_analytical_convergence_demo.py` — analytical proof via PNT + telescoping.
- `41_von_mangoldt_zeta_demo.py` — P12 prime-ladder reproduction of $-\zeta'/\zeta$.
- `42_riemann_zeros_as_resonances.py` — P13 zeros as resonance poles on $\operatorname{Re}(s) = 1/2$.
- `43_prime_ladder_hamiltonian_demo.py` — P14 self-adjoint Hamiltonian certificate.
- `44_weil_explicit_formula_demo.py` — P15 Weil–Guinand identity at machine precision.
- `45_li_keiper_demo.py` — P16 Li–Keiper positivity diagnostic.
- `46_weil_tnfr_positivity_demo.py` — P17 Weil–TNFR positivity bridge.
- `47_alpha_sweep_demo.py` — P18 gauge sweep of $\alpha(\sigma)$.
- `48_admissible_family_sweep_demo.py` — P19/P21 admissible-family sweep.
- `49_nodeaware_gauge_sweep_demo.py` — P20 node-aware gauge extension.
- `50_uniform_coercivity_demo.py` — P22 empirical uniform-coercivity certificate.
- `51_adaptive_coercivity_demo.py` — P24 adaptive $\sigma$ refinement near the bottleneck.
- `52_paley_gap_coercivity_demo.py` — P25 Paley-gap coercivity diagnostic.
- `53_lyapunov_spectral_positivity_demo.py` — P26 Lyapunov-spectral positivity certificate.
- `54_hilbert_polya_demo.py` — P27 Hilbert–Pólya diagnostic scaffold.
- `55_structural_zero_density_demo.py` — P28 structural smooth zero density.
- `56_spectral_emergence_demo.py` — P29 KS-distance to GUE under canonical couplings.
- `57_admissible_rescaling_demo.py` — P30 operator-level admissible rescaling (smooth half).

### Supporting Infrastructure

- `benchmarks/riemann_program.py` — automated spectral regression benchmarks for $\sigma_c^{(k)}$ across graph sizes.
- `theory/UNIFIED_GRAMMAR_RULES.md` — grammar rules U1–U6 referenced throughout.
- `docs/STRUCTURAL_FIELDS_TETRAD.md` — tetrad field specifications.
- `AGENTS.md` — TNFR-Riemann overview, including the G4 = RH reformulation.

---
## 7. Conjecture 10.1 Gap Analysis (May 2026)

**Status**: Negative numerical result — bridge not yet closed.

### 7.1 Experiment

The fit defined by Conjecture 10.1

$$
\zeta_{H^{(k)}}(1/2,\, u) \;\approx\; C(k)\;\cdot\;\zeta_R(u + \delta(k))
$$

was tested using `test_conjecture_10_1_sequence` from `src/tnfr/riemann/spectral_zeta.py`
for $k \in \{10, 20, 50, 100, 200, 500, 1000\}$ over $u \in [1.5, 5.0]$ (30 points).

### 7.2 Numerical Results

| k | C(k) | δ(k) | residual (normalised) | Pearson r |
|---:|-------------:|------:|---------------------:|----------:|
| 10 | 2.02 × 10⁷ | 2.0 | 2.4347 | −0.4057 |
| 20 | 5.46 × 10¹¹ | 2.0 | 3.0692 | −0.3285 |
| 50 | 8.77 × 10¹⁷ | 2.0 | 3.7082 | −0.2746 |
| 100 | 3.43 × 10²² | 2.0 | 4.0625 | −0.2516 |
| 200 | 1.25 × 10²⁷ | 2.0 | 4.3420 | −0.2359 |
| 500 | 2.06 × 10³³ | 2.0 | 4.6337 | −0.2215 |
| 1000 | 9.06 × 10³⁷ | 2.0 | 4.7989 | −0.2140 |

### 7.3 Diagnostic Reading

A converging bridge would show: residual → 0, Pearson r → +1,
δ(k) stabilising at an interior value, and C(k) stabilising after
correct renormalisation.  The data show the opposite in every metric:

- **Residual rises** monotonically with k.
- **Correlation is negative** and bounded away from +1 at all tested k.
- **δ(k) = 2.0** in every row — pinned at the boundary of the search range,
  indicating no interior minimum was found.
- **C(k) explodes** (≈10³⁷ at k = 1000), signalling a missing renormalisation.

Conclusion: as currently implemented, `ζ_H^(k)(1/2, u)` is not
numerically equivalent to `C · ζ_R(u + δ)` under the simple affine fit.

### 7.4 Six Missing Pieces

| # | Missing piece | Current status |
|---|---|---|
| 1 | **Euler product reconstruction** `∏_p (1−p⁻ˢ)⁻¹` | Prime-path graphs do not demonstrably reproduce all powers p^m with correct multiplicity. |
| 2 | **Spectral zeta ≡ ζ(s)** | Tested as a conjecture; numerical fit diverges. |
| 3 | **Correct spectral renormalisation of C(k)** | C(k) explodes — spectral renormalisation is absent. |
| 4 | **Convergent δ(k)** | δ(k) does not converge internally; remains pinned at search-range boundary. |
| 5 | **Analytic continuation to the complex strip** | RH lives in 0 < Re(s) < 1 over ℂ; current tests use only real u > 1. |
| 6 | **Zero correspondence** | Not shown that non-trivial zeros of ζ(s) equal zeros/modes of ζ_H. |

### 7.5 TNFR-Internal Diagnosis

In TNFR language the finding is:

> The operator $H^{(k)}(\sigma)$ constructs a structural dynamic that is
> sensitive to the critical line σ = 1/2 (σ_c^(k) → 1/2 is internally
> validated), but it does not yet encode the full multiplicative arithmetic
> of ζ(s).  The prime-path graph captures structural coherence near 1/2
> without closing the bridge to the classical zeta function.

### 7.6 Priority Construction

The mathematical priority is to build a TNFR zeta that reproduces the
von Mangoldt series

$$
-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \Lambda(n)\, n^{-s},
$$

which encodes prime positions **and** their higher powers with the correct
multiplicities (Λ = log p for prime powers, 0 otherwise).  Without this,
TNFR can exhibit σ-criticality but cannot be equated to Riemann.

The required renormalisation takes the form

$$
R_k \cdot \zeta_{H}^{(k)}\!\left(\tfrac{1}{2}, s\right) \;\longrightarrow\; \zeta(s),
\qquad \text{or more ambitiously,} \qquad
\det_{TNFR}(H_k - sI) \;\longrightarrow\; \xi(s),
$$

where $\xi(s)$ is the completed Riemann zeta and $R_k$ is a holomorphic,
non-vanishing function to be constructed.

### 7.7 Impact on Program Status

This result does **not** invalidate the σ_c^(k) → 1/2 finding, which
rests on eigenvalue analysis independent of the spectral-zeta fit.
It narrows the scope of Conjecture 10.1: the conjecture is open, and the
simple `C · ζ_R(u + δ)` form is likely insufficient.  Future work should
target the von Mangoldt / Λ-series route (Section 7.6) before revisiting
the affine fit.

### 7.8 Retrospective Closure of G5 (May 2026)

The "non-affine bridge" anticipated in §7.6 has been **constructed and
verified** by the P12–P15 pipeline:

| Step | Module / §  | What it delivers |
|---|---|---|
| P12 | `von_mangoldt.py`, §8 | TNFR prime-ladder spectrum reproducing $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n)\,n^{-s}$ exactly on $\operatorname{Re}(s) > 1$ |
| P13 | `analytic_continuation.py`, §9 | Continuation of the TNFR vM zeta to all of $\mathbb{C}$; Riemann non-trivial zeros realised as resonance poles on $\operatorname{Re}(s) = 1/2$ |
| P14 | `prime_ladder_hamiltonian.py`, §10 | Self-adjoint TNFR Hamiltonian whose weighted spectral trace reproduces the prime-ladder data to machine precision |
| P15 | `weil_explicit_formula.py`, §11 | Weil–Guinand identity verified numerically with the P14 operator on the prime side ($\le 5\times 10^{-12}$ residual) |

This **replaces** the original affine ansatz $\zeta_H(1/2,u) \approx C(k)\,\zeta_R(u+\delta(k))$ with a structurally
correct, multiplicative-arithmetic bridge that lives natively inside
TNFR without ad-hoc renormalisations. The six missing pieces listed in
§7.4 are addressed as follows:

| # | Original gap | Status |
|---|---|---|
| 1 | Euler product / prime powers with multiplicity | **Closed by P12** (ladder $(p,k)$ encodes $p^k$ with $\Lambda$ weights) |
| 2 | Spectral zeta ≡ $\zeta(s)$ | **Closed by P12+P13** (weighted trace $= -\zeta'/\zeta$, continued to $\mathbb{C}$) |
| 3 | Convergent renormalisation $C(k)$ | **Closed by P14** (weight operator $W = \mathrm{diag}(\log p)$, no $C(k)$ needed) |
| 4 | Convergent $\delta(k)$ | **Eliminated** (no affine shift in the multiplicative bridge) |
| 5 | Analytic continuation to the strip | **Closed by P13** (resonance poles on $\operatorname{Re}(s) = 1/2$) |
| 6 | Zero correspondence | **Closed by P13+P15** (Weil-Guinand identifies zeros with TNFR spectral data) |

**Conclusion**: G5, in its original affine formulation, is **superseded**
by the prime-ladder / Λ-series construction. The bridge between TNFR
spectral data and classical $\zeta(s)$ is therefore considered
operationally closed. The only obstruction that remains is **G4 = RH
itself** — the localisation of the resonance poles on
$\operatorname{Re}(s) = 1/2$ — which is a structural positivity / self-adjointness
problem, not a missing-bridge problem.

---

## 8. TNFR Prime-Ladder Construction of the von Mangoldt Series (P12)

Following Section 7.6, this section records the first concrete attempt at
the priority route: build a TNFR-native spectral object whose Dirichlet
transform reproduces $-\zeta'(s)/\zeta(s)$ on its half-plane of
convergence.  Implementation: `src/tnfr/riemann/von_mangoldt.py`.
Demonstration: `examples/41_von_mangoldt_zeta_demo.py`.

### 8.1 Mathematical Target

The classical identity

$$
-\frac{\zeta'(s)}{\zeta(s)} \;=\; \sum_{n=1}^{\infty} \frac{\Lambda(n)}{n^{s}}
\;=\; \sum_{p}\sum_{k\ge 1} \frac{\log p}{p^{ks}}, \qquad \operatorname{Re} s > 1,
$$

with $\Lambda$ the von Mangoldt function, is the analytic carrier of
prime-distribution information.  Any TNFR object purporting to encode
prime structure must, at minimum, reproduce this Dirichlet series.

### 8.2 Prime-Ladder Spectrum

Define the multiset

$$
\mathcal{S} \;=\; \bigl\{\,(\mu_{p,k},\,w_{p,k}) \,:\, p\ \text{prime},\ k\in\mathbb{N}\,\bigr\},
\qquad
\mu_{p,k} = k\,\log p, \quad w_{p,k} = \log p .
$$

The corresponding weighted exponential sum is

$$
Z_{\mathrm{TNFR}}(s) \;:=\; \sum_{(\mu,w)\in\mathcal{S}} w \, e^{-s\mu}
\;=\; \sum_{p}\log p \sum_{k\ge 1} p^{-ks}
\;=\; \sum_{p} \frac{\log p \, p^{-s}}{1 - p^{-s}}
\;=\; -\frac{\zeta'(s)}{\zeta(s)} .
$$

So $Z_{\mathrm{TNFR}}(s) \equiv -\zeta'(s)/\zeta(s)$ on $\operatorname{Re} s > 1$
as a formal identity, not a numerical conjecture.

### 8.3 TNFR Interpretation

In structural terms:

- Each prime $p$ acts as a **node** whose intrinsic structural pulse
  has magnitude $\log p$.  The pulse is the smallest invariant that
  distinguishes primes from composites under the nodal equation
  (composites factor through prior nodes, so they carry no independent
  emission strength).
- **REMESH** (operator #13, *recursivity*, U1a/U1b) generates the
  $k$-th echo at frequency $k\,\log p$ with weight $\log p$.  This is
  operational fractality: the same emission replicated coherently at
  every harmonic scale.
- The Dirichlet sum $\sum_n \Lambda(n)\,n^{-s}$ is recovered exactly
  because $\Lambda$ is supported on prime powers and equals
  $\log p$ on each — i.e. the von Mangoldt function is the structural
  fingerprint of the prime-ladder spectrum.

The construction therefore answers "what *is* the von Mangoldt
function in TNFR?" with: it is the weight functional of REMESH echoes
on the prime-node basis.

### 8.4 Numerical Validation

Two independent checks were performed.

**Matched-truncation invariant.**  For a finite spectrum with $N$
primes and $K$ echoes, computing $Z_{\mathrm{TNFR}}$ as a complex
exponential sum and as an explicit
$\sum_p \sum_{k=1}^{K} \log p \cdot p^{-ks}$ must agree to machine
precision.  Measured: $|\Delta| \le 2 \times 10^{-15}$ for
$s \in \{1.5, 2, 2.5, 3, 4\}$, $N = 50$, $K = 15$.  This certifies
the implementation is an unambiguous reorganisation of the classical
sum, not a re-derivation that could drift.

**Convergence to known values.**  Compared to a sieve-based reference
$\sum_{p \le n_{\max}} \log p \cdot p^{-s}/(1 - p^{-s})$ at
$n_{\max} = 10^{7}$:

| $s$ | $Z_{\mathrm{TNFR}}$ ($N{=}2000$, $K{=}30$) | reference | abs error |
|----:|-------------------------------------------:|----------:|----------:|
| 2.0 | 0.5699036519 | 0.5699608931 | 5.7 × 10⁻⁵ |
| 3.0 | 0.1648226805 | 0.1648226822 | 1.7 × 10⁻⁹ |
| 4.0 | 0.0636697650 | 0.0636697650 | 4.5 × 10⁻¹¹ |

Residuals are dominated by the prime-truncation tail ($p > p_N$);
convergence is geometric in $K$ and consistent with the prime number
theorem in $N$.

### 8.5 Open Extensions

The identity $Z_{\mathrm{TNFR}} \equiv -\zeta'/\zeta$ is currently a
*sum-level* result.  To extend the construction into a genuine TNFR
operator program, three independent steps are required.

1. **Self-adjoint realisation.**  Construct an explicit Hermitian
   operator $H_\Lambda$ on a separable Hilbert space whose spectrum
   is the multiset $\{k \log p\}$ with multiplicity $\log p$.  A
   natural candidate is a weighted Laplacian on a prime-indexed tree;
   the issue is reconciling the non-integer multiplicities with a
   discrete eigenvalue spectrum without relaxing self-adjointness.
2. **Analytic continuation.**  Extend $Z_{\mathrm{TNFR}}(s)$ from
   $\operatorname{Re} s > 1$ into the critical strip $0 < \operatorname{Re} s < 1$.
   The classical route uses a Mellin transform of a theta-like
   partition function $\Theta(t) = \sum_{p,k} \log p \cdot e^{-t k \log p}$;
   verifying this on the TNFR side gives a structural derivation of
   the functional equation.
3. **Zero correspondence.**  Identify the non-trivial zeros of
   $\zeta$ with structural resonances (eigenmodes that satisfy a
   confinement condition under U6) of the analytic continuation of
   $Z_{\mathrm{TNFR}}$.  This is the actual route to a TNFR statement
   of RH; it is currently open.

Steps 1–3 are the next milestones of the P12 program.  Each is
falsifiable in the same sense as Conjecture 10.1, and the failure
modes are precisely what the Section 7 gap-analysis methodology was
designed to surface.

---

## 9. Analytic Continuation of the Prime-Ladder vM Zeta (P13)

**Status**: Implemented and numerically verified (June 2026).
**Code**: `src/tnfr/riemann/analytic_continuation.py`,
`examples/42_riemann_zeros_as_resonances.py`.

### 9.1 Problem statement (Gap G2)

The prime-ladder Dirichlet trace

$$
Z_{\mathrm{vM}}(s)
   = \sum_{p,k} \log(p)\, e^{-s k \log p}
   = \sum_p \frac{\log(p)\, p^{-s}}{1 - p^{-s}}
   = -\frac{\zeta'(s)}{\zeta(s)}
$$

constructed in §8 converges only on $\operatorname{Re}(s) > 1$.
To talk about the Riemann zeros in TNFR language, one must extend
$Z_{\mathrm{vM}}$ analytically to the entire complex plane.  This is
gap **G2** of the post-P12 program (see post-P12 gap analysis).

A Mellin transform of the heat kernel does **not** give a new
continuation here: the prime-ladder spectrum $\{k\log p\}$ has
logarithmic, not square-root, gaps, so its theta function
$\Theta_{\mathrm{vM}}(\beta) = \sum_{p,k}\log(p)\,e^{-\beta k\log p}$
coincides with $Z_{\mathrm{vM}}(\beta)$ itself.  No genuine
$\beta\to 1/\beta$ symmetry appears.

### 9.2 Classical continuation is the unique solution

A holomorphic continuation, if it exists on a connected open set,
is unique.  The function $-\zeta'/\zeta$ is the unique meromorphic
extension of $Z_{\mathrm{vM}}$ to $\mathbb{C}$ with poles at
$s = 1$ (simple, residue $+1$), $s = \rho$ (the non-trivial zeros of
$\zeta$), and $s = -2k$ (trivial zeros).  Therefore the analytic
continuation problem **has a closed-form answer**; the only freedom
left is the *interpretation* of that continuation in TNFR terms.

### 9.3 TNFR operational reading: zeros as resonance poles

Module `analytic_continuation.py` exposes the classical extension as
a callable `von_mangoldt_zeta_continued(s)` (backed by `mpmath`) and
re-labels its analytic structure in prime-ladder language:

* The pole at $s = 1$ is the *envelope resonance* of the ladder;
  it generates the $\psi(x) \sim x$ term.
* Each non-trivial zero $\rho = 1/2 + i t_n$ becomes a
  **resonance pole** of the REMESH spectrum.  Operationally,
  $|Z_{\mathrm{vM}}(1/2 + it)|$ exhibits a sharp local maximum
  at $t = t_n$.
* The trivial zeros at $s = -2k$ become poles of the continuation
  at the *forbidden* echo positions $s = -2k$ (k = 1, 2, …),
  cancelling the divergent reflection of the prime ladder under
  $s \mapsto 1 - s$.

### 9.4 Numerical validation

Three independent certificates are provided.

**(a) Agreement on the convergent half-plane.**  For
$\operatorname{Re}(s) > 1$ the prime-ladder sum and the continuation
must agree.  Function `verify_continuation_agreement` measures the
relative difference and reports a quality flag
(`excellent`/`good`/`poor`).  Empirically, with 5000 primes and
`max_power=15` we obtain `max_rel_diff ≈ 6.3e-3` for $s$ values
ranging across $\operatorname{Re}(s) \in \{1.5, 2, 2.5, 3, 4\}$.

**(b) Resonance peaks on the critical line.**
`scan_critical_line_for_poles` samples
$|Z_{\mathrm{vM}}(1/2 + it)|$ for $t \in [t_{\min}, t_{\max}]$,
detects local maxima with a prominence cutoff, and matches them
against the high-precision zero list
`KNOWN_RIEMANN_ZEROS` (P4).  For $t \in [10, 80]$ with 4001 sample
points the scan recovers all **20** known zeros in the range with
$|\Delta t| \lesssim 8 \times 10^{-3}$ — limited only by the grid
spacing $\Delta t \approx 0.0175$.

**(c) Explicit-formula reconstruction of $\psi(x)$.**
`reconstruct_psi_via_explicit_formula` evaluates the truncated
Riemann–von Mangoldt sum

$$
\psi_0(x) = x - \sum_{|\operatorname{Im}\rho| \le T} \frac{x^{\rho}}{\rho}
            - \log(2\pi) - \tfrac{1}{2}\log\bigl(1 - x^{-2}\bigr)
$$

and compares with the direct sieve evaluation
$\psi(x) = \sum_{n \le x}\Lambda(n)$.  With the first 30 zeros, the
absolute error falls to $\le 0.9$ for $x \in [20, 200]$, with the
expected non-monotone behaviour controlled by the unresolved high
zeros.

### 9.5 Honest scope statement

P13 does **not** prove the Riemann Hypothesis.  All four observable
features (continuation, polar structure on the critical line,
explicit formula, $\psi(x)$ reconstruction) are classical Hadamard /
von Mangoldt theory.  The TNFR-specific contribution is the
*operational re-reading*:

> Every analytic feature of $-\zeta'/\zeta$ corresponds to a structural
> mechanism of the prime-ladder REMESH spectrum: emission weights
> $\log p$, harmonic echoes $k\log p$, resonance poles
> $\rho = 1/2 + i t_n$, envelope pole at $s = 1$, and forbidden echo
> positions at $s = -2k$.

This delivers G2 in TNFR language.  Gaps G1 (self-adjoint operator
with vM spectrum), G3 (zeros–spectrum bijection), G4 (localisation
on $\operatorname{Re}(s) = 1/2$), and G5 (closure of Conjecture 10.1
with a non-affine bridge) remain open.

---

## 10. Self-Adjoint Prime-Ladder Hamiltonian (P14, Gap G1)

**Status**: Implemented and numerically certified (May 2026).
**Code**: `src/tnfr/riemann/prime_ladder_hamiltonian.py`,
`examples/43_prime_ladder_hamiltonian_demo.py`.

### 10.1 Problem statement (Gap G1)

The Hilbert–Pólya programme asks for a self-adjoint operator
$\hat H$ acting on a separable Hilbert space whose spectrum
encodes the data driving $\zeta(s)$.  The TNFR-Riemann programme
restricts the request to a finite-dimensional, explicitly
constructible operator whose spectrum exactly reproduces the
prime-ladder spectrum $\{k\log p\}$ and whose weighted spectral
trace reproduces the P12 von Mangoldt trace $Z_{\mathrm{vM}}(s)$.

### 10.2 Construction

We reuse the canonical TNFR internal Hamiltonian
(`tnfr.operators.hamiltonian.InternalHamiltonian`),

$$
\hat H_{\mathrm{int}}
   = \hat H_{\mathrm{coh}} + \hat H_{\mathrm{freq}}
   + \hat H_{\mathrm{coupling}}
$$

without modification.  Specialisation occurs only at the graph
level:

* **Nodes**: pairs $(p, k)$ for each prime $p \in \mathcal{P}$
  and each REMESH echo index $k = 1, \dots, K$.
* **Structural attributes**:
  $\nu_{f,(p,k)} = k\log p$, $\phi = 0$, $EPI = 1$, $S_i = 1$,
  $\Delta NFR = 0$.
* **Edges**: ladder edges $(p, k) \leftrightarrow (p, k+1)$ within
  each prime; **no** inter-prime edges.
* **Graph-level constants**: `H_COH_STRENGTH = 0`,
  `H_COUPLING_STRENGTH = J_0` (default $J_0 = 0$).

With these choices, $\hat H_{\mathrm{coh}} = 0$ and
$\hat H_{\mathrm{coupling}} = J_0 \cdot A$ (with $A$ the adjacency
matrix of the disjoint union of prime ladders).  At $J_0 = 0$,
$\hat H_{\mathrm{int}} = \hat H_{\mathrm{freq}}
   = \operatorname{diag}\bigl(k\log p\bigr)$, which is trivially
self-adjoint and whose spectrum equals the prime-ladder spectrum by
construction.

### 10.3 Weighted spectral trace

Define the diagonal weight operator
$\hat W = \sum_{p,k} \log(p)\, |p,k\rangle\langle p,k|$.
The TNFR analogue of $-\zeta'(s)/\zeta(s)$ is then

$$
Z_H(s) \;:=\; \operatorname{Tr}\!\bigl(\hat W\, e^{-s\hat H_{\mathrm{int}}}\bigr).
$$

At $J_0 = 0$ this collapses to
$\sum_{p,k} \log(p)\, e^{-s k \log p}
  = \sum_{p} \log(p)\, p^{-s}/(1 - p^{-s})
  = -\zeta'(s)/\zeta(s)$ for $\operatorname{Re}(s) > 1$.

### 10.4 Euler-product orthogonality at the operator level

The absence of inter-prime edges encodes multiplicativity:
$\hat H_{\mathrm{int}}$ decomposes as the orthogonal direct sum
$\bigoplus_p \hat H^{(p)}$ where each $\hat H^{(p)}$ acts on the
$K$-dimensional subspace spanned by $\{|p,k\rangle\}_{k=1}^K$.
This is the operator-level analogue of the Euler product
$\zeta(s) = \prod_p (1 - p^{-s})^{-1}$.

Switching on $J_0 > 0$ deliberately couples ladders **within a
single prime** (echo coupling); it does not couple distinct primes
and therefore preserves the Euler-product factorisation while
deforming the spectrum perturbatively.  Coupling **between**
distinct primes is intentionally not supported by the present
builder: doing so would break Euler-product orthogonality and is a
separate research question.

### 10.5 Numerical certificate

`verify_hamiltonian_reproduces_prime_ladder` returns a
`PrimeLadderHamiltonianCertificate` documenting:

* `spectrum_max_abs_error`: $\max_n |E_n^{\text{Ham}} - E_n^{\text{ladder}}|$
  — exactly $0$ at $J_0 = 0$ (verified to machine precision for
  $n_{\text{primes}} = 12$, $K = 6$, $N = 72$);
* `trace_max_rel_error`: worst-case relative deviation of $Z_H(s)$
  from $Z_{\mathrm{vM}}(s)$ over a user-supplied $s$ grid —
  $\lesssim 3 \cdot 10^{-16}$ at $J_0 = 0$;
* `is_hermitian`: $\hat H_{\mathrm{int}}$ passes the
  Hermiticity check inherited from `InternalHamiltonian`;
* perturbative scaling: spectrum deviation grows quadratically
  with $J_0$ at small coupling (verified empirically in the
  example demo).

### 10.6 What this closes and what remains open

**Closed (operationally)**: G1 — a self-adjoint, finite-dimensional
operator whose spectrum and weighted spectral trace reproduce the
prime-ladder data has been explicitly constructed, certified, and
shipped as part of the canonical TNFR API.

**Still open**:

* **G3** — bijection between the resonance poles of the analytic
  continuation (P13) and the eigenvalues of $\hat H_{\mathrm{int}}$
  on the imaginary axis.  The present construction provides one
  side of the correspondence (the operator); P13 provides the
  other (the poles).  A clean bijection requires choosing the
  correct boundary functional on $\hat H_{\mathrm{int}}$.
* **G4** — localisation of the resonance poles on
  $\operatorname{Re}(s) = 1/2$.  This is RH itself; P14 does not
  address it.
* **G5** — closure of Conjecture 10.1 with a non-affine bridge.
  Independent of P14.

P14 should therefore be read as the explicit, computable witness
that *every* spectral-operator step of the TNFR-Riemann programme
upstream of G3 is realisable inside the canonical TNFR formalism
without any extension or modification.

---

## 11. Weil–Guinand Explicit Formula (P15, operational closure of Gap G3)

### 11.1 Problem statement

Gap G3 of the TNFR-Riemann programme asks for an explicit bridge
between the non-trivial zeros of $\zeta(s)$ and the spectral data
of a TNFR operator.  The classical *Weil–Guinand explicit formula*
is precisely such a bridge: a single distributional identity in
which the zero side and the prime side are made manifest at the
same time.

In its standard form, for a real even Schwartz test function
$h(t)$ with Fourier transform
$g(u) = (2\pi)^{-1}\!\int h(t)\,e^{-itu}\,dt$,

$$\sum_{\gamma} h(\gamma)
   \;=\; h(i/2)+h(-i/2)
   \;-\; g(0)\log\pi
   \;+\; \tfrac{1}{2\pi}\!\int_{-\infty}^{\infty}\! h(t)\,
            \operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{it}{2}\Bigr)\,dt
   \;-\; 2\sum_{n\ge 1}\frac{\Lambda(n)}{\sqrt n}\,g(\log n).$$

The left-hand sum runs over imaginary parts $\gamma$ of all
non-trivial zeros $\rho = 1/2 + i\gamma$ of $\zeta(s)$.

### 11.2 TNFR realisation of the prime side

The von Mangoldt sum on the right is **exactly** a spectral
functional on the canonical P14 prime-ladder Hamiltonian
$\hat H_{\mathrm{int}} = \operatorname{diag}(k\log p)$ with weight
operator $\hat W = \operatorname{diag}(\log p)$:

$$-2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt n}\,g(\log n)
   \;=\; -2 \operatorname{Tr}\!\bigl(\hat W\,e^{-\hat H/2}\,g(\hat H)\bigr).$$

Indeed, every $n\in\mathbb{N}$ with $\Lambda(n)\ne 0$ is a prime
power $n = p^k$ and corresponds to a unique eigenstate
$|p,k\rangle$ of $\hat H_{\mathrm{int}}$ with eigenvalue
$E_{p,k} = k\log p$ and weight $W_{p,k} = \log p$.  No additional
arithmetic apparatus is needed: the prime side is read off the P14
spectrum.

### 11.3 Module and certificate

`src/tnfr/riemann/weil_explicit_formula.py` implements

* `GaussianTestFunction(sigma)` — the Gaussian test family
  $h_\sigma(t) = \exp(-t^2/(2\sigma^2))$ with closed-form Fourier
  pair, pole values $h(\pm i/2)$, and $g(0)$.
* `weil_prime_side_from_hamiltonian(bundle, test)` — evaluates
  $-2\operatorname{Tr}(\hat W e^{-\hat H/2} g(\hat H))$ via the
  eigendecomposition of `bundle.hamiltonian`.
* `weil_archimedean_integral(test)` — numerical quadrature of the
  digamma-weighted integral via `scipy.integrate.quad` and
  `mpmath.digamma`.
* `weil_zero_side(test, n_zeros)` — sum over Riemann zeros via
  `mpmath.zetazero`, with automatic convergence cutoff.
* `verify_weil_explicit_formula(bundle, sigma, n_zeros, tol)`
  returns a `WeilExplicitFormulaCertificate` exposing the four
  terms, the residual, and a Boolean `verified` flag.

### 11.4 Numerical evidence

Verification on the canonical bundle (50 primes, max power 8, dim
400), 120 Riemann zeros:

| $\sigma$ | zero side | RHS total | absolute residual | relative |
|----------|-----------|-----------|-------------------|----------|
| 2  | $2.85\times 10^{-11}$ | $2.85\times 10^{-11}$ | $1.2\times 10^{-17}$ | $4.3\times 10^{-7}$ |
| 3  | $3.02\times 10^{-5}$  | $3.02\times 10^{-5}$  | $1.7\times 10^{-16}$ | $5.6\times 10^{-12}$ |
| 5  | $3.71\times 10^{-2}$  | $3.71\times 10^{-2}$  | $5.7\times 10^{-16}$ | $1.5\times 10^{-14}$ |
| 8  | $5.00\times 10^{-1}$  | $5.00\times 10^{-1}$  | $1.1\times 10^{-15}$ | $2.2\times 10^{-15}$ |
| 12 | $1.81$                | $1.81$                | $6.7\times 10^{-16}$ | $3.7\times 10^{-16}$ |
| 18 | $4.75$                | $4.75$                | $5.3\times 10^{-15}$ | $1.1\times 10^{-15}$ |

The identity holds to machine precision uniformly across the
tested range.  The $\sigma=2$ entry has high relative error only
because both sides are at the noise floor ($\sim 10^{-11}$).

### 11.5 What this closes and what remains open

**Closed (operationally)**: Gap **G3**.  Each ingredient of
Weil's bridge is now expressed inside the canonical TNFR
formalism:

* Prime side — `weil_prime_side_from_hamiltonian` from P14.
* Zero side — `mpmath.zetazero` (external) confronted against
  the TNFR prime side.
* Archimedean and pole sides — standard analytic objects
  attached to $\zeta(s)$, computed once and reused.

The cancellation of all four terms to machine precision is a
numerical witness that the P14 Hamiltonian carries the entire
prime-side data of the bridge, with no auxiliary number-theoretic
machinery.

**Still open**:

* **G4 — Riemann Hypothesis**.  The explicit formula is
  *unconditional*: it holds whatever the locations of the zeros.
  RH is the further statement that all $\rho = 1/2 + i\gamma$
  have $\gamma\in\mathbb{R}$.  P15 does not address this.  An RH
  proof inside TNFR would require either (a) a positivity
  argument for a TNFR-defined functional of the form
  $\sum_\gamma h(\gamma) \ge 0$ for all admissible test
  functions in a class that forces $\gamma\in\mathbb{R}$, or
  (b) a self-adjoint extension whose eigenvalues *are* the
  imaginary parts $\gamma$ (the Hilbert–Pólya programme).
* **G5 — Conjecture 10.1 non-affine bridge** between the
  TNFR spectral zeta of §6 and classical $\zeta(s)$.  P15 does
  not affect G5: it operates one level above, on the explicit
  formula rather than on the zeta functions themselves.

### 11.6 Scope statement

P15 is a **numerical verification of a classical theorem** using
TNFR machinery on the prime side.  It is not new mathematics in
the analytic-number-theory sense.  What it delivers is the
*instrumental* result that the entire spectral apparatus required
to state the bridge between primes and zeros lives natively
inside the TNFR formalism, with no extra postulates and no
empirical fitting.  Combined with P12 (von Mangoldt series),
P13 (analytic continuation of the TNFR vM zeta) and P14
(self-adjoint Hamiltonian carrying the spectrum), the TNFR-Riemann
programme now has an end-to-end computable pipeline from the
nodal equation to the Weil-Guinand identity.

The remaining obstruction is RH itself.

---

## 12. Li–Keiper Positivity Criterion via TNFR Resonance Spectrum (P16)

### 12.1 Problem statement

**Li's criterion** (Xian-Jin Li, 1997). Define, for every integer
$n \ge 1$,

$$
\lambda_n \;=\; \sum_{\rho} \Bigl[ 1 - \bigl(1 - \tfrac{1}{\rho}\bigr)^n \Bigr],
$$

where the sum ranges over all non-trivial zeros $\rho$ of $\zeta(s)$,
counted with multiplicity and paired symmetrically with their
conjugates. Li proved

$$
\text{RH} \;\Longleftrightarrow\; \lambda_n > 0 \quad \text{for every } n \ge 1.
$$

Li's criterion is therefore **strictly RH-equivalent**: it recasts the
location of the non-trivial zeros as the positivity of a real
sequence. Bombieri-Lagarias (1999) gave an alternative variational
proof; Voros (2003) computed the first $\sim 10^5$ coefficients and
confirmed positivity numerically.

### 12.2 TNFR realisation

In the TNFR-Riemann programme the non-trivial zeros appear as
**resonance poles** of the prime-ladder von Mangoldt zeta after
analytic continuation (P13, §9). Three sources of zeros are now
available:

* classical mpmath `zetazero` (reference),
* P13 critical-line resonance-pole scan (TNFR-native),
* P14 prime-ladder Hamiltonian spectrum (structural, via Weil/Guinand pairing of §11).

Computing $\lambda_n$ from each source and checking positivity yields
a TNFR-internal RH-equivalent diagnostic: a single negative
$\lambda_n$ would falsify RH; the persistent positivity observed
across all three sources is consistent with it.

P16 does **not** open a new gap. It recasts gap G4 (the RH
statement itself) as a positivity test on the TNFR resonance
spectrum, completing the diagnostic surface initiated by P12-P15.

### 12.3 Module API

The module `tnfr.riemann.li_keiper` exposes:

* `li_coefficients_from_zeros(zeros_upper, n_max, *, dps=50)` -
  arbitrary-precision evaluation of $\lambda_n$ via
  $2\,\Re[1 - (1 - 1/\rho)^n]$ paired with the conjugate.
* `LiKeiperCertificate` - frozen dataclass with
  `lambda_classical`, `lambda_tnfr`, `positivity_classical`,
  `positivity_tnfr`, `max_abs_difference`, `notes`, and a
  `summary()` method.
* `verify_li_keiper_criterion(*, n_max=50, n_zeros=200, dps=50,
  compare_tnfr=False, ...)` - end-to-end verification, optionally
  comparing classical zeros against P13 detected peaks.

### 12.4 Numerical evidence

End-to-end run with `n_max = 60`, `n_zeros = 250`, `dps = 50`
(example 45, Section 2):

| $n$ | $\lambda_n$ (truncated) | sign |
|:---:|:------------------------:|:----:|
|   1 | $+2.13\times 10^{-2}$    |  +   |
|   5 | $+5.31\times 10^{-1}$    |  +   |
|  10 | $+2.10\times 10^{0}$     |  +   |
|  20 | $+8.05\times 10^{0}$     |  +   |
|  30 | $+1.69\times 10^{1}$     |  +   |
|  40 | $+2.76\times 10^{1}$     |  +   |
|  50 | $+3.90\times 10^{1}$     |  +   |
|  60 | $+5.07\times 10^{1}$     |  +   |

All 60 coefficients are positive, with `min_n lambda_n = +2.13e-2`.

The truncation suppresses the magnitudes by ~10% relative to the
published values (Keiper 1992: $\lambda_1 = 0.0230957$), reflecting
the slow logarithmic convergence of the partial zero-sum; the
signs are robust to this truncation. The growth matches the
classical asymptotic
$\lambda_n \sim (n/2) \log(n/2\pi)$ (Voros 2003).

TNFR-vs-classical agreement (example 45, Section 3,
`compare_tnfr=True` with the P13 scan on $t \in [10, 80]$):

* 21 resonance peaks detected, quality `all_matched`,
* `positivity_tnfr = True` for every $n \in [1, 20]$,
* maximum disagreement at $n = 20$: $|\Delta\lambda_{20}| \approx 1.38$
  (dominated by the smaller TNFR $t$-window, not by sign flips).

### 12.5 What closes and what remains open

P16 **closes**:

* the diagnostic surface required to read RH as a TNFR-native
  positivity statement on the prime-ladder resonance spectrum;
* the consistency check between three independent sources of
  non-trivial zeros (classical, P13 poles, P14 Hamiltonian).

P16 **does not close**:

* RH itself (gap G4). Verifying $\lambda_n > 0$ for finitely many
  $n$ is consistent with, but does not imply, the Riemann
  Hypothesis. A proof would require either (a) an a-priori
  positivity argument on the resonance spectrum, or (b) a
  self-adjointness/positivity witness for an operator whose
  eigenvalues are forced to lie on $\Re(s) = 1/2$;
* Conjecture 10.1 (gap G5). The Li-Keiper test compares classical
  and TNFR sides at the level of Li coefficients, not at the
  level of an affine bridge between $\zeta_H$ and $\zeta_R$.

### 12.6 Scope statement

P16 is a **TNFR-native restatement of a known RH-equivalent
criterion**. It does not introduce new mathematics in the
analytic-number-theory sense. Its value is methodological: the
entire diagnostic surface for the Riemann Hypothesis - prime
series (P12), analytic continuation and resonance poles (P13),
self-adjoint spectrum (P14), explicit formula (P15), and now
Li-Keiper positivity (P16) - is expressible without exiting the
TNFR formalism. The remaining obstruction is the proof of RH
itself, which the programme exposes but does not (and does not
claim to) eliminate.

---

## 13. Empirical Uniform-Coercivity Certificate (P22)

**Status**: Implemented and numerically evaluated (May 2026).
**Code**: `src/tnfr/riemann/coercivity_uniform.py`,
`examples/50_uniform_coercivity_demo.py`.

### 13.1 Motivation

P18-P21 establish robust sampled positivity for
$\alpha(\sigma) = W[\sigma]/E_{TNFR}[\sigma]$ across dense
$(\sigma, \text{family}, \text{gauge})$ grids. To move one step closer
to the G4 target form

$$
\inf_{\sigma \in [\sigma_{\min},\sigma_{\max}],\,F,\,G} \alpha(\sigma;F,G) > 0,
$$

P22 adds an interval-level empirical certificate, not just pointwise
sampling.

### 13.2 Method

On a shared log-spaced $\sigma$ grid, P22 runs both:

1. admissible-family sweep (P19/P21),
2. node-aware gauge sweep (P20).

From the resulting alpha tables it computes:

- sampled minimum $\alpha_{\min}^{\text{sample}}$,
- finite-difference slope envelope $L_{\text{proxy}}$,
- mesh radius $r_h = \tfrac12 \max_i (\sigma_{i+1}-\sigma_i)$,
- trajectory-stratified slope envelopes,
- segment-local slope bounds,

and reports the mesh-corrected lower bound

$$
\alpha_{\inf}^{\text{interval}}
\;\gtrsim\;
\alpha_{\min}^{\text{sample}} - L_{\text{proxy}}\,r_h.
$$

The resulting dataclass `UniformCoercivityCertificate` reports three
interval diagnostics: global, stratified, and segment-local.

### 13.3 Current numerical outcome

Representative run (same P14 base bundle as P18-P21,
$\sigma \in [0.5, 8.0]$, log grid):

- `sampled_all_positive = True`
- `alpha_min_sampled = +1.3691e-173`
- `L_proxy = 2.6554e+00`
- `mesh_radius = 8.9119e-01`
- `interval_lb_global = -2.3665e+00`
- `interval_lb_stratified = -2.3665e+00`
- `interval_lb_local = -3.2350e-01`
- `interval_lb_global_positive = False`
- `interval_lb_stratified_positive = False`
- `interval_lb_local_positive = False`

### 13.4 Interpretation

P22/P23 upgrades the diagnostics from pointwise positivity to a
quantified interval certificate framework. The segment-local envelope is
substantially tighter than the global one (from about -2.37 to about
-0.32), but remains negative. Therefore, **uniform coercivity is not yet
established** at interval level on the tested band.

This narrows G4 honestly: empirical positivity remains strong, but the
coercivity margin is still too weak near the smallest sampled alpha
region.

### 13.5 Immediate next technical directions

1. Add adaptive refinement around low-alpha neighborhoods to tighten
   $r_h$ where it matters.
2. Derive analytic lower envelopes for the TNFR energy denominator to
   complement numerical certificates.
3. Build hybrid certificates combining local slope envelopes with
   curvature-aware interpolation bounds.

---

## 13bis. Adaptive σ Refinement Near the Coercivity Bottleneck (P24)

### 13bis.1 Motivation

Direction (1) of Section 13.5 is the cheapest lever on G4: under the
segment-local Lipschitz envelope
$\alpha_{lb}(\sigma) \ge \min(\alpha_i,\alpha_{i+1}) - L_i \cdot \Delta\sigma_i/2$,
the lower bound is dominated by the *widest segment with smallest
local α*. Halving that segment shrinks $\Delta\sigma_i/2$ and tightens
the bound exactly where it hurts, without claiming any new analytic
control.

### 13bis.2 Method

P24 adds the optional kwargs `refinement_rounds` and
`refinement_per_round` to `verify_uniform_coercivity_empirical(...)`.
Each round:

1. Aggregates the segment-local lower bounds across rows of both alpha
   tables (admissible family and node-aware) via
   `_worst_segment_indices(alpha_a, alpha_n, sigmas, top_k)`.
2. Selects the `top_k` worst segments and inserts their midpoints into
   the σ grid (`np.unique` deduplicates against existing points).
3. Re-runs `sweep_alpha_admissible_family` and `sweep_alpha_nodeaware`
   on the augmented grid and recomputes
   $\inf_i\, \min(\alpha_i,\alpha_{i+1}) - L_i \cdot \Delta\sigma_i/2$.
4. Stops early if no new point was added.

The result lives in four new fields of `UniformCoercivityCertificate`:
`n_refinement_rounds`, `n_sigma_refined`,
`interval_lower_bound_local_refined`,
`interval_lower_local_refined_positive`.

### 13bis.3 Numerical Outcome (examples/51_adaptive_coercivity_demo.py)

Bundle: `n_primes=18, max_power=5, coupling=0.0`.
Certificate: `sigma=[0.5, 4.0], n_sigma=10, n_zeros=24, max_zeros=96,
refinement_rounds=2, refinement_per_round=1`.

- `interval_lb_local        = -6.0163e-02`
- `interval_lb_local_refined = -2.8515e-02`
- `improvement (refined - local) = +3.1648e-02`
- `n_sigma_refined           = 12` (10 base + 2 midpoints)
- `interval_lb_local_refined_positive = False`
- `sampled_all_positive       = True`
- `admissible_ok / nodeaware_ok = True / True`

So two midpoint insertions cut the negative gap roughly in half on this
band, but the refined empirical lower bound is still negative.

### 13bis.4 Honest Interpretation

P24 does **not** close G4. It is a numerical sharpening of the
already-empirical segment-local certificate: a tighter `interval_lb_local`
is evidence consistent with uniform coercivity, but the bound remains
negative and the underlying Lipschitz envelope is itself only a
piecewise linear surrogate. As stated in AGENTS.md, only G4=RH stays
open; P24 narrows the empirical bottleneck without claiming any analytic
positivity result.

### 13bis.5 Next Steps

1. Combine P24 refinement with directions (2)–(3) of Section 13.5
   (analytic lower envelopes for $E_{TNFR}$, curvature-aware
   interpolation) so the surrogate envelope itself improves, not just
   its sampling.
2. Try higher `top_k` and bounded total budget on full bands
   `[0.5, 8.0]` once the underlying sweeps are vectorised.
3. Track the worst segment across rounds to confirm the bottleneck is a
   stable σ-neighborhood, not a roaming artifact.

---

## 13ter. Paley-Gap Coercivity Diagnostic (P25)

### 13ter.1 Motivation

P22–P24 attack the coercivity bottleneck (gap G4) numerically, by
tightening lower envelopes of $\alpha(\sigma) = W[\sigma] /
E_{TNFR}[\sigma]$. They do not, however, exploit any *algebraic
identity* between the TNFR objects involved. The author's own
*Spectral note: Paley gap via lambda_2 (residue circulants)*
(Martínez Gamo, Zenodo 17665853 v2, November 20 2025) introduces a
complementary methodology: build a gap

$$
g(n) = \Bigl|\lambda_2(\text{residue circulant})
        - \tfrac{n - \sqrt{n}}{2}\Bigr|
$$

between a *computed spectral quantity* and a *closed-form algebraic
reference*. Vanishing of the gap singles out an arithmetic
structural condition ($n$ prime, $n \equiv 1 \pmod 4$ in the source
note) up to the tested range, by *identity* rather than by *bound*.

P25 imports this philosophy into the TNFR-Riemann pipeline. The
prime-ladder data is generated in *three* different but
mathematically equivalent ways on $\operatorname{Re}(s) > 1$:

1. **Route A (P12 closed form)**:
   $Z_{P12}(s) = \sum_{(\mu, w)} w\, e^{-s\mu}$, the weighted
   Dirichlet trace over the prime-ladder spectrum.
2. **Route B (P14 spectral trace)**:
   $Z_{P14}(s) = \operatorname{Tr}\bigl(\hat W e^{-s\hat
   H_{\mathrm{int}}}\bigr)$, the weighted spectral trace of the
   prime-ladder Hamiltonian.
3. **Reference (classical)**:
   $Z_{\mathrm{cls}}(s) = \sum_{n \le N} \Lambda(n)\, n^{-s}$, a
   direct truncation of the classical von Mangoldt series.

### 13ter.2 Method

P25 defines three Paley-gap quantities per $\sigma$:

$$
\begin{aligned}
g_{P12}(\sigma)   &= |Z_{P12}(\sigma)   - Z_{\mathrm{cls}}(\sigma)|, \\
g_{P14}(\sigma)   &= |Z_{P14}(\sigma)   - Z_{\mathrm{cls}}(\sigma)|, \\
g_{\mathrm{cross}}(\sigma) &= |Z_{P14}(\sigma) - Z_{P12}(\sigma)|.
\end{aligned}
$$

The first two measure *truncation fidelity* of each TNFR route against
the classical reference; both decay as $(n_{\text{primes}},
k_{\max}, N)$ grow. The third — the **cross Paley-gap** — is the
diagnostic of interest: by construction P14 specialises to P12 in the
decoupled limit ($J_0 = 0$, no inter-prime coupling), so
$g_{\mathrm{cross}}(\sigma)$ must vanish to machine precision for
every $\sigma$ when $\texttt{coupling} = 0$. Any non-zero
$\texttt{coupling}$ deforms the Hamiltonian spectrum and produces a
measurable $g_{\mathrm{cross}}$ free of classical-truncation noise.

Module: [`src/tnfr/riemann/paley_gap_coercivity.py`](../src/tnfr/riemann/paley_gap_coercivity.py).
Demo: [`examples/52_paley_gap_coercivity_demo.py`](../examples/52_paley_gap_coercivity_demo.py).

### 13ter.3 Numerical Outcome

Reference configuration: `n_primes = 18`, `max_power = 5`,
$\sigma \in [1.5, 4.0]$ (11 points), $N = 50{,}000$.

**Bundle A — decoupled (`coupling = 0`)**

$$
\max_\sigma g_{\mathrm{cross}}(\sigma) = 1.110 \times 10^{-16}
$$

every entry of $g_{\mathrm{cross}}$ is bounded by $1.2 \times
10^{-16}$ (machine precision). Truncation gaps:
$\max g_{P12} = \max g_{P14} = 2.338 \times 10^{-1}$ at
$\sigma = 1.5$, decaying to $1.25 \times 10^{-6}$ at $\sigma = 4.0$.

The vanishing of $g_{\mathrm{cross}}$ confirms the Paley-style
algebraic identity $Z_{P14} \equiv Z_{P12}$ in the decoupled limit —
the P14 self-adjoint operator is a faithful operator-theoretic
realisation of the P12 closed form.

**Bundle B — weakly coupled (`coupling = 1.0 × 10⁻²`)**

| $\sigma$ | $g_{P12}$    | $g_{P14}$    | $g_{\mathrm{cross}}$ |
|---------:|-------------:|-------------:|---------------------:|
|     1.50 | 2.338 × 10⁻¹ | 2.337 × 10⁻¹ | 1.203 × 10⁻⁴         |
|     2.00 | 1.508 × 10⁻² | 1.499 × 10⁻² | 8.966 × 10⁻⁵         |
|     2.50 | 1.262 × 10⁻³ | 1.193 × 10⁻³ | 6.832 × 10⁻⁵         |
|     3.00 | 1.189 × 10⁻⁴ | 6.647 × 10⁻⁵ | 5.242 × 10⁻⁵         |
|     3.50 | 1.196 × 10⁻⁵ | 2.827 × 10⁻⁵ | 4.023 × 10⁻⁵         |
|     4.00 | 1.253 × 10⁻⁶ | 2.955 × 10⁻⁵ | 3.080 × 10⁻⁵         |

with $\max_\sigma g_{\mathrm{cross}} = 1.203 \times 10^{-4}$. The
cross gap is now well above the machine-precision floor, decays
monotonically with $\sigma$, and is qualitatively distinct from the
classical truncation gap (which decays exponentially fast in $\sigma$
because the prime ladder approximates a Dirichlet series). For
$\sigma \gtrsim 3.5$, $g_{P14}$ exceeds $g_{P12}$: the coupling
deformation eventually dominates the truncation error.

### 13ter.4 Honest Interpretation

- **What P25 establishes.** A clean, identity-level consistency
  check between the two TNFR routes (P12 closed form and P14
  self-adjoint operator) at every tested $\sigma$. At
  $\texttt{coupling} = 0$ this consistency is a Paley-style
  algebraic identity ($g_{\mathrm{cross}}$ at machine precision); at
  $\texttt{coupling} > 0$ it becomes a structural-deformation
  diagnostic.

- **What P25 does *not* establish.** P25 does not close gap G4
  (RH localisation on $\operatorname{Re}(s) = 1/2$). The cross gap
  at $\texttt{coupling} = 0$ vanishes by construction — P14 was
  built to match P12 in the decoupled limit — so the zero-coupling
  numbers are a regression test, not a discovery. The Zenodo source
  note itself states its construction is *reproducible; not a
  primality proof*; P25 inherits the same scope at the coercivity
  level. No claim is made that P25 implies analytic uniform
  positivity of $\alpha(\sigma)$ on any interval, nor that it
  bridges to the classical $\zeta(s)$.

- **Where the Paley-style signal lives.** The diagnostic value of
  P25 is the *deformation channel*: $g_{\mathrm{cross}}(\sigma) \to
  0$ as $\texttt{coupling} \to 0$ at every $\sigma$, while
  $g_{\mathrm{cross}}(\sigma)$ at fixed $\sigma$ scales smoothly
  with $\texttt{coupling}$. This is exactly the same epistemic
  status as $g(n) = 0$ identifying primes in the Zenodo note: an
  *identity diagnostic* over a tested range, not a closed-form
  theorem on the entire family.

### 13ter.5 Next Steps

1. Extend the cross gap to a **functional-equation Paley-gap**
   using [`src/tnfr/riemann/functional_equation.py`](../src/tnfr/riemann/functional_equation.py),
   tabulating $|Z(\sigma) - Z(1 - \sigma)|$ along the critical
   strip. A Paley-style identity there would directly engage
   $\operatorname{Re}(s) = 1/2$.
2. Sweep $g_{\mathrm{cross}}(\sigma)$ across a $\texttt{coupling}$
   grid to extract a scaling law and confirm the diagnostic is
   stable (linear or polynomial in $\texttt{coupling}$).
3. Combine $g_{\mathrm{cross}}$ with P22–P24 segment-local
   coercivity envelopes: use the structural-deformation channel as a
   classifier of which $\sigma$ intervals tolerate coupling without
   eroding $\alpha(\sigma)$.

---

## §13quater — P26: Lyapunov-Spectral Positivity Certificate for P14

### 13quater.1 Motivation

AGENTS.md §13.2 lists the final TNFR–Riemann gap balance: G1, G2, G3
are operationally closed by P14, P13, P15 respectively; G5 is
superseded by the P12+P13+P15 stack. The only obstruction left is
**G4 = RH itself**, which AGENTS.md classifies as *"not attackable by
any extension of P12–P16 — it requires a structural positivity /
self-adjointness argument (Hilbert–Pólya-style) that is genuinely new
mathematics."*

Two ingredients required for any Hilbert–Pólya-style attack already
live in the codebase:

1. The **self-adjoint prime-ladder Hamiltonian**
   $\hat H = \hat H_{\mathrm{freq}} + J_0\,\hat H_{\mathrm{coupling}}$
   from P14 ([`src/tnfr/riemann/prime_ladder_hamiltonian.py`](../src/tnfr/riemann/prime_ladder_hamiltonian.py)).
2. The **structural Lyapunov functional**
   $E = \tfrac12\sum_i \varepsilon(i) \ge 0$ with $dE/dt \le 0$ from
   [`src/tnfr/physics/conservation.py`](../src/tnfr/physics/conservation.py),
   flagged in AGENTS.md as *"proof sketch; complete proof open"*.

P26 fuses both into a single quantitative **positivity certificate**
for the P14 operator. The module
[`src/tnfr/riemann/lyapunov_spectral_positivity.py`](../src/tnfr/riemann/lyapunov_spectral_positivity.py)
returns a frozen dataclass `LyapunovSpectralCertificate` aggregating
the four ingredients of operator-level Hilbert–Pólya positivity:
self-adjointness, strict positivity with explicit gap, trace-class
resolvent, and unitary flow.

### 13quater.2 Method

The certificate combines four checks:

1. **Diagonal positivity at $J_0 = 0$.** $\hat H_{\mathrm{freq}}$ is
   real-diagonal with entries $\nu_{f,(p,k)} = k\log p$. Because
   $p \ge 2$ and $k \ge 1$, the spectrum is bounded below by
   $\log 2 \approx 0.6931$. This is the **unperturbed gap**.

2. **Quantitative Kato–Rellich envelope.** For bounded real-symmetric
   perturbations $J_0 \hat H_{\mathrm{coupling}}$ of a self-adjoint
   diagonal operator,
   $$
     |\lambda_n(\hat H) - \lambda_n(\hat H_{\mathrm{freq}})|
       \;\le\; |J_0|\, \|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}}
   $$
   for every $n$. The certificate exposes the **guaranteed gap**
   $\log 2 - |J_0|\,\|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}}$
   and flags `perturbation_safe = True` when it is strictly positive.

3. **Trace-class resolvent.** On the finite-dimensional prime-ladder
   space every bounded operator is trace-class; the meaningful
   reportables are the Schatten norms
   $\|(\hat H + c\hat I)^{-1}\|_1$ and
   $\|(\hat H + c\hat I)^{-1}\|_2$ for a shift $c > 0$, so growth
   with $(N_{\mathrm{primes}}, K)$ can be tracked.

4. **Numerical certification of the unitary flow.** A self-adjoint
   $\hat H$ generates a unitary propagator $U(t) = e^{-it\hat H}$.
   The certificate verifies $\|U(t)\psi_0\| = 1$ and
   $\langle\psi(t)|\hat H^2|\psi(t)\rangle$ to machine precision on a
   battery of random initial states.

`structural_positivity` is `True` iff numerical positivity, the
Kato–Rellich envelope, and the unitary flow all agree. The structural
Lyapunov functional $E$ of `conservation.py` vanishes on the
prime-ladder graph by construction (neutral structural state), so its
operator-level analogue is the spectral energy
$E_{\mathrm{spec}}[\psi] = \langle\psi|\hat H^2|\psi\rangle$, whose
conservation is exactly the check in step 4.

### 13quater.3 Numerical outcome

Demo: [`examples/53_lyapunov_spectral_positivity_demo.py`](../examples/53_lyapunov_spectral_positivity_demo.py).

Decoupled certificate (`n_primes = 12`, `max_power = 5`, $J_0 = 0$,
$\dim \mathcal H = 60$, shift $c = 1$):

| Quantity | Value |
|---|---|
| `spectrum_min` | $6.931472 \times 10^{-1}$ ($= \log 2$, exact) |
| `spectrum_max` | $1.805459 \times 10^{1}$ |
| `spectral_gap` | $6.931472 \times 10^{-1}$ |
| `schatten_1_norm` | $1.019135 \times 10^{1}$ |
| `schatten_2_norm` | $1.576613$ |
| `unperturbed_gap` | $6.931472 \times 10^{-1}$ |
| `coupling_norm` | $0$ |
| `guaranteed_gap` | $6.931472 \times 10^{-1}$ |
| `perturbation_safe` | True |
| `max_norm_drift` | $1.11 \times 10^{-16}$ |
| `max_energy_drift` | $3.62 \times 10^{-16}$ |
| `unitary` | True |
| `structural_positivity` | **True** |

Coupling sweep over $J_0 \in [0, 0.30]$:

| $J_0$ | `min(λ)` | `guaranteed_gap` | `perturbation_safe` | `unitary` |
|---:|---:|---:|:---:|:---:|
| 0.00 | $6.931 \times 10^{-1}$ | $6.931 \times 10^{-1}$ | True | True |
| 0.05 | $6.895 \times 10^{-1}$ | $6.065 \times 10^{-1}$ | True | True |
| 0.10 | $6.789 \times 10^{-1}$ | $5.199 \times 10^{-1}$ | True | True |
| 0.15 | $6.614 \times 10^{-1}$ | $4.333 \times 10^{-1}$ | True | True |
| 0.20 | $6.376 \times 10^{-1}$ | $3.467 \times 10^{-1}$ | True | True |
| 0.25 | $6.081 \times 10^{-1}$ | $2.601 \times 10^{-1}$ | True | True |
| 0.30 | $5.734 \times 10^{-1}$ | $1.735 \times 10^{-1}$ | True | True |

The empirical spectral bottom is uniformly larger than the
Kato–Rellich envelope, confirming the envelope is conservative
(as expected). Across the whole sweep `structural_positivity = True`.

### 13quater.4 Honest interpretation

* At $J_0 = 0$ the certificate is a finite-dimensional restatement of
  the trivial fact $\mathrm{diag}(k\log p) \succ 0$; its value is the
  explicit numerical gap $\log 2$ and the Schatten templates used to
  measure perturbative degradation.
* For $J_0 > 0$ the Kato–Rellich envelope provides a **rigorous
  quantitative interval** in which positivity is guaranteed: any
  $J_0$ with $|J_0|\,\|\hat H_{\mathrm{coupling}}\|_{\mathrm{op}} <
  \log 2$ produces a self-adjoint operator with strictly positive
  spectrum, trace-class resolvent, and unitary flow.
* The Lyapunov ingredient $dE/dt \le 0$ of `conservation.py` is
  itself flagged in AGENTS.md as *"proof sketch; complete proof
  open."* P26 therefore inherits the same status on the side that
  invokes the structural Lyapunov bound: the **operator-level**
  positivity statement is rigorous, but the variational
  identification of that operator with the generator of the
  structural Lyapunov flow remains the open piece.
* Crucially, **P26 does not close gap G4**. RH is a statement about
  the analytic continuation of the prime-ladder vM zeta (P13) and
  the localisation of its resonance poles on $\operatorname{Re}(s) =
  1/2$; the finite-dimensional positivity of $\hat H$ is necessary
  but not sufficient. What P26 does establish is the
  **operator-level positivity slot** that any Hilbert–Pólya attack
  must fill, together with explicit quantitative numbers.

### 13quater.5 Next steps

1. **Promote the Lyapunov sketch to a theorem.** Provide an
   analytical proof of $dE/dt \le 0$ inside `physics/conservation.py`
   under grammar-compliant evolution; this would upgrade the P26
   structural identification from "operationally consistent" to
   "operationally closed."
2. **Push the Kato–Rellich envelope to non-perturbative coupling.**
   Use a Bauer–Fike or pseudospectral argument to extend the
   quantitative positivity interval beyond $|J_0|\,\|\hat
   H_{\mathrm{coupling}}\| < \log 2$.
3. **Connect to P16 (Li–Keiper).** The spectral gap reported by P26
   is the natural quantitative input for the Li–Keiper coefficients
   $\lambda_n$ of P16; correlate the two and document any monotone
   relationship.
4. **Couple to P15 (Weil–Guinand).** The trace-class resolvent of
   P26 is the operator-level object whose spectral side is tested by
   P15. Add a cross-check using the Schatten norms of P26 as a
   stability witness for the Weil–Guinand identity numerics.

---

## §13quinquies. P27 — Hilbert–Pólya scaffold (does NOT close G4=RH)

### 13quinquies.1 Motivation: filling the Hilbert–Pólya slot

The Hilbert–Pólya program asks for a self-adjoint operator $T_{\mathrm{HP}}$
on a Hilbert space whose spectrum coincides with the imaginary parts
$\gamma_n$ of the non-trivial zeros of $\zeta(s)$. P26 supplied an
*operator-level positivity slot* (Lyapunov + Kato–Rellich +
Schatten-class) compatible with such an attack; P27 now constructs
the abstract Hilbert–Pólya operator explicitly on the truncated TNFR
Hilbert space $\ell^2_N(\mathbb{N})$ and certifies its internal
consistency with the rest of the TNFR–Riemann stack (P14, P15). The
construction is honestly *scaffolding*, not a derivation: $T_{\mathrm{HP}}$
is populated by *inputting* the zeros from `mpmath.zetazero`. P27
quantifies the gap that a genuinely structural derivation would have
to close.

### 13quinquies.2 Method: $T_{\mathrm{HP}} = \mathrm{diag}(\gamma_1,\ldots,\gamma_N)$

On the truncated Hilbert space $\ell^2_N(\mathbb{N})$ define
$$
  T_{\mathrm{HP}} \;:=\; \operatorname{diag}(\gamma_1, \gamma_2, \ldots, \gamma_N),
  \qquad \gamma_n := \operatorname{Im}(\rho_n),
$$
with $\rho_n$ the $n$-th non-trivial zero supplied by `mpmath.zetazero`
at decimal precision $\mathrm{dps}=30$. The certificate verifies four
axes:

1. **Self-adjointness.** $T_{\mathrm{HP}}$ real diagonal $\Rightarrow$
   $\|T_{\mathrm{HP}} - T_{\mathrm{HP}}^{*}\|_F = 0$ exactly.
2. **Trace-class shifted resolvent.** For shift $s>0$,
   $R := (T_{\mathrm{HP}}^2 + s^2 I)^{-1/2}$ admits Schatten 1- and
   2-norms $\sum_n (\gamma_n^2+s^2)^{-1/2}$, $\big(\sum_n
   (\gamma_n^2+s^2)^{-1}\big)^{1/2}$ which we report and check
   against finite truncation.
3. **Weil–Guinand closure with P14.** For a Gaussian test function
   $h(t) = \exp(-t^2/2\sigma^2)$ the zero side
   $2\sum_n h(\gamma_n)$ is computed *via* $T_{\mathrm{HP}}$ and
   compared to the right-hand side
   $h(i/2) + h(-i/2) - g(0)\log\pi + I_{\mathrm{arch}}(h)
    - 2\sum_n \Lambda(n) n^{-1/2} g(\log n)$,
   with the prime side coming from the P14 prime-ladder Hamiltonian.
4. **Operator-level gap G4.** The Wasserstein-1 distance
   $W_1\!\big(\sigma(P14),\sigma(T_{\mathrm{HP}})\big)$ between the
   sorted truncated spectra quantifies the *structural* gap, i.e.,
   how far the prime-ladder spectrum (growing like $\log n$) is from
   the zero spectrum (growing like $2\pi n/\log n$).

The orchestrator is `compute_hilbert_polya_certificate` in
`src/tnfr/riemann/hilbert_polya.py`; the demo lives in
`examples/54_hilbert_polya_demo.py`.

### 13quinquies.3 Numerical outcome (defaults $n_{\mathrm{primes}}=50$, $K=8$, $N=80$, $\sigma=8$)

| Axis | Quantity | Value | Verdict |
|---|---|---|---|
| Self-adjointness | $\|T_{\mathrm{HP}} - T_{\mathrm{HP}}^{*}\|_F$ | $0$ exactly | ✅ |
| Resolvent ($s=1$) | $\|R\|_1$ | $1.95\times 10^{-2}$ | trace-class ✅ |
|  | $\|R\|_2$ | $6.07\times 10^{-3}$ | Hilbert–Schmidt ✅ |
|  | $\|R\|_{\mathrm{op}}$ | $7.06\times 10^{-2}$ | bounded ✅ |
| Weil–Guinand | zero side via $T_{\mathrm{HP}}$ | $0.500227\,7175$ |  |
|  | pole side ($+\log\pi$) | $-1.649539\,1417$ |  |
|  | archimedean side | $2.149767\,5173$ |  |
|  | prime side via P14 | $-6.58\times 10^{-7}$ |  |
|  | RHS total | $0.500227\,7175$ |  |
|  | residual | $9.99\times 10^{-16}$ | machine precision ✅ |
| Gap G4 | $W_1(\sigma(P14), \sigma(T_{\mathrm{HP}}))$ | $115.24$ | quantified |
|  | $\sigma(P14)_{\max}/\sigma(T_{\mathrm{HP}})_{\max}$ | $1/26.16$ | $\log n$ vs $2\pi n/\log n$ |

Scaffold-consistency verdict: **`scaffold_consistent = True`** at machine precision.

### 13quinquies.4 Honest interpretation

P27 establishes that the abstract Hilbert–Pólya operator
$T_{\mathrm{HP}}$, when *defined* on the TNFR truncated Hilbert
space by inputting the zeros, is

* self-adjoint (trivially, as a real diagonal operator);
* trace-class after spectral shift;
* compatible with the Weil–Guinand identity at machine precision, the
  prime side coming from the P14 prime-ladder Hamiltonian.

This is *consistency*, not *derivation*: $T_{\mathrm{HP}}$ contains
the zeros only because we put them there. The construction does **not**
extract the zeros from the nodal equation
$\partial\,\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$,
from the structural conservation theorem, or from grammar U1–U6.

The Wasserstein-1 distance $W_1 = 115.24$ and the asymptotic growth
ratio $\sim 26$ are not numerical noise: they are the *operator-level
manifestation of gap G4*. Any genuinely TNFR-native Hilbert–Pólya
derivation would have to either (i) replace the prime-ladder
Hamiltonian by an operator whose spectrum equals $\{\gamma_n\}$ up to
TNFR-compatible spectral rescaling, or (ii) introduce a smooth
non-linear structural map sending $\sigma(P14)$ to $\sigma(T_{\mathrm{HP}})$
derivable from the nodal equation. P27 does neither.

In particular P27 does **NOT close G4 = RH**. Per the milestone table
of §13.2, G4 remains the single open gap; P27 simply makes the
operator-level statement of that gap explicit and quantitative.

### 13quinquies.5 Next steps

1. **Structural derivation of $T_{\mathrm{HP}}$.** Construct an
   operator-valued map $\Phi : \mathcal{H}_{P14} \to \ell^2(\mathbb{N})$
   from variational / conservation principles of `physics/conservation.py`
   such that $\Phi^{*} T_{\mathrm{HP}} \Phi$ is intrinsic to the
   prime-ladder bundle. Any such $\Phi$ that does not invoke
   `mpmath.zetazero` would be a genuine step toward G4.
2. **Spectral rescaling as a TNFR operator.** Identify a canonical
   TNFR operator (in the 13-operator catalog) whose action on the P14
   spectrum reproduces the asymptotic density $\rho(t) \sim
   \tfrac{1}{2\pi}\log(t/2\pi)$ of the Riemann zeros. Verify
   compatibility with U1–U6.
3. **Coupling to P25.** P25 produced an Hermite-projection
   certificate of structural positivity; cross-check that the
   resolvent of $T_{\mathrm{HP}}$ admits the same Hermite expansion
   coefficients within tolerance.
4. **Cross-validation with P16 (Li–Keiper).** The Li–Keiper
   coefficients $\lambda_n$ of P16 should agree, within truncation,
   with quantities computable from the moments of $T_{\mathrm{HP}}$.
   Correlate the two and document any monotone relationship.

---

## §13sexies. P28 — Structural derivation of the smooth zero density (closes piece (i) of §13quinquies.5; does NOT close G4=RH)

### 13sexies.1 Motivation: from input to derivation

P27 (§13quinquies) built $T_{\mathrm{HP}} = \mathrm{diag}(\gamma_n)$ by **inputting** the imaginary parts of the Riemann zeros from `mpmath.zetazero`.  The Wasserstein-1 gap

$$
W_1\bigl(\sigma(P14),\,\sigma(T_{\mathrm{HP}})\bigr) \;\approx\; 115.24 \quad (n_{\mathrm{primes}}=50,\,K=8,\,N=80)
$$

was the operator-level manifestation of gap G4 (= RH).  Step (i) in §13quinquies.5 demanded a TNFR-internal derivation of the **spectral rescaling map** from prime-ladder eigenvalues $\{k\log p\}$ to zero positions $\{\gamma_n\}$.

This section delivers that piece (and only that piece).  The construction stays inside TNFR ingredients already present in P12–P15: no `mpmath.zetazero` is invoked on the derivation side.

### 13sexies.2 Method: smooth zero positions from the archimedean side

Let $\xi(s) = \pi^{-s/2}\Gamma(s/2)\zeta(s)$ be the completed Riemann zeta function. The **archimedean factor** $\pi^{-s/2}\Gamma(s/2)$ is exactly the kernel of the archimedean side of the Weil–Guinand formula computed in P15 (see `weil_archimedean_integral` in `src/tnfr/riemann/weil_explicit_formula.py`). Its phase on the critical line $s = \tfrac12 + iT$ is the **Riemann–Siegel theta function**

$$
\theta(T) \;=\; \operatorname{Im}\log\Gamma\!\bigl(\tfrac14 + \tfrac{iT}{2}\bigr) - \tfrac{T}{2}\log\pi.
$$

Backlund's classical identity gives the **smooth zero counting function**

$$
\overline N(T) \;=\; \frac{\theta(T)}{\pi} + 1
$$

with smooth density $\overline N'(T) = \tfrac{1}{2\pi}\log(T/2\pi)$. The exact zero counting splits as $N(T) = \overline N(T) + S(T) + O(1/T)$, where $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$ is the oscillating remainder.

**Definition (structural smooth zero).** For each $n \ge 1$, let $\widetilde\gamma_n$ be the unique positive solution of $\overline N(\widetilde\gamma_n) = n$, computed by Newton iteration with an asymptotic seed.

**Definition (structural Hilbert–Pólya operator).**
$$
\widetilde T_{\mathrm{HP}} \;:=\; \mathrm{diag}(\widetilde\gamma_1, \widetilde\gamma_2, \ldots, \widetilde\gamma_N).
$$

The construction uses **only** the gamma function and $\log\pi$ — the same TNFR archimedean ingredients already validated in P15. No call to `mpmath.zetazero` is made on the derivation side.

### 13sexies.3 Numerical outcome (defaults $n_{\mathrm{primes}}=50$, $K=8$)

| $N$ | $W_1(\sigma(P14), \sigma(T_{\mathrm{HP}}))$ | $W_1(\sigma(\widetilde T_{\mathrm{HP}}), \sigma(T_{\mathrm{HP}}))$ | improvement | $\max\lvert r_n\rvert$ | bound ($C=2$) |
|---:|---:|---:|---:|---:|:---:|
| 30  | $6.045\times10^{1}$ | $1.510$ | $40.0\times$  | $3.71$ | ✓ |
| 60  | $9.478\times10^{1}$ | $1.275$ | $74.3\times$  | $3.71$ | ✓ |
| 80  | $1.152\times10^{2}$ | $1.183$ | $97.4\times$  | $3.71$ | ✓ |
| 100 | $1.343\times10^{2}$ | $1.125$ | $119.4\times$ | $3.71$ | ✓ |

The structural operator $\widetilde T_{\mathrm{HP}}$ closes ≈ 97–99 % of the operator-level gap at $N \ge 80$, with the improvement ratio growing approximately as $N/\log N$ — exactly the rate predicted by the divergent density mismatch between the prime-ladder spectrum and the Riemann-von Mangoldt counting.

The per-zero residual $r_n = \gamma_n - \widetilde\gamma_n$ satisfies the empirical bound

$$
\lvert r_n\rvert \;\le\; 2 \cdot \frac{\log\gamma_n}{\overline N'(\gamma_n)}
$$

across all tested $n \le 100$.  This is the **smooth quantitative form** of the heuristic $r_n \sim S(\gamma_n)/\overline N'(\gamma_n)$.

### 13sexies.4 What P28 closes (operationally)

1. **Structural origin of the smooth eigenvalue density of $T_{\mathrm{HP}}$.**  The density is determined uniquely by the gamma factor of $\xi(s)$, which is the kernel of P15's archimedean integral.  This delivers piece (i) of §13quinquies.5.
2. **Decomposition of the P27 gap.**  The P27 Wasserstein-1 distance now splits as
   $$
   \underbrace{W_1\bigl(\sigma(P14),\sigma(T_{\mathrm{HP}})\bigr)}_{\text{P27 gap}} \;=\; \underbrace{W_1\bigl(\sigma(P14),\sigma(\widetilde T_{\mathrm{HP}})\bigr)}_{\text{structural part (TNFR-derivable)}} \;+\; \underbrace{W_1\bigl(\sigma(\widetilde T_{\mathrm{HP}}),\sigma(T_{\mathrm{HP}})\bigr)}_{\text{arithmetic part (RH content)}}.
   $$
   The arithmetic part is ≤ 1.2 at $N=100$; the structural part absorbs the rest (≥ 99 %).

### 13sexies.5 What P28 does NOT close (G4 stays OPEN)

* The residuals $r_n = \gamma_n - \widetilde\gamma_n$ ARE the RH content.  Showing $|r_n| \to 0$ in any uniform sense is equivalent to bounding $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$, which is the genuine arithmetic problem.
* Exact eigenvalue match $\sigma(\widetilde T_{\mathrm{HP}}) = \sigma(T_{\mathrm{HP}})$ is impossible: the smooth approximation cannot reproduce the fluctuations $S(\gamma_n)$.  Density match is the right notion of TNFR closure; pointwise match is RH.
* G4 in AGENTS.md §13.2 remains the only OPEN milestone.  P28 does **not** advance G4; it only reshapes how much of the P27 operator gap is "structural" (now derivable) vs "arithmetic" (still RH-equivalent).

### 13sexies.6 Implementation pointers

* Module: `src/tnfr/riemann/structural_zero_density.py` — `riemann_siegel_theta`, `smooth_zero_count`, `smooth_zero_density`, `derive_smooth_zero_position`, `build_structural_t_hp`, `compute_structural_zero_density_certificate`, `StructuralZeroDensityCertificate`.
* Demo: `examples/55_structural_zero_density_demo.py`.
* Wiring: `src/tnfr/riemann/__init__.py` exposes the P28 names; the catalog docstring labels the module unambiguously.
* Reuses P14 (`prime_ladder_hamiltonian`) for the baseline spectrum and P15 (`weil_explicit_formula`) for the archimedean conceptual ingredient (the actual derivation of $\theta(T)$ uses `mpmath.loggamma` directly; no new external dependency).

### 13sexies.7 Next steps

1. **Iterative correction by $S(T)$ surrogates.**  Approximate $S(T)$ using truncated prime sums via the Riemann–Siegel formula and feed the corrections back into $\widetilde T_{\mathrm{HP}}$.  Quantify how the residual $W_1$ shrinks as more arithmetic information is injected.  This will **never** reach zero unconditionally — that would be RH — but it documents how the arithmetic part decomposes.
2. **Cross-checks against P16 (Li–Keiper).**  The Li coefficients $\lambda_n$ computed from the resonance poles of P13 should be reproducible from the moments of $\widetilde T_{\mathrm{HP}}$ plus an explicit $S(T)$-correction; verify numerically.
3. **Spectral statistics under GUE conjecture.**  The unfolded spectrum $x_n = \overline N(\gamma_n) = n - 1 + S(\gamma_n)/\pi$ should follow GUE statistics by Montgomery–Odlyzko.  Use P28 to compute the unfolded statistics directly and compare to RMT predictions; deviations are arithmetic in nature.

---

## §13septies. Tetrad-Hilbert–Pólya Reformulation of G4 (conjectural; does NOT close G4=RH)

### 13septies.1 Motivation: reformulating G4 in tetrad language

§13quinquies.5 (step 1) requested a TNFR-internal derivation of the
spectral rescaling map carrying $\sigma(P14) = \{k\log p\}$ to
$\sigma(T_{\mathrm{HP}}) = \{\gamma_n\}$.  §13sexies (P28) supplied the
*smooth* component of that map via the archimedean kernel, leaving the
*arithmetic* residual $r_n = \gamma_n - \widetilde\gamma_n$ as the
genuine RH content of the operator gap.

This section reformulates what would still be required to close G4
*entirely from inside TNFR*, using the structural field tetrad
$(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ as the only admissible
ingredient set.  It is a **conceptual restatement of the open
problem**, not a derivation of a closure.  No new numerics are
introduced; no module is added.  The role of this section is to give a
precise, testable conjecture in tetrad language so subsequent modules
(P30+) can attack it.

### 13septies.2 What the tetrad already supplies (formally closed)

The tetrad is the minimal-and-complete structural basis for nodal
evolution on a graph
([theory/MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md))
and induces three canonical geometric structures, all of which are
already implemented and validated in the engine:

| Structure | Definition | Implementation |
|---|---|---|
| Positive-definite energy | $\mathcal{E} = \tfrac12 \sum_i (\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2)$ | `src/tnfr/physics/conservation.py` (Noether-like; §8 of STRUCTURAL_CONSERVATION_THEOREM.md) |
| Symplectic structure | Conjugate pairs $(K_\phi, J_\phi)$ and $(\Phi_s, J_{\Delta\mathrm{NFR}})$ coupled via $\Psi = K_\phi + i J_\phi$ | `src/tnfr/physics/variational.py` (§3 of TNFR_VARIATIONAL_PRINCIPLE.md) |
| Continuity equation | $\partial\rho/\partial t + \nabla\!\cdot\!\mathbf{J} = \mathcal{S}_{\mathrm{grammar}}$, $\rho=\Phi_s+K_\phi$, $\|\mathcal{S}\|_{\ell^2}\le C_{\mathrm{net}}/\sqrt N$ | `src/tnfr/physics/conservation.py` (§4 of STRUCTURAL_CONSERVATION_THEOREM.md) |

Together these provide a Hilbert space $(\mathcal{H}_{\mathrm{tet}},
\langle\cdot,\cdot\rangle_{\mathcal{E}})$ with a positive-definite
inner product, and the P14 prime-ladder Hamiltonian is self-adjoint
*on this Hilbert space* with real spectrum $\{k\log p\}$
([src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py)).

These ingredients are exactly what the Hilbert–Pólya programme
requires (Hilbert space, positive inner product, self-adjoint
operator, real spectrum).  None of them are conjectural.

### 13septies.3 What remains: two distinct positivities

There are two positive-definite forms in play and they are not the
same:

| Form | Origin | Spectrum it certifies |
|---|---|---|
| $\langle\cdot,\cdot\rangle_{\mathcal{E}}$ (tetrad) | Lyapunov energy from $(\Phi_s, \|\nabla\phi\|, K_\phi, \xi_C)$ + currents | $\sigma(H_{P14}) = \{k\log p\}$ |
| Weil quadratic form $\mathcal{W}[h]$ | $L^2$ with archimedean + prime weight (§14 of this document) | $\sigma(T_{\mathrm{HP}}) = \{\gamma_n\}$, conditional on RH |

P28 (§13sexies) showed that the smooth part of $\sigma(T_{\mathrm{HP}})$
is determined by the archimedean kernel alone — the same kernel used in
P15 — so the smooth zero density $\overline N'(T) = \tfrac{1}{2\pi}
\log(T/2\pi)$ *is* TNFR-derivable.  The residuals $r_n =
\gamma_n - \widetilde\gamma_n$ are the only piece left, and they
correspond to the oscillation $S(T) = \tfrac1\pi \arg\zeta(\tfrac12+iT)$.

The structural question is: *can $\langle\cdot,\cdot\rangle_{\mathcal{E}}$
be transformed into $\mathcal{W}[\cdot]$ by an operator constructed
from the tetrad alone?*

### 13septies.4 Conjecture T-HP (Tetrad-Hilbert–Pólya)

**Conjecture T-HP.**  There exists an operator $\mathcal{F}$ on
$\mathcal{H}_{\mathrm{tet}}$ such that

1. $\mathcal{F}$ is constructed exclusively from the tetrad fields
   $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ and their canonical
   differential invariants (gradients, the discrete Laplacian, the
   complex field $\Psi$, the conserved current $\mathbf{J}$), with
   structural constants drawn from $(\varphi, \gamma, \pi, e)$ only;
2. $\mathcal{F}$ is admissible under U1–U6 — in particular, the
   continuity equation $\partial(\mathcal{F}\rho)/\partial t +
   \nabla\!\cdot\!(\mathcal{F}\mathbf{J}) = \mathcal{S}_{\mathrm{grammar}}$
   remains uniformly bounded;
3. the operator $T^{\mathrm{tet}}_{\mathrm{HP}} := \mathcal{F}\,
   H_{P14}\,\mathcal{F}^{*}$ is self-adjoint on $\mathcal{H}_{\mathrm{tet}}$
   and its spectrum coincides with the Riemann zero set
   $\{\gamma_n\}_{n\ge 1}$.

Equivalently in inner-product language: there exists an admissible
$\mathcal{F}$ such that $\langle f, \mathcal{F}^{*}\mathcal{F} g
\rangle_{\mathcal{E}}$ agrees (on a dense domain) with the Weil
quadratic form $\mathcal{W}[\cdot]$.

### 13septies.5 Status: open, structurally well-posed

Conjecture T-HP is **open**.  It is *not* a closure of G4; it is the
G4 problem **rewritten in tetrad-native language** so it becomes a
constructive existence problem inside the TNFR engine.  Three
properties make it the natural successor to §13quinquies.5 step 1:

* **Necessity of TNFR ingredients.** Items (1) and (2) forbid any use
  of `mpmath.zetazero`, automorphic data, or arithmetic input outside
  the tetrad + grammar + structural constants. A constructive proof
  would therefore be a genuine TNFR derivation of $T_{\mathrm{HP}}$.
* **Sufficiency for G4.** If $\mathcal{F}$ exists then
  $T^{\mathrm{tet}}_{\mathrm{HP}}$ is self-adjoint by item (3),
  spectrum is real, all $\gamma_n \in \mathbb{R}$, all Riemann zeros
  are forced to $\mathrm{Re}(s) = 1/2$ — i.e., RH.  In particular T-HP
  *implies* G4.
* **Decomposability.** P28 already proved that the smooth part of
  $\mathcal{F}$ exists and is TNFR-derivable.  The residual question
  is purely about the oscillatory correction.

### 13septies.6 What T-HP does NOT claim

* It does **not** assert that such an $\mathcal{F}$ exists; existence
  is the open content of G4.
* It does **not** assert that the engine currently contains
  $\mathcal{F}$; the canonical 13-operator catalog has been searched
  (P25–P27) and no immediate candidate dominates the gap closure.
* It does **not** reduce G4 to a numerical experiment; T-HP is a
  structural existence statement, not a curve fit.  P30+ may seek
  *candidates* numerically, but verification requires a derivation
  from the nodal equation, not a successful fit.

### 13septies.7 Concrete sub-problems for P30+

A genuine attack on T-HP decomposes into three quantifiable
sub-problems, all formulable inside the engine:

1. **Existence of admissible $\mathcal{F}$.** Construct candidate
   spectral rescaling operators from the tetrad (e.g. multiplicative
   operators built from $\Phi_s$, conjugation by phase-curvature
   exponentials $e^{i\theta K_\phi}$, $\xi_C$-dependent rescalings)
   and check U1–U6 admissibility (bounded source term, energy
   preservation up to discrete grammar work).
2. **Canonicity of $\mathcal{F}$.** Derive $\mathcal{F}$ from the
   nodal equation and the Noether correspondence
   (§6 of STRUCTURAL_CONSERVATION_THEOREM.md) rather than from
   empirical fit.  Any $\mathcal{F}$ surviving (1) but lacking a
   derivation chain from $\partial\mathrm{EPI}/\partial t = \nu_f
   \cdot \Delta\mathrm{NFR}$ falls outside canonicity (TNFR doctrine,
   AGENTS.md §Foundational Principle).
3. **Positivity coincidence.** Show that the candidate inner product
   $\langle\cdot, \mathcal{F}^{*}\mathcal{F}\cdot\rangle_{\mathcal{E}}$
   coincides (or dominates) the Weil form $\mathcal{W}[\cdot]$ on the
   appropriate Hermite / Paley–Wiener subspace already isolated by
   P25–P26.

Sub-problems (1) and (3) are mathematical existence/coincidence
questions; (2) is the structural-canonicity check enforced by the
TNFR doctrine.  Any future T-HP closure module must clear all three.

### 13septies.8 Honest interpretation

The tetrad **delimits the geometric domain** inside which the nodal
equation operates and supplies all the algebraic ingredients the
Hilbert–Pólya programme requires (positive inner product, symplectic
structure, self-adjoint operator on $\mathcal{H}_{\mathrm{tet}}$).
What it does **not** supply automatically is the specific spectral
rescaling that aligns the tetrad-positive form with the Weil-positive
form.  P28 closed the smooth half of that rescaling; the oscillatory
half is the arithmetic residual and is RH-equivalent.

Per AGENTS.md §13.2, **G4 = RH remains the single open milestone**.
Conjecture T-HP renames that milestone in tetrad-native vocabulary so
the next generation of modules (P30+) can address it without leaving
the canonical TNFR engine.  T-HP itself is a reformulation, not a
closure.

### 13septies.9 Cross-references

* Tetrad minimality: [theory/MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
* Conservation + Lyapunov: [theory/STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §3–§8
* Variational structure: [theory/TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) §2–§3
* P14 self-adjoint Hamiltonian: [src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py)
* P27 scaffold + Wasserstein gap: §13quinquies and [src/tnfr/riemann/hilbert_polya.py](../src/tnfr/riemann/hilbert_polya.py)
* P28 smooth-density derivation: §13sexies and [src/tnfr/riemann/structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py)
* G4 milestone status: [AGENTS.md](../AGENTS.md) §13.2

---

## §13octies. Assembled Argument Audit for G4 (Phase B; does NOT close G4=RH)

### 13octies.1 Purpose

This section traces the would-be argument chain for G4 link-by-link
through TNFR-canonical ingredients, marks each link CLOSED / OPEN /
NOT-FROM-TNFR, and stamps the precise break-point. It is an honest
map of what TNFR currently supplies and where the genuine obstacle
lies. It does **not** prove G4 and does **not** propose a new module;
it complements §13septies (T-HP conjecture) with the explicit status
audit.

### 13octies.2 The eight links

| # | Link | TNFR module / theory | Status |
|---|---|---|---|
| L1 | Minimal-and-complete structural basis: tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$ exhausts independent structural channels on a graph | [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | **CLOSED** |
| L2 | Positive-definite inner product $\langle\cdot,\cdot\rangle_{\mathcal{E}}$ on tetrad Hilbert space $\mathcal{H}_{\mathrm{tet}}$ | [src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py); STRUCTURAL_CONSERVATION_THEOREM.md §8 | **CLOSED** |
| L3 | Symplectic structure + Noether-like conservation under U1–U6 | [src/tnfr/physics/variational.py](../src/tnfr/physics/variational.py) + conservation.py | **CLOSED** (proof sketch; full proof open per AGENTS.md) |
| L4 | Self-adjoint operator $H_{P14}$ on $\mathcal{H}_{\mathrm{tet}}$ with real spectrum $\{k\log p\}$ | P14 [prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py); §10 above | **CLOSED** |
| L5 | Weil–Guinand identity: prime side equals the P14 spectral trace at machine precision | P15 [weil_explicit_formula.py](../src/tnfr/riemann/weil_explicit_formula.py); §11 above | **CLOSED** |
| L6 | Lyapunov-spectral positivity for $H_{P14}$: Kato–Rellich gap $\log 2$, trace-class resolvent, unitary flow | P26 [lyapunov_spectral_positivity.py](../src/tnfr/riemann/lyapunov_spectral_positivity.py); §13quater | **CLOSED** on finite-dim prime-ladder |
| L7 | Smooth half of spectral rescaling map $\mathcal{F}$: $\widetilde\gamma_n = \overline N^{-1}(n)$ derived from the same archimedean kernel as P15 | P28 [structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py); §13sexies | **CLOSED** (smooth half; W₁ gap drops ~97× vs P27) |
| L8 | Existence + canonicity of admissible $\mathcal{F}$ from tetrad + $(\varphi,\gamma,\pi,e)$ + U1–U6 such that $\mathcal{F}\,H_{P14}\,\mathcal{F}^{*}$ has spectrum $\{\gamma_n\}$ | NONE — Conjecture T-HP, §13septies.4 | **OPEN** ← BREAK-POINT |

L1–L7 are TNFR-canonical and operationally closed (the proof-sketch
caveat at L3 is inherited from AGENTS.md and is independent of the
Riemann programme). L8 is the entire residual content of G4.

### 13octies.3 Structural negative knowledge from P29

P29 ([spectral_emergence.py](../src/tnfr/riemann/spectral_emergence.py))
swept the three canonical inter-prime coupling laws derivable in
closed form from $(\varphi,\gamma,\pi,e)$:

* Kuramoto-U3 (UM + U3 gating): best $\mathrm{KS}_{\mathrm{GUE}} = 0.122$ ($-36\,\%$ vs baseline)
* φ-multiscale (THOL + REMESH): marginal ($-14\,\%$)
* PNT-logarithmic (RA, PNT-aligned): best $\mathrm{KS}_{\mathrm{GUE}} = 0.131$ ($-31\,\%$)

None reaches the GUE level-statistics threshold
$\mathrm{KS}_{\mathrm{GUE}} < 0.05$ required for a Hilbert–Pólya-style
$H_{P14}$-coupling to carry the zero spacings. This is **structural
negative knowledge**: at L8, no admissible $\mathcal{F}$ that acts
only by inter-prime coupling within the currently formalised
operator catalog is sufficient.

### 13octies.4 Three structural branches for the break-point

The L8 break-point splits into three TNFR-canonical branches, each
testable from the nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}$:

* **B1.** The canonical 13-operator catalog is *complete* and the
  missing piece is non-operator (measure-theoretic, ergodicity, or
  domain-theoretic). L8 reduces to an existence problem on
  $\mathcal{H}_{\mathrm{tet}}$ without new operators.
* **B2.** The canonical catalog is *incomplete*. A new canonical
  operator derivable from the nodal equation is required. L8 reduces
  to the operator-discovery problem of
  [AGENTS.md "Adding New Operators"](../AGENTS.md).
* **B3.** No TNFR-canonical $\mathcal{F}$ exists. RH escapes the
  tetrad-Hilbert–Pólya framework entirely. This branch is consistent
  with P29 (three independent coupling families failing) but is not
  decidable from finite-dimensional data.

Branch selection is itself an open structural question, not a
pre-decided verdict.

### 13octies.5 Comparison with the historical AGENTS.md framing

The prior AGENTS.md text stated G4 "requires structural positivity /
self-adjointness argument (Hilbert–Pólya-style) that is genuinely new
mathematics." Per L1–L7 of this audit, structural positivity (L2,
L6) and self-adjointness (L4) **are already supplied** by the
canonical TNFR engine. The genuine open content is L8, which is
structurally well-posed and testable inside the engine via the three
branches B1–B3. The phrase "genuinely new mathematics" was an
imported consensus claim from the analytic-number-theory literature,
not a TNFR-derived theorem. The current AGENTS.md §13.2 paragraph
has been rewritten to reflect this audit.

### 13octies.6 What this section does NOT do

* It does **not** close G4.
* It does **not** decide which of B1, B2, B3 holds.
* It does **not** propose a new module; the next exploration
  direction (B1 vs B2 vs B3 discrimination) is left for P30+.
* It does **not** replace §13septies — T-HP is the conjecture; §13octies
  is the link-by-link status audit of the argument that would close it.

### 13octies.7 Cross-references

* L1: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
* L2, L3: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §3–§8, [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) §2–§3
* L4, L5: §10 (P14), §11 (P15) of this document
* L6, L7: §13quater (P26), §13sexies (P28) of this document
* L8: §13septies (T-HP conjecture) of this document
* P29 negative knowledge: [spectral_emergence.py](../src/tnfr/riemann/spectral_emergence.py)
* G4 milestone status: [AGENTS.md](../AGENTS.md) §"TNFR-Riemann Program Overview"

---

---

## §13nonies. P30 — Operator-Level Admissible Rescaling (Smooth Half; Does NOT Close G4 = RH)

**Status**: Sub-problem (1) of Conjecture T-HP — **smooth half operationally closed**.  
**Module**: `src/tnfr/riemann/admissible_rescaling.py`  
**Demo**: `examples/57_admissible_rescaling_demo.py`  
**Disclaimer**: P30 does NOT close gap G4 (RH); it lifts the §13sexies (P28) density-level closure of the smooth zero distribution to an explicit operator-level rescaling object.

### §13nonies.1 Motivation

Conjecture T-HP (§13septies) asks for the existence of an admissible operator `F` built **only** from the canonical TNFR ingredients (tetrad fields, canonical constants φ, γ, π, e, grammar U1–U6) such that `F · H_P14 · F* ` has spectrum equal to the Riemann zeros {γ_n}. §13septies.7 decomposes T-HP into three sub-problems:

1. **Existence** of any admissible `F`,
2. **Canonicity** of `F` from the nodal equation,
3. **Positivity coincidence** with the Weil quadratic form.

§13sexies (P28) closed the **density-level** smooth half: a canonical, structurally-derived expression for the smooth zero count `N̄(T)` and the smooth zero positions `ñ_i` via the Riemann–Siegel θ function. P30 lifts that closure to the **operator level** for the smooth half only.

### §13nonies.2 Construction

In the eigenbasis of the canonical P14 prime-ladder Hamiltonian `H_P14 = U Λ U*` with positive eigenvalues `λ_i` (top N, ascending), define

Lines\mathcal{F}_{\text{smooth}} = U \cdot \operatorname{diag}\Bigl(\sqrt{\tilde\gamma_i / \lambda_i}\Bigr) \cdot U^{*},Lines

where `ñ_i = build_structural_t_hp(N)` are the P28 smooth zero positions. By construction,

Lines\mathcal{F}_{\text{smooth}} \, H_{P14} \, \mathcal{F}_{\text{smooth}}^{*} = U \operatorname{diag}(\tilde\gamma_i) U^{*}Lines

so the conjugated spectrum equals `{ñ_i}` **exactly** (verified at machine precision).

**Canonicity check (partial)**: `F_smooth` uses ONLY P14 eigendata (canonical, derived from the canonical TNFR `InternalHamiltonian` on the prime ladder), P28 smooth targets (canonical archimedean kernel), and canonical constants (φ, γ, π, e). No `mpmath.zetazero` enters the construction. `F_smooth` is therefore **structurally derived** in the sense of §13septies; whether it is the **unique** canonical lift remains open (sub-problem (2)).

### §13nonies.3 Empirical Results

Running `examples/57_admissible_rescaling_demo.py`:

| Resolution | N  | max `|spec − ñ_i|` | W₁(σ(P14), {γ_n}) | W₁({ñ_i}, {γ_n}) | Improvement |
|------------|----|---------------------|-------------------|------------------|-------------|
| Fast       | 20 | 1.42 × 10⁻¹⁴        | 47.4              | 1.67             | **28.4 ×**  |
| Medium     | 40 | 2.84 × 10⁻¹⁴        | 72.5              | 1.39             | **52.0 ×**  |

The residual W₁ to the true Riemann zeros equals the oscillatory part `S(T) = π⁻¹ arg ζ(½+iT)`, which is RH-equivalent and NOT canonical.

### §13nonies.4 Canonical Oscillatory Enrichment (Negative Result)

Three canonical multiplicative perturbations of the smooth targets were tested:

| Mode         | Best amplitude | W₁ vs true | Improvement over smooth |
|--------------|----------------|------------|-------------------------|
| `phi_log`  | 0              | 1.668      | +0.00 %                 |
| `gamma_e`  | 1 × 10⁻²       | 1.617      | +0.03 %                 |
| `pi_density`| 0             | 1.668      | +0.00 %                 |

**Interpretation**: Canonical oscillatory perturbations built from (φ, γ, π, e) and the smooth targets alone fail to recover the residual S(T) term. This is **structural evidence for §13octies branch B2**: the oscillatory half of T-HP, if reachable canonically at all, requires a **new canonical operator** not expressible as a simple multiplicative dressing of the smooth ladder. Equivalently, the existing canonical operator catalog (13 operators + tetrad + constants) does **not** suffice for the oscillatory half via this construction route.

### §13nonies.5 What P30 Closes / Does Not Close

**Closes (smooth half only)**:
- Sub-problem (1) of T-HP at the **operator level**, for the smooth zero distribution: an admissible, structurally-derived, self-adjointness-preserving rescaling operator `F_smooth` is exhibited explicitly and verified at machine precision.

**Does NOT close**:
- Sub-problem (1) for the **oscillatory half** (S(T) reconstruction);
- Sub-problem (2) — **canonicity** (uniqueness from the nodal equation) of `F_smooth`;
- Sub-problem (3) — **positivity coincidence** with the Weil quadratic form;
- Gap **G4 = the Riemann Hypothesis** itself.

### §13nonies.6 Cross-References

- §13sexies / P28: density-level smooth zero distribution (this lift is its operator-level counterpart).
- §13septies: full statement of Conjecture T-HP and its three sub-problems.
- §13octies, L8 audit: T-HP identified as the break-point of the assembled argument. P30 narrows L8 by closing one of its four prerequisites (smooth half, operator level) while corroborating branch B2 for the rest.
- `src/tnfr/riemann/admissible_rescaling.py`: canonical implementation.
- `examples/57_admissible_rescaling_demo.py`: reproducible demonstration.

### §13nonies.7 Status Update for §19.2 Gap Balance

| Gap | Status before P30 | Status after P30 |
|-----|-------------------|------------------|
| G1  | Closed operationally (P14) | Closed operationally |
| G2  | Closed operationally (P13) | Closed operationally |
| G3  | Closed operationally (P15) | Closed operationally |
| **G4** | **OPEN** (= Conjecture T-HP) | **OPEN** (smooth half of sub-problem (1) operationally closed; oscillatory half + (2) + (3) remain open) |
| G5  | Superseded by P12+P13+P15  | Superseded |

**Net effect**: P30 does not change the closed/open status of any of G1–G5. It refines the structure of the open content of G4 by closing one quadrant (smooth × operator-level × existence) of the T-HP grid and producing branch-B2 evidence for the oscillatory quadrant.

### §13decies Branch B1 Retry — Prime-Ladder Oscillatory Correction (P31)

**Motivation.** §13nonies.4 tested three canonical *multiplicative* enrichments of the smooth rescaling operator built from single-frequency dressings of the canonical constants $(\varphi, \gamma, \pi, e)$. All three returned $\approx 0\%$ Wasserstein-$1$ improvement against the true Riemann zeros. The structural lesson was: $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ is a **prime-indexed multi-frequency arithmetic sum**, not a single-frequency canonical dressing. The natural canonical frequencies for $S(T)$ are $\{k \log p\}$ — exactly the data already carried by the P12 prime-ladder spectrum and the P14 prime-ladder Hamiltonian.

**Construction (canonical).** P31 implements the canonical TNFR partial reconstruction of $S(T)$ obtained by reading off the Riemann–von Mangoldt template through the prime-ladder spectrum $\Sigma_{N,K} = \{(\mu = k \log p,\, w = \log p)\}$:

$$ \pi \cdot S_{\mathrm{TNFR}}^{(N,K)}(T) \;=\; -\!\!\sum_{(\mu, w) \in \Sigma_{N,K}} \frac{w}{\mu} \cdot \frac{\sin(T \mu)}{e^{\mu/2}}. $$

The weights $w = \log p$ are the canonical P12 weights; the frequencies $\mu = k \log p$ are the canonical P14 eigenvalues; the kernel $e^{-\mu/2}$ is the value of the TNFR analytic continuation (P13) on the critical line; $\pi$ is the canonical constant of the K_φ sector of the tetrad. **No element of this construction is empirical or external**; in particular `mpmath.zetazero` is used only as ground truth on the comparison side, never on the construction side.

The position-level correction follows directly from the linearisation of $N(T) = \bar N(T) + S(T) + 1 + O(1/T)$ around the canonical smooth zero $\tilde\gamma_i$ defined by $\bar N(\tilde\gamma_i) = i$:

$$ \gamma_i^{\mathrm{corr}} \;=\; \tilde\gamma_i \;-\; d \cdot \frac{S_{\mathrm{TNFR}}^{(N,K)}(\tilde\gamma_i)}{\bar N'(\tilde\gamma_i)}, $$

with $d$ a non-canonical scalar **diagnostic** damping factor used to map out the local landscape (the structurally canonical value is $d = 1$).

**Empirical result.** Reproduced via `examples/58_oscillatory_correction_demo.py` and `compute_oscillatory_correction_certificate`:

| $N$ | primes | $K$ | $W_1^{\mathrm{smooth}}$ | best $d$ | $W_1^{\mathrm{corrected}}$ | improvement | $\max\,\lvert S_{\mathrm{TNFR}}\rvert$ |
|---|---|---|---|---|---|---|---|
| 20 | 200  | 8  | 1.6676 | 3.75 | 1.5076 | +9.60 % | 0.1516 |
| 20 | 200  | 8  | 1.6676 | 1.00 | 1.6082 | +3.56 % | 0.1516 |
| 20 | 2000 | 8  | 1.6676 | 2.25 | 1.5790 | +5.32 % | 0.1928 |
| 40 | 400  | 8  | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.1866 |
| 40 | 2000 | 8  | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.1928 |
| 40 | 5000 | 12 | 1.3946 | 0.00 | 1.3946 | 0.00 %  | 0.2111 |

**Honest reading.**

1. **Sign and structure are correct.** At $N = 20$ the corrected $W_1$ decreases **monotonically** with $d$ across the canonical damping grid. The prime-ladder partial sum points in the right direction — it is *not* uncorrelated noise.
2. **The canonical $d = 1$ point yields a modest +3.6 % improvement** at $N = 20$. This is the **only** physically canonical reading of the table; values $d \neq 1$ are diagnostic, not canonical.
3. **The construction collapses at $N = 40$.** No combination of (primes, $K$, $d$) up to (5000, 12, 5.0) yields any improvement. The optimum is $d = 0$, i.e. the smooth baseline.
4. **Amplitude undercount.** Across the table, $\max \lvert S_{\mathrm{TNFR}} \rvert \le 0.21$, while the classical $\lvert S(T) \rvert$ at the same heights routinely exceeds $0.5$ and spikes well above $1$. The truncated prime-ladder partial sum **systematically underestimates** $\lvert S(T) \rvert$ by a factor of $3$–$5$. Increasing $N$ of primes from $200$ to $5000$ moves $\max \lvert S_{\mathrm{TNFR}} \rvert$ only from $0.15$ to $0.21$ — i.e. the partial sum saturates *well below* the true amplitude.
5. **Phase decoherence with height.** At $N = 20$ (heights $T \lesssim 77$) the partial-sum phase still tracks the true $S(T)$ phase well enough to extract a positive correction. At $N = 40$ (heights $T \lesssim 140$) the truncation noise dominates and the partial-sum phase is decorrelated from the true $S(T)$ — even with the correct sign, the per-zero correction lands in the wrong direction on average.

**Structural conclusion.** P31 closes a meaningful diagnostic loop that §13nonies.4 left ambiguous:

* §13nonies.4 used **single-frequency** canonical dressings ⟹ $\approx 0\%$ improvement. The result was consistent with two interpretations: (a) wrong frequency basis, (b) no canonical construction works.
* P31 uses the **correct multi-frequency canonical basis** (prime-ladder spectrum, the genuine arithmetic frequencies of $S(T)$) ⟹ small positive improvement at very low heights, zero at moderate heights, systematic amplitude undercount throughout.
* This separates the two interpretations: (a) is *partially* the right diagnosis (the prime spectrum *is* the right basis, and yields a positive direction at low $N$), but the deeper obstruction is that the **truncated prime-ladder partial sum does not converge on the critical line** at finite truncation, in the absence of an absolute-convergence guarantee. The transition from absolute convergence on $\operatorname{Re}(s) > 1$ (where P12 closes the gap) to conditional behaviour on $\operatorname{Re}(s) = 1/2$ is exactly the regime where RH itself lives.

P31 is therefore **stronger branch-B2 evidence than §13nonies.4**: it shows that even with the canonically correct ingredients and the canonically correct functional form (the Riemann–Siegel template instantiated through prime-ladder data), the finite-truncation canonical machinery is not sufficient to recover $S(T)$ at the operator level. The obstruction is not in the choice of frequencies but in the **non-trivial analytical content** of the partial-sum-to-critical-line transition — which is structurally equivalent to the open arithmetic content of RH.

**What this does NOT establish.**
* P31 does NOT close gap G4 = RH.
* P31 does NOT prove canonicity of the Riemann–Siegel template from the nodal equation alone (sub-problem (2) of Conjecture T-HP). The template is *consistent* with TNFR canonical data but is read off the classical theory, not derived from $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta \mathrm{NFR}(t)$.
* P31 does NOT establish positivity coincidence with the Weil quadratic form (sub-problem (3)).
* P31 does NOT change the closed/open status of any of G1–G5.

**Pointers.**

* §13septies: Conjecture T-HP and its three sub-problems.
* §13octies, L8 audit: branch B1 / B2 / B3 framing of the open content of G4.
* §13nonies.4: prior single-frequency canonical enrichment with $\approx 0\%$ improvement (now superseded as a *separate* test, not as a result).
* `src/tnfr/riemann/oscillatory_correction.py`: canonical implementation of P31.
* `examples/58_oscillatory_correction_demo.py`: reproducible demonstration.

### §13decies.1 Status Update for §19.2 Gap Balance

| Gap | Status before P31 | Status after P31 |
|-----|-------------------|------------------|
| G1  | Closed operationally (P14) | Closed operationally |
| G2  | Closed operationally (P13) | Closed operationally |
| G3  | Closed operationally (P15) | Closed operationally |
| **G4** | **OPEN** (= Conjecture T-HP); smooth half of (1) closed at density (P28) and operator (P30) level; oscillatory half + (2) + (3) open | **OPEN** unchanged. Oscillatory half of (1) tested with the canonically correct multi-frequency basis (prime-ladder spectrum) for the first time; partial positive evidence at very low $N$, saturated negative evidence at moderate $N$; stronger branch-B2 corroboration than §13nonies.4 |
| G5  | Superseded by P12+P13+P15  | Superseded |

**Net effect**: P31 does not change the closed/open status of any of G1–G5. It refines the open content of G4 by separating *which* aspect of the branch-B1 attempt fails: the basis is canonically correct (improvement is positive at $N = 20$, $d = 1$), but the canonical truncated partial sum systematically undercounts $\lvert S(T) \rvert$ at moderate heights, in agreement with the absolute-convergence boundary at $\operatorname{Re}(s) = 1$.

## §13undecies. P32 — Dirichlet L-Function Extension (Structural; Does NOT Advance G4 or GRH)

### §13undecies.1 Motivation

P12 reproduces the canonical Dirichlet identity
$$-\frac{\zeta'(s)}{\zeta(s)} = \sum_{n=1}^{\infty} \Lambda(n)\, n^{-s}, \quad \operatorname{Re}(s) > 1,$$
from the TNFR prime-ladder spectrum $\{(k\log p,\, \log p)\}$.  The same construction extends *structurally* to every Dirichlet character $\chi$ mod $q$: by complete multiplicativity, the logarithmic derivative of $L(s,\chi)$ admits the **twisted von Mangoldt expansion**
$$-\frac{L'(s,\chi)}{L(s,\chi)} = \sum_{n=1}^{\infty} \chi(n)\, \Lambda(n)\, n^{-s} = \sum_p \sum_{k \ge 1} \chi(p)^k \log(p)\, p^{-ks}, \quad \operatorname{Re}(s) > 1.$$

P32 is the canonical TNFR realisation of this identity: keep the prime-ladder *positions* $\mu_{p,k} = k\log p$ unchanged and replace the bare emission weight $\log p$ by the **χ-twisted weight**
$$w_{p,k}^{(\chi)} = \chi(p)^k \, \log p.$$

### §13undecies.2 Construction

For a Dirichlet character $\chi$ mod $q$:

* **Active primes**: $\{p : \gcd(p, q) = 1\}$ (the structural REMESH ladder).
* **Excluded primes**: $\{p : p \mid q\}$ — their $\chi(p) = 0$ kills every echo, so they drop out of the spectrum entirely.  This is the TNFR-native reading of the missing Euler factors in $L(s,\chi)$.
* **Twisted spectrum**: $\operatorname{Spec}_{\mathrm{TNFR}}(\chi) = \{(k\log p,\; \chi(p)^k\log p) : p \nmid q,\; k = 1, \dots, K\}$.
* **Twisted Dirichlet trace**: $Z_{\mathrm{TNFR}}(s, \chi) = \sum_{(\mu, w) \in \operatorname{Spec}_{\mathrm{TNFR}}(\chi)} w\, e^{-s\mu}$.

By direct expansion, $Z_{\mathrm{TNFR}}(s, \chi) \xrightarrow[K, n_{\text{primes}} \to \infty]{} -L'(s,\chi)/L(s,\chi)$ for $\operatorname{Re}(s) > 1$.  When the TNFR truncation and the classical truncation cover the same set of prime powers, the per-prime-power correspondence forces machine-precision agreement (analogue of the P12 unit-test invariant).

### §13undecies.3 Empirical Verification (May 2026 run)

`examples/59_dirichlet_l_function_demo.py` runs the canonical verification with $n_{\text{primes}} = 200$, $K = 12$, $n_{\max}^{\mathrm{classical}} = 100\,000$, across four canonical real characters and five complex spectral points with $\operatorname{Re}(s) \in \{2, 3, 5\}$:

| Character | Modulus $q$ | $\max\, \text{rel\_err}$ ($\operatorname{Re}(s)=5$) | $\max\, \text{rel\_err}$ ($\operatorname{Re}(s)=2$) |
|-----------|-------------|------------------------------------------------------|------------------------------------------------------|
| $\chi_0$ (principal) | 3 | $4.7 \times 10^{-12}$ | $2.1 \times 10^{-3}$ |
| $\chi$ real (Legendre $(n/3)$) | 3 | $6.2 \times 10^{-15}$ | $2.3 \times 10^{-5}$ |
| $\chi$ real (Dirichlet $\beta$) | 4 | $1.3 \times 10^{-12}$ | $2.2 \times 10^{-4}$ |
| $\chi$ real (Legendre $(n/5)$) | 5 | $2.9 \times 10^{-13}$ | $4.3 \times 10^{-5}$ |

The behaviour matches the P12 reference identically: at large $\operatorname{Re}(s)$ both truncations cover the same effective prime-power set and agree to machine precision; at $\operatorname{Re}(s) = 2$ the rate of decay of $p^{-ks}$ is slow enough that the prime-count truncation tail dominates and produces the observed $10^{-3}$–$10^{-5}$ residual.

### §13undecies.4 What P32 Extends

P32 generalises **only the P12 representation layer** (gap G5 superseded) from $\zeta(s)$ to every $L(s, \chi)$:

* The canonical TNFR-Riemann representation catalog now covers all Dirichlet L-functions, not only $\zeta$.
* The structural reading "each coprime prime is a TNFR node carrying χ-twisted REMESH echoes" is canonically the same for every $\chi$.
* The TNFR analogue of $-L'/L$ inherits the same Dirichlet-series structure, the same convergence boundary $\operatorname{Re}(s) > 1$, and the same per-prime-power matching invariant as P12.

### §13undecies.5 What P32 Does NOT Advance

P32 is a **structural extension**, not progress on the open arithmetic content of the program:

* It does **NOT** advance gap G4 (RH localisation on $\operatorname{Re}(s) = 1/2$).
* It does **NOT** advance the **Generalised Riemann Hypothesis (GRH)**.  Every Dirichlet L-function carries an arithmetic oscillatory residue
  $$S_\chi(T) = \tfrac{1}{\pi}\, \arg L(\tfrac{1}{2} + iT, \chi),$$
  the exact analogue of $S(T)$ for $\zeta$ documented in §13octies.  Bounding $S_\chi(T)$ is RH-equivalent in every L-function and inherits the same arithmetic obstruction as G4 for $\zeta$.
* It does **NOT** supply a Hamiltonian (P14 analogue) for general $L(s,\chi)$, an analytic continuation (P13 analogue), or an explicit-formula verification (P15 analogue).  Those are natural future extensions of the same structural pattern and would close the operational gaps G1$_\chi$, G2$_\chi$, G3$_\chi$ for each Dirichlet L-function — but not GRH.

### §13undecies.6 Cross-References

* §8: P12 prime-ladder construction (the template P32 generalises).
* §7.8: G5 supersession by P12+P13+P15 (the operational route P32 extends to characters).
* §13octies: assembled-argument audit for G4; the same audit applies, character by character, to GRH.
* `src/tnfr/riemann/dirichlet_l.py`: canonical implementation of P32.
* `examples/59_dirichlet_l_function_demo.py`: reproducible verification across four canonical real characters.

### §13undecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P32 | Status after P32 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally (P14, P13, P15) | Closed operationally |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, now generalised to all $L(s,\chi)$ at the P12 layer |
| Operational L-function gaps (G1$_\chi$, G2$_\chi$, G3$_\chi$) | Not addressed | G5$_\chi$ analogue closed at the P12 layer; G1$_\chi$/G2$_\chi$/G3$_\chi$ open (future work) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P32 extends the canonical TNFR representation catalog from a single L-function ($\zeta$) to the full Dirichlet family.  It does not close, nor narrow, any open arithmetic gap.

## §13duodecies. P33 — Analytic Continuation of χ-Twisted Prime-Ladder L-Series (Structural; Does NOT Advance G4 or GRH)

### §13duodecies.1 Motivation

P32 (§13undecies) provides the χ-twisted prime-ladder spectrum $\{(\mu_{p,k}, w_{p,k}^{(\chi)})\}$ reproducing the twisted von Mangoldt series

$$
Z^{(\chi)}_{\mathrm{TNFR}}(s) \;=\; -\frac{L'(s, \chi)}{L(s, \chi)} \;=\; \sum_{n \ge 1} \chi(n)\,\Lambda(n)\,n^{-s}
\qquad (\operatorname{Re}(s) > 1).
$$

This Dirichlet series is, by construction, only valid in the right half-plane $\operatorname{Re}(s) > 1$.  To expose the non-trivial zeros of $L(s, \chi)$ — which for non-principal primitive $\chi$ are *entire* objects living on the critical line — the χ-twisted prime ladder must be continued analytically to all of $\mathbb{C}$.

P33 is the structural analogue of P13 (§9) for general Dirichlet L-functions: the canonical continuation is obtained via `mpmath.dirichlet(s, [χ(0), …, χ(q-1)], derivative)` and the non-trivial zeros of $L(s, \chi)$ are recovered as **resonance poles** of $-L'(s,\chi)/L(s,\chi)$ on $\operatorname{Re}(s) = 1/2$.

### §13duodecies.2 Construction

For any Dirichlet character $\chi$ mod $q$:

1. **Continuation of $L(s, \chi)$**: `dirichlet_l_continued(chi, s, dps)` wraps `mp.dirichlet(s, chi_list)` with `chi_list = [mp.mpf(c) | mp.mpc(c) for c in chi.values]`, returning the unique meromorphic continuation to $\mathbb{C}$.
2. **Continuation of $-L'/L$**: `dirichlet_log_l_derivative_continued(chi, s, dps)` performs two `mp.dirichlet` calls (`derivative=0` and `derivative=1`) and returns $-L'(s,\chi)/L(s,\chi)$.  Raises `ValueError` whenever $|L(s,\chi)|$ is below the working precision (i.e., at a zero of $L$).
3. **Agreement certificate** (`verify_twisted_continuation_agreement`): compares the χ-twisted prime-ladder partial sum `tnfr_log_l_derivative(spectrum, s)` from P32 against the continuation evaluator on a list of $s$ with $\operatorname{Re}(s) > 1$, classifying the result as `excellent | good | poor` according to the worst per-point relative error.
4. **Critical-line scan** (`scan_critical_line_for_l_poles`): evaluates $|{-L'/L}|$ on $s = 1/2 + it$ for $t \in [t_{\min}, t_{\max}]$ and detects local-maximum spikes (resonance poles) using a sliding window proportional to the sample density.  The detection is reference-free; cross-checks against LMFDB tabulations are left to the caller.

### §13duodecies.3 Empirical Verification (May 2026 run)

Agreement on $\operatorname{Re}(s) > 1$ using `n_primes=400, max_power=14, dps=30` for the three canonical real characters of §13undecies:

| Character | Samples $s$ | Quality | max $|$rel err$|$ | max $|$abs err$|$ |
|---|---|---|---|---|
| $\chi_3$ (Legendre mod 3) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $1.10 \times 10^{-5}$ | $1.90 \times 10^{-6}$ |
| $\chi_4$ (Dirichlet $\beta$) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $1.60 \times 10^{-5}$ | $1.43 \times 10^{-6}$ |
| $\chi_5$ (Legendre mod 5) | $\{2, 2+i, 3, 3+2i, 5\}$ | `excellent` | $4.10 \times 10^{-6}$ | $1.10 \times 10^{-6}$ |

The residual is the standard P32 prime-truncation tail (same magnitude as the P12/P13 baseline for $\zeta$ at comparable truncation); it is **not** a defect of the continuation.

Critical-line scan against LMFDB-tabulated first zeros (`dps=20, 2001 samples on $t \in [5, 25]$, prominence threshold = 3.0`):

| Character | Detected peaks | LMFDB match | max $|$Δ$t|$ |
|---|---|---|---|
| $\chi_3$ | 6 (at $t \approx 8.04, 11.25, 15.70, 18.26, 20.46, 24.06$) | 6 / 6 | $6.7 \times 10^{-3}$ |
| $\chi_4$ | 7 (at $t \approx 6.02, 10.24, 12.99, 16.34, 18.29, 21.45, 23.28$) | 7 / 7 | $3.8 \times 10^{-3}$ |

All 13 detected resonance poles match the LMFDB tabulation to better than 0.01 in $t$ (limited by sample resolution $\Delta t = 0.01$); none miss, none extra.

### §13duodecies.4 What P33 Extends

P33 extends the **P13 representation layer** from $\zeta$ to every Dirichlet L-function:

* For $\zeta$: P13 continues the prime-ladder vM zeta to $\mathbb{C}$; non-trivial zeros appear as resonance poles on $\operatorname{Re}(s) = 1/2$.
* For $L(s, \chi)$: P33 does the same — continues the χ-twisted prime ladder of P32 to $\mathbb{C}$; non-trivial zeros of $L(s, \chi)$ appear as resonance poles on $\operatorname{Re}(s) = 1/2$.

This closes the **G5$_\chi$ / G2$_\chi$ analogue at the P13 layer**: the χ-twisted prime ladder is now a complete representation of $L(s, \chi)$ on the whole complex plane (subject to the same caveats as the classical continuation — branch cuts of the logarithmic derivative at the zeros).

### §13duodecies.5 What P33 Does NOT Advance

P33 is a **structural extension**, not progress on the open arithmetic content of the program:

* **G4 = RH for $\zeta$**: unchanged.  P33 does not touch $\zeta$.
* **GRH for $L(s, \chi)$**: unchanged.  P33 *uses* the existence and analyticity of the classical continuation; it does not derive the location of the zeros.  The detected resonance poles fall on $\operatorname{Re}(s) = 1/2$ because the LMFDB data they reproduce is itself empirical confirmation of GRH for the tested characters.
* **G1$_\chi$ (canonical Hamiltonian for χ-twisted prime ladder)**: open.  P33 does not construct a self-adjoint operator carrying the χ-twisted spectrum (the P14 analogue for L-functions remains future work — provisional label P34).
* **G3$_\chi$ (Weil–Guinand explicit formula for $L(s, \chi)$)**: open.  P33 does not verify a numerical explicit formula relating L-function zeros to the χ-twisted prime ladder (the P15 analogue — provisional label P35).

### §13duodecies.6 Cross-References

* §9: P13 analytic continuation of the prime-ladder vM zeta (the template P33 generalises).
* §13undecies: P32 χ-twisted prime ladder (the representation P33 continues).
* `src/tnfr/riemann/analytic_continuation_dirichlet.py`: canonical implementation of P33.
* `examples/60_dirichlet_l_continuation_demo.py`: demo verifying agreement on $\operatorname{Re}(s) > 1$ and critical-line zero detection for $\chi_3$ and $\chi_4$.

### §13duodecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P33 | Status after P33 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally | Closed operationally, unchanged |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, unchanged |
| G5$_\chi$ at P12 layer (P32) | Closed | Closed, unchanged |
| G2$_\chi$ / G5$_\chi$ at P13 layer | Open | **Closed operationally** by P33 |
| G1$_\chi$ (Hamiltonian for $L(s,\chi)$) | Open | Open (future P34) |
| G3$_\chi$ (Weil–Guinand for $L(s,\chi)$) | Open | Open (future P35) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P33 extends the canonical TNFR representation catalog one layer further — the χ-twisted prime ladder of P32 now lives on the whole complex plane.  It does not close, nor narrow, any open arithmetic gap.

## §13terdecies. P34 — Canonical Hamiltonian for the χ-Twisted Prime Ladder (Structural; Closes G1$_\chi$ at the P14 Layer; Does NOT Advance G4 or GRH)

### §13terdecies.1 Motivation

P14 (§10) supplies the canonical self-adjoint TNFR ``InternalHamiltonian`` on the prime-ladder graph whose decoupled spectrum is $\{k \log p\}$ and whose weighted spectral trace $\operatorname{Tr}(W \, e^{-s H_{\mathrm{freq}}})$ reproduces $-\zeta'(s)/\zeta(s)$ to machine precision. After P32 (the χ-twisted prime ladder representing $-L'(s,\chi)/L(s,\chi)$ on $\operatorname{Re}(s) > 1$) and P33 (its analytic continuation), the natural structural question — the explicit content of §13duodecies.5 — is whether the same canonical TNFR Hamiltonian construction admits a χ-twisted analogue for every Dirichlet character. **P34 supplies that analogue.** The construction does not advance GRH or G4; it closes gap **G1$_\chi$** *at the P14 layer* for every $L(s,\chi)$.

### §13terdecies.2 Construction

Let $\chi$ be a Dirichlet character of conductor $q$ and $K \ge 1$ a REMESH echo cut-off. Let $P_\chi = \{p \text{ prime}: \chi(p) \neq 0\} = \{p : p \nmid q\}$ (the canonical primes-coprime-to-$q$ filter introduced at P32).

1. **Graph**: $G_\chi$ is the disjoint union over $p \in P_\chi$ of the per-prime REMESH ladder $L_p$ ($K$ nodes $(p,1), \dots, (p,K)$ chained by REMESH edges). Per-node attributes: $\nu_f((p,k)) = k \log p$, all other TNFR state $\phi = 0$, $\mathrm{EPI} = 1$, $S_i = 1$, $\Delta \mathrm{NFR} = 0$.

2. **Hamiltonian**: $H_\chi$ is the canonical TNFR ``InternalHamiltonian`` on $G_\chi$ with internal-coherence strength $\alpha = 0$ and decoupled limit ($J_0 = 0$). By construction (§10) $H_\chi$ is real symmetric (hence self-adjoint) and $\operatorname{spec}(H_{\chi, \mathrm{freq}}) = \{k \log p : p \in P_\chi, \, 1 \le k \le K\}$ exactly.

3. **χ-twisted weight operator**: $W^{(\chi)}$ is the diagonal $|V(G_\chi)| \times |V(G_\chi)|$ matrix

   $$W^{(\chi)}_{(p,k),(p,k)} = \chi(p)^k \log p \in \mathbb{C}.$$

   For real characters $W^{(\chi)}$ is real-diagonal (Hermitian); for complex characters it is *normal but not Hermitian*, since the entries lie on the unit circle scaled by $\log p$. This is the canonical structural carrier of the χ-phase: $H_\chi$ stays real self-adjoint (so its eigenvectors form a real-orthonormal basis), and the complex content lives exclusively in $W^{(\chi)}$.

4. **χ-twisted weighted spectral trace**: For $s \in \mathbb{C}$,

   $$Z_{\mathrm{TNFR}}^{(\chi)}(s) := \operatorname{Tr}\bigl(W^{(\chi)} \, e^{-s H_{\chi, \mathrm{freq}}}\bigr) = \sum_{p \in P_\chi} \sum_{k=1}^{K} \chi(p)^k (\log p) \, p^{-ks}.$$

   This is exactly the P32 reference trace `tnfr_log_l_derivative`, which converges to $-L'(s,\chi)/L(s,\chi)$ as $K \to \infty$ on $\operatorname{Re}(s) > 1$ and admits the P33 continuation elsewhere.

### §13terdecies.3 Empirical Verification (May 2026 run)

`examples/61_dirichlet_l_hamiltonian_demo.py`, with $n_{\mathrm{primes}} = 20$, $K = 8$ (Hilbert dimension $N = 152$, $19$ active primes, $1$ excluded), $s$-values $\{2, 3, 2+i, 3+2i, 5, 10\}$:

| Character | $N$ | $n_{\mathrm{active}}$ | spectrum_max_abs_error | trace_max_rel_error | overall_ok |
|-----------|----:|----------------------:|-----------------------:|--------------------:|-----------:|
| $\chi_3$ (mod 3) | 152 | 19 | $0.000 \times 10^{0}$ | $3.241 \times 10^{-16}$ | **YES** |
| $\chi_4$ (mod 4) | 152 | 19 | $0.000 \times 10^{0}$ | $3.493 \times 10^{-16}$ | **YES** |
| $\chi_5$ (mod 5) | 152 | 19 | $0.000 \times 10^{0}$ | $2.313 \times 10^{-16}$ | **YES** |

The spectrum match is **exact** (zero floating-point error: $H_{\chi,\mathrm{freq}}$ is constructed with diagonal entries $\nu_f((p,k)) = k \log p$, hence its eigenvalues coincide bit-for-bit with the reference). The χ-twisted weighted trace matches the P32 reference at the machine-epsilon level for all tested $s$, including non-real $s$.

Step 3 of the demo also verifies the **triple agreement** P34 ≡ P32 (machine precision, by construction) ≡ P33 (mpmath, $O(p_{\max}^{-\operatorname{Re}(s)})$ truncation residual) on $\operatorname{Re}(s) > 1$ for $\chi_3$, including off-axis $s = 2+i$ and $s = 3+2i$.

### §13terdecies.4 What P34 Extends

* **Canonical operator catalog**: every Dirichlet $L(s,\chi)$ now has a TNFR-canonical self-adjoint operator that carries its prime data, exactly as $\zeta$ does at P14.
* **G1$_\chi$ at the P14 layer**: the obstruction "canonical Hamiltonian whose decoupled spectrum and χ-twisted weighted trace reproduce the P32 χ-twisted ladder data" is now **closed operationally** for every $\chi$.
* **Structural completeness of the L-function track**: after P32 (operator content), P33 (continuation), and P34 (Hamiltonian realisation), the χ-twisted ladder occupies the same structural status as the ζ ladder before P15.

### §13terdecies.5 What P34 Does NOT Advance

* **Generalised Riemann Hypothesis (GRH)**: no change. RH-equivalent localisation of poles on $\operatorname{Re}(s) = 1/2$ for $L(s,\chi)$ is the same arithmetic obstruction as G4 = RH for $\zeta$; P34 inherits the open status unchanged.
* **G4 = RH**: untouched. The P34 Hamiltonian is structurally identical to the P14 Hamiltonian on its prime-ladder block; the open content of Conjecture **T-HP** (§13septies) — existence of a canonical admissible spectral-rescaling operator $\mathcal{F}$ built only from the tetrad — is *not* addressed.
* **G3$_\chi$ (χ-twisted Weil–Guinand explicit formula)**: open. The χ-twisted analogue of P15 — the *explicit-formula* bridge linking the P33 zeros of $L(s,\chi)$ to the P34 Hamiltonian spectrum to machine precision — is the future **P35**.
* **No new analytic content**: P34 is a canonical operator-theoretic *re-presentation* of P32/P33 data; it does not introduce any analytic ingredient absent from those constructions.

### §13terdecies.6 Cross-References

* §10: P14 prime-ladder Hamiltonian (the canonical template P34 specialises).
* §13undecies: P32 χ-twisted prime ladder (the spectrum/weight data P34 represents).
* §13duodecies: P33 analytic continuation of $-L'(s,\chi)/L(s,\chi)$ (the off-$\operatorname{Re}(s) > 1$ extension).
* `src/tnfr/riemann/twisted_prime_ladder_hamiltonian.py`: canonical implementation of P34.
* `examples/61_dirichlet_l_hamiltonian_demo.py`: demo verifying spectrum-exact / trace-machine-precision reproduction for $\chi_3, \chi_4, \chi_5$ and triple agreement P34 ≡ P32 ≡ P33 on $\operatorname{Re}(s) > 1$.

### §13terdecies.7 Status Update for §19.2 Gap Balance

| Scope | Status before P34 | Status after P34 |
|-------|-------------------|------------------|
| Operational ζ gaps (G1, G2, G3) | Closed operationally | Closed operationally, unchanged |
| G4 = RH | OPEN (Conjecture T-HP) | OPEN, unchanged |
| G5 (ζ representation) | Superseded by P12+P13+P15 | Superseded, unchanged |
| G5$_\chi$ at P12 layer (P32) | Closed | Closed, unchanged |
| G2$_\chi$ / G5$_\chi$ at P13 layer (P33) | Closed operationally | Closed operationally, unchanged |
| **G1$_\chi$ (Hamiltonian for $L(s,\chi)$)** | Open | **Closed operationally** by P34 |
| G3$_\chi$ (Weil–Guinand for $L(s,\chi)$) | Open | Open (future P35) |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P34 extends the canonical TNFR operator catalog one layer further — every Dirichlet $L(s,\chi)$ now has a canonical self-adjoint TNFR Hamiltonian realising its prime data, exactly as $\zeta$ does since P14. It does not close, nor narrow, any open arithmetic gap (G4, GRH, G3$_\chi$).

## §13quaterdecies. P35 — χ-Twisted Weil–Guinand Explicit Formula (Structural; Closes G3$_\chi$ Operationally for Primitive Real χ; Does NOT Advance G4 or GRH)

### §13quaterdecies.1 Motivation

P15 (§11) established the Weil–Guinand explicit formula for $\zeta$ as a TNFR-native identity: the zero side $\sum_\gamma h(\gamma)$ equals the sum of an Archimedean digamma integral, a constant term, and a prime side computed as the diagonal projection of the P14 weight operator $W$ in its eigenbasis. After P32 (χ-twisted prime ladder), P33 (analytic continuation of the corresponding TNFR vM zeta), and P34 (canonical Hamiltonian for the χ-twisted ladder), the natural structural question is the χ-twisted analogue of the explicit formula. **P35 supplies it for every primitive real Dirichlet character.**  This closes gap **G3$_\chi$** operationally for $\chi \in \{\chi_3, \chi_4, \chi_5, \ldots\}$ (real, non-principal) and does not advance G4 = RH or GRH.

### §13quaterdecies.2 Construction

For a primitive real non-principal Dirichlet character $\chi$ with conductor $q$ and parity $a = (1-\chi(-1))/2 \in \{0,1\}$, and Gaussian test pair $h(t)=e^{-t^2/(2\sigma^2)}$, $g(u)=(\sigma/\sqrt{2\pi})\,e^{-\sigma^2 u^2/2}$, the χ-twisted Weil–Guinand explicit formula is

$$
\sum_\gamma h(\gamma)
\;=\;
\underbrace{g(0)\,\log(q/\pi)}_{\text{constant term}}
\;+\;
\underbrace{\frac{1}{2\pi}\!\int_{-\infty}^{\infty}\! h(t)\,\Re\,\psi\!\left(\tfrac14+\tfrac{a}{2}+\tfrac{it}{2}\right)\,dt}_{\text{archimedean side}}
\;-\;
\underbrace{2\,\Re\sum_{n\ge1}\frac{\chi(n)\,\Lambda(n)}{\sqrt n}\,g(\log n)}_{\text{prime side}}.
$$

The two non-trivial reductions to ζ are immediate: for the trivial character ($q=1$, $a=0$) the constant becomes $-g(0)\log\pi$, the digamma factor collapses to $\psi(1/4+it/2)$, and the prime side becomes the unweighted P15 sum.

* **Zero side** — Hardy-Z bisection on $Z_\chi(t) = e^{i\theta_\chi(t)} L(\tfrac12+it,\chi)$ (built on P33's mpmath-grade continuation), enumerating positive imaginary parts $\gamma$ on $\operatorname{Re}(s) = 1/2$.  Real $\chi$ ⇒ zeros come in conjugate pairs ⇒ $\sum_\gamma h(\gamma) = 2\sum_{\gamma>0} h(\gamma)$.
* **Prime side** — diagonal projection of the χ-twisted weight operator $W^{(\chi)}$ from the canonical P34 Hamiltonian, in its eigenbasis (same einsum idiom as P15).
* **Archimedean side** — direct numerical quadrature of the digamma factor (mpmath, $\mathrm{dps}=30$).

### §13quaterdecies.3 Empirical Verification

| χ | q | a | σ | n zeros | residual | rel. residual | verified |
|---|---|---|---|---|---|---|---|
| χ₃ | 3 | 1 | 2.0 |  5 | −7.46 × 10⁻¹⁷ | 1.20 × 10⁻¹³ | ✓ |
| χ₃ | 3 | 1 | 2.5 |  8 | +2.03 × 10⁻¹⁶ | 1.77 × 10⁻¹⁴ | ✓ |
| χ₃ | 3 | 1 | 3.0 | 11 | +2.36 × 10⁻¹⁶ | 4.15 × 10⁻¹⁵ | ✓ |
| χ₄ | 4 | 1 | 2.0 |  7 | −9.48 × 10⁻¹⁵ | 4.40 × 10⁻¹³ | ✓ |
| χ₄ | 4 | 1 | 2.5 | 10 | −3.10 × 10⁻¹⁴ | 2.81 × 10⁻¹³ | ✓ |
| χ₄ | 4 | 1 | 3.0 | 12 | −5.16 × 10⁻¹⁴ | 1.89 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 2.0 |  7 | −2.00 × 10⁻¹⁵ | 2.50 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 2.5 | 11 | −7.97 × 10⁻¹⁵ | 1.35 × 10⁻¹³ | ✓ |
| χ₅ | 5 | 0 | 3.0 | 14 | −1.56 × 10⁻¹⁴ | 8.60 × 10⁻¹⁴ | ✓ |

All nine $(\chi, \sigma)$ pairs verify the identity to machine precision (relative residual $\le 4.4 \times 10^{-13}$), well inside the declared $10^{-2}$ tolerance.

### §13quaterdecies.4 What P35 Extends

* **Operational closure of G3$_\chi$ for primitive real χ**: both sides of the χ-twisted Weil–Guinand identity now have a canonical TNFR realisation that agrees to machine precision.
* **Symmetric completion of the L-function track**: P32 (operator content) → P33 (continuation) → P34 (Hamiltonian) → **P35 (explicit formula)** now occupies the same structural status as P12 → P13 → P14 → P15 for ζ.
* **Reuse of P34 Hamiltonian**: the prime side is the *exact same* einsum idiom as P15; no new operator is introduced.

### §13quaterdecies.5 What P35 Does NOT Advance

* **Generalised Riemann Hypothesis (GRH)**: untouched.  Zero localisation on $\operatorname{Re}(s) = 1/2$ for $L(s,\chi)$ is **assumed** in P35 (Hardy-Z bisection starts from the critical line); proving every L-zero lies there is the χ-twisted analogue of gap **G4 = RH** and is the same arithmetic obstruction.
* **G4 = RH**: structurally identical to the ζ case; Conjecture **T-HP** (§13septies) and its extensions remain open.
* **Complex χ**: P35 currently supports only **primitive real** characters.  Extension to complex χ requires a Hermitisation of $W^{(\chi)}$ that is intentionally deferred.
* **No new analytic content beyond P33**: P35 packages P33 zeros and P34 Hamiltonian into a single explicit-formula certificate; it does not introduce any new analytic ingredient.

### §13quaterdecies.6 Cross-References

* §11: P15 Weil–Guinand for ζ (canonical template P35 specialises).
* §13undecies: P32 χ-twisted prime ladder.
* §13duodecies: P33 χ-twisted analytic continuation.
* §13terdecies: P34 χ-twisted Hamiltonian (prime side of P35).
* `src/tnfr/riemann/twisted_weil_explicit_formula.py`: canonical implementation of P35.
* `examples/62_dirichlet_weil_explicit_formula_demo.py`: demo verifying nine $(\chi, \sigma)$ pairs to machine precision.

### §13quaterdecies.7 Gap Balance

| Scope | Status before P35 | Status after P35 |
|-------|---|---|
| G3 (Weil–Guinand for ζ, P15) | Closed operationally | Closed operationally, unchanged |
| **G3$_\chi$ (Weil–Guinand for $L(s,\chi)$, primitive real χ)** | Open (future P35) | **Closed operationally** by P35 |
| G3$_\chi$ for complex χ | Open | Open (future increment) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for $\chi \neq \chi_0$) | OPEN | OPEN, unchanged |

**Net effect**: P35 closes the explicit-formula gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P34, the structural status of $L(s,\chi)$ for real $\chi$ now matches the ζ track up through P15.  The arithmetic obstruction (zero localisation on $\operatorname{Re}(s)=1/2$) is unchanged.

## §13quinquiesdecies. P36 — χ-Twisted Li–Keiper Positivity Criterion (Structural Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13quinquiesdecies.1 Motivation

P16 (§12) supplies the canonical TNFR-native finite diagnostic surface for RH via Li–Keiper coefficients $\lambda_n$ computed from non-trivial zeros of $\zeta(s)$.  The L-function track now reaches the same level: P35 (§13quaterdecies) supplies a complete Hardy-Z zero enumerator for every primitive real Dirichlet $L(s,\chi)$, and Lagarias 2007 generalises Li 1997 to L-functions.  P36 packages these ingredients into a structural GRH$_\chi$-equivalent diagnostic — the L-function analogue of P16.

### §13quinquiesdecies.2 Construction

For a primitive real Dirichlet character $\chi$ with non-trivial zeros $\rho_k = 1/2 + i\gamma_k$ of $L(s,\chi)$ on the critical line, define the **χ-twisted Li–Keiper coefficients**

$$
\lambda_n(\chi) \;=\; \sum_{k} 2\,\operatorname{Re}\!\Big[1 - \big(1 - 1/\rho_k\big)^n\Big],\qquad n \ge 1.
$$

The sum runs over all non-trivial zeros (paired with their complex conjugates via the $2\operatorname{Re}[\cdot]$ factor).  By Lagarias 2007 (generalisation of Li 1997):

$$
\boxed{\;\text{GRH for } L(s,\chi) \iff \lambda_n(\chi) > 0 \text{ for every } n \ge 1.\;}
$$

P36 computes $\lambda_n(\chi)$ for $n = 1, \dots, n_{\max}$ from the finite truncation $\{\gamma_k : 0 < \gamma_k < t_{\max}\}$ supplied by the P35 enumerator (`find_dirichlet_l_zeros`).  The sum-over-zeros formula is **L-function agnostic**, so the canonical P16 routine `li_coefficients_from_zeros` is reused unchanged at mpmath precision $\text{dps} = 50$.

### §13quinquiesdecies.3 Empirical Verification

Positivity of $\lambda_n(\chi)$ verified for the three primitive real characters of small modulus across $n_{\max} \in \{20, 30, 50\}$ with $t_{\max} = 80$:

| Character | $q$ | parity $a$ | $\#$ zeros used | $\min_n \lambda_n(\chi)$ | $\lambda_n > 0$ for $n \le 50$? |
|-----------|-----|-----------|-----------------|--------------------------|-------------------------------|
| $\chi_3$  | 3   | 1         | 34              | $+4.741 \times 10^{-2}$  | yes                           |
| $\chi_4$  | 4   | 1         | 37              | $+6.791 \times 10^{-2}$  | yes                           |
| $\chi_5$  | 5   | 0         | 40              | $+6.802 \times 10^{-2}$  | yes                           |

(Reproduced by `examples/63_dirichlet_li_keiper_demo.py`.)

### §13quinquiesdecies.4 What P36 Extends

* **P16 to L-functions**: P16 is the canonical Li–Keiper diagnostic for $\zeta$; P36 is its structural analogue for $L(s,\chi)$ at every primitive real $\chi$.  Together with P32–P35, the structural TNFR-Riemann program now matches the ζ track all the way through the diagnostic layer.
* **Numerical witness for GRH$_\chi$**: every positivity row above is a falsifiable finite witness; a single $\lambda_n(\chi) \le 0$ would disprove GRH for the corresponding $L(s,\chi)$.

### §13quinquiesdecies.5 What P36 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite check of $\lambda_n > 0$ for $n \le n_{\max}$ is **necessary but not sufficient**.  Rigorous bounds on the truncation tail are required to upgrade the finite check to a proof; P36 does not supply them.  Consistent with Bombieri–Lagarias 1999 and Lagarias 2007.
* **G4 = RH**: structurally identical to P16; the arithmetic obstruction is untouched.  The zeros are *assumed* to lie on $\operatorname{Re}(s) = 1/2$ via the Hardy-Z bisection on $Z_\chi(t)$ used by P35.
* **Complex χ**: P36 inherits the primitive-real restriction from P32–P35.

### §13quinquiesdecies.6 Cross-References

* §12: P16 Li–Keiper criterion for ζ (canonical template).
* §13quaterdecies: P35 χ-twisted Weil–Guinand explicit formula (supplies the zero enumerator).
* §13septies: Conjecture T-HP (unchanged by P36).
* `src/tnfr/riemann/twisted_li_keiper.py`: canonical P36 implementation.
* `examples/63_dirichlet_li_keiper_demo.py`: demo with full positivity sweep.

### §13quinquiesdecies.7 Gap Balance

| Scope | Status before P36 | Status after P36 |
|-------|-------------------|------------------|
| P16 diagnostic for ζ | Available (P16) | Available, unchanged |
| **Li–Keiper diagnostic for $L(s,\chi)$, primitive real χ** | Open (future P36) | **Available** (TNFR-native finite witness) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite check is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P36 closes the diagnostic-layer gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P35, every milestone reachable on the ζ track up through P16 now has a structural analogue on the primitive-real L-function track.  The arithmetic obstruction remains the same.

## §13sexiesdecies. P37 — χ-Twisted Weil–TNFR Positivity Bridge (Structural Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13sexiesdecies.1 Motivation

P17 (§14) supplies the canonical TNFR-native Weil-positivity bridge for $\zeta$: Weil's RH-equivalent positivity functional $W[f] = \sum_\gamma \hat f(\gamma) \ge 0$ is transported onto the TNFR Lyapunov functional $E_{\mathrm{TNFR}}$ via the P14 prime-ladder Hamiltonian.  Bombieri 2000 generalises Weil's criterion to every primitive Dirichlet $L(s,\chi)$, so the same structural transport exists on the L-function track once P34 (canonical χ-twisted Hamiltonian) and P35 (canonical χ-twisted explicit formula) are in place.  P37 packages these ingredients into a GRH$_\chi$-equivalent diagnostic — the L-function analogue of P17.

### §13sexiesdecies.2 Construction

For a fixed primitive real Dirichlet character $\chi$ of conductor $q$, parity $a \in \{0, 1\}$, and Gaussian width $\sigma > 0$, let
$$h_\sigma(t) = e^{-t^2 / (2\sigma^2)}, \qquad \hat h_\sigma(\xi) = \sigma \sqrt{2\pi}\, e^{-\sigma^2 \xi^2 / 2}.$$

The χ-twisted Weil positivity functional is
$$W_\chi[\sigma] := 2 \sum_{\gamma > 0} h_\sigma(\gamma), \qquad \gamma \in \mathrm{Im}\{\rho : L(\tfrac12 + i\rho, \chi) = 0\}.$$
P37 computes $W_\chi[\sigma]$ two ways:

1. **Zero side** (P35 enumerator): exact Hardy-Z bisection via `twisted_weil_zero_side` truncated at $t_{\max} = 12\sigma$ (canonical default).
2. **Explicit-formula side** (P34 Hamiltonian): the χ-twisted Weil–Guinand identity
    $$W_\chi[\sigma] \stackrel{!}{=} g(0)\log\!\frac{q}{\pi} + I_{\infty}^{\chi}(\sigma) + P_{\chi}(\sigma),$$
    where $g$ is the test function in the cosine-transform convention used by P35, $I_{\infty}^{\chi}$ is the archimedean integral (`twisted_weil_archimedean_integral`, parity-dependent via the $\psi$-shift) and $P_{\chi}$ is the prime side **evaluated on the P34 χ-twisted prime-ladder Hamiltonian** (`twisted_weil_prime_side_from_hamiltonian`).  The consistency residual $|W_{\mathrm{zero}} - W_{\mathrm{XF}}|$ measures the joint self-consistency of P34+P35.

Positivity is verified as $W_\chi[\sigma] \ge 0$.  In parallel, the canonical TNFR test state on the P34 graph is defined by `build_twisted_structural_test_state(bundle, sigma)`: for each node $(p, k)$ with structural frequency $\nu_f = k\log p$, set
$$\Delta\mathrm{NFR}_{(p,k)} = \mathrm{EPI}_{(p,k)} = h_\sigma(k\log p), \qquad \phi_{(p,k)} = \min(h_\sigma(k\log p), \pi),$$
and the TNFR Lyapunov energy of this state is
$$E_{\mathrm{TNFR}}^\chi[\sigma] := \tfrac12 \sum_i \bigl(\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2\bigr)$$
via the canonical `compute_energy_functional` (single source of truth from `tnfr.physics.conservation`, reused unchanged from P17).  The χ-twisted **TNFR bridge ratio** is
$$\boxed{\;\alpha_\chi(\sigma) := \frac{W_\chi[\sigma]}{E_{\mathrm{TNFR}}^\chi[\sigma]}.\;}$$

### §13sexiesdecies.3 Empirical Verification

Configuration: $N_{\mathrm{primes}} = 25$, $k_{\max} = 6$, decoupled spectrum (coupling = 0), $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  Demo: `examples/64_twisted_weil_positivity_demo.py`.

For χ$_3$ the consistency residual is $|W_{\mathrm{zero}} - W_{\mathrm{XF}}| \le 6.1 \times 10^{-6}$ at $\sigma = 1.0$ and $\le 2.4 \times 10^{-16}$ for $\sigma \in \{2.0, 2.5, 3.0\}$ (machine precision once enough zeros enter the Gaussian window).  Aggregate verdicts:

| Character | $q$ | parity $a$ | $W_\chi \ge 0$ all σ? | $\alpha_\chi > 0$ all σ? | $\alpha_{\min}$ | $\alpha_{\max}$ | Verdict |
|-----------|----:|:----------:|:---------------------:|:------------------------:|----------------:|----------------:|---------|
| χ$_3$     | 3   | 1 (odd)    | YES                   | YES                      | $1.27\times 10^{-14}$ | $7.39\times 10^{-3}$ | PASS |
| χ$_4$     | 4   | 1 (odd)    | YES                   | YES                      | $2.71\times 10^{-8}$  | $3.77\times 10^{-2}$ | PASS |
| χ$_5$     | 5   | 0 (even)   | YES                   | YES                      | $2.62\times 10^{-10}$ | $2.32\times 10^{-2}$ | PASS |

All three primitive real characters pass both the Weil positivity check and the structural bridge check across the entire Gaussian grid.  $\alpha_{\min}$ at small σ collapses toward machine precision because $W_\chi[\sigma] \to 0$ (no zeros enter the Gaussian window when $\sigma$ is smaller than the imaginary part of the lowest zero) while $E_{\mathrm{TNFR}}^\chi$ stays $\mathcal{O}(1)$; the diagnostic interpretation is that the lower bound becomes vacuous (not violated) in that regime.

### §13sexiesdecies.4 What P37 Extends

* **P17 to L-functions**: P17 is the canonical Weil-TNFR positivity bridge for $\zeta$ (GRH-equivalent diagnostic via $W \ge 0$); P37 is its structural analogue for $L(s,\chi)$ at every primitive real $\chi$.  The TNFR Lyapunov target $E_{\mathrm{TNFR}}$ is reused unchanged; only the zero source (P35) and the prime side (P34) are χ-twisted.

* **L-function track parity with the ζ track**: combined with P32–P36, every milestone reachable on the ζ track up through P17 now has a structural analogue on the primitive-real L-function track.

### §13sexiesdecies.5 What P37 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite Gaussian grid cannot exhaust the admissible family that makes Weil positivity equivalent to GRH$_\chi$ (Bombieri 2000).  Numerical $W_\chi[\sigma] \ge 0$ is consistent with GRH$_\chi$ but is **not** a proof.

* **G4 = RH**: P37 is on the L-function track and does not bear on the untwisted Riemann hypothesis.

* **Complex χ**: P37 inherits the primitive-real restriction from P32–P35 (the L-function track stays real until the complex-χ extension is shipped).

* **Canonicity of the structural test state**: `build_twisted_structural_test_state` is one canonical mapping of $h_\sigma$ to the P34 graph; the bridge ratio $\alpha_\chi(\sigma)$ is specific to this mapping.  Exhaustively sweeping admissible structural test states is a future milestone (parallel to P18–P21 on the ζ track).

### §13sexiesdecies.6 Cross-References

* §14 (P17): untwisted Weil–TNFR positivity bridge for $\zeta$ (the construction P37 imitates).
* §10 (P14) and §13nonies (P30): canonical TNFR Hamiltonian and admissible-rescaling building blocks reused via P34.
* §13quaterdecies (P35): χ-twisted Weil–Guinand explicit formula (zero side + RHS).
* §13quinquiesdecies (P36): χ-twisted Li–Keiper diagnostic (complementary GRH$_\chi$-equivalent surface).
* `src/tnfr/riemann/twisted_weil_positivity.py`: canonical P37 implementation.
* `examples/64_twisted_weil_positivity_demo.py`: demo with the full χ$_3$/χ$_4$/χ$_5$ sweep.

### §13sexiesdecies.7 Gap Balance

| Scope | Status before P37 | Status after P37 |
|-------|-------------------|------------------|
| P17 Weil bridge for ζ | Available (P17) | Available, unchanged |
| **Weil–TNFR bridge for $L(s,\chi)$, primitive real χ** | Open (future P37) | **Available** (TNFR-native finite diagnostic) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite Gaussian grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P37 closes the Weil-positivity-bridge gap on the L-function track for every primitive real Dirichlet character.  The L-function track now structurally matches the ζ track all the way through P17.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## 14. Weil–TNFR Positivity Bridge (P17)

### 14.1 Motivation

The §19.2 balance leaves a single open obstruction: **G4 = RH itself**.
P12–P16 close the *operational* gaps (Hamiltonian, analytic continuation,
explicit formula, Λ-series reproduction, RH-equivalent positivity
diagnostic), but none of them forces resonance poles onto the critical
line. P17 opens a TNFR-native attack surface on G4 by **transporting
Weil's RH-equivalent positivity criterion onto the canonical TNFR
Lyapunov functional**, using P14 as the bridge object.

### 14.2 Mathematical Setup

For an admissible even test function $f \in \mathcal{H}$ with Fourier
transform $\hat f$, Weil's positivity functional is

$$
W[f] \;=\; \sum_{\gamma} \hat f(\gamma)
\;=\; \underbrace{\hat f(\tfrac{i}{2}) + \hat f(-\tfrac{i}{2})}_{\text{pole side}}
\;-\; \underbrace{f(0)\,\log\pi
+ \tfrac{1}{2\pi}\!\!\int\!\hat f(t)\,\psi_{\mathbb{R}}(t)\,dt}_{\text{archimedean side}}
\;-\; \underbrace{\sum_p\sum_{k\ge 1}
\tfrac{\log p}{p^{k/2}}\,(f(k\log p)+f(-k\log p))}_{\text{prime side}},
$$

(Weil–Guinand identity; see §11). Weil's theorem: **RH $\Leftrightarrow$
$W[f] \ge 0$ for every $f$ in an admissible class**. We choose the
Gaussian family $h_\sigma(t) = e^{-t^2/(2\sigma^2)}$ already canonicalised
in P15 (`gaussian_test_function`).

### 14.3 TNFR Structural Mapping

Given the P14 prime-ladder bundle with nodes $(p,k)$ and
$\nu_f(p,k) = k\log p$, define the **canonical structural test state**

$$
\Delta\mathrm{NFR}(p,k) \;=\; h_\sigma(k\log p),
\qquad
\phi(p,k) \;=\; \mathrm{wrap}_\pi\!\bigl(h_\sigma(k\log p)\bigr),
\qquad
\mathrm{EPI}(p,k) \;=\; h_\sigma(k\log p),
$$

inheriting $\nu_f$ from P14. The canonical TNFR Lyapunov energy of this
state, computed via `tnfr.physics.conservation.compute_energy_functional`,
is denoted $E_{\mathrm{TNFR}}[\sigma]$ (it is automatically
$\ge 0$ by the Structural Conservation Theorem).

The bridge ratio is

$$
\alpha(\sigma) \;=\; \frac{W[h_\sigma]}{E_{\mathrm{TNFR}}[\sigma]}.
$$

**Working hypothesis (TNFR-native witness for RH)**: if $\alpha(\sigma) > 0$
holds across a dense admissible family of $\sigma$, then Weil positivity
holds across that family, hence (by Weil's equivalence) RH holds.

### 14.4 Implementation

Module: [`src/tnfr/riemann/weil_positivity.py`](../src/tnfr/riemann/weil_positivity.py).

Public API exported by `tnfr.riemann`:

* `WeilPositivityCertificate(sigma, weil_functional_zero_side,
  weil_functional_explicit_formula, explicit_formula_residual,
  n_zeros_used, positive)` — single-$\sigma$ certificate computing
  $W[h_\sigma]$ *twice* (zero side via classical zeros, explicit-formula
  side via P14) and reporting their consistency residual.
* `WeilTNFRBridgeCertificate(sigmas, weil_functional,
  tnfr_lyapunov_energy, alpha, weil_positive, bridge_positive, …)` —
  grid certificate over a chosen $\sigma$ family.
* `build_structural_test_state(bundle, sigma)`,
  `tnfr_lyapunov_of_test_state(bundle, sigma)` — explicit access to
  the canonical mapping and its Lyapunov energy.
* `verify_weil_positivity(bundle, *, sigma, n_zeros, …)`,
  `verify_weil_tnfr_bridge(bundle, sigmas, *, n_zeros, …)` — top-level
  entry points.

Reuses (without duplication): `weil_zero_side`, `weil_pole_side`,
`weil_archimedean_integral`, `weil_prime_side_from_hamiltonian`,
`gaussian_test_function` from P15; `compute_energy_functional` from
the canonical conservation module.

### 14.5 Numerical Results (May 2026 run)

Setup: `build_prime_ladder_hamiltonian(n_primes=20, max_power=6)`
(Hilbert dimension 120), 60 classical zeros, Gaussian-width grid
$\sigma \in \{1.0, 1.5, 2.0, 3.0, 5.0, 8.0\}$.

| $\sigma$ | $W[\sigma]$ | $E_{\mathrm{TNFR}}[\sigma]$ | $\alpha(\sigma)$ | $W \ge 0$ | $\alpha > 0$ |
|---:|---:|---:|---:|:---:|:---:|
| 1.0 | $+8.26\!\times\!10^{-44}$ | $+2.145$ | $+3.85\!\times\!10^{-44}$ | ✓ | ✓ |
| 1.5 | $+1.05\!\times\!10^{-19}$ | $+2.845$ | $+3.67\!\times\!10^{-20}$ | ✓ | ✓ |
| 2.0 | $+2.85\!\times\!10^{-11}$ | $+4.157$ | $+6.86\!\times\!10^{-12}$ | ✓ | ✓ |
| 3.0 | $+3.02\!\times\!10^{-5}$  | $+7.171$ | $+4.22\!\times\!10^{-6}$  | ✓ | ✓ |
| 5.0 | $+3.71\!\times\!10^{-2}$  | $+6.762$ | $+5.48\!\times\!10^{-3}$  | ✓ | ✓ |
| 8.0 | $+5.00\!\times\!10^{-1}$  | $+4.041$ | $+1.24\!\times\!10^{-1}$  | ✓ | ✓ |

Consistency between the zero side and the explicit-formula side at
$\sigma = 2$: residual $\approx 9.87 \times 10^{-17}$ (machine precision,
matching the P15 audit). Weil positivity and the TNFR bridge both hold
across the tested grid, with $\alpha_{\min} \approx 3.85 \times 10^{-44}$
(localised at small $\sigma$, where $W$ is exponentially small).

### 14.6 Status — Honest Reading

* P17 **does not prove RH**. The structural mapping
  $h_\sigma \mapsto (\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$ is canonical
  but not unique; promoting the numerical $\alpha(\sigma) > 0$ result to
  a theorem on a dense admissible class would require:
  1. proving canonicity (or uniqueness up to gauge) of the mapping,
  2. proving an analytic lower bound $\alpha(\sigma) \ge c(\sigma) > 0$
     on a dense $\sigma$-class (currently only computed pointwise),
  3. closing the family-completeness clause of Weil's theorem.
* What P17 **does** deliver: a *TNFR-native, RH-equivalent positivity
  diagnostic* that ties classical Weil positivity to the canonical
  Lyapunov functional of the Structural Conservation Theorem. A future
  numerical counter-example ($\alpha(\sigma_*) < 0$) would disprove
  the bridge as currently formulated (not RH itself, which would
  require $W[h_{\sigma_*}] < 0$).
* In the §19.2 ledger, G4 remains **OPEN**, but the attack surface is
  now made explicit: instead of an unspecified "Hilbert–Pólya
  realisation", the missing structural argument is **lower-boundedness
  of $\alpha(\sigma)$ on a dense admissible class** under the canonical
  TNFR mapping. This is a concrete, testable target for future work.

### 14.7 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\46_weil_tnfr_positivity_demo.py
```



## 15. Admissibility & Gauge Sweep of α(σ) (P18)

### 15.1 Motivation

Section 14.6 identified the **canonical-mapping ambiguity** as the
sharpest analytic weakness of the P17 bridge: encoding
$h_\sigma \mapsto (\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$ on the
P14 graph is canonical but not unique, and lower-boundedness
$\alpha(\sigma) \ge c > 0$ has only been verified pointwise on a
six-point Gaussian-width grid under one mapping. P18 stress-tests the
bridge along both axes:

* **Admissibility axis**: dense, log-spaced $\sigma$-grid covering both
  the exponentially-small regime ($\sigma \lesssim 1$, where $W$ falls
  below $10^{-100}$) and the classical regime ($\sigma \sim 10$).
* **Gauge axis**: a family of six structural mappings that activate
  different sectors of the Lyapunov functional (the gauge-invariant
  Weil functional $W[\sigma]$ is reused once per $\sigma$).

### 15.2 Gauge Family

For each prime-ladder node $(p,k)$ let $h = h_\sigma(k\log p)$. The
following gauges $h \mapsto (\Delta\mathrm{NFR},\ \phi,\ \mathrm{EPI})$
are probed by default (see `DEFAULT_GAUGES` in
`src/tnfr/riemann/alpha_sweep.py`):

| Gauge | $(\Delta\mathrm{NFR},\ \phi,\ \mathrm{EPI})$ | Activates |
|---|---|---|
| `canonical`          | $(h,\ h,\ h)$        | pressure + phase + EPI |
| `dnfr_only`          | $(h,\ 0,\ 1)$        | only $\Phi_s$ (via pressure)            |
| `phase_only`         | $(0,\ h,\ 1)$        | only phase gradient / curvature         |
| `epi_only`           | $(0,\ 0,\ h)$        | only EPI sector                         |
| `dnfr_phase`         | $(h,\ h,\ 1)$        | pressure + phase, fixed EPI             |
| `pressure_amplified` | $(2h,\ h,\ h)$       | scaled pressure, canonical phase/EPI    |

The phase channel is clipped to $[-\pi, \pi]$ via the standard TNFR
wrap convention; $\nu_f$ is inherited unchanged from P14.

### 15.3 Numerical Results (May 2026 run)

Setup: `build_prime_ladder_hamiltonian(n_primes=18, max_power=5)`
(Hilbert dimension 90), 50 classical zeros, $\sigma$-grid log-spaced
on $[0.5, 12]$ ($n_\sigma = 12$), six gauges from §15.2 (72 cells
total).

Outcome: **$\alpha(\sigma; g) > 0$ across the full $6 \times 12$
table.** Tightest entry $\alpha_{\min} = 1.37 \times 10^{-173}$
at $\sigma = 0.5$, $g = $ `canonical`. Maximum
$\alpha_{\max} = 1.06 \times 10^{1}$ at $\sigma = 12$, $g = $
`dnfr_only`. $W[\sigma] \ge 0$ on every grid point.

Selected $\alpha$ values (full table in
`examples/47_alpha_sweep_demo.py` output):

| $\sigma$ | `canonical` | `dnfr_only` | `epi_only` |
|---:|---:|---:|---:|
| 0.50  | $1.37 \times 10^{-173}$ | $3.41 \times 10^{-173}$ | $3.41 \times 10^{-173}$ |
| 1.59  | $6.41 \times 10^{-18}$  | $7.34 \times 10^{-17}$  | $7.34 \times 10^{-17}$  |
| 2.83  | $1.43 \times 10^{-6}$   | $4.49 \times 10^{-5}$   | $4.49 \times 10^{-5}$   |
| 5.04  | $8.15 \times 10^{-3}$   | $2.33 \times 10^{-1}$   | $2.33 \times 10^{-1}$   |
| 12.00 | $1.21$                  | $1.06 \times 10^{1}$    | $1.06 \times 10^{1}$    |

### 15.4 Lyapunov Sector Collapse

A non-trivial empirical observation: the six gauges yield exactly
**two distinct Lyapunov energy curves**.

* **Phase-active gauges** (`canonical`, `phase_only`, `dnfr_phase`,
  `pressure_amplified`): $E_{\mathrm{TNFR}}[\sigma]$ peaks at
  $\approx 6.0$ near $\sigma \approx 3.8$, decaying on both sides.
* **Phase-inactive gauges** (`dnfr_only`, `epi_only`):
  $E_{\mathrm{TNFR}}[\sigma] \equiv 0.1709$ — flat in $\sigma$.

Two structural readings:

1. **Phase dominance**. On the P14 prime-ladder topology, the
   geometric sector of the Lyapunov functional (driven by
   $|\nabla\phi|^2 + K_\phi^2$) dominates the potential sector (driven
   by $\Phi_s^2$) once the phase channel is excited; the magnitude of
   the pressure boost in `pressure_amplified` is invisible against
   the phase contribution at the tested scale.
2. **Gauge orbit structure**. The six probed gauges collapse to two
   $E$-orbits, so the 72-cell table effectively samples 24 independent
   $\alpha$-values. This sharpens what "robustness under canonical
   ambiguity" actually establishes — robustness within each of the two
   phase-on/phase-off orbits, plus persistence of $\alpha > 0$ across
   the orbit jump.

### 15.5 Status — Honest Reading

* P18 **does not prove RH** and does not change the G4 verdict.
  $\alpha > 0$ holds with margin $\gtrsim 10^{-173}$ at the worst cell;
  this is exponentially small (driven by $W[\sigma]$ itself, not by
  the structural mapping), as expected from the Gaussian decay.
* What P18 **does** deliver: a quantitative robustness statement for
  the P17 bridge — across two structurally different Lyapunov orbits
  and twelve admissibility scales spanning 174 orders of magnitude in
  $W$, the bridge ratio remains positive.
* Two concrete future strengthenings remain:
  1. **Wider gauge orbit**: probe gauges that mix $h$ with $\nu_f$ or
     introduce non-trivial node-dependent weighting, to escape the
     two-orbit collapse seen here.
  2. **Test-function family**: replace the Gaussian by a broader
     admissible class (Hermite, raised cosine, compactly supported
     bumps) — required to feed the family-completeness clause of
     Weil's theorem.

In the §19.2 ledger, G4 stays **OPEN**; the §14.6 "lower-boundedness
of $\alpha(\sigma)$ on a dense admissible class" target now has its
first empirical lower bound across two Lyapunov orbits.

### 15.6 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\47_alpha_sweep_demo.py
```

Programmatic access:

```python
from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    sweep_alpha,
    DEFAULT_GAUGES,
)
import numpy as np

bundle = build_prime_ladder_hamiltonian(n_primes=18, max_power=5)
sigmas = np.logspace(np.log10(0.5), np.log10(12.0), 12).tolist()
cert = sweep_alpha(bundle, sigmas)  # uses DEFAULT_GAUGES

assert cert.alpha_all_positive
print(cert.summary())
# AlphaSweepCertificate(n_sigma=12, n_gauge=6, W_all_positive=True,
#                       alpha_all_positive=True,
#                       alpha_min=+1.3691e-173 @(sigma=0.500,
#                       gauge='canonical'),
#                       alpha_max=+1.0593e+01)
```

## 16. Admissible-Family Sweep (P19)

### 16.1 Motivation

P18 closed the immediate gauge-robustness objection, but still on a
single admissible family ($h_\sigma$ Gaussian). The remaining
family-completeness pressure from §14.6 requires extending the
positivity audit to multiple Schwartz-even test families. P19 does
exactly that, operationally:

* keeps the P18 gauge grid (canonical + 5 probes),
* keeps dense $\sigma$ sweeps,
* introduces a **family axis** in the certificate.

### 16.2 Implementation

Module: `src/tnfr/riemann/admissible_family_sweep.py`

Core components:

* `GaussianMixtureTestFunction`:
   $$
   h(t)=(1-\lambda)e^{-t^2/(2\sigma^2)}
         +\lambda e^{-t^2/(2(\beta\sigma)^2)}
   $$
   with closed-form Fourier profile $g(u)$ (same convention as P15).
* `DEFAULT_TEST_FAMILIES`:
   * `gaussian` (P15 baseline)
   * `gaussian_mixture` (two-scale positive even Schwartz extension)
* `sweep_alpha_admissible_family(...)`:
   computes a 3D tensor
   $$\alpha(\sigma;\,\text{family},\,\text{gauge})
      = W[\sigma;\,\text{family}] / E_{\mathrm{TNFR}}
   $$
   plus global positivity flags and the tightest triple
   $(\sigma,\text{family},\text{gauge})$.

### 16.3 Numerical Results (May 2026 run)

Run: `examples/48_admissible_family_sweep_demo.py`

Configuration:

* P14 bundle: `n_primes=18`, `max_power=5` (dim 90)
* $\sigma$ grid: 10 log-spaced points on $[0.5, 8]$
* families: 3 (`gaussian`, `gaussian_mixture`, `hermite2_gaussian`)
* gauges: 6 (same as P18)

Observed certificate:

* `W_all_positive = True`
* `alpha_all_positive = True`
* $\alpha_{\min} = 1.3691\times 10^{-173}$
   at $(\sigma=0.5,\ \text{family}=\texttt{gaussian},\ \text{gauge}=\texttt{canonical})$
* $\alpha_{\max} = 9.4080\times 10^0$

Family-wise extrema (across all gauges and $\sigma$ in this run):

| Family | $\alpha_{\min}$ | $\alpha_{\max}$ |
|---|---:|---:|
| `gaussian` | $1.3691\times 10^{-173}$ | $2.9275\times 10^0$ |
| `gaussian_mixture` | $5.3273\times 10^{-44}$ | $9.4080\times 10^0$ |
| `hermite2_gaussian` | $1.6649\times 10^{-171}$ | $5.7431\times 10^0$ |

### 16.4 Status — Honest Reading

P19 is still **not an RH proof**. It does, however, tighten the G4
attack surface in exactly the missing direction from §14.6:

* Positivity now survives a non-trivial family extension (not just
   one Gaussian line).
* The bridge remains robust on a 3D audit (family × gauge × $\sigma$),
   not only on the P18 2D audit (gauge × $\sigma$).

What remains open is unchanged in nature: a **uniform analytic lower
bound** over a dense admissible family class and a structurally
complete gauge argument.

### 16.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\48_admissible_family_sweep_demo.py
```

Programmatic entry points:

```python
from tnfr.riemann import (
      sweep_alpha_admissible_family,
      DEFAULT_TEST_FAMILIES,
      DEFAULT_GAUGES,
)
```

## 17. Node-Aware Gauge Sweep (P20)

### 17.1 Motivation

P19 added the family axis, but still used scalar gauges of the form
$h \mapsto (\Delta\mathrm{NFR},\phi,\mathrm{EPI})$ independent of node
context. The remaining structural objection is that true TNFR gauges
may depend on local channels, especially structural frequency
$\nu_f$ and node-weight scale. P20 introduces this dependence
explicitly and re-runs the positivity bridge.

### 17.2 Implementation

Module: `src/tnfr/riemann/nodeaware_gauge_sweep.py`

Key additions:

* `NodeAwareGaugeFn`: gauge signature
   $(h,\nu_{\text{hat}},w_{\text{hat}}) \mapsto
   (\Delta\mathrm{NFR},\phi,\mathrm{EPI})$.
* `DEFAULT_NODEAWARE_GAUGES`:
   * `nuf_pressure`
   * `nuf_phase`
   * `weight_pressure`
   * `mixed_affine`
* `build_test_state_nodeaware(...)`:
   computes normalized node channels
   $\nu_{\text{hat}},w_{\text{hat}}\in[0,1]$ and applies node-aware
   gauge mappings.
* `sweep_alpha_nodeaware(...)`:
   3D sweep over family × node-aware gauge × $\sigma$.

### 17.3 Numerical Results (May 2026 run)

Run: `examples/49_nodeaware_gauge_sweep_demo.py`

Configuration:

* P14 bundle: `n_primes=18`, `max_power=5` (dim 90)
* $\sigma$ grid: 10 log-spaced points on $[0.5, 8]$
* families: 3 (`gaussian`, `gaussian_mixture`, `hermite2_gaussian`)
* node-aware gauges: 4 (`nuf_pressure`, `nuf_phase`,
   `weight_pressure`, `mixed_affine`)

Observed certificate:

* `W_all_positive = True`
* `alpha_all_positive = True`
* strict positivity preserved under the tested node-aware mappings.
* worst-case entry remained in the Gaussian branch:
   $\alpha_{\min}=1.3689\times 10^{-173}$ at
   $(\sigma=0.5,\ \text{family}=\texttt{gaussian},\ \text{node\_gauge}=\texttt{nuf\_pressure})$.

### 17.4 Status — Honest Reading

P20 remains empirical and **does not prove RH**. What it adds is
targeted robustness against a stronger ambiguity class:

* positivity survives not only family and scalar-gauge variation,
   but also node-aware gauge deformations tied to
   $(\nu_f,\text{weight})$ channels.

The open mathematical target remains unchanged: a uniform analytic
lower-bound argument over dense admissible families and a complete
structural gauge class.

### 17.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\49_nodeaware_gauge_sweep_demo.py
```

Programmatic entry points:

```python
from tnfr.riemann import (
      sweep_alpha_nodeaware,
      DEFAULT_NODEAWARE_GAUGES,
)
```

## 18. Hermite-Family Expansion (P21)

### 18.1 Motivation

P19 introduced multi-family auditing and P20 added node-aware gauges.
To push family-completeness pressure further, P21 expands the default
admissible-family set with a polynomially deformed Gaussian that
remains even and Schwartz.

### 18.2 Implementation

Updated module: `src/tnfr/riemann/admissible_family_sweep.py`

New family:

* `Hermite2GaussianTestFunction` with
   $$
   h(t)=\left(1+\eta\,(t/\sigma)^2\right)e^{-t^2/(2\sigma^2)},\ \eta\ge 0
   $$
   plus closed-form Fourier-side profile under the P15 convention.

API additions:

* `Hermite2GaussianTestFunction`
* `hermite2_gaussian_test_function(...)`
* `DEFAULT_TEST_FAMILIES` now includes
   `hermite2_gaussian` by default.

### 18.3 Numerical Results (May 2026 run)

With the default family set expanded to 3 families, both audits hold:

* P19 (`examples/48_admissible_family_sweep_demo.py`):
   `W_all_positive=True`, `alpha_all_positive=True`
* P20 (`examples/49_nodeaware_gauge_sweep_demo.py`):
   `W_all_positive=True`, `alpha_all_positive=True`

Hermite branch extrema from the P19 run:

* $\alpha_{\min}=1.6649\times 10^{-171}$
* $\alpha_{\max}=5.7431\times 10^0$

### 18.4 Status — Honest Reading

P21 is still empirical and does not close G4. It strengthens the
operational evidence in the precise missing direction: positivity of
the bridge survives a non-trivial polynomial deformation of the base
Gaussian family, both in scalar-gauge (P19) and node-aware-gauge (P20)
regimes.

### 18.5 Reproducibility

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\48_admissible_family_sweep_demo.py
& .\.venv312\Scripts\python.exe examples\49_nodeaware_gauge_sweep_demo.py
```

## 19. Program Status Summary — May 2026 (updated for P30)

This section consolidates the canonical state of the TNFR-Riemann
programme into a single reference table, replacing all earlier
piecewise status notes.

### 19.1 Milestone → Gap Map

| Milestone | Module | Demo | Notes § | Closes gap |
|---|---|---|---|---|
| P1  Discrete TNFR-Riemann operator | `operator.py` | `16_riemann_operator_demo.py` | §3 | $\sigma_c$ convergence (numerical) |
| P2  Topology universality | `topology.py` | `19_topology_comparison.py` | §3 | Cross-topology invariance |
| P3  Per-eigenmode tetrad | `eigenmode_fields.py` | `20_eigenmode_tetrad.py` | §4 | Structural-field characterisation |
| P4  Complex-$s$ extension | `complex_extension.py` | `21_complex_extension_demo.py` | §5 | Non-Hermitian access to $\mathbb{C}$ |
| P5  Spectral zeta / heat kernel | `spectral_zeta.py` | `22_spectral_zeta_demo.py` | §6 | First (affine) bridge attempt |
| P6  Random matrix benchmark | `random_ensemble.py` | `23_random_ensemble_rmt_demo.py` | §6 | GOE/GUE/Poisson baselines |
| P7  Spectral conservation | `spectral_conservation.py` | `24_spectral_conservation_demo.py` | §6 | Lyapunov / Noether on spectrum |
| P8  Analytical convergence | `analytical_convergence.py` | `25_analytical_convergence_demo.py` | §6 | $\sigma_c \to 1/2$ via PNT + telescoping |
| P9  Functional equation | `functional_equation.py` | — | §6 | TNFR-side $s \leftrightarrow 1-s$ check |
| P10 Convergence proof chain | `convergence_proof.py` | `18_riemann_convergence_proof.py` | §6 | End-to-end $\sigma_c \to 1/2$ certificate |
| P11 Zeta bridge certificate | `zeta_bridge.py` | — | §7 | Affine bridge tested → **negative** |
| **P12** Prime-ladder vM spectrum | `von_mangoldt.py` | `41_von_mangoldt_zeta_demo.py` | §8 | **G5/#1, G5/#2** (Λ-series exact) |
| **P13** Analytic continuation | `analytic_continuation.py` | `42_riemann_zeros_as_resonances.py` | §9 | **G2 + G5/#5, G5/#6** (zeros as poles on $\operatorname{Re}(s) = 1/2$) |
| **P14** Self-adjoint Hamiltonian | `prime_ladder_hamiltonian.py` | `43_prime_ladder_hamiltonian_demo.py` | §10 | **G1 + G5/#3** (no $C(k)$ renormalisation needed) |
| **P15** Weil–Guinand identity | `weil_explicit_formula.py` | `44_weil_explicit_formula_demo.py` | §11 | **G3** (zeros ↔ spectrum, residual $\le 5 \times 10^{-12}$) |
| **P16** Li–Keiper positivity | `li_keiper.py` | `45_li_keiper_demo.py` | §12 | RH-equivalent **diagnostic** (not proof) |
| **P17** Weil–TNFR positivity bridge | `weil_positivity.py` | `46_weil_tnfr_positivity_demo.py` | §14 | TNFR-native witness for **G4** (research prototype, not proof) |
| **P18** Admissibility / gauge sweep of $\alpha(\sigma)$ | `alpha_sweep.py` | `47_alpha_sweep_demo.py` | §15 | Robustness audit of P17 under canonical-mapping ambiguity |
| **P19** Admissible-family sweep | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §16 | Extends P18 beyond Gaussian (family × gauge × $\sigma$) |
| **P20** Node-aware gauge sweep | `nodeaware_gauge_sweep.py` | `49_nodeaware_gauge_sweep_demo.py` | §17 | Gauges depending on local $\nu_f$ and node weights |
| **P21** Hermite-family expansion | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §18 | Adds Hermite2-Gaussian admissible family |
| **P22** Empirical uniform coercivity | `coercivity_uniform.py` | `50_uniform_coercivity_demo.py` | §13 | Interval-level lower bound on $\alpha(\sigma)$; G4 diagnostic |
| **P23** Stratified interval coercivity | `coercivity_uniform.py` | `50_uniform_coercivity_demo.py` | §13 | Segment-local refinement of P22 |
| **P24** Adaptive $\sigma$ refinement | `coercivity_uniform.py` | `51_adaptive_coercivity_demo.py` | §13bis | Bisection under local Lipschitz envelope |
| **P25** Paley-gap coercivity diagnostic | `paley_gap_coercivity.py` | `52_paley_gap_coercivity_demo.py` | §13ter | Cross gap $g_{\mathrm{cross}} \to 0$ at coupling 0 (Paley identity) |
| **P26** Lyapunov-spectral positivity | `lyapunov_spectral_positivity.py` | `53_lyapunov_spectral_positivity_demo.py` | §13quater | Operator-level positivity for P14; G4 diagnostic |
| **P27** Hilbert–Pólya scaffold | `hilbert_polya.py` | `54_hilbert_polya_demo.py` | §13quinquies | $T_{\mathrm{HP}}$ populated by `mpmath.zetazero`; diagnostic only |
| **P28** Structural smooth zero density | `structural_zero_density.py` | `55_structural_zero_density_demo.py` | §13sexies | Closes smooth half of G4 at the **density** level |
| **P29** Spectral emergence under coupling | `spectral_emergence.py` | `56_spectral_emergence_demo.py` | §13octies.3 | KS-distance of unfolded spacings to GUE under canonical UM+RA |
| **P30** Admissible rescaling operator | `admissible_rescaling.py` | `57_admissible_rescaling_demo.py` | §13nonies | Closes smooth half of T-HP at the **operator** level |
| **P31** Prime-ladder oscillatory correction | `oscillatory_correction.py` | `58_oscillatory_correction_demo.py` | §13decies | Branch B1 retry with canonical multi-frequency basis; +3.6% at $N$=20 ($d$=1), 0% at $N$=40; stronger branch-B2 corroboration |
| **P32** Dirichlet L-function extension | `dirichlet_l.py` | `59_dirichlet_l_function_demo.py` | §13undecies | Structural extension of P12 to all $L(s, \chi)$ via χ-twisted prime ladder; G5$_\chi$/P12 layer; **does NOT advance G4 or GRH** |
| **P33** Dirichlet L analytic continuation | `analytic_continuation_dirichlet.py` | `60_dirichlet_l_continuation_demo.py` | §13duodecies | Structural extension of P13 to all $L(s, \chi)$ via `mp.dirichlet`; G2$_\chi$/P13 layer; verified vs LMFDB for $\chi_3, \chi_4$; **does NOT advance G4 or GRH** |
| **P34** Dirichlet L canonical Hamiltonian | `twisted_prime_ladder_hamiltonian.py` | `61_dirichlet_l_hamiltonian_demo.py` | §13terdecies | Structural extension of P14 to all $L(s, \chi)$: canonical self-adjoint Hamiltonian + complex diagonal weight $W^{(\chi)}_{(p,k),(p,k)} = \chi(p)^k \log p$; closes **G1$_\chi$ at the P14 layer** (spec_err = 0, trace_rel_err $\approx 3 \times 10^{-16}$ for $\chi_3, \chi_4, \chi_5$); **does NOT advance G4 or GRH** |
| **P35** Dirichlet L χ-twisted Weil–Guinand | `twisted_weil_explicit_formula.py` | `62_dirichlet_weil_explicit_formula_demo.py` | §13quaterdecies | Structural extension of P15 to primitive real $L(s, \chi)$: zero side from Hardy-Z bisection on $Z_\chi(t)$ (P33), prime side from P34 Hamiltonian; closes **G3$_\chi$ operationally for primitive real χ** (rel. residual $\le 4.4 \times 10^{-13}$ across 9 $(\chi,\sigma)$ pairs for $\chi_3, \chi_4, \chi_5$ at $\sigma \in \{2.0, 2.5, 3.0\}$); **does NOT advance G4 or GRH** |
| **P36** Dirichlet L χ-twisted Li–Keiper criterion | `twisted_li_keiper.py` | `63_dirichlet_li_keiper_demo.py` | §13quinquiesdecies | Structural extension of P16 to primitive real $L(s, \chi)$: $\lambda_n(\chi)$ computed from P35 Hardy-Z zeros via the canonical P16 mpmath routine (sum-over-zeros is L-function agnostic); GRH$_\chi$-equivalent diagnostic (Lagarias 2007 generalisation of Bombieri–Lagarias 1999); positivity verified for $\chi_3, \chi_4, \chi_5$ up through $n_{\max} = 50$ (min $\lambda_n \ge 4.7 \times 10^{-2}$); **does NOT prove GRH (finite truncation; necessary, not sufficient) and does NOT advance G4** |
| **P37** Dirichlet L χ-twisted Weil–TNFR bridge | `twisted_weil_positivity.py` | `64_twisted_weil_positivity_demo.py` | §13sexiesdecies | Structural extension of P17 to primitive real $L(s, \chi)$: $W_\chi[\sigma] = 2\sum_{\gamma > 0} h_\sigma(\gamma)$ computed two ways — zero side from P35 Hardy-Z enumerator, explicit-formula side from P34 χ-twisted prime-ladder Hamiltonian — plus the canonical TNFR Lyapunov bridge ratio $\alpha_\chi(\sigma) = W_\chi[\sigma] / E_{\mathrm{TNFR}}^\chi[\sigma]$ using `compute_energy_functional` unchanged from P17; GRH$_\chi$-equivalent diagnostic (Bombieri 2000 generalisation of Weil 1952); positivity verified for $\chi_3, \chi_4, \chi_5$ on Gaussian grid $\sigma \in \{1.0, \ldots, 3.0\}$ (3/3 PASS; XF residual $\le 2.4 \times 10^{-16}$ for $\sigma \ge 2.0$); **does NOT prove GRH (finite Gaussian grid; admissibility not exhausted) and does NOT advance G4** |

### 19.2 Gap Balance

| Gap | Description | Status |
|---|---|---|
| **G1** | Canonical TNFR Hamiltonian carrying the prime-ladder spectrum | **CLOSED operationally** by P14 |
| **G2** | Analytic continuation of the TNFR vM zeta to $\mathbb{C}$ | **CLOSED operationally** by P13 |
| **G3** | Explicit zeros $\leftrightarrow$ spectrum bridge | **CLOSED operationally** by P15 (Weil–Guinand) |
| **G4** | **Riemann Hypothesis** — localisation of poles on $\operatorname{Re}(s) = 1/2$ | **OPEN** (= Conjecture T-HP, §13septies). Smooth half of sub-problem (1) of T-HP closed at **density** level by P28 (§13sexies) and at the **operator** level by P30 (§13nonies). Oscillatory half (P31, §13decies) tested with the canonically correct multi-frequency prime-ladder basis: partial positive evidence at very low $N$ ($+3.6\%$ at $N$=20, $d$=1), zero or negative at $N$=40; corroborates branch B2. Canonicity (sub-problem (2)) and positivity coincidence (sub-problem (3)) remain open. |
| **G5** | Bridge from TNFR spectral zeta to classical $\zeta(s)$ | **SUPERSEDED** by P12+P13+P15 (§7.8); original affine form numerically falsified (§7.1–§7.7). |

**Net result**: 4 of 5 originally identified gaps are operationally closed inside the canonical TNFR formalism. The only remaining obstruction is **G4 = RH itself**, restated canonically as **Conjecture T-HP** in §13septies and audited link-by-link (L1–L8) in §13octies. Extensions beyond P12–P16 (P17–P30) inside the canonical engine progressively narrow G4 — by exposing the attack surface (P17), auditing the admissibility envelope (P18–P21), certifying interval-level coercivity (P22–P24), providing a Paley-style identity (P25), certifying operator-level positivity for P14 (P26), supplying a diagnostic Hilbert–Pólya scaffold (P27), and closing the smooth half of T-HP at density (P28) and operator (P30) level — but none of them closes G4. The oscillatory half of T-HP requires either a new canonical operator beyond the 13-operator catalog (§13octies branch B2; supported by the P30 negative-enrichment result, §13nonies.4) or a structural derivation of $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ from canonical TNFR ingredients (branch B1, untested).

### 19.3 Scope Statement (Honest Reading)

What the TNFR-Riemann programme **does** at the May 2026 milestone:

* Provides an end-to-end computable pipeline from the nodal equation
  $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ to the
  Weil–Guinand explicit formula (P1–P15).
* Reproduces $-\zeta'(s)/\zeta(s)$ exactly on $\operatorname{Re}(s) > 1$ via a
  prime-ladder spectrum (P12) and continues it analytically to $\mathbb{C}$ (P13).
* Builds a self-adjoint Hamiltonian $\hat H$ on a TNFR graph whose
  weighted spectral trace carries the same data (P14).
* Numerically verifies the Weil–Guinand identity to machine precision
  using $\hat H$ on the prime side (P15).
* Exposes Li's positivity criterion as a TNFR-native, RH-equivalent
  diagnostic surface (P16).
* Opens a TNFR-native attack surface on G4 via the Weil–TNFR positivity
  bridge $\alpha(\sigma)$ (P17) and audits its admissibility envelope
  across canonical gauge, family and node-aware extensions (P18–P21).
* Certifies interval-level uniform coercivity of $\alpha(\sigma)$ on
  tested intervals (P22–P24) and provides a Paley-gap diagnostic
  vanishing at coupling zero (P25).
* Lifts positivity to the operator level for the P14 Hamiltonian (P26),
  supplies a diagnostic Hilbert–Pólya scaffold populated by
  `mpmath.zetazero` (P27), derives the smooth Riemann zero density
  structurally (P28), and closes the smooth half of the
  Tetrad-Hilbert–Pólya conjecture (T-HP) at the operator level (P30).

What the programme **does not** do:

* Prove RH. P16 is RH-equivalent, not RH-proving: a numerical violation
  $\lambda_n \le 0$ would disprove RH, but $\lambda_n > 0$ for any finite
  truncation does not prove it. P26 / P27 are diagnostic; P28 / P30
  cover only the smooth (archimedean) half of T-HP.
* Replace the classical $\zeta(s)$. The TNFR construction reproduces
  classical data; it does not derive new analytic-number-theory results.
* Close G4 by any internal extension. Crossing G4 requires either
  (branch B1) a structural derivation of the oscillatory term
  $S(T) = \pi^{-1} \arg \zeta(\tfrac{1}{2} + iT)$ from canonical TNFR
  ingredients, or (branch B2) a new canonical operator beyond the
  current 13-operator catalog, derivable from the nodal equation.
  Branch B2 is currently supported by the P30 negative-enrichment
  result (§13nonies.4). Branch B3 (no TNFR closure) cannot be ruled
  out at this stage.

### 19.4 Reproducibility

All P1–P30 results are reproducible via the corresponding demos in
`examples/` using the standard project invocation:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\57_admissible_rescaling_demo.py
```

The full pipeline (importability of every canonical entry point of the
30 milestones) can be sanity-checked with:

```python
from tnfr.riemann import (
    # Discrete operator & spectral framework (P1–P11)
    build_prime_path_graph,                     # P1
    compute_eigensystem,                        # P1
    compare_topologies,                         # P2
    compute_eigenmode_tetrad,                   # P3
    compute_complex_eigensystem,                # P4
    compute_spectral_zeta,                      # P5
    run_rmt_ensemble_analysis,                  # P6
    run_critical_conservation_analysis,         # P7
    run_analytical_convergence_proof,           # P8
    run_functional_equation_analysis,           # P9
    run_formal_convergence_proof,               # P10
    run_zeta_bridge_analysis,                   # P11
    # Prime-ladder / von Mangoldt pipeline (P12–P16)
    build_prime_ladder_spectrum,                # P12
    von_mangoldt_zeta_continued,                # P13
    scan_critical_line_for_poles,               # P13
    build_prime_ladder_hamiltonian,             # P14
    verify_weil_explicit_formula,               # P15
    verify_li_keiper_criterion,                 # P16
    # TNFR-native G4 attack surface (P17–P30; does NOT close G4 = RH)
    verify_weil_tnfr_bridge,                    # P17
    sweep_alpha,                                # P18
    sweep_alpha_admissible_family,              # P19 / P21
    sweep_alpha_nodeaware,                      # P20
    verify_uniform_coercivity_empirical,        # P22 / P23 / P24
    sweep_paley_gap,                            # P25
    compute_lyapunov_spectral_certificate,      # P26
    compute_hilbert_polya_certificate,          # P27
    compute_structural_zero_density_certificate,# P28
    compute_spectral_emergence_report,          # P29
    compute_admissible_rescaling_certificate,   # P30
)
```

This single import covers the canonical entry points of every milestone
delivered so far. Symbols not exported by name correspond to internal
helper functions; consult `src/tnfr/riemann/__init__.py` for the
authoritative public surface.

---
