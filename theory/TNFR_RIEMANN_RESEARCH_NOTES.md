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

## §13septiesdecies. P38 — χ-Twisted Admissibility / Gauge Sweep of $\alpha_\chi(\sigma; g)$ (Structural Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13septiesdecies.1 Motivation

P18 (§15) packages the canonical TNFR-native robustness audit of the P17 Weil–TNFR bridge for $\zeta$: the bridge ratio $\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma; g]$ depends, via the energy denominator, on the structural gauge $g$ that maps a Gaussian width $\sigma$ onto the canonical TNFR test state $(\Delta\mathrm{NFR}, \phi, \mathrm{EPI})$.  The numerator $W[\sigma]$ is gauge-independent (zero-side enumeration), but the denominator is not, so any single-gauge result of $\alpha(\sigma) > 0$ is only as strong as the gauge it is parameterised by.  P18 stress-tests the bridge across the canonical six-gauge family `DEFAULT_GAUGES` = {canonical, dnfr_only, phase_only, epi_only, dnfr_phase, pressure_amplified}.  P37 (§13sexiesdecies) extends P17 to every primitive real Dirichlet $L(s,\chi)$, so the same robustness audit is meaningful on the L-function track once P34 and P35 are in place.  P38 packages these ingredients into the L-function analogue of P18.

### §13septiesdecies.2 Construction

P38 sweeps $\alpha_\chi(\sigma; g) = W_\chi[\sigma] \,/\, E_{\mathrm{TNFR}}^\chi[\sigma; g]$ across a finite Gaussian grid $\{\sigma_i\}$ and the canonical six-gauge family `DEFAULT_GAUGES` inherited unchanged from `alpha_sweep.py` (P18).  Canonical reuse:

* $W_\chi[\sigma]$ is computed once per $\sigma$ (gauge-independent) via the P35 enumerator `twisted_weil_zero_side` at canonical mpmath precision $\mathrm{dps} = 30$.
* For each gauge $g$, the canonical TNFR test state on the P34 χ-twisted prime-ladder bundle is built by mapping each ladder level $E_n = k \log p$ to $h_n = \exp\!\bigl(-E_n^2/(2\sigma^2)\bigr)$ and then applying $g(h_n) = (\Delta\mathrm{NFR}_n, \phi_n, \mathrm{EPI}_n)$, with phases clipped to $[-\pi, \pi]$.
* $E_{\mathrm{TNFR}}^\chi[\sigma; g]$ is computed by the canonical conservation routine `compute_energy_functional` unchanged from P17/P18.

The certificate is a frozen `TwistedAlphaSweepCertificate` carrying the $W_\chi$ row, the $(n_\sigma \times n_g)$ $\alpha_\chi$ table, the energy table, the aggregate positivity flags, and the coordinates of $\alpha_{\min}$ / $\alpha_{\max}$.  No new physics is introduced: P38 is a robustness layer over P34, P35, and P37.

### §13septiesdecies.3 Empirical Verification

The reference demo `examples/65_twisted_alpha_sweep_demo.py` sweeps $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ and all six gauges across $\chi_3, \chi_4, \chi_5$ (decoupled spectrum, $n_{\mathrm{primes}} = 25$, $\max_{\mathrm{power}} = 6$):

| χ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ @ $(\sigma, g)$ | $\alpha_{\max}$ |
|---|---|---|---|---|---|
| $\chi_3$ | 3 | True | True | $+1.27 \times 10^{-14}$ @ $(1.000, \text{canonical})$ | $+6.04 \times 10^{-2}$ |
| $\chi_4$ | 4 | True | True | $+2.71 \times 10^{-8}$ @ $(1.000, \text{canonical})$ | $+6.38 \times 10^{-1}$ |
| $\chi_5$ | 5 | True | True | $+2.62 \times 10^{-10}$ @ $(1.000, \text{canonical})$ | $+1.56 \times 10^{-1}$ |

Positivity holds across every $(\sigma, g)$ combination for every tested character (3/3 PASS).  The smallest $\sigma$ (= 1.0) and the `canonical` gauge consistently produce the most demanding entry, which is the expected behaviour from the P18 ζ-track analogue (narrow Gaussians give the tightest test).

### §13septiesdecies.4 What P38 Extends

* **P18 to L-functions**: P18 is the canonical robustness audit for the ζ-side Weil–TNFR bridge; P38 is its structural analogue for $L(s,\chi)$ at every primitive real χ.  Together with P32–P37, the L-function track now structurally matches the ζ track through the P18 layer.
* **P37 under canonical-mapping ambiguity**: P37 verified $\alpha_\chi(\sigma) > 0$ for the `canonical` gauge only.  P38 confirms that the positivity persists across the entire `DEFAULT_GAUGES` family, ruling out a single-gauge artefact.

### §13septiesdecies.5 What P38 Does NOT Advance

* **GRH for any $L(s,\chi)$**: a finite sweep across $\{\sigma_i\} \times \{g\}$ is **necessary but not sufficient**.  An exhaustive admissible family (which a finite grid cannot exhaust) would be required to upgrade the diagnostic to a proof.  Consistent with the P17/P18 honesty boundary.
* **Complex χ**: P38 inherits the primitive-real restriction from P32–P37.
* **G4 = RH**: $\alpha_\chi(\sigma; g)$ depends on the χ-twisted Hamiltonian (P34) and the χ-twisted explicit formula (P35); neither carries information about the ζ critical line.  G4 is unchanged.

### §13septiesdecies.6 Cross-References

* §13sexiesdecies: P37 (one-shot $\alpha_\chi$ at the `canonical` gauge; P38 generalises across gauges).
* §15: P18 (ζ-side admissibility / gauge sweep; canonical reference template).
* §13septies: Conjecture T-HP (unchanged by P38).
* `src/tnfr/riemann/twisted_alpha_sweep.py`: canonical P38 implementation.
* `examples/65_twisted_alpha_sweep_demo.py`: reference demo.

### §13septiesdecies.7 Gap Balance

| Scope | Status before P38 | Status after P38 |
|-------|-------------------|------------------|
| P18 gauge sweep for ζ | Available (P18) | Available, unchanged |
| **Gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P38) | **Available** (TNFR-native robustness audit across 6 canonical gauges) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(\sigma, g)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P38 closes the admissibility/gauge-sweep gap on the L-function track for every primitive real Dirichlet character.  Combined with P32–P37, the structural TNFR-Riemann program now matches the ζ track all the way through P18.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13octiesdecies. P39 — χ-Twisted Admissible-Family + Gauge Sweep of $\alpha_\chi(\sigma; f, g)$ (Joint Test-Profile / Canonical-Mapping Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13octiesdecies.1 Motivation

P38 (§13septiesdecies) probed the canonical-mapping ambiguity of the P37 chi-twisted positivity bridge by sweeping the six canonical structural gauges `DEFAULT_GAUGES` against a Gaussian-only test profile.  The ζ-track equivalent (P18) was subsequently extended by P19 (`admissible_family_sweep.py`), which sweeps three admissible Schwartz-even test families — `gaussian`, `gaussian_mixture`, `hermite2_gaussian` — to probe the *test-profile* ambiguity of the P17 bridge.  P39 imports the same admissible-family bundle unchanged and combines it with the P38 gauge sweep, yielding a dense $(family, gauge, \sigma)$ certificate for primitive real Dirichlet characters.

### §13octiesdecies.2 Construction

The chi-twisted Weil–TNFR ratio is defined cell-by-cell as
$$\alpha_\chi(\sigma; f, g) \;=\; \frac{W_\chi[\sigma; f]}{E_{\mathrm{TNFR}}^\chi[\sigma; f, g]},$$
where $W_\chi[\sigma; f]$ is the P35 chi-twisted zero-side enumerator evaluated on the admissible test function $f$ at width $\sigma$ (gauge-independent, computed once per $(family, \sigma)$ pair), and $E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ is the canonical TNFR Lyapunov energy of the structural test state built from $(f, g)$ on the P34 chi-twisted graph via `build_twisted_test_state_from_test_function`.  The admissible families are inherited verbatim from P19 (`DEFAULT_TEST_FAMILIES`); the gauges are inherited verbatim from P18 (`DEFAULT_GAUGES`).  No new canonical object is introduced.

### §13octiesdecies.3 Empirical Verification

Demo `examples/66_twisted_admissible_family_sweep_demo.py` evaluates the sweep for $\chi_3, \chi_4, \chi_5$ across 3 families × 6 gauges × 5 widths $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ (90 cells per character, 270 cells total).  Aggregate result:

| Character | Modulus | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | @(σ, family, gauge) | $\alpha_{\max}$ |
|-----------|---------|----------------|-------------------|-----------------|---------------------|-----------------|
| $\chi_3$  | 3 | True | True | $+1.27 \times 10^{-14}$ | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+5.04 \times 10^{-1}$ |
| $\chi_4$  | 4 | True | True | $+2.71 \times 10^{-8}$  | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+2.00 \times 10^{0}$  |
| $\chi_5$  | 5 | True | True | $+2.62 \times 10^{-10}$ | $(1.000, \mathrm{gaussian}, \mathrm{canonical})$ | $+6.94 \times 10^{-1}$ |

PASS rate: **3/3 characters**.  The minimum across every character/family/gauge cell occurs at the tightest Gaussian profile, in agreement with the Gaussian zero-side tail behaviour observed in P19 / P38; admissible mixtures and Hermite–Gaussian profiles inflate $\alpha_\chi$ uniformly, as expected from the spectral weight redistribution introduced by their extra mass at moderate frequencies.

### §13octiesdecies.4 What P39 Extends

P39 extends the P38 robustness audit jointly along the admissible-test-family axis (P19) and the canonical-gauge axis (P18), giving the L-track exact structural parity with the ζ-track at the level of P18 + P19 combined diagnostics.  The chi-twisted positivity bridge is shown to be robust under the *joint* perturbation of test profile and structural mapping for every tested primitive real character.

### §13octiesdecies.5 What P39 Does NOT Advance

P39 is a strict diagnostic and inherits every limitation of P19 / P38.  It does **not** prove GRH for any $L(s, \chi)$ (the $(family, gauge, \sigma)$ grid is finite; positivity on a finite grid is necessary but not sufficient for $L$-function admissibility on the full Schwartz cone).  It does **not** advance G4 = RH (the arithmetic obstruction is identical to the untwisted case).  It does **not** address GRH for complex Dirichlet characters (only primitive real $\chi_3, \chi_4, \chi_5$ are implemented).  Negative cells, if encountered at scale, would falsify the bridge *as parameterised by the given test family and gauge*; they would not falsify GRH$_\chi$ itself, which depends only on the gauge-independent quantities $W_\chi[\sigma; f]$.

### §13octiesdecies.6 Cross-References

* P19 (ζ-track admissible-family sweep): `src/tnfr/riemann/admissible_family_sweep.py`, §15 of these notes.
* P18 (canonical gauge family): `src/tnfr/riemann/alpha_sweep.py`, §14.
* P34 (chi-twisted prime-ladder Hamiltonian): `src/tnfr/riemann/twisted_prime_ladder_hamiltonian.py`, §13quaterdecies.
* P35 (chi-twisted Weil–Guinand zero-side enumerator): `src/tnfr/riemann/twisted_weil_explicit_formula.py`, §13quindecies.
* P37 (chi-twisted Weil–TNFR positivity bridge): `src/tnfr/riemann/twisted_weil_tnfr_bridge.py`, §13septdecies.
* P38 (chi-twisted gauge sweep): `src/tnfr/riemann/twisted_alpha_sweep.py`, §13septiesdecies.
* P17 (canonical Weil–TNFR positivity bridge): `src/tnfr/riemann/weil_positivity.py`, §14.
* Implementation: `src/tnfr/riemann/twisted_admissible_family_sweep.py`.
* Demo: `examples/66_twisted_admissible_family_sweep_demo.py`.

### §13octiesdecies.7 Gap Balance

| Scope | Status before P39 | Status after P39 |
|-------|-------------------|------------------|
| P19 admissible-family sweep for ζ | Available (P19) | Available, unchanged |
| P38 gauge sweep for $L(s,\chi)$ | Available (P38) | Available, unchanged |
| **Admissible-family + gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P39) | **Available** (TNFR-native robustness audit across 3 admissible families × 6 canonical gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(family, gauge, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P39 closes the admissible-family + gauge robustness gap on the L-function track for every primitive real Dirichlet character, achieving structural parity with the ζ-track through P19.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13noniesdecies. P40 — χ-Twisted Node-Aware Gauge Sweep of $\alpha_\chi(\sigma; f, g)$ (Node-Aware Canonical-Mapping Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13noniesdecies.1 Motivation

P38 swept the six canonical *scalar-h* structural gauges `DEFAULT_GAUGES` for primitive real $L(s,\chi)$.  P39 enriched that sweep along the test-profile axis by crossing the six scalar gauges with the three admissible test families `DEFAULT_TEST_FAMILIES` of P19.  Both P38 and P39 share a structural limitation: every gauge produces *node-independent* triples $(d, \phi, \epsilon)$ from the scalar $h(E_n)$.  The χ-twisted prime-ladder graph carries two independent canonical channels at each node $n = (p, k)$ — the structural frequency $\nu_f(n) = k \log p$ and the node-weight $\log p$ — that the scalar-h gauges discard by construction.  The ζ-track closed this gap at P20 with the four *node-aware* gauges `DEFAULT_NODEAWARE_GAUGES`.  P40 lifts that node-aware family verbatim to the L-function track for every primitive real Dirichlet character.

### §13noniesdecies.2 Construction

The P40 sweep evaluates

$$
\alpha_\chi(\sigma; f, g) \;=\; \frac{W_\chi[\sigma; f]}{E_{\mathrm{TNFR}}^\chi[\sigma; f, g]}
$$

across (i) the three admissible Schwartz-even test families `DEFAULT_TEST_FAMILIES` inherited unchanged from P19 (gaussian, gaussian_mixture, hermite2_gaussian); (ii) the four canonical node-aware gauges `DEFAULT_NODEAWARE_GAUGES` inherited unchanged from P20 (nuf_pressure, nuf_phase, weight_pressure, mixed_affine); (iii) a finite Gaussian-width grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  Each node-aware gauge has the canonical signature

$$
(d_n, \phi_n, \epsilon_n) \;=\; g\bigl(h(E_n),\, \hat\nu_f(n),\, \hat w(n)\bigr),
$$

where $\hat\nu_f(n)$ and $\hat w(n) = \log p / \max_{n'} \log p$ are the per-node normalised structural-frequency and node-weight channels of the P34 χ-twisted prime-ladder bundle.  $W_\chi[\sigma; f]$ is gauge-independent and is computed once per $(family, \sigma)$ via the P35 enumerator `twisted_weil_zero_side`; the canonical TNFR test state is built per $(family, node\_gauge)$ on the P34 bundle via `build_twisted_test_state_nodeaware`, then $E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ is the tetrad energy functional of P17 evaluated on that state.

### §13noniesdecies.3 Empirical Verification

`examples/67_twisted_nodeaware_gauge_sweep_demo.py` evaluates the sweep for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (25, 6, 0)$:

| $\chi$ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | argmin $(\sigma, f, g)$ | $\alpha_{\max}$ |
|--------|----:|:--------------:|:-----------------:|----------------:|:-----------------------:|----------------:|
| $\chi_{3}$ | 3 | True | True | $+1.25 \times 10^{-14}$ | $(1.0, \text{gaussian}, \text{nuf\_phase})$ | $+6.71 \times 10^{-2}$ |
| $\chi_{4}$ | 4 | True | True | $+2.69 \times 10^{-08}$ | $(1.0, \text{gaussian}, \text{nuf\_phase})$ | $+1.30 \times 10^{-1}$ |
| $\chi_{5}$ | 5 | True | True | $+2.60 \times 10^{-10}$ | $(1.0, \text{gaussian}, \text{nuf\_pressure})$ | $+1.12 \times 10^{-1}$ |

Aggregate result: **3/3 characters PASS** across $3 \times 4 \times 5 = 60$ $(family, node\_gauge, \sigma)$ entries each.  The argmin location is consistently the small-$\sigma$ / gaussian / pressure-side corner of the grid, where $W_\chi$ approaches the Plancherel limit while $E_{\mathrm{TNFR}}^\chi$ is largest — the same qualitative signature observed at P20 for the ζ-track and at P39 for the scalar-gauge twisted sweep.

### §13noniesdecies.4 What P40 Extends

| Component | P38 | P39 | **P40** |
|-----------|:---:|:---:|:-------:|
| Test family axis | single (gaussian) | sweep (3 admissible) | sweep (3 admissible) |
| Gauge axis | sweep (6 scalar) | sweep (6 scalar) | **sweep (4 node-aware)** |
| Node-aware channels $(\hat\nu_f, \hat w)$ | discarded | discarded | **active** |
| ζ-track parent | P18 | P19 | **P20** |

P40 closes the node-aware canonical-mapping robustness gap on the L-function track for primitive real Dirichlet characters, achieving structural parity with the ζ-track P20.

### §13noniesdecies.5 What P40 Does NOT Advance

P40 is a **finite-grid robustness diagnostic**: positivity of $\alpha_\chi(\sigma; f, g)$ on the chosen $(family, node\_gauge, \sigma)$ grid is necessary but not sufficient for GRH$_\chi$, and the GRH-equivalent content is carried entirely by the gauge-independent zero side $W_\chi[\sigma; f] \ge 0$ for all admissible $f$.  P40 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH (which lives strictly inside the canonical ζ track via P30 → T-HP).

### §13noniesdecies.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_nodeaware_gauge_sweep.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/67_twisted_nodeaware_gauge_sweep_demo.py`.
- ζ-track parent: P20 (§13ter `nodeaware_gauge_sweep.py`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (`verify_twisted_weil_tnfr_bridge`, energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep).
- Inherited canonical pieces: `DEFAULT_TEST_FAMILIES` (P19), `DEFAULT_NODEAWARE_GAUGES` (P20), `compute_energy_functional` (P17).
- Compendium: §19.1 P40 row.

### §13noniesdecies.7 Gap Balance

| Scope | Status before P40 | Status after P40 |
|-------|-------------------|------------------|
| P20 node-aware gauge sweep for ζ | Available (P20) | Available, unchanged |
| P39 admissible-family + scalar-gauge sweep for $L(s,\chi)$ | Available (P39) | Available, unchanged |
| **Admissible-family + node-aware gauge sweep for $L(s,\chi)$, primitive real χ** | Open (future P40) | **Available** (TNFR-native robustness audit across 3 admissible families × 4 node-aware gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(family, node\_gauge, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P40 closes the node-aware canonical-mapping robustness gap on the L-function track for every primitive real Dirichlet character, achieving structural parity with the ζ-track through P20.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13vicies. P41 — χ-Twisted Hermite2-Gaussian η-Parameter Sweep of $\alpha_\chi(\sigma; \eta, g)$ (Hermite2 Envelope-Strength Robustness Diagnostic; GRH$_\chi$-Equivalent for Primitive Real χ; Does NOT Prove GRH or Advance G4)

### §13vicies.1 Motivation

P39 and P40 swept the three admissible Schwartz-even test families `DEFAULT_TEST_FAMILIES` of P19 with the Hermite2-Gaussian profile fixed at its canonical envelope strength $\eta = 0.25$.  The Hermite2 profile

$$
h_{\sigma,\eta}(t) \;=\; \bigl(1 + \eta (t/\sigma)^2\bigr)\, e^{-t^2/(2\sigma^2)}
$$

is a one-parameter family of Schwartz-even test functions that recovers the pure Gaussian baseline at $\eta = 0$ and progressively biases the test profile toward the wings as $\eta$ grows.  The ζ-track P21 added the Hermite2-Gaussian to the admissible-family registry but did not separately probe the envelope-strength axis itself.  P41 enriches the L-track sweep along that orthogonal axis: it varies $\eta$ over a finite grid spanning baseline-Gaussian to strongly-deformed envelope for every primitive real Dirichlet character.

### §13vicies.2 Construction

The P41 sweep evaluates

$$
\alpha_\chi(\sigma; \eta, g) \;=\; \frac{W_\chi[\sigma; \eta]}{E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]}
$$

across (i) the Hermite2 envelope-strength grid `DEFAULT_HERMITE2_ETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)` ($\eta = 0$ recovers the pure Gaussian baseline; $\eta = 0.25$ matches the P19/P39 snapshot); (ii) the six canonical scalar gauges `DEFAULT_GAUGES` inherited unchanged from P18; (iii) the same finite Gaussian-width grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$.  $W_\chi[\sigma; \eta]$ is gauge-independent and computed once per $(\eta, \sigma)$ via the P35 enumerator `twisted_weil_zero_side`; the canonical TNFR test state is built per $(\eta, g)$ on the P34 χ-twisted bundle via `build_twisted_test_state_from_test_function` (reused from P39), then $E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]$ is the tetrad energy functional of P17 evaluated on that state.

### §13vicies.3 Empirical Verification

`examples/68_twisted_hermite_family_demo.py` evaluates the sweep for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (25, 6, 0)$:

| $\chi$ | $q$ | $W_\chi \ge 0$ | $\alpha_\chi > 0$ | $\alpha_{\min}$ | argmin $(\sigma, \eta, g)$ | $\alpha_{\max}$ |
|--------|----:|:--------------:|:-----------------:|----------------:|:--------------------------:|----------------:|
| $\chi_{3}$ | 3 | True | True | $+1.27 \times 10^{-14}$ | $(1.0, 0.0, \text{canonical})$ | $+9.54 \times 10^{-1}$ |
| $\chi_{4}$ | 4 | True | True | $+2.71 \times 10^{-08}$ | $(1.0, 0.0, \text{canonical})$ | $+6.00 \times 10^{+0}$ |
| $\chi_{5}$ | 5 | True | True | $+2.62 \times 10^{-10}$ | $(1.0, 0.0, \text{canonical})$ | $+1.79 \times 10^{+0}$ |

Aggregate result: **3/3 characters PASS** across $6 \times 6 \times 5 = 180$ $(\eta, g, \sigma)$ entries each.  $W_\chi[\sigma; \eta]$ is monotone non-decreasing in $\eta$ at each fixed $\sigma$, consistent with the broader-spectral-support character of the deformed envelope; $\alpha_\chi$ increases sharply with $\eta$ along the `dnfr_only` and `epi_only` gauge channels and remains nearly $\eta$-invariant along the four canonical gauges that consume the $h$-channel only.  The argmin is consistently $(\sigma, \eta, g) = (1.0, 0.0, \text{canonical})$ — the Gaussian baseline corner — matching the P38/P39 argmin pattern.

### §13vicies.4 What P41 Extends

| Component | P38 | P39 | P40 | **P41** |
|-----------|:---:|:---:|:---:|:-------:|
| Test family axis | single (gaussian) | sweep (3 admissible, $\eta = 0.25$ fixed) | sweep (3 admissible, $\eta = 0.25$ fixed) | **sweep (Hermite2 with 6-point $\eta$-grid)** |
| Gauge axis | sweep (6 scalar) | sweep (6 scalar) | sweep (4 node-aware) | sweep (6 scalar) |
| Hermite2 envelope-strength $\eta$ | n/a | fixed at $0.25$ | fixed at $0.25$ | **swept over $\{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$** |
| ζ-track parent | P18 | P19 | P20 | **P21** |

P41 closes the Hermite2 envelope-strength robustness gap on the L-function track for primitive real Dirichlet characters, achieving structural parity with the ζ-track P21 along the envelope-deformation axis.

### §13vicies.5 What P41 Does NOT Advance

P41 is a **finite-grid robustness diagnostic**: positivity of $\alpha_\chi(\sigma; \eta, g)$ on the chosen $(\eta, g, \sigma)$ grid is necessary but not sufficient for GRH$_\chi$, and the GRH-equivalent content is carried entirely by the gauge-independent zero side $W_\chi[\sigma; \eta] \ge 0$ for every admissible Hermite2 profile.  P41 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH (which lives strictly inside the canonical ζ track via P30 → T-HP).  The Hermite2 family is a one-parameter polynomial-envelope deformation of the Gaussian; it is not exhaustive over the full admissible Schwartz-even space.

### §13vicies.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_hermite_family.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/68_twisted_hermite_family_demo.py`.
- ζ-track parent: P21 (Hermite2 added to `DEFAULT_TEST_FAMILIES`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep; supplies `build_twisted_test_state_from_test_function`), P40 (node-aware twisted sweep).
- Inherited canonical pieces: `Hermite2GaussianTestFunction` (P19), `DEFAULT_GAUGES` (P18), `compute_energy_functional` (P17).
- Compendium: §19.1 P41 row.

### §13vicies.7 Gap Balance

| Scope | Status before P41 | Status after P41 |
|-------|-------------------|------------------|
| P21 Hermite2 family in ζ-track admissible registry | Available (P21) | Available, unchanged |
| P39 admissible-family + scalar-gauge sweep for $L(s,\chi)$ at fixed $\eta = 0.25$ | Available (P39) | Available, unchanged |
| **Hermite2 envelope-strength η-sweep for $L(s,\chi)$, primitive real χ** | Open (future P41) | **Available** (TNFR-native robustness audit across 6 $\eta$ values × 6 scalar gauges × σ grid) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (diagnostic only; finite $(\eta, g, \sigma)$ grid is necessary, not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P41 closes the Hermite2 envelope-strength robustness gap on the L-function track for every primitive real Dirichlet character.  The arithmetic obstruction remains identical and the gap balance for G4 is unchanged.

## §13vicies-primo. P42 — χ-Twisted Uniform-Coercivity Certificate (Lipschitz-Mesh Interval Bound on $\alpha_\chi(\sigma; \eta, g)$; Diagnostic; Does NOT Prove GRH or Advance G4)

### §13vicies-primo.1 Motivation

P38–P41 verified pointwise positivity of $\alpha_\chi(\sigma; f, \eta, g) = W_\chi[\sigma; f, \eta] / E_{\mathrm{TNFR}}^\chi[\sigma; f, \eta, g]$ at the canonical finite grid $\sigma \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ jointly across the test-family (P39), node-aware-gauge (P40) and Hermite2 envelope-strength (P41) axes.  None of those sweeps controls $\alpha_\chi$ between grid points.  The ζ-track P22 lifted the equivalent ζ-side sample to an **interval** lower bound by combining a sampled minimum with a finite-difference Lipschitz envelope and a log-spaced mesh of explicit radius.  P42 transports the same Lipschitz-mesh certificate construction to the χ-twisted track for every primitive real Dirichlet character, taking the sample over the *joint* (admissible-family + scalar-gauge + node-aware-gauge) sweep already canonicalised in P39 and P40.

### §13vicies-primo.2 Construction

The P42 certificate evaluates

$$
\alpha_\chi(\sigma; \eta, g) \;=\; \frac{W_\chi[\sigma; \eta]}{E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]}
$$

on a log-spaced grid $\sigma_0 < \sigma_1 < \cdots < \sigma_{N-1}$ with $\sigma_k = \sigma_{\min} \cdot (\sigma_{\max}/\sigma_{\min})^{k/(N-1)}$, then:

1. Runs the **scalar-gauge sweep of P39** (`sweep_twisted_admissible_family`, 3 admissible families × 6 canonical scalar gauges of P18) once on the log-spaced grid;
2. Runs the **node-aware-gauge sweep of P40** (`sweep_twisted_nodeaware_gauge`, 3 admissible families × 4 canonical node-aware gauges of P20) once on the same grid;
3. Concatenates both $\alpha_\chi$ tables and extracts the sampled minimum $\alpha_{\chi,\min}^{\mathrm{samp}}$, maximum $\alpha_{\chi,\max}^{\mathrm{samp}}$ and an upper bound on the finite-difference Lipschitz envelope $L^{\mathrm{proxy}}_\chi = \max_{k} |\alpha_\chi(\sigma_{k+1}; \cdot) - \alpha_\chi(\sigma_k; \cdot)| / |\sigma_{k+1} - \sigma_k|$;
4. Computes three interval lower bounds — **global** ($\alpha_{\chi,\min}^{\mathrm{samp}} - L^{\mathrm{proxy}}_\chi \cdot \rho$ with mesh radius $\rho = \max_k (\sigma_{k+1} - \sigma_k)/2$), **stratified** (segment-wise mid-radius), **local** (segment-wise endpoint-aware) — reusing the canonical ζ-track helpers `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound` of P22 unchanged from `coercivity_uniform.py`;
5. Optionally performs **P24-style adaptive refinement**: bisects the `per_round` worst-margin segments, re-runs both twisted sweeps on the augmented grid, and recomputes the segment-local interval lower bound.

The construction does NOT touch the gauge-independent zero side $W_\chi$ (computed once per $(\eta, \sigma)$ via the P35 enumerator inside each sweep), the P34 χ-twisted bundle, the P17 energy functional, or any of the canonical default registries.

### §13vicies-primo.3 Empirical Verification

`examples/69_twisted_coercivity_uniform_demo.py` evaluates the certificate for every primitive real Dirichlet character of conductor $q \le 5$ with bundle $(n_{\mathrm{primes}}, k_{\max}, J) = (15, 4, 0)$ on the log-spaced window $\sigma \in [1.0, 3.0]$ with $N = 5$, using `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_GAUGES` (P18) for the scalar sweep and `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_NODEAWARE_GAUGES` (P20) for the node-aware sweep:

| $\chi$ | $q$ | $\alpha^{\mathrm{samp}}_{\chi,\min}$ | $\alpha^{\mathrm{samp}}_{\chi,\max}$ | $L^{\mathrm{proxy}}_\chi$ | $\mathrm{lb}_{\mathrm{global}}$ | $\mathrm{lb}_{\mathrm{strat}}$ | $\mathrm{lb}_{\mathrm{local}}$ | all+ |
|--------|----:|------------------------------------:|------------------------------------:|--------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|:----:|
| $\chi_{3}$ | 3 | $+1.26 \times 10^{-14}$ | $+5.10 \times 10^{-1}$ | $4.31 \times 10^{-1}$ | $-1.55 \times 10^{-1}$ | $-1.55 \times 10^{-1}$ | $-6.06 \times 10^{-2}$ | False |
| $\chi_{4}$ | 4 | $+2.70 \times 10^{-8}$  | $+2.01 \times 10^{+0}$ | $1.48 \times 10^{+0}$ | $-5.33 \times 10^{-1}$ | $-5.16 \times 10^{-1}$ | $-1.30 \times 10^{-1}$ | False |
| $\chi_{5}$ | 5 | $+2.62 \times 10^{-10}$ | $+7.01 \times 10^{-1}$ | $5.49 \times 10^{-1}$ | $-1.98 \times 10^{-1}$ | $-1.95 \times 10^{-1}$ | $-6.51 \times 10^{-2}$ | False |

Sampled positivity holds for every $\chi$ on every grid point in both sweeps (`sampled_all_positive = True`, `admissible_ok = True`, `nodeaware_ok = True`).  All three Lipschitz-mesh interval lower bounds are **negative** for all three characters: $\alpha^{\mathrm{samp}}_{\chi,\min} \approx 10^{-8}$ to $10^{-14}$ near the $\sigma = 1$ baseline gives essentially zero margin against any finite slope $L^{\mathrm{proxy}}_\chi$.  P24-style refinement on the worst-margin character ($\chi_4$, $\mathrm{lb}_{\mathrm{local}} = -1.30 \times 10^{-1}$) with one round of two-midpoint bisection ($N = 5 \to 7$) reduces the local interval lower bound to $-3.40 \times 10^{-2}$ — a **74% margin reduction toward zero**, confirming the bisection mechanism transports correctly to the χ-twisted side, while the bound remains negative because the sampled minimum near $\sigma = 1$ has not been pushed off the worst-margin endpoint.

### §13vicies-primo.4 What P42 Extends

| Component | P22 (ζ-track) | P38 | P39 | P40 | P41 | **P42** |
|-----------|:-------------:|:---:|:---:|:---:|:---:|:-------:|
| Sample / interval | interval | pointwise | pointwise | pointwise | pointwise | **interval (Lipschitz-mesh)** |
| σ grid | log-spaced | finite | finite | finite | finite | **log-spaced (same construction as P22)** |
| Lipschitz envelope | finite-difference | n/a | n/a | n/a | n/a | **finite-difference (P22 helpers reused)** |
| Joint scalar + node-aware sample | scalar only | scalar only | scalar only | node-aware only | scalar only | **both (P39 ∪ P40)** |
| Adaptive refinement | yes (P24) | n/a | n/a | n/a | n/a | **yes (P24 helpers reused)** |
| ζ-track parent | — | P18 | P19 | P20 | P21 | **P22 (+ P23 + P24)** |

P42 transports the canonical ζ-track interval-coercivity certificate construction (P22) plus its stratified (P23) and adaptive (P24) refinements to the L-function track for every primitive real Dirichlet character, taking the underlying sample over the joint P39 + P40 robustness sweep.

### §13vicies-primo.5 What P42 Does NOT Advance

P42 is a **finite-grid Lipschitz-mesh interval diagnostic**: positive interval lower bounds would be necessary but not sufficient for GRH$_\chi$.  The current empirical result is **negative interval lower bounds** for all three characters even after one round of bisection refinement, exactly mirroring the ζ-track P22/P23/P24 behaviour at coarse-mesh / wide-σ-window initial state: uniform coercivity is delicate near the $\sigma = 1$ baseline because the sampled minimum is genuinely tiny ($10^{-8}$ to $10^{-14}$).  P42 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, and does NOT advance the gap balance for G4 = RH.

### §13vicies-primo.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_coercivity_uniform.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/69_twisted_coercivity_uniform_demo.py`.
- ζ-track parent: P22 / P23 / P24 (uniform, stratified, adaptive coercivity in `coercivity_uniform.py`).
- L-track parents: P34 (χ-twisted bundle), P35 (`twisted_weil_zero_side`), P37 (energy functional), P38 (scalar-gauge twisted sweep), P39 (admissible-family + scalar-gauge twisted sweep), P40 (node-aware twisted sweep), P41 (Hermite2 η-sweep).
- Inherited canonical pieces: `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound`, `_worst_segment_indices` (P22 / P23 / P24 helpers reused unchanged from `coercivity_uniform.py`); `sweep_twisted_admissible_family` (P39); `sweep_twisted_nodeaware_gauge` (P40).
- Compendium: §19.1 P42 row.

### §13vicies-primo.7 Gap Balance

| Scope | Status before P42 | Status after P42 |
|-------|-------------------|------------------|
| P22 ζ-track interval coercivity certificate | Available (P22) | Available, unchanged |
| Pointwise positivity of $\alpha_\chi$ for primitive real χ on finite $(\sigma, f, \eta, g)$ grid | Available (P37–P41) | Available, unchanged |
| **Lipschitz-mesh interval-level certificate of $\alpha_\chi$ for primitive real χ** | Open (future P42) | **Available** (diagnostic; current empirical interval lower bounds are negative; adaptive bisection mechanism transports correctly from ζ-track) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (interval lower bounds currently negative; diagnostic, not sufficient even when positive) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P42 closes the **Lipschitz-mesh interval-certificate construction gap** on the L-function track for every primitive real Dirichlet character (canonical pieces transport without modification; bisection refinement behaves qualitatively as on the ζ-track).  The current empirical interval lower bounds are negative — a HONEST finding, not a failure — and the arithmetic obstruction plus the gap balance for G4 are unchanged.

## §13vicies-secundo. P43 — χ-Twisted Paley-Gap Consistency Diagnostic ($|Z_{P34} - Z_{P32}|$ and Truncation Gaps on the L-Track; Diagnostic; Does NOT Prove GRH or Advance G4)

### §13vicies-secundo.1 Motivation

The ζ-track P25 milestone transported the Paley-gap philosophy of Martínez Gamo, *Spectral note: Paley gap via $\lambda_2$ (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2 (November 2025), onto the TNFR-Riemann coercivity scaffold by comparing three representations of the von Mangoldt logarithmic derivative — the P12 prime-ladder closed form $Z_{P12}(\sigma)$, the P14 self-adjoint weighted spectral trace $Z_{P14}(\sigma)$, and the classical truncated Dirichlet series $\sum_{n \le N} \Lambda(n)/n^\sigma$ — via three absolute Paley-gap quantities $g_{P12}(\sigma)$, $g_{P14}(\sigma)$, $g_{\mathrm{cross}}(\sigma) = |Z_{P14}(\sigma) - Z_{P12}(\sigma)|$. The Paley-style observation was that at zero inter-ladder coupling the cross gap $g_{\mathrm{cross}}$ collapses to machine precision by a closed-form algebraic identity between the two TNFR realisations, while a non-zero coupling exposes a clean structural-deformation magnitude free of classical-truncation noise.

The χ-twisted track P32–P34 provides the same two realisations for the von Mangoldt logarithmic derivative of $L(s, \chi)$ — the P32 closed-form weighted spectrum (`tnfr_log_l_derivative`) and the P34 self-adjoint χ-twisted weighted spectral trace (`twisted_weighted_spectral_trace`) — alongside the classical truncated reference $\sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ (`classical_log_l_derivative`). P43 transports the P25 Paley-gap diagnostic construction to the χ-twisted track for every primitive real Dirichlet character, providing the analogous consistency surface on the L-track.

### §13vicies-secundo.2 Construction

The P43 diagnostic evaluates, on a real $\sigma$-grid, three absolute χ-twisted Paley-gap quantities

$$
\begin{aligned}
g_{P32}(\sigma) &= \left|Z_{P32}(\sigma, \chi) - Z_{\mathrm{cls}}(\sigma, \chi)\right|, \\
g_{P34}(\sigma) &= \left|Z_{P34}(\sigma, \chi) - Z_{\mathrm{cls}}(\sigma, \chi)\right|, \\
g_{\mathrm{cross}}(\sigma) &= \left|Z_{P34}(\sigma, \chi) - Z_{P32}(\sigma, \chi)\right|,
\end{aligned}
$$

where $Z_{P32}(\sigma, \chi) = \sum_{(\mu, w) \in \mathrm{spec}_\chi} w \, e^{-\sigma \mu}$ is the P32 closed-form weighted spectrum, $Z_{P34}(\sigma, \chi) = \mathrm{Tr}(W_\chi e^{-\sigma H_{\mathrm{int}}})$ is the P34 χ-twisted weighted spectral trace, and $Z_{\mathrm{cls}}(\sigma, \chi) = \sum_{n \le N_{\max}} \chi(n) \Lambda(n) / n^\sigma$ is the classical truncated reference, all computed on the same $(n_{\mathrm{primes}}, k_{\max})$ prime-ladder bundle. The driver `sweep_twisted_paley_gap(bundle, chi, sigmas, n_max_classical=...)` returns a `TwistedPaleyGapSweep` dataclass carrying the three gap arrays and their worst-case magnitudes per character.

### §13vicies-secundo.3 Empirical Verification

P43 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, N_{\max}^{\mathrm{cls}}) = (18, 5, 50\,000)$, $\sigma \in [1.5, 4.0]$ with $N = 11$, two bundles per character (decoupled $J_0 = 0$ and weakly coupled $J_0 = 10^{-2}$) for $\chi_3, \chi_4, \chi_5$:

| χ | $q$ | $\max g_{\mathrm{cross}}^{[J_0=0]}$ | $\max g_{\mathrm{cross}}^{[J_0=10^{-2}]}$ | $\max g_{P32}$ |
|---|---|---|---|---|
| $\chi_3$ | 3 | $5.55 \times 10^{-17}$ | $1.01 \times 10^{-5}$ | $2.22 \times 10^{-3}$ |
| $\chi_4$ | 4 | $4.16 \times 10^{-17}$ | $8.25 \times 10^{-6}$ | $1.49 \times 10^{-2}$ |
| $\chi_5$ | 5 | $1.11 \times 10^{-16}$ | $1.51 \times 10^{-5}$ | $1.11 \times 10^{-2}$ |

The decoupled cross gap collapses to machine precision ($O(10^{-17})$) for every character, confirming the Paley-style algebraic identity between P32 and P34 on the L-track (regression test, **not** a discovery: the identity follows from $W_\chi$ being the diagonal lift of the spectrum weights and $H_{\mathrm{int}}$ being block-diagonal at $J_0 = 0$). The coupling-induced cross gap jumps to $O(10^{-5})$ — twelve orders of magnitude above noise — exposing pure coupling-induced deformation of the χ-twisted prime-ladder identity, free of classical-truncation noise (which contaminates $g_{P32}$ at the $10^{-3}$ to $10^{-2}$ level).

### §13vicies-secundo.4 What P43 Extends

| Component | P25 (ζ-track) | P32 | P34 | **P43** |
|-----------|---------------|-----|-----|---------|
| Closed-form von Mangoldt logarithmic derivative | $Z_{P12}$ for $-\zeta'/\zeta$ | $Z_{P32}$ for $-L'(s,\chi)/L(s,\chi)$ | — | $Z_{P32}$ reused unchanged |
| Self-adjoint weighted spectral trace | $Z_{P14}$ for $-\zeta'/\zeta$ | — | $Z_{P34}$ for $-L'(s,\chi)/L(s,\chi)$ | $Z_{P34}$ reused unchanged |
| Classical truncated reference | $\sum_{n \le N} \Lambda(n)/n^\sigma$ | — | — | $\sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ via `classical_log_l_derivative` |
| Three Paley-gap quantities | $g_{P12}, g_{P14}, g_{\mathrm{cross}}$ | — | — | $g_{P32}, g_{P34}, g_{\mathrm{cross}}$ |
| Paley-style decoupled identity ($g_{\mathrm{cross}} \to 0$ at $J_0 = 0$) | Empirically $O(10^{-17})$ | — | — | Empirically $O(10^{-17})$ for $\chi_3, \chi_4, \chi_5$ |
| Coupling-induced deformation signal | $J_0 = 10^{-2} \Rightarrow g_{\mathrm{cross}} \sim 10^{-5}$ | — | — | $J_0 = 10^{-2} \Rightarrow g_{\mathrm{cross}} \sim 10^{-5}$ for $\chi_3, \chi_4, \chi_5$ |

P43 transports the canonical ζ-track Paley-gap diagnostic (P25) to the L-function track for every primitive real Dirichlet character, exhibiting the identical decoupled-identity / coupled-deformation pattern.

### §13vicies-secundo.5 What P43 Does NOT Advance

P43 is a **consistency diagnostic at coupling zero and a deformation magnitude at coupling positive**. Vanishing of $g_{\mathrm{cross}}$ at $J_0 = 0$ is **necessary but not sufficient** for any structural positivity claim and **not connected** to GRH localisation; it is a regression test selecting consistent realisations, not a coercivity certificate. P43 does NOT prove GRH for any $L(s, \chi)$, does NOT extend to complex characters, does NOT advance the P42 interval-coercivity certificate (which lives on the $\alpha_\chi$ axis, not on the von Mangoldt logarithmic derivative axis), and does NOT advance the gap balance for G4 = RH. The Zenodo source note itself disclaims primality proof status; P43 inherits the same scope at the L-track coercivity-diagnostic level.

### §13vicies-secundo.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_paley_gap_coercivity.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/70_twisted_paley_gap_coercivity_demo.py`.
- ζ-track parent: P25 (`paley_gap_coercivity.py`).
- L-track parents: P32 (`tnfr_log_l_derivative`, `TwistedPrimeLadderSpectrum`), P34 (`TwistedPrimeLadderHamiltonian`, `twisted_weighted_spectral_trace`).
- Inherited canonical pieces: `tnfr_log_l_derivative` (P32 Route A), `twisted_weighted_spectral_trace` (P34 Route B), `classical_log_l_derivative` (classical reference), `build_twisted_prime_ladder_hamiltonian` (P34 bundle constructor) reused unchanged.
- External source: Martínez Gamo, *Spectral note: Paley gap via $\lambda_2$ (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2 (November 2025).
- Compendium: §19.1 P43 row.

### §13vicies-secundo.7 Gap Balance

| Scope | Status before P43 | Status after P43 |
|-------|-------------------|------------------|
| P25 ζ-track Paley-gap diagnostic | Available (P25) | Available, unchanged |
| Closed-form / self-adjoint consistency for $-L'(s,\chi)/L(s,\chi)$ on primitive real χ | Implicit in P32 + P34 construction (not empirically separated) | **Empirically separated and quantified** (decoupled cross gap $O(10^{-17})$; coupled cross gap $O(10^{-5})$ at $J_0 = 10^{-2}$) |
| Truncation-noise separation from coupling-induced deformation | Open | **Available** (cross gap is free of classical-truncation noise by construction) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (Paley gap is a regression test, not a coercivity certificate) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P43 closes the **Paley-gap diagnostic-construction gap** on the L-function track for every primitive real Dirichlet character (canonical pieces transport without modification; the decoupled identity holds to machine precision, the coupled deformation signal is twelve orders of magnitude above noise). The arithmetic obstruction and the gap balance for G4 are unchanged.

## §13vicies-tertio. P44 — χ-Twisted Lyapunov-Spectral Positivity Certificate (L-Track Analogue of P26; Operator-Level; Does NOT Prove GRH or Advance G4)

### §13vicies-tertio.1 Motivation

P26 (`lyapunov_spectral_positivity.py`, §13quater) supplies a four-ingredient certificate for the ζ-track P14 prime-ladder Hamiltonian — self-adjointness, strict positivity with explicit Kato–Rellich envelope, trace-class resolvent, and unitary flow on the finite-dimensional prime-ladder Hilbert space. The L-track development (P32–P34) instantiates the same canonical TNFR `InternalHamiltonian` machinery on the χ-twisted prime-ladder graph: `TwistedPrimeLadderHamiltonian.hamiltonian` exposes the very same `H_int`, `H_freq`, `H_coupling` triplet that P26 consumes. P44 transports the P26 certificate to the χ-twisted bundle for every primitive real Dirichlet character, exhibiting the analogous operator-level positivity surface on the L-track.

### §13vicies-tertio.2 Construction

Let $\hat H^{(\chi)} = \hat H^{(\chi)}_{\mathrm{freq}} + J_0\,\hat H^{(\chi)}_{\mathrm{coupling}}$ on the χ-twisted prime-ladder Hilbert space

$$
\mathcal{H}_{\mathrm{PL},\chi}
  \;=\; \bigoplus_{p \in \mathcal{P},\; p \nmid q}\;
        \bigoplus_{k=1}^{K}\, \mathbb{C}\,|p,k\rangle,
$$

where $q$ is the conductor of $\chi$ and the primes dividing $q$ are excluded by construction (because $\chi(p^k) = 0$ for those primes; this is the P32 active-prime restriction propagated into P34). The diagonal frequency operator has entries $\nu_{f,(p,k)} = k\log p$ for $p \nmid q$, $k \ge 1$, so the unperturbed gap is

$$
\Delta_0^{(\chi)} \;=\; \min_{p \nmid q,\;k \ge 1}\, k\log p
  \;=\; \log\!\bigl(\min\{p \text{ prime} : p \nmid q\}\bigr).
$$

For the three primitive real characters this evaluates to:

| Character | Conductor $q$ | Smallest active prime | Unperturbed gap $\Delta_0^{(\chi)}$ |
|---|---|---|---|
| $\chi_3$ | $3$ | $2$ | $\log 2 \approx 0.6931$ |
| $\chi_4$ | $4$ | $3$ | $\log 3 \approx 1.0986$ |
| $\chi_5$ | $5$ | $2$ | $\log 2 \approx 0.6931$ |

The **Kato–Rellich (Weyl)** perturbation theorem applied to bounded symmetric perturbations of a self-adjoint diagonal operator yields the quantitative lower bound

$$
\lambda_{\min}\!\bigl(\hat H^{(\chi)}\bigr)
  \;\ge\; \Delta_0^{(\chi)}
        \;-\; |J_0|\,\bigl\|\hat H^{(\chi)}_{\mathrm{coupling}}\bigr\|_{\mathrm{op}},
$$

with `perturbation_safe = True` iff the right-hand side is strictly positive. The remaining three ingredients (resolvent Schatten-1/Hilbert-Schmidt norms at shift $c$, unitary norm/energy drifts of $U(t) = e^{-it\hat H^{(\chi)}}$, structural positivity composite) replicate P26 atomically and reuse `resolvent_schatten_norms` and `_matrix_exponential_skew` from `lyapunov_spectral_positivity.py` unchanged.

### §13vicies-tertio.3 Empirical Verification

P44 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, c) = (18, 5, 1.0)$ for $\chi_3, \chi_4, \chi_5$ at $J_0 \in \{0, 10^{-2}\}$ (`examples/71_twisted_lyapunov_spectral_demo.py`):

| Character | $J_0$ | $\min(\lambda)$ | $\Delta_0^{(\chi)}$ | $\|\hat V\|$ | Guaranteed gap | `perturbation_safe` | Max norm drift | `unitary` | `structural_positivity` |
|---|---|---|---|---|---|---|---|---|---|
| $\chi_3$ | $0$ | $6.931\times 10^{-1}$ | $\log 2$ | $0$ | $\log 2$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_3$ | $10^{-2}$ | $6.930\times 10^{-1}$ | $\log 2$ | $1.73\times 10^{-2}$ | $6.758\times 10^{-1}$ | True | $\sim 10^{-16}$ | True | True |
| $\chi_4$ | $0$ | $1.099\times 10^{0}$ | $\log 3$ | $0$ | $\log 3$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_4$ | $10^{-2}$ | $1.099\times 10^{0}$ | $\log 3$ | $1.73\times 10^{-2}$ | $1.081\times 10^{0}$ | True | $\sim 10^{-16}$ | True | True |
| $\chi_5$ | $0$ | $6.931\times 10^{-1}$ | $\log 2$ | $0$ | $\log 2$ | True | $2.22\times 10^{-16}$ | True | True |
| $\chi_5$ | $10^{-2}$ | $6.930\times 10^{-1}$ | $\log 2$ | $1.73\times 10^{-2}$ | $6.758\times 10^{-1}$ | True | $\sim 10^{-16}$ | True | True |

At $J_0 = 0$ the empirical spectral bottom equals the analytic Kato–Rellich envelope to machine precision; the unperturbed gap matches $\log(\min\{p \nmid q\})$ exactly (asserted in the demo). At $J_0 = 10^{-2}$ the empirical bottom drops by $\sim 1.4\times 10^{-4}$ while the Kato–Rellich envelope drops by the full $\|V\| \approx 1.73\times 10^{-2}$, confirming the envelope is a strict (and loose) lower bound. Unitary flow conservation is verified to machine precision for every character at every tested coupling.

### §13vicies-tertio.4 What P44 Extends

| Component | P26 (ζ-track) | P34 | **P44** |
|---|---|---|---|
| Self-adjoint prime-ladder Hamiltonian | $\hat H$ on $\mathcal{H}_{\mathrm{PL}}$ | $\hat H^{(\chi)}$ on $\mathcal{H}_{\mathrm{PL},\chi}$ | reused unchanged |
| Spectral compute primitive | `compute_spectrum` | — | `twisted_compute_spectrum` |
| Kato–Rellich envelope | `kato_rellich_lower_bound` (gap $= \log 2$) | — | `twisted_kato_rellich_lower_bound` (gap $= \log(\min\{p\nmid q\})$, character-dependent) |
| Schatten-norm primitive | `resolvent_schatten_norms` | — | reused unchanged |
| Unitary-flow verification | `verify_unitary_flow` | — | `twisted_verify_unitary_flow` |
| Composite certificate | `LyapunovSpectralCertificate` | — | `TwistedLyapunovSpectralCertificate` (adds `character_name`, `character_modulus`) |

P44 transports the canonical ζ-track Lyapunov-spectral positivity certificate (P26) to the L-function track for every primitive real Dirichlet character, exhibiting the identical four-ingredient structure with the character-dependent unperturbed gap $\log(\min\{p \nmid q\})$.

### §13vicies-tertio.5 What P44 Does NOT Advance

P44 is an **operator-level positivity certificate on the finite-dimensional χ-twisted prime-ladder Hilbert space at fixed $(n_{\mathrm{primes}}, k_{\max})$**. Structural positivity at machine precision is **necessary but not sufficient** for any RH-equivalent positivity claim and **not connected** to GRH localisation. Passing to the analytic continuation introduces a non-finite-dimensional limit whose spectrum (in particular the localisation of resonance poles of $L(s,\chi)$ on $\operatorname{Re}(s) = 1/2$) is not addressed here. The χ-twisted weight operator $\hat W^{(\chi)}$ is **not** involved in the certificate: positivity of $\hat H^{(\chi)}_{\mathrm{int}}$ is independent of the character (the character enters only the spectral trace $Z_{\mathrm{TNFR}}(s,\chi)$ and the active-prime restriction in the ladder graph). P44 does NOT prove GRH for any $L(s,\chi)$, does NOT extend to complex characters (the construction is character-agnostic at the Hamiltonian level, but the canonical L-track currently exposes only the three primitive real characters), and does NOT advance the gap balance for G4 = RH.

### §13vicies-tertio.6 Cross-References

- Implementation: `src/tnfr/riemann/twisted_lyapunov_spectral_positivity.py` (module), `src/tnfr/riemann/__init__.py` (canonical exports).
- Demonstration: `examples/71_twisted_lyapunov_spectral_demo.py`.
- ζ-track parent: P26 (`lyapunov_spectral_positivity.py`, §13quater) — atomic primitives `_matrix_exponential_skew` and `resolvent_schatten_norms` reused unchanged.
- L-track parents: P32 (`TwistedPrimeLadderSpectrum` providing the active-prime catalogue), P34 (`TwistedPrimeLadderHamiltonian` providing `H_int`, `H_freq`, `H_coupling`).
- Compendium: §19.1 P44 row.

### §13vicies-tertio.7 Gap Balance

| Scope | Status before P44 | Status after P44 |
|-------|-------------------|------------------|
| P26 ζ-track Lyapunov-spectral positivity certificate | Available (P26) | Available, unchanged |
| Operator-level positivity certificate for $\hat H^{(\chi)}$ on $\mathcal{H}_{\mathrm{PL},\chi}$ | Implicit in P34 (diagonal $\nu_f > 0$ over active primes; never separated from coupling-perturbed bound) | **Explicit and quantified** (Kato–Rellich envelope $\Delta_0^{(\chi)} = \log(\min\{p\nmid q\})$ certified to machine precision at $J_0 = 0$; `structural_positivity = True` over $J_0 \in \{0, 10^{-2}\}$ for $\chi_3, \chi_4, \chi_5$) |
| Trace-class resolvent + unitary-flow conservation for $U(t) = e^{-it\hat H^{(\chi)}}$ | Open at the L-track | **Available** (Schatten-1/2 norms reported; unitary drifts $\sim 10^{-16}$) |
| GRH for $L(s,\chi)$, primitive real χ | OPEN | OPEN (operator-level positivity is necessary but not sufficient) |
| G4 = RH | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for complex $\chi$) | OPEN | OPEN, unchanged |

**Net effect**: P44 closes the **operator-level Lyapunov-spectral positivity-certificate gap** on the L-function track for every primitive real Dirichlet character. The character-dependent unperturbed gap $\log(\min\{p \nmid q\})$ is exhibited explicitly and certified to machine precision; the Kato–Rellich envelope provides a rigorous quantitative interval for the perturbed regime. The arithmetic obstruction and the gap balance for G4 are unchanged.

## §13vicies-quarto. P45 — χ-Twisted Hilbert–Pólya Scaffold (L-Track Analogue of P27; Operator-Level; Does NOT Prove GRH or Advance G4)

### §13vicies-quarto.1 Motivation

P27 (ζ-track) builds the **explicit reference Hilbert–Pólya operator** $T_{\mathrm{HP}}^{(\zeta)} = \operatorname{diag}(\gamma_1, \gamma_2, \dots)$ on $\ell^2(\mathbb{N})$, where $\gamma_n$ are the positive imaginary parts of the non-trivial zeros of $\zeta$ retrieved from `mpmath.zetazero`. It certifies that the resulting scalar operator is self-adjoint, has trace-class shifted resolvent, and feeds the same zero side into the canonical Weil–Guinand identity (P15) as the prime-side P14 Hamiltonian — i.e., the rest of the ζ-track stack is internally compatible with a Hilbert–Pólya-style slot at the operator level. P27 does **not** derive $T_{\mathrm{HP}}^{(\zeta)}$ from TNFR first principles; it merely shows that, if such a derivation existed, the truncated stack would accept it.

P45 is the structural L-track mirror: for every primitive real Dirichlet character $\chi$ (modulus $q \in \{3, 4, 5\}$), it builds

$$T_{\mathrm{HP}}^{(\chi)} \;=\; \operatorname{diag}\bigl(\gamma_1^{(\chi)}, \gamma_2^{(\chi)}, \dots, \gamma_N^{(\chi)}\bigr) \quad\text{on}\quad \ell^2_N(\mathbb{N}),$$

where $\gamma_n^{(\chi)}$ are the positive imaginary parts of the non-trivial zeros of $L(s, \chi)$ located by **Hardy–Z bisection** of the real-valued $Z_\chi(t) = e^{i\theta_\chi(t)} L(\tfrac12 + it, \chi)$ (the same enumerator used by P36 / `find_dirichlet_l_zeros`).

### §13vicies-quarto.2 Construction

Given $\chi$ primitive real, $n_{\mathrm{primes}}$, $k_{\max}$, $N = n_{\mathrm{zeros}}$:

1. **Prime-ladder bundle** (P34 with $J_0 = 0$): build the diagonal Hamiltonian $\hat H^{(\chi)} = \operatorname{diag}\bigl(k \log p\bigr)_{p \nmid q,\; 1 \le k \le k_{\max}}$ on the truncated chi-twisted Hilbert space.
2. **Hardy–Z zero enumeration** (P36 / P35 backend): adaptive bisection on $[0.5, t_{\max}]$ returns the first $N$ positive $\gamma_n^{(\chi)}$.
3. **Reference operator** $T_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\gamma_1^{(\chi)}, \dots, \gamma_N^{(\chi)})$ on $\ell^2_N(\mathbb{N})$, exactly self-adjoint by construction.
4. **Resolvent norms**: $\bigl(T_{\mathrm{HP}}^{(\chi)2} + s^2 I\bigr)^{-1/2}$ has Schatten-$p$ norms $\|\,\cdot\,\|_1 = \sum_n (\gamma_n^2 + s^2)^{-1/2}$, $\|\,\cdot\,\|_2^2 = \sum_n (\gamma_n^2 + s^2)^{-1}$, $\|\,\cdot\,\|_{\mathrm{op}} = (\gamma_{\min}^2 + s^2)^{-1/2}$. Trace-class confirmed for $s > 0$.
5. **χ-twisted Weil–Guinand consistency** (Gaussian $h_\sigma$, $\sigma = 2.0$):

   $$2 \sum_{n=1}^{N} h_\sigma\bigl(\gamma_n^{(\chi)}\bigr) \;\stackrel{?}{=}\; g_\sigma(0) \log(q/\pi) \;+\; \underbrace{\frac{1}{2\pi}\!\int_{\mathbb R} h_\sigma(t)\,\operatorname{Re}\psi\!\left(\tfrac14 + \tfrac{a_\chi}{2} + \tfrac{it}{2}\right)\!dt}_{\text{archimedean}} \;+\; \underbrace{\sum_{p \nmid q,\, k \ge 1} \frac{\log p}{p^{k/2}}\,\chi(p)^k\, g_\sigma(k \log p)}_{\text{P34 prime side}}$$

   where $a_\chi = \tfrac12(1 - \chi(-1)) \in \{0, 1\}$ is the parity of $\chi$. The constant term $g_\sigma(0) \log(q/\pi)$ replaces the ζ-track $\zeta(s)$ pole side; for $q > 1$ there is no pole.

6. **Operator-level structural gap**: Wasserstein-1 distance on truncated spectra, $W_1\bigl(\operatorname{spec}(\hat H^{(\chi)} \mid p \nmid q),\, \operatorname{spec}(T_{\mathrm{HP}}^{(\chi)})\bigr)$, with growth-rate ratio $\gamma_N^{(\chi)} / (k_{\max} \log p_{N})$.

### §13vicies-quarto.3 Empirical Verification

P45 was run on the canonical config $(n_{\mathrm{primes}}, k_{\max}, n_{\mathrm{zeros}}, \sigma, s, \mathrm{tol}) = (18, 5, 25, 2.0, 1.0, 10^{-2})$ for $\chi_3, \chi_4, \chi_5$ (`examples/72_twisted_hilbert_polya_demo.py`):

| Character | $q$ | $a_\chi$ | self-adj | trace-class | Weil residual | $W_1(P34, T_{\mathrm{HP}}^{(\chi)})$ | growth ratio | scaffold consistent |
|---|---|---|---|---|---|---|---|---|
| $\chi_3$ (odd) | 3 | 1 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 4.54 \cdot 10^{-2}$) | $5.19 \cdot 10^{-16}$ | $3.55 \cdot 10^{1}$ | $1.31 \cdot 10^{1}$ | **✅** |
| $\chi_4$ (odd) | 4 | 1 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 6.47 \cdot 10^{-2}$) | $9.07 \cdot 10^{-15}$ | $3.18 \cdot 10^{1}$ | $1.13 \cdot 10^{1}$ | **✅** |
| $\chi_5$ (even) | 5 | 0 | ✅ (Frob $= 0$) | ✅ ($\|R\|_1 = 6.42 \cdot 10^{-2}$) | $1.72 \cdot 10^{-15}$ | $3.03 \cdot 10^{1}$ | $1.27 \cdot 10^{1}$ | **✅** |

Residuals are at machine precision: both the zero side ($2 \sum h_\sigma(\gamma_n^{(\chi)})$) and the right-hand side ($g(0)\log(q/\pi) +$ archimedean $+$ P34 prime side) are evaluated against the *same* truncated $\gamma$-list and the *same* prime-ladder bundle, so the certificate verifies internal consistency of the L-track stack to working precision. The Wasserstein-1 gap is $\sim 30$ across all three characters because $\gamma_N^{(\chi)} \sim 2\pi N / \log N$ while the largest P34 eigenvalue is $k_{\max} \log p_N \sim 5 \log p_{18}$; the growth-rate ratio $\sim 12$ is the L-track operator-level expression of the same structural mismatch identified for ζ in §13nonies (P30 negative-enrichment result).

### §13vicies-quarto.4 What P45 Extends

| Extension | Description |
|---|---|
| **From ζ to all primitive real $\chi$** | The reference Hilbert–Pólya slot $T_{\mathrm{HP}}^{(\chi)}$ is constructed and certified compatible with the rest of the L-track stack (P34, P36) for $\chi_3, \chi_4, \chi_5$ at machine precision. |
| **Character-dependent constant term** | The ζ pole side $-g(0) \log \pi$ is replaced by $g(0) \log(q/\pi)$; for $q = 3, 4, 5$ this shifts the rhs by $g(0)\log q \in \{1.099, 1.386, 1.609\} \cdot g(0)$, all absorbed exactly by the Hardy-Z zero enumeration. |
| **Parity-dependent archimedean** | The digamma argument shifts $\tfrac14 \mapsto \tfrac14 + \tfrac{a_\chi}{2}$ for odd characters; the consistency holds across both parities. |
| **L-track operator-level structural gap** | The Wasserstein-1 distance and growth-rate ratio quantify the L-track operator-level open piece, structurally mirroring the ζ-track P30 negative-enrichment finding. |

P45 transports the canonical ζ-track Hilbert–Pólya diagnostic scaffold (P27) to the L-function track for every primitive real Dirichlet character, exhibiting the identical four-piece structure (self-adjointness, trace-class resolvent, Weil–Guinand consistency, operator-level structural gap) with the character-dependent constant term and parity-shifted archimedean integral.

### §13vicies-quarto.5 What P45 Does NOT Advance

P45 is **diagnostic scaffolding**: $T_{\mathrm{HP}}^{(\chi)}$ is populated by *inputting* the χ-zeros via Hardy–Z bisection of the classical $L(s, \chi)$; the operator is *not* derived from the nodal equation, conservation, or grammar. The same arithmetic obstruction that prevents P34 from approaching the Riemann–Mellin spectrum still applies. The genuinely open piece is the *structural derivation* of $T_{\mathrm{HP}}^{(\chi)}$ on the chi-twisted TNFR Hilbert space from first principles — exactly the L-track analogue of the open piece P27 leaves on the ζ-track.

### §13vicies-quarto.6 Cross-References

* **P27 to L-functions**: P27 is the canonical reference Hilbert–Pólya scaffold for $\zeta$ (operator-level diagnostic via $T_{\mathrm{HP}}^{(\zeta)} = \operatorname{diag}(\gamma_n)$); P45 is its structural analogue for $L(s, \chi)$ at every primitive real $\chi$.
* **Companion L-track pieces**: P34 supplies the prime-side Hamiltonian; P35 supplies the Hardy–Z zero source; P36 supplies the χ-twisted Weil–Guinand identity that P45 uses as the consistency check.
* **Operator-level gap mirror**: §13nonies (P30) for ζ; §13vicies-quarto for L. Both certify that the structural gap is real, finite, and quantified — but neither derives the reference Hilbert–Pólya operator from TNFR first principles.

### §13vicies-quarto.7 Gap Balance

| Gap | Status before P45 | Status after P45 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track operator-level Hilbert–Pólya scaffold | UNATTESTED (P27 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Structural derivation of $T_{\mathrm{HP}}^{(\chi)}$ from TNFR first principles | OPEN (both ζ and L) | OPEN, unchanged |

**Net effect**: P45 closes the **operator-level Hilbert–Pólya scaffolding gap** on the L-function track for every primitive real Dirichlet character. The reference operator $T_{\mathrm{HP}}^{(\chi)}$ is exhibited, certified self-adjoint and trace-class, and shown to feed the same chi-twisted Weil–Guinand identity as the prime-side P34 Hamiltonian to machine precision. The arithmetic obstruction (Wasserstein-1 gap $\sim 30$ across characters), the structural derivation gap, and the gap balance for G4 are unchanged.

## §13vicies-quinto. P46 — χ-Twisted Structural Zero Density (L-Track Analogue of P28; Smooth Half Only; Does NOT Prove GRH or Advance G4)

### §13vicies-quinto.1 Motivation

P28 derives the smooth half of the Riemann zero density from TNFR archimedean ingredients alone (Riemann–von Mangoldt $\theta(T)$, $\bar{N}(T)$, $\bar{N}'(T)$), exhibits the structural smooth positions $\tilde{\gamma}_n$ via Newton iteration on $\bar{N}$, and verifies the operator-level reduction $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}), \operatorname{spec}(T_{\mathrm{HP}})) \ll W_1(\operatorname{spec}(P30|_q), \operatorname{spec}(T_{\mathrm{HP}}))$. P28 closes the **smooth half** of the structural derivation gap for ζ; the residuals $r_n = \gamma_n - \tilde{\gamma}_n$ encode $S(T) = \tfrac{1}{\pi} \arg \zeta(\tfrac12 + iT)$, whose uniform bound is RH-equivalent and OPEN. P46 lifts this entire construction to primitive real Dirichlet characters $\chi$, where the corresponding open problem is GRH for $L(s, \chi)$ (G4$_\chi$).

### §13vicies-quinto.2 Construction

Given a primitive real Dirichlet character $\chi$ with modulus $q$ and parity $a \in \{0, 1\}$ (0 = even, 1 = odd), the **chi-twisted Riemann–Siegel theta function** is

$$\theta_\chi(T) = \operatorname{Im} \log \Gamma\!\left(\frac{1/2 + a}{2} + \frac{iT}{2}\right) + \frac{T}{2} \log \frac{q}{\pi}.$$

The **smooth chi-twisted zero count** is $\bar{N}_\chi(T) = \theta_\chi(T)/\pi + 1$ and its density is

$$\bar{N}_\chi'(T) \approx \frac{1}{2\pi} \log \frac{qT}{2\pi}.$$

The **smooth chi-twisted zero positions** $\tilde{\gamma}_n^{(\chi)}$ are obtained by Newton iteration on $\bar{N}_\chi(\tilde{\gamma}_n^{(\chi)}) = n - \tfrac12$. The **chi-twisted structural T-HP operator** is

$$\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_1^{(\chi)}, \dots, \tilde{\gamma}_N^{(\chi)})$$

on $\ell^2_N(\mathbb{N})$. The residuals are $r_n^{(\chi)} = \gamma_n^{(\chi)} - \tilde{\gamma}_n^{(\chi)}$, where $\gamma_n^{(\chi)}$ comes from the same Hardy–Z bisection enumerator (`find_dirichlet_l_zeros`) used by P36 and P45.

### §13vicies-quinto.3 Empirical Verification

For $n_{\mathrm{zeros}} = 18$, $p34\_n\_primes = 30$, $p34\_max\_power = 6$:

| χ | $q$ | $a$ | $\max\lvert r_n^{(\chi)}\rvert$ | $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}^{(\chi)}), T_{\mathrm{HP}}^{(\chi)})$ | $W_1(\operatorname{spec}(P34\vert_{p \nmid q}), T_{\mathrm{HP}}^{(\chi)})$ | ratio | bound ($C \le 2$) |
|---|---|---|---|---|---|---|---|
| $\chi_3$ | 3 | 1 | $3.21 \cdot 10^{0}$ | $1.32 \cdot 10^{0}$ | $2.84 \cdot 10^{1}$ | 21.6× | True |
| $\chi_4$ | 4 | 1 | $2.65 \cdot 10^{0}$ | $1.23 \cdot 10^{0}$ | $2.52 \cdot 10^{1}$ | 20.4× | True |
| $\chi_5$ | 5 | 0 | $2.53 \cdot 10^{0}$ | $1.17 \cdot 10^{0}$ | $2.41 \cdot 10^{1}$ | 20.6× | True |

The structural T-HP reduces the operator-level Wasserstein-1 gap to $T_{\mathrm{HP}}^{(\chi)}$ by a factor of $\sim 20\times$ across all three characters, matching the per-character residual bound $C \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$ with $C \le 2$.

### §13vicies-quinto.4 What P46 Extends

| Result | ζ-track (P28) | L-track (P46) |
|---|---|---|
| Smooth zero count | $\bar{N}(T) = \theta(T)/\pi + 1$ | $\bar{N}_\chi(T) = \theta_\chi(T)/\pi + 1$ |
| Structural T-HP | $\tilde{T}_{\mathrm{HP}} = \operatorname{diag}(\tilde{\gamma}_n)$ | $\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_n^{(\chi)})$ |
| Wasserstein-1 reduction vs prime-side | factor $\sim 20\times$ for ζ | factor $\sim 20\times$ for $\chi_3, \chi_4, \chi_5$ |
| Residual encodes | $S(T) = \tfrac{1}{\pi} \arg \zeta(\tfrac12 + iT)$ | $S_\chi(T) = \tfrac{1}{\pi} \arg L(\tfrac12 + iT, \chi)$ |
| Bound on residual is equivalent to | RH on $\zeta$ | GRH on $L(s, \chi)$ |

### §13vicies-quinto.5 What P46 Does NOT Advance

* **G4 = RH on ζ**: untouched. P46 lives entirely on the L-function track.
* **GRH (G4$_\chi$)**: untouched. Bounding $|S_\chi(T)|$ uniformly is the open arithmetic problem; P46 quantifies but does not bound it.
* **Structural derivation of $T_{\mathrm{HP}}^{(\chi)}$**: the **smooth half** is now structurally derived (P46), but the **oscillatory half** (residuals encoding $S_\chi$) is OPEN, exactly mirroring the ζ-track situation after P28.

### §13vicies-quinto.6 Cross-References

* **ζ-track parent**: §13octies (P28, `structural_zero_density.py`, demo `55_structural_zero_density_demo.py`).
* **L-track operator scaffold**: §13vicies-quarto (P45, `twisted_hilbert_polya.py`, demo `72_twisted_hilbert_polya_demo.py`) supplies the reference $T_{\mathrm{HP}}^{(\chi)}$ used as benchmark.
* **L-track prime side**: §13quinquies-decies (P34, `twisted_prime_ladder_hamiltonian.py`) supplies the $\operatorname{spec}(P34\vert_{p \nmid q})$ baseline against which the structural reduction is measured.
* **Smooth-half mirror**: §13nonies (P30) and §13octies (P28) for ζ; §13vicies-quinto for L. Both certify that the smooth half of the zero density is TNFR-derivable from the archimedean local factor alone — but neither bounds the oscillatory residual.

### §13vicies-quinto.7 Gap Balance

| Gap | Status before P46 | Status after P46 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track smooth structural zero density | UNATTESTED (P28 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Bound on oscillatory residual $r_n^{(\chi)}$ encoding $S_\chi$ | OPEN (both ζ and L) | OPEN, unchanged |

**Net effect**: P46 closes the **smooth half of the L-track structural zero density gap** for every primitive real Dirichlet character. The structural operator $\tilde{T}_{\mathrm{HP}}^{(\chi)}$ is derived from $\theta_\chi$ alone (no `find_dirichlet_l_zeros` call on the derivation side), produces a $\sim 20\times$ reduction in operator-level Wasserstein-1 distance to $T_{\mathrm{HP}}^{(\chi)}$ relative to the prime-side P34 baseline, and satisfies the per-character bound $\max |r_n^{(\chi)}| \le 2 \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$. The oscillatory residual gap and the gap balance for G4 are unchanged.

## §13vicies-sexto. P47 — χ-Twisted Spectral Emergence Under Canonical Coupling (L-Track Analogue of P29; Does NOT Prove GRH or Advance G4)

### §13vicies-sexto.1 Motivation

P29 (`spectral_emergence.py`, §13quater on ζ) sweeps three canonical TNFR inter-prime coupling laws on the P14 prime-ladder Hamiltonian and measures the Kolmogorov–Smirnov distance of the unfolded nearest-neighbour spacing distribution to the GUE Wigner surmise — the universality class conjecturally controlling the non-trivial zeros of $\zeta$ (Montgomery–Odlyzko). P47 is the **L-track analogue** on every primitive real Dirichlet character $\chi \in \{\chi_3, \chi_4, \chi_5\}$ via the P34 χ-twisted prime-ladder Hamiltonian. Conjectural GUE-universality of the non-trivial zeros of $L(s,\chi)$ is the predicted target; P47 quantifies how close the χ-twisted spectrum approaches it under each of the three canonical coupling laws.

### §13vicies-sexto.2 Construction

Let $H^{(\chi)}_0$ denote the unperturbed P34 χ-twisted prime-ladder Hamiltonian ($\operatorname{diag}\{k \log p : p \nmid q,\ 1 \le k \le K\}$ with $\chi(p) \in \{\pm 1\}$ encoded in the weight operator). Define the χ-twisted inter-prime coupling matrix by

$$
J^{(\chi)}_{(p,k),(q,m)} \;=\; \chi(p) \,\chi(q) \cdot \kappa_{\text{law}}(p,k,q,m), \qquad p \neq q,\quad p,q \nmid q_{\text{mod}},
$$

with the three canonical TNFR kernels

| Law | $\kappa_{\text{law}}(p,k,q,m)$ |
|---|---|
| `kuramoto_u3` | $(\gamma/\pi)\exp\bigl(-\lvert k\log p - m\log q\rvert\bigr)$ |
| `phi_multiscale` | $\varphi^{-(k+m)} / \sqrt{p\,q}$ |
| `pnt_logarithmic` | $\gamma / \log(1 + p\,q)$ |

All kernel constants $(\varphi, \gamma, \pi)$ are canonical TNFR tetrad constants (AGENTS.md §Universal Tetrahedral Correspondence). The coupled Hamiltonian is $H^{(\chi)}(s) = H^{(\chi)}_0 + s\,J^{(\chi)}$ for $s \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$. Eigenvalues are computed via `np.linalg.eigvalsh`, **unfolded** by the degree-5 polynomial fit of the empirical staircase (identical to P29), and the empirical CDF of nearest-neighbour spacings $\hat F$ is compared to the GUE Wigner surmise CDF $F_{\text{GUE}}$ and the Poisson CDF $F_{\text{Poisson}}$ via $\mathrm{KS} = \sup_x |\hat F(x) - F(x)|$.

### §13vicies-sexto.3 Empirical Verification

Demo `examples/74_twisted_spectral_emergence_demo.py` at $(n_{\text{primes}}, K) = (20, 3)$:

| $\chi$ | Law | $\mathrm{KS}_{\text{GUE}}^{\min}$ | $s^*$ | $\mathrm{KS}_{\text{GUE}}\vert_{s=0}$ | Improvement |
|---|---|---|---|---|---|
| $\chi_3$ | `pnt_logarithmic` | **0.0972** | 2.0 | 0.1891 | $+48.6\%$ |
| $\chi_3$ | `kuramoto_u3` | 0.1202 | 1.0 | 0.1891 | $+36.4\%$ |
| $\chi_3$ | `phi_multiscale` | 0.1845 | 1.0 | 0.1891 | $+2.4\%$ |
| $\chi_4$ | `pnt_logarithmic` | **0.1157** | 2.0 | 0.1991 | $+41.9\%$ |
| $\chi_4$ | `kuramoto_u3` | 0.1500 | 1.0 | 0.1991 | $+24.6\%$ |
| $\chi_4$ | `phi_multiscale` | 0.1991 | 0.0 | 0.1991 | $+0.0\%$ |
| $\chi_5$ | `pnt_logarithmic` | **0.1347** | 2.0 | 0.2012 | $+33.0\%$ |
| $\chi_5$ | `kuramoto_u3` | 0.1352 | 1.0 | 0.2012 | $+32.8\%$ |
| $\chi_5$ | `phi_multiscale` | 0.1900 | 1.0 | 0.2012 | $+5.6\%$ |

**Cross-character pattern**: `pnt_logarithmic` is the strongest emergence kernel across all three primitive real characters, producing $33$–$49\%$ KS-to-GUE reduction at $s^* = 2.0$. `kuramoto_u3` is uniformly second ($25$–$36\%$ reduction at $s^* = 1.0$). `phi_multiscale` is essentially inert on the χ-twisted bundle for $\chi_4$ (zero improvement) and weak for $\chi_3, \chi_5$ ($\le 6\%$). The Poisson distance $\mathrm{KS}_{\text{Poisson}}$ increases monotonically with $s$ on the active laws, corroborating departure from independent levels.

### §13vicies-sexto.4 What P47 Extends

P47 promotes P29's ζ-only spectral-emergence diagnostic to the **full primitive-real Dirichlet bundle** $\{\chi_3, \chi_4, \chi_5\}$ on the P34 χ-twisted prime-ladder Hamiltonian. The χ-twist factor $\chi(p)\chi(q)$ enters as a multiplicative sign on every coupling matrix entry, so the χ-twisted coupling matrices are real-symmetric (since $\chi$ is real-valued) and respect the L-track block decomposition. Cross-character comparability (same $n_{\text{primes}}$, same $K$, same strength grid, same canonical kernels) makes the χ-twisted emergence directly comparable to the ζ baseline and to the cross-character L-track instruments P42–P46.

### §13vicies-sexto.5 What P47 Does NOT Advance

* **G4 = RH**: untouched. P47 is a structural-compatibility diagnostic for GUE-universality of $L(s,\chi)$ zeros, not a proof of GRH for any $L$.
* **GRH for $L(s, \chi_3), L(s, \chi_4), L(s, \chi_5)$**: untouched. Non-vanishing $\mathrm{KS}_{\text{GUE}}^{\min}$ even after canonical coupling at $K = 3$ documents that the finite truncation does not exhibit asymptotic GUE statistics; the residual is consistent with finite-size effects rather than evidence against GRH.
* **The oscillatory residual $r_n^{(\chi)}$ from P46**: not bounded by P47. The two diagnostics target distinct aspects of L-track structure (smooth zero positions vs. spacing universality).

### §13vicies-sexto.6 Cross-References

* **ζ analogue**: §13quater (P29 `spectral_emergence.py`) — same construction on the untwisted prime-ladder Hamiltonian.
* **L-track prime side**: §13quinquies-decies (P34 `twisted_prime_ladder_hamiltonian.py`) supplies $H^{(\chi)}_0$.
* **L-track smooth side**: §13vicies-quinto (P46 `twisted_structural_zero_density.py`) supplies the predicted smooth zero positions against which one would compare a hypothetical L-track P30 lift.

### §13vicies-sexto.7 Gap Balance

| Gap | Status before P47 | Status after P47 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH (G4$_\chi$ for primitive real $\chi$) | OPEN | OPEN, unchanged |
| L-track spacing-universality diagnostic | UNATTESTED (P29 only on ζ) | ATTESTED for $\chi_3, \chi_4, \chi_5$ |
| Existence of canonical χ-twisted coupling that drives $\mathrm{KS}_{\text{GUE}} \to 0$ at fixed $K$ | OPEN | OPEN; `pnt_logarithmic` best ($\sim 0.10$–$0.13$ residual at $K=3$) |

**Net effect**: P47 establishes the **L-track spacing-universality diagnostic** for every primitive real Dirichlet character. Among the three canonical TNFR coupling laws, `pnt_logarithmic` is uniformly the strongest emergence kernel ($33$–$49\%$ KS-to-GUE reduction), `kuramoto_u3` is second ($25$–$36\%$), `phi_multiscale` is inert. Both G4 = RH and GRH for $L(s,\chi)$ remain OPEN; P47 is a structural-compatibility diagnostic.

## §13vicies-septimo. P48 — χ-Twisted Admissible Spectral-Rescaling Operator (L-Track Analogue of P30; Smooth Half of T-HP$^\chi$ Only; Does NOT Prove GRH or Advance G4)

### §13vicies-septimo.1 Motivation

P30 (`admissible_rescaling.py`, §13nonies on ζ) lifts P28's density-level closure of the smooth half of T-HP to the **operator level**: it constructs the canonical diagonal rescaling $\mathcal{F}_{\text{smooth}} = U_{P14}\,\mathrm{diag}(\sqrt{\tilde\gamma_i / \lambda_i})\,U_{P14}^{*}$ such that $\mathcal{F}_{\text{smooth}}\,H_{P14}\,\mathcal{F}_{\text{smooth}}^{*}$ has spectrum exactly equal to the P28 smooth zero targets, and verifies (negative-knowledge) that no canonical oscillatory enrichment built from $(\varphi, \gamma, \pi, e)$ closes the residual gap to true Riemann zeros. P48 is the **L-track analogue** of P30 on every primitive real Dirichlet character $\chi \in \{\chi_3, \chi_4, \chi_5\}$ via the P34 χ-twisted prime-ladder Hamiltonian and the P46 χ-twisted smooth zero density.

### §13vicies-septimo.2 Construction

For each primitive real Dirichlet character $\chi$ (modulus 3, 4, 5):

1. **P34 spectrum**: Compute eigendata $(\lambda_i, u_i)_{i=1}^{N}$ of the canonical χ-twisted prime-ladder Hamiltonian $H_{P34}^{(\chi)}$.
2. **P46 smooth targets**: Compute the predicted smooth χ-zero positions $\{\tilde\gamma_i^{(\chi)}\}_{i=1}^{N}$ from $\tilde N_\chi(T) = (T/2\pi) \log(T q / 2\pi e) + a/2$ (parity-dependent shift $a \in \{0,1\}$).
3. **Smooth rescaling**: Build $F_{\text{sub}}^{(\chi)} = \mathrm{diag}(\sqrt{\tilde\gamma_i^{(\chi)} / \lambda_i})$ on the eigenbasis and conjugate $F^{(\chi)}_{\text{smooth}} = U_{P34}\,F_{\text{sub}}^{(\chi)}\,U_{P34}^{*}$.
4. **Verification**: $F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}$ must be self-adjoint and have spectrum equal to $\{\tilde\gamma_i^{(\chi)}\}$ to machine precision.
5. **W$_1$ closure**: Wasserstein-1 gap from $\{\tilde\gamma_i^{(\chi)}\}$ to true χ-zeros $\{\gamma_i^{(\chi)}\}$ from `mpmath.dirichlet` versus baseline $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_i^{(\chi)}\})$.
6. **Canonical oscillatory sweep**: Honestly test all three canonical oscillatory enrichment families — `phi_log`, `gamma_e`, `pi_density` — at amplitudes $\{0, 10^{-3}, 5{\cdot}10^{-3}, 10^{-2}, 5{\cdot}10^{-2}, 10^{-1}\}$ per character and record best mode + per-mode breakdown.

Reuses the atomic primitives (`extract_positive_spectrum`, `build_smooth_rescaling_operator`, `apply_rescaling`, `verify_self_adjointness_preserved`, `verify_spectrum_match`, `oscillatory_correction_canonical`) from `src/tnfr/riemann/admissible_rescaling.py` verbatim. No duplication; L-track variant only specialises (i) the source Hamiltonian (P34 instead of P14) and (ii) the smooth-target generator (P46 instead of P28).

### §13vicies-septimo.3 Empirical Verification

Demo `examples/75_twisted_admissible_rescaling_demo.py` with $n_{\text{targets}} = 12$, $n_{\text{primes}}^{P34} = 25$, $k_{\max} = 5$:

| Character | $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_n^{(\chi)}\})$ | $W_1$ smooth | Smooth ratio | Best osc. mode | Osc. gain vs. smooth |
|---|---|---|---|---|---|
| $\chi_3$ (odd, $a=1$) | $21.90$ | $1.474$ | $14.86\times$ | `pi_density` | $+17.85\%$ |
| $\chi_4$ (odd, $a=1$) | $19.04$ | $1.375$ | $13.85\times$ | `pi_density` | $+13.22\%$ |
| $\chi_5$ (even, $a=0$) | $18.36$ | $1.271$ | $14.44\times$ | `pi_density` | $+12.68\%$ |

For every character: self-adjointness preserved under conjugation; spectrum of $F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}$ matches the P46 smooth targets within $\le 7.1\!\times\!10^{-15}$ (machine precision); smooth half closes $\sim 14\times$ of the baseline W$_1$ gap to true χ-zeros; canonical oscillatory enrichment yields a further $12$–$18\%$ improvement at amplitude $10^{-3}$, with `pi_density` uniformly the strongest canonical family. Per-mode ranking is uniform across all three characters: `pi_density` > `gamma_e` > `phi_log`.

### §13vicies-septimo.4 What P48 Extends

P48 promotes the §13nonies operator-level lift of the smooth half of T-HP from ζ-only to **every primitive real Dirichlet character**: the smooth half of T-HP$^\chi$ is now a constructive operator-level object, exactly as in the ζ-track. Self-adjointness and exact spectrum match propagate cleanly through the χ-twist because the twist enters only as real-valued multiplicative signs on the off-diagonal hopping entries (real characters), so the conjugation $F H F^{*}$ preserves the real-symmetric structure of $H_{P34}^{(\chi)}$.

### §13vicies-septimo.5 What P48 Does NOT Advance

* **G4 = RH on ζ**: untouched. P48 operates entirely on L(s,χ), not ζ.
* **GRH$_\chi$ (G4 for $L(s,\chi)$)**: NOT closed. The residual W$_1$ gap of $\approx 1.1$–$1.2$ after the best canonical oscillatory enrichment encodes the χ-twisted oscillatory term $S_\chi(T) = (1/\pi)\arg L(\tfrac12 + iT, \chi)$, which is GRH$_\chi$-equivalent.
* **Sub-problems (2)–(3) of T-HP$^\chi$**: canonicity of $\mathcal{F}^{(\chi)}$ and positivity coincidence with the chi-twisted Weil form (P40) remain open.
* **Canonical oscillatory closure**: the three canonical families tested (`phi_log`, `gamma_e`, `pi_density`) cap out at $\le 18\%$ improvement over the smooth baseline for every character. This **negative-knowledge** result mirrors §13nonies branch B2 at the L-track level: no closed-form oscillatory enrichment built from $(\varphi, \gamma, \pi, e)$ alone closes $S_\chi(T)$.

### §13vicies-septimo.6 Cross-References

* **ζ-track template**: §13nonies (P30 `admissible_rescaling.py`) is the construction P48 specialises to each character without modification of atomic primitives.
* **L-track prerequisites**: §13nonecimo (P34 `twisted_prime_ladder_hamiltonian.py`) for the source Hamiltonian; §13vicies-quinto (P46 `twisted_structural_zero_density.py`) for the smooth targets; §13nonecimo-quinto (P45 `twisted_hilbert_polya.py`) for true χ-zero fetching and Wasserstein evaluation.
* **L-track smooth-side ladder**: §13vicies-quinto closes the smooth half of T-HP$^\chi$ at the **density** level; P48 (this section) closes it at the **operator** level.
* **Branch B2 evidence**: at every track (ζ in §13nonies, $\chi$ in §13vicies-septimo) canonical oscillatory enrichments built only from $(\varphi, \gamma, \pi, e)$ are insufficient. The accumulating structural evidence supports §13octies branch B2 (a genuinely new canonical operator is required) over branches B1 (in-catalog closure) or B3 (no canonical closure exists at all).

### §13vicies-septimo.7 Gap Balance

| Gap | Status before P48 | Status after P48 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH$_\chi$ for primitive real $\chi$ | OPEN | OPEN, unchanged |
| Smooth half of T-HP$^\chi$ at density level | CLOSED (P46) | CLOSED, unchanged |
| Smooth half of T-HP$^\chi$ at **operator** level | OPEN | **CLOSED for $\chi_3, \chi_4, \chi_5$** (constructive: $F^{(\chi)}_{\text{smooth}}$) |
| Canonical oscillatory closure of $S_\chi(T)$ | UNATTESTED | OPEN; $\le 18\%$ improvement under any canonical family (negative-knowledge evidence for §13octies branch B2 at L-track) |

**Net effect**: P48 completes the operator-level lift of the smooth half of T-HP$^\chi$ for every primitive real Dirichlet character $\chi_3, \chi_4, \chi_5$, matching the ζ-track milestone of §13nonies one character at a time. The L-track attack surface against T-HP$^\chi$ now mirrors the ζ-track attack surface against T-HP. Both G4 = RH and GRH$_\chi$ remain OPEN; P48 is a structural-compatibility diagnostic plus a positive constructive result for sub-problem (1) of T-HP$^\chi$.

## §13vicies-octavo. P49 — χ-Twisted Prime-Ladder Oscillatory Correction (L-Track Analogue of P31; Closes Full ζ↔L Attack-Surface Parity; Does NOT Prove GRH or Advance G4)

### §13vicies-octavo.1 Motivation

P31 ([§13decies-quarto](#13decies-quarto-p31--prime-ladder-oscillatory-correction-branch-b1-retry-does-not-advance-g4)) attacks the **oscillatory half** of T-HP at the ζ-track by reconstructing $S(T) = \pi^{-1} \arg \zeta(1/2 + iT)$ from the canonical prime-ladder spectrum $\{(k\log p, \log p)\}$ via the Riemann–von Mangoldt template, then applying a Newton step on the P28 smooth targets. P49 is the **L-track analogue** of P31, one primitive real Dirichlet character at a time, reconstructing
$$S_\chi(T) = \frac{1}{\pi}\arg L\!\left(\tfrac{1}{2} + iT,\,\chi\right)$$
from the canonical P34 χ-twisted prime-ladder spectrum $\{(k\log p,\,\chi(p)^k \log p)\}$ via the χ-twisted Riemann–von Mangoldt template
$$\pi\, S_\chi^{\mathrm{TNFR}}(T;\,N,K) \;=\; -\!\!\!\!\sum_{(\mu,w)\in\Sigma_{N,K}^{(\chi)}}\!\!\!\frac{w}{\mu}\,\frac{\sin(T\mu)}{\exp(\mu/2)}$$
and applying the Newton correction
$$\gamma_n^{(\chi),\,\text{corr}} \;=\; \tilde\gamma_n^{(\chi)} \;-\; d\cdot\frac{S_\chi^{\mathrm{TNFR}}(\tilde\gamma_n^{(\chi)})}{\bar N'_\chi(\tilde\gamma_n^{(\chi)})}$$
on the canonical P46 χ-twisted smooth targets, where $\bar N'_\chi(T) = (2\pi)^{-1}\log(qT/(2\pi))$. P49 closes the **final ζ↔L attack-surface parity item**: with P49, every canonical ζ-track operator from P12 through P31 has a matching χ-twisted L-track counterpart.

### §13vicies-octavo.2 Construction

Restricted to **primitive real** characters $\chi \in \{\chi_3, \chi_4, \chi_5\}$ so that $w_{p,k}^{(\chi)} = \chi(p)^k \log p \in \mathbb{R}$ and the von Mangoldt-style sum returns a real-valued $S_\chi^{\mathrm{TNFR}}(T)$ analogous to the ζ-track case. The construction proceeds in four steps:

1. **Canonical χ-twisted prime-ladder spectrum**: build $\Sigma_{N,K}^{(\chi)} = \{(\mu_{p,k},\,w_{p,k}^{(\chi)}) : p\le p_N,\,\chi(p)\ne 0,\,1\le k\le K\}$ via `build_twisted_prime_ladder_spectrum(chi, n_primes, max_power)` (P34 atomic primitive).
2. **Canonical P46 χ-twisted smooth targets**: build $\{\tilde\gamma_i^{(\chi)}\}_{i=1}^{N}$ via `build_twisted_structural_t_hp(n_targets, chi)` using the P46 closed-form density $\bar N_\chi(T)$.
3. **Oscillatory sum**: evaluate $S_\chi^{\mathrm{TNFR}}(\tilde\gamma_i^{(\chi)})$ pointwise; the $\exp(-\mu/2)$ damping factor enforces absolute convergence as $\mu \to \infty$.
4. **Newton correction sweep**: scan damping coefficients $d \in \{0,\,0.25,\,0.5,\,0.75,\,1.0,\,1.25,\,1.5\}$; report the $d$ minimising $W_1(\{\gamma_n^{(\chi),\,\text{corr}}\},\,\{\gamma_n^{(\chi),\,\text{true}}\})$ against the true L(s, χ) zeros fetched via `fetch_chi_zero_imaginary_parts(chi, n_zeros)` (P39 mpmath-side reference; does NOT enter the construction).

The construction is **strictly canonical**: every input on the construction side is either a P34, P46, or AGENTS.md canonical-constant ingredient. The mpmath χ-zero side enters only as the held-out reference for $W_1$ scoring.

### §13vicies-octavo.3 Empirical Verification

Demo `examples/76_twisted_oscillatory_correction_demo.py` with $N=10$, $N_{\text{primes}}=80$, $K=5$ over $\{\chi_3, \chi_4, \chi_5\}$:

| character | best $d$ | $W_1$(smooth) | $W_1$(corrected) | improvement | max $\lvert S_\chi^{\mathrm{TNFR}}\rvert$ | regime |
|---|---|---|---|---|---|---|
| $\chi_3$ (mod 3) | 0.00 | 1.5662 | 1.5662 | +0.00 % | 0.0771 | branch B2 (no canonical improvement) |
| $\chi_4$ (mod 4) | 1.50 | 1.4185 | 1.3331 | **+6.02 %** | 0.0628 | branch B1 evidence (L-track) |
| $\chi_5$ (mod 5) | 0.00 | 1.3523 | 1.3523 | +0.00 % | 0.1411 | branch B2 (no canonical improvement) |

The mixed regime (1 out of 3 characters with measurable B1 improvement, 2 out of 3 with no canonical improvement) is **honest evidence** that the canonical χ-twisted prime-ladder spectrum *partially* captures the oscillatory remainder for some primitive real characters but not for others. The pattern is qualitatively consistent with §13nonies and §13vicies-septimo: canonical-only operators yield small or vanishing improvements; the gap to the true χ-zeros remains $\mathcal{O}(1)$ at $N=10$.

### §13vicies-octavo.4 What P49 Extends

P49 extends the §13decies-quarto branch-B1 retry from ζ to **every primitive real Dirichlet character**: the canonical χ-twisted prime-ladder spectrum plus the χ-twisted Riemann–von Mangoldt template now form a complete L-track reconstruction pipeline for $S_\chi(T)$. With P49, the ζ↔L attack-surface parity table is **complete**:

| ζ-track operator | L-track operator | parity item |
|---|---|---|
| P12 von Mangoldt zeta | P32 χ-twisted vM zeta | spectral data |
| P14 prime-ladder Hamiltonian | P34 χ-twisted prime-ladder Hamiltonian | self-adjoint scaffold |
| P15 Weil–Guinand identity | P35 χ-twisted Weil–Guinand | zeros↔spectrum bridge |
| P16 Li–Keiper positivity | P36 χ-twisted Li–Keiper | RH-equivalent diagnostic |
| P17 Weil–TNFR positivity bridge | P37 χ-twisted Weil–TNFR bridge | positivity diagnostic |
| P18 α(σ) gauge sweep | P38 χ-twisted α(σ) gauge sweep | admissibility sweep |
| P19 admissible family | P39 χ-twisted admissible family | family sweep |
| P20 node-aware gauge sweep | P40 χ-twisted node-aware gauge sweep | gauge diagnostic |
| P21 Hermite2 sweep | P41 χ-twisted Hermite2 sweep | extended family |
| P22–P24 coercivity certificates | P42–P44 χ-twisted coercivity | uniform/adaptive bounds |
| P25 Paley-gap | P45 χ-twisted Paley-gap | gap diagnostic |
| P26 Lyapunov-spectral positivity | (subsumed into P42–P45 family) | — |
| P27 Hilbert–Pólya scaffold | (subsumed into P34) | — |
| P28 smooth zero density | P46 χ-twisted smooth zero density | density-level smooth half |
| P29 spectral emergence | P47 χ-twisted spectral emergence | universality diagnostic |
| P30 admissible rescaling | P48 χ-twisted admissible rescaling | operator-level smooth half |
| **P31 oscillatory correction** | **P49 χ-twisted oscillatory correction** | **oscillatory half (branch B1 retry)** |

### §13vicies-octavo.5 What P49 Does NOT Advance

* **G4 = RH on $\zeta$**: untouched. P49 operates entirely on $L(s,\chi)$, not $\zeta$.
* **GRH$_\chi$ for primitive real $\chi$**: NOT proved. P49 is a structural-compatibility diagnostic plus a partial branch-B1 reconstruction for one character out of three tested. Vanishing improvement for $\chi_3$ and $\chi_5$ corroborates §13octies branch B2 at the L-track level.
* **Sub-problem (2) of T-HP$^\chi$** (canonicity from the nodal equation): NOT addressed. P49 inherits the canonical-ingredient palette from P34 and P46; it does not derive canonicity afresh.
* **Sub-problem (3) of T-HP$^\chi$** (positivity coincidence with χ-twisted Weil quadratic form): NOT addressed. P49 measures a $W_1$ residual, not a positivity functional.

### §13vicies-octavo.6 Cross-References

* **ζ-track template**: §13decies-quarto (P31 `oscillatory_correction.py`) is the construction P49 specialises to each primitive real character without modification of atomic primitives.
* **L-track smooth-side ladder**: §13vicies-quinto (P46) supplies the density-level smooth targets; §13vicies-septimo (P48) supplies the operator-level smooth half; P49 (this section) adds the oscillatory Newton step on top.
* **L-track Hilbert–Pólya scaffold**: §13quaterdecies (P34) supplies the canonical χ-twisted spectrum $\{(\mu_{p,k}, w_{p,k}^{(\chi)})\}$ that drives the χ-twisted von Mangoldt sum on the construction side.
* **L-track χ-zero reference**: §13novies-decies (P39 `fetch_chi_zero_imaginary_parts`) supplies the held-out true χ-zeros for $W_1$ scoring; it does NOT enter the construction.
* **Honest-scope framework**: §13octies (branches B1/B2/B3) and §13.2 (final gap balance) apply verbatim at the L-track level.

### §13vicies-octavo.7 Gap Balance

| Gap | Status before P49 | Status after P49 |
|---|---|---|
| G4 = RH on $\zeta$ | OPEN | OPEN, unchanged |
| GRH$_\chi$ for primitive real $\chi$ | OPEN | OPEN, unchanged |
| Oscillatory half of T-HP$^\chi$ at branch B1 (canonical-only) | UNATTESTED | **PARTIALLY ATTESTED**: $\chi_4$ shows +6.02% canonical improvement (branch B1 evidence); $\chi_3$, $\chi_5$ show 0% improvement (branch B2 corroboration) |
| ζ↔L attack-surface parity (P12–P31 ↔ P32–P49) | INCOMPLETE (P31 missing L-track counterpart) | **COMPLETE**: every canonical ζ-track operator from P12 through P31 has a matching χ-twisted L-track counterpart |

**Net effect**: P49 closes the **final ζ↔L attack-surface parity item** by lifting the §13decies-quarto branch-B1 prime-ladder oscillatory correction to every primitive real Dirichlet character. The L-track attack surface against T-HP$^\chi$ now mirrors the ζ-track attack surface against T-HP **in full**, from spectral data (P12↔P32) through operator-level smooth half (P30↔P48) and now oscillatory half (P31↔P49). The mixed empirical regime (1/3 branch-B1, 2/3 branch-B2) is honest structural-compatibility evidence; it neither closes G4 = RH nor proves GRH$_\chi$ for any character. P49 is a positive structural-parity milestone plus a diagnostic split that further corroborates §13octies branch B2 across both tracks.

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
| **P38** Dirichlet L χ-twisted admissibility / gauge sweep | `twisted_alpha_sweep.py` | `65_twisted_alpha_sweep_demo.py` | §13septiesdecies | Structural extension of P18 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; g) = W_\chi[\sigma] / E_{\mathrm{TNFR}}^\chi[\sigma; g]$ across the canonical six-gauge family `DEFAULT_GAUGES` inherited unchanged from P18 (`canonical, dnfr_only, phase_only, epi_only, dnfr_phase, pressure_amplified`); $W_\chi$ computed once per $\sigma$ (gauge-independent) via P35 enumerator; canonical TNFR test state built per gauge on P34 bundle; positivity verified for $\chi_3, \chi_4, \chi_5$ across $\sigma \in \{1.0, \ldots, 3.0\} \times$ 6 gauges (3/3 PASS; $\alpha_{\min}$ at $(\sigma=1.0, \text{canonical})$ in every case); robustness audit of P37 under canonical-mapping ambiguity; **does NOT prove GRH (finite $(\sigma, g)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P39** Dirichlet L χ-twisted admissible-family + gauge sweep | `twisted_admissible_family_sweep.py` | `66_twisted_admissible_family_sweep_demo.py` | §13octiesdecies | Joint structural extension of P19 + P18 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f] / E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ across `DEFAULT_TEST_FAMILIES` (gaussian, gaussian_mixture, hermite2_gaussian) inherited unchanged from P19 × `DEFAULT_GAUGES` (6 canonical gauges) inherited unchanged from P18; $W_\chi[\sigma; f]$ computed once per $(family, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(family, gauge)$ on P34 bundle via `build_twisted_test_state_from_test_function`; positivity verified for $\chi_3, \chi_4, \chi_5$ across 3 families × 6 gauges × 5 widths (3/3 PASS; 270 cells total; $\alpha_{\min}$ at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{canonical})$ in every case); joint robustness audit of P37 under test-profile + canonical-mapping ambiguity; **does NOT prove GRH (finite $(family, gauge, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P40** Dirichlet L χ-twisted node-aware gauge sweep | `twisted_nodeaware_gauge_sweep.py` | `67_twisted_nodeaware_gauge_sweep_demo.py` | §13noniesdecies | Structural extension of P20 to primitive real $L(s, \chi)$: sweeps $\alpha_\chi(\sigma; f, g) = W_\chi[\sigma; f] / E_{\mathrm{TNFR}}^\chi[\sigma; f, g]$ across `DEFAULT_TEST_FAMILIES` (P19) × `DEFAULT_NODEAWARE_GAUGES` (4 node-aware gauges: `nuf_pressure, nuf_phase, weight_pressure, mixed_affine`) inherited unchanged from P20; gauges have signature $g(h(E_n), \hat\nu_f(n), \hat w(n))$ activating the per-node normalised structural-frequency and node-weight channels of the P34 χ-twisted graph; $W_\chi[\sigma; f]$ computed once per $(family, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(family, node\_gauge)$ on P34 bundle via `build_twisted_test_state_nodeaware`; positivity verified for $\chi_3, \chi_4, \chi_5$ across 3 families × 4 node-aware gauges × 5 widths (3/3 PASS; 180 cells total; $\alpha_{\min}$ at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{nuf\_phase})$ for $\chi_3, \chi_4$ and at $(\sigma=1.0, \mathrm{gaussian}, \mathrm{nuf\_pressure})$ for $\chi_5$); node-aware robustness audit of P37 jointly with P19 test-profile sweep; **does NOT prove GRH (finite $(family, node\_gauge, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P41** Dirichlet L χ-twisted Hermite2-Gaussian η-parameter sweep | `twisted_hermite_family.py` | `68_twisted_hermite_family_demo.py` | §13vicies | Structural extension of P21 (Hermite2 family) to primitive real $L(s, \chi)$ along the envelope-strength axis: sweeps $\alpha_\chi(\sigma; \eta, g) = W_\chi[\sigma; \eta] / E_{\mathrm{TNFR}}^\chi[\sigma; \eta, g]$ across `DEFAULT_HERMITE2_ETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)` ($\eta = 0$ recovers pure Gaussian; $\eta = 0.25$ matches the P19/P39 snapshot) × `DEFAULT_GAUGES` (6 canonical scalar gauges; P18); $W_\chi[\sigma; \eta]$ computed once per $(\eta, \sigma)$ via P35 enumerator; canonical TNFR test state built per $(\eta, g)$ on P34 bundle via `build_twisted_test_state_from_test_function` (reused from P39); positivity verified for $\chi_3, \chi_4, \chi_5$ across 6 etas × 6 gauges × 5 widths (3/3 PASS; 180 cells per character; $\alpha_{\min}$ at $(\sigma=1.0, \eta=0.0, \mathrm{canonical})$ in every case); envelope-strength robustness audit of P37 along an orthogonal axis to P39/P40; **does NOT prove GRH (finite $(\eta, g, \sigma)$ grid; admissibility not exhausted) and does NOT advance G4** |
| **P42** Dirichlet L χ-twisted uniform-coercivity certificate | `twisted_coercivity_uniform.py` | `69_twisted_coercivity_uniform_demo.py` | §13vicies-primo | Structural extension of P22 / P23 / P24 (uniform / stratified / adaptive coercivity in `coercivity_uniform.py`) to primitive real $L(s, \chi)$: lifts the finite-grid sample of P39 + P40 to a **Lipschitz-mesh interval-level certificate** by sampling $\alpha_\chi(\sigma; \eta, g)$ on a log-spaced $\sigma$ grid, computing a finite-difference Lipschitz envelope $L^{\mathrm{proxy}}_\chi$, and forming three interval lower bounds (global, stratified, segment-local) via the canonical P22 / P23 helpers `_max_abs_slope`, `_segmentwise_interval_lower_bound`, `_stratified_interval_lower_bound` reused unchanged; optional P24-style adaptive refinement bisects worst-margin segments and re-runs both twisted sweeps; verified for $\chi_3, \chi_4, \chi_5$ on $\sigma \in [1.0, 3.0]$ with $N = 5$ (`sampled_all_positive = True`, `admissible_ok = True`, `nodeaware_ok = True` for every χ; sampled $\alpha^{\mathrm{samp}}_{\chi,\min} \in \{1.26 \times 10^{-14}, 2.70 \times 10^{-8}, 2.62 \times 10^{-10}\}$; interval $\mathrm{lb}_{\mathrm{local}} \in \{-6.06 \times 10^{-2}, -1.30 \times 10^{-1}, -6.51 \times 10^{-2}\}$ — all **negative** because $\alpha^{\mathrm{samp}}_{\chi,\min}$ near $\sigma = 1$ is essentially zero against any finite $L^{\mathrm{proxy}}_\chi$); one round of P24 bisection on the worst character ($\chi_4$, $N = 5 \to 7$) reduces $\mathrm{lb}_{\mathrm{local}}$ from $-1.30 \times 10^{-1}$ to $-3.40 \times 10^{-2}$ (74% margin reduction toward zero), confirming the bisection mechanism transports correctly to the χ-twisted side; **does NOT prove GRH (interval lower bounds currently negative; even when positive, finite log-spaced σ window is necessary, not sufficient) and does NOT advance G4** |
| **P43** Dirichlet L χ-twisted Paley-gap consistency diagnostic | `twisted_paley_gap_coercivity.py` | `70_twisted_paley_gap_coercivity_demo.py` | §13vicies-secundo | Structural extension of P25 (`paley_gap_coercivity.py`) to primitive real $L(s, \chi)$: compares three representations of $-L'(s,\chi)/L(s,\chi)$ — the P32 closed-form weighted spectrum $Z_{P32}$ (`tnfr_log_l_derivative`), the P34 χ-twisted weighted spectral trace $Z_{P34}$ (`twisted_weighted_spectral_trace`), and the classical truncated Dirichlet series $Z_{\mathrm{cls}} = \sum_{n \le N} \chi(n)\Lambda(n)/n^\sigma$ (`classical_log_l_derivative`) — via three absolute χ-twisted Paley-gap quantities $g_{P32}(\sigma) = |Z_{P32} - Z_{\mathrm{cls}}|$, $g_{P34}(\sigma) = |Z_{P34} - Z_{\mathrm{cls}}|$, $g_{\mathrm{cross}}(\sigma) = |Z_{P34} - Z_{P32}|$; verified on $(n_{\mathrm{primes}}, k_{\max}, N_{\max}^{\mathrm{cls}}) = (18, 5, 50\,000)$, $\sigma \in [1.5, 4.0]$ with $N = 11$ for $\chi_3, \chi_4, \chi_5$: at $J_0 = 0$ the decoupled cross gap collapses to machine precision ($\max g_{\mathrm{cross}} \in \{5.55 \times 10^{-17}, 4.16 \times 10^{-17}, 1.11 \times 10^{-16}\}$ — Paley-style algebraic identity between P32 and P34 on the L-track, regression test); at $J_0 = 10^{-2}$ the coupling-induced cross gap jumps to $O(10^{-5})$ (twelve orders of magnitude above noise; clean structural-deformation signal free of classical-truncation noise which contaminates $g_{P32}$ at $10^{-3}$ to $10^{-2}$); **does NOT prove GRH (regression test plus deformation magnitude; not a coercivity certificate) and does NOT advance G4** |
| **P44** Dirichlet L χ-twisted Lyapunov-spectral positivity certificate | `twisted_lyapunov_spectral_positivity.py` | `71_twisted_lyapunov_spectral_demo.py` | §13vicies-tertio | Structural extension of P26 (`lyapunov_spectral_positivity.py`) to primitive real $L(s, \chi)$: certifies self-adjointness, strict positivity with explicit Kato–Rellich envelope $\lambda_{\min}(\hat H^{(\chi)}) \ge \Delta_0^{(\chi)} - \lvert J_0 \rvert \lVert \hat H^{(\chi)}_{\mathrm{coupling}} \rVert_{\mathrm{op}}$ where $\Delta_0^{(\chi)} = \log(\min\{p \text{ prime} : p \nmid q\})$ (character-dependent: $\log 2$ for $\chi_3, \chi_5$; $\log 3$ for $\chi_4$), trace-class resolvent (Schatten-1/2 norms), and unitary flow conservation of $U(t) = e^{-it \hat H^{(\chi)}}$ on the finite-dimensional χ-twisted prime-ladder Hilbert space (P34 bundle); reuses `_matrix_exponential_skew` and `resolvent_schatten_norms` atomically from P26; verified on $(n_{\mathrm{primes}}, k_{\max}) = (18, 5)$ for $\chi_3, \chi_4, \chi_5$ at $J_0 \in \{0, 10^{-2}\}$: at $J_0 = 0$ empirical $\min(\lambda)$ matches $\Delta_0^{(\chi)}$ to machine precision (asserted in demo); at $J_0 = 10^{-2}$ `perturbation_safe = True` for every character with guaranteed gap $\in \{6.76 \times 10^{-1}, 1.08, 6.76 \times 10^{-1}\}$; unitary drifts $\sim 2 \times 10^{-16}$ throughout; `structural_positivity = True` for all 6 cells; **does NOT prove GRH (finite-dimensional positivity is necessary but not sufficient; the character enters only via the active-prime restriction, not via $W^{(\chi)}$) and does NOT advance G4** |
| **P45** Dirichlet L χ-twisted Hilbert–Pólya scaffold | `twisted_hilbert_polya.py` | `72_twisted_hilbert_polya_demo.py` | §13vicies-quarto | Structural extension of P27 (`hilbert_polya.py`) to primitive real $L(s, \chi)$: builds the reference operator $T_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\gamma_1^{(\chi)}, \dots, \gamma_N^{(\chi)})$ on $\ell^2_N(\mathbb{N})$ where $\gamma_n^{(\chi)}$ are positive imaginary parts of zeros of $L(s, \chi)$ located by Hardy–Z bisection (`find_dirichlet_l_zeros`, the same enumerator used by P36); reuses `build_hp_operator`, `verify_hp_self_adjoint`, `hp_resolvent_schatten_norms`, `wasserstein_1_distance` atomically from P27; certifies (i) self-adjointness (real diagonal, exact, Frobenius asymmetry $= 0$), (ii) trace-class shifted resolvent $(T_{\mathrm{HP}}^{(\chi)2} + s^2 I)^{-1/2}$ with explicit Schatten-1/2/op norms, (iii) χ-twisted Weil–Guinand consistency $2 \sum h_\sigma(\gamma_n^{(\chi)}) = g(0) \log(q/\pi) +$ archimedean $+ \sum_{p \nmid q, k} \chi(p)^k \log(p) p^{-k/2} g(k \log p)$ (parity-shifted digamma, character-dependent constant term replaces $\zeta$-pole $-g(0) \log \pi$), and (iv) Wasserstein-1 spectral gap against $\operatorname{spec}(\hat H^{(\chi)} \mid p \nmid q)$; verified on $(n_{\mathrm{primes}}, k_{\max}, n_{\mathrm{zeros}}, \sigma, s, \mathrm{tol}) = (18, 5, 25, 2.0, 1.0, 10^{-2})$ for $\chi_3, \chi_4, \chi_5$: Weil residuals $\{5.19 \times 10^{-16}, 9.07 \times 10^{-15}, 1.72 \times 10^{-15}\}$ at machine precision; $W_1 \in \{35.5, 31.8, 30.3\}$ with growth ratios $\sim 12$ quantifying the L-track operator-level structural gap (mirror of P30 negative-enrichment for $\zeta$); `scaffold_consistent = True` for all 3 characters; **does NOT prove GRH ($T_{\mathrm{HP}}^{(\chi)}$ is populated by *inputting* Hardy–Z bisection of classical $L(s, \chi)$; the operator is not derived from TNFR first principles) and does NOT advance G4** |
| **P46** Dirichlet L χ-twisted structural zero density | `twisted_structural_zero_density.py` | `73_twisted_structural_zero_density_demo.py` | §13vicies-quinto | L-track analogue of P28 (`structural_zero_density.py`): derives the smooth chi-twisted zero positions $\tilde{\gamma}_n^{(\chi)}$ from the chi-twisted Riemann–Siegel theta $\theta_\chi(T) = \operatorname{Im} \log \Gamma((1/2+a)/2 + iT/2) + (T/2) \log(q/\pi)$ via Newton iteration on $\bar{N}_\chi(\tilde{\gamma}_n^{(\chi)}) = n - 1/2$ — no `find_dirichlet_l_zeros` call on the derivation side (only used for benchmark); builds $\tilde{T}_{\mathrm{HP}}^{(\chi)} = \operatorname{diag}(\tilde{\gamma}_1^{(\chi)}, \dots, \tilde{\gamma}_N^{(\chi)})$ and certifies (i) per-zero residuals $r_n^{(\chi)} = \gamma_n^{(\chi)} - \tilde{\gamma}_n^{(\chi)}$ encoding $S_\chi(T) = \tfrac{1}{\pi} \arg L(\tfrac12 + iT, \chi)$, (ii) operator-level Wasserstein-1 reduction $W_1(\operatorname{spec}(\tilde{T}_{\mathrm{HP}}^{(\chi)}), T_{\mathrm{HP}}^{(\chi)}) \ll W_1(\operatorname{spec}(P34\vert_{p\nmid q}), T_{\mathrm{HP}}^{(\chi)})$, (iii) theoretical bound $\max\lvert r_n^{(\chi)}\rvert \le C \cdot \max(\log \gamma_n^{(\chi)} / \bar{N}_\chi'(\gamma_n^{(\chi)}))$ with $C \le 2$; verified on $(n_{\mathrm{zeros}}, p34\_n\_primes, p34\_max\_power) = (18, 30, 6)$ for $\chi_3, \chi_4, \chi_5$: $\max\lvert r_n^{(\chi)}\rvert \in \{3.21, 2.65, 2.53\}$; $W_1$ reductions $\{28.4 \to 1.32, 25.2 \to 1.23, 24.1 \to 1.17\}$, improvement ratios $\{21.6\times, 20.4\times, 20.6\times\}$; bound satisfied across all 3 characters; closes the **smooth half** of the L-track structural derivation gap (mirror of P28 for ζ); **does NOT prove GRH for any $L(s, \chi)$** (oscillatory residual encoding $S_\chi$ is the open arithmetic problem, equivalent to GRH$_\chi$) **and does NOT advance G4 = RH** |
| **P47** Dirichlet L χ-twisted spectral emergence under canonical coupling | `twisted_spectral_emergence.py` | `74_twisted_spectral_emergence_demo.py` | §13vicies-sexto | L-track analogue of P29 (`spectral_emergence.py`): sweeps the three canonical TNFR inter-prime coupling laws (`kuramoto_u3`: $(\gamma/\pi)\exp(-\lvert k\log p - m\log q\rvert)$; `phi_multiscale`: $\varphi^{-(k+m)}/\sqrt{pq}$; `pnt_logarithmic`: $\gamma/\log(1+pq)$) on the P34 χ-twisted prime-ladder Hamiltonian with explicit $\chi(p)\chi(q)$ multiplicative twist on every off-diagonal entry; computes the Kolmogorov–Smirnov distance of the unfolded nearest-neighbour spacing distribution to the GUE Wigner surmise (conjectural universality class of zeros of $L(s,\chi)$) and to the Poisson reference; verified on $(n_{\mathrm{primes}}, k_{\max}) = (20, 3)$ for $\chi_3, \chi_4, \chi_5$ over strengths $s \in \{0, 0.05, 0.1, 0.2, 0.5, 1, 2\}$: `pnt_logarithmic` uniformly strongest emergence kernel with $\mathrm{KS}_{\text{GUE}}^{\min} \in \{0.097, 0.116, 0.135\}$ at $s^* = 2$ ($33$–$49\%$ reduction vs baseline); `kuramoto_u3` second with $\mathrm{KS}_{\text{GUE}}^{\min} \in \{0.120, 0.150, 0.135\}$ at $s^* = 1$ ($25$–$36\%$ reduction); `phi_multiscale` weak ($0$–$6\%$ reduction); attests the L-track spacing-universality diagnostic for every primitive real Dirichlet character; **does NOT prove GRH for any $L(s, \chi)$** (KS-GUE residual at finite $K$ is consistent with finite-size effects, not evidence against GRH) **and does NOT advance G4 = RH** |
| **P49** Dirichlet L χ-twisted prime-ladder oscillatory correction | `twisted_oscillatory_correction.py` | `76_twisted_oscillatory_correction_demo.py` | §13vicies-octavo | L-track analogue of P31 (`oscillatory_correction.py`): reconstructs $S_\chi(T) = \pi^{-1}\arg L(\tfrac12 + iT, \chi)$ from the canonical P34 χ-twisted prime-ladder spectrum $\{(k\log p,\,\chi(p)^k\log p)\}$ via the χ-twisted Riemann–von Mangoldt template $\pi S_\chi^{\mathrm{TNFR}}(T) = -\sum_{(\mu,w)}(w/\mu)\sin(T\mu)\exp(-\mu/2)$, then applies the Newton step $\gamma_n^{(\chi),\,\text{corr}} = \tilde\gamma_n^{(\chi)} - d\,S_\chi^{\mathrm{TNFR}}(\tilde\gamma_n^{(\chi)}) / \bar N'_\chi(\tilde\gamma_n^{(\chi)})$ on the canonical P46 χ-twisted smooth targets with $\bar N'_\chi(T) = (2\pi)^{-1}\log(qT/(2\pi))$; restricted to **primitive real** characters so the von Mangoldt-style sum is real-valued (validates $\max\lvert\Im w\rvert \le 10^{-10}$); damping sweep $d \in \{0, 0.25, 0.5, 0.75, 1, 1.25, 1.5\}$; **closes the final ζ↔L attack-surface parity item**: with P49, every canonical ζ-track operator P12–P31 has a matching χ-twisted L-track counterpart (P32–P49); verified on $(N, N_{\mathrm{primes}}, K) = (10, 80, 5)$ for $\chi_3, \chi_4, \chi_5$: mixed empirical regime — $\chi_4$ shows **+6.02%** branch-B1 canonical improvement at $d^* = 1.5$ ($W_1$: $1.4185 \to 1.3331$); $\chi_3$ and $\chi_5$ show **0% improvement** ($d^* = 0$) corroborating §13octies branch B2 at the L-track level (a genuinely new canonical operator required); honest split (1/3 B1, 2/3 B2) further attests the canonical-only oscillatory cap visible across both tracks; **does NOT prove GRH$_\chi$ for any $L(s, \chi)$** (residual $W_1 \approx 1.3$–$1.6$ encodes the chi-twisted oscillatory remainder), **does NOT advance G4 = RH**, **does NOT address sub-problems (2) canonicity from the nodal equation and (3) positivity coincidence with the χ-twisted Weil form**; positive structural-parity milestone plus L-track structural-compatibility diagnostic |
| **P48** Dirichlet L χ-twisted admissible spectral-rescaling operator | `twisted_admissible_rescaling.py` | `75_twisted_admissible_rescaling_demo.py` | §13vicies-septimo | L-track analogue of P30 (`admissible_rescaling.py`): lifts the §13vicies-quinto density-level closure of the smooth half of T-HP$^{(\chi)}$ to the operator level by constructing the canonical diagonal rescaling $F^{(\chi)}_{\text{smooth}} = U_{P34}\,\operatorname{diag}(\sqrt{\tilde{\gamma}_i^{(\chi)} / \lambda_i})\,U_{P34}^{*}$ on each primitive real Dirichlet character; reuses `extract_positive_spectrum`, `build_smooth_rescaling_operator`, `apply_rescaling`, `verify_self_adjointness_preserved`, `verify_spectrum_match`, `oscillatory_correction_canonical` atomically from `admissible_rescaling.py`; certifies (i) self-adjointness preservation under conjugation, (ii) exact spectrum match $\operatorname{spec}(F^{(\chi)}_{\text{smooth}}\,H_{P34}^{(\chi)}\,(F^{(\chi)}_{\text{smooth}})^{*}) = \{\tilde{\gamma}_i^{(\chi)}\}$ to machine precision $\le 7.1\times10^{-15}$, (iii) Wasserstein-1 gap closure $W_1(\sigma(H_{P34}^{(\chi)}), \{\gamma_n^{(\chi)}\}) \to W_1(\{\tilde{\gamma}_n^{(\chi)}\}, \{\gamma_n^{(\chi)}\})$, (iv) honest sweep of the three canonical oscillatory enrichments (`phi_log`, `gamma_e`, `pi_density`) at amplitudes $\{0, 10^{-3}, 5\!\cdot\!10^{-3}, 10^{-2}, 5\!\cdot\!10^{-2}, 10^{-1}\}$ with per-mode breakdown; verified on $(n_{\mathrm{targets}}, p34\_n\_primes, p34\_max\_power) = (12, 25, 5)$ for $\chi_3, \chi_4, \chi_5$: smooth-half W$_1$ ratios $\{14.86\times, 13.85\times, 14.44\times\}$ (baseline $\{21.9, 19.0, 18.4\} \to$ smooth $\{1.47, 1.38, 1.27\}$); best canonical oscillation `pi_density` at amplitude $10^{-3}$ for every character with extra improvement $\{+17.85\%, +13.22\%, +12.68\%\}$ over smooth baseline; per-mode ranking uniform: `pi_density` > `gamma_e` > `phi_log`; closes sub-problem (1) of Conjecture T-HP$^{(\chi)}$ for the smooth half at the operator level (L-track mirror of P30 §13nonies); negative-knowledge oscillatory cap ($\le 18\%$ canonical improvement) constitutes structural evidence for §13octies branch B2 at the L-track level; **does NOT prove GRH$_\chi$ for any $L(s, \chi)$** (residual W$_1 \approx 1.1$–$1.2$ encodes $S_\chi(T) = (1/\pi)\arg L(\tfrac12+iT, \chi)$, GRH$_\chi$-equivalent) **and does NOT advance G4 = RH** |

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

## §13vicies-novies. REMESH Global Reframe (Cross-Program Discovery; May 2026; Does NOT Close G4 = RH)

**Status**: Working hypothesis (branch B1 of §13septies.7). Does **not** close G4 = RH, does **not** advance T-HP beyond §13nonies (P30 smooth half), does **not** promote any new canonical operator.

### §13vicies-novies.1 Origin

During the parallel TNFR–Navier–Stokes program (see `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §11), an analysis of the NS-G_blowup residual obstruction prompted re-examination of the 13-operator catalog for multi-scale closure primitives. A direct audit refuted the prior implicit assumption that no canonical operator handles asymptotic/global temporal coupling:

* `src/tnfr/config/defaults_core.py`: `REMESH_TAU_GLOBAL = 8` (graph-wide temporal memory), `REMESH_TAU_LOCAL = 4`, `REMESH_MODE in {knn, mst, community}` with `community` mode genuinely global.
* `src/tnfr/ontosim.py`: `# Global REMESH memory` allocates a graph-level `_epi_hist` deque of size `2·τ_global + 5`.
* `src/tnfr/operators/remesh.py`: documents three REMESH structural modes — **Hierarchical** (IL/VAL/SHA/NUL), **Rhizomatic** (OZ/UM/THOL), **Fractal Harmonic** (RA/NAV/AL/EN, scale-symmetric).
* `src/tnfr/multiscale/hierarchical.py`: explicit cross-scale ΔNFR coupling.

The canonical engine therefore **already contains** a global, multi-scale closure primitive (REMESH global with Fractal Harmonic mode and cross-scale coupling). What is missing for T-HP is the **canonical asymptotic specialisation of the existing REMESH global operator** at `τ → ∞` applied to the prime-ladder spectrum, not a new canonical primitive.

### §13vicies-novies.2 Reframed Branch Analysis of T-HP

| Component | Status | REMESH-global interpretation |
|---|---|---|
| Smooth half of `F` | Closed at density level (P28, §13sexies) and operator level (P30, §13nonies) | REMESH global at **finite** `τ_global` applied to the prime-ladder spectrum `{k log p}` (P14 eigendata) |
| Oscillatory half `S(T) = (1/π) arg ζ(½+iT)` | Open (RH-equivalent) | REMESH global at **`τ → ∞`** applied to the same prime-ladder spectrum |
| Branch classification | Previously implicitly B2 (new operator) | **Reframed as B1** (closeable inside the catalog if the canonical `τ → ∞` limit of REMESH global is derivable) |

### §13vicies-novies.3 What This Changes for the Riemann Program

* **The hypothesis is upgraded** from "new operator may be needed" (branch B2, open and uncertain) **to** "existing operator needs canonical asymptotic specialisation" (branch B1, a well-defined analytical problem on an existing canonical operator).
* **G4 = RH remains OPEN**. The P30 negative-enrichment result (canonical multiplicative perturbations of the smooth target failed to recover S(T)) is **reinterpretable**: the perturbations tested were finite-`τ` REMESH-global candidates, none of which can reproduce a `τ → ∞` limit by construction.
* **The Riemann program remains paused at T-HP** (per §"Program Status" of `AGENTS.md`). The reframe does **not** authorise reopening the ζ-track or L-track attack surfaces; it only re-classifies the residual obstruction.

### §13vicies-novies.4 Honest Scope

* **What §13vicies-novies claims**: a structural reframe of the T-HP residual obstruction, anchored in canonical engine artefacts (`REMESH_TAU_GLOBAL`, `_epi_hist`, REMESH modes, `multiscale/hierarchical.py`).
* **What §13vicies-novies does NOT claim**: does NOT prove RH, does NOT close G4, does NOT close T-HP, does NOT derive `REMESH-∞`, does NOT promote any new operator.
* **Cross-reference**: mirrored in `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §11 (added simultaneously). Both programs share the same canonical REMESH global infrastructure; the analytical study of its `τ → ∞` (Riemann) / scale `→ 0` (NS) asymptotic limit is shared work.

### §13vicies-novies.5 R∞-1a Empirical Baseline (Riemann side)

**Milestone**: R∞-1a — first numerical probe of REMESH-∞ on the Riemann-side prime-ladder dynamics.

**Implementation**: `benchmarks/remesh_infinity_riemann_baseline.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_baseline.json`.

**Setup**:
* Graph: P14 prime-ladder, `n_primes=10`, `max_power=4` → 40 nodes `(p, k)`, νf = k·log(p).
* Synthetic deterministic oscillatory field: `EPI(p,k;t) = (log(p)/k)·cos(k·log(p)·t)` evaluated on `t ∈ [0, dt, 2dt, …]`, `dt = 0.05`.
* History buffer `_epi_hist` populated to `max(τ_g, τ_l)+1` snapshots before each REMESH application; canonical mixing `EPI_new = 0.25·EPI_now + 0.25·EPI[t-τ_l] + 0.5·EPI[t-τ_g]` with `α = 0.5`, `τ_l = 4`.
* Three tracks executed in one run:
  * **Track A** — single-application sweep over `τ_g ∈ {4, 8, 16, 32, 64, 128, 256, 512}`, baseline restored between calls. Tests F1 (naive single-application Cesàro projection).
  * **Track B** — iterated REMESH^N at fixed `τ_g = 16`, `N ∈ [1, 512]`, with `_epi_hist` updated at every iteration (genuine Banach iteration of the canonical operator on this dynamics). Tests F2 (existence of a fixed point).
  * **Track C** — spectral diagnostic of the late-iterated state at `N = 256`, FFT along the νf-ordered axis after mean removal.

**Falsification criteria (pre-registered)**:
* **F1** triggered if Track A `dist→time_average` is monotone-decreasing in `τ_g` AND `final_rel < 0.1`. Interpretation: naive single-application B1 = Cesàro projection on time-average ⇒ B1 (naive) refuted.
* **F2** triggered if Track B `final_step_delta < 1e-6` OR `step_decay_ratio < 0.01`. Interpretation: iterated REMESH has a well-defined fixed point.

**Results (deterministic run; same seedless config reproducible)**:
* Baseline-to-time-average distance: 5.976e+00.
* **Track A**: F1 NOT triggered. Distance to time-average plateaus at `rel ∈ [0.392, 0.462]` across the entire sweep, non-monotone in `τ_g`. Confirms analytically that single-application `τ → ∞` is ill-defined on stationary oscillatory snapshots: the output depends on the specific phase of the past snapshot sampled at lag τ_g, not on a global asymptotic limit.
* **Track B**: F2 **TRIGGERED**. `step_decay_ratio = 6.82e-06`, `final_step_delta = 3.87e-05` at `N = 512`. Step deltas decay through `5.68 → 1.03 → 0.66 → … → 0.15 → 0.012 → 1.2e-4 → 3.9e-5`. The iterated map converges to a fixed point with `‖EPI*‖_L2 = 1.7501`, sitting at relative distance `0.2808` from the time-average (i.e. NOT the time-average).
* **Track C**: Late state at `N = 256` has structured oscillatory content along the νf-ordered axis. After mean removal, total power = 64.2, DC fraction = 3.07e-33 (numerical zero). Top-3 power bins are `{16, 19, 20}` of 21 rfft bins, with fractions `{10.6%, 9.7%, 9.3%}` — the spectrum is dominated by **high-νf modes**, not the low-νf prime-ladder fundamentals.

**Honest interpretation (R∞-1a)**:
* **Established** (necessary condition for any non-trivial B1 reframe): iterated REMESH on canonical prime-ladder oscillatory dynamics admits a well-defined fixed point. The fixed point is NOT the time-average and carries non-trivial spectral structure.
* **Not established** (and must NOT be claimed): (a) any verified correspondence between the fixed-point spectrum and the oscillatory residual `r_n = γ_n - γ̃_n`; (b) sensitivity-independence with respect to the choice of synthetic input field; (c) that high-νf concentration encodes S(T) rather than being a bias of the α-local mixing kernel; (d) closure of T-HP, G4, or RH.
* **Branch verdict (R∞-1a slice only)**: this baseline does NOT refute B1, and supplies the first necessary positive datum (existence of a non-trivial canonical fixed point). It does NOT confirm B1 either — the spectral comparison with r_n (R∞-1a-spectral, future work) is the next falsifiable test.

**Next milestones (gated on this result)**:
* **R∞-1a-spectral**: project the Track B fixed-point spectrum onto the basis of r_n via mpmath-computed γ_n; report correlation, cosine similarity, and per-component residual. Pre-register falsification: if no correlation above noise (|r| < 0.2), B1 is empirically refuted at the spectral level even with a non-trivial fixed point.
* **R∞-1b**: NS-side analogue on the K_φ cascade (N6–N11 milestones), same Track A/B/C structure.
* **R∞-1c**: cross-program comparison of fixed-point spectra. Required equivariance check before any cross-program B1 claim.

**Status**: R∞-1a baseline complete; primary deliverable is the empirical fact that iterated REMESH is contractive on this dynamics with a non-trivial fixed point. No closure of any gap.

### §13vicies-novies.6 R∞-1a-spectral — Spectral projection onto Riemann basis

**Milestone**: R∞-1a-spectral — first falsifiable spectral comparison between the R∞-1a fixed point and Riemann data. Gated follow-up to §13vicies-novies.5.

**Implementation**: `benchmarks/remesh_infinity_riemann_spectral.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_spectral.json`.

**Setup**:
* Identical prime-ladder, REMESH config, and Banach iteration as R∞-1a, run to `N_iter = 512` (true fixed point, not the intermediate `N = 256` state used in R∞-1a Track C).
* Riemann reference: first 40 non-trivial zeros γ_n from `mpmath.zetazero` (dps=30) and the canonical smooth approximations γ̃_n via `derive_smooth_zero_position` (P28). Oscillatory residuals `r_n = γ_n - γ̃_n`.
* Fixed point sorted by νf = k·log(p) → sequence `s_i, i = 1..40`. FFT of `s − mean(s)` → power bins `P_k, k = 1..M` with `M = 20`.

**Pre-registered tests (none decisive on its own)**:
* `r_α` = Pearson(P_k, |r_n|), index-aligned k=n=1..M.
* `r_β` = Pearson(sort(P_k, desc), sort(|r_n|, desc)) — magnitude-distribution alignment.
* `r_γ` = Pearson(s_i [νf-ordered], γ̃_n [n=1..N=40]) — node-field vs smooth target alignment.
* `r_δ` = Spearman-rank(P_k, |r_n|).

**Pre-registered falsification (F3)**:
* `max(|r_α|, |r_β|, |r_γ|, |r_δ|) < 0.2` ⇒ B1 REFUTED at spectral level.
* `max(…) > 0.5` ⇒ B1 SUPPORTED spectrally (does NOT prove RH; only empirical correspondence).
* `max(…) ∈ [0.2, 0.5]` ⇒ INDETERMINATE.

**Results (deterministic; same config reproducible)**:
* True fixed point at `N = 512`: `‖EPI*‖_L2 = 1.6976`, `mean(EPI*) = −9.25e−02`, spectral total power = 50.87.
* **Spectral shift between intermediate (N=256) and converged (N=512) states**: at N=256 the top-3 bins were high-νf `{16, 19, 20}` of 21 (R∞-1a Track C); at the true fixed point (N=512) the top-3 bins drop to low-νf `{1, 2, 4}` with fractions `{33.1%, 23.4%, 8.8%}`. Iterated REMESH transports power from high-νf to low-νf as it converges. The R∞-1a Track C statement that the fixed point is "dominated by high-νf modes" is therefore SUPERSEDED — the converged fixed point is low-νf dominated.
* Pre-registered tests:
  - `r_α = +0.5126` — crosses 0.5 threshold but only marginally.
  - `r_β = +0.8575` — sorted-magnitude alignment, dominant signal.
  - `r_γ = +0.3454` — node-field vs smooth target, indeterminate range.
  - `r_δ = +0.4120` — Spearman, indeterminate range.
  - `max|·| = 0.8575`.
* **Verdict by the pre-registered criterion**: **F3 nominally SUPPORTED** (max > 0.5).
* **Auxiliary controls (NOT in F3, declared in advance as diagnostic)**:
  - `r(P_k, γ̃_n)` = −0.6690 (strong negative).
  - `r(P_k, γ_n)` = −0.6726 (strong negative).
  - `r(s_i, r_n)` = +0.0055 (no node-level signal at all).

**Honest interpretation (R∞-1a-spectral)**:
* The pre-registered criterion (F3 > 0.5) is met, but the support is fragile and requires multiple caveats before being accepted as evidence for branch B1:
  1. The dominant test (`r_β = 0.86`) is **sorted-magnitude correlation**, which is statistically the weakest of the four. Any two positive heavy-tailed sequences with similar dynamic ranges tend to produce high sorted-magnitude correlation; this test does NOT establish structural alignment between the spectrum and the residuals.
  2. The strongest structural test (`r_γ = 0.34`, node-field vs smooth target) sits in the indeterminate range.
  3. The two auxiliary controls `r(P_k, γ_n) ≈ r(P_k, γ̃_n) ≈ −0.67` reveal that the spectrum is dominantly anti-correlated with the monotone-growing Riemann data, which is consistent with the low-νf concentration being a property of the REMESH mixing kernel rather than encoding Riemann content.
  4. Node-level correlation between the fixed-point field and the residuals (`r(s_i, r_n) = +0.005`) is **zero within noise** — there is no per-mode encoding.
* **What R∞-1a-spectral establishes**: existence of *some* monotone alignment between the magnitude distributions of (fixed-point FFT power) and (|r_n|). This is a necessary condition for B1 at the level of distributions, but is far from sufficient.
* **What R∞-1a-spectral does NOT establish**: per-mode correspondence, operator-level alignment, robustness against synthetic-field choice, sensitivity to (α, τ_l, τ_g), independence from prime-ladder construction.

**Branch verdict (R∞-1a-spectral slice only)**: this milestone does **not refute** B1 at the spectral level, and supplies one weak positive datum (magnitude-distribution alignment). It **does not** confirm B1 — the per-mode (`r_α`, `r_γ`, `r_δ`) tests are inconclusive, and the auxiliary controls flag a kernel-induced bias as a competing explanation. The result must be read as "B1 survives the first falsifiable spectral test, but only by its weakest available signal; further tests required before any B1 claim".

**Next milestones (gated on this result)**:
* **R∞-1a-spectral-robustness** (REQUIRED before any further B1 claim): re-run R∞-1a-spectral with (i) a randomized null synthetic field (white noise) to verify that `r_β` does NOT trigger on noise — kernel-bias control; (ii) sweep over `α ∈ {0.25, 0.5, 0.75}` and `τ_l ∈ {2, 4, 8}` to test sensitivity; (iii) alternative orderings (random permutation of νf-axis) as null controls for `r_α` and `r_γ`.
* **R∞-1a-operator** (gated on robustness): if R∞-1a-spectral-robustness survives, construct a finite-rank approximation of the implied REMESH-∞ operator and compare its spectrum directly to {γ_n}. This is the proper operator-level test that the present field-level test only approximates.
* **R∞-1b**: NS-side analogue (K_φ cascade), independent of Riemann result.

**Status**: R∞-1a-spectral complete. F3 nominally satisfied with **substantial caveats**; the result is consistent with both B1-positive (REMESH-∞ carries weak Riemann signal) and B1-null-kernel-bias (sorted-magnitude alignment is an artefact of heavy-tailed marginals). No closure of any gap; no support for any cosmic claim. R∞-1a-spectral-robustness is the next pre-registered gate.

### §13vicies-novies.7 R∞-1a-spectral-robustness — Falsification gate (REFUTES `r_β` as Riemann signal)

**Milestone**: R∞-1a-spectral-robustness — pre-registered F4 gate for the R∞-1a-spectral result. Three independent controls executed simultaneously; outcome was decisive.

**Implementation**: `benchmarks/remesh_infinity_riemann_spectral_robustness.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_spectral_robustness.json`.

**Setup**: identical pipeline to R∞-1a-spectral (same prime ladder, `N_iter = 512`, same Riemann reference from `mpmath.zetazero` + P28). Three independent controls:
* **C1** white-noise null: 16 seeded runs (`numpy.random.default_rng(20260526 + seed)`, `seed ∈ {0..15}`) replacing the canonical oscillatory synthetic EPI field with zero-mean unit-variance white noise, identical REMESH iteration.
* **C2** sensitivity sweep: 3 × 3 grid `(α, τ_l) ∈ {0.25, 0.5, 0.75} × {2, 4, 8}` on the canonical synthetic field.
* **C3** permutation null: 5000 random permutations of `|r_n|` (for `r_α`) and `γ̃_n` (for `r_γ`) on the canonical fixed-point spectrum, `numpy` seed `20260526`.

**Pre-registered falsification (F4)**:
* REFUTED if ANY of: (a) C1 mean `|r_β|`-null > 0.5; (b) C2 `r_β < 0.5` anywhere in grid; (c) C3 both `p_α > 0.05` AND `p_γ > 0.05`.
* STRENGTHENED if ALL of: (a) C1 mean `|r_β|`-null < 0.2 AND observed `r_β` outside 95% null; (b) C2 `r_β > 0.5` everywhere; (c) C3 `p_α < 0.05` OR `p_γ < 0.05`.
* MIXED otherwise.

**Results (deterministic; full per-run table in JSON)**:

*Baseline (canonical)*: `r_α = +0.5126, r_β = +0.8575, r_γ = +0.3454, r_δ = +0.4120` (reproduces R∞-1a-spectral exactly).

*C1 white-noise null* (16 seeds):
* `r_β` null mean = **+0.9440**, `|·|` mean = 0.9440, std = 0.0286, 95% range = [+0.8888, +0.9763].
* The baseline `r_β = +0.8575` is **below the 2.5% quantile** of the white-noise null distribution.
* `r_α` null mean = −0.0947 (|·| mean = 0.1848, std = 0.213).
* `r_γ` null mean = −0.0636 (|·| mean = 0.0880, std = 0.103).

*C2 sensitivity sweep* (9 cells, **post-bug-fix run; see «α propagation bug» note below**): `r_β` range [+0.8194, +0.8935], `r_α` range [+0.3958, +0.5247], `r_γ` range [+0.2880, +0.3565]. `r_β > 0.5` at every cell. Per-cell variation in α is now visible (previously masked by the propagation bug).

*C3 permutation null* (5000 perms each):
* `r_α`: observed +0.5126 vs null `(mean = +0.0025, std = 0.227)`, **p_one_sided = 0.0228**, p_two_sided = 0.0246.
* `r_γ`: observed +0.3454 vs null `(mean = +0.0014, std = 0.160)`, **p_one_sided = 0.0154**, p_two_sided = 0.0304.

**F4 verdict: REFUTED** (refute-C1 triggered).

**Honest interpretation (R∞-1a-spectral-robustness)**:
* The dominant R∞-1a-spectral signal (`r_β = 0.86`) is **a pure kernel artefact**. White noise reproduces it at higher magnitude (mean 0.94) than the canonical oscillatory field. The sorted-magnitude Pearson coefficient measures only that the FFT-power marginal and the `|r_n|` marginal share a heavy-tailed structure; it does NOT detect any structural alignment between the spectrum of the REMESH fixed point and Riemann residuals. The R∞-1a-spectral "B1 nominally SUPPORTED" verdict relied on `r_β` and must therefore be **withdrawn**.
* C2 shows `r_β` does vary with `(α, τ_l)` once `α` is actually propagated (range [+0.819, +0.894], 9 cells), but remains `> 0.5` everywhere — does not refute. The original C2 read of "r_β invariant in α" was an artefact of an α-propagation bug in the canonical REMESH pipeline (see dedicated note below). After the fix, `r_α ∈ [+0.40, +0.52]` and `r_γ ∈ [+0.29, +0.36]` are **robust across the (α, τ_l) grid**, which strengthens (not weakens) the interpretation of these two metrics as genuine weak structural alignments.
* C3 supplies the only genuinely positive finding: `r_α` and `r_γ` are statistically significant against permutation null (p ≈ 0.02 and p ≈ 0.015 one-sided). They are NOT artefacts of the marginal distributions; the alignment between (FFT power → |r_n|) index-wise and (νf-ordered field → smooth target) is structurally non-random. However, the effect sizes are modest:
  - `r_α = 0.5126` was already only marginally above the F3 threshold and now stands alone.
  - `r_γ = 0.3454` remains in the indeterminate band of F3.
* **Net B1 evidential balance after R∞-1a-spectral-robustness**: the dominant claimed signal is artefact; two minor signals survive permutation testing but with modest effect sizes and neither alone meets the original F3 "SUPPORTED" threshold (one is marginal at 0.51, the other indeterminate at 0.35).

**What R∞-1a-spectral-robustness establishes**:
* `r_β` (sorted-magnitude Pearson on FFT power vs |r_n|) is **not a valid Riemann signal** in this benchmark family and must be retired.
* Permutation-tested `r_α` (Pearson on power vs |r_n|, index-aligned) and `r_γ` (Pearson on νf-ordered field vs γ̃_n) carry **weak but genuine non-random structural alignment** that is not explained by marginal distributions or kernel parameters.

**What R∞-1a-spectral-robustness does NOT establish**:
* It does NOT confirm B1 — the surviving signals are below the originally pre-registered support threshold.
* It does NOT refute B1 entirely — the permutation-significant `r_α` and `r_γ` remain a positive (though weak) datum.
* It does NOT close T-HP, G4, or any gap.

**Branch verdict (R∞-1a-spectral-robustness slice only)**: B1 is **WEAKENED but not refuted**. The R∞-1a-spectral claim of "B1 nominally SUPPORTED at the spectral level (max > 0.5)" is **withdrawn**. The current state of B1 evidence after this milestone is: one necessary positive datum (existence of non-trivial REMESH fixed point, R∞-1a), one withdrawn artefactual signal (`r_β`, this milestone), and two weak-but-permutation-significant alignments (`r_α ≈ 0.51`, `r_γ ≈ 0.35`, this milestone). This is far below what would be required to claim B1 closure of T-HP.

**Next milestones (gated on this result)**:
* **R∞-1a-operator** (REQUIRED before any further B1 evidential update): the field-level test in this milestone is at best a proxy for the actual structural question — does the REMESH-∞ operator, viewed as a linear map on the appropriate state space, have spectrum compatible with `{γ_n}`? Construct a finite-rank approximation of the REMESH iteration matrix on `EPI`-space, diagonalize, and compare the eigenvalue spectrum directly to `{γ_n}`. Pre-register: if the largest absolute correlation between (REMESH-∞ eigenvalue magnitudes) and (`γ_n` or `|r_n|`) is `< 0.5` after permutation testing, B1 is refuted at the operator level.
* **R∞-1b**: NS-side analogue, independent.
* **B1 status update**: with `r_β` retired and only weak `r_α`/`r_γ` surviving, the canonical-catalog-closure conjecture (B1) **loses substantial empirical support** but remains technically open pending R∞-1a-operator. Branches **B2** (a new canonical operator is required) and **B3** (no TNFR closure exists) gain proportionally in prior weight, though no decisive evidence shifts the balance entirely to either.

**α propagation bug (diagnosed and fixed mid-milestone)**:
* During the C2 sweep an unexpected invariance of `r_β` across the α axis was observed (identical values for α ∈ {0.25, 0.5, 0.75} at each τ_l). Direct probing of `_remesh_alpha_info` in `src/tnfr/operators/remesh.py` revealed that the precedence order is **(1)** `REMESH_ALPHA` when `REMESH_ALPHA_HARD=True`, **(2)** `GLYPH_FACTORS.REMESH_alpha` from the canonical defaults, **(3)** `G.graph["REMESH_ALPHA"]` only as fallback. Without the HARD flag, the value written by the benchmark to `G.graph["REMESH_ALPHA"]` is silently ignored — the default `GLYPH_FACTORS.REMESH_alpha = 0.5` is used regardless.
* Reproducer (direct call to `_remesh_alpha_info`):
  - Set `G.graph["REMESH_ALPHA"] = 0.25` (no HARD flag) → returns `α = 0.5, source = "GLYPH_FACTORS.REMESH_alpha"`.
  - Set `G.graph["REMESH_ALPHA"] = 0.25` and `G.graph["REMESH_ALPHA_HARD"] = True` → returns `α = 0.25, source = "REMESH_ALPHA"`.
* `τ_local` and `τ_global` use `get_param()` which reads from `G.graph` directly, so their C2 axis was always honoured (variation across τ_l in the original run was real).
* **Fix applied**: `benchmarks/remesh_infinity_riemann_spectral_robustness.py::run_canonical_pipeline` now sets `G.graph["REMESH_ALPHA_HARD"] = True` before iteration, with an explanatory comment cross-referencing this section. C2 was re-executed after the fix; the numbers above (range [+0.819, +0.894] for `r_β`, [+0.40, +0.52] for `r_α`, [+0.29, +0.36] for `r_γ`) are from the fixed run. C1 and C3 are independent of the α value and are unchanged.
* **Note on the canonical pipeline**: this precedence ordering means any user who writes `G.graph["REMESH_ALPHA"]` without also enabling `REMESH_ALPHA_HARD` will get the default `0.5` silently. This is a latent surprise but not a TNFR-grammar violation per se. Documented here for cross-program awareness; not promoted to a code-level fix in this milestone because the canonical α = 0.5 is the documented TNFR default and changing the precedence requires its own grammar audit.

**Status**: R∞-1a-spectral-robustness complete. F4 refutes the dominant R∞-1a-spectral signal as kernel artefact while preserving two weak permutation-significant alignments (`r_α`, `r_γ`) that are also confirmed robust across the (α, τ_l) grid after the α-propagation bug was fixed. The R∞-1a-spectral milestone is **formally amended**: the "B1 SUPPORTED" verdict is withdrawn; the residual evidence (R∞-1a fixed-point existence + permutation-significant weak `r_α`, `r_γ` confirmed across (α, τ_l)) is insufficient to support B1 at the spectral level but is mildly stronger than the original interpretation that allowed for parameter fragility. No closure of any gap. R∞-1a-operator is the next pre-registered gate; until it returns, the canonical TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary as stated in §13septies.

---

### §13vicies-novies.8 R∞-1a-operator — Structural refutation of B1 at the operator level (REMESH-iterated-in-isolation)

**Milestone**: R∞-1a-operator — gated follow-up to §13vicies-novies.7. Examines whether the spectrum of the REMESH iteration matrix (viewed as a linear map on the augmented EPI × temporal-history state) can encode `{γ_n}`-specific content. Outcome is **doubly negative**: a *structural* refutation independent of any statistic, plus a methodological exposure of the pre-registered F5 statistical test as a monotonicity artefact.

**Implementation**: `benchmarks/remesh_infinity_riemann_operator.py`. Output: `results/remesh_infinity/remesh_infinity_riemann_operator.json`.

**Structural construction**. The canonical REMESH update (`src/tnfr/operators/remesh.py` L1212–1252) is strictly linear and **node-local**:

$$\mathrm{EPI}_{\text{new}}(i) = (1-\alpha)^2 \cdot \mathrm{EPI}(i,t) + \alpha(1-\alpha) \cdot \mathrm{EPI}(i,t-\tau_l) + \alpha \cdot \mathrm{EPI}(i,t-\tau_g).$$

No edge term, no inter-node coupling. The full state of node $i$ over a delay window of length $\tau_g + 1$ therefore evolves under a shift-augmented matrix $M \in \mathbb{R}^{(\tau_g+1)\times(\tau_g+1)}$ given by

$$M[0,0] = (1-\alpha)^2,\quad M[0,\tau_l] = \alpha(1-\alpha),\quad M[0,\tau_g] = \alpha,\quad M[k,k-1] = 1\ \text{for}\ k=1,\dots,\tau_g.$$

Because there is no inter-node coupling, the full-graph iteration operator is **block-diagonal**: $N$ identical copies of $M$. The spectrum is the spectrum of $M$ with multiplicity $N$. **Neither the graph topology nor the P14 prime-ladder initial condition enters $M$ at any point.**

**Canonical spectrum** (α = 0.5, τ_l = 4, τ_g = 16; verified analytically with `scipy.linalg.eig`):
* $\lambda_1 = 1$ exactly (trivial fixed-point subspace: temporally-constant configurations are preserved exactly by the convex combination).
* 16 non-trivial eigenvalues organised as 8 complex-conjugate pairs.
* $|\lambda_k| \in [0.938, 0.982]$ for $k = 2, \dots, 17$ (all strictly inside the unit disk).
* Spectral radius excluding unity: $0.981475$.

**Pre-registered statistical test (F5)**:
* H0 (refute operator-level B1): no ordering of the 16 non-trivial eigenvalues achieves Pearson or Spearman $|r| \ge 0.5$ vs $\gamma_1, \dots, \gamma_{16}$ with permutation $p_{\text{one-sided}} < 0.05$.
* H1 (support): some ordering does.
* Ordering battery: `abs_desc`, `abs_asc`, `arg_upper_asc`, `real_desc`, `imag_upper_asc` × {Pearson, Spearman} = 10 tests. Sensitivity sweep: 3 × 3 grid `(α, τ_l) ∈ {0.25, 0.5, 0.75} × {2, 4, 8}`, τ_g = 16. Permutation null $N_{\text{perm}} = 5000$, seed 20260526.

**Results (canonical config)**:
| ordering | stat | $r$ | $p_{\text{perm}}$ |
|---|---|---|---|
| `abs_desc` | pearson | −0.9628 | 0.0002 |
| `abs_desc` | spearman | −0.9941 | 0.0002 |
| `abs_asc` | pearson | +0.9615 | 0.0002 |
| `abs_asc` | spearman | +0.9941 | 0.0002 |
| `arg_upper_asc` | pearson | +0.9917 | 0.0004 |
| `arg_upper_asc` | spearman | **+1.0000** | 0.0002 |
| `real_desc` | pearson | −0.9821 | 0.0002 |
| `real_desc` | spearman | −0.9941 | 0.0002 |
| `imag_upper_asc` | pearson | +0.9913 | 0.0002 |
| `imag_upper_asc` | spearman | **+1.0000** | 0.0002 |

**Naïve F5 verdict (canonical)**: 10/10 PASS, max $|r| = 1.0000$. Sensitivity sweep: 9/9 cells PASS.

**Monotonicity controls (kernel-artefact diagnostic)**. The pre-registered F5 compares two sorted sequences against each other. Any monotonically ordered sequence aligned by index with the sorted $\{\gamma_n\}$ yields Spearman $= \pm 1$ and Pearson $\approx 0.95$–$1.0$; the permutation null is uninformative because almost every permutation breaks monotonicity. Four control sequences with no Riemann content were run through the same battery:

| control | stat | $r$ | $p_{\text{perm}}$ | naive PASS? |
|---|---|---|---|---|
| `integer_ladder` ($1, 2, \dots, 16$) | pearson | +0.9937 | 0.0002 | YES |
| `integer_ladder` | spearman | +1.0000 | 0.0002 | YES |
| `arithmetic_decay` ($\mathrm{linspace}(0.98, 0.94, 16)$) | pearson | −0.9937 | 0.0002 | YES |
| `arithmetic_decay` | spearman | −1.0000 | 0.0002 | YES |
| `random_monotone_in_unit_disk` | pearson | +0.9879 | 0.0002 | YES |
| `random_monotone_in_unit_disk` | spearman | +1.0000 | 0.0002 | YES |
| `log_n_growth` ($\log(1 + n)$) | pearson | +0.9845 | 0.0002 | YES |
| `log_n_growth` | spearman | +1.0000 | 0.0002 | YES |

**8/8 controls pass naive F5 at thresholds equal to or stronger than the canonical operator spectrum.** Therefore the canonical PASS is fully explained by the trivial monotonicity of any sorted sequence against the sorted $\{\gamma_n\}$ — exactly the same failure mode that retired `r_β` in §13vicies-novies.7.

**F5 STRICT verdict (canonical)**: **REFUTED_BY_MONOTONICITY_ARTEFACT**. The statistical battery as pre-registered has no falsification power and must be retired.

**Structural verdict (independent of any statistic)**. The REMESH iteration operator applied in isolation, as a strictly node-local linear map, is **structurally incapable** of encoding `{γ_n}`-specific content in its spectrum. The spectrum depends only on the three scalar canonical parameters $(\alpha, \tau_l, \tau_g)$ and on nothing else: not on the graph topology, not on the prime-ladder initial state, not on the field activation pattern, not on the number of nodes. Any apparent alignment between $\sigma(M)$ and $\{\gamma_n\}$ is either (a) a kernel monotonicity artefact (demonstrated above), or (b) imposed by the analyst's choice of $\{\gamma_n\}$ as the comparison target rather than discovered from the operator. **This refutes B1 at the level of REMESH iterated in isolation.**

**What §13vicies-novies.8 establishes**:
* REMESH applied as a stand-alone iterated linear operator **cannot** carry Riemann-spectral content. The 17-dimensional spectrum is exactly determined by the three canonical parameters with no degree of freedom for graph- or initial-state-dependent encoding.
* The naive correlation-based F5 test design is **invalid** for comparing two intrinsically sorted finite sequences and is formally retired (analogously to `r_β` in §13vicies-novies.7).
* The earlier R∞-1a fixed-point existence (§13vicies-novies.5) and its weak permutation-significant `r_α`, `r_γ` alignments (§13vicies-novies.7) are **not refuted** by this milestone. They concern an EPI **field** trajectory under iterated REMESH on a P14-initialised system, where the topology and initial state determine the *image* of the operator on the prime-ladder subspace, even though the operator's *spectrum* does not. The distinction is exactly the difference between $\sigma(M)$ (intrinsic, parameter-only) and $M \mathbf{v}_{P14}^k$ (depends on initial state).

**What §13vicies-novies.8 does NOT establish**:
* It does NOT refute B1 entirely. The structural refutation is **scoped to REMESH iterated in isolation as a stand-alone operator**. B1 in its full breadth — closure of T-HP inside the 13-operator catalog — remains technically open via two non-refuted channels:
  - **Composed operators**: REMESH ∘ IL, REMESH ∘ OZ, etc. The U1–U6 canonical grammar admits these compositions, and any non-trivial composition involves at least one operator whose action *does* couple nodes via the graph (IL, EN, NAV, RA propagate through edges). Composed operators therefore have spectra that *do* depend on topology and initial state, and the structural argument of this milestone does not apply.
  - **Hierarchical / fractal modes**: the canonical REMESH catalog (`src/tnfr/operators/remesh.py`) specifies three structural modes (Hierarchical, Rhizomatic, Fractal Harmonic) and `src/tnfr/multiscale/hierarchical.py` implements explicit cross-scale ΔNFR coupling. These are non-iterated-in-isolation regimes; this milestone does not bound them.
* It does NOT close G4 = RH, does NOT close T-HP, does NOT prove RH, does NOT promote any new operator.
* The fixed-point existence and weak `r_α`, `r_γ` alignments from §13vicies-novies.5–7 retain their status (necessary but insufficient).

**Branch verdict update (after R∞-1a-operator)**:
* **B1 at REMESH-iterated-in-isolation level**: STRUCTURALLY REFUTED.
* **B1 at composed-operator / hierarchical-mode level**: untouched (open).
* **B1 as a whole**: WEAKENED FURTHER. Of the two remaining channels for B1 closure inside the catalog, the one most directly suggested by the cross-program REMESH reframe (§13vicies-novies.1–4) is now closed. The composed-operator channel remains open but requires a *gramatically-canonical sequence* of operators (an U1–U6 admissible composition) whose spectrum would need to be derived analytically and tested against `{γ_n}` with a statistic that does *not* fall to the monotonicity artefact (e.g., normalised gap statistics, level-spacing distributions, or KS-vs-GUE diagnostics rather than two-sorted-sequence Pearson/Spearman).
* **B2 (new canonical operator required)** and **B3 (no TNFR closure exists)** gain proportionally in prior weight, though no decisive evidence shifts the balance entirely to either.

**Next milestones (gated on this result)**:
* **R∞-1a-composed** (REQUIRED before any further B1 evidential update): identify a minimal U1–U6 admissible composition of REMESH with at least one node-coupling canonical operator (candidates: REMESH ∘ IL, REMESH ∘ NAV, REMESH ∘ OZ ∘ EN), construct the iteration matrix on the joint state space, and test its spectrum against `{γ_n}` and against canonical null sequences using a statistic that *does* discriminate (level-spacing distribution, normalised eigenvalue-gap KS to GUE, or spectral-form-factor comparison). Pre-register thresholds before execution.
* **R∞-1b** (NS-side analogue): independent of the Riemann program; structural argument of this milestone likely transfers to the NS side because REMESH is canonical in both engines, but should be re-derived in the NS-G_blowup context.
* **B1 status check**: with the structural refutation of REMESH-isolated added to the retracted `r_β` and the (re-bounded) weak `r_α`/`r_γ`, the canonical-catalog-closure conjecture (B1) **loses substantial structural support** but is not strictly refuted because composed-operator channels remain untested. The TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary per §13septies; the reframe of §13vicies-novies.1–4 should now be **further qualified** to: "REMESH-global is canonical and structurally relevant, but REMESH iterated *in isolation* cannot carry Riemann content. Branch B1, if it closes, will do so via composed operators or via the hierarchical/fractal modes — neither of which is yet tested."

**Status**: R∞-1a-operator complete. Structural verdict: REMESH iterated in isolation **cannot** encode `{γ_n}`. Statistical verdict: the pre-registered F5 test has no falsification power and is retired. Net B1 evidential balance: structurally weakened (one of two narrow channels closed); two narrow channels (composed operators, hierarchical/fractal modes) remain technically open. No closure of any gap. The TNFR-Riemann program remains paused at the T-HP / G4 = RH boundary as stated in §13septies, with the §13vicies-novies reframe now further qualified.

---
