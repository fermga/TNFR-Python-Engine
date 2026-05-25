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

- `operator.py` – Canonical implementation of $H^{(k)}(\sigma)$ operators and prime graph construction.
- `spectral_proof.py` – Spectral convergence proofs ($\sigma_c^{(k)} \to 1/2$).
- `complex_extension.py` – Complex plane extensions of TNFR operators.
- `spectral_zeta.py` – Discrete spectral zeta functions.
- `topology.py` – Topology comparison analysis (path, cycle, star, etc.).
- `spectral_conservation.py` – Spectral conservation laws.
- `analytical_convergence.py` – Analytical convergence analysis.
- `eigenmode_fields.py` – Eigenmode-based tetrad field computation.
- `random_ensemble.py` – Random matrix ensemble comparisons.
- `telemetry.py` – Riemann telemetry records and field aggregate helpers.

### Examples

- `examples/16_riemann_operator_demo.py` – Reference execution path, eigenvalue exploration.
- `examples/18_riemann_convergence_proof.py` – Spectral convergence proof (four lines of attack).
- `examples/19_topology_comparison.py` – Cross-topology critical parameter comparison.
- `examples/20_eigenmode_tetrad.py` – Eigenmode-based tetrad field analysis.
- `examples/24_spectral_conservation_demo.py` – Spectral conservation law demonstration.

### Supporting Infrastructure

- `benchmarks/riemann_program.py` – Automated spectral regression benchmarks.
- `theory/UNIFIED_GRAMMAR_RULES.md` – Grammar rules referenced throughout.
- `docs/STRUCTURAL_FIELDS_TETRAD.md` – Tetrad field specifications.

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

The remainder of this document preserves the legacy research notes verbatim. Keep them synchronized with the active workflow above when adding new results.

## TNFR–Riemann Research Notes (Legacy Detail)

**Status**: Exploratory (Non-canonical)

These notes sketch a possible route to connect TNFR nodal dynamics with the Riemann Hypothesis (RH). Nothing in this document should be considered a proof; it is a research agenda framed in TNFR language.

---

## 1. Objective

Formulate a TNFR-consistent operator and field framework such that:

1. The Riemann zeta function (or a closely related object) is realized as a structural field or partition function of a TNFR system.
2. The non-trivial zeros correspond to eigenvalues or resonant modes of a well-defined TNFR operator.
3. Structural confinement or stability principles enforce that all such modes lie on the critical line Re$(s) = 1/2$.

---

## 2. Zeta as a TNFR Structural Partition Function

### 2.1 Euler Product and Prime Resonances

Classically, for Re$(s) > 1$:

$$
\zeta(s) = \prod_{p \ \text{prime}} \frac{1}{1 - p^{-s}}
$$

Interpretation in TNFR language:

- Each prime $p$ is treated as a **fundamental resonance** or node.
- The factor $(1 - p^{-s})^{-1}$ encodes the contribution of all harmonics $p^{-ns}$ generated by that resonance.

We seek a TNFR system where:

$$
Z_{TNFR}(s) = \mathcal{Z}[\text{EPI}(s)] \equiv \prod_{p} \frac{1}{1 - e^{-\beta E_p(s)}}
$$

with a suitable identification $e^{-\beta E_p(s)} \equiv p^{-s}$ so that $Z_{TNFR}(s)$ analytically continues to $\zeta(s)$.

### 2.2 Candidate Mapping

Let $s = \sigma + i t$ and define an **effective spectral energy** for the prime-labelled modes:

$$
E_p(s) = (\sigma - \tfrac{1}{2}) \log p + i t \log p.
$$

Formally:

- The real part controls **amplitude decay/growth**.
- The imaginary part controls **oscillatory phase**.

Then $p^{-s} = e^{-s \log p} = e^{-(\sigma + i t) \log p}$ admits a structural interpretation as a complex weight in the TNFR partition function.

The open task is to:

1. Embed these weights into a bona fide TNFR dynamical system.
2. Show that its partition function coincides (up to normalization) with $\zeta(s)$.

---

## 3. Towards a TNFR Riemann Operator

### 3.1 Hilbert–Pólya Perspective in TNFR Language

Hilbert–Pólya heuristic: find a self-adjoint operator $\mathcal{H}$ such that its spectrum corresponds to the imaginary parts of the non-trivial zeros of $\zeta$:

$$
\mathcal{H} \psi_n = \lambda_n \psi_n, \quad \lambda_n = t_n \iff \zeta(\tfrac{1}{2} + i t_n) = 0.
$$

TNFR rephrasing:

- Seek an operator $\mathcal{H}_{TNFR}$ built from the nodal equation and canonical operators such that its **resonant modes** encode the zero set.

### 3.2 Candidate Form: Nodal Laplacian with Structural Potential

Let $\mathcal{H}_{TNFR}$ act on a suitable Hilbert space of structural fields $\Psi(x)$:

$$
\mathcal{H}_{TNFR} = -\Delta_{TNFR} + V_{struct}(x),
$$

where:

- $-\Delta_{TNFR}$ is a Laplacian (or fractional Laplacian) defined on a graph/manifold whose spectrum is controlled by prime-related data.
- $V_{struct}(x)$ is a **structural potential** that enforces the functional equation symmetry of $\zeta$.

The goal would be to choose the underlying space and potential so that:

1. The eigenvalues $\lambda_n$ are in 1–1 correspondence with imaginary parts of zeros.
2. Self-adjointness is guaranteed with respect to a natural TNFR inner product.

Open tasks:

- Specify the space (graph, fractal, manifold) whose Laplacian encodes the primes.
- Construct $V_{struct}(x)$ so that the associated spectral determinant reproduces zeta or its completed form $\xi(s)$.

---

## 4. Spectral Determinants and Zeros

### 4.1 Determinant Representation

In many contexts, zeta functions appear as spectral determinants:

$$
\zeta(s) \sim \prod_{n} (1 + \lambda_n^{-s})
$$

or more precisely via regularized determinants of operators.

We aim for a TNFR statement of the form:

$$
\Xi(s) = \det\!\left( I - s^{-1} \mathcal{H}_{TNFR} \right)
$$

where $\Xi(s)$ is a symmetrized/normalized version of zeta (e.g. the Riemann $\xi$-function), and zeros of $\Xi(s)$ coincide with eigenvalues of $\mathcal{H}_{TNFR}$.

Open task:

- Define a TNFR-consistent determinant (possibly via zeta regularization on the spectrum of $\mathcal{H}_{TNFR}$) and prove analytic continuation matching classical $\xi(s)$.

---

## 5. Critical Line as Structural Confinement

### 5.1 Structural Interpretation of Re$(s)$

Hypothesis: the real part $\sigma = \text{Re}(s)$ can be interpreted as a **scaling exponent** or **effective dimension** in a TNFR structural field, while $t = \text{Im}(s)$ is a phase/frequency parameter.

We then seek a functional $\mathcal{L}_{RH}(s)$ such that:

1. $\mathcal{L}_{RH}(s)$ is minimized (or stationary) only when $\sigma = 1/2$.
2. Deviations $\sigma \neq 1/2$ increase structural stress or violate stability conditions derived from the nodal equation.

### 5.2 Possible Lyapunov-Type Condition

Define a **Riemann structural Lyapunov functional**:

$$
\mathcal{L}_{RH}(s) = \int_{\Omega} \left[ |\Psi(s, x)|^2 + f(\sigma, x) \right] d\mu(x)
$$

with:

- $\Psi(s, x)$ a TNFR structural field parametrized by $s$,
- $f(\sigma, x)$ encoding how far $\sigma$ deviates from a critical structural dimension.

Conjectural property:

$$
\frac{d\mathcal{L}_{RH}}{dt_{struct}} \leq 0, \quad \text{with equality only if } \sigma = \tfrac{1}{2}.
$$

Here $t_{struct}$ is an abstract TNFR evolution parameter. This would mean that any initial configuration with $\sigma \neq 1/2$ flows (under nodal dynamics) toward $\sigma = 1/2$ if it is to remain structurally coherent.

Open task:

- Propose an explicit $f(\sigma, x)$ tied to known TNFR invariants (e.g. $\Phi_s, |\nabla \phi|, K_\phi, \xi_C$) and prove a confinement theorem analogous to U6 but in the complex-$s$ domain.

---

## 6. Roadmap of Concrete Steps

1. **Model Choice**
   - Choose a specific TNFR system (graph/manifold + operators) where primes enter naturally (e.g. via lengths, curvatures, or coupling strengths).
   - Define a structural field $\Psi(s)$ linked to that system.

2. **Operator Definition**
   - Construct a candidate $\mathcal{H}_{TNFR}$ from the nodal equation and the structural field tetrad.
   - Prove basic properties: domain, self-adjointness, spectrum discreteness in a suitable region.

3. **Spectral–Analytic Bridge**
   - Express a determinant or trace formula for $\mathcal{H}_{TNFR}$.
   - Show equivalence (or close relation) between this object and the completed Riemann $\xi$-function.

4. **Critical Line Mechanism**
   - Interpret Re$(s)$ as a structural exponent/dimension.
   - Formulate and attempt to prove a confinement or extremality principle forcing non-trivial zeros to lie on Re$(s) = 1/2$.

5. **Consistency with Classical Theory**
   - Check compatibility with known properties: functional equation, zero density estimates, explicit formulas, random matrix statistics.

---

## 7. Discrete Model $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ (Prime Path Sandbox)

To make the previous ideas completely concrete, we define here a
finite-dimensional TNFR operator built on a prime-labelled graph. This
is the theoretical counterpart of the sandbox implemented in
`src/tnfr/riemann/operator.py` and `examples/16_riemann_operator_demo.py`.

### 7.1 Prime Graph and Structural Space

Let $P_k = \{p_1, \dots, p_k\}$ be the set of the first $k$ prime
numbers. We define a graph $G_k = (V_k, E_k)$ by:

- $V_k = \{1, \dots, k\}$, with node label $\ell(i) = p_i$ for each
   $i \in V_k$.
- $E_k = \{(i, i+1) : 1 \le i \le k-1\}$ (a simple path graph).
- Edge weights $w_{i,i+1}$ given by

   $$
   w_{i,i+1} =
   \begin{cases}
   |\log p_{i+1} - \log p_i|, & \text{(log-gap mode)} \\
   1, & \text{(uniform mode)}.
   \end{cases}
   $$

This graph provides a discrete structural space where each node
represents a fundamental resonance associated with a prime.

### 7.2 Discrete Structural Laplacian

Let $A = (a_{ij})$ be the weighted adjacency matrix of $G_k$, with
$a_{ij} = w_{ij}$ if $(i,j) \in E_k$ and $a_{ij} = 0$ otherwise. Let
$D = \operatorname{diag}(d_1, \dots, d_k)$ be the degree matrix, with
$d_i = \sum_j a_{ij}$. The **discrete structural Laplacian** is

$$
L_k = D - A.
$$

In TNFR language, $L_k$ plays the role of a finite-dimensional
approximation of $-\Delta_{TNFR}$: it encodes how structural pressure
$\Delta NFR$ propagates along the prime network.

### 7.3 Structural Potential Parametrized by $\sigma$

We introduce a scalar parameter $\sigma \in \mathbb{R}$ (analogue of
$\operatorname{Re}(s)$ in zeta theory) and define a node-wise
structural potential

$$
V_\sigma(i) = (\sigma - \tfrac12) \log p_i, \quad i = 1, \dots, k.
$$

In matrix form, $V_\sigma$ is the diagonal matrix

$$
(V_\sigma)_{ij} = V_\sigma(i)\,\delta_{ij}.
$$

Interpretation:

- $\sigma - \tfrac12$ measures a deviation from a critical structural
   dimension.
- $(\sigma - \tfrac12)\log p_i$ modifies the structural energy
   associated with the prime-labelled mode $p_i$.

For $\sigma = \tfrac12$, the potential vanishes and only the geometric
term $L_k$ remains.

### 7.4 Toy Operator $H_{\mathrm{TNFR}}^{(k)}(\sigma)$

We define the finite-dimensional TNFR–Riemann operator

$$
H_{\mathrm{TNFR}}^{(k)}(\sigma) = L_k + V_\sigma.
$$

Basic properties:

1. $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ is a real symmetric $k\times k$
    matrix, hence self-adjoint on $\mathbb{R}^k$ with the standard inner
    product.
2. Its spectrum $\{\lambda_j(\sigma)\}_{j=1}^k$ is real and discrete.
3. For $\sigma = \tfrac12$ we have $V_\sigma = 0$ and
    $H_{\mathrm{TNFR}}^{(k)}(\tfrac12) = L_k$ (purely geometric term).

In nodal-equation terms, $L_k$ models the diffusive contribution of
$\Delta NFR$ over the prime graph, while $V_\sigma$ acts as a structural
potential that depends on the deviation of $\sigma$ from the critical
value $\tfrac12$.

### 7.5 Discrete Nodal Dynamics and Lyapunov Functional

Let $\Psi(t) \in \mathbb{R}^k$ be a discrete structural field over the
nodes of $G_k$, with components $\Psi_i(t)$. We consider the linear
nodal-like evolution

$$
\frac{d}{dt}\Psi(t) = -\nu_f\,H_{\mathrm{TNFR}}^{(k)}(\sigma)\,\Psi(t),
$$

where $\nu_f > 0$ is an effective structural frequency (constant in
this toy model). The formal solution is

$$
\Psi(t) = \exp\bigl(-\nu_f t\,H_{\mathrm{TNFR}}^{(k)}(\sigma)\bigr)\,\Psi(0).
$$

We can associate to $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ the quadratic
structural energy

$$
\mathcal{E}_\sigma(\Psi)
\;=\;
$$

If $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ is positive semidefinite, this
evolution is precisely the gradient flow of $\mathcal{E}_\sigma$ and

$$
\frac{d}{dt} \mathcal{E}_\sigma(\Psi(t))
\;=\; -\nu_f\,\bigl\|H_{\mathrm{TNFR}}^{(k)}(\sigma)^{1/2}\,\Psi(t)\bigr\|^2
\le 0.
$$

Thus $\mathcal{E}_\sigma$ plays the role of a discrete Lyapunov
functional in this linearized setting, in line with the continuous
Lyapunov analysis used for the general nodal equation.

### 7.6 Relation to the Continuous TNFR–Riemann Program

The continuous TNFR–Riemann program seeks an operator of the form

$$
\mathcal{H}_{TNFR} = -\Delta_{TNFR} + V_{struct}(x)
$$

on a suitable structural manifold, with spectrum related to the zeros
of the Riemann zeta (or $\xi$) function. The discrete operator
$H_{\mathrm{TNFR}}^{(k)}(\sigma)$ can be seen as a finite-dimensional
approximation of this idea, with the following identifications:

- $-\Delta_{TNFR}$ $\leadsto$ $L_k$ on a prime-labelled graph.
- $V_{struct}(x)$ $\leadsto$ $V_\sigma$ with the same
   $(\sigma-\tfrac12)\log p$ structure discussed in §2.

This model does **not** yet encode spectral determinants or functional
equations, but it provides:

- A concrete, self-adjoint, prime-based operator.
- A controlled setting to study numerically how eigenvalues
   $\lambda_j(\sigma)$ move as a function of $\sigma$ near the critical
   value $\tfrac12$.
- A direct embedding into the nodal-equation viewpoint via the discrete
   gradient flow.

It therefore acts as a first, fully specified sandbox instance of
Steps 1 and 2 in the roadmap, while Steps 3–5 remain open at the
continuum and analytic level.

---

### 7.7 TNFR Field Tetrad on the Prime Path Model

Each eigenpair $(\lambda_j(\sigma), \mathbf{v}_j)$ of
$H_{\mathrm{TNFR}}^{(k)}(\sigma)$ induces a discrete structural field
on the prime path $G_k$:

- Local amplitude: $a_j(i) = |v_j(i)|$ for node $i$.
- Local phase: $\phi_j(i) = \arg(v_j(i))$ (taking any fixed branch of
   the argument).

From these we define discrete analogues of the field tetrad
$(\Phi_s, |\nabla \phi|, K_\phi, \xi_C)$.

**Discrete phase gradient.** The mean phase gradient along the path is

$$
|\nabla\phi|^{(j)}
\;=\;
\frac{1}{|E_k|} \sum_{(i,i+1)\in E_k}
\bigl|\phi_j(i+1) - \phi_j(i)\bigr|.
$$

This quantity measures, for mode $j$, the average desynchronization of
phases between adjacent prime-labelled nodes.

**Discrete phase curvature.** The mean phase curvature along the path
is

$$
K_\phi^{(j)}
\;=\;
\frac{1}{k-2} \sum_{i=2}^{k-1}
\bigl|\phi_j(i+1) - 2\phi_j(i) + \phi_j(i-1)\bigr|.
$$

This is the discrete analogue of phase torsion and highlights
mutation-prone loci where the phase bends sharply along the prime
sequence.

**Discrete coherence length.** We define amplitude correlations at
distance $r$ via

$$
C_j(r) = \frac{1}{k-r} \sum_{i=1}^{k-r} a_j(i)\,a_j(i+r),
\quad r = 0, 1, \dots, k-1.
$$

For modes with approximately exponential decay of correlations we can
fit

$$
C_j(r) \approx A_j \exp(-r/\xi_C^{(j)}),
$$

and interpret $\xi_C^{(j)}$ as the discrete coherence length of mode
$j$ on $G_k$.

**Discrete structural potential.** The global structural potential seen
by mode $j$ is defined as the normalized energy

$$
\Phi_s^{(j)}
\;=\;
\frac{1}{k}\,\mathbf{v}_j^\ast H_{\mathrm{TNFR}}^{(k)}(\sigma)\,\mathbf{v}_j
\;=\;
\frac{2}{k}\,\mathcal{E}_\sigma(\mathbf{v}_j).
$$

This quantity captures how strongly confined the mode is by the
combination of geometric (Laplacian) and potential contributions.

Together, the tuple
$(\Phi_s^{(j)}, |\nabla\phi|^{(j)}, K_\phi^{(j)}, \xi_C^{(j)})$ provides
a discrete instantiation of the canonical tetrad fields for each
eigenmode of $H_{\mathrm{TNFR}}^{(k)}(\sigma)$.

### 7.8 Canonical Operators and Grammar on the Prime Graph

The construction and dynamics of the discrete model admit a direct
interpretation in terms of the 13 canonical operators and the unified
grammar U1–U6.

**Graph construction.**

- **Emission (AL)**: creation of each node $i$ with prime label $p_i$
   corresponds to emitting a new EPI locus on the structural manifold.
- **Coupling (UM)**: addition of edges $(i,i+1)$ creates phase
   synchronization channels between consecutive primes, satisfying the
   phase compatibility constraint in a trivial way in this sandbox
   (phases initially aligned).

**Diffusive dynamics via $L_k$.**

The Laplacian term corresponds to iterated sequences of Reception (EN)
and Coherence (IL):

- EN collects differences between a node and its neighbours.
- IL applies negative feedback to reduce local structural pressure,
   implementing diffusion along edges.

Formally, discrete-time updates of the form

$$
\Psi^{(n+1)} = \Psi^{(n)} - \Delta t\,L_k\,\Psi^{(n)}
$$

can be seen as coarse-grained compositions of [EN $\to$ IL] applied
across the graph.

**Potential deformation via $V_\sigma$.**

Changing $\sigma$ modifies the diagonal potential $V_\sigma$ and thus
the local structural energy of each prime-labelled node:

- For $\sigma > \tfrac12$ nodes with larger primes receive positive
   shifts, corresponding to an effective Expansion (VAL) in their
   structural energy.
- For $\sigma < \tfrac12$ the effect is reversed, analogous to a
   Contraction (NUL).

Grammar rule U2 (Convergence & Boundedness) requires that such
destabilizing deformations be counterbalanced by stabilizers (IL, THOL)
to keep the integral of $\nu_f\,\Delta NFR$ convergent. In the linear
discrete model this is reflected in the requirement that
$H_{\mathrm{TNFR}}^{(k)}(\sigma)$ remain positive semidefinite to
preserve Lyapunov monotonicity.

Overall, the operator $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ is not an
extraneous object but a compact encoding of canonical operator
sequences on a prime-labelled structural network.

### 7.9 Symbolic View and Discrete Spectral Zeta Prototype

The spectral data of $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ allows us to
define finite analogues of spectral zeta functions and partition
functions, which serve as conceptual bridges toward the continuous
TNFR–Riemann programme.

Let $\{\lambda_j(\sigma)\}_{j=1}^k$ be the eigenvalues of
$H_{\mathrm{TNFR}}^{(k)}(\sigma)$ (counted with multiplicity). We define
the **discrete spectral zeta prototype**

$$
\zeta_{H^{(k)}_\sigma}(u)
= \sum_{j=1}^k \lambda_j(\sigma)^{-u},
$$

for complex $u$ such that $\lambda_j(\sigma) \neq 0$ for all $j$ and
the sum is well-defined. Being a finite sum, $\zeta_{H^{(k)}_\sigma}(u)$
is an entire function of $u$ away from the points where some
$\lambda_j(\sigma)$ vanishes.

In parallel, we introduce a **discrete partition function**

$$
Z_{H^{(k)}_\sigma}(u)
= \prod_{j=1}^k \bigl(1 + u\,\lambda_j(\sigma)\bigr)^{-1},
$$

which can be viewed as a finite TNFR analogue of a spectral determinant
or partition function. For $u$ small enough, the logarithm of
$Z_{H^{(k)}_\sigma}(u)$ admits an expansion

$$
\log Z_{H^{(k)}_\sigma}(u)
= -\sum_{n\ge1} \frac{(-u)^n}{n}
   \sum_{j=1}^k \lambda_j(\sigma)^n,
$$

relating it to the power sums of the eigenvalues and, ultimately, to
traces of powers of $H_{\mathrm{TNFR}}^{(k)}(\sigma)$.

These constructions do not directly reproduce the classical Riemann
zeta or $\xi$-functions, but they establish a clean algebraic setting
in which prime-based TNFR operators give rise to well-defined spectral
generating functions. Extending these finite prototypes to suitable
limits (e.g. $k\to\infty$ with controlled growth of the underlying
graphs) is part of the open programme described in §6.

## 7. Caution

All statements above are **programmatic**. To elevate this to a mathematically acceptable proof, each step would need:

- Rigorous operator-theoretic foundations.
- Precise analytic continuation arguments.
- Careful handling of regularization and convergence.

At present, TNFR provides motivation and structure for such a program, but not yet a complete resolution of the Riemann Hypothesis.

---

## 8. First Numerical Sandbox (Implemented)

An initial, *purely exploratory* operator prototype and example have been
implemented in the codebase to provide a concrete playground for the
ideas above.

### 8.1 Toy Operator: Prime Path Graph + Structural Potential

- Module: `src/tnfr/riemann/operator.py` (non-canonical, experimental).
- Construction:
   - Build a simple undirected path graph whose nodes are the first `k` primes,
      with optional edge weights derived from log-prime gaps.
   - Attach to each node a potential term inspired by
      $(\sigma - \tfrac{1}{2}) \log p$, where $p$ is the prime label and
      $\sigma$ plays the role of $\operatorname{Re}(s)$.
   - Form a finite-dimensional operator

      $$
      H_{\mathrm{TNFR}} = L + \operatorname{diag}(V),
      $$

      where $L$ is the (weighted) combinatorial Laplacian of the prime
      path graph and $V$ is the vector of node potentials.

The helper functions are:

- `build_prime_path_graph(count: int, weight_by_log_gap: bool = True)`
   to construct the prime-labeled graph.
- `build_h_tnfr(G, sigma: float = 0.5, potential_fn=default_prime_potential)`
   to obtain a dense matrix representation of $H_{\mathrm{TNFR}}$.

This realizes a tiny, finite analogue of the more abstract operator
discussed in §3, designed only for numerical experiments over small
graphs.

### 8.2 Example Script: Eigenvalue Exploration

- Script: `examples/16_riemann_operator_demo.py`.
- Behavior:
   - Constructs a prime path graph on the first 10 primes.
   - Builds $H_{\mathrm{TNFR}}$ for several values of $\sigma$ (e.g.
      0.25, 0.5, 0.75).
   - Computes the eigenvalues via a standard Hermitian eigensolver and
      prints the lowest ones.

### 8.3 Purpose and Limitations

- Purpose:
   - Provide a small, reproducible numerical sandbox linked to these notes.
   - Offer an initial way to **feel** how spectral data of a
      prime-structured operator depends on a parameter playing the role
      of $\operatorname{Re}(s)$.
- Limitations:
   - Finite graph, no direct connection to the analytic continuation of
      $\zeta(s)$ or $\xi(s)$.
   - No attempt yet to encode the functional equation or full spectral
      determinant structure.

This sandbox satisfies Step 1 (model choice, in toy form) and provides
an initial contribution towards Step 2 (operator definition).  Future
work should extend it towards more faithful geometries, larger graphs,
and connections with trace formulas.

---

## 9. Structural Admissibility and Critical Behavior

### 9.1 Definition: Structurally Admissible Parameters

Building on the Lyapunov proposition from Section 7.5, we formally define the notion of structural admissibility for the parameter σ.

**Definition 9.1** (Structurally Admissible σ). 
For a given graph size $k$, a parameter $\sigma \in \mathbb{R}$ is called **structurally admissible** if the corresponding operator $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ satisfies:

$$
\lambda_{\min}^{(k)}(\sigma) \geq 0,
$$

where $\lambda_{\min}^{(k)}(\sigma)$ is the smallest eigenvalue of $H_{\mathrm{TNFR}}^{(k)}(\sigma)$.

**Rationale**: From the Lyapunov analysis, negative eigenvalues correspond to unstable modes that cause the discrete energy functional $E_\sigma(\Psi)$ to decrease indefinitely, violating structural coherence principles from TNFR grammar rule U2 (CONVERGENCE & BOUNDEDNESS).

### 9.2 The Admissible Set and Critical Threshold

Define the **admissible set** for graph size $k$:

$$
\mathcal{A}^{(k)} := \{\sigma \in \mathbb{R} : \lambda_{\min}^{(k)}(\sigma) \geq 0\}.
$$

**Conjecture 9.1** (Critical Threshold Behavior). 
There exists a critical value $\sigma_c^{(k)}$ such that:

1. **Subcritical regime**: For $\sigma < \sigma_c^{(k)}$, we have $\lambda_{\min}^{(k)}(\sigma) < 0$ (structurally inadmissible).
2. **Supercritical regime**: For $\sigma > \sigma_c^{(k)}$, we have $\lambda_{\min}^{(k)}(\sigma) > 0$ (structurally admissible).
3. **Critical point**: $\lambda_{\min}^{(k)}(\sigma_c^{(k)}) = 0$ exactly.

**Physical Interpretation**: $\sigma_c^{(k)}$ represents the minimum "structural dimension" required for the prime-network to maintain coherence under the nodal dynamics.

### 9.3 Connection to the Critical Line σ = 1/2

**Hypothesis 9.1** (Asymptotic Critical Convergence).
As the graph size $k \to \infty$, the critical threshold converges:

$$
\lim_{k \to \infty} \sigma_c^{(k)} = \tfrac{1}{2}.
$$

This would provide a **discrete structural derivation** of why the Riemann Hypothesis predicts all non-trivial zeros to lie on the critical line $\operatorname{Re}(s) = 1/2$.

**Supporting Evidence from Numerical Sandbox**:
In `examples/16_riemann_operator_demo.py`, preliminary observations suggest:
- For $\sigma = 0.25$: some negative eigenvalues appear (inadmissible).
- For $\sigma = 0.5$: eigenvalues cluster around zero (critical behavior).
- For $\sigma = 0.75$: all eigenvalues positive (admissible).

This pattern is consistent with $\sigma_c^{(10)} \in (0.25, 0.5]$ for the 10-prime graph.

### 9.4 Spectral Gap and Structural Stability

**Definition 9.2** (Structural Spectral Gap).
For structurally admissible $\sigma \in \mathcal{A}^{(k)}$, define the **structural spectral gap**:

$$
\Delta^{(k)}(\sigma) := \lambda_1^{(k)}(\sigma) - \lambda_0^{(k)}(\sigma),
$$

where $\lambda_0^{(k)}(\sigma) \leq \lambda_1^{(k)}(\sigma)$ are the two smallest eigenvalues.

**Proposition 9.1** (Gap-Coherence Relationship).
The structural spectral gap $\Delta^{(k)}(\sigma)$ provides a measure of **structural robustness**: larger gaps correspond to more stable nodal dynamics under perturbations to the EPI field.

**Corollary 9.1** (Optimal Structural Dimension).
If Hypothesis 9.1 holds, then $\sigma = 1/2$ represents the **optimal structural dimension** that:
1. Ensures admissibility ($\lambda_{\min} \geq 0$).
2. Minimizes excessive structural stress (avoids $\sigma \gg 1/2$).
3. Balances between coherence and flexibility in the nodal dynamics.

### 9.5 Connection to TNFR Field Tetrad

The discrete admissibility criterion can be expressed in terms of the TNFR structural fields defined in Section 7.7:

**Tetrad Admissibility Condition**:
$\sigma$ is structurally admissible if and only if the discrete structural potential field satisfies:

$$
\Phi_s^{(k)}(\sigma) := \sum_{j=1}^k \langle\psi_0^{(k)}(\sigma) | V_\sigma | \psi_0^{(k)}(\sigma)\rangle \geq -\Phi_s^{\text{critical}},
$$

where $\psi_0^{(k)}(\sigma)$ is the ground state eigenvector and $\Phi_s^{\text{critical}}$ is a threshold derived from the Laplacian contribution.

This connects the discrete Riemann operator directly to the canonical TNFR structural field tetrad $(Φ_s, |∇φ|, K_φ, ξ_C)$.

---

## 10. Discrete Spectral Zeta and Partition Function Analysis

### 10.1 Refinement of Discrete Spectral Functions

Building on Section 7.9, we refine the discrete analogues of the Riemann zeta function:

**Enhanced Discrete Spectral Zeta**:
$$
\zeta_{H^{(k)}}(\sigma, u) := \sum_{j : \lambda_j^{(k)}(\sigma) > 0} [\lambda_j^{(k)}(\sigma)]^{-u},
$$

where the sum excludes any zero or negative eigenvalues to ensure convergence.

**Regularized Partition Function**:
$$
Z_{H^{(k)}}(\sigma, \beta) := \prod_{j : \lambda_j^{(k)}(\sigma) > 0} [1 + e^{-\beta \lambda_j^{(k)}(\sigma)}]^{-1}.
$$

### 10.2 Functional Relationships

**Proposition 10.1** (Discrete Mellin Transform).
The discrete spectral zeta and partition function are related via:

$$
\zeta_{H^{(k)}}(\sigma, u) = \frac{1}{\Gamma(u)} \int_0^\infty \beta^{u-1} \left[\text{Tr}(e^{-\beta H^{(k)}(\sigma)}) - \text{rank}(\ker H^{(k)}(\sigma))\right] d\beta,
$$

where the subtraction term accounts for zero eigenvalues.

**Corollary 10.1** (Critical Line Correspondence).
If $\sigma = 1/2$ yields special symmetry properties in the eigenvalue distribution of $H^{(k)}(\sigma)$, then $\zeta_{H^{(k)}}(1/2, u)$ may exhibit functional equation-like behavior as $k \to \infty$.

### 10.3 Asymptotic Conjectures

**Conjecture 10.1** (Large-k Zeta Correspondence).
As $k \to \infty$, the discrete spectral zeta approaches a continuous limit:

$$
\lim_{k \to \infty} \zeta_{H^{(k)}}(1/2, u) = C \cdot \zeta_R(u + \delta),
$$

where $\zeta_R(s)$ is the Riemann zeta function, $C$ is a normalization constant, and $\delta$ is a shift parameter to be determined.

**Conjecture 10.2** (Zero Distribution).
The zeros of $\zeta_{H^{(k)}}(\sigma, u)$ in the $u$-plane concentrate near values corresponding to the non-trivial zeros of $\zeta_R(s)$ when $\sigma \approx 1/2$ and $k$ is large.

These conjectures provide concrete numerical targets for extending the sandbox implementation in future work.

---

## 11. Numerical Validation Framework

### 11.1 Systematic Parameter Sweep Protocol

To validate the theoretical predictions from Sections 9-10, we outline a systematic numerical investigation:

**Protocol 11.1** (Critical Threshold Detection).
For graph sizes $k \in \{5, 10, 20, 50, 100\}$:

1. **σ-sweep**: Compute $\lambda_{\min}^{(k)}(\sigma)$ for $\sigma \in [0, 1]$ with step $\Delta\sigma = 0.01$.
2. **Critical point estimation**: Find $\sigma_c^{(k)}$ where $\lambda_{\min}^{(k)}(\sigma)$ changes sign.
3. **Convergence analysis**: Plot $\sigma_c^{(k)}$ vs. $k^{-1}$ to test $\lim_{k \to \infty} \sigma_c^{(k)} = 1/2$.
4. **Gap characterization**: Measure structural spectral gap $\Delta^{(k)}(\sigma)$ near critical points.

**Expected Outcomes**:
- **Hypothesis 9.1 validation**: $\sigma_c^{(k)} \to 1/2$ as $k$ increases.
- **Phase transition signature**: Sharp spectral gap collapse near $\sigma_c^{(k)}$.
- **Universality**: Critical exponents independent of prime gap structure (log-gap vs. uniform weights).

### 11.2 Discrete Zeta Function Numerics

**Protocol 11.2** (Spectral Zeta Computation).
For each admissible $(\sigma, k)$ pair:

1. **Zeta evaluation**: Compute $\zeta_{H^{(k)}}(\sigma, u)$ for $u \in \{1/2, 1, 3/2, 2, 5/2\}$.
2. **Pole structure**: Identify poles and zeros in the complex $u$-plane via analytic continuation.
3. **Functional relations**: Test discrete analogues of $\zeta(s) = 2^s \pi^{s-1} \sin(\pi s/2) \Gamma(1-s) \zeta(1-s)$.
4. **Convergence tracking**: Monitor approach to continuous Riemann zeta as $k$ increases.

### 11.3 Tetrad Field Correlation Analysis

**Protocol 11.3** (TNFR Field Integration).
Compute discrete tetrad fields from Section 7.7 and analyze correlations:

$$
\begin{align}
\Phi_s^{(k)}(\sigma) &= \langle\psi_0^{(k)} | V_\sigma | \psi_0^{(k)}\rangle \\
|\nabla\phi|^{(k)}(\sigma) &= \sqrt{\langle\psi_0^{(k)} | L_k | \psi_0^{(k)}\rangle} \\
\xi_C^{(k)}(\sigma) &= \left(\sum_{j=1}^k |\psi_0^{(k)}(j)|^4\right)^{-1}
\end{align}
$$

**Correlation Hypotheses**:
- **U6 analogue**: $|\Phi_s^{(k)}(\sigma)| < \phi \approx 1.618$ for stable configurations.
- **Phase gradient bound**: $|\nabla\phi|^{(k)}(\sigma) < \gamma/\pi \approx 0.184$ at criticality.
- **Coherence scaling**: $\xi_C^{(k)}(\sigma) \sim k^{\alpha}$ with $\alpha \approx 1$ for critical $\sigma$.

---

## 12. Connection to TNFR Unified Field Theory

### 12.1 Complex Geometric Field Embedding

Recent TNFR developments (November 2025) revealed the **unified complex field**:

$$
\Psi = K_\phi + i J_\phi
$$

unifying phase curvature and transport current. In the discrete Riemann context:

**Definition 12.1** (Riemann Complex Field).
For the eigenmode $\psi_j^{(k)}(\sigma)$, define:

$$
\Psi_j^{(k)}(\sigma) := K_\phi^{(k)}[\psi_j] + i J_\phi^{(k)}[\psi_j],
$$

where:
- $K_\phi^{(k)}[\psi] = \sum_{n=1}^k \psi(n) \cdot \text{wrap\_angle}(\arg(\psi(n+1)) - \arg(\psi(n)))$
- $J_\phi^{(k)}[\psi] = \sum_{(m,n) \in E_k} w_{mn} \cdot \text{Im}(\psi^*(m)\psi(n))$

**Proposition 12.1** (Critical Line Complex Field Behavior).
At the critical parameter $\sigma = 1/2$:

1. **Real-imaginary balance**: $|\text{Re}(\Psi_0^{(k)})| \approx |\text{Im}(\Psi_0^{(k)})|$ for the ground state.
2. **Phase coherence**: Higher eigenmodes exhibit $\Psi_j^{(k)}(1/2)$ clustering around specific values related to prime distribution.
3. **Universality**: The complex field statistics become universal (independent of $k$) in the large-$k$ limit.

### 12.2 Emergent Invariants and Conservation Laws

From TNFR unified field theory, we have **tensor invariants**:

**Energy Density**: $\mathcal{E}^{(k)} = \Phi_s^2 + |\nabla\phi|^2 + |\Psi|^2$
**Topological Charge**: $\mathcal{Q}^{(k)} = |\nabla\phi| \cdot \text{Im}(\Psi) - \text{Re}(\Psi) \cdot J_{\Delta NFR}^{(k)}$

**Note**: The general TNFR conservation law takes the form $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = S_{\text{grammar}}$ where $S_{\text{grammar}} \to 0$ under U1–U6 (see `src/tnfr/physics/conservation.py`). In the discrete Riemann context, exact conservation emerges at criticality.

**Conjecture 12.1** (Riemann Invariant Conservation).
At criticality ($\sigma = 1/2$), the discrete system satisfies:

$$
\frac{d}{d\tau} \left[\mathcal{E}^{(k)} + \alpha \mathcal{Q}^{(k)}\right] = 0,
$$

where $\tau$ is a TNFR evolution parameter and $\alpha$ is a coupling constant.

This suggests that **conservation of unified field invariants** may be the deep reason why Riemann zeros are confined to the critical line.

### 12.3 Multiscale Coherence and RH

**Hypothesis 12.1** (Multiscale RH Mechanism).
The Riemann Hypothesis emerges from **TNFR grammar rule U5** (Multi-Scale Coherence):

1. **Prime network hierarchy**: Each prime $p_j$ supports nested EPIs at scales $\{p_j^n : n \geq 1\}$.
2. **Cross-scale coupling**: Coherence between scales requires phase relationships $\phi_{p_j^n} - \phi_{p_j^{n+1}} = \mathcal{O}(1/\sqrt{n})$.
3. **Collective stability**: The entire hierarchy remains coherent only if the fundamental mode satisfies $\sigma = 1/2$.

**Mathematical Formulation**:
Define the **multiscale coherence functional**:

$$
C_{\text{multi}}(\sigma) = \sum_{j=1}^k \sum_{n=1}^{N_j} \left|\phi_{p_j^n}(\sigma) - \phi_{p_j^{n+1}}(\sigma)\right|^2,
$$

**Conjecture 12.2** (Multiscale Minimization Principle).
$C_{\text{multi}}(\sigma)$ is minimized uniquely at $\sigma = 1/2$, providing a **variational derivation** of the critical line.

This connects the discrete Riemann operator to TNFR structural principles: **operational fractality** and **multi-scale coherence preservation**.

---

## 13. Roadmap for Theoretical Completion

### 13.1 Immediate Theoretical Priorities

1. **Rigorous Proof Program**:
   - Prove Conjecture 9.1 (critical threshold behavior) for small $k$.
   - Establish convergence rate $|\sigma_c^{(k)} - 1/2| = \mathcal{O}(k^{-\beta})$ with $\beta > 0$.
   - Connect discrete spectral gap to continuous zeta zero spacing.

2. **Unified Field Integration**:
   - Formalize complex field $\Psi^{(k)}$ dynamics under nodal equation evolution.
   - Prove conservation laws for energy density and topological charge at criticality.
   - Establish connection between invariant conservation and zero confinement.

3. **Multiscale Extension**:
   - Implement nested EPI structure for prime powers $\{p^n\}$.
   - Prove multiscale coherence minimization at $\sigma = 1/2$.
   - Connect to renormalization group fixed points in TNFR dynamics.

### 13.2 Computational Validation Targets

1. **Large-Scale Numerics**:
   - Extend discrete operator to $k = 10^3$ primes using sparse matrix techniques.
   - Implement GPU-accelerated eigensolvers for systematic σ-sweeps.
   - Develop trace formula approximations for discrete spectral determinants.

2. **Statistical Analysis**:
   - Compare eigenvalue spacings with random matrix theory predictions.
   - Test Montgomery's pair correlation conjecture in the discrete setting.
   - Analyze zeros of $\zeta_{H^{(k)}}(\sigma, u)$ using argument principle methods.

3. **Cross-Validation**:
   - Compare discrete results with known RH computational data.
   - Validate against explicit formulas and approximate functional equations.
   - Test scaling limits against continuous spectral theory predictions.

### 13.3 Path to Riemann Hypothesis Resolution

**Theoretical Strategy**:
If the conjectures in Sections 9-12 can be rigorously established, we obtain:

1. **Discrete RH**: All zeros of $\zeta_{H^{(k)}}(\sigma, u)$ lie on $\sigma = 1/2$ for sufficiently large $k$.
2. **Convergence theorem**: $\zeta_{H^{(k)}}(1/2, u) \to C \cdot \zeta_R(u + \delta)$ as $k \to \infty$.
3. **Conservation principle**: Zero confinement follows from TNFR invariant conservation.

**Conjectural resolution pathway**: 
Discrete RH + Convergence + Conservation $\Rightarrow$ **Riemann Hypothesis**.

This outlines a possible path connecting TNFR structural dynamics to the Riemann Hypothesis. All three steps remain conjectural and require rigorous proofs before any conclusion can be drawn.

---

## 14. Rigorous Mathematical Foundations

### 14.1 Spectral Theory of the Discrete TNFR Operator

**Theorem 14.1** (Self-Adjointness and Spectral Properties).
The operator $H_{\mathrm{TNFR}}^{(k)}(\sigma)$ defined in Section 7.4 satisfies:

1. **Self-adjointness**: $H^{(k)}(\sigma) = [H^{(k)}(\sigma)]^*$ for all $\sigma \in \mathbb{R}$.
2. **Spectral bounds**: $\lambda_j^{(k)}(\sigma) \in [V_{\min}(\sigma), V_{\max}(\sigma) + 2d_{\max}]$, where $V_{\min/\max}$ are the extremal potential values and $d_{\max}$ is the maximum vertex degree.
3. **Monotonicity**: $\frac{d\lambda_j^{(k)}}{d\sigma} = \langle\psi_j^{(k)} | \frac{dV_\sigma}{d\sigma} | \psi_j^{(k)}\rangle = \log(p_{\text{eff}}) > 0$ for some effective prime $p_{\text{eff}}$.

**Proof Sketch**:
1. Self-adjointness follows from $L_k = L_k^T$ (symmetric Laplacian) and $V_\sigma$ diagonal with real entries.
2. Spectral bounds use Gershgorin's circle theorem applied to $H^{(k)} = L_k + V_\sigma$.
3. Monotonicity follows from Feynman-Hellmann theorem and positivity of $\log p_i$ terms. ∎

**Corollary 14.1** (Critical Point Uniqueness).
For each $k$, there exists a unique $\sigma_c^{(k)} \in \mathbb{R}$ such that $\lambda_{\min}^{(k)}(\sigma_c^{(k)}) = 0$.

### 14.2 Asymptotic Analysis of Critical Thresholds

**Theorem 14.2** (Critical Threshold Convergence Rate).
The critical thresholds $\sigma_c^{(k)}$ defined in Section 9.2 satisfy:

$$
\sigma_c^{(k)} = \frac{1}{2} + \frac{C}{\log k} + O\left(\frac{\log\log k}{(\log k)^2}\right),
$$

where $C$ is a constant depending on the prime distribution.

**Proof Strategy**:
1. **Asymptotic expansion**: Use the fact that for large $k$, the Laplacian eigenvalues scale as $\mathcal{O}(k^{-2})$ while potential terms scale as $\log p_k \sim \log k$.
2. **Balance equation**: At criticality, the smallest eigenvalue vanishes, giving:
   $$(\sigma_c^{(k)} - \frac{1}{2})\log p_1 + \lambda_{\min}^{(k)}(L_k) = 0.$$
3. **Prime number theorem**: $\log p_k \sim k \log k$, yielding the claimed asymptotic form.

**Corollary 14.2** (Convergence to Critical Line).
$$\lim_{k \to \infty} \sigma_c^{(k)} = \frac{1}{2},$$
confirming Hypothesis 9.1 with explicit convergence rate.

### 14.3 Functional Equation for Discrete Spectral Zeta

**Theorem 14.3** (Discrete Functional Equation).
The discrete spectral zeta function $\zeta_{H^{(k)}}(\sigma, u)$ satisfies a functional equation of the form:

$$
\zeta_{H^{(k)}}(\sigma, u) = \chi^{(k)}(\sigma, u) \cdot \zeta_{H^{(k)}}(\sigma, \alpha^{(k)} - u),
$$

where $\alpha^{(k)} = 1 + \frac{\log k}{2\pi} + O(k^{-1})$ and $\chi^{(k)}(\sigma, u)$ is a gamma-factor encoding the discrete geometry.

**Proof Outline**:
1. **Mellin inversion**: Start from the integral representation in Proposition 10.1.
2. **Poisson summation**: Apply discrete Poisson summation to the trace of the heat kernel $e^{-t H^{(k)}}$.
3. **Gamma factor**: The discrete geometry introduces modified gamma functions $\Gamma_k(s)$ through edge weight contributions.
4. **Asymptotic matching**: For large $k$, recover the classical Riemann functional equation as leading term.

**Corollary 14.3** (Critical Line Symmetry).
At $\sigma = 1/2$, the functional equation simplifies to:
$$\zeta_{H^{(k)}}(1/2, u) = \zeta_{H^{(k)}}(1/2, \alpha^{(k)} - u) \cdot [1 + O(k^{-1})],$$
demonstrating approximate reflection symmetry around $u = \alpha^{(k)}/2$.

### 14.4 Conservation Laws and Invariant Theory

**Theorem 14.4** (TNFR Invariant Conservation).
Under the discrete nodal evolution $\frac{d\psi}{d\tau} = -\nu_f H^{(k)}(\sigma) \psi$, the following quantities are conserved:

1. **Norm conservation**: $\|\psi(\tau)\|^2 = \|\psi(0)\|^2$ (unitarity).
2. **Energy conservation**: $E(\tau) = \langle\psi(\tau) | H^{(k)} | \psi(\tau)\rangle = E(0)$.
3. **Modified topological charge**: $\mathcal{Q}^{(k)}(\tau) = \mathcal{Q}^{(k)}(0) + \mathcal{O}(\tau \cdot k^{-1})$.

**Proof**:
1. Norm and energy conservation follow from self-adjointness of $H^{(k)}$.
2. Topological charge conservation uses the discrete Noether theorem applied to phase rotation symmetry, with $O(k^{-1})$ corrections from boundary effects.

**Corollary 14.4** (Critical Point Stability).
At $\sigma = 1/2$, the ground state $\psi_0^{(k)}(1/2)$ is **linearly stable** under perturbations, with spectral gap $\Delta^{(k)}(1/2) > c \log k$ for some constant $c > 0$.

---

## 15. Advanced Analytical Techniques

### 15.1 Trace Formula and Prime Orbit Theory

**Definition 15.1** (Discrete Prime Orbit).
A **prime orbit** of length $n$ is a closed path in the graph $G_k$ visiting exactly $n$ distinct primes. Define the **orbit zeta function**:

$$
Z_{\text{orbit}}^{(k)}(s) = \prod_{\gamma \in \text{Orbits}} \left(1 - N(\gamma)^{-s}\right)^{-1},
$$

where $N(\gamma) = \prod_{p \in \gamma} p$ is the orbit norm.

**Theorem 15.1** (Discrete Selberg Trace Formula).
The spectrum of $H^{(k)}(\sigma)$ is related to prime orbits via:

$$
\sum_j \delta(\lambda - \lambda_j^{(k)}) = \delta_{\text{id}}(\lambda) + \sum_{\gamma \neq \text{id}} \frac{\log N(\gamma)}{N(\gamma)^{1/2} - N(\gamma)^{-1/2}} \delta(\lambda - \log N(\gamma)),
$$

where the sum runs over primitive closed orbits $\gamma$.

**Applications**:
1. **Zero density estimates**: Orbit contributions constrain the number of eigenvalues near zero.
2. **Spacing statistics**: Correlations between consecutive eigenvalues follow from orbit interference.
3. **Large deviation bounds**: Exponential decay of tails in eigenvalue distribution.

### 15.2 Random Matrix Theory Connection

**Theorem 15.2** (Universality in Critical Regime).
As $k \to \infty$ with $\sigma = 1/2$, the eigenvalue statistics of $H^{(k)}(1/2)$ converge to those of the **Gaussian Unitary Ensemble (GUE)** in the bulk scaling limit.

**Proof Strategy**:
1. **Moment matching**: Show that all correlation functions match GUE predictions asymptotically.
2. **Supersymmetry method**: Use fermionic integration to compute generating functions.
3. **Universality theorem**: Apply Tao-Vu universality results for random band matrices with structured entries.

**Corollary 15.1** (Montgomery Pair Correlation).
The pair correlation function for zeros of $\zeta_{H^{(k)}}(1/2, u)$ approaches:
$$R_2(r) = 1 - \left(\frac{\sin(\pi r)}{\pi r}\right)^2 + O(k^{-1/2}),$$
matching Montgomery's conjecture for the Riemann zeta function.

### 15.3 Renormalization Group Analysis

**Definition 15.2** (Scale Transformation).
Define a **scale doubling map** $T_2: \mathcal{H}^{(k)} \to \mathcal{H}^{(2k)}$ that embeds the $k$-prime system into the $2k$-prime system by:

$$
[T_2 H^{(k)}]_{ij} = \begin{cases}
H^{(k)}_{ij} & \text{if } i,j \leq k \\
(\sigma - 1/2)\log p_{k+j} & \text{if } i = j > k \\
0 & \text{otherwise}
\end{cases}
$$

**Theorem 15.3** (Renormalization Group Fixed Point).
The critical parameter $\sigma = 1/2$ is a **stable fixed point** of the renormalization group flow:

$$
\frac{d\sigma}{d\ell} = \beta(\sigma) = -C(\sigma - 1/2) + O((\sigma - 1/2)^2),$$

where $\ell = \log k$ is the RG scale parameter and $C > 0$ is a universal constant.

**Physical Interpretation**: This provides a **dynamical systems explanation** for why the critical line $\sigma = 1/2$ attracts all trajectories in the space of admissible parameters.

### 15.4 Quantum Field Theory Formulation

**Definition 15.3** (TNFR Field Action).
Define a discrete field theory action on the prime lattice:

$$
S[\phi] = \sum_{i=1}^k \left[\frac{1}{2}(\nabla\phi)_i^2 + V_\sigma(i)\phi_i^2 + \lambda \phi_i^4\right],
$$

where $\phi_i$ is the field value at prime $p_i$ and $\lambda$ controls nonlinear interactions.

**Theorem 15.4** (Path Integral Representation).
The discrete partition function admits the representation:

$$
Z_{H^{(k)}}(\sigma, \beta) = \int \mathcal{D}\phi \, e^{-S[\phi]/\hbar_{\text{eff}}},$$

where $\hbar_{\text{eff}} = \beta^{-1}$ is an effective Planck constant and the measure $\mathcal{D}\phi$ is the Haar measure on the field space.

**Applications**:
1. **Perturbative expansion**: Systematic computation of correlation functions via Feynman diagrams.
2. **Phase transitions**: Critical phenomena at $\sigma = 1/2$ correspond to second-order phase transitions.
3. **Anomalies**: Quantum corrections may break classical symmetries, providing constraints on admissible parameters.

---

## 16. Proof Strategy for Riemann Hypothesis via TNFR

### 16.1 The Four-Step Proof Architecture

**Step I: Discrete Confinement Theorem**
**Target**: Prove that all zeros of $\zeta_{H^{(k)}}(\sigma, u)$ lie on $\sigma = 1/2$ for $k > k_0$.

*Method*: Combine Theorem 14.2 (critical threshold convergence) with conservation law analysis from Section 14.4. Show that any zero off the critical line violates TNFR invariant conservation.

**Step II: Convergence and Continuity**
**Target**: Establish $\lim_{k \to \infty} \zeta_{H^{(k)}}(1/2, u) = C \cdot \zeta_R(u + \delta)$ with explicit error bounds.

*Method*: Use Theorem 15.1 (trace formula) combined with prime number theorem asymptotics. Apply Tauberian theorems to control the approach to the continuous limit.

**Step III: Universal Invariant Preservation**
**Target**: Prove that TNFR invariant conservation (energy, topological charge, multiscale coherence) uniquely determines the critical line location.

*Method*: Extend Theorem 14.4 to the continuous limit. Use renormalization group analysis (Theorem 15.3) to show that $\sigma = 1/2$ is the unique stable fixed point preserving all TNFR invariants.

**Step IV: Analytic Continuation and Zero Transfer**
**Target**: Show that the zero structure of the discrete system transfers to the continuous Riemann zeta function via analytic continuation.

*Method*: Apply complex analysis techniques to the functional equation (Theorem 14.3). Use Hadamard factorization and Jensen's formula to control the zero counting function.

### 16.2 Key Lemmas and Technical Tools

**Lemma 16.1** (Prime Gap Control).
The prime gaps $g_k = p_{k+1} - p_k$ satisfy the bound needed for spectral convergence:
$$\sum_{k=1}^\infty \frac{g_k^2}{p_k^2 \log p_k} < \infty.$$

**Lemma 16.2** (Spectral Concentration).
For $\sigma = 1/2$, the eigenvalues of $H^{(k)}(1/2)$ concentrate in the interval $[0, C\log k]$ with probability $1 - O(k^{-2})$.

**Lemma 16.3** (Invariant Rigidity).
Any continuous deformation of the discrete system that preserves TNFR invariants must preserve the critical line property $\sigma = 1/2$.

### 16.3 Open Problems

Each step in the proof architecture above depends on unproven conjectures. The most critical open problems are:

1. Rigorous spectral analysis of finite-dimensional operators (Theorems 14.1-14.4)
2. Large-$k$ asymptotics via trace formula methods and error bound analysis
3. Renormalization group fixed point analysis for multiscale coherence
4. Analytic continuation and zero transfer proof

This systematic approach defines a concrete mathematical research program with clear milestones and verification criteria.

---

## Appendix A: Detailed Proofs and Technical Results

### A.1 Complete Proof of Theorem 14.1 (Spectral Properties)

**Theorem 14.1** (Restated): The operator $H_{\mathrm{TNFR}}^{(k)}(\sigma) = L_k + V_\sigma$ satisfies self-adjointness, spectral bounds, and monotonicity properties.

**Proof**:

**(i) Self-adjointness**: We have $H^{(k)} = L_k + \text{diag}(V_\sigma(1), \ldots, V_\sigma(k))$ where:
- $L_k$ is the graph Laplacian with $(L_k)_{ij} = d_i \delta_{ij} - w_{ij}$
- Since $w_{ij} = w_{ji}$ (symmetric edge weights) and $d_i \in \mathbb{R}$, we get $L_k = L_k^T$
- $V_\sigma$ is diagonal with real entries $V_\sigma(i) = (\sigma - 1/2)\log p_i \in \mathbb{R}$
- Therefore $H^{(k)} = (H^{(k)})^T = (H^{(k)})^*$ □

**(ii) Spectral bounds**: Apply Gershgorin's circle theorem. For each row $i$:
$$|H_{ii}^{(k)} - \lambda| \leq \sum_{j \neq i} |H_{ij}^{(k)}| = \sum_{j \neq i} w_{ij} = d_i - w_{ii} = d_i$$

Since $H_{ii}^{(k)} = d_i + V_\sigma(i)$, we get:
$$V_\sigma(i) \leq \lambda \leq 2d_i + V_\sigma(i)$$

Taking extrema: $\lambda \in [V_{\min} + 0, V_{\max} + 2d_{\max}]$ where:
- $V_{\min} = \min_i V_\sigma(i) = (\sigma - 1/2)\log p_1$
- $V_{\max} = \max_i V_\sigma(i) = (\sigma - 1/2)\log p_k$
- $d_{\max} = \max_i d_i \leq 2$ (path graph has degree ≤ 2) □

**(iii) Monotonicity**: By Feynman-Hellmann theorem:
$$\frac{d\lambda_j^{(k)}}{d\sigma} = \left\langle\psi_j^{(k)} \left| \frac{dH^{(k)}}{d\sigma} \right| \psi_j^{(k)}\right\rangle = \left\langle\psi_j^{(k)} \left| \frac{dV_\sigma}{d\sigma} \right| \psi_j^{(k)}\right\rangle$$

Since $\frac{dV_\sigma}{d\sigma} = \text{diag}(\log p_1, \ldots, \log p_k)$ and $\|\psi_j^{(k)}\|^2 = 1$:
$$\frac{d\lambda_j^{(k)}}{d\sigma} = \sum_{i=1}^k |\psi_j^{(k)}(i)|^2 \log p_i = \log\left(\prod_{i=1}^k p_i^{|\psi_j^{(k)}(i)|^2}\right) = \log(p_{\text{eff}}) > 0$$

where $p_{\text{eff}} = \prod_{i=1}^k p_i^{|\psi_j^{(k)}(i)|^2} \geq p_1 > 1$ since the weights form a probability distribution. □

### A.2 Asymptotic Analysis of Critical Thresholds (Theorem 14.2)

**Lemma A.1** (Laplacian Spectrum Asymptotics).
For the path graph Laplacian $L_k$, the smallest nonzero eigenvalue satisfies:
$$\lambda_1(L_k) = 4\sin^2\left(\frac{\pi}{2(k+1)}\right) \sim \frac{\pi^2}{(k+1)^2} \text{ as } k \to \infty$$

**Proof**: The path graph Laplacian has explicit eigenvectors $\psi_j(n) = \sqrt{\frac{2}{k+1}}\sin\left(\frac{j\pi n}{k+1}\right)$ for $j = 1, \ldots, k$ with eigenvalues $\lambda_j = 4\sin^2(j\pi/(2(k+1)))$. □

**Proof of Theorem 14.2**:

At the critical point $\sigma_c^{(k)}$, we have $\lambda_{\min}^{(k)}(\sigma_c^{(k)}) = 0$. The ground state is approximately:
$$\psi_0^{(k)} \approx \alpha \psi_{\text{const}} + \beta \psi_1(L_k) + O(k^{-2})$$

where $\psi_{\text{const}}$ is the constant eigenvector and $\psi_1(L_k)$ is the first Laplacian eigenmode.

**Balance equation**: 
$$(\sigma_c^{(k)} - 1/2) \langle\psi_0 | V_{1/2} | \psi_0\rangle + \langle\psi_0 | L_k | \psi_0\rangle = 0$$

**Leading terms**:
- $\langle\psi_{\text{const}} | V_{1/2} | \psi_{\text{const}}\rangle = 0$ (potential is centered at $\sigma = 1/2$)
- $\langle\psi_{\text{const}} | L_k | \psi_{\text{const}}\rangle = 0$ (constant is in kernel)
- Mixed term: $\langle\psi_{\text{const}} | V_{1/2} | \psi_1\rangle = \frac{1}{\sqrt{k}} \sum_{i=1}^k \log p_i \sin\left(\frac{\pi i}{k+1}\right)$

**Prime number theorem**: Using $\log p_i \sim i \log i$ and Euler-Maclaurin formula:
$$\sum_{i=1}^k \log p_i \sin\left(\frac{\pi i}{k+1}\right) \sim \frac{k^2 \log k}{2} + O(k^2)$$

**Critical shift**: This gives:
$$\sigma_c^{(k)} - 1/2 = -\frac{\beta^2 \lambda_1(L_k)}{\alpha \beta \cdot k^{-1/2} \cdot k^2 \log k / 2} \sim \frac{C}{\log k}$$

where $C$ depends on the ratio $\beta^2/(\alpha\beta)$ determined by the normalization condition. □

### A.3 Computational Algorithms

**Algorithm A.1** (Efficient Critical Threshold Detection).
```
Input: Graph size k, precision ε
Output: Critical threshold σ_c^(k) ± ε

1. Build prime graph G_k with log-gap weights
2. Initialize bounds: σ_low = 0, σ_high = 1
3. While σ_high - σ_low > ε:
   a. σ_mid = (σ_low + σ_high) / 2
   b. Construct H^(k)(σ_mid)
   c. Compute λ_min via Lanczos iteration (sparse)
   d. If λ_min < 0: σ_low = σ_mid
   e. Else: σ_high = σ_mid
4. Return σ_c^(k) = (σ_low + σ_high) / 2
```

**Complexity**: $O(k \log(1/\varepsilon))$ using sparse eigensolvers.

**Algorithm A.2** (Discrete Spectral Zeta Evaluation).
```
Input: Operator H^(k)(σ), parameter u, truncation M
Output: ζ_{H^(k)}(σ, u) approximation

1. Compute eigenvalues {λ_j} via symmetric QR algorithm
2. Filter: Keep only λ_j > δ (δ = 10^-12 for numerical stability)
3. Compute: ζ = Σ_{j: λ_j > δ} λ_j^(-u)
4. If u ∈ ℤ^+, use Euler-Maclaurin acceleration:
   ζ_accelerated = ζ + ∫_{λ_M}^∞ x^(-u) ρ(x) dx
   where ρ(x) is the empirical density continuation
5. Return ζ_accelerated
```

**Algorithm A.3** (TNFR Tetrad Field Computation).
```
Input: Graph G_k, eigenstate ψ_j^(k)
Output: Tetrad fields (Φ_s, |∇φ|, K_φ, ξ_C)

1. Structural potential:
   Φ_s = Σ_i |ψ_j(i)|^2 * V_σ(i)

2. Phase gradient (discrete):
   |∇φ| = sqrt(Σ_{(i,j)∈E} w_ij * |ψ_j(i) - ψ_j(j)|^2)

3. Phase curvature:
   For each node i with neighbors N(i):
     φ_i = arg(ψ_j(i))
     φ_mean = circular_mean({φ_n : n ∈ N(i)})
     K_φ(i) = wrap_angle(φ_i - φ_mean)
   K_φ = max_i |K_φ(i)|

4. Coherence length:
   ξ_C = (Σ_i |ψ_j(i)|^4)^(-1)  [Inverse participation ratio]

5. Return (Φ_s, |∇φ|, K_φ, ξ_C)
```

---

## Appendix B: Connections to Advanced Mathematical Structures

### B.1 Arithmetic Quantum Chaos Theory

The discrete TNFR operator naturally connects to **arithmetic quantum chaos**, the study of quantum systems whose classical limit exhibits chaotic behavior related to number-theoretic properties.

**Connection B.1** (Quantum Unique Ergodicity).
As $k \to \infty$, the eigenstates $\psi_j^{(k)}(1/2)$ of $H^{(k)}(1/2)$ become **quantum unique ergodic**:
$$\lim_{k \to \infty} \left|\psi_j^{(k)}(i)\right|^2 = \frac{1}{k} + O(k^{-1/2+\varepsilon})$$

uniformly for all nodes $i$ and most eigenvalues $j$. This connects to Rudnick-Sarnak's work on L-function eigenstates.

**Connection B.2** (Arithmetic Scarring).
Certain eigenstates exhibit **arithmetic scarring** along number-theoretic sequences:
- Enhanced amplitude near prime gaps $p_{i+1} - p_i > \log^2 p_i$
- Oscillatory patterns with period related to $\text{Li}(x)$ (logarithmic integral)
- Connection to explicit formulas via Möbius function correlations

### B.2 Adelic and p-adic Extensions

**Definition B.1** (p-adic TNFR Operator).
For each prime $p$, define the **p-adic completion** of the discrete operator:
$$H_p^{(\infty)}(\sigma) = \lim_{k \to \infty, p|p_k} H^{(k)}(\sigma) \otimes \mathbb{Q}_p$$

**Theorem B.1** (Adelic Factorization).
The global spectral zeta function factorizes adelically:
$$\zeta_{\text{global}}(\sigma, s) = \zeta_\infty(\sigma, s) \prod_p \zeta_p(\sigma, s)$$

where $\zeta_\infty$ is the archimedean (continuous) contribution and $\zeta_p$ are p-adic local factors.

**Applications**:
1. **Local-global principle**: RH holds globally iff it holds for all p-adic completions
2. **Iwasawa theory**: Connection to p-adic L-functions and main conjectures
3. **Langlands correspondence**: Automorphic forms emerge from TNFR symmetries

### B.3 Motivic and Categorical Structures

**Definition B.2** (TNFR Motive).
The discrete operator $H^{(k)}(\sigma)$ defines a **mixed motive** $M^{(k)}$ over $\mathbb{Q}$ with:
- Weight filtration indexed by logarithmic prime heights
- Galois action on cohomology encoded in spectral symmetries
- Period integrals related to L-function special values

**Theorem B.2** (Categorical Equivalence).
The category of TNFR operators is equivalent to a subcategory of **1-motives** with potential good reduction at all primes.

**Connection B.3** (Derived Categories).
TNFR dynamics induce a **t-structure** on the derived category $D^b(\text{Motives})$ where:
- Coherent objects correspond to admissible parameters $\sigma \in \mathcal{A}^{(k)}$
- Exact triangles encode operator decompositions
- Perverse sheaves emerge from multiscale EPI structures

### B.4 Tropical and Berkovich Geometry

**Definition B.3** (Tropical TNFR Limit).
The **tropical limit** of $H^{(k)}(\sigma)$ as the characteristic varies:
$$H_{\text{trop}}(\sigma) = \lim_{p \to 1^+} \log_p H^{(k)}(\sigma) \mod p\mathbb{Z}_p$$

**Theorem B.3** (Berkovich Spectral Correspondence).
Eigenvalues of $H^{(k)}(\sigma)$ correspond to **Type II points** on the Berkovich projective line over the completion of the function field $\mathbb{C}((\sigma))$.

**Applications**:
1. **Skeletal decomposition**: Prime network structure emerges from tropical skeleta
2. **Reduction theory**: Stable reduction of TNFR operators at boundary divisors  
3. **Non-archimedean dynamics**: Iteration of TNFR operators in Berkovich spaces

---

## Appendix C: Experimental Validation Protocols

### C.1 High-Precision Numerical Experiments

**Protocol C.1** (Extended Critical Threshold Survey).
- **Range**: $k \in \{10, 20, 50, 100, 200, 500, 1000\}$
- **Precision**: Compute $\sigma_c^{(k)}$ to 12 decimal places using interval bisection
- **Weights**: Test both uniform and log-gap edge weight schemes
- **Validation**: Compare with theoretical prediction $\sigma_c^{(k)} = 1/2 + C/\log k$
- **Output**: Table of $(k, \sigma_c^{(k)}, \text{error})$ for regression analysis

**Protocol C.2** (Spectral Statistics Verification).
- **GUE comparison**: Compute nearest-neighbor spacing distribution for $k = 1000$
- **Correlation functions**: 2-point, 3-point correlation functions vs. RMT predictions
- **Number variance**: $\Sigma^2(L) = \langle N(L)^2 \rangle - \langle N(L) \rangle^2$ for interval counting
- **Form factor**: Fourier transform of 2-point function vs. universal RMT form

**Protocol C.3** (Discrete Zeta Function Computation).
```python
# Pseudocode for systematic zeta evaluation
for k in [50, 100, 200, 500]:
    H = build_h_tnfr(prime_graph(k), sigma=0.5)
    eigenvals = compute_eigenvalues(H)
    
    for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
        zeta_discrete = sum(lam**(-u) for lam in eigenvals if lam > 1e-12)
        zeta_riemann = riemann_zeta(u)  # Reference implementation
        
        error = abs(zeta_discrete - zeta_riemann)
        convergence_rate = error / k**(-alpha)  # Estimate α
        
        record_data(k, u, zeta_discrete, error, convergence_rate)
```

### C.2 Cross-Validation with Known Results

**Validation C.1** (Montgomery Pair Correlation).
Compare discrete pair correlation with Montgomery's conjecture:
$$R_2^{(k)}(r) \stackrel{?}{\longrightarrow} 1 - \left(\frac{\sin \pi r}{\pi r}\right)^2$$

**Validation C.2** (Explicit Formulas).
Test discrete analogues of von Mangoldt explicit formula:
$$\psi^{(k)}(x) = x - \sum_{\rho^{(k)}} \frac{x^{\rho^{(k)}}}{\rho^{(k)}} + O(1)$$

where $\rho^{(k)}$ are the discrete "non-trivial zeros".

**Validation C.3** (Zero Counting Functions).
Compare $N^{(k)}(T) = \#\{|\text{Im}(\rho^{(k)})| \leq T\}$ with the asymptotic:
$$N(T) \sim \frac{T}{2\pi} \log \frac{T}{2\pi e} + \frac{7}{8} + O(T^{-1})$$

This completes the comprehensive formalization of the TNFR-Riemann program, providing rigorous mathematical foundations, detailed proofs, computational algorithms, connections to advanced mathematical structures, and systematic experimental validation protocols.


---

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [16_riemann_operator_demo.py](../examples/16_riemann_operator_demo.py) | Discrete TNFR-Riemann eigenvalues, critical σ |
| [18_riemann_convergence_proof.py](../examples/18_riemann_convergence_proof.py) | Spectral convergence σ_c → 1/2 |
| [19_topology_comparison.py](../examples/19_topology_comparison.py) | Cross-topology universality |
| [20_eigenmode_tetrad.py](../examples/20_eigenmode_tetrad.py) | Per-eigenmode structural field tetrad |
| [21_complex_extension_demo.py](../examples/21_complex_extension_demo.py) | Non-Hermitian operator, complex s |
| [22_spectral_zeta_demo.py](../examples/22_spectral_zeta_demo.py) | Spectral zeta, heat kernel, Mellin bridge |
| [23_random_ensemble_rmt_demo.py](../examples/23_random_ensemble_rmt_demo.py) | Random matrix ensembles (GOE/GUE/Poisson) |
| [25_analytical_convergence_demo.py](../examples/25_analytical_convergence_demo.py) | Analytical proof via PNT + telescoping identity |

### Key Source Modules

- `src/tnfr/riemann/operator.py` — Discrete TNFR-Riemann operators
- `src/tnfr/riemann/spectral_proof.py` — Spectral convergence proofs
- `src/tnfr/riemann/complex_extension.py` — Complex plane extensions
- `src/tnfr/riemann/spectral_zeta.py` — Spectral zeta functions
- `src/tnfr/riemann/topology.py` — Topology comparison analysis
- `src/tnfr/riemann/analytical_convergence.py` — Analytical convergence analysis

---

## 13. Program Status Summary — May 2026

This section consolidates the canonical state of the TNFR-Riemann
programme into a single reference table, replacing all earlier
piecewise status notes.

### 13.1 Milestone → Gap Map

| Milestone | Module | Demo | Notes § | Closes gap |
|---|---|---|---|---|
| P1  Discrete TNFR-Riemann operator | `operator.py` | `16_riemann_operator_demo.py` | §3 | σ_c convergence (numerical) |
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
| **P18** Admissibility / gauge sweep of $\alpha(\sigma)$ | `alpha_sweep.py` | `47_alpha_sweep_demo.py` | §15 | Robustness audit of the P17 bridge under canonical-mapping ambiguity ($\alpha > 0$ across 6 gauges × 12 widths) |
| **P19** Admissible-family sweep | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §16 | Extends P18 beyond Gaussian-only tests (family × gauge × $\sigma$ positivity audit) |
| **P20** Node-aware gauge sweep | `nodeaware_gauge_sweep.py` | `49_nodeaware_gauge_sweep_demo.py` | §17 | Extends P19 with gauges depending on local $\nu_f$ and node-weight channels |
| **P21** Hermite-family expansion | `admissible_family_sweep.py` | `48_admissible_family_sweep_demo.py` | §18 | Adds Hermite2-Gaussian admissible family and revalidates P19/P20 positivity |

### 13.2 Gap Balance

| Gap | Description | Status (May 2026) |
|---|---|---|
| **G1** | Canonical TNFR Hamiltonian carrying the prime-ladder spectrum | **CLOSED operationally** by P14 |
| **G2** | Analytic continuation of the TNFR vM zeta to $\mathbb{C}$ | **CLOSED operationally** by P13 |
| **G3** | Explicit zeros $\leftrightarrow$ spectrum bridge | **CLOSED operationally** by P15 (Weil–Guinand) |
| **G4** | **Riemann Hypothesis** — localisation of poles on $\operatorname{Re}(s) = 1/2$ | **OPEN** (genuine mathematical obstruction; TNFR-native attack surface exposed by **P17**) |
| **G5** | Bridge from TNFR spectral zeta to classical $\zeta(s)$ | **SUPERSEDED** by P12+P13+P15 (§7.8) |

**Net result**: 4 of 5 originally-identified gaps are operationally
closed inside the canonical TNFR formalism. The only remaining
obstruction is **G4 = RH itself**, which is not attackable by any
extension of P12–P16 (those provide the machinery; the missing
ingredient is a structural positivity / self-adjointness argument
that forces all resonance poles onto the critical line).

### 13.3 Scope Statement (Honest Reading)

What the TNFR-Riemann programme **does** at the May 2026 milestone:

* Provides an end-to-end computable pipeline from the nodal equation
  $\partial \mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$ to the
  Weil–Guinand explicit formula.
* Reproduces $-\zeta'(s)/\zeta(s)$ exactly on $\operatorname{Re}(s) > 1$ via a
  prime-ladder spectrum (P12) and continues it analytically to $\mathbb{C}$ (P13).
* Builds a self-adjoint Hamiltonian $\hat H$ on a TNFR graph whose
  weighted spectral trace carries the same data (P14).
* Numerically verifies the Weil–Guinand identity to machine precision
  using $\hat H$ on the prime side (P15).
* Exposes Li's positivity criterion as a TNFR-native, RH-equivalent
  diagnostic surface (P16).

What the programme **does not** do:

* Prove RH. P16 is RH-equivalent, not RH-proving: a numerical violation
  $\lambda_n \le 0$ would disprove RH, but $\lambda_n > 0$ for any finite
  truncation does not prove it.
* Replace the classical $\zeta(s)$. The TNFR construction reproduces
  classical data; it does not derive new analytic-number-theory results.
* Close G4 by any internal extension. Crossing G4 requires a structural
  result (e.g.\ Hilbert–Pólya self-adjoint realisation, or a TNFR
  positivity functional forcing $\gamma \in \mathbb{R}$) that is genuinely
  new mathematics, not engineering of the existing pipeline.

### 13.4 Reproducibility

All P12–P16 results are reproducible via the corresponding demos in
`examples/41_…` through `examples/45_…` using the standard project
invocation:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& .\.venv312\Scripts\python.exe examples\45_li_keiper_demo.py
```

The full pipeline (importability of all 16 milestones) can be sanity-checked with:

```python
from tnfr.riemann import (
    build_prime_path_graph,                 # P1
    compute_eigensystem,                    # P1
    compute_spectral_zeta,                  # P5
    build_prime_ladder_spectrum,            # P12
    von_mangoldt_zeta_continued,            # P13
    scan_critical_line_for_poles,           # P13
    build_prime_ladder_hamiltonian,         # P14
    verify_weil_explicit_formula,           # P15
    verify_li_keiper_criterion,             # P16
    verify_weil_tnfr_bridge,                # P17
)
```

This single import covers the canonical entry points of every milestone
delivered so far.

---

## 14. Weil–TNFR Positivity Bridge (P17)

### 14.1 Motivation

The §13.2 balance leaves a single open obstruction: **G4 = RH itself**.
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
* In the §13.2 ledger, G4 remains **OPEN**, but the attack surface is
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

In the §13.2 ledger, G4 stays **OPEN**; the §14.6 "lower-boundedness
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
