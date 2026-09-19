# TNFR Additive Dynamics — Controlled Additive-Character Number Theory (R6)

**Status**: finite cyclic convolution identity **DERIVED**, with numerical
residual checks; the selected phase statistic does not outperform the selected
power statistic on the recorded controls. No additional information beyond the
complete invertible Fourier transform is possible for a deterministic summary
of the same input. A useful TNFR-specific dynamics/measurement bridge remains
**OPEN / not established** (`NT-P06`). No proof
of Goldbach or any open problem; the target property is never used to define the
phase.
**Module**: [src/tnfr/mathematics/additive_resonance.py](../src/tnfr/mathematics/additive_resonance.py) ·
**Tests**: [tests/mathematics/test_additive_resonance.py](../tests/mathematics/test_additive_resonance.py) ·
**Benchmark**: [benchmarks/additive_resonance_controls.py](../benchmarks/additive_resonance_controls.py) ·
**Depends on**: C5 (claim manifest, circularity audit).

## 1. The object and the honest question

The implemented cyclic representation function is the self-convolution of the
finite prime indicator on `Z/NZ`,

$$r_2(m) = (1_{\mathbb P} * 1_{\mathbb P})(m)
= \#\{(a, b) : a + b \equiv m,\; a, b \text{ prime}\}.$$

Its additive-character (Fourier) transform factorises by the finite convolution
theorem: `\hat r_2(k) = \hat{1_{\mathbb P}}(k)^2`. Modular wraparound counts
representations of `m` modulo `N`; identifying ordinary integer Goldbach counts
requires a separate no-aliasing range or padding. R6 does **not** try to prove
Goldbach. It asks a falsifiable internal question:

> Does a **TNFR** reading of the additive phase `χ_k(n) = e^{2πi k n / N}` supply a
> useful summary for a declared discrimination task compared with a particular
> power-spectrum summary, under explicit control distributions?

The phase is derived **only** from the additive characters; the target property
(`r_2`, Goldbach) is never used to define it.

## 2. The exact reduction (DERIVED + MEASURED)

In exact arithmetic the cyclic function equals the inverse transform of the
squared spectrum:

$$r_2 = \operatorname{IFFT}\big(\hat{1_{\mathbb P}}^2\big).$$

Measured residual `~1e-14` for `N ≤ 1024` (`goldbach_fourier_residual`).
This measured residual is not a proved universal FFT error bound. Both `r_2`
and the power spectrum are quadratic, not linear, in the input. More generally,
the complete DFT is invertible: every deterministic input statistic, linear or
nonlinear, is a function of that spectrum. A nonlinear statistic can improve a
particular compressed read-out or task cost; it cannot add information beyond
the full DFT. Such utility requires a specified comparison and does not derive
new nodal dynamics.

## 3. The controlled negative (MEASURED)

The candidate is the mean squared second difference of an **unwrapped FFT
phase**, implemented by `np.unwrap`, `np.angle` and `np.diff(...,2)`,

$$\mathcal{E}_{K_\phi}(1) = \big\langle (\Delta^2 \arg \hat 1(k))^2 \big\rangle_k,$$

on the interior frequency indices. It is not the canonical graph circular-mean
`K_phi`: no graph-neighbor curvature owner is called, and the FFT's zero
coefficients do not supply a physically defined phase. The naming in the legacy
API does not establish that bridge. It is compared against **power concentration**
`max_{k≠0}|\hat 1|^2 / \text{mean}`. For each observable a z-score measures how
far the primes sit from a control distribution. Shuffle and matched-random
preserve the finite prime count. The Cramér branch draws independent Bernoulli
values using `1/log(m)` and does not preserve that count; it is a different
baseline. The historical table is retained with those actual control semantics:

| control | classical `z` | TNFR `z` | TNFR exceeds |
|---------|---------------|----------|--------------|
| matched | 167.8 | −0.58 | **No** |
| Cramér | 148.5 | −0.21 | **No** |
| shuffle | 190.6 | −0.51 | **No** |

The classical Fourier observable strongly detects the arithmetic structure of the
primes in these samples (`\|z\| ≫ 1`); it does not compute or prove a singular-series
asymptotic. The candidate phase statistic shows no large z-score in the reported
samples (`\|z\| < 1`), and never
exceeds the classical observable (`excess_over_fourier(...).tnfr_exceeds ==
False`). Prime constellations (twin-prime starts) also stand out — but again under
the power statistic. These are comparisons of selected summaries, not proofs of
the absence of all possible useful phase statistics.

**Invariance controls.** The observables respect the required conventions:
`power_concentration` is invariant under translation `n → n + c` and under
character conjugation (`n → −n`), and `cyclic_goldbach` is translation-equivariant
with the output shifted by `2c` for a shift `c` of the input. These properties
do not certify origin invariance of the unwrapped-phase second-difference
statistic, which has its own branch and endpoint conventions.

## 4. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `r_2 = IFFT(\hat 1^2)` (additive reading is Fourier) | convolution theorem | **DERIVED** + MEASURED (`~1e-14`) |
| power concentration distinguishes the supplied samples | selected control distributions | **MEASURED** (`\|z\| ≫ 1`) |
| TNFR phase-curvature exceeds Fourier under controls | — | **NEGATIVE** (`\|z\| < 1`, never exceeds) |
| extra information beyond the complete DFT of the same input | DFT invertibility | **EXCLUDED** for deterministic summaries |
| useful canonical nodal contribution for this task | no dynamics/measurement bridge | **OPEN** (`NT-P06`) |

R6 retains a finite convolution baseline and a comparison of two compressed
statistics. The selected phase statistic showed no advantage in the recorded
experiment. A canonical phase-curvature interpretation, a useful nodal evolution
and any claim about ordinary Goldbach representations require separate work;
this note does not open another active research queue.
