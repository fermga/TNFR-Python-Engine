# TNFR Additive Dynamics — Controlled Additive-Character Number Theory (R6)

**Status**: additive→Fourier reduction **DERIVED** + **MEASURED** (exact identity
to machine precision); the TNFR-phase excess over Fourier is **NEGATIVE** for the
tested observable, so the claim `NT-P06` is **OPEN / not established**. No proof
of Goldbach or any open problem; the target property is never used to define the
phase.
**Module**: [src/tnfr/mathematics/additive_resonance.py](../src/tnfr/mathematics/additive_resonance.py) ·
**Tests**: [tests/mathematics/test_additive_resonance.py](../tests/mathematics/test_additive_resonance.py) ·
**Benchmark**: [benchmarks/additive_resonance_controls.py](../benchmarks/additive_resonance_controls.py) ·
**Depends on**: C5 (claim manifest, circularity audit).

## 1. The object and the honest question

The Goldbach representation function is the self-convolution of the prime
indicator,

$$r_2(m) = (1_{\mathbb P} * 1_{\mathbb P})(m)
= \#\{(a, b) : a + b \equiv m,\; a, b \text{ prime}\}.$$

Its additive-character (Fourier) transform on `ℤ/Nℤ` factorises — the circle
method — so `\hat r_2(k) = \hat{1_{\mathbb P}}(k)^2`. R6 does **not** try to prove
Goldbach. It asks a falsifiable internal question:

> Does a **TNFR** reading of the additive phase `χ_k(n) = e^{2πi k n / N}` supply a
> structural observable that is **more** than a renaming of the Fourier transform
> or the Hardy–Littlewood singular series, once density-matched controls are in
> place?

The phase is derived **only** from the additive characters; the target property
(`r_2`, Goldbach) is never used to define it.

## 2. The exact reduction (DERIVED + MEASURED)

The cyclic Goldbach function equals the inverse transform of the squared
spectrum, exactly (to FFT backward error):

$$r_2 = \operatorname{IFFT}\big(\hat{1_{\mathbb P}}^2\big),\qquad
\max\big|r_2 - \operatorname{IFFT}(\hat 1^2)\big| \le \varepsilon\, N\, \lVert 1\rVert_2^2.$$

Measured residual `~1e-14` for `N ≤ 1024` (`goldbach_fourier_residual`).
Consequently **every linear additive-character observable** of the prime
indicator — `r_2`, the power spectrum, any character average — is a function of
the Fourier spectrum `\{\hat 1(k)\}`. A genuine TNFR contribution must therefore
be a **non-linear** phase observable *and* must be shown to exceed Fourier under
controls.

## 3. The controlled negative (MEASURED)

The candidate non-linear observable is the **phase-curvature energy**, the TNFR
`K_φ` read of the spectrum phase,

$$\mathcal{E}_{K_\phi}(1) = \big\langle (\Delta^2 \arg \hat 1(k))^2 \big\rangle_k,$$

compared against the classical **power concentration**
`max_{k≠0}|\hat 1|^2 / \text{mean}`. For each observable a z-score measures how
far the primes sit from a distribution of density-matched controls (Cramér,
shuffle, matched-random, all preserving the prime count / density):

| control | classical `z` | TNFR `z` | TNFR exceeds |
|---------|---------------|----------|--------------|
| matched | 167.8 | −0.58 | **No** |
| Cramér | 148.5 | −0.21 | **No** |
| shuffle | 190.6 | −0.51 | **No** |

The classical Fourier observable strongly detects the arithmetic structure of the
primes (`|z| ≫ 1`, the singular series), as it must. The TNFR phase-curvature
observable detects **nothing beyond the density controls** (`|z| < 1`), and never
exceeds the classical observable (`excess_over_fourier(...).tnfr_exceeds ==
False`). Prime constellations (twin-prime starts) also stand out — but again under
the **classical** observable, i.e. via the singular series, not via any TNFR
excess.

**Invariance controls.** The observables respect the required conventions:
`power_concentration` is invariant under translation `n → n + c` and under
character conjugation (`n → −n`), and `cyclic_goldbach` is translation-equivariant
— so no result rides on an arbitrary phase or index origin.

## 4. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `r_2 = IFFT(\hat 1^2)` (additive reading is Fourier) | convolution theorem | **DERIVED** + MEASURED (`~1e-14`) |
| classical Fourier detects prime structure | singular series | **MEASURED** (`|z| ≫ 1`) |
| TNFR phase-curvature exceeds Fourier under controls | — | **NEGATIVE** (`|z| < 1`, never exceeds) |
| TNFR additive phase adds beyond Fourier | — | **OPEN** (`NT-P06`; tested observable is negative) |

**Bottom line.** R6 establishes that the additive-character reading of Goldbach is
exactly the Fourier / circle-method description, and provides a controlled test
for any TNFR excess. The natural non-linear phase observable tested here shows no
excess over the classical power spectrum under density-matched controls. The R6
claim (`NT-P06`) is therefore **open with negative evidence**: no TNFR-specific
additive observable beyond Fourier has been demonstrated, and none is claimed.
