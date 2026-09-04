# TNFR Directed Non-Normal Dynamics (R9)

**Status**: normal/non-normal classification and transient amplification
**DERIVED** (Kreiss) + **MEASURED** (exact circulant normality; transient gain and
pseudospectral bound); the U2 generalization to non-normal transients is
**OPEN / CONJECTURAL** (`NT-P09`). SciPy is optional — the Schur certificate is
gated; the core is numpy-only.
**Modules**: [src/tnfr/physics/directed_diffusion.py](../src/tnfr/physics/directed_diffusion.py),
[src/tnfr/physics/spectral_projectors.py](../src/tnfr/physics/spectral_projectors.py) (extended) ·
**Tests**: [tests/physics/test_directed_diffusion.py](../tests/physics/test_directed_diffusion.py) ·
**Benchmark**: [benchmarks/directed_nonnormal_dynamics.py](../benchmarks/directed_nonnormal_dynamics.py) ·
**Depends on**: C1 (outgoing `L_rw`), C4 (normal-only projectors).

## 1. Why non-normality matters

The structural-diffusion operator `L_rw = I − D_out⁻¹ W` is symmetric (hence
**normal**) only for undirected or vertex-transitive graphs. A general **directed**
graph makes `L_rw` non-normal, and the naive spectral reading fails:

- `eigh` is invalid — the operator is not symmetric;
- a **stable spectrum** does not exclude **transient amplification**;
- right eigenvectors are not orthogonal, so `Q Qᴴ` is **not** the spectral
  projector (C4's `spectral_clusters` correctly raises `NonNormalOperatorError`
  and points here).

This module handles exactly the operators C4 refuses, with certificates that use
`eigvals` / Schur, never `eigh`.

## 2. Two regimes (MEASURED)

| digraph | normal | `α(−L)` | transient gain | Kreiss bound | amplifies |
|---------|--------|---------|----------------|--------------|-----------|
| circulant `C7{1,2}` (R2 residue) | **yes** | 0.000 | 1.000 | 1.000 | no |
| circulant `C8{1,3}` | **yes** | 0.000 | 1.000 | 1.000 | no |
| feed-forward chain + self-loops | **no** | −0.000 | 1.987 | 1.665 | **yes** |
| asymmetric directed graph | **no** | −0.000 | 1.032 | 1.016 | **yes** |

**Directed circulants are the benign case.** The R2 residue digraphs are
circulant, hence normal (`commutator = 0`), so their diffusion is a contraction:
transient gain `= 1`, no growth. The R2/C4 spectral machinery applies to them
unchanged.

**General digraphs amplify.** The feed-forward and asymmetric graphs have a
**stable** spectrum (`α(−L) ≤ 0`, asymptotically decaying) yet a transient gain
**> 1** — `‖e^{−tL}‖₂` grows before it decays. Non-normality, not instability, is
the source.

## 3. The certificates (DERIVED)

Added to [spectral_projectors.py](../src/tnfr/physics/spectral_projectors.py):

- `spectral_abscissa(A) = max_i Re λ_i(A)` — asymptotic growth rate
  (`α(−L) ≤ 0` ⇒ stable), via `eigvals`.
- `matrix_exponential(A)` — numpy-only scaling-and-squaring, so the transient
  measures need no SciPy.
- `transient_gain(A) = max_t ‖e^{tA}‖₂` — peak amplification; `≤ 1` for a normal
  stable `A`, possibly `> 1` for a non-normal one.
- `pseudospectral_bound(A) = sup_{Re z>0} Re(z)·‖(zI − A)⁻¹‖₂` — the Kreiss lower
  bound. The **Kreiss matrix theorem** gives `K(A) ≤ sup_t ‖e^{tA}‖`, so a value
  `> 1` **proves** transient growth from the resolvent alone (verified:
  `Kreiss ≤ transient gain` in every non-normal case).
- `schur_residual(A) = ‖A − Q T Qᴴ‖₂` — the correct unitary factorisation for
  non-normal operators; **SciPy-gated** (`scipy.linalg.schur`), raising a clear
  error when SciPy is absent so the numpy core never mishandles it silently.

`certify_directed_dynamics` bundles these into a `DirectedDynamicsCertificate`
for the generator `−L_rw`.

## 4. U2 scope — asymptotic only (OPEN)

The convergence reading `r_c = ν_f λ₂` describes **asymptotic** relaxation. For a
non-normal generator the positive transient gain means `C(t)` can **dip before it
relaxes**, a temporal risk for U6 monitoring. The U2 integral convergence
`∫ ν_f ΔNFR dt < ∞` still holds asymptotically (the spectrum is stable), but:

> No generalized U2 bound is claimed for non-normal operators. Deriving how the
> transient enters the integral convergence — and its interaction with U6
> potential confinement — is left **open** (`NT-P09`).

## 5. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| directed circulants are normal | commutator `= 0` | **MEASURED** (exact) |
| circulant diffusion has unit transient gain | normal contraction | **DERIVED** + MEASURED |
| non-normal stable digraphs amplify transiently | Kreiss theorem | **DERIVED** + MEASURED (`gain > 1`) |
| `pseudospectral_bound ≤ transient_gain` | Kreiss matrix theorem | **DERIVED** + MEASURED |
| Schur residual `= 0` | unitary factorisation | **MEASURED** (SciPy-gated) |
| generalized U2 bound for non-normal transients | — | **OPEN / CONJECTURAL** (`NT-P09`) |

**Bottom line.** R9 separates the normal (directed circulant) regime, where the
existing spectral machinery is exact, from the general non-normal regime, where a
stable spectrum coexists with transient amplification. It certifies the latter
with Kreiss / Schur measures — never `eigh` — and keeps the U2 reading explicitly
asymptotic, leaving the transient's role in the convergence integral open.
