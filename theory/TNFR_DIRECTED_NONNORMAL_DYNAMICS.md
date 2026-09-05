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

## 4b. Metric layer and U2 semantics (N03)

The transient gain is **metric-dependent**, so no physical U2 conclusion may be
drawn without an explicit norm. The N03 layer
([directed_diffusion.py](../src/tnfr/physics/directed_diffusion.py)) exposes two
norms and both integral readings, and **does not** decide U2.

**The Euclidean transient is a metric artefact (DERIVED + MEASURED).** Let
`P = D_out⁻¹ W` (row-stochastic) with stationary distribution `π` (`πᵀP = πᵀ`,
`π > 0` for a strongly connected digraph). In the weighted inner product
`⟨f, g⟩_π = Σ π_i f_i g_i`, Jensen gives

$$\lVert P f\rVert_{2,\pi}^2 = \sum_i \pi_i\Big(\sum_j P_{ij} f_j\Big)^2
\le \sum_i \pi_i \sum_j P_{ij} f_j^2
= \sum_j f_j^2 \underbrace{\sum_i \pi_i P_{ij}}_{=\,\pi_j}
= \lVert f\rVert_{2,\pi}^2,$$

so `‖P‖_{2,π} ≤ 1`. Since `e^{−tL} = e^{−t}\sum_k \tfrac{t^k}{k!} P^k` is a convex
combination of `π`-contractions, `‖e^{−tL}‖_{2,π} ≤ 1`: **the diffusion semigroup
is a contraction in `L²(π)`**. Measured (`stationary_transient_gain`,
`is_stationary_contraction`): for every non-normal strongly connected digraph the
Euclidean gain exceeds 1 (e.g. `1.03`, `1.02`) while the **stationary gain is
exactly `1.0`**. The Euclidean transient amplification is a coordinate effect of
the non-normal basis, not a `π`-weighted energy growth.

**The U2 integral has two readings (report §16.2).** `u2_integral_readings`
distinguishes the **signed** displacement `‖x(∞) − x(0)‖ = ‖∫₀^∞ ẋ\,dt‖`
(reorganizations may cancel) from the **total structural variation**
`∫₀^∞ ‖ẋ(s)‖\,ds` (no cancellation); always `net ≤ total`. Which one U2
canonically means — "existence of the EPI limit" (net) vs "total pressure
absorbed" (total) — is **not decided here**.

**The metric decision stays OPEN.** Three outcomes remain possible (report
§16.3): (i) the transient is real in the canonical TNFR per-node energy (the
unweighted / Euclidean norm) and must enter U2/U6; (ii) it is a coordinate effect
and the canonical metric is `L²(π)`; (iii) both norms are useful and are exposed
with distinct semantics. **U2 in [AGENTS.md](../AGENTS.md) §6 is not modified**
until the gate (defined quantity, chosen metric, EPI-channel proof, normal +
non-normal certificate, temporal U6 bound) is met.

## 4c. Structural-time theorem — νf as a clock (N04)

For a scalar common frequency `ν_f(t) ≥ 0`, the linear EPI transport
`ẋ = −ν_f(t) L x` on a fixed graph has the **exact** solution

$$x(t) = e^{-s(t)\,L}\,x_0,\qquad s(t) = \int_0^t \nu_f(\tau)\,d\tau,$$

because `L` is constant and all `ν_f(τ)L` commute. So `ν_f` is a **clock change**
(a mobility, not a mass): it rescales the speed along a **fixed** state-space
trajectory. Verified (`clock_change_residual`): an RK4 integrator of the
time-varying ODE converges to `e^{−s(t)L}x_0` (`~3e-7`, shrinking with
refinement).

**Total reorganization is clock-invariant.** With the substitution `s = s(t)`,
`ds = ν_f dt`,

$$\int_0^\infty \nu_f(t)\,\lVert L\,e^{-s(t)L} Q x_0\rVert\,dt
= \int_0^\infty \lVert L\,e^{-sL} Q x_0\rVert\,ds,$$

so the accumulated structural variation depends on the **trajectory**, not the
speed (`reorganization_time_invariance_residual`, `~3e-6`). Here `Q = I − 1πᵀ` is
the projection out of the consensus mode; since `L·1 = 0` and `πᵀL = 0`,
`LQ = QL = L` and `Q` commutes with the semigroup.

**Finite reorganization (the linear-EPI-channel U2 form).** On the non-consensus
subspace `‖e^{−sL}Q‖ ≤ M e^{−ωs}` with `M = ` `sustained_gain` (`= 1` normal,
`> 1` non-normal — the transient cost) and `ω = ` `nonconsensus_abscissa` (the
spectral gap). Hence

$$J = \int_0^\infty \lVert L\,e^{-sL} Q x_0\rVert\,ds \;\le\; \frac{M\,\lVert LQ\rVert\,\lVert x_0\rVert}{\omega},$$

`certify_structural_time` verifies `J ≤ bound` (measured: `J = 1.70 ≤ 2.20`). This
is the exact, EPI-channel, scalar-frequency form of U2's `∫ ν_f ΔNFR dt < ∞`: the
integral converges, and the non-normal cost is the finite factor `M`. It is
**restricted** to the linear EPI channel with a **scalar** `ν_f` on a fixed graph;
heterogeneous nodal `ν_f` (a diagonal `D_{ν_f}(t)`) is **not** a clock change and
is out of scope (N13, `NT-P09` heterogeneous).

## 5. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| directed circulants are normal | commutator `= 0` | **MEASURED** (exact) |
| circulant diffusion has unit transient gain | normal contraction | **DERIVED** + MEASURED |
| non-normal stable digraphs amplify transiently (Euclidean) | Kreiss theorem | **DERIVED** + MEASURED (`gain > 1`) |
| `pseudospectral_bound ≤ transient_gain` | Kreiss matrix theorem | **DERIVED** + MEASURED |
| Schur residual `= 0` | unitary factorisation | **MEASURED** (SciPy-gated) |
| diffusion contracts in `L²(π)` (stationary gain `≤ 1`) | Jensen (N03) | **DERIVED** + MEASURED |
| the Euclidean transient is metric-dependent | N03 metric layer | **MEASURED** (eucl `> 1`, stat `= 1`) |
| `net ≤ total` U2 integral readings | triangle inequality | **DERIVED** + MEASURED |
| `x(t) = e^{−s(t)L}x₀`, `s = ∫ν_f` (scalar-`ν_f` clock change) | commuting flow (N04) | **DERIVED** + MEASURED (RK4 `~3e-7`) |
| total reorganization is clock-invariant | change of variables (N04) | **DERIVED** + MEASURED (`~3e-6`) |
| `J ≤ M‖LQ‖‖x₀‖/ω` (finite reorganization) | exponential decay on `Q` (N04) | **DERIVED** + MEASURED |
| which norm/integral is the canonical U2 | — | **OPEN** (`NT-P09b/c`; gate not met) |
| generalized U2 bound for non-normal transients | — | **OPEN / CONJECTURAL** (`NT-P09`) |

**Bottom line.** R9 separates the normal (directed circulant) regime, where the
existing spectral machinery is exact, from the general non-normal regime, where a
stable spectrum coexists with transient amplification. It certifies the latter
with Kreiss / Schur measures — never `eigh` — and keeps the U2 reading explicitly
asymptotic, leaving the transient's role in the convergence integral open.
