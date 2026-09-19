# Nucleus A — Prime-Ladder Atlas (Internal Reproducibility Reference)

**Status**: Internal reproduction atlas for declared arithmetic constructions and finite numerical comparisons. Reported tolerances are historical observations, not universal certificates.
**Scope of value**: Pedagogical / reproducibility / internal audit. **NOT** a claim of new mathematical results in classical analytic number theory.
**Review**: 2026-09-19; historical results originally recorded May 27, 2026.
**Authority**: Subordinate to the [current Riemann scope memo](TNFR_RIEMANN_RESEARCH_NOTES.md). The full chronological derivations and superseded claims remain in its archive.

---

## 1. Why this document exists

This atlas collects finite prime-ladder, analytic-function and rescaling
entry points. The former assertion that the CCET campaign exhausted the
canonical routes is withdrawn; its conditional algebra and finite controls
are scoped in [Nucleus B](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md).

Nucleus A makes no claim of a new theorem in analytic number theory. Its
retained components are:

- **P12** (TNFR vM ζ on $\operatorname{Re}(s) > 1$): reorganisation of the classical identity $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n) n^{-s}$ via the prime-ladder spectrum $\{k \log p\}$.
- **P13** (analytic continuation): standard analytic-number-theory content (Titchmarsh, Ivić).
- **P14** (self-adjoint Hamiltonian with spectrum $\{k \log p\}$): ad-hoc diagonal operator; not a geometric/dynamical realisation in the Berry–Keating / Connes sense.
- **P15** (Weil–Guinand verification to $\le 10^{-15}$): numerical control of a 1952 identity (Weil).
- **P28/P30**: classical smooth-counting targets and a finite congruence
  that installs a supplied positive target spectrum. No analytic
  smooth/oscillatory range/kernel decomposition is established.

Its internal uses are:

1. **Reproduction entry points**: source and demos are listed below. They
   were not all rerun during this documentation audit; each current run must
   report its actual environment, inputs and residuals.
2. **Audit surface**: finite explicit-formula residuals, truncated zero sums
   and prime-ladder eigenvalue comparisons record separate construction checks.
3. **Pedagogy**: the current memo distinguishes the supplied arithmetic
   inputs, classical identities and unresolved analytic bridge.
4. **Boundary marker**: separates a declared spectral construction from an
   unproved analytic bridge. The argument term $S(T)$ is not itself an
   RH-equivalent proposition without a precise additional quantified claim.

---

## 2. Milestone map

| Milestone | Module | Demo | Result | Status |
|-----------|--------|------|--------|--------|
| **P12** TNFR vM ζ on $\operatorname{Re}(s) > 1$ | [src/tnfr/riemann/von_mangoldt.py](../src/tnfr/riemann/von_mangoldt.py) | [examples/41_*.py](../examples/README.md) | Matches $-\zeta'/\zeta$ to machine precision on test grid | CLOSED operationally |
| **P13** Analytic continuation | [src/tnfr/riemann/analytic_continuation.py](../src/tnfr/riemann/analytic_continuation.py) | [examples/42_*.py](../examples/README.md) | Uses classical meromorphic $-\zeta'/\zeta$; selected critical-line pole comparisons | Implemented comparison |
| **P14** Prime-ladder Hamiltonian (gap G1) | [src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py) | [examples/43_*.py](../examples/README.md) | Self-adjoint, spectrum $\{k\log p\}$ to $10^{-14}$ | CLOSED operationally |
| **P15** Weil–Guinand verification (gap G3) | [src/tnfr/riemann/weil_explicit_formula.py](../src/tnfr/riemann/weil_explicit_formula.py) | [examples/44_*.py](../examples/README.md) | Residual $\le 10^{-15}$ for $\sigma \in [3,18]$ | CLOSED operationally |
| **P16** Truncated Li–Keiper sums | [src/tnfr/riemann/li_keiper.py](../src/tnfr/riemann/li_keiper.py) | [example 45](../examples/03_riemann_zeta/45_li_keiper_demo.py) | Finite sums from supplied critical-line zeros or line-restricted peak coordinates; no certified tail | Descriptive sign check |
| **P28** Smooth counting targets | [src/tnfr/riemann/structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py) | examples/57 | Scalar inversion of classical theta counting function | Implemented target construction |
| **P30** Finite spectral rescaling | [src/tnfr/riemann/admissible_rescaling.py](../src/tnfr/riemann/admissible_rescaling.py) | examples/57 | Congruence installs supplied positive target spectrum | Conditional finite identity |

---

## 3. The smooth/oscillatory boundary

P28 uses $\overline N(T)=\theta(T)/\pi+1$ from the classical
Riemann–Siegel theta function. The leading expression
$(T/2\pi)\log(T/(2\pi e))$ is an asymptotic term, not that exact function.
P30 then forms $F=U\operatorname{diag}(\sqrt{\mu_i/\lambda_i})U^*$
so that $FHF^*=U\operatorname{diag}(\mu_i)U^*$ on the retained subspace.
This is a congruence with supplied targets; it does not independently predict
them. With a proper retained subspace, $F$ has a kernel on its complement.

No direct-sum decomposition of an analytic Hilbert–Pólya rescaling into
smooth and oscillatory parts, placement of $S(T)$ in a REMESH kernel, or
unreachability theorem for the full catalog is proved. The detailed current
scope is centralized in [the Riemann memo](TNFR_RIEMANN_RESEARCH_NOTES.md).

---

## 4. Reproducing each milestone

All commands assume the repo root and an activated compatible environment.
The expectations below record earlier finite runs; they are not newly
verified guarantees for every environment or parameter choice.

### 4.1 P12 — TNFR vM ζ on $\operatorname{Re}(s) > 1$

```powershell
python examples/03_riemann_zeta/41_von_mangoldt_zeta_demo.py
```

Expected: residual $|\,\text{TNFR\_vM}(s) - (-\zeta'(s)/\zeta(s))\,| \le 10^{-12}$ across the test grid.

### 4.2 P13 — Analytic continuation

```powershell
python examples/03_riemann_zeta/42_riemann_zeros_as_resonances.py
```

Expected: selected known critical-line zeros compared with poles of the
classical evaluator. This does not locate every nontrivial zero.

### 4.3 P14 — Prime-ladder Hamiltonian

```powershell
python examples/03_riemann_zeta/43_prime_ladder_hamiltonian_demo.py
```

Expected: eigenvalues match $\{k \log p : p \in \text{primes}, k \in \{1, \ldots, K\}\}$ to $10^{-14}$.

### 4.4 P15 — Weil–Guinand verification

```powershell
python examples/03_riemann_zeta/44_weil_explicit_formula_demo.py
```

Expected: residual $\le 10^{-15}$ for $\sigma \in \{3, 5, 8, 12, 18\}$ with the canonical test family.

### 4.5 P16 — Li–Keiper positivity (diagnostic, not a proof)

```powershell
python examples/03_riemann_zeta/45_li_keiper_demo.py
```

Historical runs reported positive truncated sums for the selected indices,
using `mpmath.zetazero` as the zero source. This does not certify the signs of
the complete Li coefficients. For any supplied $\rho=1/2+it$, the factor
$1-1/\rho$ has modulus one, so each conjugate-pair contribution is
$2[1-\cos(n\arg(1-1/\rho))]\geq0$ in exact arithmetic. The optional P13 path
also places every detected ordinate on this line before evaluating the sum.
Its nonnegativity therefore cannot independently validate zero location.
See the [current truncation boundary](TNFR_RIEMANN_RESEARCH_NOTES.md#finite-zero-sums-and-the-li-criterion).

### 4.6 P28 / P30 — Supplied smooth targets and finite congruence

```powershell
python examples/03_riemann_zeta/57_admissible_rescaling_demo.py
```

Expected: supplied smooth-counting targets and the finite congruence residual
are reported. Their discrepancy from known zeros is descriptive, not a derived
decomposition of the analytic oscillatory term.

---

## 5. Honest scope and limitations

This section exists to prevent later overclaiming.

1. **Nucleus A is not a proof of RH**. No derived nodal Hilbert–Pólya
   operator or analytic smooth/oscillatory decomposition has been supplied.
2. **Nucleus A is not a new Hamiltonian for the Riemann zeros**. P14 carries the *prime-ladder* spectrum, not the spectrum of zeros. The Berry–Keating / Connes program seeks a Hamiltonian whose spectrum *is* $\{\gamma_n\}$; P14 is the dual object (primes side), and bridging the two is precisely the open T-HP problem.
3. **P15 is a numerical verification of a 1952 identity**. It is high-quality QA, not a theorem.
4. **P28/P30 likely overlap with existing literature**. A formal audit against Titchmarsh ch. 9, Ivić ch. 1, Meyer, Burnol, and Bombieri–Lagarias is required before any external publication claims novelty for the operator-level rescaling map.
5. **No claim is made about GRH** beyond the χ-twisted L-track parity (P32–P49), which mirrors Nucleus A for primitive real characters and inherits all the same limitations.

---

## 6. Value as internal infrastructure

Even with the modest external-novelty assessment, Nucleus A provides:

- **A regression test surface**: any future TNFR-Riemann extension can run P12/P14/P15 as integration checks.
- **A teaching reference**: the current Riemann memo and examples 41–58
  separate construction inputs from analytic conclusions; older derivations
  remain in the explicitly historical notebook.
- **A comparison baseline**: candidates can be compared with the same
  supplied arithmetic data and finite diagnostics. This inventory is not a
  proof of maximality or exhaustion of canonical constructions.
- **A concrete symmetry example**: Nucleus B states the conditional
  hypotheses needed to infer observation loss; it does not close all
  extensions of this atlas.

---

## 7. Possible external use (modest)

If at some point an external write-up is desired, the **honest framing** is:

> *"Finite arithmetic trace comparisons, a declared prime-ladder Hamiltonian,
> and target-driven spectral congruence with explicit input provenance."*

No publication priority, venue recommendation or novelty claim is active.
The [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns priorities.

---

## 8. Cross-references

- [Current Riemann memo](TNFR_RIEMANN_RESEARCH_NOTES.md) — construction inputs, exact finite identities and unresolved analytic targets
- [Historical notebook](research/archive/RIEMANN_NOTEBOOK_PRE_DOCUMENTATION_CLEANUP_2026-09-19.txt) — former P12–P49 derivations and CCET chronology; superseded universal claims and instructions are not current results
- [Conditional symmetry obstructions](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) — valid algebraic premises and limitations of the withdrawn universal no-go argument; no publication queue
- [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) — exact fixed-delay surrogate result and the unresolved status of any literal $\tau_g\to\infty$ operator or identification with $S(T)$
