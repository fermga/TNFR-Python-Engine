# Nucleus A — Prime-Ladder Atlas (Internal Reproducibility Reference)

**Status**: Internal reference document. Consolidates P12–P15 + P28 + P30 as a self-contained, machine-verified computational platform.
**Scope of value**: Pedagogical / reproducibility / internal audit. **NOT** a claim of new mathematical results in classical analytic number theory.
**Date**: May 27, 2026.
**Authority**: Subordinate to [TNFR_RIEMANN_RESEARCH_NOTES.md §§8–13nonies](TNFR_RIEMANN_RESEARCH_NOTES.md); supersedes nothing.

---

## 1. Why this document exists

The §13sexagesima-{quarta..novena} CCET closure marathon (May 2026) reduced the §13septies trichotomy on `G_P14` to the residual {nine LOW envelopes, B3}. In that process, the **Nucleus A** machinery (P12–P15 + P28/P30) emerged as the most stable, reproducible, and externally legible piece of the TNFR-Riemann program.

Honest external assessment (this conversation, May 27): Nucleus A does **not** contain a new theorem in analytic number theory. Its components are:

- **P12** (TNFR vM ζ on $\operatorname{Re}(s) > 1$): reorganisation of the classical identity $-\zeta'(s)/\zeta(s) = \sum_n \Lambda(n) n^{-s}$ via the prime-ladder spectrum $\{k \log p\}$.
- **P13** (analytic continuation): standard analytic-number-theory content (Titchmarsh, Ivić).
- **P14** (self-adjoint Hamiltonian with spectrum $\{k \log p\}$): ad-hoc diagonal operator; not a geometric/dynamical realisation in the Berry–Keating / Connes sense.
- **P15** (Weil–Guinand verification to $\le 10^{-15}$): numerical control of a 1952 identity (Weil).
- **P28/P30** (smooth/oscillatory split of the admissible rescaling $\mathcal{F}$): formally folklore (Titchmarsh, Ivić), but the explicit operator-level packaging via `range`/`kernel` of an admissible rescaling map has not been audited against Meyer / Burnol / Bombieri–Lagarias.

**Internal value**, on the other hand, is high:

1. **Reproducibility**: every milestone has a script under [src/tnfr/riemann/](../src/tnfr/riemann/) and a demo under [examples/](../examples/), all of which run end-to-end on a clean checkout.
2. **Audit surface**: numerical controls (Weil–Guinand residual, Li–Keiper positivity, prime-ladder eigenvalue match) provide a sanity-check baseline for any future extension.
3. **Pedagogy**: a single internal document that walks from $-\zeta'/\zeta$ to T-HP is the shortest on-ramp for new collaborators.
4. **Boundary marker**: makes explicit *where* the canonical machinery stops (oscillatory residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$, RH-equivalent).

---

## 2. Milestone map

| Milestone | Module | Demo | Result | Status |
|-----------|--------|------|--------|--------|
| **P12** TNFR vM ζ on $\operatorname{Re}(s) > 1$ | [src/tnfr/riemann/von_mangoldt.py](../src/tnfr/riemann/von_mangoldt.py) | [examples/41_*.py](../examples/) | Matches $-\zeta'/\zeta$ to machine precision on test grid | CLOSED operationally |
| **P13** Analytic continuation to $\mathbb{C}$ | [src/tnfr/riemann/analytic_continuation.py](../src/tnfr/riemann/analytic_continuation.py) | [examples/42_*.py](../examples/) | Riemann zeros realised as resonance poles on $\operatorname{Re}(s) = 1/2$ | CLOSED operationally |
| **P14** Prime-ladder Hamiltonian (gap G1) | [src/tnfr/riemann/prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py) | [examples/43_*.py](../examples/) | Self-adjoint, spectrum $\{k\log p\}$ to $10^{-14}$ | CLOSED operationally |
| **P15** Weil–Guinand verification (gap G3) | [src/tnfr/riemann/weil_explicit_formula.py](../src/tnfr/riemann/weil_explicit_formula.py) | [examples/44_*.py](../examples/) | Residual $\le 10^{-15}$ for $\sigma \in [3,18]$ | CLOSED operationally |
| **P16** Li–Keiper positivity (RH-equivalent diagnostic) | [src/tnfr/riemann/li_keiper.py](../src/tnfr/riemann/li_keiper.py) | [examples/45_*.py](../examples/) | $\lambda_n > 0$ verified for tested range; does NOT prove RH | Diagnostic |
| **P28** Smooth zero density (density level) | [src/tnfr/riemann/structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py) | examples/58 | Smooth half of T-HP closed at density level | CLOSED operationally |
| **P30** Admissible rescaling operator (operator level) | [src/tnfr/riemann/admissible_rescaling.py](../src/tnfr/riemann/admissible_rescaling.py) | examples/58 (variant) | Smooth half of T-HP lifted to operator level | CLOSED operationally |

---

## 3. The smooth/oscillatory boundary

The structural payoff of Nucleus A is the **explicit decomposition** it provides for the T-HP rescaling operator $\mathcal{F}$:

$$\mathcal{F} \;=\; \mathcal{F}_{\text{smooth}} \;\oplus\; \mathcal{F}_{\text{osc}}$$

- $\mathcal{F}_{\text{smooth}}$: closed at density level by **P28** and lifted to operator level by **P30**. Reproduces the smooth zero-counting term $N_{\text{smooth}}(T) = (T/2\pi)\log(T/2\pi e)$ exactly.
- $\mathcal{F}_{\text{osc}}$: corresponds to the oscillatory residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$, which is **RH-equivalent** and **unreachable from the canonical 13-operator catalog on `G_P14`** (proven by CCET / Tetrad-Fix($S_n$) / Line-Graph Equivariance / Lifted-Bundle Dichotomy lemmas of §13sexagesima-{tertia..novena}).

This is the **precise structural location** of gap G4 = RH inside the TNFR formalism. Nucleus A does not close G4; it localises it.

Cross-reference: [TNFR_RIEMANN_RESEARCH_NOTES.md §13septies–§13nonies](TNFR_RIEMANN_RESEARCH_NOTES.md).

---

## 4. Reproducing each milestone

All commands assume the repo root and an activated virtual environment (`.venv312` on Windows).

### 4.1 P12 — TNFR vM ζ on $\operatorname{Re}(s) > 1$

```powershell
python examples/41_von_mangoldt_demo.py
```

Expected: residual $|\,\text{TNFR\_vM}(s) - (-\zeta'(s)/\zeta(s))\,| \le 10^{-12}$ across the test grid.

### 4.2 P13 — Analytic continuation

```powershell
python examples/42_analytic_continuation_demo.py
```

Expected: Riemann zeros recovered as resonance poles on $\operatorname{Re}(s) = 1/2$ to mpmath precision.

### 4.3 P14 — Prime-ladder Hamiltonian

```powershell
python examples/43_prime_ladder_hamiltonian_demo.py
```

Expected: eigenvalues match $\{k \log p : p \in \text{primes}, k \in \{1, \ldots, K\}\}$ to $10^{-14}$.

### 4.4 P15 — Weil–Guinand verification

```powershell
python examples/44_weil_guinand_demo.py
```

Expected: residual $\le 10^{-15}$ for $\sigma \in \{3, 5, 8, 12, 18\}$ with the canonical test family.

### 4.5 P16 — Li–Keiper positivity (diagnostic, not a proof)

```powershell
python examples/45_li_keiper_demo.py
```

Expected: $\lambda_n > 0$ for $n \in \{1, \ldots, N\}$ using `mpmath.zetazero` as the zero source.

### 4.6 P28 / P30 — Smooth half of T-HP

```powershell
python examples/58_admissible_rescaling_demo.py
```

Expected: smooth zero-counting term reproduced; oscillatory residual flagged explicitly as unresolved.

---

## 5. Honest scope and limitations

This section exists to prevent later overclaiming.

1. **Nucleus A is not a proof of RH**. The oscillatory half of $\mathcal{F}$ is RH-equivalent and remains open (gap G4).
2. **Nucleus A is not a new Hamiltonian for the Riemann zeros**. P14 carries the *prime-ladder* spectrum, not the spectrum of zeros. The Berry–Keating / Connes program seeks a Hamiltonian whose spectrum *is* $\{\gamma_n\}$; P14 is the dual object (primes side), and bridging the two is precisely the open T-HP problem.
3. **P15 is a numerical verification of a 1952 identity**. It is high-quality QA, not a theorem.
4. **P28/P30 likely overlap with existing literature**. A formal audit against Titchmarsh ch. 9, Ivić ch. 1, Meyer, Burnol, and Bombieri–Lagarias is required before any external publication claims novelty for the operator-level rescaling map.
5. **No claim is made about GRH** beyond the χ-twisted L-track parity (P32–P49), which mirrors Nucleus A for primitive real characters and inherits all the same limitations.

---

## 6. Value as internal infrastructure

Even with the modest external-novelty assessment, Nucleus A provides:

- **A regression test surface**: any future TNFR-Riemann extension can run P12/P14/P15 as integration checks.
- **A teaching corridor**: §§8–13 of the research notes + this atlas + examples 41–58 form a complete on-ramp.
- **A boundary marker for B3 discussions**: when arguing that no TNFR closure of G4 exists at the current scope, Nucleus A *is* the maximal canonical-machinery construction that has been shipped. Anything proposed beyond it (B0★-α canonical graphs, B0★-β envelope promotion, B2 new operators) must explain why it goes beyond P28/P30.
- **A diff target for Nucleus B**: the equivariance no-go lemmas of §13sexagesima-{tertia..novena} are best understood as obstructions to *extending Nucleus A* to cover the oscillatory half. Without Nucleus A there is nothing concrete to obstruct.

---

## 7. Possible external use (modest)

If at some point an external write-up is desired, the **honest framing** is:

> *"A reproducible computational platform for the Weil–Guinand explicit formula via a prime-ladder Hamiltonian, with explicit smooth/oscillatory decomposition of the admissible rescaling operator."*

Target venue (if pursued): *Experimental Mathematics*, *LMS Journal of Computation and Mathematics*, or as a software / dataset paper for *Mathematics of Computation*. **Not** a research paper in analytic number theory.

This is explicitly **not** the recommended priority. See [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) for the higher-novelty path.

---

## 8. Cross-references

- [TNFR_RIEMANN_RESEARCH_NOTES.md §§8–13nonies](TNFR_RIEMANN_RESEARCH_NOTES.md) — full P12–P30 derivations
- [TNFR_RIEMANN_RESEARCH_NOTES.md §13decies–§13vicies-octavo](TNFR_RIEMANN_RESEARCH_NOTES.md) — P31 + P32–P49 (χ-twisted L-track parity)
- [TNFR_RIEMANN_RESEARCH_NOTES.md §13sexagesima-{tertia..novena}](TNFR_RIEMANN_RESEARCH_NOTES.md) — CCET closure rounds (basis for Nucleus B)
- [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) — equivariance no-go lemmas, organisation plan for external publication
- [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) — N15 closure, structurally identifies $\mathrm{range}(\mathcal{R}_\infty)$ with the smooth half of $\mathcal{F}$ (P28/P30) and $\ker(\mathcal{R}_\infty)$ with the oscillatory residue $S(T)$
