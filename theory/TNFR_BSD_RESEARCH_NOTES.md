# TNFR–Birch–Swinnerton-Dyer Structural-Pressure Research Notes

**Status**: Pre-registered research programme; BSD-1 diagnostic implemented; obstruction classified as Branch B (open)
**Date**: 2026-06-13
**Scope**: TNFR-internal structural-pressure accumulation across the prime network; **not** a proof of the Clay Birch–Swinnerton-Dyer conjecture
**Primary anchors**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, structural pressure `ΔNFR`, the shipped TNFR L-track (P32–P49, GL(1) Dirichlet), the P14 von-Mangoldt prime-ladder Hamiltonian (GL(1))

---

## 0. Terminology Discipline

This programme is formulated in TNFR language only. References to the
Birch–Swinnerton-Dyer conjecture are treated as an **external comparison
target**. The TNFR object is a nodal structural question on the prime
network of an elliptic curve: *does the accumulated structural pressure
separate curves by the number of their independent rational points?*

No claim in this document should be read as a solution of the Clay
Millennium Problem. The Clay BSD conjecture asserts the rigorous equality of
the **algebraic rank** of the Mordell–Weil group `E(Q)` and the **analytic
order of vanishing** of `L(E, s)` at `s = 1`. Nothing here establishes that
equality.

---

## 1. The GL(1) → GL(2) Gap (existing base and what is missing)

The shipped TNFR L-track builds **Dirichlet** L-functions — a GL(1) object:

| Component | Existing source | Euler factor |
| --- | --- | --- |
| χ-twisted prime ladder (P32) | `src/tnfr/riemann/dirichlet_l.py` | `(1 − χ(p) p^{-s})^{-1}`, `|χ(p)| = 1` |
| von-Mangoldt Hamiltonian (P14) | `src/tnfr/riemann/prime_ladder_hamiltonian.py` | spectrum `{k log p}` |
| Twisted continuation / Weil (P33–P49) | `src/tnfr/riemann/twisted_*` | GL(1) functional equation |

Elliptic-curve L-functions are **GL(2)**:

$$
L(E, s) = \prod_p \bigl(1 - a_p\, p^{-s} + p^{1-2s}\bigr)^{-1},
\qquad a_p = p + 1 - \#E(\mathbb{F}_p), \quad |a_p| \le 2\sqrt{p}\ (\text{Hasse}).
$$

The degree-2 Euler factor carries the coefficient `a_p`, which the GL(1)
track does not. **Building an `a_p`-weighted prime-ladder Hamiltonian (the
GL(2) analogue of P14) is the open milestone BSD-2** and is not assumed.

---

## 2. TNFR-Native Reformulation

Read each prime `p` as a node. The deviation of the local point count from
the neutral value `p + 1`,

$$
a_p = p + 1 - \#E(\mathbb{F}_p),
$$

is the **structural pressure** at prime `p` — the arithmetic analogue of
`ΔNFR` (how far the local reorganisation departs from the neutral count),
bounded by Hasse `|a_p| ≤ 2√p`. The accumulated product

$$
P(X) = \prod_{p \le X} \frac{\#E(\mathbb{F}_p)}{p}
$$

is the accumulated structural coherence of the curve across the prime
network.

> **BSD-1**: Does structural-pressure accumulation `P(X)` separate elliptic
> curves by rank — i.e. does `P(X) ∼ C (log X)^r` with `r` increasing with
> the Mordell–Weil rank?

> **BSD-2** (open): Build the `a_p`-weighted prime-ladder Hamiltonian (GL(2)
> analogue of P14) whose spectral data reproduces `L(E, s)`, and test whether
> the order of vanishing at the central point matches the rank.

BSD-2 (and the rigorous rank ↔ vanishing equality) is the Clay-hard boundary
and is **not** assumed.

---

## 3. BSD-1 Result (DONE)

Birch and Swinnerton-Dyer discovered the conjecture (EDSAC computer, 1965 —
the strictest empirical method) precisely through the growth of `P(X)`.
Reproduced and reframed structurally in
`examples/09_millennium/110_bsd_rank_structural_pressure.py`, using brute-force point
counting `#E(F_p)` (the arithmetic side — **not** an analytic L-function
library), over primes up to 4000 for the standard smallest-conductor curves
of each rank:

| Curve (Cremona) | true rank | `P(X_max)` | empirical slope `r` |
| --- | ---: | ---: | ---: |
| 11a   | 0 | 6.95 | 0.019 |
| 37a   | 1 | 71.1 | 1.137 |
| 389a  | 2 | 311  | 2.047 |
| 5077a | 3 | 2511 | 3.060 |

The slope is `d(log P)/d(log log X)` over the tail, which equals `r` under
`P(X) ∼ C (log X)^r`. The slopes are **strictly ordered by rank** and track
`0, 1, 2, 3`.

**BSD-1 verdict**: structural-pressure accumulation separates the ranks —
the TNFR-native reproduction of the original 1965 BSD empirical discovery.

---

## 4. Honest Obstruction Classification

Using the same A/B trichotomy as the other TNFR Millennium programs:

- **Branch A** (closure inside the existing catalog) — *not established*.
  BSD-1 is the measurement side; it uses **known** ranks and the GL(1) track
  cannot carry `a_p`.
- **Branch B** (open; current classification) — the rank-separation signal is
  real and clean, but (i) the GL(2) `a_p`-weighted prime-ladder Hamiltonian
  (BSD-2) is unbuilt, and (ii) the Clay content — rigorous equality of
  algebraic rank and analytic order of vanishing — is untouched.
- **Branch B3** (no TNFR closure) — not decidable from BSD-1.

This obstruction is structurally analogous to the open residuals of the
sibling programs: the Riemann `S(T)` oscillatory half, the Navier–Stokes
cascade at scale → 0, the Yang–Mills continuum gap (YMG-5), and the P-vs-NP
synthesis trapping (PNP-2). In each case TNFR reformulates and **localises**
the obstruction without closing it.

---

## 5. Milestone Roadmap

| BSD | Title | Status |
| --- | --- | --- |
| BSD-1 | Rank separation via structural-pressure accumulation `P(X)` | **DONE** (`examples/110`) |
| BSD-2 | `a_p`-weighted GL(2) prime-ladder Hamiltonian (analogue of P14) | open |
| BSD-3 | Central order of vanishing of the GL(2) construction vs rank | open |
| BSD-4 | Functional equation / analytic continuation of the GL(2) L-function | open |
| BSD-5 | Rigorous rank ↔ order-of-vanishing equality (Clay-hard boundary) | open, **not assumed** |

BSD-5 is the Clay-strength statement and is not claimed.

---

## 6. What This Program Does and Does Not Do

**Does**: provide a TNFR-native reformulation of BSD as structural-pressure
(`a_p`) accumulation across the prime network; reproduce the original 1965
empirical rank-separation `P(X) ∼ C (log X)^r` from first-principles point
counting; document the GL(1) → GL(2) gap precisely; classify the obstruction
honestly (Branch B, open).

**Does not**: prove BSD; derive the ranks (they are known inputs); build the
GL(2) `a_p`-weighted Hamiltonian (BSD-2, open); establish the rank ↔
order-of-vanishing equality (the Clay content). The TNFR value-add is the
structural FRAMING, not a new mechanism. The program follows the disciplined
pattern of the Riemann, Navier–Stokes, Yang–Mills, and P-vs-NP programs:
reformulate, measure one clean diagnostic, localise the obstruction, and
remain honest about the open boundary.
