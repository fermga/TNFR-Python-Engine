# TNFR–Birch–Swinnerton-Dyer Structural-Pressure Research Notes

**Status**: Auxiliary finite point-count diagnostic; BSD and any nodal dynamics bridge remain open
**Date**: 2026-06-13
**Scope**: Declared arithmetic point-count products and comparison with known rank labels; **not** a nodal derivation or a proof of the Clay Birch–Swinnerton-Dyer conjecture
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
| χ-twisted prime ladder (P32) | `src/tnfr/riemann/dirichlet_l.py` | `(1 − χ(p) p^{-s})^{-1}`; `\|χ(p)\|=1` away from the character modulus, `χ(p)=0` at its prime divisors |
| von-Mangoldt Hamiltonian (P14) | `src/tnfr/riemann/prime_ladder_hamiltonian.py` | spectrum `{k log p}` |
| Twisted continuation / Weil (P33–P49) | `src/tnfr/riemann/twisted_*` | GL(1) functional equation |

For an elliptic curve, the following **degree-two** Euler factor applies at
primes of good reduction:

$$
L(E, s) = \prod_{p\ \mathrm{good}} \bigl(1 - a_p\, p^{-s} + p^{1-2s}\bigr)^{-1}
\prod_{p\ \mathrm{bad}} L_p(E,s),
\qquad a_p = p + 1 - \#E(\mathbb{F}_p), \quad |a_p| \le 2\sqrt{p}\ (\text{Hasse}).
$$

Bad-reduction factors require their own local definition. The Hasse bound in
the display is stated for the smooth reduced elliptic curve. A uniform
degree-two product over every prime would be incorrect.

The degree-2 Euler factor carries the coefficient `a_p`, which the GL(1)
track does not. **Building an `a_p`-weighted prime-ladder Hamiltonian (the
GL(2) analogue of P14) is the open milestone BSD-2** and is not assumed.

---

## 2. Declared arithmetic comparison

Read each prime `p` as a node. The deviation of the local point count from
the neutral value `p + 1`,

$$
a_p = p + 1 - \#E(\mathbb{F}_p),
$$

is an arithmetic deviation, proposed as a pressure-like diagnostic at good
primes. Calling it pressure does not supply an EPI chart, a constitutive
response, a capacity law or a phase/support evolution. The product

$$
P(X) = \prod_{p \le X} \frac{\#E(\mathbb{F}_p)}{p}
$$

is a classical point-count statistic. It is neither the canonical coherence
`1/(1+|pressure|+|dEPI|)` nor the nodal accumulated change `integral nu*p dt`.
The example includes point counts from the displayed equations at bad primes
without constructing the full local L-function factors; that finite product
must not be substituted for the good-reduction Euler product above.

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

**BSD-1 verdict**: the recorded finite slopes are ordered for these four
curves with supplied rank labels. This is not a proved asymptotic, a general
rank detector, an out-of-sample prediction or a TNFR generation mechanism.

---

## 4. Honest Obstruction Classification

Using the same A/B trichotomy as the other TNFR Millennium programs:

- **Branch A** (closure inside the existing catalog) — *not established*.
  BSD-1 is the measurement side; it uses **known** ranks and the GL(1) track
  cannot carry `a_p`.
- **Branch B** (open; current classification) — the rank-separation signal is
  recorded for the selected examples, but (i) the GL(2) `a_p`-weighted prime-ladder Hamiltonian
  (BSD-2) is unbuilt, and (ii) the Clay content — rigorous equality of
  algebraic rank and analytic order of vanishing — is untouched.
- **Branch B3** (no TNFR closure) — not decidable from BSD-1.

The sibling programs also distinguish finite diagnostics from unproved
theorems. Their mathematical obstructions are not thereby equivalent, nor
does the shared vocabulary provide a common nodal law.

---

## 5. Milestone Roadmap

| BSD | Title | Status |
| --- | --- | --- |
| BSD-1 | Rank separation via structural-pressure accumulation `P(X)` | **DONE** (`examples/110`) |
| BSD-2 | `a_p`-weighted GL(2) prime-ladder Hamiltonian (analogue of P14) | open |
| BSD-3 | Central order of vanishing of the GL(2) construction vs rank | open |
| BSD-4 | Functional equation / analytic continuation of the GL(2) L-function | open |
| BSD-5 | Rigorous rank ↔ order-of-vanishing equality (Clay-hard boundary) | open, **not assumed** |

BSD-5 is not claimed. These milestones are a historical comparison inventory;
the [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns active priorities.

---

## 6. What This Program Does and Does Not Do

**Does**: compute finite point-count products for supplied curves and compare
their fitted slopes with supplied ranks; expose the missing degree-two local
data and nodal dynamics bridge. The reported fit is a finite observation,
not a reproduction of a proved asymptotic.

**Does not**: prove BSD; derive the ranks (they are known inputs); build the
GL(2) `a_p`-weighted Hamiltonian (BSD-2, open); establish the rank ↔
order-of-vanishing equality (the Clay content). The TNFR value-add is the
structural FRAMING, not a new mechanism. The program follows the disciplined
pattern of the Riemann, Navier–Stokes, Yang–Mills, and P-vs-NP programs:
reformulate, measure one clean diagnostic, localise the obstruction, and
remain honest about the open boundary.
