# TNFR–Hodge Discrete Cochain Research Notes

**Status**: Pre-registered research programme; HC-1 diagnostic implemented; obstruction classified **Branch B3-leaning** (strong negative — no discrete TNFR closure of the actual conjecture)
**Date**: 2026-06-13
**Scope**: TNFR-internal discrete/combinatorial Hodge theory on the tetrad cochain tower; **not** a proof or attack on the Clay Hodge conjecture
**Primary anchors**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, structural field tetrad `(Φ_s, |∇φ|, K_φ, ξ_C)`, the k=1 Helmholtz–Hodge decomposition already shipped in `examples/107`

---

## 0. Terminology Discipline

This programme is formulated in TNFR language only. References to the Hodge
conjecture are treated as an **external comparison target**.

No claim in this document should be read as a solution of the Clay Millennium
Problem. The Clay Hodge conjecture asserts: on a non-singular complex
projective variety, every Hodge class (a rational cohomology class of type
`(p,p)`) is a rational combination of cohomology classes of **algebraic
cycles** (subvarieties cut out by polynomial equations). Nothing here
establishes that statement; this programme delivers an **honest strong
negative** about the reach of the discrete/structural setting.

---

## 1. TNFR-Native Object: the Tetrad Cochain Tower

Example 107 established the `k=1` Helmholtz–Hodge decomposition of the phase
field on a graph (gradient ⊕ cycle). Extending the field to a 2-complex
(triangles) gives the **full** discrete Hodge decomposition. The tetrad
supplies the natural cochain degrees:

| Degree | Tetrad object | Cochain |
| --- | --- | --- |
| 0 | phase value | vertex 0-cochain |
| 1 | phase gradient `|∇φ|` | edge 1-cochain |
| 2 | phase curvature `K_φ` | triangle 2-cochain (discrete curl / holonomy) |

With simplicial boundary maps `d1` (edges → vertices) and `d2` (triangles →
edges), the combinatorial Hodge Laplacians are

$$
L_0 = d_1 d_1^{\mathsf T},\quad
L_1 = d_1^{\mathsf T} d_1 + d_2 d_2^{\mathsf T},\quad
L_2 = d_2^{\mathsf T} d_2 .
$$

Eckmann's theorem (1944): harmonic `k`-cochains `≅` homology `H_k`, so
`dim ker L_k = b_k` (the `k`-th Betti number).

---

## 2. HC-1 Result (DONE)

Reproduced in `examples/09_millennium/111_hodge_discrete_and_honest_gap.py`.

- **Chain complex.** `d1 d2 = 0` to machine precision (the tetrad cochain
  tower is a genuine complex).
- **Eckmann, exact.** On a triangulated torus (`|V|=25, |E|=75, |T|=50`,
  Euler `0`): harmonic dimensions `(dim ker L_0, L_1, L_2) = (1, 2, 1)` =
  Betti `(1, 2, 1)`. The 2 harmonic 1-forms are closed (`|d1 h| ~ 1e-16`) and
  co-closed (`|d2^T h| ~ 1e-15`).
- **Topology tracking.** Octahedron (sphere) gives harmonic dims `(1, 0, 1)` =
  Betti `(1, 0, 1)`; the torus gives `(1, 2, 1)`. The harmonic count is a
  faithful topological invariant: the sphere has no 1-loops, the torus has 2
  (the loops a TNFR phase field can wind around; cf. `examples/107`).

**HC-1 verdict**: the TNFR cochain tower carries a complete discrete Hodge
decomposition; harmonic = homology exactly, across spaces.

---

## 3. The Honest Gap (why this is NOT the Hodge conjecture)

Two features constitute the difficulty of the Hodge conjecture, and the
discrete TNFR setting has **neither**:

- **A. Complex `(p,p)` bigrading.** The conjecture lives in the Hodge
  decomposition `H^k = ⊕_{p+q=k} H^{p,q}` of a Kähler manifold, requiring a
  **complex structure**. The real combinatorial Laplacian `L_k` has no
  `(p,q)` bigrading — only one real harmonic space per degree.
- **B. Algebraicity.** An "algebraic cycle" is cut out by polynomial
  equations — strictly stronger than an "integer topological cycle." In the
  combinatorial setting **every** harmonic class is already an integer
  simplicial cycle (Eckmann), so the discrete analogue of the conjecture is
  **trivially true** — precisely because the discrete setting cannot even
  express the algebraicity distinction that is the whole difficulty.

So TNFR's discrete Hodge captures the **topological** half (harmonic =
homology) exactly and is **structurally blind** to the complex-algebraic
content. This is the honest analogue of the Riemann result that the emergent
substrate is "blind" to arithmetic content.

---

## 4. Honest Obstruction Classification

- **Branch A** (closure inside the catalog) — not applicable; the conjecture
  cannot even be posed discretely.
- **Branch B** (open attack surface) — *not* the right classification here.
  Unlike P-vs-NP (PNP-2) or BSD (BSD-2), there is **no concrete discrete next
  milestone** toward the conjecture: bridging to `(p,p)` bigrading and
  algebraicity requires leaving the discrete/structural setting entirely.
- **Branch B3-leaning** (no TNFR closure) — current classification. The
  discrete TNFR setting is **structurally blind** to the actual conjecture;
  the value delivered is a precise localisation of *why* Hodge is hard, not an
  attack surface on it.

This is the **strongest negative** of the TNFR Millennium programs. Where the
Riemann `S(T)` residual, the NS cascade, the Yang–Mills continuum gap, and the
P-vs-NP trapping are *open obstructions with attack surfaces*, the Hodge gap is
a *qualitative blindness*: the discrete cochain tower cannot represent the
algebraic-complex structure at all.

---

## 5. Milestone Roadmap

| HC | Title | Status |
| --- | --- | --- |
| HC-1 | Discrete Hodge on the tetrad cochain tower (Eckmann); honest gap | **DONE** (`examples/111`) |
| HC-2 | Whether any TNFR-native complex structure induces a `(p,p)` bigrading | open, **expected negative** |
| HC-3 | Whether algebraicity has any structural (non-topological) TNFR analogue | open, **expected negative** |

HC-2 and HC-3 are recorded for completeness; the honest a-priori expectation
is that the discrete/structural setting cannot supply either ingredient.

---

## 6. What This Program Does and Does Not Do

**Does**: provide a TNFR-native discrete Hodge decomposition on the tetrad
cochain tower; verify Eckmann (harmonic = homology) exactly; localise
precisely the two features (complex `(p,p)` bigrading, algebraicity) that the
discrete setting cannot represent; classify the obstruction honestly as the
strongest negative (Branch B3-leaning).

**Does not**: prove, disprove, or even attack the Hodge conjecture; claim any
bridge from discrete harmonic classes to algebraic cycles; introduce a complex
or Kähler structure. The TNFR value-add is a precise honest delimitation of
the reach of the structural setting — consistent with the disciplined pattern
of the Riemann, Navier–Stokes, Yang–Mills, P-vs-NP, and BSD programs.
