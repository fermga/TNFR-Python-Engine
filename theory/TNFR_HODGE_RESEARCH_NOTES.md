# TNFR–Hodge Auxiliary Simplicial Baseline Notes

**Status**: HC-1 auxiliary simplicial baseline implemented; no TNFR tetrad-to-cochain bridge has been derived
**Date**: 2026-06-13
**Scope**: standard finite simplicial Hodge theory used as an auxiliary comparison; **not** a proof or attack on the Clay Hodge conjecture
**Primary anchors**: independently constructed incidence matrices in example 111; the canonical TNFR tetrad supplies only a comparison target

---

## 0. Terminology Discipline

References to the Hodge conjecture are treated as an **external comparison
target**. The executable result below is standard combinatorial Hodge theory,
not a consequence of the nodal equation or the 13 operators.

No claim in this document should be read as a solution of the Clay Millennium
Problem. The Clay Hodge conjecture asserts: on a non-singular complex
projective variety, every Hodge class (a rational cohomology class of type
`(p,p)`) is a rational combination of cohomology classes of **algebraic
cycles** (subvarieties cut out by polynomial equations). Nothing here
establishes that statement; this programme delivers an **honest strong
negative** about the reach of the discrete/structural setting.

---

## 1. Auxiliary simplicial cochain baseline

Example 107 verifies an incidence-matrix orthogonality for an oriented EPI edge
gradient and a selected circulation. Example 111 independently extends a graph
to a finite 2-complex and computes its standard simplicial Hodge Laplacians.
The carriers in that auxiliary complex are:

| Degree | Carrier | Cochain |
| --- | --- | --- |
| 0 | vertices | scalar vertex values |
| 1 | oriented edges | signed edge values |
| 2 | oriented faces | signed face values |

The canonical `compute_phase_gradient` returns a nonnegative mean per **node**
and discards edge orientation and sign. `compute_phase_curvature` also returns a
wrapped scalar per **node**, not a face value. They therefore do not instantiate
the 1- and 2-cochains in this table. A genuine TNFR bridge would need distinct
oriented edge and face observables plus an explicit compatibility proof; neither
is implemented here.

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

## 2. HC-1 auxiliary result (DONE)

Reproduced in `examples/09_millennium/111_hodge_discrete_and_honest_gap.py`.

- **Chain complex.** The independently constructed incidence matrices satisfy
  `d1 d2 = 0` to machine precision, as required by a simplicial complex.
- **Eckmann baseline.** On a triangulated torus (`|V|=25, |E|=75, |T|=50`,
  Euler `0`), the numerically detected harmonic dimensions
  `(dim ker L_0, L_1, L_2) = (1, 2, 1)` match Betti `(1, 2, 1)`. The two sampled
  harmonic 1-forms are closed (`|d1 h| ~ 1e-16`) and co-closed
  (`|d2^T h| ~ 1e-15`). Eckmann's theorem is the exact external result; the
  reported ranks use a declared numerical tolerance.
- **Topology comparison.** An octahedral sphere gives harmonic dimensions
  `(1, 0, 1)` and the torus gives `(1, 2, 1)`. Betti numbers distinguish these
  two complexes but are not complete invariants of topological spaces.

**HC-1 verdict**: the auxiliary finite complexes reproduce the standard
combinatorial Hodge decomposition. This does not establish a tetrad cochain
tower or a new TNFR theorem.

---

## 3. The Honest Gap (why this is NOT the Hodge conjecture)

Two features constitute the difficulty of the Hodge conjecture, and the
discrete TNFR setting has **neither**:

- **A. Complex `(p,p)` bigrading.** The conjecture lives in the Hodge
  decomposition `H^k = ⊕_{p+q=k} H^{p,q}` of a Kähler manifold, requiring a
  **complex structure**. The real combinatorial Laplacian `L_k` has no
  `(p,q)` bigrading — only one real harmonic space per degree.
- **B. Algebraicity.** An "algebraic cycle" is cut out by polynomial
  equations — strictly stronger than a topological cycle. Finite simplicial
  Hodge theory identifies harmonic representatives with real cohomology; it
  neither makes every harmonic representative integral nor supplies the
  algebraic-cycle subspace. The Hodge conjecture therefore has no faithful
  analogue in this baseline.

The auxiliary discrete model captures topological cohomology and is blind to
the complex-algebraic content. Since no map from the canonical TNFR fields to
its cochains has been proved, even this topological result must not be promoted
to a property of the tetrad.

---

## 4. Honest Obstruction Classification

- **Branch A** (closure inside the catalog) — not applicable; the conjecture
  cannot even be posed discretely.
- **Branch B** (open attack surface) — *not* the right classification here.
  Unlike P-vs-NP (PNP-2) or BSD (BSD-2), there is **no concrete discrete next
  milestone** toward the conjecture: bridging to `(p,p)` bigrading and
  algebraicity requires leaving the discrete/structural setting entirely.
- **Branch B3-leaning** (no closure through this baseline) — current
  classification. This particular auxiliary finite model cannot express the
  actual conjecture. That is a scope result about the model, not a proof that
  every possible TNFR extension is incapable of doing so.

This is the **strongest negative** of the TNFR Millennium programs. Where the
Riemann `S(T)` residual, the NS cascade, the Yang–Mills continuum gap, and the
P-vs-NP trapping are *open obstructions with attack surfaces*, the Hodge gap is
a *qualitative blindness*: the discrete cochain tower cannot represent the
algebraic-complex structure at all.

---

## 5. Milestone Roadmap

| HC | Title | Status |
| --- | --- | --- |
| HC-1 | Auxiliary finite simplicial Hodge baseline (Eckmann); honest gap | **DONE** (`examples/111`) |
| HC-1b | Derive oriented TNFR edge/face cochains and prove compatibility with canonical telemetry | **OPEN** |
| HC-2 | Whether any TNFR-native complex structure induces a `(p,p)` bigrading | open, **expected negative** |
| HC-3 | Whether algebraicity has any structural (non-topological) TNFR analogue | open, **expected negative** |

HC-2 and HC-3 are recorded for completeness; the honest a-priori expectation
is that the discrete/structural setting cannot supply either ingredient.

---

## 6. What This Program Does and Does Not Do

**Does**: reproduce standard finite simplicial Hodge calculations; compare two
finite complexes with their expected Betti numbers; identify the missing
complex `(p,p)` bigrading and algebraic-cycle data; expose the absent
tetrad-to-cochain bridge.

**Does not**: prove, disprove, or attack the Hodge conjecture; show that
`|∇φ|` is an oriented 1-cochain or `K_φ` a face 2-cochain; claim a bridge from
discrete harmonic classes to algebraic cycles; introduce a complex or Kähler
structure.
