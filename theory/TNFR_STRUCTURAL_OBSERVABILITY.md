# TNFR Structural Observability — the Symmetry-Sector Theorem (R1)

**Status**: diffusion-sector base case — DERIVED (representation theory) +
MEASURED (implementation-certified). The full operator-by-operator equivariance
theorem is a later R1 stage and is **not** claimed here.
**Modules**: [src/tnfr/physics/symmetry_sectors.py](../src/tnfr/physics/symmetry_sectors.py),
[src/tnfr/physics/equivariance.py](../src/tnfr/physics/equivariance.py) ·
**Tests**: [tests/physics/test_symmetry_sectors.py](../tests/physics/test_symmetry_sectors.py) ·
**Origin**: formalises [example 123](../examples/08_emergent_geometry/123_symmetry_sector_decomposition.py).

## 1. The question

What information can a TNFR dynamics generate or resolve when the graph, the
initial condition, `νf`, the node selection and the operators all respect a
symmetry group? The answer bounds every "wall" in the §9.5–§9.10 number-theory
arc: some information is local, some lives in collective modes, some was injected
from outside.

## 2. The diffusion-sector result (DERIVED)

Let `Γ = Aut(G, W)` act on `ℝᴺ` by the permutation representation `σ ↦ P_σ`. The
canonical structural-diffusion operator `L_rw = I − D⁻¹W` (the EPI channel of the
nodal equation, [structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py))
is **equivariant**:

$$P_\sigma\, L_{rw} = L_{rw}\, P_\sigma \qquad \forall\,\sigma\in\Gamma .$$

The **Reynolds projector** $Q_\Gamma = \tfrac{1}{|\Gamma|}\sum_{\sigma}P_\sigma$
satisfies $\operatorname{im}(Q_\Gamma)=\mathrm{Fix}(\Gamma)$ and
$\ker(Q_\Gamma)=\mathrm{Fix}(\Gamma)^\perp$, with

$$\dim\mathrm{Fix}(\Gamma) = \#\{\text{vertex orbits of }\Gamma\}.$$

By Schur's lemma an equivariant operator commutes with $Q_\Gamma$, so it
block-diagonalises:

$$\mathbb{R}^N = \mathrm{Fix}(\Gamma)\ \oplus\ \mathrm{Fix}(\Gamma)^\perp,\qquad
[L_{rw}, Q_\Gamma] = 0 .$$

**Consequence for the flow.** With an orbit-constant `νf` (so `D_νf` also
commutes with `P_σ`), the overdamped nodal-equation flow
$\dot x = -D_{\nu f} L_{rw} x$ preserves both sectors. A symmetric seed stays in
`Fix(Γ)`; the canonical dynamics alone cannot manufacture per-node structure that
separates two nodes in the same orbit. Any such separation must come from the
seed or an external per-node lever, not from the operator.

## 3. Certification (MEASURED)

[verify_diffusion_equivariance](../src/tnfr/physics/equivariance.py) measures, on
each test graph, the equivariance residual $\max_\sigma\lVert P_\sigma L - L
P_\sigma\rVert_2$ and the sector-preservation residual $\lVert L Q_\Gamma -
Q_\Gamma L\rVert_2$, both against the derived tolerance $\tau=\sqrt{\varepsilon}\,
\lVert L\rVert_2$. Across cycle, complete, star, path and torus:

| Graph | orbits = `dim Fix(Γ)` | equivariant | sectors preserved |
|---|---|---|---|
| cycle `C₈` (vertex-transitive) | 1 | yes | yes |
| complete `K₆` | 1 | yes | yes |
| star `K₁,₅` | 2 (center, leaves) | yes | yes |
| path `P₆` | 3 (ends, near-ends, middle) | yes | yes |
| torus `C₃×C₃` | 1 | yes | yes |

`rank Q_Γ = #orbits` in every case; the residuals are at machine precision. The
star shows the result does **not** need a symmetric operator: `L_rw` there is
non-symmetric yet still commutes with `Aut = S₅`. Results are relabel-invariant
and weight/direction aware (a distinguished heavy edge lowers the symmetry and
raises the orbit count; a directed cycle stays single-orbit under rotations).

## 4. Pointed graphs (declared symmetry breaking)

Selecting a single origin node `o` (a localized emission) is **not** equivariant.
It changes the object of study from `G` to the **pointed graph** `(G, o)` and
reduces the relevant group to the stabilizer `Γ_o`. This is legitimate provided
the break is *declared*: the reduced symmetry is stated, not presented as
spontaneous emergence. This distinction is the basis of the arithmetic-pulse
programme (R2), which studies the pointed residue network `(G_{p,k}, 0)`.

## 4b. Per-operator equivariance (all 13, certified under isolation)

The non-linear stage asks whether each operator `O` is equivariant,
`O(P_σ x) = P_σ O(x)`. The testable form: an equivariant operator maps a
`Fix(Γ)` state to a `Fix(Γ)` state, so applying it at `v` and at `σ(v)` on a
σ-invariant seed gives σ-related results
([operator_equivariance.py](../src/tnfr/physics/operator_equivariance.py),
[test_operator_equivariance.py](../tests/operators/test_operator_equivariance.py)).

**Result (MEASURED).** On the vertex-transitive cycle (uniform seed) and the
two-orbit star (orbit-constant seed), **all 13 canonical operators** — AL, EN,
IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH — are equivariant to
machine precision (residual $<10^{-6}$, most exactly $0$). Combined with the
diffusion base case, a grammar-composed *word* of equivariant operators is
equivariant, so no canonical word can move a symmetric state into
$\mathrm{Fix}(\Gamma)^\perp$ — the §8.1.7 falsification search finds no
counterexample.

**Root cause and fix (N01).** The certification requires each operator to be
measured from an **independent, clean graph cache**. TNFR content-keyed caches
(`_dnfr_prep_cache`, `_node_set_checksum_cache`, the graph cache managers) live in
`G.graph` and survive `networkx`'s `G.copy()` by **shared reference**; keyed on a
**label-independent** node-set checksum, they collide between the two isomorphic
copies `A` and `B = σ(A)`, so `B` reads `A`'s cached ΔNFR prep. On reused seed
graphs this leaked across operators and made a *batch* audit report spurious
$\sim 10^{-3}$ residuals (verified: Silence is $0$ in isolation, $\sim 10^{-3}$
only after other operators run). The fix (`_isolate_graph_caches`) drops those
content caches on each copy, so every measurement is an **independent
experiment** with a private cache (report R1-T01: prefer per-experiment local
cache when a global content key cannot distinguish isomorphic siblings). With it
the batch audit is exactly $0$ for all 13 operators, independent of cache warmth,
operator order, or repeated runs
([test_equivariance_cache_isolation.py](../tests/physics/test_equivariance_cache_isolation.py)).
The config keys (`_dnfr_weights`, `_DNFR_META`) are preserved — only the caches
are isolated.

## 5. Honest scope

This is the representation theory of graph automorphisms (Schur's lemma)
re-expressed in the canonical emergent operator, with an implementation
certificate. It **explains and unifies** the walls of the §9.5–§9.10 arc — the
residue-digraph vertex-transitivity wall (ex 120), the substrate blindness, the
spectral primality (ex 119) and the Riemann residual
$S(T)\in\ker(\mathcal R_\infty)\cap\mathrm{Fix}(S_n)^\perp$ — as one
`Fix(Γ)/Fix(Γ)^⊥` split for different groups. It is **not** new mathematics and
closes no open problem. The diffusion base case **and** the per-operator audit
(§4b, all 13 equivariant under isolation) are certified; what remains for the
*fully general* theorem is (i) a formal — not only measured — proof that
grammatical composition preserves equivariance, and (ii) the treatment of
non-equivariant **selectors** (a specific-node selector is pointed, §4), which is
a property of the selection policy, not of the operators.
