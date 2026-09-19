# TNFR Structural Observability — the Symmetry-Sector Theorem (R1)

**Status**: diffusion-sector base case — DERIVED (representation theory) +
MEASURED (finite implementation checks). Pointed operator/word covariance is
tested on selected fixtures. A universal catalog equivariance theorem, a
complete observability theorem and an analytic Riemann obstruction are **not**
established here.
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

By averaging the commutation identities over the finite group, an equivariant
linear operator commutes with $Q_\Gamma$, so it
block-diagonalises:

$$\mathbb{R}^N = \mathrm{Fix}(\Gamma)\ \oplus\ \mathrm{Fix}(\Gamma)^\perp,\qquad
[L_{rw}, Q_\Gamma] = 0 .$$

**Consequence for the flow.** With an orbit-constant `νf` (so `D_νf` also
commutes with `P_σ`), the overdamped nodal-equation flow
$\dot x = -D_{\nu f} L_{rw} x$ preserves both sectors. A symmetric seed stays in
`Fix(Γ)`; this fixed-support diffusion cannot manufacture per-node structure that
separates two nodes in the same orbit. Any such separation must come from the
seed or a departure from the stated equivariant evolution assumptions. This is
not a claim that all possible TNFR evolutions, changing supports or effective
descriptions satisfy those assumptions.

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

**Formal structure (N07).** The stabilizer
`Γ_o = {g ∈ Aut(G) : g(o) = o}` is a subgroup of `Aut(G)`
([pointed_symmetry.py](../src/tnfr/physics/pointed_symmetry.py)), and three facts
make the break *exactly* `Aut(G) → Γ_o`:

- **Orbit–stabilizer** (DERIVED group theory, MEASURED): `|Aut(G)| = |Γ_o| · |orbit(o)|`
  — verified `24 = 6·4` (star leaf) and `12 = 2·6` (cycle vertex).
- **Residual sectors refine, never coarsen** (DERIVED): `Fix(Aut(G)) ⊆ Fix(Γ_o)`
  and `dim Fix(Γ_o) = #orbits(Γ_o) ≥ #orbits(Aut(G))`. The origin splits its
  orbit: star `2 → 3` sectors (`{0},{1,2,3,4} → {0},{1},{2,3,4}`), cycle `1 → 4`.
- **Break localization** (MEASURED): a pointed Emission maps `Fix(Aut(G))` **out**
  of `Fix(Aut(G))` (`break_magnitude ≈ 0.07 > 0`) yet **into** `Fix(Γ_o)`
  (`stabilizer_residual = 0`). The possible marked origins are parameterized by
  the coset set `Aut(G)/Γ_o`, in bijection with `orbit(o)`; this discrete set
  is not a linear space of broken directions. A singleton-orbit origin (the star centre) breaks
  nothing (`break_magnitude = 0`).

**No privileged origin (conjugation).** Origins in one orbit are conjugate:
`Γ_{g(o)} = g Γ_o g⁻¹` and `O@{g(o)} = g (O@o) g⁻¹` (MEASURED), so the pointed
structures at `o` and `g(o)` are isomorphic — the choice of `0` in `(G_{p,k}, 0)`
is a labelling convention, and the arithmetic-pulse observables are independent
of it.

## 4b. Pointed per-operator covariance (finite isolated probes)

The probes compare actions at corresponding selected nodes:
`O_(σv)(P_σ x) = P_σ O_v(x)`, evaluated on a σ-invariant seed.
This covariance of the family of pointed maps differs from equivariance of
one map with a fixed selector. A single selected-node emission can break the
seed symmetry while still obeying the pointed covariance identity
([operator_equivariance.py](../src/tnfr/physics/operator_equivariance.py),
[test_operator_equivariance.py](../tests/operators/test_operator_equivariance.py)).

**Result (MEASURED).** On the vertex-transitive cycle (uniform seed) and the
two-orbit star (orbit-constant seed), **all 13 canonical operators** — AL, EN,
IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH — are equivariant to
the selected tolerance (residual $<10^{-6}$, most exactly $0$) on these
pointed comparisons. The helper reads four scalar channels on the original
node set after refreshing pressure. It does not compare every history,
created node, changed edge, selector or other state attribute; its boolean
`is_equivariant` is a finite probe verdict, not an all-state theorem.

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

## 4c. Word composition closure (N06)

The following algebraic theorem assumes complete equivariance of each map on
its stated domain. The finite pointed probes in §4b do not prove that premise.

**Theorem (composition closure).** If each factor `O_i` is Γ-equivariant
(`O_i ρ(g) = ρ(g) O_i` for every automorphism `g`), then the grammar word
`W = O_k ∘ … ∘ O_1` is Γ-equivariant. *Proof (induction on length).* Length 1 is
the per-operator base case. For `W' = O ∘ W` with `W, O` equivariant,

$$W' \rho(g) = O\,(W \rho(g)) = O\,(\rho(g)\,W) = (O \rho(g))\,W = \rho(g)\,(O W) = \rho(g)\,W'. \qquad\blacksquare$$

**Corollary (`Fix(Γ)` preservation).** An equivariant `W` maps `Fix(Γ)` into
`Fix(Γ)`: if `ρ(g)x = x` for all `g` then `ρ(g) W(x) = W(ρ(g)x) = W(x)`. A
symmetric configuration therefore has no new component in `Fix(Γ)^⊥` under
such a word. Grammar admission alone does not imply the premise, and no
representation placing analytic `S(T)` in this finite complement is supplied.

**Status.** The conditional composition theorem is exact. Small residuals on a
finite base-case sample do not establish its hypothesis on later inputs, or
control error accumulation under arbitrary composition. The independent probes in
[word_equivariance.py](../src/tnfr/physics/word_equivariance.py) report that the
five canonical words (Bootstrap, Bootstrap+close, Stabilize, Propagate, Explore)
have **exactly zero** residual on both test cases, every prefix stays within
tolerance (`composition_closure_holds`), and a Γ-symmetric sweep keeps the seed
orbit-constant (`word_preserves_fix`)
([test_word_equivariance.py](../tests/physics/test_word_equivariance.py)). This
checks the recorded words and prefixes. A sequential sweep also needs its
ordering respected by the group action; merely visiting every node does not
prove equivariance. The pointed-selector result remains distinct (§4).

## 5. Honest scope

The reusable results are fixed-graph diffusion equivariance, the conditional
composition theorem, and the group theory of pointed selectors. The finite
operator/word residuals supplement those proofs without extending their
quantifiers. An invariant input and equivariant observer explain nodewise
constancy on vertex-transitive residue fixtures; arbitrary perturbed states
fall outside that conclusion.

A global spectral invariant is not a vector in the nontrivial sector merely
because it distinguishes graphs. No map identifies analytic `S(T)` with a
finite symmetry complement or REMESH kernel. The R1 decomposition is useful
but not a complete joint-state observability theory, an autonomous symmetry
selection mechanism, or a solution to an external open problem. Current
cross-scale dynamic and geometric output obligations are maintained in
[TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).
