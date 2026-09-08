# TNFR p-adic Dynamics — the Projective Network Tower (R4)

**Status**: transport consistency **DERIVED** (fiber-averaging intertwining) +
**MEASURED** (exact rational residual `0`; spectrum containment `~1e-15`). The
tower→REMESH identification (`NT-P04`) is **CONJECTURAL** — the REMESH operator
contract is deliberately left **unverified**. No factoring, complexity,
cryptographic, or Millennium claim; no extrapolation to the limit `ℤ_p`.
**Module**: [src/tnfr/mathematics/padic_tower.py](../src/tnfr/mathematics/padic_tower.py) ·
**Tests**: [tests/mathematics/test_padic_tower.py](../tests/mathematics/test_padic_tower.py) ·
**Benchmark**: [benchmarks/padic_scale_consistency.py](../benchmarks/padic_scale_consistency.py) ·
**Depends on**: R3 (multiscale composition), C4 (spectral tolerance),
C5 (claim manifest).

## 1. The tower and its scale maps

The p-adic tower is the inverse system

$$\mathbb{Z}/p\mathbb{Z} \xleftarrow{\pi_1} \mathbb{Z}/p^2\mathbb{Z}
\xleftarrow{\pi_2} \mathbb{Z}/p^3\mathbb{Z} \xleftarrow{\pi_3}\cdots,
\qquad \pi_e(x) = x \bmod p^e.$$

Each coarse node has a **fiber** of `p` fine nodes. Two linear maps encode the
scale change between level `e+1` and level `e`:

- the **fiber-averaging** (aggregation) map `R_e` (`projective_scale_map`),
  a row-stochastic `p^e × p^{e+1}` operator,
  $(R_e f)(y) = \tfrac1p \sum_{x \equiv y\,(p^e)} f(x)$;
- the **lift** (prolongation) map `Lift_e` (`padic_lift_map`), the
  `p^{e+1} × p^e` pullback $(Lift_e\,g)(x) = g(x \bmod p^e)$, constant on fibers.

The scale map is given the **neutral** name `projective_scale_map` — **not**
REMESH (§5).

## 2. The reduction-compatible family

Transport is projective only for connection sets that reduce uniformly over
fibers. The canonical family lifts any base pattern mod `p` by ignoring higher
p-adic digits:

$$S_e = \{\, x \in \mathbb{Z}/p^e\mathbb{Z} : x \bmod p \in \text{base} \,\},
\qquad \text{base} \subseteq (\mathbb{Z}/p\mathbb{Z})\setminus\{0\}.$$

Because fiber-averaging only sees the residue mod `p`, every coarse connection
element is represented with equal multiplicity `p` at the finer level — the
condition that makes the diagrams below commute. Special cases: `base = {1}`
(a lifted successor pattern) and `base = (ℤ/pℤ)^*` (the unit tower, connecting to
R3).

## 3. Projective transport (DERIVED + MEASURED, exact over ℚ)

Writing `P_e = (1/|S_e|) W_e` for the random-walk transition and `L_e = I − P_e`,
the compatible family satisfies **three exact identities** (`fractions.Fraction`,
residual `0`):

$$\boxed{\,R_e\,P_{e+1} = P_e\,R_e\,}\qquad
\boxed{\,P_{e+1}\,Lift_e = Lift_e\,P_e\,}\qquad
\boxed{\,R_e\,Lift_e = I\,}.$$

- **Commutation** `R_e P_{e+1} = P_e R_e`: fiber-averaging the fine transport
  equals coarse-transporting the averaged state — transport is *projective*
  (`projective_commutation_residual`, and its Laplacian form
  `laplacian_commutation_residual`).
- **Intertwining** `P_{e+1} Lift_e = Lift_e P_e`: the lift is an operator
  intertwiner (`lift_intertwining_residual`).
- **Right inverse** `R_e Lift_e = I`: averaging a lifted (fiber-constant) state
  returns it (`lift_reduction_residual`).

**Proof sketch (commutation).** For `s ∈ S_{e+1}`, `(x+s) mod p^e` depends only
on `x mod p^e` and `s mod p^e`; since each coarse shift `t ∈ S_e` has exactly `p`
fine preimages in `S_{e+1}`, summing over the fiber of `y` and over `S_{e+1}`
reproduces `p·d_e` copies of the coarse sum, and the two `1/p` and `1/d`
normalisations collapse to `P_e R_e`. The control (`test_non_uniform_fine_set_
breaks_commutation`) drops one fiber element to make the multiplicity non-uniform
and the residual becomes non-zero — compatibility is necessary.

## 4. Surviving modes (DERIVED + MEASURED)

The intertwining `P_{e+1} Lift_e = Lift_e P_e` has an immediate spectral
consequence. If `P_e v = λ v` then

$$P_{e+1}(Lift_e\,v) = Lift_e(P_e\,v) = \lambda\,(Lift_e\,v),$$

and `Lift_e v ≠ 0` (the lift is injective), so `λ ∈ spec(P_{e+1})`. Hence

$$\operatorname{spec}(L_e) \subseteq \operatorname{spec}(L_{e+1}):$$

**every coarse mode survives the lift**, and level `e+1` only *adds* finer,
fiber-varying modes (those annihilated by `R_e`). Numerically the coarse
spectrum sits within the fine spectrum to `~1e-15`
(`surviving_spectrum_containment`, below the derived tolerance `√ε·‖L‖₂`). The
measured spectral gap `λ₂(L_e)` decreases from level 1 and then **stabilises**
(e.g. `p = 3`, unit base: `1.5 → 1.0 → 1.0`) — the slow-mode timescale reaches a
fixed p-adic scale.

## 5. REMESH is not claimed (the contract audit)

TNFR's REMESH operator (recursivity, [AGENTS.md](../AGENTS.md) §5) has a specific
contract: a recursive **EPI echo across scales**, **NETWORK**-scale
generator/closure, **preserved nodal identity**, and **U5** multiscale coherence.
Projective transport consistency (§3–4) is **necessary but not sufficient** for
that contract. `RemeshContractAudit` records the four conditions, all
**unverified**, so `remesh_contract_audit().realizes_remesh == False` and the
map keeps its neutral name. The claim *"a compatible p-adic tower realises
REMESH"* (`NT-P04`) is therefore **CONJECTURAL**; naming the reduction REMESH is
forbidden until every contract field is independently established.

### 5b. The R4b campaign — temporal memory and the U5 boundary (N09)

The campaign measures the missing temporal ingredient
([remesh_audit.py](../src/tnfr/mathematics/remesh_audit.py)) without conflating
field dispersion with hierarchy coherence. The static tower candidate is the
same-scale projection `P = Lift_e · R_e`; the temporal candidate is
`EPI_new = (1-α)² EPI(t) + α(1-α) EPI(t-τ_l) + α EPI(t-τ_g)`.
Their default audits are:

| candidate | temporal echo | NETWORK probe | identity probe | `1/(1+std(EPI))` | declared U5 | REMESH |
|-----------|---------------|---------------|----------------|------------------|-------------|--------|
| static `Lift·R_e` | **no** (`0`) | yes | yes | preserved | unverified | **no** |
| temporal recurrence | yes (`≈ 1.5`) | yes | yes | preserved | unverified | **no** |

The dispersion score is now reported as `field_uniformity_*`. It reads only the
sampled scalar EPI field; it does not read `DeltaNFR` or `dEPI` and declares no
parent/child relation. It therefore cannot set `u5_multiscale_verified`.
The default campaign establishes only that the recurrence responds to delayed
inputs while the projection does not: `temporal_echo_discriminates == True`, but
`audit_discriminates == False` because U5 has no declared evidence.

A caller can provide candidate-specific `RemeshU5Evidence` for a concrete graph
materialized after the update. The audit then invokes
`assess_u5_parent_child_coherence` with the declared parent, children, alpha and
tolerance. A satisfying assessment can complete the temporal candidate's scoped
four-condition audit. This is evidence for that hierarchy and state; it is not a
universal U5-preservation theorem for later states or arbitrary recurrences.

## 6. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `R_e P_{e+1} = P_e R_e` (compatible family) | uniform-fiber reduction | **DERIVED** + MEASURED (exact `0`) |
| `P_{e+1} Lift_e = Lift_e P_e` | operator intertwining | **DERIVED** + MEASURED (exact `0`) |
| `R_e Lift_e = I` | fiber-constant averaging | **DERIVED** + MEASURED (exact `0`) |
| `spec(L_e) ⊆ spec(L_{e+1})` | intertwining corollary | **DERIVED** + MEASURED (`~1e-15`) |
| compatibility necessary | non-uniform control | **MEASURED** (residual `≠ 0`) |
| tower realises REMESH | contract unverified | **CONJECTURAL** (`NT-P04`, `realizes_remesh = False`) |
| static `Lift·R_e` lacks temporal echo and declared U5 evidence | scoped audit (N09) | **MEASURED** (2/4 pass, echo `= 0`) |
| temporal recurrence has delayed-input sensitivity | temporal probe (N09) | **MEASURED** (`temporal_echo_discriminates = True`) |
| default campaign realises REMESH | no declared hierarchy | **UNVERIFIED** (`audit_discriminates = False`) |
| recurrence with declared hierarchy evidence | canonical U5 assessment | **SCOPED** to the supplied post-update state |

**Scope.** The reduction-compatible p-adic tower carries a genuinely
projective transport: coarse and fine dynamics commute with fiber-averaging, the
lift intertwines the levels, and coarse modes survive exactly. This is a
structural scale-consistency result on small `p` and low exponents. It does
**not** identify the scale map with REMESH, and it closes no open problem.
