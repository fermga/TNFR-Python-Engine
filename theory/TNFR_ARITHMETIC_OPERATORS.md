# TNFR Arithmetic Operators — Transformation-to-Operator Certification (R8)

**Status**: certificates **MEASURED** (contract residuals, grammar words); two
positive mappings, four negative (boundary) results. No fourteenth operator is
invented; the general "which arithmetic maps are operators" question is **OPEN**
(`NT-P08`).
**Module**: [src/tnfr/mathematics/operator_certificates.py](../src/tnfr/mathematics/operator_certificates.py) ·
**Tests**: [tests/mathematics/test_arithmetic_operator_certificates.py](../tests/mathematics/test_arithmetic_operator_certificates.py) ·
**Benchmark**: [benchmarks/arithmetic_operator_certification.py](../benchmarks/arithmetic_operator_certification.py) ·
**Depends on**: C2 (U3 hard invariant), C5 (claim manifest); contracts from
[operator_contracts.py](../src/tnfr/operators/operator_contracts.py).

## 1. The certificate

Whether an arithmetic transformation *is* a TNFR operator is a **contract**
question. `ArithmeticOperatorCertificate` records, for each candidate, the
transformation, the canonical operator (or `None`), the nodal `state_channel` and
`scale`, the pre/postconditions, the grammar word, and the measured `residuals`.
A mapping is certified only when its measured effect matches the operator's
contract (channel, scale, postcondition — from `contract_for`) **and** its grammar
word validates against U1-U6 (`validate_sequence`). Decorative naming is rejected.

## 2. Positive certificates (MEASURED)

| transformation | operator | channel / scale | residual |
|----------------|----------|-----------------|----------|
| localized **emission at zero** (`e_0` seed) | **Emission (AL)** | EPI / NODE | `epi_direction_violation = 0` (`ΔEPI = +1`) |
| **propagation over residue edges** (additive Cayley transport) | **Resonance (RA)** | EPI / NODE | `conservation_defect = 0` (identity preserved) |

Emission at zero is the R2 pointed-pulse seed: it creates EPI from vacuum at the
neutral node (`∂EPI/∂t ≥ 0`), matching the Emission contract with the valid word
`[emission, coherence, silence]`. Residue-edge propagation is the additive
random-walk transport: one step conserves the total EPI (the identity is
preserved), matching the Resonance contract under the **U3** phase-compatibility
precondition, with the valid word `[emission, resonance, coupling, silence]`.

## 3. Negative certificates — the boundary result (MEASURED)

A negative certificate is a **useful** result: it shows an arithmetic operation
does not fit the 13 operators without an external axiom.

| transformation | verdict | evidence |
|----------------|---------|----------|
| **CRT projection** `σ` | rejected — relabeling | bijection, `channel_modification = 0` |
| **p-adic lift** `ℤ/p^eℤ → ℤ/p^{e+1}ℤ` | rejected — REMESH unverified | `commutation = 0` but `remesh_conditions_unmet = 4` |
| **affine** `x ↦ ax + b` | rejected — automorphism | bijection, `channel_modification = 0` |
| **power map** `x ↦ x^k` | rejected — endomorphism | `many_to_one_factor = gcd(k, ord)` |

- **CRT projection** (R3) is a pure coordinate change; it modifies no nodal
  channel, so it realizes U5 *structure* without being an operator.
- **p-adic lift** (R4) is transport-consistent, but the REMESH contract (EPI
  recursion, NETWORK scale, preserved identity, U5) is unverified — naming it
  REMESH is forbidden until `remesh_contract_audit().realizes_remesh` is `True`.
- **affine** maps are network automorphisms (R1 symmetry sectors), not
  state-modifying operators.
- **power maps** are group endomorphisms; their "contraction" reduces the state
  space to a subgroup, not a node's `νf`, so certifying them as Contraction would
  be decorative.

## 4. Structural morphisms — the emergent taxonomy (N08)

The four rejections are not ad-hoc labels: they are kinds of a **taxonomy that
emerges from the nodal equation** itself
([structural_morphism.py](../src/tnfr/physics/structural_morphism.py)).

**Genus (DERIVED).** For the EPI channel with a common `ν_f` the nodal equation is
`dEPI/dt = −ν_f L · EPI` with flow `EPI(t) = e^{−ν_f t L} EPI(0)`. A linear map
`M : (V_src, L_src) → (V_tgt, L_tgt)` carries **every** source solution to a target
solution, `M e^{−s L_src} = e^{−s L_tgt} M` for all `s`, **iff** it intertwines the
generators, `M L_src = L_tgt M` (differentiate at `s = 0` for ⇒; both sides solve
the same ODE `d/ds(·) = −L_tgt(·)` with equal initial data for ⇐). So a
**structural morphism is exactly a map that transports the nodal-equation flow** —
the intertwining defect equals the nodal-flow-preservation defect
(`nodal_flow_preservation_residual`). This is not an imported category; it is a
property of `∂EPI/∂t = ν_f · ΔNFR`.

**Species.** An intertwiner is fixed by two invariants — the dimension change and
the rank type — giving a grid whose non-empty cells are the kinds:

| kind | dimension | rank | canonical root | emerges |
|------|-----------|------|----------------|---------|
| `AUTOMORPHISM` ⊆ `RELABELING` | preserve | iso (permutation) | R1 symmetry / relabel-invariance (C1) | yes |
| `INTERTWINER` | preserve | iso (non-permutation) | change of coordinates | yes |
| `PROJECTION` ⊇ `COARSE_GRAINING` | preserve / reduce | idempotent / quotient | R1 sector projector `Q_Γ` / U5 fiber quotient (R4) | yes |
| `LIFT` | increase | embedding | U5 prolongation (R4) | yes |
| `ENDOMORPHISM` | preserve | rank-deficient **fold** | — (does not intertwine) | **no** |

Measured (`audit_structural_morphisms`): six kinds have `nodal_flow_residual ≈ 0`
(they emerge); the folding power map `x ↦ x² (mod p)` has `≈ 1.9` — it does **not**
emerge, the R8 boundary. Notably the **Reynolds sector projector `Q_Γ`** (R1) is an
idempotent intertwiner: a rank test alone would misfile it as an endomorphism, but
it is the emergent `PROJECTION` onto `Fix(Γ)`.

**Operator vs morphism (the R8 separation).** A canonical operator acts *within* a
network, `X_G → X_G`, and reorganizes nodal state through `∂EPI/∂t = ν_f · ΔNFR`; a
structural morphism maps *between* networks, `X_G → X_H`, transporting the flow
without reorganizing. Both are grounded in the nodal equation, which is why the
rejections are precise: a relabeling, an automorphism, a fiber quotient and a lift
are morphisms (`is_operator = False`), not the fourteenth operator.

## 5. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| emission at zero → Emission | EPI-increase contract + valid word | **MEASURED** (residual 0) |
| residue propagation → Resonance | EPI conservation + U3 + valid word | **MEASURED** (residual 0) |
| CRT / affine are operators | relabeling changes no channel | **NEGATIVE** |
| p-adic lift is REMESH | REMESH contract unverified | **NEGATIVE** (R4) |
| power map is Contraction | endomorphism, not νf | **NEGATIVE** |
| a fourteenth operator is needed | — | **not invented** (`NT-P08` OPEN) || morphism taxonomy emerges from the nodal equation | intertwiner = nodal-flow transport (N08) | **DERIVED** + MEASURED |
| Reynolds `Q_Γ` is a sector `PROJECTION` (not an endomorphism) | idempotent intertwiner (N08) | **MEASURED** |
| the folding endomorphism does not emerge | `nodal_flow_residual ≈ 1.9` (N08) | **MEASURED** (boundary) |
**Bottom line.** The certification framework maps two arithmetic transformations
to canonical operators with verified contracts and grammar words, and rejects four
others with explicit, measured reasons. No new operator is created; the negatives
sharpen the boundary of the 13-operator catalogue. The general classification of
arithmetic maps into operators is left open (`NT-P08`).
