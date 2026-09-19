# TNFR Arithmetic Operators — Transformation-to-Operator Certification (R8)

**Status**: two candidate contract comparisons and four rejected identifications.
The positive records do not execute AL/RA or certify a full engine transition.
No fourteenth operator is
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
The current `certified` property only checks that `canonical_operator` is not
`None`; it is not a sealed executor certificate. `grammar_valid` separately
checks the listed word through `validate_sequence`. These records reuse
`contract_for` metadata and selected synthetic residuals. They do not validate
every operator pre/postcondition or run a graph-owned history, and the U3 phase
premise of the propagation record is recorded as text rather than measured.

## 2. Positive candidate records (synthetic comparisons)

| transformation | operator | channel / scale | residual |
|----------------|----------|-----------------|----------|
| localized **emission at zero** (`e_0` seed) | **Emission (AL)** | EPI / NODE | `epi_direction_violation = 0` (`ΔEPI = +1`) |
| **propagation over residue edges** (additive Cayley transport) | **Resonance (RA)** | EPI / NODE | `conservation_defect = 0` (identity preserved) |

The first function assigns synthetic scalars `epi_before=0`, `epi_after=1`.
It tests the sign of that increment and records an AL contract name plus the
word `[emission, coherence, silence]`; it neither invokes AL nor derives
creation from vacuum. Zero EPI is not absence of a nodal substrate.

The second multiplies a localized vector by a declared Cayley transition
matrix. Total preservation follows from that matrix's column sums, not from
executing RA. It records the word `[emission, resonance, coupling, silence]`
and a U3 premise without constructing phases. EPI-sum preservation alone is
not preservation of a complete pattern identity or a proof of RA equivalence.

## 3. Negative certificates — the boundary result (MEASURED)

A negative record documents why the tested identification was not certified.
It does not exclude every future canonical realization of an arithmetic map.

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
- **affine** maps with invertible multiplier are permutations. For a Cayley
  graph, they are automorphisms only when the multiplier preserves its
  connection set. The current helper checks bijectivity, not edge preservation.
- **power maps** are group endomorphisms; their "contraction" reduces the state
  space to a subgroup, not a node's `νf`, so certifying them as Contraction would
  be decorative.

## 4. Structural morphisms of the declared linear EPI flow (N08)

The companion helper classifies selected finite linear maps by shape, rank and
intertwining residual
([structural_morphism.py](../src/tnfr/physics/structural_morphism.py)).

**Genus (DERIVED).** For the EPI channel with a fixed common `ν_f>0` the nodal equation is
`dEPI/dt = −ν_f L · EPI` with flow `EPI(t) = e^{−ν_f t L} EPI(0)`. A linear map
`M : (V_src, L_src) → (V_tgt, L_tgt)` carries **every** source solution to a target
solution, `M e^{−s L_src} = e^{−s L_tgt} M` for all `s`, **iff** it intertwines the
generators, `M L_src = L_tgt M` (differentiate at `s = 0` for ⇒; both sides solve
the same ODE `d/ds(·) = −L_tgt(·)` with equal initial data for ⇐). So a
**linear flow morphism is exactly an intertwiner for this fixed EPI model**.
`intertwining_residual` measures the generator defect. The distinct
`nodal_flow_preservation_residual` samples a trajectory from one initial
state; its numerical magnitude need not equal the generator defect, and a
zero sampled result does not prove all-state transport. This uses the
additional constitutive choice `pressure=-L*EPI`, fixed support and common
capacity; it is not derived from the nodal identity alone.

**Classification vocabulary.** Dimension change, rank type and additional
matrix predicates supply labels for the tested maps. These labels do not
uniquely determine a map or classify all intertwiners:

| helper label | dimension | rank | checked construction | finite intertwining result |
|------|-----------|------|----------------|---------|
| `AUTOMORPHISM` ⊆ `RELABELING` | preserve | iso (permutation) | R1 symmetry / relabel-invariance (C1) | yes |
| `INTERTWINER` | preserve | iso (non-permutation) | change of coordinates | yes |
| `PROJECTION` ⊇ `COARSE_GRAINING` | preserve / reduce | idempotent / quotient | R1 sector projector `Q_Γ` / U5 fiber quotient (R4) | yes |
| `LIFT` | increase | embedding | U5 prolongation (R4) | yes |
| `ENDOMORPHISM` | preserve | rank-deficient **fold** | — (does not intertwine) | **no** |

Measured (`audit_structural_morphisms`): six supplied constructions have
`nodal_flow_residual ≈ 0`; the selected folding power map `x ↦ x² (mod p)`
has `≈ 1.9`, so it fails that generator test. This is not a classification
of every folding map. Notably the **Reynolds sector projector `Q_Γ`** (R1) is an
idempotent intertwiner: a rank test alone would misfile it as an endomorphism, but
it is the helper's `PROJECTION` onto `Fix(Γ)`.

**Operator vs linear morphism.** A relabeling, an automorphism, a fiber
quotient and a lift are classified as morphisms (`is_operator = False`) in
this helper. The distinction is not
a universal "within versus between" dichotomy: canonical THOL can change
support, and a projector can act on one state space. A joint morphism must
also transport the actual phase, capacity, support and output laws.

## 5. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| synthetic emission compared with AL | assigned EPI increment + contract metadata | **CANDIDATE**; AL not executed |
| transition step compared with RA | EPI sum + declared U3 premise | **CANDIDATE**; RA/U3 not executed |
| CRT / affine are operators | relabeling changes no channel | **NEGATIVE** |
| p-adic lift is REMESH | REMESH contract unverified | **NEGATIVE** (R4) |
| power map is Contraction | endomorphism, not νf | **NEGATIVE** |
| a fourteenth operator is needed | — | **not established** (`NT-P08` OPEN) |
| intertwiner transports every solution of the fixed linear EPI law | differentiate semigroup / ODE uniqueness | **DERIVED**, with stated hypotheses |
| Reynolds `Q_Γ` is a sector `PROJECTION` (not an endomorphism) | idempotent intertwiner (N08) | **MEASURED** |
| the selected folding map fails to intertwine | `nodal_flow_residual ≈ 1.9` (N08) | **MEASURED** (fixture boundary) |

The current records support selected arithmetic-to-contract comparisons and
linear-flow identities. Full AL/RA transition equivalence, joint-state
morphisms and a general classification remain open. Source API names such as
`certified` retain compatibility and must be read within this scope.
