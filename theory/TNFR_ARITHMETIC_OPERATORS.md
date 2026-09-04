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

## 4. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| emission at zero → Emission | EPI-increase contract + valid word | **MEASURED** (residual 0) |
| residue propagation → Resonance | EPI conservation + U3 + valid word | **MEASURED** (residual 0) |
| CRT / affine are operators | relabeling changes no channel | **NEGATIVE** |
| p-adic lift is REMESH | REMESH contract unverified | **NEGATIVE** (R4) |
| power map is Contraction | endomorphism, not νf | **NEGATIVE** |
| a fourteenth operator is needed | — | **not invented** (`NT-P08` OPEN) |

**Bottom line.** The certification framework maps two arithmetic transformations
to canonical operators with verified contracts and grammar words, and rejects four
others with explicit, measured reasons. No new operator is created; the negatives
sharpen the boundary of the 13-operator catalogue. The general classification of
arithmetic maps into operators is left open (`NT-P08`).
