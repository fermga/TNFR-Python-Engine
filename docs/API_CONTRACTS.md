# TNFR Operator API Contracts

**Status:** Active normative view
**Repository version:** 0.0.3.5
**Owner:** [`tnfr.operators.operator_contracts`](../src/tnfr/operators/operator_contracts.py)

This document is a readable view of the canonical operator contracts. The code
module above owns channel, scale, direction, postcondition, and source anchors.
Changes begin in that module and must pass its consistency assertions and tests.

## Contract model

Every canonical operator has:

- a public English name and internal glyph;
- one primary nodal-equation channel;
- node or network scale;
- a direct-effect direction;
- a verifiable postcondition;
- independent grammar and state preconditions.

The channel classification does not replace grammar roles. For example,
Mutation acts primarily on phase while also being a U2 destabilizer and U4
transformer.

## Canonical contracts

| Operator | Glyph | Primary channel | Scale | Direct postcondition |
| --- | --- | --- | --- | --- |
| Emission | AL | EPI | node | EPI does not decrease |
| Reception | EN | EPI | node | Coherence does not decrease during coherent integration |
| Resonance | RA | EPI | node | EPI structural identity is preserved |
| Silence | SHA | structural frequency | node | Structural frequency does not increase |
| Expansion | VAL | structural frequency | node | Structural frequency does not decrease |
| Contraction | NUL | structural frequency | node | Structural frequency does not increase |
| Coupling | UM | phase | node | Pressure magnitude does not increase under mutual stabilization |
| Mutation | ZHIR | phase | node | Phase is transformed when mutation preconditions hold |
| Coherence | IL | pressure | node | Pressure magnitude and coherence do not worsen |
| Dissonance | OZ | pressure | node | Pressure magnitude does not decrease |
| Self-organization | THOL | pressure | node | Global form is preserved without catastrophic coherence loss |
| Transition | NAV | pressure | node | At least one controlled state channel changes |
| Recursivity | REMESH | EPI | network | EPI mixes toward temporal or multi-scale history |

For exact wording and measured context, inspect
[`OPERATOR_CONTRACTS`](../src/tnfr/operators/operator_contracts.py).

## Six global invariants

The canonical invariant list is owned by
[AGENTS.md](../AGENTS.md#8-canonical-invariants):

1. nodal-equation integrity;
2. phase-coherent coupling;
3. multi-scale fractality;
4. grammar compliance;
5. structural metrology;
6. reproducible dynamics.

This document does not define a parallel invariant list.

## Valid execution example

```python
from tnfr.operators.definitions import Coherence, Emission, Silence
from tnfr.structural import create_nfr, run_sequence

G, node = create_nfr("seed", epi=0.1, vf=1.0, theta=0.0)
run_sequence(G, node, [Emission(), Coherence(), Silence()])
```

The word starts with a U1 generator, contains a stabilizer, and ends with a U1
closure. State-dependent operator preconditions remain mandatory during
execution. Coupling and Resonance additionally enforce U3 before mutation.

## Contract verification

- [`test_operator_contracts.py`](../tests/operators/test_operator_contracts.py)
  verifies catalog coverage and direct effects.
- [`test_u3_hard_invariant.py`](../tests/operators/test_u3_hard_invariant.py)
  verifies rejection before Coupling or Resonance mutation.
- [`test_canonical_operators_modern.py`](../tests/operators/test_canonical_operators_modern.py)
  covers operator behavior and latency.
- [Grammar Physics Verification Map](grammar/PHYSICS_VERIFICATION.md) maps U1-U6
  to their implementation and scope.

## Extension rule

The catalog is fixed at 13 canonical operators. A domain feature should compose
existing operators or remain a diagnostic/morphism outside the catalog. Any
proposal to alter the catalog requires a nodal-equation channel, scale,
postcondition, grammar classification, tests, and an update to the canonical
synthesis; adding a class or registry entry alone does not establish
canonicity.
