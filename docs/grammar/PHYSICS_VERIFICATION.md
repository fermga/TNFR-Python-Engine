# TNFR Grammar Physics Verification Map

**Status:** Active scope and verification index
**Canonical specification:** [Unified Grammar Rules](../../theory/UNIFIED_GRAMMAR_RULES.md)

This document maps grammar requirements to their physical interpretation,
implementation, and tests. It does not claim that all U1-U6 requirements are
theorems of the bare differential identity.

## Scope

The nodal equation

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}
$$

supplies the dynamical meaning of form, capacity, and pressure. Grammar U1-U6
adds initialization, operator-composition, state-dependent coupling, nesting,
and monitoring contracts. These contracts are mandatory for the TNFR engine,
but their mathematical status differs:

| Rule | Engine requirement | Mathematical status |
| --- | --- | --- |
| U1 | Generator and closure context | Initialization and finite-word boundary contract |
| U2 | Bound destabilizer debt with stabilizers | Finite-word policy calibrated from a mean-rate relaxation surrogate |
| U3 | Verify wrapped phase compatibility for Coupling and Resonance | Runtime precondition tied to the phase gate |
| U4 | Require handlers and recent destabilizer context | Bifurcation composition contract with a calibrated recency window |
| U5 | Stabilize declared nested EPI depth | Multi-scale identity and execution contract |
| U6 | Monitor structural-potential drift | Read-only safety policy, not a sequence constraint |

The complete hypotheses, counterexamples, and limits are centralized in
[Diagnostic and Grammar Scope](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

## Corrections to the historical proof narrative

- `EPI = 0` does not make the nodal derivative undefined. Finite `nu_f` and
  `Delta NFR` can produce a finite derivative at zero. U1 therefore requires
  operator initialization context rather than following from a singularity.
- A destabilizer token does not by itself prove exponential divergence. U2
  limits uncompensated structural-pressure debt in finite words.
- The canonical U2 debt capacity and U4 recency window evaluate to 2 and 3 under
  the documented mean-rate calibration. They are not uniform modal relaxation
  theorems for every graph.
- U5 preserves nested identity operationally. A general inequality between
  parent and child coherence requires an explicit hierarchy and normalization.
- The `pi/4` potential magnitude and `pi/2` drift values are selected safety
  policies. Crossing them records a U6 warning; it does not prove
  fragmentation.

## Source chain

1. Operator predicates and classifications:
   [`physics_derivation.py`](../../src/tnfr/config/physics_derivation.py)
2. Canonical rule materialization:
   [`grammar_canon.py`](../../src/tnfr/operators/grammar_canon.py)
3. Public validation facade:
   [`grammar.py`](../../src/tnfr/operators/grammar.py)
4. Runtime state checks:
   [operator preconditions](../../src/tnfr/operators/preconditions/)
5. Grammar-aware application:
   [`grammar_application.py`](../../src/tnfr/operators/grammar_application.py)

Consumers must import the shared classifications instead of reproducing
operator sets.

## Verification coverage

| Contract | Principal tests |
| --- | --- |
| Role derivation and source consistency | [`test_grammar_canon.py`](../../tests/operators/test_grammar_canon.py), [`test_grammar_canonical_consistency.py`](../../tests/operators/test_grammar_canonical_consistency.py) |
| U1-U4 context and history | [`test_grammar_dynamics.py`](../../tests/operators/test_grammar_dynamics.py) |
| U3 rejection before mutation | [`test_u3_hard_invariant.py`](../../tests/operators/test_u3_hard_invariant.py) |
| Operator postconditions | [`test_operator_contracts.py`](../../tests/operators/test_operator_contracts.py) |
| Structural fields and U6 telemetry | [`test_tetrad_bounds.py`](../../tests/physics/test_tetrad_bounds.py) |

Passing these finite tests verifies implemented contracts over their test
surfaces. It does not establish an unrestricted asymptotic theorem.
