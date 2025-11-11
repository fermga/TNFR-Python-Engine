# Cross-Reference Index: U1-U5 Constraints

**Complete traceability map for TNFR canonical grammar constraints**

Version: 2.0  
Last Updated: 2025-11-10  
Status: CANONICAL

---

## Purpose

This document provides a **complete cross-reference index** for the unified TNFR grammar constraints (U1-U5), mapping:

- Theory → Documentation → Implementation → Tests → Examples

**Use this index to:**
- Find all resources related to a specific constraint
- Trace a constraint from physics to code
- Locate tests for validation
- Find executable examples

---

## U1: STRUCTURAL INITIATION & CLOSURE

### U1a: Initiation

**Physics Basis**: ∂EPI/∂t undefined at EPI=0

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U1](../../UNIFIED_GRAMMAR_RULES.md)
- [TNFR.pdf § 2.1 - Nodal Equation](../../TNFR.pdf)
- [AGENTS.md § Invariant #1](../../AGENTS.md)

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U1a](02-CANONICAL-CONSTRAINTS.md#u1a-initiation-generators)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_initiation()`
- `src/tnfr/operators/grammar.py::GENERATORS`

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU1aInitiation`
  - `test_epi_zero_requires_generator`
  - `test_epi_zero_non_generator_fails`
  - `test_epi_nonzero_no_generator_needed`
  - `test_all_generators_valid_for_epi_zero`
- `tests/integration/test_mutation_sequences.py::test_u1a_satisfied_with_emission`

**Examples**:
- `examples/u1-initiation-closure-examples.py::example_u1a_valid`
- `examples/u1-initiation-closure-examples.py::example_u1a_invalid`
- `examples/u1-initiation-closure-examples.py::example_u1a_context_matters`

**Operators**:
- Generators: `{emission, transition, recursivity}` → `{AL, NAV, REMESH}`

**Anti-Patterns**:
- Using Reception as initiator
- Forgetting generator when reusing sequences
- Assuming EPI exists without checking

---

### U1b: Closure

**Physics Basis**: Sequences as action potentials need endpoints

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U1](../../UNIFIED_GRAMMAR_RULES.md)
- [AGENTS.md § Invariant #4 - Operator Closure](../../AGENTS.md)

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U1b](02-CANONICAL-CONSTRAINTS.md#u1b-closure-endpoints)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_closure()`
- `src/tnfr/operators/grammar.py::CLOSURES`

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU1bClosure`
  - `test_sequence_must_have_closure`
  - `test_non_closure_end_fails`
  - `test_all_closures_valid`
  - `test_empty_sequence_fails_closure`
- `tests/integration/test_mutation_sequences.py::test_u1b_closure_satisfied`
- `tests/unit/operators/test_remesh_operator_integration.py::test_remesh_as_closure_U1b`

**Examples**:
- `examples/u1-initiation-closure-examples.py::example_u1b_valid`
- `examples/u1-initiation-closure-examples.py::example_u1b_invalid`
- `examples/u1-initiation-closure-examples.py::example_dual_role_operators`

**Operators**:
- Closures: `{silence, transition, recursivity, dissonance}` → `{SHA, NAV, REMESH, OZ}`

**Anti-Patterns**:
- Ending with Coherence (not a closure)
- Ending with data gathering operations
- Confusing closure with stabilization

---

## U2: CONVERGENCE & BOUNDEDNESS

**Physics Basis**: ∫νf·ΔNFR dt must converge (integral convergence theorem)

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U2](../../UNIFIED_GRAMMAR_RULES.md)
- [TNFR.pdf § 2.1 - Integrated Dynamics](../../TNFR.pdf)
- [AGENTS.md § Convergence & Boundedness](../../AGENTS.md)

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U2](02-CANONICAL-CONSTRAINTS.md#u2-convergence--boundedness)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_convergence()`
- `src/tnfr/operators/grammar.py::STABILIZERS`
- `src/tnfr/operators/grammar.py::DESTABILIZERS`

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU2Convergence`
  - `test_destabilizer_needs_stabilizer`
  - `test_no_destabilizers_passes`
  - `test_destabilizer_stabilizer_pairs`
  - `test_multiple_destabilizers_need_stabilizer`
  - `test_multiple_destabilizers_without_stabilizer_fail`
- `tests/integration/test_mutation_sequences.py::test_u2_satisfied_with_stabilizers`
- `tests/unit/operators/test_canonical_grammar_legacy.py::test_rc2_maps_to_u2`
- `tests/unit/operators/test_grammar_c1_c3_deprecation.py::test_validate_c2_boundedness_*`

**Examples**:
- `examples/u2-convergence-examples.py::example_u2_valid`
- `examples/u2-convergence-examples.py::example_u2_invalid`
- `examples/u2-convergence-examples.py::example_masking_antipattern`
- `examples/u2-convergence-examples.py::example_interleaving_pattern`

**Operators**:
- Destabilizers: `{dissonance, mutation, expansion}` → `{OZ, ZHIR, VAL}`
- Stabilizers: `{coherence, self_organization}` → `{IL, THOL}`

**Anti-Patterns**:
- Masking with weak stabilizers (multiple destabilizers, single stabilizer)
- Assuming order doesn't matter (stabilizer before destabilizer ineffective)
- Ignoring accumulation effects (long sequence of destabilizers)

---

## U3: RESONANT COUPLING

**Physics Basis**: Resonance physics + AGENTS.md Invariant #5

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U3](../../UNIFIED_GRAMMAR_RULES.md)
- [AGENTS.md § Invariant #5 - Phase Verification](../../AGENTS.md)

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U3](02-CANONICAL-CONSTRAINTS.md#u3-resonant-coupling)
- [03-OPERATORS-AND-GLYPHS.md § Coupling (UM)](03-OPERATORS-AND-GLYPHS.md)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_resonant_coupling()`
- `src/tnfr/operators/grammar.py::COUPLING_RESONANCE`
- Operator preconditions check phase compatibility

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU3ResonantCoupling`
  - `test_coupling_requires_phase_awareness`
  - `test_no_coupling_not_applicable`
  - `test_coupling_resonance_ops_trigger_u3`
  - `test_multiple_coupling_ops_trigger_u3`
- `tests/unit/operators/test_coupling_preconditions.py::test_um_phase_compatibility_*`
- `tests/unit/metrics/test_phase_compatibility.py::test_grammar_u3_compliance`
- `tests/unit/operators/test_canonical_grammar_legacy.py::test_rc3_maps_to_u3`

**Examples**:
- `examples/u3-resonant-coupling-examples.py::example_phase_compatibility`
- `examples/u3-resonant-coupling-examples.py::example_antipattern_no_check`
- `examples/u3-resonant-coupling-examples.py::example_antipattern_phase_drift`
- `examples/u3-resonant-coupling-examples.py::example_wave_interference`

**Operators**:
- Coupling/Resonance: `{coupling, resonance}` → `{UM, RA}`

**Phase Condition**:
- Formula: `|φᵢ - φⱼ| ≤ Δφ_max`
- Typical threshold: `π/2 radians`

**Anti-Patterns**:
- Coupling nodes without phase check
- Assuming small phase differences are always OK
- Ignoring phase drift during sequences

---

## U4: BIFURCATION DYNAMICS

### U4a: Bifurcation Triggers Need Handlers

**Physics Basis**: Contract OZ + bifurcation theory (∂²EPI/∂t² > τ)

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U4a](../../UNIFIED_GRAMMAR_RULES.md)
- [AGENTS.md § Contract OZ](../../AGENTS.md)
- [03-OPERATORS-AND-GLYPHS.md § Dissonance (OZ)](03-OPERATORS-AND-GLYPHS.md)

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U4a](02-CANONICAL-CONSTRAINTS.md#u4a-triggers-need-handlers)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_bifurcation_triggers()`
- `src/tnfr/operators/grammar.py::BIFURCATION_TRIGGERS`
- `src/tnfr/operators/grammar.py::BIFURCATION_HANDLERS`

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU4aBifurcationTriggers`
  - `test_triggers_require_handlers`
  - `test_triggers_without_handlers_fail`
  - `test_trigger_handler_pairs`
  - `test_no_triggers_not_applicable`
  - `test_multiple_triggers_need_handler`
- `tests/unit/operators/test_controlled_bifurcation.py::test_multiple_bifurcations_*`
- `tests/unit/operators/test_bifurcation.py::test_bifurcation_above_threshold`

**Examples**:
- `examples/u4-bifurcation-examples.py::example_u4a_valid`
- `examples/u4-bifurcation-examples.py::example_u4a_invalid`
- `examples/u4-bifurcation-examples.py::example_antipattern_cascade`

**Operators**:
- Triggers: `{dissonance, mutation}` → `{OZ, ZHIR}`
- Handlers: `{self_organization, coherence}` → `{THOL, IL}`

**Anti-Patterns**:
- Uncontrolled bifurcation cascades (multiple triggers without handlers)
- Wrong handler for trigger type
- Assuming handler proximity doesn't matter

---

### U4b: Transformers Need Context

**Physics Basis**: Threshold energy needed for phase transition

**Theory**:
- [UNIFIED_GRAMMAR_RULES.md § U4b](../../UNIFIED_GRAMMAR_RULES.md)
- [AGENTS.md § Contract OZ + ZHIR Requirements](../../AGENTS.md)
- [U4B_AUDIT_REPORT.md](../../U4B_AUDIT_REPORT.md) - Complete U4b analysis

**Documentation**:
- [02-CANONICAL-CONSTRAINTS.md § U4b](02-CANONICAL-CONSTRAINTS.md#u4b-transformers-need-context)
- [03-OPERATORS-AND-GLYPHS.md § Mutation (ZHIR)](03-OPERATORS-AND-GLYPHS.md)

**Implementation**:
- `src/tnfr/operators/grammar.py::GrammarValidator.validate_transformer_context()`
- `src/tnfr/operators/grammar.py::TRANSFORMERS`
- Window size: ~3 operators

**Tests**:
- `tests/unit/operators/test_unified_grammar.py::TestU4bTransformerContext`
  - `test_transformer_needs_recent_destabilizer`
  - `test_transformer_without_destabilizer_fails`
  - `test_mutation_needs_prior_coherence`
  - `test_recent_window_is_three_ops`
  - `test_destabilizer_within_window_valid`
  - `test_self_organization_needs_destabilizer`
  - `test_no_transformers_not_applicable`
- `tests/integration/test_mutation_sequences.py::test_u4b_satisfied_in_canonical_sequence`
- `tests/unit/operators/test_controlled_bifurcation.py::test_transformer_at_sequence_start_fails`
- `tests/unit/operators/test_zhir_u4b_validation.py`
- `tests/unit/operators/test_mutation_metrics_comprehensive.py::test_grammar_u4b_validation`

**Examples**:
- `examples/u4-bifurcation-examples.py::example_u4b_valid`
- `examples/u4-bifurcation-examples.py::example_u4b_invalid`
- `examples/u4-bifurcation-examples.py::example_zhir_requirements`
- `examples/u4-bifurcation-examples.py::example_antipattern_window`

**Operators**:
- Transformers: `{mutation, self_organization}` → `{ZHIR, THOL}`
- Context: Recent destabilizer from `{dissonance, mutation, expansion}`
- ZHIR specific: Prior coherence (IL) for stable base

**Anti-Patterns**:
- Transformer without sufficient energy (too far from destabilizer)
- ZHIR without stable base (missing prior coherence)
- Confusing context window (which destabilizer provides context?)

---

## JSON Schema

**Schema**: Unified constraints schema (updated to include U5)

Complete machine-readable specification with:
- Operator sets
- Physics basis
- Validation functions
- Test references
- Example references
- Anti-patterns

---

## Quick Navigation

**By Document Type**:

| Type | Location |
|------|----------|
| Theory | [UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md) |
| Physics | [TNFR.pdf](../../TNFR.pdf) |
| Documentation | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) |
| Implementation | `src/tnfr/operators/grammar.py` |
| Tests | `tests/unit/operators/test_unified_grammar.py` |
| Examples | `examples/u*-examples.py` |
| Schema | Unified constraints schema (includes U5) |
| Invariants | [AGENTS.md](../../AGENTS.md) |

**By Constraint**:

| Constraint | Theory | Docs | Code | Tests | Examples |
|------------|--------|------|------|-------|----------|
| U1a | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u1a-initiation-generators) | `validate_initiation()` | TestU1aInitiation | u1-examples.py |
| U1b | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u1b-closure-endpoints) | `validate_closure()` | TestU1bClosure | u1-examples.py |
| U2 | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u2-convergence--boundedness) | `validate_convergence()` | TestU2Convergence | u2-examples.py |
| U3 | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u3-resonant-coupling) | `validate_resonant_coupling()` | TestU3ResonantCoupling | u3-examples.py |
| U4a | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u4a-triggers-need-handlers) | `validate_bifurcation_triggers()` | TestU4aBifurcationTriggers | u4-examples.py |
| U4b | [Link](../../UNIFIED_GRAMMAR_RULES.md) | [Link](02-CANONICAL-CONSTRAINTS.md#u4b-transformers-need-context) | `validate_transformer_context()` | TestU4bTransformerContext | u4-examples.py |

---

## Verification Checklist

**To verify a constraint implementation**:

- [ ] Physics derivation in UNIFIED_GRAMMAR_RULES.md
- [ ] Documentation section in 02-CANONICAL-CONSTRAINTS.md
- [ ] Implementation in grammar.py
- [ ] Operator sets defined
- [ ] Test class exists with multiple test cases
- [ ] Executable examples with valid/invalid patterns
- [ ] Anti-patterns documented
- [ ] JSON schema entry complete
- [ ] Cross-references working

---

## Update Procedure

**When modifying a constraint**:

1. Update **theory** in UNIFIED_GRAMMAR_RULES.md
2. Update **documentation** in 02-CANONICAL-CONSTRAINTS.md
3. Update **implementation** in grammar.py
4. Update **tests** in test_unified_grammar.py
5. Update **examples** in u*-examples.py
6. Update the unified constraints schema to include U5
7. Update **this index** if cross-references change

**Maintain bidirectional traceability at all times.**

---

<div align="center">

**Complete traceability: Theory → Docs → Code → Tests → Examples**

---

*Single source of truth for TNFR canonical grammar*

</div>
