# Grammar Code-Documentation Cross-Reference Index

**Purpose**: Bidirectional mapping between TNFR grammar documentation and implementation.

**Last Updated**: 2025-11-10

---

## Documentation → Code

### Constraint Rules

| Rule | Documentation | Code Implementation |
|------|---------------|---------------------|
| **U1a: Initiation** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_initiation()` in `src/tnfr/operators/grammar.py:476` |
| **U1b: Closure** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_closure()` in `src/tnfr/operators/grammar.py:520` |
| **U2: Convergence** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_convergence()` in `src/tnfr/operators/grammar.py:558` |
| **U3: Resonant Coupling** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_resonant_coupling()` in `src/tnfr/operators/grammar.py:616` |
| **U4a: Bifurcation Triggers** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_bifurcation_triggers()` in `src/tnfr/operators/grammar.py:672` |
| **U4b: Transformer Context** | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) | `GrammarValidator.validate_transformer_context()` in `src/tnfr/operators/grammar.py:727` |
| **U2-REMESH** | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) | `GrammarValidator.validate_remesh_amplification()` in `src/tnfr/operators/grammar.py:807` |

### Operator Sets

| Set | Documentation | Code Definition |
|-----|---------------|-----------------|
| **GENERATORS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:210` → `{"emission", "transition", "recursivity"}` |
| **CLOSURES** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:213` → `{"silence", "transition", "recursivity", "dissonance"}` |
| **STABILIZERS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:216` → `{"coherence", "self_organization"}` |
| **DESTABILIZERS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:219` → `{"dissonance", "mutation", "expansion"}` |
| **COUPLING_RESONANCE** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:222` → `{"coupling", "resonance"}` |
| **BIFURCATION_TRIGGERS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:225` → `{"dissonance", "mutation"}` |
| **BIFURCATION_HANDLERS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:228` → `{"self_organization", "coherence"}` |
| **TRANSFORMERS** | [03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md) | `grammar.py:231` → `{"mutation", "self_organization"}` |

### Functions

| Function | Documentation | Code Location |
|----------|---------------|---------------|
| `validate_grammar()` | [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) | `grammar.py:966` |
| `GrammarValidator.validate()` | [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) | `grammar.py:897` |
| `glyph_function_name()` | [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) | `grammar.py:90` |
| `function_name_to_glyph()` | [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) | `grammar.py:141` |

---

## Code → Documentation

### From `grammar.py`

| Code Element | Line | Referenced In |
|--------------|------|---------------|
| Module docstring | 1-36 | [README.md](README.md), [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) |
| `GENERATORS` | 210 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U1a |
| `CLOSURES` | 213 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U1b |
| `STABILIZERS` | 216 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U2 |
| `DESTABILIZERS` | 219 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U2 |
| `validate_initiation()` | 476 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U1a, [examples/u1-initiation-closure-examples.py](examples/u1-initiation-closure-examples.py) |
| `validate_closure()` | 520 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U1b, [examples/u1-initiation-closure-examples.py](examples/u1-initiation-closure-examples.py) |
| `validate_convergence()` | 558 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U2, [examples/u2-convergence-examples.py](examples/u2-convergence-examples.py) |
| `validate_resonant_coupling()` | 616 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U3, [examples/u3-resonant-coupling-examples.py](examples/u3-resonant-coupling-examples.py) |
| `validate_bifurcation_triggers()` | 672 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U4a, [examples/u4-bifurcation-examples.py](examples/u4-bifurcation-examples.py) |
| `validate_transformer_context()` | 727 | [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) § U4b, [examples/u4-bifurcation-examples.py](examples/u4-bifurcation-examples.py) |
| `validate_remesh_amplification()` | 807 | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U2-REMESH |

### From Examples

| Example File | Uses Code |
|--------------|-----------|
| [u1-initiation-closure-examples.py](examples/u1-initiation-closure-examples.py) | `validate_grammar()`, `GrammarValidator.validate()`, `GENERATORS`, `CLOSURES` |
| [u2-convergence-examples.py](examples/u2-convergence-examples.py) | `validate_grammar()`, `GrammarValidator.validate()`, `STABILIZERS`, `DESTABILIZERS` |
| [u3-resonant-coupling-examples.py](examples/u3-resonant-coupling-examples.py) | `validate_grammar()`, `GrammarValidator.validate()`, `COUPLING_RESONANCE` |
| [u4-bifurcation-examples.py](examples/u4-bifurcation-examples.py) | `validate_grammar()`, `GrammarValidator.validate()`, `BIFURCATION_TRIGGERS`, `BIFURCATION_HANDLERS`, `TRANSFORMERS` |
| [01-basic-bootstrap.py](examples/01-basic-bootstrap.py) | `validate_grammar()`, All operator sets |
| [02-intermediate-exploration.py](examples/02-intermediate-exploration.py) | `GrammarValidator.validate()`, Complex sequences |
| [03-advanced-bifurcation.py](examples/03-advanced-bifurcation.py) | `validate_bifurcation_triggers()`, `validate_transformer_context()` |

---

## Test Coverage

| Code Element | Test File | Test Function |
|--------------|-----------|---------------|
| `validate_initiation()` | `tests/unit/operators/test_unified_grammar.py` | `TestU1aInitiation::test_*` |
| `validate_closure()` | `tests/unit/operators/test_unified_grammar.py` | `TestU1bClosure::test_*` |
| `validate_convergence()` | `tests/unit/operators/test_unified_grammar.py` | `TestU2Convergence::test_*` |
| `validate_resonant_coupling()` | `tests/unit/operators/test_unified_grammar.py` | `TestU3ResonantCoupling::test_*` |
| `validate_bifurcation_triggers()` | `tests/unit/operators/test_unified_grammar.py` | `TestU4aBifurcationTriggers::test_*` |
| `validate_transformer_context()` | `tests/unit/operators/test_unified_grammar.py` | `TestU4bTransformerContext::test_*` |
| Operator sets | `tests/unit/operators/test_unified_grammar.py` | `TestOperatorSets::test_*` |

---

## Schema Mapping

| Schema Element | Code Element |
|----------------|--------------|
| `canonical-operators.json::operators[].name` | Operator function names in `definitions.py` |
| `canonical-operators.json::operators[].glyph` | `GLYPH_TO_FUNCTION` mapping in `grammar.py:70` |
| `canonical-operators.json::operators[].classification.generator` | Membership in `GENERATORS` set |
| `canonical-operators.json::operators[].classification.closure` | Membership in `CLOSURES` set |
| `canonical-operators.json::operators[].classification.stabilizer` | Membership in `STABILIZERS` set |
| (etc for all classifications) | (corresponding sets in grammar.py) |

---

## Physics Basis Traceability

| Grammar Rule | Physics Basis | Code Implementation | Documentation |
|--------------|---------------|---------------------|---------------|
| U1a | ∂EPI/∂t undefined at EPI=0 | `validate_initiation()` checks `epi_initial == 0` | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U1 |
| U1b | Sequences need coherent endpoints | `validate_closure()` checks last operator | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U1 |
| U2 | ∫νf·ΔNFR dt must converge | `validate_convergence()` checks stabilizers | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U2 |
| U3 | \|φᵢ - φⱼ\| ≤ Δφ_max | Phase checked in operator preconditions | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U3 |
| U4a | ∂²EPI/∂t² > τ requires handlers | `validate_bifurcation_triggers()` | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U4a |
| U4b | Transformers need threshold energy | `validate_transformer_context()` checks window | [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) § U4b |

---

## Maintenance Notes

**When updating code**:
1. Run `python tools/sync_documentation.py --all` to verify sync
2. Update this cross-reference if new functions added
3. Update schema if operator sets change
4. Re-run examples to verify they still work

**When updating documentation**:
1. Verify code references are still accurate
2. Check line numbers in cross-references
3. Test that examples still execute correctly

**Sync Tool**: Use `python tools/sync_documentation.py` for automated validation:
- `--audit`: Audit grammar.py only
- `--validate`: Test all examples
- `--all`: Full synchronization check (default)

---

**Last Sync**: 2025-11-10 via `tools/sync_documentation.py`
**Status**: ✅ All checks passing
