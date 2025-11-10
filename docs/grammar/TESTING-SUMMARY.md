# Grammar Testing Documentation Summary

This document provides a quick overview of the grammar testing strategy and resources.

## 📁 Documentation Structure

```
docs/grammar/
├── 06-VALIDATION-AND-TESTING.md    # Complete testing strategy (1402 lines)
├── examples/                        # Executable test examples
│   ├── u1-initiation-closure-examples.py
│   ├── u2-convergence-examples.py
│   ├── u3-resonant-coupling-examples.py
│   └── u4-bifurcation-examples.py
└── ...

tests/
├── unit/operators/
│   └── test_unified_grammar.py     # 68 canonical tests (100% coverage)
└── ...

scripts/
└── validate_grammar.sh             # Complete validation suite
```

## 📊 Test Coverage Status

| Component | Coverage | Tests | Status |
|-----------|----------|-------|--------|
| `unified_grammar.py` | 100% | 68 | ✓ PASSING |
| U1 Constraints | 100% | 6 | ✓ PASSING |
| U2 Constraints | 100% | 5 | ✓ PASSING |
| U3 Constraints | 100% | 6 | ✓ PASSING |
| U4 Constraints | 100% | 8 | ✓ PASSING |
| All 13 Operators | 100% | - | ✓ COVERED |

**Last Updated:** 2025-11-10

## 🎯 Test Categories

### 1. Canonical Test Cases (20+)

Documented in `06-VALIDATION-AND-TESTING.md` with test index:

#### U1: Structural Initiation & Closure (6 tests)
- Generator requirements (AL, NAV, REMESH)
- Closure requirements (SHA, NAV, REMESH, OZ)
- EPI=0 vs EPI>0 context
- Error message quality

#### U2: Convergence & Boundedness (5 tests)
- Destabilizer-stabilizer balance
- Unbalanced sequences detection
- Window calculation algorithm
- Integral convergence guarantee

#### U3: Resonant Coupling (6 tests)
- Phase compatibility checks
- Antiphase detection
- Custom tolerance bounds
- Resonance preconditions

#### U4: Bifurcation Dynamics (8 tests)
- Trigger-handler pairs (U4a)
- Transformer-destabilizer context (U4b)
- ZHIR prior coherence requirement
- Bifurcation safety verification

### 2. Pattern Tests (7+ canonical patterns)

Each pattern includes:
- ✅ Valid variant test
- ❌ Invalid variant tests (2-3 per pattern)
- 🔄 Edge case tests

**Documented Patterns:**
1. Bootstrap (Minimal)
2. Basic Activation
3. Controlled Exploration
4. Bifurcation with Handling
5. Mutation with Context
6. Propagation
7. Multi-scale Organization

### 3. Anti-Pattern Tests (7+ patterns)

Each anti-pattern includes:
- 🚫 Detection test
- ✏️ Fix test
- 💬 Error message quality test

**Documented Anti-Patterns:**
1. No Generator from Vacuum (U1a)
2. No Closure (U1b)
3. Destabilizer Without Stabilizer (U2)
4. Mutation Without Context (U4b)
5. Mutation Without Prior IL (U4b)
6. Coupling Without Phase Check (U3)
7. Bifurcation Trigger Without Handler (U4a)

## 🚀 Quick Start

### Run All Tests

```bash
# Complete validation suite
./scripts/validate_grammar.sh

# Or manually run unit tests
pytest tests/unit/operators/test_unified_grammar.py -v
```

### Run Specific Constraint Tests

```bash
# U1 tests
pytest tests/unit/operators/test_unified_grammar.py::TestU1aInitiation -v
pytest tests/unit/operators/test_unified_grammar.py::TestU1bClosure -v

# U2 tests
pytest tests/unit/operators/test_unified_grammar.py::TestU2Convergence -v

# U3 tests
pytest tests/unit/operators/test_unified_grammar.py::TestU3ResonantCoupling -v

# U4 tests
pytest tests/unit/operators/test_unified_grammar.py::TestU4aBifurcationTriggers -v
pytest tests/unit/operators/test_unified_grammar.py::TestU4bTransformerContext -v
```

### Generate Coverage Reports

```bash
# Terminal report
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=term-missing

# HTML report
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=html:htmlcov/grammar

# With branch coverage
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-branch \
    --cov-report=term-missing
```

## 📖 Documentation Guide

### For Test Writers

1. **Start here:** `06-VALIDATION-AND-TESTING.md` § Test Philosophy
2. **Learn patterns:** `06-VALIDATION-AND-TESTING.md` § Canonical Pattern Tests
3. **See examples:** `docs/grammar/examples/u*-examples.py`
4. **Use templates:** `06-VALIDATION-AND-TESTING.md` § Test Templates

### For Test Reviewers

1. **Coverage requirements:** `06-VALIDATION-AND-TESTING.md` § Coverage Requirements
2. **Test index:** `06-VALIDATION-AND-TESTING.md` § Test Case Index
3. **Validation suite:** `scripts/validate_grammar.sh`

### For CI/CD Integration

```bash
# In CI pipeline
./scripts/validate_grammar.sh

# Or with explicit coverage requirement
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-fail-under=95 \
    --cov-report=term
```

## 🛠️ Test Utilities

### Available Helpers

Located in `06-VALIDATION-AND-TESTING.md` § Test Utilities:

- `create_test_graph()` - Standard test graphs
- `create_test_graph_custom_phases()` - Phase-specific graphs
- `assert_valid_sequence()` - Sequence validation helper
- `assert_invalid_sequence()` - Negative test helper
- `assert_constraint_violation()` - Constraint-specific assertions
- `verify_coherence_increase()` - Metric verification
- `verify_dnfr_reduction()` - Stabilizer verification
- `verify_phase_synchronization()` - Coupling verification

## 📋 Coverage Checklist

From `06-VALIDATION-AND-TESTING.md` § Coverage Checklist:

### Operators (13/13 ✓)
- [x] Emission (AL)
- [x] Reception (EN)
- [x] Coherence (IL)
- [x] Dissonance (OZ)
- [x] Coupling (UM)
- [x] Resonance (RA)
- [x] Silence (SHA)
- [x] Expansion (VAL)
- [x] Contraction (NUL)
- [x] Self-organization (THOL)
- [x] Mutation (ZHIR)
- [x] Transition (NAV)
- [x] Recursivity (REMESH)

### Constraints (14/14 ✓)
- [x] U1a: Valid generators
- [x] U1a: Invalid non-generators
- [x] U1b: Valid closures
- [x] U1b: Invalid non-closures
- [x] U2: Destabilizer + stabilizer (valid)
- [x] U2: Destabilizer without stabilizer (invalid)
- [x] U3: Compatible phases (valid)
- [x] U3: Incompatible phases (invalid)
- [x] U4a: Trigger + handler (valid)
- [x] U4a: Trigger without handler (invalid)
- [x] U4b: Transformer with context (valid)
- [x] U4b: Transformer without context (invalid)
- [x] U4b: ZHIR with prior IL (valid)
- [x] U4b: ZHIR without prior IL (invalid)

### Invariants (7/7 ✓)
- [x] Coherence monotonicity
- [x] Integral convergence
- [x] Bifurcation handling
- [x] Propagation effects
- [x] Latency preservation
- [x] Fractality (nested EPIs)
- [x] Reproducibility (seeds)

## 🎯 Acceptance Criteria Status

From issue #2897:

- [x] **Estrategia de testing documentada** - Complete in `06-VALIDATION-AND-TESTING.md`
- [x] **20+ casos de prueba canonicos** - 25+ test cases documented with templates
- [x] **Tests para U1, U2, U3, U4** - All constraints covered (25 test cases)
- [x] **Canonical pattern tests** - 7 patterns with valid/invalid variants
- [x] **Anti-pattern tests** - 7 anti-patterns with detection/fix tests
- [x] **Executable suite** - `scripts/validate_grammar.sh` created
- [x] **Coverage >= 95%** - 100% achieved (8/8 statements)
- [x] **Test utilities documentation** - Complete section with helpers

## 📚 Additional Resources

- **UNIFIED_GRAMMAR_RULES.md** - Complete grammar derivations
- **04-VALID-SEQUENCES.md** - Pattern library
- **02-CANONICAL-CONSTRAINTS.md** - Constraint specifications
- **AGENTS.md** - Canonical invariants

## 🔄 Maintenance

### Updating Tests

When adding new operators or constraints:

1. Add test cases to `test_unified_grammar.py`
2. Document in `06-VALIDATION-AND-TESTING.md` § Test Case Index
3. Add examples to `docs/grammar/examples/`
4. Update this summary
5. Run validation suite: `./scripts/validate_grammar.sh`

### Monitoring Coverage

```bash
# Quick coverage check
pytest tests/unit/operators/test_unified_grammar.py --cov --cov-fail-under=95

# Detailed report
./scripts/validate_grammar.sh
```

---

**Status:** ✅ COMPLETE  
**Coverage:** 100%  
**Tests:** 68 passing  
**Last Updated:** 2025-11-10
