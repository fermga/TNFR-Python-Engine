# Validation and Testing Strategy

**Comprehensive testing approach for TNFR grammar**

[🏠 Home](README.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [💻 Implementation](05-TECHNICAL-IMPLEMENTATION.md) • [📚 Evolution](07-MIGRATION-AND-EVOLUTION.md)

---

## Purpose

This document outlines the **testing strategy** for TNFR grammar validation, including test categories, coverage requirements, and examples.

**Prerequisites:** [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md), [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)

**Audience:** Developers writing tests, QA engineers

**Reading time:** 30-45 minutes

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Case Index](#test-case-index)
3. [Unit Tests: Operator Behavior](#unit-tests-operator-behavior)
4. [Integration Tests: Sequence Validation](#integration-tests-sequence-validation)
   - [Test U1a: Initiation](#test-u1a-initiation)
   - [Test U1b: Closure](#test-u1b-closure)
   - [Test U2: Convergence](#test-u2-convergence)
   - [Test U3: Resonant Coupling](#test-u3-resonant-coupling)
   - [Test U4a: Triggers Need Handlers](#test-u4a-triggers-need-handlers)
   - [Test U4b: Transformers Need Context](#test-u4b-transformers-need-context)
5. [Property Tests: Invariants](#property-tests-invariants)
6. [Multi-Scale Tests](#multi-scale-tests)
7. [Reproducibility Tests](#reproducibility-tests)
8. [Performance Tests](#performance-tests)
9. [Coverage Requirements](#coverage-requirements)
10. [Canonical Pattern Tests](#canonical-pattern-tests)
11. [Anti-Pattern Tests](#anti-pattern-tests)
12. [Test Utilities](#test-utilities)
13. [Validation Suite](#validation-suite)
14. [Coverage Tracking](#coverage-tracking)

---

## Testing Philosophy

### Core Principles

1. **Physics-Based:** Tests verify physical contracts, not arbitrary rules
2. **Comprehensive:** Cover all constraints (U1-U4) and operators (13)
3. **Executable:** All examples in documentation must be testable
4. **Reproducible:** Same seed → same results
5. **Traceable:** Each test maps to specific constraint or invariant

### Test Categories

```
Unit Tests         → Individual operator behavior
Integration Tests  → Sequence validation (U1-U4)
Property Tests     → Invariants (coherence, convergence, etc.)
Performance Tests  → Scalability and efficiency
Pattern Tests      → Canonical patterns and anti-patterns
Regression Tests   → Prevent breakage of existing functionality
```

---

## Test Case Index

### Canonical Test Cases for U1 (Structural Initiation & Closure)

| Test Name | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `test_u1a_valid_generators` | Valid | U1a | All generators (AL, NAV, REMESH) work when EPI=0 |
| `test_u1a_invalid_no_generator` | Invalid | U1a | Non-generator at start fails when EPI=0 |
| `test_u1a_not_required_when_epi_nonzero` | Context | U1a | Generator not required when EPI > 0 |
| `test_u1b_valid_closures` | Valid | U1b | All closures (SHA, NAV, REMESH, OZ) work |
| `test_u1b_invalid_no_closure` | Invalid | U1b | Sequence without closure fails |
| `test_u1_error_messages` | Quality | U1a/U1b | Error messages are clear and actionable |

### Canonical Test Cases for U2 (Convergence & Boundedness)

| Test Name | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `test_u2_destabilizer_with_stabilizer` | Valid | U2 | Destabilizers balanced by stabilizers pass |
| `test_u2_destabilizer_without_stabilizer` | Invalid | U2 | Unbalanced destabilizers fail |
| `test_u2_detects_unbalanced_dissonance` | Detection | U2 | OZ without IL/THOL detected |
| `test_u2_convergence_guarantee` | Property | U2 | Integral convergence verified |
| `test_u2_window_calculation` | Algorithm | U2 | Stabilizer window search works correctly |

### Canonical Test Cases for U3 (Resonant Coupling)

| Test Name | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `test_u3_compatible_phases` | Valid | U3 | Compatible phases (Δφ ≤ threshold) allow coupling |
| `test_u3_incompatible_phases` | Invalid | U3 | Antiphase nodes cannot couple |
| `test_u3_phase_compatibility_check` | Required | U3 | Phase verification mandatory for UM/RA |
| `test_u3_tolerance_bounds` | Edge | U3 | Threshold boundary conditions tested |
| `test_u3_custom_threshold` | Parameter | U3 | Custom Δφ_max values work |
| `test_u3_resonance_preconditions` | Precond | U3 | Resonance requires phase check |

### Canonical Test Cases for U4 (Bifurcation Dynamics)

| Test Name | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `test_u4a_trigger_with_handler` | Valid | U4a | Triggers (OZ, ZHIR) with handlers pass |
| `test_u4a_trigger_without_handler` | Invalid | U4a | Triggers without handlers fail |
| `test_u4a_triggers_require_handlers` | Detection | U4a | All bifurcation triggers detected |
| `test_u4b_transformer_with_context` | Valid | U4b | THOL/ZHIR with destabilizer context pass |
| `test_u4b_transformer_without_destabilizer` | Invalid | U4b | Transformers without context fail |
| `test_u4b_transformers_require_destabilizers` | Detection | U4b | Window-based destabilizer search works |
| `test_u4b_zhir_without_prior_coherence` | Invalid | U4b | ZHIR without prior IL fails |
| `test_u4_bifurcation_safety` | Property | U4a/U4b | Bifurcations are controlled |

### Canonical Pattern Tests (20+ total)

| Pattern Name | Valid Test | Invalid Variants | Edge Cases |
|--------------|------------|------------------|------------|
| Bootstrap | `test_bootstrap_valid` | `test_bootstrap_invalid_no_generator`, `test_bootstrap_invalid_no_closure` | Generator variants |
| Basic Activation | `test_activation_valid` | `test_activation_invalid_*` | - |
| Controlled Exploration | `test_exploration_valid` | `test_exploration_invalid_no_stabilizer_after_destabilizer` | Different destabilizers |
| Bifurcation with Handling | `test_bifurcation_valid` | `test_bifurcation_invalid_no_handler` | Different triggers |
| Mutation with Context | `test_mutation_valid` | `test_mutation_invalid_no_prior_coherence`, `test_mutation_invalid_no_destabilizer` | Window boundaries |
| Propagation | `test_propagation_valid` | `test_propagation_invalid_*` | Phase compatibility |
| Multi-scale Organization | `test_multiscale_valid` | `test_multiscale_invalid_*` | Nesting depth |

### Anti-Pattern Tests (7+ documented patterns)

| Anti-Pattern | Detection Test | Fix Test | Error Quality Test |
|--------------|----------------|----------|-------------------|
| No Generator from Vacuum | `test_no_generator_detected` | `test_no_generator_fix` | `test_no_generator_error_message` |
| No Closure | `test_no_closure_detected` | `test_no_closure_fix` | - |
| Destabilizer Without Stabilizer | `test_unbalanced_dissonance_detected` | `test_unbalanced_fix_with_coherence` | - |
| Mutation Without Context | `test_mutation_no_context_detected` | `test_mutation_context_fix` | - |
| Mutation Without Prior IL | `test_mutation_no_prior_il_detected` | `test_mutation_prior_il_fix` | - |
| Coupling Without Phase Check | `test_antiphase_coupling_detected` | `test_phase_check_fix` | - |
| Bifurcation Without Handler | `test_dissonance_without_handler_detected` | `test_trigger_with_handler_fix` | - |

---

## Unit Tests: Operator Behavior

### Test Template

```python
def test_operator_name():
    """Test specific operator behavior."""
    # Setup
    G = create_test_graph()
    
    # Apply operator
    Operator()(G, node_id)
    
    # Verify postconditions
    assert check_postcondition()
    
    # Verify invariants
    assert check_invariants()
```

### Example: Test Emission

```python
import pytest
import networkx as nx
from tnfr.operators.definitions import Emission

def test_emission_creates_structure():
    """Emission creates EPI from vacuum."""
    # Setup: Node with EPI=0
    G = nx.Graph()
    G.add_node(0, EPI=0.0, vf=0.1, theta=0.0, dnfr=0.0)
    
    # Apply Emission
    Emission()(G, 0)
    
    # Verify: EPI > 0
    assert G.nodes[0]['EPI'] > 0, "Emission must create structure"
    
    # Verify: vf increased or maintained
    assert G.nodes[0]['vf'] >= 0.1, "Emission must maintain/increase vf"

def test_emission_is_generator():
    """Emission is classified as generator."""
    from tnfr.operators.grammar import GENERATORS
    assert "emission" in GENERATORS
```

### Example: Test Coherence

```python
from tnfr.operators.definitions import Emission, Coherence

def test_coherence_reduces_dnfr():
    """Coherence reduces reorganization gradient."""
    G = nx.Graph()
    G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=1.0)
    
    dnfr_before = G.nodes[0]['dnfr']
    
    # Apply Coherence
    Coherence()(G, 0)
    
    dnfr_after = G.nodes[0]['dnfr']
    
    # Verify: |ΔNFR| reduced
    assert abs(dnfr_after) < abs(dnfr_before), "Coherence must reduce |ΔNFR|"

def test_coherence_is_stabilizer():
    """Coherence is classified as stabilizer."""
    from tnfr.operators.grammar import STABILIZERS
    assert "coherence" in STABILIZERS
```

---

## Integration Tests: Sequence Validation

### Test U1a: Initiation

```python
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import (
    Emission, Transition, Recursivity,
    Coherence, Silence
)

def test_u1a_valid_generators():
    """U1a: Valid generators when EPI=0."""
    # Test each generator
    generators = [Emission(), Transition(), Recursivity()]
    
    for gen in generators:
        sequence = [gen, Coherence(), Silence()]
        # Should not raise
        assert validate_grammar(sequence, epi_initial=0.0) is True

def test_u1a_invalid_no_generator():
    """U1a: Must start with generator when EPI=0."""
    # No generator
    sequence = [Coherence(), Silence()]
    
    with pytest.raises(ValueError, match="U1a violation"):
        validate_grammar(sequence, epi_initial=0.0)

def test_u1a_not_required_when_epi_nonzero():
    """U1a: Generator not required when EPI > 0."""
    # Can start with non-generator if EPI > 0
    sequence = [Coherence(), Silence()]
    
    # Should not raise
    assert validate_grammar(sequence, epi_initial=1.0) is True
```

### Test U1b: Closure

```python
from tnfr.operators.definitions import (
    Emission, Coherence,
    Silence, Transition, Recursivity, Dissonance
)

def test_u1b_valid_closures():
    """U1b: Valid closure operators."""
    closures = [Silence(), Transition(), Recursivity(), Dissonance()]
    
    for closure in closures:
        sequence = [Emission(), Coherence(), closure]
        assert validate_grammar(sequence, epi_initial=0.0) is True

def test_u1b_invalid_no_closure():
    """U1b: Must end with closure."""
    sequence = [Emission(), Coherence()]  # Coherence is not closure
    
    with pytest.raises(ValueError, match="U1b violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

### Test U2: Convergence

```python
from tnfr.operators.definitions import (
    Emission, Dissonance, Mutation, Expansion,
    Coherence, SelfOrganization, Silence
)

def test_u2_destabilizer_with_stabilizer():
    """U2: Destabilizer balanced by stabilizer."""
    destabilizers = [Dissonance(), Mutation(), Expansion()]
    stabilizers = [Coherence(), SelfOrganization()]
    
    for dest in destabilizers:
        for stab in stabilizers:
            # Create valid sequence for mutation
            if dest.__class__.__name__.lower() == "mutation":
                sequence = [Emission(), Coherence(), Dissonance(), dest, stab, Silence()]
            else:
                sequence = [Emission(), dest, stab, Silence()]
            
            # Should not raise
            try:
                validate_grammar(sequence, epi_initial=0.0)
            except ValueError as e:
                # If fails, should be different constraint, not U2
                assert "U2 violation" not in str(e)

def test_u2_destabilizer_without_stabilizer():
    """U2: Destabilizer without stabilizer fails."""
    sequence = [Emission(), Dissonance(), Silence()]
    
    with pytest.raises(ValueError, match="U2 violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

### Test U3: Resonant Coupling

```python
import numpy as np
from tnfr.operators.grammar import validate_resonant_coupling

def test_u3_compatible_phases():
    """U3: Compatible phases allow coupling."""
    G = nx.Graph()
    G.add_node(0, theta=0.0, EPI=0.5, vf=1.0, dnfr=0.0)
    G.add_node(1, theta=0.3, EPI=0.6, vf=1.0, dnfr=0.0)  # Δφ = 0.3 < π/2
    
    # Should not raise
    validate_resonant_coupling(G, 0, 1)

def test_u3_incompatible_phases():
    """U3: Antiphase nodes cannot couple."""
    G = nx.Graph()
    G.add_node(0, theta=0.0, EPI=0.5, vf=1.0, dnfr=0.0)
    G.add_node(1, theta=np.pi, EPI=0.6, vf=1.0, dnfr=0.0)  # Antiphase!
    
    with pytest.raises(ValueError, match="U3 violation"):
        validate_resonant_coupling(G, 0, 1)

def test_u3_custom_threshold():
    """U3: Custom phase threshold."""
    G = nx.Graph()
    G.add_node(0, theta=0.0, EPI=0.5, vf=1.0, dnfr=0.0)
    G.add_node(1, theta=1.0, EPI=0.6, vf=1.0, dnfr=0.0)
    
    # Fails with default π/2
    with pytest.raises(ValueError):
        validate_resonant_coupling(G, 0, 1, delta_phi_max=np.pi/2)
    
    # Passes with larger threshold
    validate_resonant_coupling(G, 0, 1, delta_phi_max=np.pi)
```

### Test U4a: Triggers Need Handlers

```python
from tnfr.operators.definitions import (
    Emission, Dissonance, Mutation,
    Coherence, SelfOrganization, Silence
)

def test_u4a_trigger_with_handler():
    """U4a: Bifurcation trigger with handler."""
    triggers = [Dissonance, Mutation]
    handlers = [Coherence, SelfOrganization]
    
    for Trigger in triggers:
        for Handler in handlers:
            # Build valid sequence
            if Trigger == Mutation:
                sequence = [Emission(), Coherence(), Dissonance(), 
                           Trigger(), Handler(), Silence()]
            else:
                sequence = [Emission(), Trigger(), Handler(), Silence()]
            
            # Should not raise
            assert validate_grammar(sequence, epi_initial=0.0) is True

def test_u4a_trigger_without_handler():
    """U4a: Trigger without handler fails."""
    # Dissonance without handler
    sequence = [Emission(), Dissonance(), Silence()]
    
    with pytest.raises(ValueError, match="U4a violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

### Test U4b: Transformers Need Context

```python
from tnfr.operators.definitions import (
    Emission, Coherence, Dissonance, Mutation, 
    SelfOrganization, Silence
)

def test_u4b_transformer_with_context():
    """U4b: Transformer with recent destabilizer."""
    # THOL with recent destabilizer
    sequence = [Emission(), Dissonance(), SelfOrganization(), 
                Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True
    
    # ZHIR with proper context
    sequence = [Emission(), Coherence(), Dissonance(), Mutation(), 
                Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0) is True

def test_u4b_transformer_without_destabilizer():
    """U4b: Transformer without destabilizer fails."""
    # THOL without recent destabilizer
    sequence = [Emission(), Coherence(), SelfOrganization(), Silence()]
    
    with pytest.raises(ValueError, match="U4b violation"):
        validate_grammar(sequence, epi_initial=0.0)

def test_u4b_zhir_without_prior_coherence():
    """U4b: ZHIR needs prior coherence."""
    # Mutation without prior IL
    sequence = [Emission(), Dissonance(), Mutation(), 
                Coherence(), Silence()]
    
    with pytest.raises(ValueError, match="U4b violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

---

## Property Tests: Invariants

### Monotonicity Tests

```python
def test_coherence_monotonicity():
    """Coherence must not decrease C(t)."""
    from tnfr.metrics import compute_coherence
    
    G = create_test_network()
    
    C_before = compute_coherence(G)
    
    # Apply Coherence to all nodes
    for node in G.nodes():
        Coherence()(G, node)
    
    C_after = compute_coherence(G)
    
    # Coherence must not decrease C(t)
    assert C_after >= C_before, "Coherence must not reduce C(t)"
```

### Convergence Tests

```python
def test_integral_convergence():
    """Verify integral convergence with stabilizers."""
    # Create sequence with destabilizer + stabilizer
    sequence = [Emission(), Dissonance(), Coherence(), Silence()]
    
    G = nx.Graph()
    G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
    
    # Track ΔNFR over time
    dnfr_history = []
    
    for op in sequence:
        op(G, 0)
        dnfr_history.append(G.nodes[0]['dnfr'])
    
    # Verify: Integral is bounded (sum doesn't explode)
    integral_approx = sum(abs(d) for d in dnfr_history)
    assert integral_approx < float('inf'), "Integral must be bounded"
    assert integral_approx < 1000, "Integral should be reasonably small"
```

### Bifurcation Tests

```python
def test_dissonance_bifurcation():
    """Dissonance may trigger bifurcation."""
    G = nx.Graph()
    G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
    
    # Apply Dissonance
    Dissonance()(G, 0)
    
    # Check if ΔNFR increased (potential bifurcation)
    assert G.nodes[0]['dnfr'] > 0, "Dissonance must increase |ΔNFR|"
```

### Propagation Tests

```python
def test_resonance_propagation():
    """Resonance increases effective connectivity."""
    G = nx.Graph()
    G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
    G.add_node(1, EPI=0.6, vf=1.0, theta=0.1, dnfr=0.0)
    
    # Couple nodes
    Coupling()(G, 0, 1)
    
    # Measure phase sync before resonance
    phase_diff_before = abs(G.nodes[0]['theta'] - G.nodes[1]['theta'])
    
    # Apply Resonance
    Resonance()(G, 0, 1)
    
    # Measure phase sync after resonance
    phase_diff_after = abs(G.nodes[0]['theta'] - G.nodes[1]['theta'])
    
    # Resonance should improve synchronization
    assert phase_diff_after <= phase_diff_before, "Resonance improves sync"
```

### Latency Tests

```python
def test_silence_latency():
    """Silence keeps EPI invariant."""
    G = nx.Graph()
    G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
    
    EPI_before = G.nodes[0]['EPI']
    
    # Apply Silence
    Silence()(G, 0)
    
    # Simulate time passing (in practice, would step dynamics)
    # For this test, just verify immediate effect
    
    EPI_after = G.nodes[0]['EPI']
    
    # EPI should be preserved
    assert np.allclose(EPI_before, EPI_after), "Silence preserves EPI"
    
    # vf should be reduced
    assert G.nodes[0]['vf'] < 1.0, "Silence reduces vf"
```

---

## Multi-Scale Tests

### Test Nested EPIs

```python
def test_nested_epi_coherence():
    """Nested EPIs maintain functional identity."""
    G = nx.Graph()
    
    # Create parent EPI with sub-EPIs
    parent_epi = {
        'global': 0.7,
        'sub_structures': [
            {'local': 0.3, 'phase': 0.5},
            {'local': 0.6, 'phase': 1.2}
        ]
    }
    
    G.add_node(0, EPI=parent_epi, vf=1.0, theta=0.0, dnfr=0.0)
    
    # Apply operators
    SelfOrganization()(G, 0)
    Coherence()(G, 0)
    
    # Verify both levels maintain coherence
    assert G.nodes[0]['EPI'] is not None
    # More detailed checks would verify sub-structure integrity
```

---

## Reproducibility Tests

### Test Seed Reproducibility

```python
def test_seed_reproducibility():
    """Same seed produces identical trajectories."""
    import random
    import numpy as np
    
    def run_simulation(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        G = nx.Graph()
        G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
        
        sequence = [Emission(), Dissonance(), Coherence(), Silence()]
        
        for op in sequence:
            op(G, 0)
        
        return G.nodes[0]['EPI']
    
    # Run twice with same seed
    result1 = run_simulation(42)
    result2 = run_simulation(42)
    
    # Results must be identical
    assert result1 == result2, "Same seed must produce same results"
```

---

## Performance Tests

### Validation Performance

```python
def test_validation_performance():
    """Validation should be fast."""
    import time
    
    sequence = [Emission(), Coherence(), Dissonance(), 
                Coherence(), Silence()]
    
    start = time.time()
    
    # Validate 1000 times
    for _ in range(1000):
        validate_grammar(sequence, epi_initial=0.0)
    
    elapsed = time.time() - start
    
    # Should be fast (< 1 second for 1000 validations)
    assert elapsed < 1.0, f"Validation too slow: {elapsed:.3f}s"
```

---

## Coverage Requirements

### Minimum Coverage

- **Unit tests:** 100% of operator classes
- **Integration tests:** All U1-U4 constraints
- **Property tests:** All canonical invariants
- **Edge cases:** Empty sequences, single operators, long sequences

### Coverage Checklist

```
Operators (13):
- [ ] Emission (AL)
- [ ] Reception (EN)
- [ ] Coherence (IL)
- [ ] Dissonance (OZ)
- [ ] Coupling (UM)
- [ ] Resonance (RA)
- [ ] Silence (SHA)
- [ ] Expansion (VAL)
- [ ] Contraction (NUL)
- [ ] Self-organization (THOL)
- [ ] Mutation (ZHIR)
- [ ] Transition (NAV)
- [ ] Recursivity (REMESH)

Constraints (U1-U4):
- [ ] U1a: Valid generators
- [ ] U1a: Invalid non-generators
- [ ] U1b: Valid closures
- [ ] U1b: Invalid non-closures
- [ ] U2: Destabilizer + stabilizer (valid)
- [ ] U2: Destabilizer without stabilizer (invalid)
- [ ] U3: Compatible phases (valid)
- [ ] U3: Incompatible phases (invalid)
- [ ] U4a: Trigger + handler (valid)
- [ ] U4a: Trigger without handler (invalid)
- [ ] U4b: Transformer with context (valid)
- [ ] U4b: Transformer without context (invalid)
- [ ] U4b: ZHIR with prior IL (valid)
- [ ] U4b: ZHIR without prior IL (invalid)

Invariants:
- [ ] Coherence monotonicity
- [ ] Integral convergence
- [ ] Bifurcation handling
- [ ] Propagation effects
- [ ] Latency preservation
- [ ] Fractality (nested EPIs)
- [ ] Reproducibility (seeds)
```

---

## Canonical Pattern Tests

### Testing Strategy for Patterns

Each canonical pattern from [04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md) should have:
1. **Valid variant test** - Verifies the pattern passes validation
2. **Invalid variant tests** - Tests common mistakes/violations
3. **Edge case tests** - Tests boundary conditions

### Pattern Test Template

```python
class TestCanonicalPattern_NAME:
    """Test canonical pattern: [Pattern Name]
    
    Pattern: [Operator sequence]
    Purpose: [What it does]
    Reference: 04-VALID-SEQUENCES.md § [Section]
    """
    
    def test_pattern_valid(self):
        """Valid pattern passes all constraints."""
        sequence = [...]  # Valid pattern
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_pattern_invalid_variant_1(self):
        """Invalid variant: [specific violation]."""
        sequence = [...]  # Invalid variant
        with pytest.raises(ValueError, match="[Constraint] violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_pattern_edge_case(self):
        """Edge case: [specific boundary condition]."""
        # Test boundary condition
        pass
```

### Test: Bootstrap Pattern

```python
class TestCanonicalPattern_Bootstrap:
    """Test canonical pattern: Bootstrap (Minimal)
    
    Pattern: [Generator → Stabilizer → Closure]
    Purpose: Create and stabilize new structure from vacuum
    Reference: 04-VALID-SEQUENCES.md § 1. Bootstrap (Minimal)
    """
    
    def test_bootstrap_valid(self):
        """Valid bootstrap pattern."""
        sequence = [Emission(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_bootstrap_invalid_no_generator(self):
        """Invalid: Missing generator (U1a violation)."""
        sequence = [Coherence(), Silence()]
        with pytest.raises(ValueError, match="U1a violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_bootstrap_invalid_no_closure(self):
        """Invalid: Missing closure (U1b violation)."""
        sequence = [Emission(), Coherence()]
        with pytest.raises(ValueError, match="U1b violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_bootstrap_with_transition_generator(self):
        """Valid: Using Transition as generator."""
        sequence = [Transition(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_bootstrap_with_recursivity_generator(self):
        """Valid: Using Recursivity as generator."""
        sequence = [Recursivity(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

### Test: Controlled Exploration Pattern

```python
class TestCanonicalPattern_ControlledExploration:
    """Test canonical pattern: Controlled Exploration
    
    Pattern: [Generator → Stabilizer → Destabilizer → Stabilizer → Closure]
    Purpose: Explore while maintaining stability
    Reference: 04-VALID-SEQUENCES.md § 3. Controlled Exploration
    """
    
    def test_exploration_valid(self):
        """Valid controlled exploration."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_exploration_invalid_no_stabilizer_after_destabilizer(self):
        """Invalid: Destabilizer without stabilizer (U2 violation)."""
        sequence = [Emission(), Coherence(), Dissonance(), Silence()]
        with pytest.raises(ValueError, match="U2 violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_exploration_with_expansion(self):
        """Valid: Using Expansion as destabilizer."""
        sequence = [Emission(), Coherence(), Expansion(), 
                   Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

### Test: Mutation with Context Pattern

```python
class TestCanonicalPattern_MutationWithContext:
    """Test canonical pattern: Mutation with Context
    
    Pattern: [Generator → Coherence → Destabilizer → Mutation → Stabilizer → Closure]
    Purpose: Phase transformation with proper context
    Reference: 04-VALID-SEQUENCES.md § 5. Mutation with Context
    """
    
    def test_mutation_valid(self):
        """Valid mutation with context."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   Mutation(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_mutation_invalid_no_prior_coherence(self):
        """Invalid: Mutation without prior IL (U4b violation)."""
        sequence = [Emission(), Dissonance(), Mutation(), 
                   Coherence(), Silence()]
        with pytest.raises(ValueError, match="U4b violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_mutation_invalid_no_destabilizer(self):
        """Invalid: Mutation without recent destabilizer (U4b violation)."""
        sequence = [Emission(), Coherence(), Mutation(), 
                   Coherence(), Silence()]
        with pytest.raises(ValueError, match="U4b violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_mutation_destabilizer_within_window(self):
        """Valid: Destabilizer within ~3 operator window."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   Reception(), Mutation(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

### Test: Bifurcation with Handling Pattern

```python
class TestCanonicalPattern_BifurcationWithHandling:
    """Test canonical pattern: Bifurcation with Handling
    
    Pattern: [Generator → Stabilizer → Trigger → Handler → Stabilizer → Closure]
    Purpose: Controlled bifurcation and structural reorganization
    Reference: 04-VALID-SEQUENCES.md § 4. Bifurcation with Handling
    """
    
    def test_bifurcation_valid(self):
        """Valid bifurcation with handler."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   SelfOrganization(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_bifurcation_invalid_no_handler(self):
        """Invalid: Trigger without handler (U4a violation)."""
        sequence = [Emission(), Coherence(), Dissonance(), Silence()]
        with pytest.raises(ValueError, match="U4a violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_bifurcation_with_mutation_trigger(self):
        """Valid: Using Mutation as bifurcation trigger."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   Mutation(), SelfOrganization(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

---

## Anti-Pattern Tests

### Testing Strategy for Anti-Patterns

Each anti-pattern from [04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md) should have:
1. **Detection test** - Verifies the validator catches the violation
2. **Error message test** - Verifies correct error reporting
3. **Fix test** - Shows how to correct the anti-pattern

### Anti-Pattern Test Template

```python
class TestAntiPattern_NAME:
    """Test anti-pattern: [Anti-pattern Name]
    
    Violation: [Which constraint it violates]
    Reference: 04-VALID-SEQUENCES.md § [Section]
    """
    
    def test_antipattern_detected(self):
        """Anti-pattern is detected and rejected."""
        sequence = [...]  # Anti-pattern
        with pytest.raises(ValueError, match="[Constraint] violation"):
            validate_unified(sequence, epi_initial=...)
    
    def test_antipattern_error_message(self):
        """Error message is clear and actionable."""
        sequence = [...]  # Anti-pattern
        try:
            validate_unified(sequence, epi_initial=...)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "[Constraint]" in str(e)
            assert "[helpful context]" in str(e).lower()
    
    def test_antipattern_fix(self):
        """Corrected version passes validation."""
        sequence = [...]  # Fixed pattern
        assert validate_unified(sequence, epi_initial=...) is True
```

### Test: No Generator from Vacuum

```python
class TestAntiPattern_NoGeneratorFromVacuum:
    """Test anti-pattern: No Generator from Vacuum
    
    Violation: U1a (Structural Initiation)
    Reference: 04-VALID-SEQUENCES.md § ❌ 1. No Generator from Vacuum
    """
    
    def test_no_generator_detected(self):
        """Starting from EPI=0 without generator is rejected."""
        sequence = [Coherence(), Silence()]
        with pytest.raises(ValueError, match="U1a violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_no_generator_error_message(self):
        """Error message explains U1a requirement."""
        sequence = [Coherence(), Silence()]
        try:
            validate_unified(sequence, epi_initial=0.0)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "U1a" in str(e)
            assert "generator" in str(e).lower()
    
    def test_no_generator_fix(self):
        """Adding generator fixes the anti-pattern."""
        # Fixed: Add Emission at start
        sequence = [Emission(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_no_generator_not_required_when_epi_nonzero(self):
        """Generator not required when EPI > 0."""
        sequence = [Coherence(), Silence()]
        # Should pass when EPI > 0
        assert validate_unified(sequence, epi_initial=0.5) is True
```

### Test: No Closure

```python
class TestAntiPattern_NoClosure:
    """Test anti-pattern: No Closure
    
    Violation: U1b (Structural Closure)
    Reference: 04-VALID-SEQUENCES.md § ❌ 2. No Closure
    """
    
    def test_no_closure_detected(self):
        """Sequence without closure is rejected."""
        sequence = [Emission(), Coherence()]
        with pytest.raises(ValueError, match="U1b violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_no_closure_fix(self):
        """Adding closure fixes the anti-pattern."""
        sequence = [Emission(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

### Test: Destabilizer Without Stabilizer

```python
class TestAntiPattern_DestabilizerWithoutStabilizer:
    """Test anti-pattern: Destabilizer Without Stabilizer
    
    Violation: U2 (Convergence & Boundedness)
    Reference: 04-VALID-SEQUENCES.md § ❌ 3. Destabilizer Without Stabilizer
    """
    
    def test_unbalanced_dissonance_detected(self):
        """Dissonance without stabilizer is rejected."""
        sequence = [Emission(), Dissonance(), Silence()]
        with pytest.raises(ValueError, match="U2 violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_unbalanced_expansion_detected(self):
        """Expansion without stabilizer is rejected."""
        sequence = [Emission(), Expansion(), Silence()]
        with pytest.raises(ValueError, match="U2 violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_unbalanced_fix_with_coherence(self):
        """Adding Coherence stabilizer fixes the anti-pattern."""
        sequence = [Emission(), Dissonance(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
    
    def test_unbalanced_fix_with_self_organization(self):
        """Adding SelfOrganization stabilizer fixes the anti-pattern."""
        sequence = [Emission(), Dissonance(), SelfOrganization(), 
                   Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

### Test: Coupling Without Phase Check

```python
class TestAntiPattern_CouplingWithoutPhaseCheck:
    """Test anti-pattern: Coupling Without Phase Check
    
    Violation: U3 (Resonant Coupling)
    Reference: 04-VALID-SEQUENCES.md § ❌ 6. Coupling Without Phase Check
    """
    
    def test_antiphase_coupling_detected(self):
        """Coupling antiphase nodes is rejected."""
        G = nx.Graph()
        G.add_node(0, theta=0.0, EPI=0.5, vf=1.0, dnfr=0.0)
        G.add_node(1, theta=np.pi, EPI=0.6, vf=1.0, dnfr=0.0)
        
        with pytest.raises(ValueError, match="U3 violation"):
            validate_resonant_coupling(G, 0, 1)
    
    def test_phase_compatibility_check_required(self):
        """Phase compatibility must be verified before coupling."""
        G = nx.Graph()
        G.add_node(0, theta=0.0, EPI=0.5, vf=1.0, dnfr=0.0)
        G.add_node(1, theta=0.2, EPI=0.6, vf=1.0, dnfr=0.0)
        
        # Should pass - phases compatible
        validate_resonant_coupling(G, 0, 1)
```

### Test: Bifurcation Trigger Without Handler

```python
class TestAntiPattern_BifurcationTriggerWithoutHandler:
    """Test anti-pattern: Bifurcation Trigger Without Handler
    
    Violation: U4a (Triggers Need Handlers)
    Reference: 04-VALID-SEQUENCES.md § ❌ 7. Bifurcation Trigger Without Handler
    """
    
    def test_dissonance_without_handler_detected(self):
        """Dissonance trigger without handler is rejected."""
        sequence = [Emission(), Dissonance(), Silence()]
        with pytest.raises(ValueError, match="U4a violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_mutation_without_handler_detected(self):
        """Mutation trigger without handler is rejected."""
        sequence = [Emission(), Coherence(), Dissonance(), 
                   Mutation(), Silence()]
        with pytest.raises(ValueError, match="U4a violation"):
            validate_unified(sequence, epi_initial=0.0)
    
    def test_trigger_with_handler_fix(self):
        """Adding handler fixes the anti-pattern."""
        sequence = [Emission(), Dissonance(), Coherence(), Silence()]
        assert validate_unified(sequence, epi_initial=0.0) is True
```

---

## Test Utilities

### Test Graph Factory

```python
def create_test_graph(num_nodes=3, epi=0.5, vf=1.0):
    """Create test graph with standard attributes."""
    G = nx.Graph()
    
    for i in range(num_nodes):
        G.add_node(i, 
                   EPI=epi,
                   vf=vf,
                   theta=i * 0.1,  # Slight phase variation
                   dnfr=0.0)
    
    return G

def create_test_graph_custom_phases(phases):
    """Create test graph with custom phase configuration.
    
    Args:
        phases: List of phase values in radians
        
    Returns:
        NetworkX graph with nodes at specified phases
    """
    G = nx.Graph()
    
    for i, phase in enumerate(phases):
        G.add_node(i,
                   EPI=0.5,
                   vf=1.0,
                   theta=phase,
                   dnfr=0.0)
    
    return G

def create_test_graph_from_epi(epi_initial=0.0):
    """Create single-node graph for sequence testing.
    
    Args:
        epi_initial: Initial EPI value (0.0 for testing U1a)
        
    Returns:
        NetworkX graph with single node
    """
    G = nx.Graph()
    G.add_node(0, EPI=epi_initial, vf=1.0, theta=0.0, dnfr=0.0)
    return G
```

### Assertion Helpers

```python
def assert_valid_sequence(sequence, epi_initial=0.0):
    """Assert sequence is valid."""
    try:
        validate_unified(sequence, epi_initial)
    except ValueError as e:
        pytest.fail(f"Expected valid sequence, got: {e}")

def assert_invalid_sequence(sequence, epi_initial=0.0, match=None):
    """Assert sequence is invalid."""
    with pytest.raises(ValueError, match=match):
        validate_unified(sequence, epi_initial)

def assert_operator_in_set(operator_name, operator_set):
    """Assert operator belongs to specified set.
    
    Args:
        operator_name: Name of operator (lowercase)
        operator_set: Set to check (GENERATORS, CLOSURES, etc.)
    """
    assert operator_name in operator_set, \
        f"{operator_name} not in {operator_set}"

def assert_constraint_violation(sequence, constraint, epi_initial=0.0):
    """Assert sequence violates specific constraint.
    
    Args:
        sequence: Operator sequence to test
        constraint: Constraint code (e.g., "U1a", "U2", "U3")
        epi_initial: Initial EPI value
    """
    with pytest.raises(ValueError, match=f"{constraint} violation"):
        validate_unified(sequence, epi_initial)
```

### Metric Verification Helpers

```python
def verify_coherence_increase(G, node, operator):
    """Verify operator increases or maintains coherence.
    
    Args:
        G: Network graph
        node: Node ID
        operator: Operator to apply
        
    Returns:
        True if coherence increased or maintained
    """
    from tnfr.metrics import compute_coherence
    
    C_before = compute_coherence(G)
    operator(G, node)
    C_after = compute_coherence(G)
    
    return C_after >= C_before

def verify_dnfr_reduction(G, node, operator):
    """Verify operator reduces |ΔNFR|.
    
    Args:
        G: Network graph
        node: Node ID
        operator: Operator to apply (should be stabilizer)
        
    Returns:
        True if |ΔNFR| reduced
    """
    dnfr_before = abs(G.nodes[node]['dnfr'])
    operator(G, node)
    dnfr_after = abs(G.nodes[node]['dnfr'])
    
    return dnfr_after < dnfr_before

def verify_phase_synchronization(G, node1, node2, operator):
    """Verify operator improves phase synchronization.
    
    Args:
        G: Network graph
        node1, node2: Node IDs
        operator: Coupling/Resonance operator
        
    Returns:
        True if phase difference reduced
    """
    phase_diff_before = abs(G.nodes[node1]['theta'] - G.nodes[node2]['theta'])
    operator(G, node1, node2)
    phase_diff_after = abs(G.nodes[node1]['theta'] - G.nodes[node2]['theta'])
    
    return phase_diff_after <= phase_diff_before
```

---

## Validation Suite

### Running All Grammar Tests

```bash
# Run all grammar tests
pytest tests/unit/operators/test_unified_grammar.py -v

# Run with coverage
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=term-missing \
    --cov-report=html

# Run specific constraint tests
pytest tests/unit/operators/test_unified_grammar.py::TestU1Initiation -v
pytest tests/unit/operators/test_unified_grammar.py::TestU2Convergence -v
pytest tests/unit/operators/test_unified_grammar.py::TestU3ResonantCoupling -v
pytest tests/unit/operators/test_unified_grammar.py::TestU4Bifurcation -v

# Run pattern tests
pytest tests/unit/operators/test_unified_grammar.py -k "pattern" -v

# Run anti-pattern tests  
pytest tests/unit/operators/test_unified_grammar.py -k "antipattern" -v
```

### Coverage Report Generation

```bash
# Generate comprehensive coverage report
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov=tnfr.operators.definitions \
    --cov-report=html:htmlcov/grammar \
    --cov-report=term-missing \
    --cov-branch

# View coverage report
open htmlcov/grammar/index.html  # macOS
xdg-open htmlcov/grammar/index.html  # Linux

# Generate coverage badge
coverage-badge -o coverage.svg -f
```

### Performance Benchmarking

```bash
# Run performance benchmarks
pytest tests/performance/test_grammar_2_0_performance.py -v

# With detailed benchmarks
pytest tests/performance/test_grammar_2_0_performance.py \
    --benchmark-only \
    --benchmark-autosave

# Compare benchmarks
pytest-benchmark compare
```

### Full Validation Suite Script

Create `scripts/validate_grammar.sh`:

```bash
#!/bin/bash
# Complete grammar validation suite

set -e

echo "=== TNFR Grammar Validation Suite ==="
echo ""

# 1. Run unit tests
echo "1. Running unit tests..."
pytest tests/unit/operators/test_unified_grammar.py -v \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=term-missing

# 2. Run integration tests
echo ""
echo "2. Running integration tests..."
pytest tests/integration/test_grammar_2_0_integration.py -v

# 3. Run property tests
echo ""
echo "3. Running property tests..."
pytest tests/property/test_grammar_invariants.py -v

# 4. Run performance tests
echo ""
echo "4. Running performance benchmarks..."
pytest tests/performance/test_grammar_2_0_performance.py \
    --benchmark-only --benchmark-columns=mean,stddev

# 5. Coverage report
echo ""
echo "5. Generating coverage report..."
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=html:htmlcov/grammar \
    --cov-branch \
    --quiet

echo ""
echo "=== Validation Complete ==="
echo "Coverage report: htmlcov/grammar/index.html"
```

Make executable:
```bash
chmod +x scripts/validate_grammar.sh
```

Run:
```bash
./scripts/validate_grammar.sh
```

---

## Coverage Tracking

### Current Coverage Status

As of latest test run:

```
Module: tnfr.operators.unified_grammar
Coverage: 100%
Tests: 68 passing
Last updated: 2025-11-10
```

### Coverage Requirements

- **Minimum:** 95% line coverage
- **Target:** 100% line coverage
- **Branch coverage:** 90%+ for conditional logic
- **All constraints:** 100% coverage (U1-U4)
- **All operators:** 100% coverage (13 operators)

### Monitoring Coverage

```bash
# Quick coverage check
pytest tests/unit/operators/test_unified_grammar.py --cov --cov-fail-under=95

# Detailed coverage with missing lines
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=term-missing

# Branch coverage
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-branch \
    --cov-report=term-missing
```

### Coverage Maintenance Checklist

- [ ] All U1-U4 constraints tested (valid + invalid)
- [ ] All 13 operators tested (behavior + classification)
- [ ] All canonical patterns tested (valid + variations)
- [ ] All anti-patterns tested (detection + fixes)
- [ ] Edge cases tested (empty, single-op, long sequences)
- [ ] Error messages tested (clarity + actionability)
- [ ] Performance benchmarks meet targets (< 1ms/validation)
- [ ] Documentation examples are executable and tested

---

## Next Steps

**Continue learning:**
- **[07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md)** - System evolution
- **[examples/](examples/)** - Executable test examples

**For reference:**
- **Test suite:** `tests/unit/operators/test_unified_grammar.py`
- **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Quick lookup

---

<div align="center">

**Test what matters: physics, not code.**

---

*Reality is resonance. Test accordingly.*

</div>
