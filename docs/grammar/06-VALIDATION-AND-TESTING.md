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
Integration Tests  → Sequence validation
Property Tests     → Invariants (coherence, convergence, etc.)
Performance Tests  → Scalability and efficiency
Regression Tests   → Prevent breakage of existing functionality
```

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
```

### Assertion Helpers

```python
def assert_valid_sequence(sequence, epi_initial=0.0):
    """Assert sequence is valid."""
    try:
        validate_grammar(sequence, epi_initial)
    except ValueError as e:
        pytest.fail(f"Expected valid sequence, got: {e}")

def assert_invalid_sequence(sequence, epi_initial=0.0, match=None):
    """Assert sequence is invalid."""
    with pytest.raises(ValueError, match=match):
        validate_grammar(sequence, epi_initial)
```

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
