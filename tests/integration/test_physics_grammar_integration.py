"""Integration test demonstrating physics-based operator derivation.

This test shows that the grammar validation uses operators derived from
TNFR physical principles rather than arbitrary lists.
"""

import pytest

from tnfr.config.operator_names import (
    EMISSION,
    RECEPTION,
    COHERENCE,
    TRANSITION,
    SILENCE,
    RECURSIVITY,
    validate_physics_derivation,
)
from tnfr.config.physics_derivation import (
    derive_start_operators_from_physics,
    derive_end_operators_from_physics,
)
from tnfr.operators.grammar import validate_sequence


def test_physics_derived_operators_are_used_by_grammar():
    """Grammar validation uses physics-derived start/end operators."""
    # Get physics-derived operators
    physics_start = derive_start_operators_from_physics()
    physics_end = derive_end_operators_from_physics()
    
    # Verify they match what the validation expects
    validation = validate_physics_derivation()
    assert validation["start_operators_valid"], (
        f"Start operators mismatch: {validation['discrepancies']}"
    )
    assert validation["end_operators_valid"], (
        f"End operators mismatch: {validation['discrepancies']}"
    )
    
    # Test that EMISSION (physics-derived start) works
    assert EMISSION in physics_start
    result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE])
    assert result.passed
    
    # Test that TRANSITION (NEW physics-derived start) works
    assert TRANSITION in physics_start
    result = validate_sequence([TRANSITION, RECEPTION, COHERENCE, SILENCE])
    assert result.passed
    
    # Test that SILENCE (physics-derived end) works
    assert SILENCE in physics_end
    result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE])
    assert result.passed
    
    # Test that RECURSIVITY (physics-derived start AND end) works
    assert RECURSIVITY in physics_start
    assert RECURSIVITY in physics_end
    result = validate_sequence([RECURSIVITY, RECEPTION, COHERENCE, RECURSIVITY])
    assert result.passed


def test_non_physics_operators_are_rejected_as_starts():
    """Operators that can't generate/activate EPI are rejected as starts."""
    physics_start = derive_start_operators_from_physics()
    
    # RECEPTION cannot start (needs external source + existing EPI)
    assert RECEPTION not in physics_start
    result = validate_sequence([RECEPTION, COHERENCE, SILENCE])
    assert not result.passed
    assert "must start" in result.message.lower()
    
    # COHERENCE cannot start (stabilizes existing, can't create)
    assert COHERENCE not in physics_start
    result = validate_sequence([COHERENCE, SILENCE])
    assert not result.passed
    assert "must start" in result.message.lower()


def test_physics_derivation_linked_to_nodal_equation():
    """Verify derivation is based on nodal equation ∂EPI/∂t = νf · ΔNFR."""
    from tnfr.config.physics_derivation import (
        can_generate_epi_from_null,
        can_stabilize_reorganization,
    )
    
    # EMISSION generates EPI: creates νf > 0 and ΔNFR > 0
    # From ∂EPI/∂t = νf · ΔNFR, this produces ∂EPI/∂t > 0
    assert can_generate_epi_from_null(EMISSION)
    
    # SILENCE stabilizes: forces νf → 0
    # From ∂EPI/∂t = νf · ΔNFR, this produces ∂EPI/∂t → 0
    assert can_stabilize_reorganization(SILENCE)


def test_transition_added_as_physics_derived_start():
    """TRANSITION was added as start operator based on physics derivation."""
    physics_start = derive_start_operators_from_physics()
    
    # TRANSITION can activate latent EPI (phase activation)
    assert TRANSITION in physics_start
    
    # Verify it works in actual grammar validation
    result = validate_sequence([TRANSITION, RECEPTION, COHERENCE, SILENCE])
    assert result.passed, f"TRANSITION start failed: {result.message}"
    
    # This is the key difference: previously TRANSITION wasn't a valid
    # start operator. Now it is, because physics derivation shows it can
    # activate nodes from another phase.


def test_backward_compatibility_maintained():
    """Original start operators (EMISSION, RECURSIVITY) still work."""
    physics_start = derive_start_operators_from_physics()
    
    # Original operators still present
    assert EMISSION in physics_start
    assert RECURSIVITY in physics_start
    
    # Original sequences still valid
    result1 = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE])
    assert result1.passed
    
    result2 = validate_sequence([RECURSIVITY, RECEPTION, COHERENCE, SILENCE])
    assert result2.passed


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
