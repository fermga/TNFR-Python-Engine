"""Tests for deprecated C1-C3 validation functions and their warnings."""

import warnings

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RESONANCE,
    SILENCE,
    SELF_ORGANIZATION,
)
from tnfr.operators.grammar import (
    validate_c1_existence,
    validate_c2_boundedness,
    validate_c3_threshold,
    # Test that unified grammar is re-exported
    UnifiedGrammarValidator,
    validate_unified,
    UNIFIED_GENERATORS,
    UNIFIED_CLOSURES,
    UNIFIED_STABILIZERS,
    UNIFIED_DESTABILIZERS,
)


def test_validate_c1_existence_emits_deprecation_warning():
    """Verify validate_c1_existence emits DeprecationWarning."""
    sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_c1_existence(sequence)
        
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "validate_c1_existence is deprecated" in str(w[0].message)
        assert "UnifiedGrammarValidator" in str(w[0].message)
        
    # Should still return correct result
    assert result is True


def test_validate_c1_existence_rejects_invalid_start():
    """Verify C1 validation catches invalid start operators."""
    sequence = [RECEPTION, COHERENCE, SILENCE]  # Invalid start
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c1_existence(sequence)
        
    assert result is False


def test_validate_c1_existence_rejects_invalid_end():
    """Verify C1 validation catches invalid end operators."""
    sequence = [EMISSION, RECEPTION, COHERENCE]  # Invalid end
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c1_existence(sequence)
        
    assert result is False


def test_validate_c2_boundedness_emits_deprecation_warning():
    """Verify validate_c2_boundedness emits DeprecationWarning."""
    sequence = [EMISSION, DISSONANCE, COHERENCE, SILENCE]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_c2_boundedness(sequence)
        
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "validate_c2_boundedness is deprecated" in str(w[0].message)
        
    # Should still return correct result
    assert result is True


def test_validate_c2_boundedness_accepts_no_destabilizers():
    """Verify C2 validation passes when no destabilizers present."""
    sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c2_boundedness(sequence)
        
    assert result is True


def test_validate_c2_boundedness_rejects_unbalanced_destabilizers():
    """Verify C2 validation catches destabilizers without stabilizers."""
    sequence = [EMISSION, DISSONANCE, RESONANCE, SILENCE]  # OZ without IL/THOL
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c2_boundedness(sequence)
        
    assert result is False


def test_validate_c3_threshold_emits_deprecation_warning():
    """Verify validate_c3_threshold emits DeprecationWarning."""
    # Need COHERENCE before MUTATION as well (C3 requirement)
    sequence = [EMISSION, DISSONANCE, COHERENCE, MUTATION, SILENCE]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = validate_c3_threshold(sequence)
        
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "validate_c3_threshold is deprecated" in str(w[0].message)
        
    # Should still return correct result
    assert result is True


def test_validate_c3_threshold_accepts_no_transformers():
    """Verify C3 validation passes when no transformers present."""
    sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c3_threshold(sequence)
        
    assert result is True


def test_validate_c3_threshold_rejects_mutation_without_context():
    """Verify C3 validation catches mutation without recent destabilizer."""
    sequence = [EMISSION, RECEPTION, MUTATION, SILENCE]  # ZHIR without recent OZ
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = validate_c3_threshold(sequence)
        
    assert result is False


def test_unified_grammar_validator_is_exported():
    """Verify UnifiedGrammarValidator is properly re-exported."""
    assert UnifiedGrammarValidator is not None
    assert hasattr(UnifiedGrammarValidator, 'validate')
    assert hasattr(UnifiedGrammarValidator, 'validate_initiation')
    assert hasattr(UnifiedGrammarValidator, 'validate_closure')
    assert hasattr(UnifiedGrammarValidator, 'validate_convergence')


def test_validate_unified_is_exported():
    """Verify validate_unified convenience function is exported."""
    assert validate_unified is not None
    assert callable(validate_unified)


def test_unified_operator_sets_are_exported():
    """Verify unified operator sets are properly re-exported."""
    # Check that all unified operator sets are available
    assert UNIFIED_GENERATORS is not None
    assert UNIFIED_CLOSURES is not None
    assert UNIFIED_STABILIZERS is not None
    assert UNIFIED_DESTABILIZERS is not None
    
    # Verify they are frozen sets
    assert isinstance(UNIFIED_GENERATORS, frozenset)
    assert isinstance(UNIFIED_CLOSURES, frozenset)
    assert isinstance(UNIFIED_STABILIZERS, frozenset)
    assert isinstance(UNIFIED_DESTABILIZERS, frozenset)
    
    # Verify they contain expected operators
    assert "emission" in UNIFIED_GENERATORS
    assert "silence" in UNIFIED_CLOSURES
    assert "coherence" in UNIFIED_STABILIZERS
    assert "dissonance" in UNIFIED_DESTABILIZERS


def test_c1_c3_mapping_to_unified():
    """Verify that C1-C3 validations map correctly to U1-U4."""
    # This sequence should pass all validations
    sequence = [EMISSION, DISSONANCE, COHERENCE, MUTATION, SELF_ORGANIZATION, SILENCE]
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        
        # Old C1-C3 validations
        c1_valid = validate_c1_existence(sequence)
        c2_valid = validate_c2_boundedness(sequence)
        c3_valid = validate_c3_threshold(sequence)
        
    # All should pass
    assert c1_valid is True
    assert c2_valid is True
    assert c3_valid is True


def test_backward_compatibility_with_existing_code():
    """Verify that existing code using validate_sequence still works."""
    from tnfr.operators.grammar import validate_sequence
    
    # This is how existing code calls validate_sequence
    result = validate_sequence([EMISSION, RECEPTION, COHERENCE, SILENCE])
    
    # Should pass without issues
    assert result.passed
    assert result.message == "ok"
