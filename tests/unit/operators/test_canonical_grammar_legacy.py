"""Tests for canonical_grammar legacy module (deprecated).

This module tests the legacy canonical_grammar.py that now delegates
to unified_grammar.py. All tests should pass with DeprecationWarnings.
"""

import warnings

import pytest

from tnfr.operators.definitions import (
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Mutation,
    Silence,
    Coupling,
    SelfOrganization,
)


class TestLegacyModuleWarnings:
    """Test that the legacy module emits proper deprecation warnings."""

    def test_module_import_emits_warning(self):
        """Importing canonical_grammar should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-import to trigger warning
            import importlib
            import tnfr.operators.canonical_grammar
            importlib.reload(tnfr.operators.canonical_grammar)
            
            # Check that a warning was emitted
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert "canonical_grammar is deprecated" in str(deprecation_warnings[0].message)
            assert "unified_grammar" in str(deprecation_warnings[0].message)


class TestLegacyValidatorMethods:
    """Test that legacy CanonicalGrammarValidator methods emit warnings."""

    def test_validate_initialization_emits_warning(self):
        """validate_initialization() should emit DeprecationWarning."""
        # Import with suppressed import warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator
        
        ops = [Emission(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = CanonicalGrammarValidator.validate_initialization(ops)
            
            # Check warning
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_initialization is deprecated" in str(w[0].message)
            
            # Check functionality still works
            assert result[0] is True
            assert "U1a" in result[1]

    def test_validate_convergence_emits_warning(self):
        """validate_convergence() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator
        
        ops = [Emission(), Dissonance(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = CanonicalGrammarValidator.validate_convergence(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_convergence is deprecated" in str(w[0].message)
            
            # Check functionality
            assert result[0] is True
            assert "U2" in result[1]

    def test_validate_phase_compatibility_emits_warning(self):
        """validate_phase_compatibility() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator
        
        ops = [Emission(), Coupling(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = CanonicalGrammarValidator.validate_phase_compatibility(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_phase_compatibility is deprecated" in str(w[0].message)
            
            # Check functionality
            assert result[0] is True
            assert "U3" in result[1]

    def test_validate_bifurcation_limits_emits_warning(self):
        """validate_bifurcation_limits() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator
        
        ops = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = CanonicalGrammarValidator.validate_bifurcation_limits(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_bifurcation_limits is deprecated" in str(w[0].message)
            
            # Check functionality
            assert result[0] is True
            assert "U4a" in result[1]

    def test_validate_method_emits_warning(self):
        """validate() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator
        
        ops = [Emission(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_valid, messages = CanonicalGrammarValidator.validate(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "CanonicalGrammarValidator.validate is deprecated" in str(w[0].message)
            
            # Check functionality
            assert is_valid is True
            assert len(messages) > 0


class TestLegacyFunctions:
    """Test that legacy standalone functions emit warnings."""

    def test_validate_canonical_only_emits_warning(self):
        """validate_canonical_only() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import validate_canonical_only
        
        ops = [Emission(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_canonical_only(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_canonical_only is deprecated" in str(w[0].message)
            assert "validate_unified" in str(w[0].message)
            
            # Check functionality
            assert result is True

    def test_validate_with_conventions_emits_warning(self):
        """validate_with_conventions() should emit DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import validate_with_conventions
        
        ops = [Emission(), Coherence(), Silence()]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_valid, messages = validate_with_conventions(ops)
            
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_with_conventions is deprecated" in str(w[0].message)
            
            # Check functionality
            assert is_valid is True
            assert len(messages) > 0


class TestLegacyOperatorSets:
    """Test that legacy operator set names are available."""

    def test_legacy_operator_sets_available(self):
        """Legacy operator set names should be available."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import (
                GENERATOR_OPS,
                STABILIZER_OPS,
                DESTABILIZER_OPS,
                COUPLING_OPS,
                BIFURCATION_TRIGGER_OPS,
                BIFURCATION_HANDLER_OPS,
                CLOSURE_OPS,
                TRANSFORMER_OPS,
            )
        
        # Check that all sets are available and contain expected operators
        assert 'emission' in GENERATOR_OPS
        assert 'coherence' in STABILIZER_OPS
        assert 'dissonance' in DESTABILIZER_OPS
        assert 'coupling' in COUPLING_OPS
        assert 'mutation' in BIFURCATION_TRIGGER_OPS
        assert 'coherence' in BIFURCATION_HANDLER_OPS
        assert 'silence' in CLOSURE_OPS
        assert 'mutation' in TRANSFORMER_OPS


class TestBackwardCompatibility:
    """Test that the legacy API produces same results as unified grammar."""

    def test_rc1_maps_to_u1a(self):
        """RC1 (initialization) maps to U1a (initiation)."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator as LegacyValidator
            from tnfr.operators.unified_grammar import UnifiedGrammarValidator
        
        ops = [Emission(), Coherence(), Silence()]
        
        # Get results from both
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            legacy_result = LegacyValidator.validate_initialization(ops)
        unified_result = UnifiedGrammarValidator.validate_initiation(ops)
        
        # Should have same validity
        assert legacy_result[0] == unified_result[0]
        # Both should mention the rule (legacy uses RC1, unified uses U1a)
        assert legacy_result[0] is True

    def test_rc2_maps_to_u2(self):
        """RC2 (convergence) maps to U2 (convergence & boundedness)."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator as LegacyValidator
            from tnfr.operators.unified_grammar import UnifiedGrammarValidator
        
        ops = [Emission(), Dissonance(), Coherence(), Silence()]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            legacy_result = LegacyValidator.validate_convergence(ops)
        unified_result = UnifiedGrammarValidator.validate_convergence(ops)
        
        assert legacy_result[0] == unified_result[0]
        assert legacy_result[0] is True

    def test_rc3_maps_to_u3(self):
        """RC3 (phase verification) maps to U3 (resonant coupling)."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator as LegacyValidator
            from tnfr.operators.unified_grammar import UnifiedGrammarValidator
        
        ops = [Emission(), Coupling(), Coherence(), Silence()]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            legacy_result = LegacyValidator.validate_phase_compatibility(ops)
        unified_result = UnifiedGrammarValidator.validate_resonant_coupling(ops)
        
        assert legacy_result[0] == unified_result[0]
        assert legacy_result[0] is True

    def test_rc4_maps_to_u4a(self):
        """RC4 (bifurcation limits) maps to U4a (bifurcation triggers)."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from tnfr.operators.canonical_grammar import CanonicalGrammarValidator as LegacyValidator
            from tnfr.operators.unified_grammar import UnifiedGrammarValidator
        
        ops = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            legacy_result = LegacyValidator.validate_bifurcation_limits(ops)
        unified_result = UnifiedGrammarValidator.validate_bifurcation_triggers(ops)
        
        assert legacy_result[0] == unified_result[0]
        assert legacy_result[0] is True
