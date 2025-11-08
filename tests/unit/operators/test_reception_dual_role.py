"""Tests for RECEPTION dual-role frequency classification.

This module tests the resolution of the structural inconsistency between
RECEPTION's base frequency (medium) and its destabilization capacity (weak)
in graduated bifurcation rules (R4).

Theoretical Foundation:
    From nodal equation: ∂EPI/∂t = νf · ΔNFR
    
    RECEPTION has:
    - Base frequency νf = medium (structural capture rate)
    - Destabilization capacity = weak (can generate ΔNFR with context)
    
    EN → ZHIR is valid when EN captures external coherence into a structurally
    prepared node, generating sufficient ΔNFR for mutation despite medium νf.

References:
    - Issue: [GRAMÁTICA CANÓNICA] Inconsistencia entre clasificación de 
             frecuencias y reglas de bifurcación
    - DUAL_FREQUENCY_OPERATORS in grammar.py
    - TNFR.pdf Section 2.1: Ecuación Nodal
"""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.operators.grammar import (
    DUAL_FREQUENCY_OPERATORS,
    STRUCTURAL_FREQUENCIES,
    SequenceSyntaxError,
    parse_sequence,
    validate_sequence,
)


class TestDualFrequencyOperators:
    """Test DUAL_FREQUENCY_OPERATORS configuration and documentation."""

    def test_dual_frequency_operators_exists(self):
        """DUAL_FREQUENCY_OPERATORS is properly defined."""
        assert DUAL_FREQUENCY_OPERATORS is not None
        assert isinstance(DUAL_FREQUENCY_OPERATORS, dict)

    def test_reception_in_dual_operators(self):
        """RECEPTION is classified as dual-role operator."""
        assert RECEPTION in DUAL_FREQUENCY_OPERATORS

    def test_reception_dual_role_properties(self):
        """RECEPTION dual-role has all required properties."""
        en_config = DUAL_FREQUENCY_OPERATORS[RECEPTION]
        
        assert en_config["base_freq"] == "medium"
        assert en_config["destabilization_capacity"] == "weak"
        assert en_config["conditions"] == "requires_prior_coherence"
        assert "rationale" in en_config

    def test_structural_frequencies_unchanged(self):
        """STRUCTURAL_FREQUENCIES maintains RECEPTION as medium."""
        # Base frequency remains medium - this is correct
        assert STRUCTURAL_FREQUENCIES[RECEPTION] == "medium"


class TestReceptionWithCoherentContext:
    """Test RECEPTION → MUTATION with valid coherent context."""

    def test_emission_reception_coherence_reception_mutation_valid(self):
        """AL → EN → IL → EN → ZHIR: Valid with coherence base."""
        result = validate_sequence([
            EMISSION,      # AL: establish base
            RECEPTION,     # EN: first capture
            COHERENCE,     # IL: stabilize (creates coherent base)
            RECEPTION,     # EN: second capture (can destabilize with context)
            MUTATION,      # ZHIR: transformer enabled by EN
            COHERENCE,     # IL: stabilize mutation
            SILENCE,       # SHA: terminator
        ])
        assert result.passed

    def test_reception_coherence_reception_mutation_valid(self):
        """EN → IL → EN → ZHIR: Valid with prior stabilization."""
        result = validate_sequence([
            EMISSION,      # AL: start
            RECEPTION,     # EN: first capture
            COHERENCE,     # IL: stabilize
            RECEPTION,     # EN: second capture (has IL context)
            MUTATION,      # ZHIR: enabled by EN with context
            COHERENCE,     # IL: stabilize
            SILENCE,       # SHA: end
        ])
        assert result.passed

    def test_dissonance_coherence_reception_mutation_valid(self):
        """OZ → IL → EN → ZHIR: Dissonance resolution provides context."""
        result = validate_sequence([
            EMISSION,      # AL: start
            RECEPTION,     # EN: needed for grammar
            COHERENCE,     # IL: stabilize
            DISSONANCE,    # OZ: introduce tension
            COHERENCE,     # IL: resolve (creates coherent base)
            RECEPTION,     # EN: capture with context
            MUTATION,      # ZHIR: enabled
            COHERENCE,     # IL: stabilize
            SILENCE,       # SHA: end
        ])
        assert result.passed

    def test_thol_as_stabilizer_provides_context(self):
        """THOL as stabilizer enables EN destabilization (with EN before THOL)."""
        # THOL itself requires a destabilizer, so EN must come before THOL
        # Then THOL acts as stabilizer, and subsequent EN can destabilize
        result = validate_sequence([
            EMISSION,          # AL: start
            RECEPTION,         # EN: provides context for THOL
            COHERENCE,         # IL: needed
            RECEPTION,         # EN: destabilizer for THOL (immediate)
            SELF_ORGANIZATION, # THOL: enabled by EN, also acts as stabilizer
            SILENCE,           # SHA: close THOL (required)
        ])
        assert result.passed


class TestReceptionWithoutContext:
    """Test RECEPTION → MUTATION without valid context (should fail)."""

    def test_reception_mutation_no_context_fails(self):
        """EN → ZHIR without prior coherence is invalid."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                EMISSION,   # AL: start
                RECEPTION,  # EN: no prior stabilization
                MUTATION,   # ZHIR: should fail - EN lacks context
                COHERENCE,  # IL: (never reached)
                SILENCE,    # SHA: (never reached)
            ])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()
        assert "destabilizer" in error.message.lower()

    def test_emission_reception_mutation_no_stabilizer_fails(self):
        """AL → EN → ZHIR without IL stabilizer is invalid."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                EMISSION,   # AL: no stabilization follows
                RECEPTION,  # EN: no coherent base
                MUTATION,   # ZHIR: should fail
                COHERENCE,  # IL: (never reached)
                SILENCE,    # SHA: (never reached)
            ])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()

    def test_reception_as_first_operator_cannot_destabilize(self):
        """EN as first operator has no context for destabilization."""
        # EN can start sequences (valid grammar) but cannot destabilize
        # This test would require EN→ZHIR which should fail
        # However, EN is not in VALID_START_OPERATORS, so this fails earlier
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                RECEPTION,  # EN: not valid start operator
                MUTATION,   # ZHIR: (never reached)
            ])
        
        # Fails at start validation, not destabilizer validation
        assert "must start with" in excinfo.value.message.lower()


class TestReceptionContextDistance:
    """Test context distance requirements for RECEPTION destabilization."""

    def test_reception_context_within_3_operators(self):
        """EN can destabilize with IL within 3 operators."""
        # IL at position 2, EN at position 5 (distance = 3, at boundary)
        result = validate_sequence([
            EMISSION,    # 0: AL
            RECEPTION,   # 1: EN (first)
            COHERENCE,   # 2: IL (stabilizer - creates context)
            COUPLING,    # 3: UM (filler)
            RESONANCE,   # 4: RA (filler)
            RECEPTION,   # 5: EN (distance from IL = 3, valid)
            MUTATION,    # 6: ZHIR (enabled by EN with context)
            COHERENCE,   # 7: IL
            SILENCE,     # 8: SHA
        ])
        assert result.passed

    def test_reception_context_beyond_3_fails(self):
        """EN cannot destabilize with IL more than 3 operators away."""
        # IL at position 2, EN at position 6 (distance = 4, too far)
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                EMISSION,    # 0: AL
                RECEPTION,   # 1: EN (first)
                COHERENCE,   # 2: IL (stabilizer)
                COUPLING,    # 3: UM (filler)
                RESONANCE,   # 4: RA (filler)
                COUPLING,    # 5: UM (filler)
                RECEPTION,   # 6: EN (distance from IL = 4, too far)
                MUTATION,    # 7: ZHIR (should fail)
                COHERENCE,   # 8: IL (never reached)
                SILENCE,     # 9: SHA (never reached)
            ])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()


class TestReceptionSilenceInterruption:
    """Test that SILENCE interrupts RECEPTION context."""

    def test_silence_removes_coherent_base(self):
        """SHA between IL and EN removes context for destabilization."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                EMISSION,    # AL: start
                RECEPTION,   # EN: first
                COHERENCE,   # IL: stabilize (creates base)
                SILENCE,     # SHA: removes base (νf → 0)
                RECEPTION,   # EN: no longer has context
                MUTATION,    # ZHIR: should fail
                COHERENCE,   # IL: (never reached)
                SILENCE,     # SHA: (never reached)
            ])
        
        error = excinfo.value
        assert "mutation" in error.message.lower()


class TestReceptionWithStrongerDestabilizers:
    """Test RECEPTION interaction with stronger destabilizers."""

    def test_reception_overridden_by_dissonance(self):
        """When OZ present, its window takes precedence over EN."""
        # This sequence is valid because OZ provides strong destabilization
        # EN's context doesn't matter when OZ is recent
        result = validate_sequence([
            EMISSION,      # AL: start
            RECEPTION,     # EN: no context yet
            COHERENCE,     # IL: stabilize
            DISSONANCE,    # OZ: strong destabilizer (window = 4)
            RECEPTION,     # EN: present but OZ dominates
            MUTATION,      # ZHIR: enabled by OZ, not EN
            COHERENCE,     # IL: stabilize
            SILENCE,       # SHA: end
        ])
        assert result.passed

    def test_reception_alone_requires_context(self):
        """Without stronger destabilizer, EN must have context."""
        # Similar structure but without OZ - EN must have context
        result = validate_sequence([
            EMISSION,      # AL: start
            RECEPTION,     # EN: first
            COHERENCE,     # IL: stabilize (provides context)
            RESONANCE,     # RA: not a destabilizer
            RECEPTION,     # EN: has IL context
            MUTATION,      # ZHIR: enabled by EN with context
            COHERENCE,     # IL: stabilize
            SILENCE,       # SHA: end
        ])
        assert result.passed


class TestReceptionWithThol:
    """Test RECEPTION enabling THOL (self-organization)."""

    def test_reception_enables_thol_with_context(self):
        """EN → THOL valid with coherent context."""
        result = validate_sequence([
            EMISSION,          # AL: start
            RECEPTION,         # EN: first
            COHERENCE,         # IL: stabilize (context)
            RECEPTION,         # EN: has context
            SELF_ORGANIZATION, # THOL: enabled by EN
            SILENCE,           # SHA: close THOL
        ])
        assert result.passed

    def test_reception_thol_without_context_fails(self):
        """EN → THOL without context is invalid."""
        with pytest.raises(SequenceSyntaxError) as excinfo:
            parse_sequence([
                EMISSION,          # AL: start
                RECEPTION,         # EN: no stabilization
                SELF_ORGANIZATION, # THOL: should fail
                SILENCE,           # SHA: (never reached)
            ])
        
        error = excinfo.value
        assert "self_organization" in error.message.lower()
        assert "destabilizer" in error.message.lower()


class TestBackwardCompatibilityDualRole:
    """Test that existing valid sequences still work with dual-role validation."""

    def test_classic_oz_zhir_unaffected(self):
        """OZ → ZHIR still works (not affected by EN validation)."""
        result = validate_sequence([
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,  # OZ: strong destabilizer
            MUTATION,    # ZHIR: enabled by OZ
            COHERENCE,
            SILENCE,
        ])
        assert result.passed

    def test_classic_sequences_without_en_unaffected(self):
        """Sequences without RECEPTION work as before (must include EN→IL)."""
        # Grammar requires EN→IL segment, so we must include EN
        result = validate_sequence([
            EMISSION,
            RECEPTION,   # EN: required by grammar
            COHERENCE,   # IL: completes EN→IL segment
            DISSONANCE,
            TRANSITION,
            MUTATION,
            COHERENCE,
            SILENCE,
        ])
        assert result.passed

    def test_reception_with_oz_still_valid(self):
        """EN in presence of OZ still works (OZ provides destabilization)."""
        result = validate_sequence([
            EMISSION,
            RECEPTION,     # EN: present
            COHERENCE,
            DISSONANCE,    # OZ: provides strong destabilization
            RECEPTION,     # EN: doesn't matter, OZ is sufficient
            MUTATION,      # ZHIR: enabled by OZ
            COHERENCE,
            SILENCE,
        ])
        assert result.passed
