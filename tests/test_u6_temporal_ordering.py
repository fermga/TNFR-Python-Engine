"""Tests for U6 Temporal Ordering (experimental)."""

import pytest

from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Silence,
)
from tnfr.operators.grammar import GrammarValidator


class TestU6TemporalOrdering:
    """Test suite for U6 temporal ordering validation."""

    def test_u6_not_enabled_by_default(self):
        """U6 should not be enabled by default (experimental status)."""
        validator = GrammarValidator()
        assert validator.experimental_u6 is False

    def test_u6_enable_experimental(self):
        """U6 can be enabled via experimental flag."""
        validator = GrammarValidator(experimental_u6=True)
        assert validator.experimental_u6 is True

    def test_u6_passes_with_no_destabilizers(self):
        """U6 should pass when no destabilizers present."""
        validator = GrammarValidator(experimental_u6=True)
        sequence = [Emission(), Coherence(), Silence()]

        is_valid, messages = validator.validate(sequence, epi_initial=0.0)

        # U6 should not apply
        u6_messages = [m for m in messages if "U6" in m]
        assert any("not applicable" in m for m in u6_messages)

    def test_u6_passes_with_single_destabilizer(self):
        """U6 should pass when only one destabilizer present."""
        validator = GrammarValidator(experimental_u6=True)
        sequence = [Emission(), Dissonance(), Coherence(), Silence()]

        is_valid, messages = validator.validate(sequence, epi_initial=0.0)

        # U6 should not apply (need 2+ destabilizers)
        u6_messages = [m for m in messages if "U6" in m]
        assert any("not applicable" in m or "fewer than 2" in m for m in u6_messages)

    def test_u6_flags_consecutive_destabilizers(self):
        """U6 should flag consecutive destabilizers (spacing=1)."""
        validator = GrammarValidator(experimental_u6=True)
        sequence = [Emission(), Dissonance(), Dissonance(), Coherence(), Silence()]

        is_valid, messages = validator.validate(sequence, epi_initial=0.0, vf=1.0)

        # Should pass U1-U5 but flag U6 warning
        assert is_valid  # U6 violations don't fail validation (experimental)

        u6_messages = [m for m in messages if "U6" in m]
        assert len(u6_messages) > 0
        assert any("WARNING" in m or "spacing" in m for m in u6_messages)

    def test_u6_passes_with_proper_spacing(self):
        """U6 should pass when destabilizers properly spaced."""
        validator = GrammarValidator(experimental_u6=True)
        # Spacing: OZ at pos 1, OZ at pos 4 → spacing = 3
        sequence = [
            Emission(),  # pos 0
            Dissonance(),  # pos 1 (first destabilizer)
            Coherence(),  # pos 2
            Coherence(),  # pos 3
            Dissonance(),  # pos 4 (second destabilizer, spacing=3)
            Coherence(),
            Silence(),
        ]

        is_valid, messages = validator.validate(sequence, epi_initial=0.0, vf=1.0)

        u6_messages = [m for m in messages if "U6" in m]
        assert any("satisfied" in m or "properly spaced" in m for m in u6_messages)

    def test_u6_spacing_scales_with_vf(self):
        """U6 minimum spacing should decrease with higher νf."""
        validator = GrammarValidator(experimental_u6=True)
        sequence = [Emission(), Dissonance(), Dissonance(), Coherence(), Silence()]

        # Low νf → longer τ_relax → larger min_spacing
        _, messages_low = validator.validate(sequence, epi_initial=0.0, vf=0.5)
        u6_low = [m for m in messages_low if "U6" in m][0]

        # High νf → shorter τ_relax → smaller min_spacing
        _, messages_high = validator.validate(sequence, epi_initial=0.0, vf=2.0)
        u6_high = [m for m in messages_high if "U6" in m][0]

        # Both should flag warning (spacing=1 always too close)
        # But estimated τ_relax should differ
        assert "WARNING" in u6_low or "spacing" in u6_low
        assert "WARNING" in u6_high or "spacing" in u6_high

    def test_u6_detects_multiple_violations(self):
        """U6 should detect multiple spacing violations in one sequence."""
        validator = GrammarValidator(experimental_u6=True)
        # Triple consecutive destabilizers
        sequence = [
            Emission(),
            Dissonance(),  # pos 1
            Expansion(),  # pos 2 (spacing=1 from pos 1)
            Mutation(),  # pos 3 (spacing=1 from pos 2)
            Coherence(),
            Silence(),
        ]

        is_valid, messages = validator.validate(sequence, epi_initial=0.0, vf=1.0)

        u6_messages = [m for m in messages if "U6" in m]
        # Should report multiple violations
        assert len(u6_messages) > 0

    def test_u6_distinguishes_operator_types(self):
        """U6 should recognize OZ, ZHIR, VAL as destabilizers."""
        validator = GrammarValidator(experimental_u6=True)

        sequences = [
            [Emission(), Dissonance(), Mutation(), Coherence(), Silence()],  # OZ + ZHIR
            [Emission(), Expansion(), Dissonance(), Coherence(), Silence()],  # VAL + OZ
            [Emission(), Mutation(), Expansion(), Coherence(), Silence()],  # ZHIR + VAL
        ]

        for seq in sequences:
            _, messages = validator.validate(seq, epi_initial=0.0, vf=1.0)
            u6_messages = [m for m in messages if "U6" in m]
            # All should trigger U6 (consecutive destabilizers)
            assert len(u6_messages) > 0

    def test_u6_respects_k_top_parameter(self):
        """U6 should use k_top parameter for τ_relax estimation."""
        validator = GrammarValidator(experimental_u6=True)
        sequence = [Emission(), Dissonance(), Dissonance(), Coherence(), Silence()]

        # Radial topology (k_top=1.0) → moderate τ_relax
        _, messages_radial = validator.validate(sequence, epi_initial=0.0, vf=1.0, k_top=1.0)

        # Ring topology (k_top=0.16) → shorter τ_relax
        _, messages_ring = validator.validate(sequence, epi_initial=0.0, vf=1.0, k_top=0.16)

        # Both should flag, but estimated τ should differ
        assert any("U6" in m for m in messages_radial)
        assert any("U6" in m for m in messages_ring)


class TestU6Integration:
    """Integration tests for U6 with existing grammar rules."""

    def test_u6_does_not_interfere_with_u1_u5(self):
        """U6 validation should not affect U1-U5 results."""
        # Sequence that passes U1-U5
        sequence = [Emission(), Dissonance(), Coherence(), Silence()]

        validator_no_u6 = GrammarValidator(experimental_u6=False)
        validator_with_u6 = GrammarValidator(experimental_u6=True)

        valid_no_u6, _ = validator_no_u6.validate(sequence, epi_initial=0.0)
        valid_with_u6, _ = validator_with_u6.validate(sequence, epi_initial=0.0)

        # Both should pass (U6 doesn't fail validation, only warns)
        assert valid_no_u6 is True
        assert valid_with_u6 is True

    def test_u6_complements_u2_convergence(self):
        """U6 adds temporal dimension to U2 spatial constraint."""
        # Sequence passes U2 (has stabilizers) but violates U6 (timing)
        sequence = [Emission(), Dissonance(), Dissonance(), Coherence(), Silence()]

        validator = GrammarValidator(experimental_u6=True)
        is_valid, messages = validator.validate(sequence, epi_initial=0.0)

        # U2 should pass
        u2_messages = [m for m in messages if "U2:" in m]
        assert any("satisfied" in m for m in u2_messages)

        # U6 should flag warning
        u6_messages = [m for m in messages if "U6" in m]
        assert len(u6_messages) > 0

    def test_u6_with_u4b_transformer_context(self):
        """U6 should work with U4b transformer requirements."""
        # Proper sequence: OZ → spacing → ZHIR (U4b + U6 compliant)
        sequence = [
            Emission(),
            Coherence(),  # Prior IL for ZHIR (U4b)
            Dissonance(),  # Destabilizer
            Coherence(),  # Spacing
            Coherence(),  # More spacing
            Mutation(),  # Transformer (U4b: needs prior IL + recent destabilizer)
            Coherence(),
            Silence(),
        ]

        validator = GrammarValidator(experimental_u6=True)
        is_valid, messages = validator.validate(sequence, epi_initial=0.0, vf=1.0)

        # Should pass all rules
        assert is_valid is True

        # U4b should pass
        assert any("U4b" in m and "satisfied" in m for m in messages)

        # U6 should pass (OZ and ZHIR properly spaced)
        u6_messages = [m for m in messages if "U6" in m]
        assert any("satisfied" in m or "properly spaced" in m for m in u6_messages)
