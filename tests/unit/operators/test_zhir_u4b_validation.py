"""Comprehensive tests for U4b grammar validation in ZHIR (Mutation).

This module tests the strict U4b validation requirements for the ZHIR operator:

U4b Requirements (AGENTS.md, UNIFIED_GRAMMAR_RULES.md):
1. **Prior IL (Coherence)**: Stable base for transformation
2. **Recent destabilizer**: Threshold energy within ~3 operators

Test Coverage:
- IL precedence requirement (Part 1)
- Destabilizer requirement (Part 2)
- Graduated destabilizer windows (strong/moderate/weak)
- Configuration flags (strict vs soft validation)
- Integration with grammar.py validator
- Error messages and telemetry

References:
- AGENTS.md: §U4b (Transformers Need Context)
- UNIFIED_GRAMMAR_RULES.md: U4b physics derivation
- src/tnfr/operators/preconditions/__init__.py: validate_mutation()
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Emission,
    Coherence,
    Dissonance,
    Mutation,
    Expansion,
    Silence,
    Reception,
)
from tnfr.operators.preconditions import validate_mutation, OperatorPreconditionError


class TestU4bILPrecedence:
    """Test U4b Part 1: ZHIR requires prior IL (Coherence) for stable base."""

    def test_zhir_without_il_passes_without_strict_validation(self):
        """ZHIR without IL should pass when strict validation disabled (default)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Default: VALIDATE_OPERATOR_PRECONDITIONS=False

        # Build history without IL
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["glyph_history"] = []  # Empty history, no IL

        # Should not raise - strict validation disabled
        validate_mutation(G, node)

    def test_zhir_without_il_fails_with_strict_validation(self):
        """ZHIR without prior IL should fail when strict validation enabled."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True  # Enable strict validation

        # Build history without IL
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        from tnfr.types import Glyph

        # Add non-IL operators to history
        G.nodes[node]["glyph_history"] = [Glyph.AL, Glyph.OZ]  # Emission, Dissonance (no Coherence)

        # Should raise error
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        error_msg = str(exc_info.value)
        assert "U4b violation" in error_msg
        assert "prior IL" in error_msg or "Coherence" in error_msg
        assert "stable" in error_msg.lower()

    def test_zhir_with_il_passes_strict_validation(self):
        """ZHIR with prior IL should pass strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Build history WITH IL
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.AL,  # Emission
            Glyph.IL,  # Coherence - REQUIRED
            Glyph.OZ,  # Dissonance
        ]

        # Should not raise
        validate_mutation(G, node)

    def test_zhir_require_il_flag_enforces_without_strict(self):
        """ZHIR_REQUIRE_IL_PRECEDENCE flag enforces IL even without strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Don't enable VALIDATE_OPERATOR_PRECONDITIONS
        G.graph["ZHIR_REQUIRE_IL_PRECEDENCE"] = True  # But enable this specific flag

        # Build history without IL
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.OZ]  # Only Dissonance, no IL

        # Should raise error
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "U4b violation" in str(exc_info.value)
        assert "prior IL" in str(exc_info.value)

    def test_zhir_il_anywhere_in_history_satisfies(self):
        """IL anywhere in history satisfies precedence requirement."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # IL early in history
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.AL,  # Emission
            Glyph.IL,  # Coherence - early
            Glyph.EN,  # Reception
            Glyph.RA,  # Resonance
            Glyph.OZ,  # Dissonance
        ]
        G.nodes[node]["epi_history"] = [0.2, 0.3, 0.4, 0.45, 0.5]

        # Should pass - IL found anywhere before mutation
        validate_mutation(G, node)


class TestU4bDestabilizerRequirement:
    """Test U4b Part 2: ZHIR requires recent destabilizer within ~3 ops."""

    def test_zhir_without_destabilizer_passes_without_strict_validation(self):
        """ZHIR without destabilizer should pass when strict validation disabled."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Default: strict validation off

        # Build history without destabilizer
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.AL, Glyph.IL]  # No destabilizer

        # Should not raise (only warning logged)
        validate_mutation(G, node)

    def test_zhir_without_destabilizer_fails_with_strict_validation(self):
        """ZHIR without recent destabilizer should fail when strict validation enabled."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Build history: IL present (satisfies Part 1) but no destabilizer
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.AL,  # Emission
            Glyph.IL,  # Coherence (satisfies IL requirement)
            Glyph.RA,  # Resonance (not a destabilizer)
        ]
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should raise error
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        error_msg = str(exc_info.value)
        assert "U4b violation" in error_msg
        assert "destabilizer" in error_msg.lower()

    def test_zhir_with_recent_dissonance_passes(self):
        """ZHIR with recent OZ (Dissonance) should pass strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Build valid history: IL + recent OZ
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.AL,  # Emission
            Glyph.IL,  # Coherence (IL precedence)
            Glyph.OZ,  # Dissonance (destabilizer)
        ]
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should pass
        validate_mutation(G, node)

    def test_zhir_with_recent_expansion_passes(self):
        """ZHIR with recent VAL (Expansion) should pass strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Build valid history: IL + recent VAL
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.IL,  # Coherence
            Glyph.VAL,  # Expansion (destabilizer)
        ]
        G.nodes[node]["epi_history"] = [0.4, 0.5]

        # Should pass
        validate_mutation(G, node)

    def test_zhir_require_destabilizer_flag_enforces_without_strict(self):
        """ZHIR_REQUIRE_DESTABILIZER flag enforces destabilizer even without strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["ZHIR_REQUIRE_DESTABILIZER"] = True  # Enable specific flag

        # Build history without destabilizer
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.IL,
            Glyph.SHA,
        ]  # No destabilizer (Silence is not a destabilizer)
        G.nodes[node]["epi_history"] = [0.4, 0.5]

        # Should raise error
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "U4b violation" in str(exc_info.value)
        assert "destabilizer" in str(exc_info.value).lower()


class TestU4bDestabilizerWindows:
    """Test graduated destabilizer windows (strong/moderate/weak)."""

    def test_strong_destabilizer_window_is_4_ops(self):
        """OZ (strong destabilizer) should be valid within 4 ops of ZHIR."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # OZ at distance 4 from mutation (should still be valid for strong)
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.IL,  # 0: Coherence
            Glyph.OZ,  # 1: Dissonance (strong destabilizer)
            Glyph.SHA,  # 2: Silence
            Glyph.SHA,  # 3: Silence
            Glyph.SHA,  # 4: Silence
            # Position 5 will be ZHIR - distance from OZ = 4
        ]
        G.nodes[node]["epi_history"] = [0.2, 0.3, 0.35, 0.40, 0.45, 0.5]

        # Should pass - OZ within window of 4
        validate_mutation(G, node)

    def test_strong_destabilizer_expired_beyond_window(self):
        """OZ beyond window of 4 ops should fail."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # OZ at distance > 4 from mutation
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.IL,  # 0: Coherence
            Glyph.OZ,  # 1: Dissonance
            Glyph.SHA,  # 2: Silence
            Glyph.SHA,  # 3: Silence
            Glyph.SHA,  # 4: Silence
            Glyph.SHA,  # 5: Silence
            Glyph.SHA,  # 6: Silence
            # Position 7 will be ZHIR - distance from OZ = 6 > 4
        ]
        G.nodes[node]["epi_history"] = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        # Should fail - OZ too far
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "destabilizer" in str(exc_info.value).lower()

    def test_moderate_destabilizer_window_is_2_ops(self):
        """VAL (moderate destabilizer) should be valid within 2 ops of ZHIR."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # VAL at distance 2 from mutation
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.IL,  # Coherence
            Glyph.VAL,  # Expansion (moderate destabilizer)
            Glyph.SHA,  # Silence
            # Next will be ZHIR - distance from VAL = 2
        ]
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.45, 0.5]

        # Should pass - VAL within window of 2
        validate_mutation(G, node)


class TestU4bIntegration:
    """Integration tests combining IL precedence and destabilizer requirements."""

    def test_valid_canonical_sequence_il_oz_zhir(self):
        """Canonical sequence IL → OZ → ZHIR should pass all U4b checks."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Perfect canonical sequence
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [
            Glyph.AL,  # Emission
            Glyph.IL,  # Coherence (IL precedence ✓)
            Glyph.OZ,  # Dissonance (recent destabilizer ✓)
        ]
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should pass all checks
        validate_mutation(G, node)

    def test_both_requirements_missing_shows_both_errors(self):
        """Missing both IL and destabilizer should fail with IL error first."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # No IL, no destabilizer
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.AL, Glyph.EN]  # Only non-IL, non-destabilizers
        G.nodes[node]["epi_history"] = [0.4, 0.5]

        # Should fail on IL check first (checked before destabilizer)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        # First error should be about IL
        assert "prior IL" in str(exc_info.value) or "Coherence" in str(exc_info.value)

    def test_run_sequence_with_strict_validation(self):
        """Test full operator sequence with strict validation."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Build initial history
        G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]

        # Valid sequence: IL → OZ → ZHIR
        run_sequence(
            G,
            node,
            [
                Coherence(),  # Provides IL precedence
                Dissonance(),  # Provides destabilizer
                Mutation(),  # Should pass U4b
            ],
        )

        # Should complete without error
        # Verify mutation context was recorded
        assert "_mutation_context" in G.nodes[node]
        context = G.nodes[node]["_mutation_context"]
        assert context["destabilizer_operator"] == "dissonance"

    def test_run_sequence_fails_without_il(self):
        """Sequence without IL should fail when strict validation enabled."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Try invalid sequence: OZ → ZHIR (no IL)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            run_sequence(
                G,
                node,
                [
                    Emission(),  # Not IL
                    Dissonance(),  # Destabilizer present
                    Mutation(),  # Should fail - no IL
                ],
            )

        assert "prior IL" in str(exc_info.value) or "Coherence" in str(exc_info.value)

    def test_run_sequence_fails_without_destabilizer(self):
        """Sequence without recent destabilizer should fail when strict validation enabled."""
        G, node = create_nfr("test", epi=0.4, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Try invalid sequence: IL → ZHIR (no destabilizer)
        with pytest.raises(OperatorPreconditionError) as exc_info:
            run_sequence(
                G,
                node,
                [
                    Coherence(),  # IL present
                    Silence(),  # Not a destabilizer
                    Mutation(),  # Should fail - no destabilizer
                ],
            )

        assert "destabilizer" in str(exc_info.value).lower()


class TestU4bErrorMessages:
    """Test error messages include helpful information."""

    def test_il_error_shows_recent_history(self):
        """IL precedence error should show recent history for debugging."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.AL, Glyph.OZ, Glyph.EN]
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        error_msg = str(exc_info.value)
        # Should include history in error message
        assert "history" in error_msg.lower() or "emission" in error_msg.lower()

    def test_destabilizer_error_shows_recent_history(self):
        """Destabilizer error should show recent history for debugging."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["ZHIR_REQUIRE_DESTABILIZER"] = True

        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.IL, Glyph.RA, Glyph.SHA]  # No destabilizers
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        error_msg = str(exc_info.value)
        # Should include recent history
        assert "history" in error_msg.lower()


class TestU4bBackwardCompatibility:
    """Test backward compatibility - strict validation off by default."""

    def test_default_behavior_is_soft_validation(self):
        """Without VALIDATE_OPERATOR_PRECONDITIONS, U4b checks are soft (warnings only)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Don't set VALIDATE_OPERATOR_PRECONDITIONS (default False)

        # Invalid sequence - but should pass (warnings only)
        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.AL, Glyph.SHA]  # No IL, no destabilizer
        G.nodes[node]["epi_history"] = [0.4, 0.5]

        # Should not raise
        validate_mutation(G, node)

    def test_explicit_flags_work_independently(self):
        """Individual flags (ZHIR_REQUIRE_*) work without global strict validation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Global strict validation OFF
        assert G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False) is False

        # But enable IL requirement specifically
        G.graph["ZHIR_REQUIRE_IL_PRECEDENCE"] = True

        from tnfr.types import Glyph

        G.nodes[node]["glyph_history"] = [Glyph.OZ]  # No IL
        G.nodes[node]["epi_history"] = [0.4, 0.5]

        # Should fail on IL even though global strict validation is off
        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_mutation(G, node)

        assert "prior IL" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
