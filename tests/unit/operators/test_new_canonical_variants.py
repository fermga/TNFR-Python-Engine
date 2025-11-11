"""Unit tests for new canonical sequence variants unlocked by high → zero transition.

This module tests the 4 new canonical patterns that became valid after updating
frequency transitions to allow high → zero (e.g., OZ → SHA, RA → SHA).

These patterns were always theoretically valid per TNFR but were blocked by
overly restrictive frequency rules. Now they can be validated and used.

New Patterns Tested:
1. CONTAINED_CRISIS (OZ → SHA direct) - Crisis containment
2. RESONANCE_PEAK_HOLD (RA → SHA direct) - Peak state preservation
3. MINIMAL_COMPRESSION (NUL → SHA direct) - Compressed latency
4. PHASE_LOCK (ZHIR → SHA direct) - Phase transition hold
"""

from __future__ import annotations

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RESONANCE,
    SILENCE,
)
from tnfr.operators.canonical_patterns import (
    CANONICAL_SEQUENCES,
    CONTAINED_CRISIS,
    MINIMAL_COMPRESSION,
    PHASE_LOCK,
    RESONANCE_PEAK_HOLD,
)
from tnfr.operators.grammar import validate_sequence
from tnfr.types import Glyph
from tnfr.validation import SequenceValidationResult


class TestNewCanonicalVariants:
    """Test that new canonical variants are properly defined and validate."""

    def test_new_variants_in_registry(self):
        """All 4 new variants should be in CANONICAL_SEQUENCES registry."""
        assert "contained_crisis" in CANONICAL_SEQUENCES
        assert "resonance_peak_hold" in CANONICAL_SEQUENCES
        assert "minimal_compression" in CANONICAL_SEQUENCES
        assert "phase_lock" in CANONICAL_SEQUENCES

    def test_contained_crisis_definition(self):
        """CONTAINED_CRISIS should have correct structure."""
        assert CONTAINED_CRISIS.name == "contained_crisis"
        assert CONTAINED_CRISIS.glyphs == [
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            Glyph.OZ,
            Glyph.SHA,
        ]
        assert "crisis" in CONTAINED_CRISIS.description.lower()
        assert "therapeutic" in CONTAINED_CRISIS.domain.lower()

    def test_resonance_peak_hold_definition(self):
        """RESONANCE_PEAK_HOLD should have correct structure."""
        assert RESONANCE_PEAK_HOLD.name == "resonance_peak_hold"
        assert RESONANCE_PEAK_HOLD.glyphs == [
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            Glyph.RA,
            Glyph.SHA,
        ]
        assert "peak" in RESONANCE_PEAK_HOLD.description.lower()
        assert "cognitive" in RESONANCE_PEAK_HOLD.domain.lower()

    def test_minimal_compression_definition(self):
        """MINIMAL_COMPRESSION should have correct structure."""
        assert MINIMAL_COMPRESSION.name == "minimal_compression"
        assert MINIMAL_COMPRESSION.glyphs == [
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            Glyph.NUL,
            Glyph.SHA,
        ]
        assert "compress" in MINIMAL_COMPRESSION.description.lower()

    def test_phase_lock_definition(self):
        """PHASE_LOCK should have correct structure."""
        assert PHASE_LOCK.name == "phase_lock"
        assert PHASE_LOCK.glyphs == [
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            Glyph.OZ,
            Glyph.ZHIR,
            Glyph.SHA,
        ]
        assert "phase" in PHASE_LOCK.description.lower()


class TestContainedCrisisValidation:
    """Test CONTAINED_CRISIS (OZ → SHA direct) sequence validation."""

    def test_contained_crisis_validates(self):
        """CONTAINED_CRISIS sequence should validate successfully."""
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"CONTAINED_CRISIS failed validation: {result.message}"
        assert isinstance(result, SequenceValidationResult)

    def test_oz_sha_direct_transition(self):
        """Direct OZ → SHA transition should be allowed."""
        # This is the key transition that was unlocked
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed
        # Verify it's actually OZ → SHA, not some other pattern
        canonical = result.canonical_tokens
        oz_index = canonical.index(DISSONANCE)
        sha_index = canonical.index(SILENCE)
        assert sha_index == oz_index + 1, "SHA should immediately follow OZ"

    def test_contained_crisis_metadata(self):
        """CONTAINED_CRISIS should have therapeutic use cases."""
        assert len(CONTAINED_CRISIS.use_cases) >= 3
        use_cases_text = " ".join(CONTAINED_CRISIS.use_cases).lower()
        assert any(
            keyword in use_cases_text for keyword in ["trauma", "crisis", "emergency", "protective"]
        )


class TestResonancePeakHoldValidation:
    """Test RESONANCE_PEAK_HOLD (RA → SHA direct) sequence validation."""

    def test_resonance_peak_hold_validates(self):
        """RESONANCE_PEAK_HOLD sequence should validate successfully."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"RESONANCE_PEAK_HOLD failed validation: {result.message}"

    def test_ra_sha_direct_transition(self):
        """Direct RA → SHA transition should be allowed."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed
        canonical = result.canonical_tokens
        ra_index = canonical.index(RESONANCE)
        sha_index = canonical.index(SILENCE)
        assert sha_index == ra_index + 1, "SHA should immediately follow RA"

    def test_resonance_peak_hold_metadata(self):
        """RESONANCE_PEAK_HOLD should have peak/flow use cases."""
        assert len(RESONANCE_PEAK_HOLD.use_cases) >= 3
        use_cases_text = " ".join(RESONANCE_PEAK_HOLD.use_cases).lower()
        assert any(
            keyword in use_cases_text for keyword in ["peak", "flow", "optimal", "coherence"]
        )


class TestMinimalCompressionValidation:
    """Test MINIMAL_COMPRESSION (NUL → SHA direct) sequence validation."""

    def test_minimal_compression_validates(self):
        """MINIMAL_COMPRESSION sequence should validate successfully."""
        sequence = [EMISSION, RECEPTION, COHERENCE, CONTRACTION, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"MINIMAL_COMPRESSION failed validation: {result.message}"

    def test_nul_sha_direct_transition(self):
        """Direct NUL → SHA transition should be allowed."""
        sequence = [EMISSION, RECEPTION, COHERENCE, CONTRACTION, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed
        canonical = result.canonical_tokens
        nul_index = canonical.index(CONTRACTION)
        sha_index = canonical.index(SILENCE)
        assert sha_index == nul_index + 1, "SHA should immediately follow NUL"

    def test_minimal_compression_metadata(self):
        """MINIMAL_COMPRESSION should have compression/storage use cases."""
        assert len(MINIMAL_COMPRESSION.use_cases) >= 3
        use_cases_text = " ".join(MINIMAL_COMPRESSION.use_cases).lower()
        assert any(
            keyword in use_cases_text for keyword in ["compress", "minimal", "storage", "essence"]
        )


class TestPhaseLockValidation:
    """Test PHASE_LOCK (ZHIR → SHA direct) sequence validation."""

    def test_phase_lock_validates(self):
        """PHASE_LOCK sequence should validate successfully."""
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed, f"PHASE_LOCK failed validation: {result.message}"

    def test_zhir_sha_direct_transition(self):
        """Direct ZHIR → SHA transition should be allowed."""
        sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, SILENCE]
        result = validate_sequence(sequence)

        assert result.passed
        canonical = result.canonical_tokens
        zhir_index = canonical.index(MUTATION)
        sha_index = canonical.index(SILENCE)
        assert sha_index == zhir_index + 1, "SHA should immediately follow ZHIR"

    def test_phase_lock_metadata(self):
        """PHASE_LOCK should have transformation/identity use cases."""
        assert len(PHASE_LOCK.use_cases) >= 3
        use_cases_text = " ".join(PHASE_LOCK.use_cases).lower()
        assert any(
            keyword in use_cases_text for keyword in ["phase", "identity", "mutation", "transform"]
        )


class TestNewVariantsVsExistingSequences:
    """Compare new variants with existing elaborate sequences."""

    def test_contained_crisis_vs_bifurcated_base(self):
        """CONTAINED_CRISIS is simpler than BIFURCATED_BASE."""
        # CONTAINED_CRISIS: AL → EN → IL → OZ → SHA (5 operators)
        # BIFURCATED_BASE: AL → EN → IL → OZ → ZHIR → IL → SHA (7 operators)
        assert len(CONTAINED_CRISIS.glyphs) < len(CANONICAL_SEQUENCES["bifurcated_base"].glyphs)
        # Both contain OZ → ... → SHA path
        assert Glyph.OZ in CONTAINED_CRISIS.glyphs
        assert Glyph.SHA in CONTAINED_CRISIS.glyphs

    def test_phase_lock_vs_bifurcated_base(self):
        """PHASE_LOCK is simpler than BIFURCATED_BASE."""
        # PHASE_LOCK: AL → EN → IL → OZ → ZHIR → SHA (6 operators)
        # BIFURCATED_BASE: AL → EN → IL → OZ → ZHIR → IL → SHA (7 operators)
        assert len(PHASE_LOCK.glyphs) < len(CANONICAL_SEQUENCES["bifurcated_base"].glyphs)
        # PHASE_LOCK skips the IL stabilization after ZHIR
        assert PHASE_LOCK.glyphs[-2] == Glyph.ZHIR  # ZHIR before SHA
        bifurcated_base = CANONICAL_SEQUENCES["bifurcated_base"]
        assert bifurcated_base.glyphs[-2] == Glyph.IL  # IL before SHA

    def test_new_variants_complement_existing(self):
        """New variants complement (don't replace) existing sequences."""
        # All 6 original sequences still present
        original_sequences = [
            "bifurcated_base",
            "bifurcated_collapse",
            "therapeutic_protocol",
            "theory_system",
            "full_deployment",
            "mod_stabilizer",
        ]
        for seq_name in original_sequences:
            assert seq_name in CANONICAL_SEQUENCES

        # 4 new variants added
        new_variants = [
            "contained_crisis",
            "resonance_peak_hold",
            "minimal_compression",
            "phase_lock",
        ]
        for seq_name in new_variants:
            assert seq_name in CANONICAL_SEQUENCES

        # Total: 10 sequences
        assert len(CANONICAL_SEQUENCES) == 10


class TestFrequencyTransitionEnablement:
    """Test that new variants depend on high → zero frequency transition."""

    def test_all_new_variants_use_high_to_zero(self):
        """All new variants end with high frequency → SHA (zero)."""
        high_freq_operators = {Glyph.OZ, Glyph.RA, Glyph.NUL, Glyph.ZHIR}

        # CONTAINED_CRISIS: OZ → SHA
        assert CONTAINED_CRISIS.glyphs[-2] in high_freq_operators
        assert CONTAINED_CRISIS.glyphs[-1] == Glyph.SHA

        # RESONANCE_PEAK_HOLD: RA → SHA
        assert RESONANCE_PEAK_HOLD.glyphs[-2] in high_freq_operators
        assert RESONANCE_PEAK_HOLD.glyphs[-1] == Glyph.SHA

        # MINIMAL_COMPRESSION: NUL → SHA
        assert MINIMAL_COMPRESSION.glyphs[-2] in high_freq_operators
        assert MINIMAL_COMPRESSION.glyphs[-1] == Glyph.SHA

        # PHASE_LOCK: ZHIR → SHA
        assert PHASE_LOCK.glyphs[-2] in high_freq_operators
        assert PHASE_LOCK.glyphs[-1] == Glyph.SHA

    def test_variants_reference_frequency_update(self):
        """New variant documentation should reference frequency transition."""
        frequency_keywords = ["frequency", "high", "zero", "transition"]

        for variant in [
            CONTAINED_CRISIS,
            RESONANCE_PEAK_HOLD,
            MINIMAL_COMPRESSION,
            PHASE_LOCK,
        ]:
            ref_text = variant.references.lower()
            # At least one frequency keyword should be in references
            assert any(
                keyword in ref_text for keyword in frequency_keywords
            ), f"{variant.name} should reference frequency transition in references"


class TestCanonicalVariantUseCases:
    """Verify use cases for each new variant are distinct and appropriate."""

    def test_each_variant_has_unique_use_cases(self):
        """Each new variant should have distinct primary use cases."""
        variants = [
            CONTAINED_CRISIS,
            RESONANCE_PEAK_HOLD,
            MINIMAL_COMPRESSION,
            PHASE_LOCK,
        ]

        # Extract primary keywords from use cases
        crisis_keywords = {"trauma", "crisis", "emergency"}
        peak_keywords = {"peak", "flow", "optimal"}
        compress_keywords = {"compress", "minimal", "storage"}
        phase_keywords = {"phase", "identity", "transform"}

        # Verify CONTAINED_CRISIS emphasizes crisis management
        crisis_text = " ".join(CONTAINED_CRISIS.use_cases).lower()
        assert any(k in crisis_text for k in crisis_keywords)

        # Verify RESONANCE_PEAK_HOLD emphasizes peak states
        peak_text = " ".join(RESONANCE_PEAK_HOLD.use_cases).lower()
        assert any(k in peak_text for k in peak_keywords)

        # Verify MINIMAL_COMPRESSION emphasizes compression
        compress_text = " ".join(MINIMAL_COMPRESSION.use_cases).lower()
        assert any(k in compress_text for k in compress_keywords)

        # Verify PHASE_LOCK emphasizes transformation
        phase_text = " ".join(PHASE_LOCK.use_cases).lower()
        assert any(k in phase_text for k in phase_keywords)

    def test_variants_have_sufficient_use_cases(self):
        """Each variant should document multiple use cases."""
        for variant in [
            CONTAINED_CRISIS,
            RESONANCE_PEAK_HOLD,
            MINIMAL_COMPRESSION,
            PHASE_LOCK,
        ]:
            assert len(variant.use_cases) >= 3, f"{variant.name} should have at least 3 use cases"
