"""Tests for canonical IL (Coherence) sequence recognition and validation.

This module tests the canonical glyph sequences involving the IL (Coherence)
operator as defined in the TNFR grammar system. Tests verify:

1. Canonical sequence recognition (AL→IL, EN→IL, OZ→IL, RA→IL, IL→ZHIR)
2. Anti-pattern detection (IL→SHA, IL→IL, SHA→IL)
3. Grammar compliance of all sequences
4. Sequence suggestion logic
5. Integration with apply_glyph_with_grammar

All sequences are validated against Grammar 2.0 rules (R1-R5) to ensure
structural coherence and compatibility.
"""

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
)
from tnfr.operators.grammar import (
    CANONICAL_IL_SEQUENCES,
    IL_ANTIPATTERNS,
    recognize_il_sequences,
    optimize_il_sequence,
    suggest_il_sequence,
)
from tnfr.types import Glyph


class TestCanonicalILSequences:
    """Test canonical coherence sequence definitions and metadata."""

    def test_canonical_sequences_defined(self):
        """Canonical coherence sequences are properly defined."""
        assert "EMISSION_COHERENCE" in CANONICAL_IL_SEQUENCES
        assert "RECEPTION_COHERENCE" in CANONICAL_IL_SEQUENCES
        assert "DISSONANCE_COHERENCE" in CANONICAL_IL_SEQUENCES
        assert "RESONANCE_COHERENCE" in CANONICAL_IL_SEQUENCES
        assert "COHERENCE_MUTATION" in CANONICAL_IL_SEQUENCES

    def test_emission_coherence_safe_activation(self):
        """emission → coherence (safe_activation) sequence has correct metadata."""
        seq = CANONICAL_IL_SEQUENCES["EMISSION_COHERENCE"]
        assert seq["name"] == "safe_activation"
        assert seq["pattern"] == [EMISSION, COHERENCE]
        assert seq["glyphs"] == [Glyph.AL, Glyph.IL]
        assert seq["optimization"] == "can_fuse"
        assert "Emission stabilized" in seq["description"]

    def test_reception_coherence_stable_integration(self):
        """reception → coherence (stable_integration) sequence has correct metadata."""
        seq = CANONICAL_IL_SEQUENCES["RECEPTION_COHERENCE"]
        assert seq["name"] == "stable_integration"
        assert seq["pattern"] == [RECEPTION, COHERENCE]
        assert seq["glyphs"] == [Glyph.EN, Glyph.IL]
        assert seq["optimization"] == "can_fuse"
        assert "Reception consolidated" in seq["description"]

    def test_dissonance_coherence_creative_resolution(self):
        """dissonance → coherence (creative_resolution) sequence has correct metadata."""
        seq = CANONICAL_IL_SEQUENCES["DISSONANCE_COHERENCE"]
        assert seq["name"] == "creative_resolution"
        assert seq["pattern"] == [DISSONANCE, COHERENCE]
        assert seq["glyphs"] == [Glyph.OZ, Glyph.IL]
        assert seq["optimization"] == "preserve"
        assert "Dissonance resolved" in seq["description"]

    def test_resonance_coherence_resonance_consolidation(self):
        """resonance → coherence (resonance_consolidation) sequence has correct metadata."""
        seq = CANONICAL_IL_SEQUENCES["RESONANCE_COHERENCE"]
        assert seq["name"] == "resonance_consolidation"
        assert seq["pattern"] == [RESONANCE, COHERENCE]
        assert seq["glyphs"] == [Glyph.RA, Glyph.IL]
        assert seq["optimization"] == "preserve"
        assert "Propagated coherence locked" in seq["description"]

    def test_coherence_mutation_stable_transformation(self):
        """coherence → mutation (stable_transformation) sequence has correct metadata."""
        seq = CANONICAL_IL_SEQUENCES["COHERENCE_MUTATION"]
        assert seq["name"] == "stable_transformation"
        assert seq["pattern"] == [COHERENCE, MUTATION]
        assert seq["glyphs"] == [Glyph.IL, Glyph.ZHIR]
        assert seq["optimization"] == "preserve"
        assert "phase transformation" in seq["structural_effect"].lower()


class TestILAntipatterns:
    """Test coherence anti-pattern definitions."""

    def test_antipatterns_defined(self):
        """Coherence anti-patterns are properly defined."""
        assert "COHERENCE_SILENCE" in IL_ANTIPATTERNS
        assert "COHERENCE_COHERENCE" in IL_ANTIPATTERNS
        assert "SILENCE_COHERENCE" in IL_ANTIPATTERNS

    def test_coherence_silence_info_severity(self):
        """coherence → silence anti-pattern has info severity (valid but redundant)."""
        anti = IL_ANTIPATTERNS["COHERENCE_SILENCE"]
        assert anti["severity"] == "info"
        assert anti["alternative"] is None  # Valid sequence, just informational
        assert "redundant" in anti["warning"].lower()

    def test_coherence_coherence_warning_severity(self):
        """coherence → coherence anti-pattern has warning severity (no structural purpose)."""
        anti = IL_ANTIPATTERNS["COHERENCE_COHERENCE"]
        assert anti["severity"] == "warning"
        assert anti["alternative"] is None
        assert "repeated" in anti["warning"].lower()

    def test_silence_coherence_error_severity(self):
        """silence → coherence anti-pattern has error severity with reformulated alternative."""
        anti = IL_ANTIPATTERNS["SILENCE_COHERENCE"]
        assert anti["severity"] == "error"
        assert anti["alternative"] == [SILENCE, EMISSION, COHERENCE]
        assert anti["alternative_glyphs"] == [Glyph.SHA, Glyph.AL, Glyph.IL]
        assert "silence → emission → coherence" in anti["warning"].lower()


class TestRecognizeILSequences:
    """Test recognize_il_sequences() function."""

    def test_recognize_al_il(self):
        """Recognize AL → IL canonical sequence."""
        seq = [Glyph.AL, Glyph.IL]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "safe_activation"
        assert recognized[0]["position"] == 0
        assert not recognized[0]["is_antipattern"]

    def test_recognize_en_il(self):
        """Recognize EN → IL canonical sequence."""
        seq = [Glyph.EN, Glyph.IL]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "stable_integration"

    def test_recognize_oz_il(self):
        """Recognize OZ → IL canonical sequence."""
        seq = [Glyph.OZ, Glyph.IL]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "creative_resolution"

    def test_recognize_ra_il(self):
        """Recognize RA → IL canonical sequence."""
        seq = [Glyph.RA, Glyph.IL]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "resonance_consolidation"

    def test_recognize_il_zhir(self):
        """Recognize IL → ZHIR canonical sequence."""
        seq = [Glyph.IL, Glyph.ZHIR]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "stable_transformation"

    def test_recognize_multiple_sequences(self):
        """Recognize multiple IL sequences in longer sequence."""
        seq = [Glyph.AL, Glyph.IL, Glyph.ZHIR]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 2
        assert recognized[0]["pattern_name"] == "safe_activation"
        assert recognized[0]["position"] == 0
        assert recognized[1]["pattern_name"] == "stable_transformation"
        assert recognized[1]["position"] == 1

    def test_recognize_il_sha_antipattern(self):
        """Recognize coherence → silence anti-pattern (info severity, no warning emitted)."""
        seq = [Glyph.IL, Glyph.SHA]
        
        # Info severity doesn't emit warnings automatically
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "coherence_silence_info"
        assert recognized[0]["is_antipattern"]
        assert recognized[0]["severity"] == "info"

    def test_recognize_il_il_antipattern(self):
        """Recognize coherence → coherence anti-pattern with warning."""
        seq = [Glyph.IL, Glyph.IL]
        
        with pytest.warns(UserWarning, match="Anti-pattern detected.*coherence"):
            recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "coherence_coherence_antipattern"
        assert recognized[0]["is_antipattern"]
        assert recognized[0]["severity"] == "warning"

    def test_recognize_sha_il_antipattern(self):
        """Recognize silence → coherence anti-pattern with error warning."""
        seq = [Glyph.SHA, Glyph.IL]
        
        with pytest.warns(UserWarning, match="Anti-pattern detected.*silence"):
            recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "silence_coherence_antipattern"
        assert recognized[0]["is_antipattern"]
        assert recognized[0]["severity"] == "error"
        assert recognized[0]["alternative"] == [SILENCE, EMISSION, COHERENCE]

    def test_recognize_with_string_names(self):
        """Recognize sequences from operator names (strings)."""
        seq = ["emission", "coherence"]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 1
        assert recognized[0]["pattern_name"] == "safe_activation"

    def test_recognize_no_match(self):
        """Return empty list when no IL sequences found."""
        seq = [Glyph.AL, Glyph.EN, Glyph.RA]
        recognized = recognize_il_sequences(seq)
        
        assert len(recognized) == 0


class TestOptimizeILSequence:
    """Test optimize_il_sequence() function."""

    def test_optimize_returns_original(self):
        """Currently returns original sequence (fusion not yet implemented)."""
        seq = [Glyph.AL, Glyph.IL, Glyph.SHA]
        optimized = optimize_il_sequence(seq)
        
        assert optimized == seq

    def test_optimize_with_fusion_disabled(self):
        """With fusion disabled, returns original sequence."""
        seq = [Glyph.AL, Glyph.IL]
        optimized = optimize_il_sequence(seq, allow_fusion=False)
        
        assert optimized == seq

    def test_optimize_recognizes_patterns(self):
        """Optimization calls recognize_il_sequences internally."""
        seq = [Glyph.AL, Glyph.IL]
        # Should not raise even if patterns are recognized
        optimized = optimize_il_sequence(seq)
        assert optimized is not None


class TestSuggestILSequence:
    """Test suggest_il_sequence() function."""

    def test_suggest_inactive_node_activation(self):
        """Suggest AL → IL for inactive node requiring stability."""
        current = {"epi": 0.05, "dnfr": 0.0, "vf": 0.85}
        goal = {"dnfr_target": "low", "consolidate": True}
        
        suggested = suggest_il_sequence(current, goal)
        
        assert suggested == [EMISSION, COHERENCE]

    def test_suggest_high_dnfr_resolution(self):
        """Suggest OZ → IL for high ΔNFR requiring reduction."""
        current = {"epi": 0.5, "dnfr": 0.85, "vf": 1.0}
        goal = {"dnfr_target": "low"}
        
        suggested = suggest_il_sequence(current, goal)
        
        assert DISSONANCE in suggested
        assert COHERENCE in suggested

    def test_suggest_moderate_dnfr_direct_coherence(self):
        """Suggest direct IL for moderate ΔNFR."""
        current = {"epi": 0.4, "dnfr": 0.4, "vf": 0.95}
        goal = {"dnfr_target": "low"}
        
        suggested = suggest_il_sequence(current, goal)
        
        assert suggested == [COHERENCE]

    def test_suggest_phase_transformation(self):
        """Suggest IL → ZHIR for phase change goal."""
        current = {"epi": 0.6, "dnfr": 0.3, "vf": 1.1}
        goal = {"phase_change": True}
        
        suggested = suggest_il_sequence(current, goal)
        
        assert COHERENCE in suggested
        assert MUTATION in suggested
        assert suggested[-2:] == [COHERENCE, MUTATION]

    def test_suggest_reactivation_from_silence(self):
        """Suggest AL → IL instead of SHA → IL anti-pattern."""
        current = {"epi": 0.0, "dnfr": 0.0, "vf": 0.02}  # Silenced state
        goal = {"reactivate": True, "consolidate": True}
        
        suggested = suggest_il_sequence(current, goal)
        
        # Should suggest AL → IL, not SHA → IL
        assert suggested == [EMISSION, COHERENCE]

    def test_suggest_consolidation_only(self):
        """Suggest simple IL for consolidation goal."""
        current = {"epi": 0.5, "dnfr": 0.2, "vf": 1.0}
        goal = {"consolidate": True}
        
        suggested = suggest_il_sequence(current, goal)
        
        assert suggested == [COHERENCE]


class TestGrammarCompliance:
    """Test that canonical coherence sequences comply with compatibility rules."""

    def test_emission_coherence_compatible(self):
        """emission → coherence has EXCELLENT compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(EMISSION, COHERENCE)
        assert level == CompatibilityLevel.EXCELLENT

    def test_reception_coherence_compatible(self):
        """reception → coherence has EXCELLENT compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(RECEPTION, COHERENCE)
        assert level == CompatibilityLevel.EXCELLENT

    def test_dissonance_coherence_compatible(self):
        """dissonance → coherence has GOOD compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(DISSONANCE, COHERENCE)
        assert level == CompatibilityLevel.GOOD

    def test_resonance_coherence_compatible(self):
        """resonance → coherence has EXCELLENT compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(RESONANCE, COHERENCE)
        assert level == CompatibilityLevel.EXCELLENT

    def test_coherence_mutation_caution(self):
        """coherence → mutation has CAUTION compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(COHERENCE, MUTATION)
        assert level == CompatibilityLevel.CAUTION

    def test_coherence_coherence_avoid(self):
        """coherence → coherence has AVOID compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(COHERENCE, COHERENCE)
        assert level == CompatibilityLevel.AVOID

    def test_silence_coherence_avoid(self):
        """silence → coherence has AVOID compatibility."""
        from tnfr.validation.compatibility import get_compatibility_level, CompatibilityLevel
        
        level = get_compatibility_level(SILENCE, COHERENCE)
        assert level == CompatibilityLevel.AVOID


class TestIntegrationWithGrammar:
    """Test integration with apply_glyph_with_grammar."""

    def test_recognition_in_apply_glyph(self):
        """Coherence sequences are recognized during apply_glyph_with_grammar."""
        import networkx as nx
        from tnfr.operators.grammar import apply_glyph_with_grammar
        from tnfr.constants import inject_defaults
        
        G = nx.DiGraph()
        inject_defaults(G)
        G.add_node("test_node", epi=0.3, vf=1.0, dnfr=0.1, theta=0.0)
        G.nodes["test_node"]["glyph_history"] = [Glyph.AL]
        
        # Apply coherence to form emission → coherence sequence
        apply_glyph_with_grammar(G, ["test_node"], Glyph.IL)
        
        # Check that sequence was recognized and logged
        assert "recognized_coherence_patterns" in G.graph
        patterns = G.graph["recognized_coherence_patterns"]
        assert len(patterns) >= 1
        assert any(p["pattern_name"] == "safe_activation" for p in patterns)

    def test_antipattern_warning_in_apply_glyph(self):
        """Anti-patterns generate warnings during apply_glyph_with_grammar."""
        import networkx as nx
        from tnfr.operators.grammar import apply_glyph_with_grammar
        from tnfr.constants import inject_defaults
        
        G = nx.DiGraph()
        inject_defaults(G)
        G.add_node("test_node", epi=0.5, vf=1.0, dnfr=0.1, theta=0.0)
        G.nodes["test_node"]["glyph_history"] = [Glyph.IL]
        
        # Apply coherence again to form coherence → coherence anti-pattern
        with pytest.warns(UserWarning, match="Anti-pattern detected.*coherence"):
            apply_glyph_with_grammar(G, ["test_node"], Glyph.IL)
        
        # Pattern should still be logged
        assert "recognized_coherence_patterns" in G.graph
        patterns = G.graph["recognized_coherence_patterns"]
        assert any(p.get("is_antipattern", False) for p in patterns)
