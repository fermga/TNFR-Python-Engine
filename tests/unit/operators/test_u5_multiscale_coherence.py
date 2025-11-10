"""Comprehensive tests for U5: Multi-Scale Coherence grammar rule.

Tests the canonical constraint that deep hierarchical structures (REMESH depth>1)
require scale stabilizers to preserve coherence across hierarchical levels.

Physical Basis
--------------
From coherence conservation principle:
    C_total = C_parent + Σ C_child_i = constant
    
For bounded evolution:
    C_parent ≥ α · Σ C_child_i
    
Where α = (1/√N) · η_phase(N) · η_coupling(N) ∈ [0.1, 0.4]

References
----------
- UNIFIED_GRAMMAR_RULES.md: § U5 Multi-Scale Coherence
- Problem statement: "El pulso que nos atraviesa.pdf"
- AGENTS.md: Invariant #7 (Operational Fractality)
"""

import pytest

from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Recursivity,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.unified_grammar import (
    RECURSIVE_GENERATORS,
    SCALE_STABILIZERS,
    UnifiedGrammarValidator,
    validate_unified,
)


class TestU5OperatorSets:
    """Test U5-specific operator set definitions."""

    def test_recursive_generators_set(self):
        """U5: Recursive generators create multi-scale hierarchies.
        
        Physics basis: Only REMESH (Recursivity) creates hierarchical
        nested EPIs that require multi-scale coherence preservation.
        """
        assert RECURSIVE_GENERATORS == frozenset({"recursivity"})

    def test_scale_stabilizers_set(self):
        """U5: Scale stabilizers preserve coherence across hierarchy levels.
        
        Physics basis: IL and THOL provide multi-level stabilization:
        - IL (Coherence): Direct coherence at each hierarchical level
        - THOL (Self-organization): Autopoietic closure across scales
        """
        assert SCALE_STABILIZERS == frozenset({"coherence", "self_organization"})


class TestU5ShallowRecursion:
    """Test that shallow recursion (depth=1) does not trigger U5."""

    def test_shallow_remesh_no_stabilizer_valid(self):
        """Shallow REMESH (depth=1) doesn't require stabilizer.
        
        Physics: Single-level recursion doesn't create multi-scale hierarchy.
        No conservation constraint across levels.
        """
        sequence = [
            Emission(),
            Recursivity(depth=1),  # Shallow recursion
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        
        # Should pass all rules including U5
        assert is_valid, f"Shallow REMESH should pass: {messages}"
        
        # Check U5 specifically
        u5_messages = [msg for msg in messages if msg.startswith("U5:")]
        assert len(u5_messages) == 1
        assert "not applicable" in u5_messages[0].lower()

    def test_default_remesh_no_stabilizer_valid(self):
        """REMESH without explicit depth defaults to depth=1.
        
        Physics: Backward compatibility - existing code assumes shallow recursion.
        """
        sequence = [
            Emission(),
            Recursivity(),  # Default depth=1
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, f"Default REMESH should pass: {messages}"

    def test_shallow_remesh_with_stabilizer_valid(self):
        """Shallow REMESH with stabilizer is valid but not required.
        
        Physics: Stabilizer doesn't hurt, just not required for shallow recursion.
        """
        sequence = [
            Emission(),
            Recursivity(depth=1),
            Coherence(),  # Optional for depth=1
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid


class TestU5DeepRecursionViolations:
    """Test that deep recursion (depth>1) without stabilizers fails U5."""

    def test_deep_remesh_no_stabilizer_fails(self):
        """Deep REMESH without stabilizer violates U5.
        
        Physics: Multi-level hierarchy without stabilization causes
        coherence fragmentation (C_parent < α·ΣC_child).
        
        This is the DECISIVE test case showing U5 independence from U2+U4b:
        - U2: ✓ No destabilizers (trivially convergent)
        - U4b: ✓ REMESH not a transformer (U4b doesn't apply)
        - U5: ✗ Deep recursivity without stabilization → fragmentation
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),  # Deep hierarchy, no stabilizer
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        
        # Should fail specifically on U5
        assert not is_valid, "Deep REMESH without stabilizer should fail U5"
        
        # Check that U5 is the violating rule
        u5_messages = [msg for msg in messages if msg.startswith("U5:")]
        assert len(u5_messages) == 1
        assert "violated" in u5_messages[0].lower()
        assert "depth=3" in u5_messages[0]
        assert "scale stabilizer" in u5_messages[0].lower()

    def test_deep_remesh_depth_2_no_stabilizer_fails(self):
        """Even depth=2 requires stabilizer.
        
        Physics: Any depth>1 creates hierarchy requiring multi-scale stabilization.
        """
        sequence = [
            Emission(),
            Recursivity(depth=2),
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid

    def test_deep_remesh_depth_10_no_stabilizer_fails(self):
        """Very deep hierarchy especially needs stabilization.
        
        Physics: Deeper hierarchy → more levels → stronger conservation requirement.
        α decreases with N (more children → harder to maintain C_parent ≥ α·ΣC_child).
        """
        sequence = [
            Emission(),
            Recursivity(depth=10),
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid


class TestU5DeepRecursionWithStabilizers:
    """Test that deep recursion with stabilizers satisfies U5."""

    def test_deep_remesh_with_coherence_valid(self):
        """Deep REMESH with IL (Coherence) satisfies U5.
        
        Physics: IL provides multi-level coherence stabilization,
        ensuring C_parent ≥ α·ΣC_child at each level.
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),
            Coherence(),  # Scale stabilizer
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, f"Deep REMESH with IL should pass: {messages}"
        
        # Check U5 specifically
        u5_messages = [msg for msg in messages if msg.startswith("U5:")]
        assert len(u5_messages) == 1
        assert "satisfied" in u5_messages[0].lower()

    def test_deep_remesh_with_self_organization_valid(self):
        """Deep REMESH with THOL (Self-organization) satisfies U5.
        
        Physics: THOL provides autopoietic closure across scales,
        creating self-limiting boundaries at each hierarchical level.
        
        Note: THOL is a transformer requiring U4b context (recent destabilizer).
        """
        sequence = [
            Emission(),
            Dissonance(),  # U4b: provides context for THOL transformer
            Recursivity(depth=3),
            SelfOrganization(),  # Scale stabilizer + transformer
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, f"Deep REMESH with THOL should pass: {messages}"

    def test_deep_remesh_stabilizer_before_valid(self):
        """Stabilizer can come before REMESH within window.
        
        Physics: Pre-stabilization prepares coherent structure
        for multi-scale expansion.
        """
        sequence = [
            Emission(),
            Coherence(),  # Pre-stabilization
            Recursivity(depth=3),
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid

    def test_deep_remesh_stabilizer_after_valid(self):
        """Stabilizer can come after REMESH within window.
        
        Physics: Post-stabilization consolidates multi-scale structure.
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),
            Coherence(),  # Post-stabilization
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid


class TestU5StabilizerWindow:
    """Test U5 stabilizer window requirements (±3 operators)."""

    def test_stabilizer_at_window_start_valid(self):
        """Stabilizer at exactly -3 positions from REMESH is valid.
        
        Physics: Within ΔNFR decay timescale (~3 operators).
        """
        sequence = [
            Emission(),
            Coherence(),  # Position -3 from REMESH
            Transition(),
            Transition(),
            Recursivity(depth=3),  # Position 4
            Silence(),
        ]
        
        is_valid, _ = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, "Stabilizer at window start should be valid"

    def test_stabilizer_at_window_end_valid(self):
        """Stabilizer at exactly +3 positions from REMESH is valid.
        
        Physics: Post-stabilization within consolidation window.
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),  # Position 1
            Transition(),
            Transition(),
            Coherence(),  # Position 4 (window_end = min(6, 4+1))
            Silence(),
        ]
        
        is_valid, _ = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, "Stabilizer at window end should be valid"

    def test_stabilizer_outside_window_fails(self):
        """Stabilizer outside ±3 window doesn't satisfy U5.
        
        Physics: Beyond ΔNFR decay timescale - stabilization effect dissipated.
        """
        sequence = [
            Emission(),
            Coherence(),  # Too far before
            Dissonance(),
            Dissonance(),
            Dissonance(),
            Dissonance(),  # More than 3 ops away
            Recursivity(depth=3),
            Silence(),
        ]
        
        is_valid, _ = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid, "Stabilizer outside window should fail"

    def test_multiple_deep_remesh_each_needs_stabilizer(self):
        """Each deep REMESH requires its own stabilizer.
        
        Physics: Each hierarchical expansion needs independent stabilization.
        """
        sequence = [
            Emission(),
            Recursivity(depth=2),
            Coherence(),  # Stabilizes first REMESH
            Transition(),
            Transition(),
            Transition(),
            Transition(),  # Outside window of second REMESH
            Recursivity(depth=2),  # No nearby stabilizer
            Silence(),
        ]
        
        is_valid, _ = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid, "Second REMESH lacks stabilizer"


class TestU5EdgeCases:
    """Test U5 edge cases and boundary conditions."""

    def test_remesh_depth_zero_invalid(self):
        """Depth=0 is invalid (no recursion).
        
        Physics: depth must be >= 1 to have any recursion.
        """
        with pytest.raises(ValueError, match="depth must be >= 1"):
            Recursivity(depth=0)

    def test_remesh_negative_depth_invalid(self):
        """Negative depth is invalid.
        
        Physics: depth is a count of hierarchical levels, must be positive.
        """
        with pytest.raises(ValueError, match="depth must be >= 1"):
            Recursivity(depth=-1)

    def test_empty_sequence_u5_not_applicable(self):
        """Empty sequence doesn't trigger U5.
        
        Physics: No REMESH present.
        """
        sequence = []
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=1.0)
        
        # Will fail on U1b (no closure) but not on U5
        u5_messages = [msg for msg in messages if msg.startswith("U5:")]
        assert len(u5_messages) == 1
        assert "not applicable" in u5_messages[0].lower()

    def test_no_remesh_u5_not_applicable(self):
        """Sequence without REMESH doesn't trigger U5.
        
        Physics: U5 only applies to hierarchical structures.
        """
        sequence = [
            Emission(),
            Coherence(),
            Silence(),
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid
        
        u5_messages = [msg for msg in messages if msg.startswith("U5:")]
        assert "not applicable" in u5_messages[0].lower()


class TestU5PhysicalBasis:
    """Test that U5 captures physical constraints not covered by U2+U4b."""

    def test_u5_independence_from_u2(self):
        """U5 operates independently of U2 (convergence).
        
        Physics: U2 is TEMPORAL (operator sequences in time).
        U5 is SPATIAL (hierarchical nesting in structure).
        
        This sequence has no destabilizers (U2 trivially satisfied)
        but violates U5 (spatial hierarchy requirement).
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),  # Deep hierarchy
            Silence(),
        ]
        
        # Check U2 separately
        valid_u2, msg_u2 = UnifiedGrammarValidator.validate_convergence(sequence)
        assert valid_u2, "U2 should pass (no destabilizers)"
        
        # Check U5 separately
        valid_u5, msg_u5 = UnifiedGrammarValidator.validate_multiscale_coherence(sequence)
        assert not valid_u5, "U5 should fail (no scale stabilizer)"
        
        # Full validation should fail on U5, not U2
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid
        
        u2_msg = [m for m in messages if m.startswith("U2:")][0]
        u5_msg = [m for m in messages if m.startswith("U5:")][0]
        
        assert "not applicable" in u2_msg.lower() or "satisfied" in u2_msg.lower()
        assert "violated" in u5_msg.lower()

    def test_u5_independence_from_u4b(self):
        """U5 operates independently of U4b (transformer context).
        
        Physics: U4b requires transformers (ZHIR, THOL) to have context.
        REMESH is not a transformer - it's a generator/closure.
        
        This sequence has no transformers (U4b not applicable)
        but violates U5 (spatial hierarchy requirement).
        """
        sequence = [
            Emission(),
            Recursivity(depth=3),  # Not a transformer
            Silence(),
        ]
        
        # Check U4b separately
        valid_u4b, msg_u4b = UnifiedGrammarValidator.validate_transformer_context(sequence)
        assert valid_u4b, "U4b should pass (no transformers)"
        
        # Check U5 separately
        valid_u5, msg_u5 = UnifiedGrammarValidator.validate_multiscale_coherence(sequence)
        assert not valid_u5, "U5 should fail (no scale stabilizer)"

    def test_u5_complements_u2_remesh(self):
        """U5 is related but distinct from U2-REMESH.
        
        Physics:
        - U2-REMESH: REMESH amplifies destabilizers → need stabilizers
        - U5: Deep REMESH creates hierarchy → need scale stabilizers
        
        Both can apply simultaneously but address different physics.
        """
        # Sequence that triggers both U2-REMESH and U5
        sequence = [
            Emission(),
            Recursivity(depth=3),  # Deep hierarchy (U5)
            Dissonance(),  # Destabilizer (U2-REMESH)
            Silence(),
        ]
        
        # Should fail both rules
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert not is_valid
        
        # Both U2-REMESH and U5 should be violated
        u2_remesh_msg = [m for m in messages if "U2-REMESH" in m][0]
        u5_msg = [m for m in messages if m.startswith("U5:")][0]
        
        assert "violated" in u2_remesh_msg.lower()
        assert "violated" in u5_msg.lower()
        
        # Add stabilizer fixes both
        sequence_fixed = [
            Emission(),
            Recursivity(depth=3),
            Dissonance(),
            Coherence(),  # Fixes both U2-REMESH and U5
            Silence(),
        ]
        
        is_valid_fixed, _ = UnifiedGrammarValidator.validate(sequence_fixed, epi_initial=0.0)
        assert is_valid_fixed


class TestU5Integration:
    """Test U5 integration with complete grammar validation."""

    def test_validate_unified_includes_u5(self):
        """validate_unified() checks U5 rule.
        
        Ensures U5 is integrated into unified grammar pipeline.
        """
        # Should fail U5
        sequence_fail = [
            Emission(),
            Recursivity(depth=3),
            Silence(),
        ]
        
        assert not validate_unified(sequence_fail, epi_initial=0.0)
        
        # Should pass U5
        sequence_pass = [
            Emission(),
            Recursivity(depth=3),
            Coherence(),
            Silence(),
        ]
        
        assert validate_unified(sequence_pass, epi_initial=0.0)

    def test_u5_with_all_other_rules(self):
        """U5 works correctly alongside U1-U4.
        
        Physics: U5 is additive - doesn't interfere with existing rules.
        """
        # Complex sequence that tests multiple rules
        sequence = [
            Emission(),  # U1a: Generator
            Dissonance(),  # U2: Destabilizer, U4a: Trigger
            Recursivity(depth=3),  # U5: Deep recursion
            Coherence(),  # U2: Stabilizer, U4a: Handler, U5: Scale stabilizer
            Silence(),  # U1b: Closure
        ]
        
        is_valid, messages = UnifiedGrammarValidator.validate(sequence, epi_initial=0.0)
        assert is_valid, f"Complex sequence should pass all rules: {messages}"
        
        # All rules should be satisfied
        for rule in ["U1a:", "U1b:", "U2:", "U3:", "U4a:", "U4b:", "U5:"]:
            rule_msgs = [m for m in messages if m.startswith(rule)]
            assert len(rule_msgs) == 1
            msg = rule_msgs[0].lower()
            assert "satisfied" in msg or "not applicable" in msg
