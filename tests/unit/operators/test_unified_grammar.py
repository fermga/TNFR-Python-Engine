"""Comprehensive tests for unified TNFR grammar.

Tests all U1-U5 constraints (temporal + multi-scale) and their physics basis.

References:
- UNIFIED_GRAMMAR_RULES.md: Complete derivations
- unified_grammar.py: Implementation
- AGENTS.md: Canonical invariants and formal contracts
"""

import pytest

from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.operators.unified_grammar import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
    UnifiedGrammarValidator,
    validate_unified,
)
from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD


class TestOperatorSets:
    """Test canonical operator set definitions.

    Verifies that operator sets match UNIFIED_GRAMMAR_RULES.md specifications
    and derive correctly from TNFR physics.
    """

    def test_generators_set(self):
        """U1a: Generators create EPI from null/dormant states.

        Physics basis: Only operators that can generate structure from EPI=0.
        - AL (Emission): Generates EPI from vacuum via emission
        - NAV (Transition): Activates latent EPI through regime transition
        - REMESH (Recursivity): Echoes dormant structure across scales
        """
        assert GENERATORS == frozenset({"emission", "transition", "recursivity"})

    def test_closures_set(self):
        """U1b: Closures leave system in coherent attractor states.

        Physics basis: Terminal states that preserve coherence.
        - SHA (Silence): Terminal closure - freezes evolution (νf → 0)
        - NAV (Transition): Handoff closure - transfers to next regime
        - REMESH (Recursivity): Recursive closure - distributes across scales
        - OZ (Dissonance): Intentional closure - preserves activation/tension
        """
        assert CLOSURES == frozenset({"silence", "transition", "recursivity", "dissonance"})

    def test_stabilizers_set(self):
        """U2: Stabilizers provide negative feedback.

        Physics basis: Reduce |ΔNFR|, ensure integral convergence.
        - IL (Coherence): Direct coherence restoration
        - THOL (Self-organization): Autopoietic closure, self-limiting boundaries
        - EN (Reception): Integration reduces ΔNFR through structural stabilization
        """
        assert STABILIZERS == frozenset({"coherence", "self_organization", "reception"})

    def test_destabilizers_set(self):
        """U2: Destabilizers increase |ΔNFR| (positive feedback).

        Physics basis: May cause ∫νf·ΔNFR dt divergence without stabilizers.
        - OZ (Dissonance): Explicit dissonance
        - ZHIR (Mutation): Phase transformation
        - VAL (Expansion): Increases structural complexity
        - NUL (Contraction): Densification increases ΔNFR
        """
        assert DESTABILIZERS == frozenset({"dissonance", "mutation", "expansion", "contraction"})

    def test_coupling_resonance_set(self):
        """U3: Ops requiring phase verification per Invariant #5.

        Physics basis: Resonance requires phase compatibility.
        |φᵢ - φⱼ| ≤ Δφ_max required for coupling/resonance.
        """
        assert COUPLING_RESONANCE == frozenset({"coupling", "resonance"})

    def test_bifurcation_triggers_set(self):
        """U4a: Ops that may trigger phase transitions.

        Physics basis: If ∂²EPI/∂t² > τ, bifurcation may occur.
        - OZ (Dissonance): May trigger bifurcation
        - ZHIR (Mutation): Phase transformation operator
        """
        assert BIFURCATION_TRIGGERS == frozenset({"dissonance", "mutation"})

    def test_bifurcation_handlers_set(self):
        """U4a: Ops that manage bifurcation dynamics.

        Physics basis: Manage reorganization when threshold crossed.
        - THOL (Self-organization): Manages bifurcation through emergence
        - IL (Coherence): Stabilizes post-bifurcation state
        """
        assert BIFURCATION_HANDLERS == frozenset({"self_organization", "coherence"})

    def test_transformers_set(self):
        """U4b: Ops that execute structural bifurcations.

        Physics basis: Require recent destabilizer for threshold energy.
        - ZHIR (Mutation): Phase change θ → θ'
        - THOL (Self-organization): Creates sub-EPIs (operational fractality)
        """
        assert TRANSFORMERS == frozenset({"mutation", "self_organization"})


class TestU1aInitiation:
    """Test U1a: Structural initiation constraint.

    Physics: If EPI=0, then ∂EPI/∂t undefined.
    Must use generator to create initial structure.

    Derivation from UNIFIED_GRAMMAR_RULES.md:
        If EPI₀ = 0:
          ∂EPI/∂t|_{EPI=0} = undefined (no gradient on empty space)
          → System CANNOT evolve
          → MUST use generator to create initial structure
    """

    def test_epi_zero_requires_generator(self):
        """EPI=0 must start with generator."""
        seq = [Emission(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_initiation(seq, epi_initial=0.0)
        assert valid
        assert "starts with generator" in msg.lower()

    def test_epi_zero_non_generator_fails(self):
        """EPI=0 with non-generator start should fail."""
        seq = [Coherence(), Emission(), Silence()]  # Coherence not a generator
        valid, msg = UnifiedGrammarValidator.validate_initiation(seq, epi_initial=0.0)
        assert not valid
        assert "U1a violated" in msg

    def test_epi_nonzero_no_generator_needed(self):
        """EPI>0 doesn't require generator."""
        seq = [Coherence(), Resonance(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_initiation(seq, epi_initial=1.0)
        assert valid
        assert "initiation not required" in msg.lower()

    @pytest.mark.parametrize(
        "generator_op",
        [
            Emission(),  # AL
            Transition(),  # NAV
            Recursivity(),  # REMESH
        ],
    )
    def test_all_generators_valid_for_epi_zero(self, generator_op):
        """All three generators valid for EPI=0 initiation."""
        seq = [generator_op, Coherence(), Silence()]
        valid, _ = UnifiedGrammarValidator.validate_initiation(seq, epi_initial=0.0)
        assert valid


class TestU1bClosure:
    """Test U1b: Structural closure constraint.

    Physics: Sequences are bounded action potentials.
    Must end in coherent attractor states.

    Physical interpretation: Like physical waves, sequences must have
    emission source AND absorption/termination that leaves system
    in stable attractor states.
    """

    def test_sequence_must_have_closure(self):
        """Sequence must end with closure operator."""
        seq = [Emission(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_closure(seq)
        assert valid
        assert "ends with closure" in msg.lower()

    def test_non_closure_end_fails(self):
        """Ending with non-closure should fail."""
        seq = [Emission(), Coherence(), Expansion()]  # Expansion not a closure
        valid, msg = UnifiedGrammarValidator.validate_closure(seq)
        assert not valid
        assert "U1b violated" in msg

    @pytest.mark.parametrize(
        "closure_op",
        [
            Silence(),  # SHA - Terminal closure
            Transition(),  # NAV - Handoff closure
            Recursivity(),  # REMESH - Recursive closure
            Dissonance(),  # OZ - Intentional closure
        ],
    )
    def test_all_closures_valid(self, closure_op):
        """All four closures valid for sequence termination."""
        seq = [Emission(), Coherence(), closure_op]
        valid, _ = UnifiedGrammarValidator.validate_closure(seq)
        assert valid

    def test_empty_sequence_fails_closure(self):
        """Empty sequence has no closure."""
        seq = []
        valid, msg = UnifiedGrammarValidator.validate_closure(seq)
        assert not valid
        assert "U1b violated" in msg


class TestU2Convergence:
    """Test U2: Convergence and boundedness constraint.

    Physics: ∫νf·ΔNFR dt must converge.
    Destabilizers without stabilizers → divergence.

    From integrated nodal equation:
        EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf(τ) · ΔNFR(τ) dτ

    Without stabilizers: ΔNFR ~ e^(λt) → integral diverges
    With stabilizers: ΔNFR bounded → integral converges
    """

    def test_destabilizers_require_stabilizers(self):
        """Destabilizers must be accompanied by stabilizers."""
        seq = [Emission(), Dissonance(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_convergence(seq)
        assert valid
        assert "stabilizers" in msg.lower()

    def test_destabilizers_without_stabilizers_fail(self):
        """Destabilizers without stabilizers should fail."""
        seq = [Emission(), Dissonance(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_convergence(seq)
        assert not valid
        assert "U2 violated" in msg
        assert "may diverge" in msg

    def test_no_destabilizers_passes(self):
        """No destabilizers → U2 not applicable."""
        seq = [Emission(), Coherence(), Resonance(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_convergence(seq)
        assert valid
        assert "not applicable" in msg

    @pytest.mark.parametrize(
        "destabilizer_op,stabilizer_op",
        [
            (Dissonance(), Coherence()),
            (Mutation(), SelfOrganization()),
            (Expansion(), Coherence()),
            (Dissonance(), SelfOrganization()),
            (Mutation(), Coherence()),
            (Expansion(), SelfOrganization()),
        ],
    )
    def test_destabilizer_stabilizer_pairs(self, destabilizer_op, stabilizer_op):
        """Each destabilizer can be bounded by stabilizers."""
        seq = [Emission(), destabilizer_op, stabilizer_op, Silence()]
        valid, _ = UnifiedGrammarValidator.validate_convergence(seq)
        assert valid

    def test_multiple_destabilizers_need_stabilizer(self):
        """Multiple destabilizers still require at least one stabilizer."""
        seq = [Emission(), Dissonance(), Expansion(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_convergence(seq)
        assert valid

    def test_multiple_destabilizers_without_stabilizer_fail(self):
        """Multiple destabilizers without stabilizer should fail."""
        seq = [Emission(), Dissonance(), Expansion(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_convergence(seq)
        assert not valid
        assert "U2 violated" in msg


class TestU3ResonantCoupling:
    """Test U3: Resonant coupling constraint.

    Physics: Invariant #5 - phase verification mandatory.
    |φᵢ - φⱼ| ≤ Δφ_max required for coupling/resonance.

    Resonance physics: Two oscillators resonate ⟺ phases compatible.
    Without phase verification: Antiphase nodes attempt coupling
    → Destructive interference → Non-physical.
    """

    def test_coupling_requires_phase_awareness(self):
        """Coupling operators must have phase verification awareness."""
        seq = [Emission(), Coupling(), Resonance(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_resonant_coupling(seq)
        assert valid
        assert "phase verification" in msg.lower()
        # Check for Invariant #5 reference
        assert "MANDATORY" in msg or "Invariant" in msg

    def test_no_coupling_not_applicable(self):
        """U3 not applicable without coupling/resonance."""
        seq = [Emission(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_resonant_coupling(seq)
        assert valid
        assert "not applicable" in msg

    @pytest.mark.parametrize(
        "coupling_op",
        [
            Coupling(),  # UM
            Resonance(),  # RA
        ],
    )
    def test_coupling_resonance_ops_trigger_u3(self, coupling_op):
        """Both coupling and resonance trigger U3 awareness."""
        seq = [Emission(), coupling_op, Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_resonant_coupling(seq)
        assert valid
        assert "phase verification" in msg.lower()

    def test_multiple_coupling_ops_trigger_u3(self):
        """Multiple coupling/resonance ops still trigger U3."""
        seq = [Emission(), Coupling(), Resonance(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_resonant_coupling(seq)
        assert valid
        assert "phase verification" in msg.lower()


class TestU4aBifurcationTriggers:
    """Test U4a: Bifurcation triggers need handlers.

    Physics: If ∂²EPI/∂t² > τ, bifurcation may occur.
    Requires handlers (THOL or IL) for stable transition.

    Contract OZ: Dissonance may trigger bifurcation if threshold crossed.
    Handlers manage reorganization when system enters bifurcation regime.
    """

    def test_triggers_require_handlers(self):
        """Bifurcation triggers must have handlers."""
        seq = [Emission(), Dissonance(), SelfOrganization(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_bifurcation_triggers(seq)
        assert valid
        assert "handlers" in msg.lower()

    def test_triggers_without_handlers_fail(self):
        """Triggers without handlers should fail."""
        seq = [Emission(), Dissonance(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_bifurcation_triggers(seq)
        assert not valid
        assert "U4a violated" in msg
        assert "unmanaged" in msg

    @pytest.mark.parametrize(
        "trigger_op,handler_op",
        [
            (Dissonance(), Coherence()),
            (Dissonance(), SelfOrganization()),
            (Mutation(), Coherence()),
            (Mutation(), SelfOrganization()),
        ],
    )
    def test_trigger_handler_pairs(self, trigger_op, handler_op):
        """Each trigger can be managed by handlers."""
        seq = [Emission(), trigger_op, handler_op, Silence()]
        valid, _ = UnifiedGrammarValidator.validate_bifurcation_triggers(seq)
        assert valid

    def test_no_triggers_not_applicable(self):
        """U4a not applicable without bifurcation triggers."""
        seq = [Emission(), Coherence(), Resonance(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_bifurcation_triggers(seq)
        assert valid
        assert "not applicable" in msg

    def test_multiple_triggers_need_handler(self):
        """Multiple triggers still require at least one handler."""
        seq = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_bifurcation_triggers(seq)
        assert valid


class TestU4bTransformerContext:
    """Test U4b: Transformers need context.

    Physics: Bifurcations require threshold energy.
    Transformers need recent destabilizer (~3 ops) for sufficient |ΔNFR|.
    ZHIR additionally needs prior IL for stable base.

    "Recent" = within ~3 operators (ΔNFR decays via structural relaxation).
    """

    def test_transformer_needs_recent_destabilizer(self):
        """Transformers require recent destabilizer within ~3 ops."""
        seq = [Emission(), Dissonance(), Coherence(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid

    def test_transformer_without_destabilizer_fails(self):
        """Transformer without recent destabilizer should fail."""
        seq = [Emission(), Coherence(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert not valid
        assert "U4b violated" in msg
        assert "recent destabilizer" in msg

    def test_mutation_needs_prior_coherence(self):
        """ZHIR (Mutation) needs prior IL (Coherence) for stable base."""
        # Valid: has prior IL
        seq = [Emission(), Coherence(), Dissonance(), Mutation(), Silence()]
        valid, _ = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid

        # Invalid: no prior IL
        seq = [Emission(), Dissonance(), Mutation(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert not valid
        assert "prior IL" in msg.lower() or "coherence" in msg.lower()

    def test_recent_window_is_three_ops(self):
        """Recent destabilizer window is ~3 operators."""
        # Destabilizer at position 1, transformer at position 5 (distance > 3)
        seq = [
            Emission(),
            Dissonance(),
            Coherence(),
            Resonance(),
            Coherence(),
            Mutation(),
            Silence(),
        ]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert not valid  # Too far apart

    def test_destabilizer_within_window_valid(self):
        """Destabilizer within 3 ops is valid."""
        # Destabilizer at 1, transformer at 3 (distance = 2, within window)
        seq = [Emission(), Dissonance(), Coherence(), Mutation(), Silence()]
        valid, _ = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid

    def test_self_organization_needs_destabilizer(self):
        """THOL (Self-organization) needs recent destabilizer."""
        # Valid
        seq = [Emission(), Dissonance(), SelfOrganization(), Silence()]
        valid, _ = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid

        # Invalid - no recent destabilizer
        seq = [Emission(), Coherence(), SelfOrganization(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert not valid
        assert "recent destabilizer" in msg

    def test_no_transformers_not_applicable(self):
        """U4b not applicable without transformers."""
        seq = [Emission(), Coherence(), Dissonance(), Silence()]
        valid, msg = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid
        assert "not applicable" in msg


class TestIntegration:
    """Test complete sequence validation.

    Integration tests validate that all U1-U5 constraints (temporal + multi-scale) work together
    correctly and produce expected validation results for complete sequences.
    """

    def test_valid_bootstrap_sequence(self):
        """Bootstrap pattern: AL → UM → IL.

        Physics: Basic activation with coupling and stabilization.
        """
        seq = [Emission(), Coupling(), Coherence()]
        validator = UnifiedGrammarValidator()
        valid, messages = validator.validate(seq, epi_initial=0.0)
        # Note: This will fail U1b (no closure), but tests integration
        assert not valid  # Missing closure
        assert any("U1b violated" in m for m in messages)

    def test_valid_bootstrap_with_closure(self):
        """Complete bootstrap: AL → UM → IL → SHA."""
        seq = [Emission(), Coupling(), Coherence(), Silence()]
        validator = UnifiedGrammarValidator()
        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid
        # All constraints should be satisfied or not applicable
        assert all("violated" not in m for m in messages)

    def test_valid_exploration_sequence(self):
        """Exploration pattern: OZ → ZHIR → IL.

        Physics: Dissonance creates bifurcation potential,
        mutation executes phase change, coherence stabilizes.
        """
        seq = [
            Emission(),
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
            Silence(),
        ]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid
        assert all("violated" not in m for m in messages)

    def test_invalid_multiple_violations(self):
        """Sequence violating multiple constraints.

        Violations:
        - No generator (U1a) when EPI=0
        - No closure (U1b)
        - Destabilizer without stabilizer (U2)
        """
        # Use sequence without stabilizer (Coherence is a stabilizer)
        seq = [Coupling(), Dissonance(), Expansion()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert not valid
        assert any("U1a violated" in m for m in messages)
        assert any("U1b violated" in m for m in messages)
        assert any("U2 violated" in m for m in messages)

    def test_validate_unified_convenience_function(self):
        """Test convenience function validate_unified().

        This function returns only boolean result for simple validation.
        """
        # Valid sequence
        seq = [Emission(), Coherence(), Silence()]
        assert validate_unified(seq, epi_initial=0.0) is True

        # Invalid sequence (no closure)
        seq = [Coherence(), Expansion()]
        assert validate_unified(seq, epi_initial=0.0) is False

    def test_complex_valid_sequence(self):
        """Complex but valid sequence with multiple operator types.

        Pattern: Activation → Destabilization → Transformation → Stabilization → Closure
        """
        seq = [
            Emission(),  # Generator
            Reception(),  # Information gathering
            Dissonance(),  # Destabilizer + bifurcation trigger
            Coherence(),  # Stabilizer + bifurcation handler
            Coupling(),  # Requires phase verification (U3)
            Expansion(),  # Destabilizer
            SelfOrganization(),  # Transformer + stabilizer + handler
            Silence(),  # Closure
        ]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid
        assert all("violated" not in m for m in messages)

    def test_resonance_requires_phase_check(self):
        """Sequences with resonance trigger U3 awareness."""
        seq = [Emission(), Resonance(), Coherence(), Silence()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid
        # Find U3 message and verify it mentions phase
        u3_msg = [m for m in messages if "U3:" in m][0]
        assert "phase verification" in u3_msg.lower()

    def test_all_messages_include_constraint_label(self):
        """All validation messages should include U1a, U1b, U2, U3, U4a, U4b, U2-REMESH, or U5 label."""
        seq = [Emission(), Coherence(), Silence()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)

        # Should have exactly 8 messages (one for each constraint including U2-REMESH and U5)
        assert len(messages) == 8

        # Each should start with constraint label
        constraint_labels = ["U1a:", "U1b:", "U2:", "U3:", "U4a:", "U4b:", "U2-REMESH:", "U5:"]
        for label in constraint_labels:
            assert any(label in m for m in messages), f"Missing {label} in messages"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_operator_sequence(self):
        """Single operator that is both generator and closure."""
        # Transition is both generator and closure
        seq = [Transition()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid

    def test_empty_sequence_validation(self):
        """Empty sequence should fail initiation and closure."""
        seq = []
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert not valid
        assert any("U1a violated" in m for m in messages)
        assert any("U1b violated" in m for m in messages)

    def test_sequence_with_only_stabilizers(self):
        """Sequence with only stabilizers (no destabilizers) is valid.

        Note: Self-organization is a transformer and requires recent destabilizer.
        Using only Coherence which is stabilizer but not transformer.
        """
        seq = [Emission(), Coherence(), Reception(), Silence()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid

    def test_destabilizer_at_window_boundary(self):
        """Destabilizer exactly 3 ops before transformer (boundary case)."""
        # Destabilizer at pos 1, transformer at pos 4 (distance = 3, at boundary)
        seq = [
            Emission(),
            Dissonance(),
            Coherence(),
            Resonance(),
            Mutation(),
            Silence(),
        ]
        valid, _ = UnifiedGrammarValidator.validate_transformer_context(seq)
        # At position 4, window is [1:4], so destabilizer at 1 is included
        assert valid

    def test_mutation_with_coherence_not_immediate(self):
        """ZHIR can have IL anywhere before it, not just immediately."""
        seq = [
            Emission(),
            Coherence(),
            Reception(),
            Dissonance(),
            Mutation(),
            Silence(),
        ]
        valid, _ = UnifiedGrammarValidator.validate_transformer_context(seq)
        assert valid

    def test_recursivity_as_generator_and_closure(self):
        """REMESH can serve as both generator and closure in same sequence."""
        seq = [Recursivity(), Coherence(), Recursivity()]
        validator = UnifiedGrammarValidator()

        valid, messages = validator.validate(seq, epi_initial=0.0)
        assert valid


class TestOperatorNameHandling:
    """Test that operator name extraction works correctly.

    The validator uses getattr(op, "canonical_name", op.name.lower())
    to extract operator names, with fallback to .name attribute.
    """

    def test_operators_have_name_attribute(self):
        """All operator instances have .name attribute."""
        operators = [
            Emission(),
            Reception(),
            Coherence(),
            Dissonance(),
            Coupling(),
            Resonance(),
            Silence(),
            Expansion(),
            Contraction(),
            SelfOrganization(),
            Mutation(),
            Transition(),
            Recursivity(),
        ]
        for op in operators:
            assert hasattr(op, "name")
            assert isinstance(op.name, str)
            assert op.name.lower() in {
                "emission",
                "reception",
                "coherence",
                "dissonance",
                "coupling",
                "resonance",
                "silence",
                "expansion",
                "contraction",
                "self_organization",
                "mutation",
                "transition",
                "recursivity",
            }

    def test_operator_names_are_lowercase(self):
        """Operator names should be normalized to lowercase."""
        # The validator uses op.name.lower()
        seq = [Emission(), Coherence(), Silence()]
        validator = UnifiedGrammarValidator()

        valid, _ = validator.validate(seq, epi_initial=0.0)
        assert valid


class TestU6StructuralPotentialConfinement:
    """Test U6: STRUCTURAL POTENTIAL CONFINEMENT (CANONICAL as of 2025-11-11).

    Tests the structural potential field Φ_s and escape threshold validation.

    Physics basis:
    - Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α (α=2)
    - Δ Φ_s < 2.0 (escape threshold)
    - corr(Δ Φ_s, ΔC) = -0.822 (validated 2,400+ experiments)
    - CV = 0.1% (topology-universal)

    References:
    - UNIFIED_GRAMMAR_RULES.md § U6: Complete physics derivation
    - docs/TNFR_FORCES_EMERGENCE.md § 14-15: Validation evidence
    - AGENTS.md § Structural Fields: Canonical status
    """

    def test_structural_potential_confined_sequence(self):
        """Valid sequence maintains Δ Φ_s < 2.0 (safe regime)."""
        import networkx as nx
        from tnfr.physics.fields import compute_structural_potential
        from tnfr.operators.grammar import validate_structural_potential_confinement

        # Create test network
        G = nx.karate_club_graph()

        # Set ΔNFR to simulate valid sequence result (low drift)
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.5  # Moderate pressure

        phi_before = compute_structural_potential(G, alpha=2.0)

        # Simulate small change (valid sequence)
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.6  # Small increase

        phi_after = compute_structural_potential(G, alpha=2.0)

        # Validate confinement
        valid, drift, msg = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )

        assert valid, f"Expected confinement, got: {msg}"
        assert drift < 2.0, f"Drift {drift:.3f} should be < 2.0"
        assert "PASS" in msg
        assert "confined" in msg

    def test_structural_potential_escape_violation(self):
        """Violation sequence produces Δ Φ_s ≥ 2.0 (fragmentation risk)."""
        import networkx as nx
        from tnfr.physics.canonical import compute_structural_potential
        from tnfr.operators.grammar import (
            validate_structural_potential_confinement,
            StructuralPotentialConfinementError,
        )

        # Create test network
        G = nx.karate_club_graph()

        # Set ΔNFR to simulate before state (low, uniform)
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.1

        phi_before = compute_structural_potential(G, alpha=2.0)

        # Simulate large non-uniform change (violation: creates hotspots)
        for i, node in enumerate(G.nodes()):
            # Alternate high/low to maximize spatial gradients
            G.nodes[node]["delta_nfr"] = 5.0 if i % 2 == 0 else 0.1

        # Use different alpha to avoid cache hit
        phi_after = compute_structural_potential(G, alpha=2.001)

        # Validate - should fail
        valid, drift, msg = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )

        assert not valid, f"Expected escape violation, got: {msg}"
        assert drift >= 2.0, f"Drift {drift:.3f} should be ≥ 2.0"
        assert "FAIL" in msg
        assert "escape" in msg.lower() or "fragmentation" in msg.lower()

    def test_structural_potential_strict_mode_raises(self):
        """strict=True raises error on violation."""
        import networkx as nx
        from tnfr.physics.canonical import compute_structural_potential
        from tnfr.operators.grammar import (
            validate_structural_potential_confinement,
            StructuralPotentialConfinementError,
        )

        # Create test network with violation
        G = nx.karate_club_graph()

        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.1
        phi_before = compute_structural_potential(G, alpha=2.0)

        # Non-uniform change to create violation
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["delta_nfr"] = 5.0 if i % 2 == 0 else 0.1
        phi_after = compute_structural_potential(G, alpha=2.001)

        # Should raise in strict mode
        with pytest.raises(StructuralPotentialConfinementError) as exc_info:
            validate_structural_potential_confinement(
                G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=True
            )

        err = exc_info.value
        assert "U6 STRUCTURAL POTENTIAL CONFINEMENT violated" in str(err)
        assert err.context["delta_phi_s"] >= 2.0
        assert err.threshold == 2.0

    def test_structural_potential_custom_threshold(self):
        """Custom threshold allows tuning escape boundary."""
        import networkx as nx
        from tnfr.physics.canonical import compute_structural_potential
        from tnfr.operators.grammar import (
            validate_structural_potential_confinement,
        )

        G = nx.karate_club_graph()

        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.2
        phi_before = compute_structural_potential(G, alpha=2.0)

        # Moderate non-uniform change (drift ~0.6-1.0)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["delta_nfr"] = 1.5 if i % 3 == 0 else 0.2
        phi_after = compute_structural_potential(G, alpha=2.001)

        # Should pass with threshold=STRUCTURAL_ESCAPE_THRESHOLD
        valid_high, drift, _ = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )
        assert valid_high

        # Should fail with threshold=0.5
        valid_low, _, _ = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=0.5, strict=False
        )
        assert not valid_low

        # Drift should be the same regardless of threshold
        _, drift_high, _ = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )
        _, drift_low, _ = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=0.5, strict=False
        )
        assert abs(drift_high - drift_low) < 1e-6

    def test_structural_potential_empty_network(self):
        """Empty network trivially satisfies U6."""
        import networkx as nx
        from tnfr.operators.grammar import validate_structural_potential_confinement

        G = nx.Graph()  # Empty network

        phi_before = {}
        phi_after = {}

        valid, drift, msg = validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )

        assert valid
        assert drift == 0.0
        assert "No nodes" in msg or "trivial" in msg.lower()

    def test_structural_potential_physics_interpretation(self):
        """Verify passive equilibrium mechanism (not active attraction)."""
        import networkx as nx
        from tnfr.physics.canonical import compute_structural_potential
        from tnfr.operators.grammar import (
            validate_structural_potential_confinement,
        )

        # Create network
        G = nx.karate_club_graph()

        # Test 1: Grammar-valid behavior (small drift ~0.6)
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.5
        phi_valid_before = compute_structural_potential(G, alpha=2.0)

        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.55  # Small change
        phi_valid_after = compute_structural_potential(G, alpha=2.001)

        valid_ok, drift_valid, _ = validate_structural_potential_confinement(
            G, phi_valid_before, phi_valid_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )

        # Test 2: Grammar-violating behavior (large drift ~3.9)
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.5
        phi_viol_before = compute_structural_potential(G, alpha=2.1)

        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 2.5  # Large change
        phi_viol_after = compute_structural_potential(G, alpha=2.101)

        valid_bad, drift_viol, _ = validate_structural_potential_confinement(
            G, phi_viol_before, phi_viol_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
        )

        # Verify passive protection: valid drift < violation drift
        assert (
            drift_valid < drift_viol
        ), "Valid sequences should have smaller drift"
        assert drift_valid < 2.0, "Valid drift below threshold"
        # Note: Actual ratio ~0.15× (valid 0.6 vs violation 3.9)"

    def test_structural_potential_topology_independence(self):
        """Φ_s validation works across different topologies."""
        import networkx as nx
        from tnfr.physics.fields import compute_structural_potential
        from tnfr.operators.grammar import (
            validate_structural_potential_confinement
        )

        topologies = [
            ("ring", nx.cycle_graph(20)),
            ("grid", nx.grid_2d_graph(5, 5)),
            ("tree", nx.balanced_tree(2, 3)),
            ("scale_free", nx.barabasi_albert_graph(20, 2)),
        ]

        for name, G in topologies:
            # Convert grid nodes to integers for consistency
            if name == "grid":
                G = nx.convert_node_labels_to_integers(G)

            # Set ΔNFR
            for node in G.nodes():
                G.nodes[node]["delta_nfr"] = 0.5
            phi_before = compute_structural_potential(G, alpha=2.0)

            # Small change (valid)
            for node in G.nodes():
                G.nodes[node]["delta_nfr"] = 0.6
            phi_after = compute_structural_potential(G, alpha=2.0)

            valid, drift, msg = validate_structural_potential_confinement(
                G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
            )

            assert valid, f"Topology {name} should pass U6: {msg}"
            assert drift < 2.0, (
                f"Topology {name} drift {drift:.3f} exceeds threshold"
            )
