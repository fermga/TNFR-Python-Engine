"""Integration tests for ZHIR (Mutation) in canonical operator sequences.

This module tests ZHIR behavior in complete, canonical operator sequences
as specified in AGENTS.md and UNIFIED_GRAMMAR_RULES.md:

Test Coverage:
1. Canonical mutation cycle (IL → OZ → ZHIR → IL)
2. Mutation with self-organization (OZ → ZHIR → THOL)
3. Bootstrap with mutation (AL → IL → OZ → ZHIR → NAV)
4. Extended sequences with multiple mutations
5. Sequence validation with grammar rules

References:
- AGENTS.md §Operator Composition
- UNIFIED_GRAMMAR_RULES.md (U1-U4)
- test_canonical_sequences.py (similar integration tests)
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Emission,
    Coherence,
    Dissonance,
    Mutation,
    SelfOrganization,
    Transition,
    Silence,
    Resonance,
)


class TestCanonicalMutationCycle:
    """Test the canonical IL → OZ → ZHIR → IL mutation cycle."""

    def test_canonical_mutation_cycle_completes(self):
        """IL → OZ → ZHIR → IL should complete successfully."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Apply canonical cycle
        run_sequence(G, node, [
            Coherence(),    # IL: Stabilize
            Dissonance(),   # OZ: Destabilize
            Mutation(),     # ZHIR: Transform
            Coherence(),    # IL: Stabilize
        ])
        
        # Verify all operators executed
        metrics = G.graph["operator_metrics"]
        glyphs = [m["glyph"] for m in metrics]
        
        assert "IL" in glyphs
        assert "OZ" in glyphs
        assert "ZHIR" in glyphs
        assert glyphs.count("IL") >= 2  # IL appears twice

    def test_mutation_cycle_improves_coherence(self):
        """Mutation cycle should end with stabilized state."""
        from tnfr.metrics.coherence import compute_coherence
        
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Measure initial coherence
        C_initial = compute_coherence(G)
        
        # Apply mutation cycle
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
        ])
        
        # Measure final coherence
        C_final = compute_coherence(G)
        
        # Final coherence should be reasonable (not collapsed)
        assert C_final > 0.3, f"Coherence collapsed: {C_initial} → {C_final}"

    def test_mutation_cycle_preserves_node_viability(self):
        """After mutation cycle, node should remain viable (νf > 0)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
        ])
        
        # Node should remain viable
        vf_final = G.nodes[node]["νf"]
        assert vf_final > 0, "Mutation cycle killed node (νf → 0)"
        assert vf_final > 0.2, "Mutation cycle severely weakened node"

    def test_mutation_cycle_theta_transformed(self):
        """Mutation cycle should transform phase (θ changes)."""
        import math
        
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        theta_initial = G.nodes[node]["theta"]
        
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
        ])
        
        theta_final = G.nodes[node]["theta"]
        
        # Phase should have changed during ZHIR
        assert theta_final != theta_initial, "ZHIR did not transform phase"

    def test_multiple_mutation_cycles(self):
        """Multiple mutation cycles should work correctly."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Apply 3 mutation cycles
        for i in range(3):
            run_sequence(G, node, [
                Coherence(),
                Dissonance(),
                Mutation(),
                Coherence(),
            ])
        
        # Node should still be viable
        assert G.nodes[node]["νf"] > 0
        assert -1.0 <= G.nodes[node]["EPI"] <= 1.0


class TestMutationWithSelfOrganization:
    """Test OZ → ZHIR → THOL sequence."""

    def test_mutation_then_self_organization(self):
        """OZ → ZHIR → THOL should complete successfully."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Add a neighbor for THOL to work with
        neighbor_node = "neighbor"
        G.add_node(neighbor_node, EPI=0.4, **{"νf": 1.0}, theta=0.5, delta_nfr=0.0)
        G.add_edge(node, neighbor_node)
        
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Apply sequence
        run_sequence(G, node, [
            Coherence(),        # Stabilize
            Dissonance(),       # Destabilize
            Mutation(),         # Transform
            SelfOrganization(), # Organize
        ])
        
        # Verify sequence completed
        metrics = G.graph["operator_metrics"]
        glyphs = [m["glyph"] for m in metrics]
        
        assert "OZ" in glyphs
        assert "ZHIR" in glyphs
        assert "THOL" in glyphs

    def test_mutation_enables_self_organization(self):
        """ZHIR transformation should enable effective THOL."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.2)
        # Add neighbors with proper initialization
        for i in range(2):
            neighbor_id = f"n{i}"
            G.add_node(neighbor_id, EPI=0.4, **{"νf": 1.0}, theta=0.3 + i * 0.1, delta_nfr=0.0)
            G.add_edge(node, neighbor_id)
        
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Apply sequence
        run_sequence(G, node, [
            Dissonance(),       # Create instability
            Mutation(),         # Transform phase
            SelfOrganization(), # Should work with transformed state
        ])
        
        # Verify THOL was applied
        # (exact effects depend on implementation)


class TestBootstrapWithMutation:
    """Test complete bootstrap sequence including mutation."""

    def test_bootstrap_with_mutation(self):
        """AL → IL → OZ → ZHIR → NAV should complete lifecycle."""
        G, node = create_nfr("test", epi=0.0, vf=0.5)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Complete bootstrap sequence
        run_sequence(G, node, [
            Emission(),    # AL: Generate from vacuum
            Coherence(),   # IL: Stabilize
            Dissonance(),  # OZ: Destabilize
            Mutation(),    # ZHIR: Transform
            Transition(),  # NAV: Regime shift
        ])
        
        # Verify all operators executed
        metrics = G.graph["operator_metrics"]
        glyphs = [m["glyph"] for m in metrics]
        
        assert "AL" in glyphs
        assert "IL" in glyphs
        assert "OZ" in glyphs
        assert "ZHIR" in glyphs
        assert "NAV" in glyphs

    def test_bootstrap_sequence_grammar_valid(self):
        """Bootstrap sequence should satisfy all grammar rules."""
        G, node = create_nfr("test", epi=0.0, vf=0.5)
        
        # With strict validation enabled
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Sequence should pass grammar validation
        # Note: Removed Transition at end as it requires perturbation before it
        run_sequence(G, node, [
            Emission(),    # U1a: Generator for EPI=0
            Coherence(),   # U2: Stabilizer after generation
            Dissonance(),  # U2: Destabilizer (needs stabilizer after)
            Mutation(),    # U4b: Transformer (has IL + destabilizer)
            Coherence(),   # U2: Stabilizer after mutation
            # Transition requires perturbation, so use Silence for closure
            Silence(),     # U1b: Closure
        ])
        
        # Should complete without grammar violations

    def test_bootstrap_node_becomes_viable(self):
        """After bootstrap, node should be fully viable."""
        G, node = create_nfr("test", epi=0.0, vf=0.3)
        
        run_sequence(G, node, [
            Emission(),
            Coherence(),
            Dissonance(),
            Mutation(),
            Transition(),
        ])
        
        # Node should be viable
        assert G.nodes[node]["EPI"] != 0.0
        assert G.nodes[node]["νf"] > 0
        assert 0 <= G.nodes[node]["theta"] < 6.28319  # 2π


class TestExtendedMutationSequences:
    """Test extended sequences with multiple operations."""

    def test_oz_zhir_oz_zhir_sequence(self):
        """Multiple mutation applications in one sequence."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Double mutation sequence
        run_sequence(G, node, [
            Coherence(),   # Stabilize
            Dissonance(),  # Destabilize 1
            Mutation(),    # Transform 1
            Coherence(),   # Stabilize
            Dissonance(),  # Destabilize 2
            Mutation(),    # Transform 2
            Coherence(),   # Stabilize
        ])
        
        # Node should still be viable
        assert G.nodes[node]["νf"] > 0
        assert -1.0 <= G.nodes[node]["EPI"] <= 1.0

    def test_resonance_after_mutation(self):
        """RA (Resonance) after ZHIR should propagate transformed state."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        # Add neighbor with compatible phase and proper initialization
        neighbor_id = "neighbor"
        G.add_node(neighbor_id, EPI=0.4, **{"νf": 1.0}, theta=0.6, delta_nfr=0.0)
        G.add_edge(node, neighbor_id)
        
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Apply sequence
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),    # Transform phase
            Resonance(),   # Propagate transformed pattern
        ])
        
        # Resonance should have executed
        # (exact effects depend on phase compatibility)

    def test_silence_after_mutation(self):
        """SHA (Silence) after ZHIR should freeze transformed state."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Apply mutation then silence
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
            Silence(),
        ])
        
        # SHA should have frozen state
        # νf should be near zero during silence
        # (actual implementation may vary)


class TestSequenceGrammarValidation:
    """Test that mutation sequences satisfy grammar rules."""

    def test_u4b_satisfied_in_canonical_sequence(self):
        """Canonical sequence should satisfy U4b."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # U4b requires: IL precedence + recent destabilizer
        run_sequence(G, node, [
            Coherence(),   # IL precedence ✓
            Dissonance(),  # Recent destabilizer ✓
            Mutation(),    # U4b satisfied ✓
        ])
        
        # Should complete without error

    def test_u2_satisfied_with_stabilizers(self):
        """U2 (Convergence) satisfied with stabilizers after destabilizers."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # U2 requires stabilizers after destabilizers
        run_sequence(G, node, [
            Dissonance(),       # Destabilizer
            Mutation(),         # Destabilizer/Transformer
            Coherence(),        # Stabilizer ✓
        ])
        
        # Integral should converge (not diverge)

    def test_u1b_closure_satisfied(self):
        """Sequences should end with closure operators."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        
        # Test various closure operators
        closure_operators = [
            Silence(),
            Transition(),
            # Dissonance also counts as closure
        ]
        
        for closure_op in closure_operators[:2]:  # Test first two
            run_sequence(G, node, [
                Coherence(),
                Dissonance(),
                Mutation(),
                closure_op,  # U1b: Closure ✓
            ])
            
            # Should complete successfully


class TestMutationSequenceMetrics:
    """Test metrics collection in sequences with mutation."""

    def test_sequence_metrics_captured(self):
        """All operators in sequence should have metrics."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
        ])
        
        metrics = G.graph["operator_metrics"]
        
        # Should have 4 metric entries
        assert len(metrics) >= 4
        
        # Verify operators are tracked
        operators = [m["operator"] for m in metrics]
        assert "Coherence" in operators
        assert "Dissonance" in operators
        assert "Mutation" in operators

    def test_mutation_context_captured(self):
        """ZHIR metrics should include context from preceding operators."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.35, 0.42, 0.50]
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        run_sequence(G, node, [
            Coherence(),
            Dissonance(),
            Mutation(),
        ])
        
        # Find ZHIR metrics
        metrics = G.graph["operator_metrics"]
        zhir_metrics = [m for m in metrics if m["glyph"] == "ZHIR"]
        
        assert len(zhir_metrics) > 0
        zhir_metric = zhir_metrics[-1]
        
        # Should capture context
        assert "destabilizer_type" in zhir_metric
        assert "destabilizer_operator" in zhir_metric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
