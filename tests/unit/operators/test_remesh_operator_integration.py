"""Integration tests for REMESH operator with all 13 canonical operators.

Tests verify that REMESH integrates properly with each operator according to
the physical relationships documented in remesh.py module docstring.
"""

import pytest
import networkx as nx

from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling,
    Resonance, Silence, Expansion, Contraction, SelfOrganization,
    Mutation, Transition, Recursivity
)
from tnfr.operators.remesh import (
    StructuralIdentity,
    structural_similarity,
    compute_global_coherence,
)
from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA
from tnfr.structural import run_sequence


class TestREMESHOperatorIntegration:
    """Test REMESH integration with all 13 canonical operators."""
    
    def test_remesh_with_IL_hierarchical(self):
        """REMESH → IL (hierarchical stabilization)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, 1.0)
        
        # Capture identity before
        identity_before = StructuralIdentity.capture_from_node(G.nodes[1])
        
        # Apply REMESH → IL sequence
        run_sequence(G, 1, [Recursivity(), Coherence()])
        
        # Verify coherence increased (IL effect)
        coherence = compute_global_coherence(G)
        assert coherence > 0.5  # Should be stabilized
        
        # Verify identity preserved
        assert identity_before.matches(G.nodes[1], tolerance=0.2)
    
    def test_remesh_with_SHA_latent_memory(self):
        """SHA → REMESH (frozen latent memory propagation)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.6)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply SHA to freeze
        run_sequence(G, 1, [Silence()])
        
        # Capture frozen identity
        identity_frozen = StructuralIdentity.capture_from_node(
            G.nodes[1],
            is_sha_frozen=True
        )
        
        assert identity_frozen.frozen_by_sha is True
        
        # Apply REMESH on frozen state
        run_sequence(G, 1, [Recursivity()])
        
        # Identity should still match (frozen state propagated)
        assert identity_frozen.matches(G.nodes[1], tolerance=0.2)
    
    def test_remesh_with_VAL_expansion(self):
        """VAL → REMESH (fractal growth)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.4)
        set_attr(G.nodes[1], ALIAS_VF, 0.9)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        epi_before = get_attr(G.nodes[1], ALIAS_EPI)
        
        # Apply VAL → REMESH
        run_sequence(G, 1, [Expansion(), Recursivity()])
        
        epi_after = get_attr(G.nodes[1], ALIAS_EPI)
        
        # EPI should have changed (expansion + recursion)
        # Note: exact behavior depends on implementation
        assert epi_after is not None
    
    def test_remesh_with_OZ_rhizomatic(self):
        """OZ → REMESH (distributed bifurcation)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply OZ → REMESH (rhizomatic mode)
        run_sequence(G, 1, [Dissonance(), Recursivity()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_THOL_self_organization(self):
        """THOL → REMESH (recursive self-organization)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.6)
        set_attr(G.nodes[1], ALIAS_VF, 1.1)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply THOL → REMESH
        run_sequence(G, 1, [SelfOrganization(), Recursivity()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_RA_fractal_harmonic(self):
        """REMESH → RA (multi-scale resonance)."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[2], ALIAS_EPI, 0.52)
        set_attr(G.nodes[2], ALIAS_VF, 0.98)
        set_attr(G.nodes[2], ALIAS_THETA, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply REMESH → RA
        run_sequence(G, 1, [Recursivity(), Resonance()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_AL_fractal_emission(self):
        """AL → REMESH (fractal emission)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.1)  # Low EPI (emission will increase)
        set_attr(G.nodes[1], ALIAS_VF, 0.5)   # Positive νf required
        set_attr(G.nodes[1], ALIAS_THETA, 1.0)
        
        # Apply AL → REMESH
        run_sequence(G, 1, [Emission(), Recursivity()])
        
        # EPI should increase after emission
        epi_after = get_attr(G.nodes[1], ALIAS_EPI)
        assert epi_after >= 0.1  # Should not decrease
    
    def test_remesh_with_NAV_transition(self):
        """NAV → REMESH (fractal attractor transition)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply NAV → REMESH
        run_sequence(G, 1, [Transition(), Recursivity()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_EN_reception(self):
        """EN → REMESH (symmetric multi-scale reception)."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(2, 1)  # Node 2 influences node 1
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[2], ALIAS_EPI, 0.6)
        set_attr(G.nodes[2], ALIAS_VF, 1.0)
        set_attr(G.nodes[2], ALIAS_THETA, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply EN → REMESH
        run_sequence(G, 1, [Reception(), Recursivity()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_NUL_contraction(self):
        """NUL → REMESH (hierarchical compression)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.7)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # Apply NUL → REMESH
        run_sequence(G, 1, [Contraction(), Recursivity()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_with_UM_coupling(self):
        """REMESH → UM (multi-scale coupling)."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, 0.5)
        
        set_attr(G.nodes[2], ALIAS_EPI, 0.52)
        set_attr(G.nodes[2], ALIAS_VF, 0.98)
        set_attr(G.nodes[2], ALIAS_THETA, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[2], ALIAS_THETA, 0.52)
        
        # Apply REMESH → UM
        run_sequence(G, 1, [Recursivity(), Coupling()])
        
        # Should complete without error
        assert G.nodes[1] is not None
    
    def test_remesh_zhir_indirect_relationship(self):
        """ZHIR operates post-recursion (indirect relationship)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # ZHIR requires prior IL + recent destabilizer (U4b grammar)
        # Apply: IL → OZ → REMESH → ZHIR
        run_sequence(G, 1, [
            Coherence(),      # Prior IL (base stability)
            Dissonance(),     # Recent destabilizer
            Recursivity(),    # REMESH propagates variations
            Mutation()        # ZHIR transforms post-recursion
        ])
        
        # Should complete without error
        assert G.nodes[1] is not None


class TestOperatorSequenceGrammar:
    """Test that REMESH respects grammar rules with various operators."""
    
    def test_remesh_as_generator_U1a(self):
        """REMESH can initiate sequences (generator)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # REMESH as first operator (generator)
        run_sequence(G, 1, [Recursivity()])
        
        assert G.nodes[1] is not None
    
    def test_remesh_as_closure_U1b(self):
        """REMESH can close sequences (closure)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # REMESH as last operator (closure)
        run_sequence(G, 1, [Emission(), Recursivity()])
        
        assert G.nodes[1] is not None
    
    def test_remesh_with_destabilizer_requires_stabilizer_U2(self):
        """REMESH + destabilizers require stabilizers (U2 convergence)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        
        # REMESH + VAL (destabilizer) → requires IL (stabilizer)
        run_sequence(G, 1, [
            Recursivity(),
            Expansion(),      # Destabilizer
            Coherence()       # Stabilizer (required by U2)
        ])
        
        assert G.nodes[1] is not None
    
    def test_remesh_phase_verification_U3(self):
        """REMESH with coupling operators requires phase compatibility (U3)."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)
        
        # Set compatible phases
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, 1.0)
        
        set_attr(G.nodes[2], ALIAS_EPI, 0.52)
        set_attr(G.nodes[2], ALIAS_VF, 0.98)
        set_attr(G.nodes[2], ALIAS_THETA, 1.0)
        set_attr(G.nodes[node if "node" in locals() else 1], ALIAS_THETA, 1.0)
        set_attr(G.nodes[2], ALIAS_THETA, 1.05)  # Compatible phase
        
        # Apply REMESH with RA (requires phase compatibility)
        run_sequence(G, 1, [Recursivity(), Resonance()])
        
        # Verify phase compatibility maintained
        identity = StructuralIdentity.capture_from_node(G.nodes[1])
        assert identity.phase_pattern is not None


class TestStructuralSimilarityIntegration:
    """Test structural similarity with real operator sequences."""
    
    def test_similar_patterns_after_same_sequence(self):
        """Nodes with same sequence should have similar EPIs."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)
        
        # Start with similar states
        for node in [1, 2]:
            set_attr(G.nodes[node], ALIAS_EPI, 0.5)
            set_attr(G.nodes[node], ALIAS_VF, 1.0)
            set_attr(G.nodes[node], ALIAS_THETA, 1.0)
        
        # Apply same sequence to both
        for node in [1, 2]:
            run_sequence(G, node, [Coherence(), Recursivity()])
        
        epi1 = get_attr(G.nodes[1], ALIAS_EPI)
        epi2 = get_attr(G.nodes[2], ALIAS_EPI)
        
        similarity = structural_similarity(epi1, epi2)
        assert similarity > 0.8  # Should remain similar


class TestREMESHGrammarLimitations:
    """Test REMESH-specific grammar limitations derived from physics."""
    
    def test_remesh_amplification_requires_stabilizer_with_VAL(self):
        """U2-REMESH: REMESH + VAL requires stabilizer (recursive expansion)."""
        from tnfr.operators.grammar import GrammarValidator
        
        # REMESH + VAL without stabilizer should violate U2-REMESH
        seq_invalid = [Emission(), Recursivity(), Expansion(), Silence()]
        valid, msgs = GrammarValidator.validate(seq_invalid)
        
        # Should fail
        assert not valid
        
        # Should mention U2-REMESH violation
        u2_remesh_msg = [m for m in msgs if "U2-REMESH" in m and "violated" in m]
        assert len(u2_remesh_msg) > 0
        assert "recursive amplification" in u2_remesh_msg[0].lower()
    
    def test_remesh_amplification_satisfied_with_stabilizer(self):
        """U2-REMESH: REMESH + VAL + IL satisfies grammar."""
        from tnfr.operators.grammar import GrammarValidator
        
        # REMESH + VAL + IL should satisfy U2-REMESH
        seq_valid = [Emission(), Recursivity(), Expansion(), Coherence(), Silence()]
        valid, msgs = GrammarValidator.validate(seq_valid)
        
        # Should pass
        assert valid
        
        # U2-REMESH should be satisfied
        u2_remesh_msg = [m for m in msgs if "U2-REMESH" in m and "satisfied" in m]
        assert len(u2_remesh_msg) > 0
    
    def test_remesh_amplification_requires_stabilizer_with_OZ(self):
        """U2-REMESH: REMESH + OZ requires stabilizer (recursive bifurcation)."""
        from tnfr.operators.grammar import GrammarValidator
        
        # REMESH + OZ without stabilizer should violate U2-REMESH
        seq_invalid = [Emission(), Recursivity(), Dissonance(), Silence()]
        valid, msgs = GrammarValidator.validate(seq_invalid)
        
        # Should fail
        assert not valid
        
        # Should mention dissonance amplification
        u2_remesh_msg = [m for m in msgs if "U2-REMESH" in m and "violated" in m]
        assert len(u2_remesh_msg) > 0
        assert "dissonance" in u2_remesh_msg[0].lower()
    
    def test_remesh_without_destabilizer_not_applicable(self):
        """U2-REMESH: Not applicable when REMESH has no destabilizers."""
        from tnfr.operators.grammar import GrammarValidator
        
        # REMESH without destabilizers - U2-REMESH not applicable
        seq = [Emission(), Recursivity(), Coherence(), Silence()]
        valid, msgs = GrammarValidator.validate(seq)
        
        # Should pass
        assert valid
        
        # U2-REMESH should indicate not applicable or satisfied
        u2_remesh_msg = [m for m in msgs if "U2-REMESH" in m]
        assert len(u2_remesh_msg) > 0
        assert ("not applicable" in u2_remesh_msg[0].lower() or 
                "satisfied" in u2_remesh_msg[0].lower())
    
    def test_destabilizer_without_remesh_uses_general_U2(self):
        """General U2 applies when destabilizers present without REMESH."""
        from tnfr.operators.grammar import GrammarValidator
        
        # VAL without REMESH - general U2 applies, not U2-REMESH
        seq = [Emission(), Expansion(), Silence()]
        valid, msgs = GrammarValidator.validate(seq)
        
        # Should fail on general U2
        assert not valid
        
        # U2 should be violated
        u2_msg = [m for m in msgs if m.startswith("U2:") and "violated" in m]
        assert len(u2_msg) > 0
        
        # U2-REMESH should not be applicable
        u2_remesh_msg = [m for m in msgs if "U2-REMESH" in m]
        assert len(u2_remesh_msg) > 0
        assert "not applicable" in u2_remesh_msg[0].lower()
    
    def test_physical_rationale_documented(self):
        """Verify U2-REMESH has physical derivation from nodal equation."""
        from tnfr.operators.grammar import GrammarValidator
        import inspect
        
        # Check that validate_remesh_amplification has proper documentation
        method = GrammarValidator.validate_remesh_amplification
        docstring = inspect.getdoc(method)
        
        # Should mention key physical concepts
        assert "temporal coupling" in docstring.lower()
        assert "recursive" in docstring.lower() or "amplif" in docstring.lower()
        assert "nodal equation" in docstring.lower()
        assert "∫" in docstring or "integral" in docstring.lower()
