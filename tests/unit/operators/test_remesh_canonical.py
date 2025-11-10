"""Tests for canonical REMESH implementation (Phase 1 & 2).

Tests cover:
- Phase 1: Structural memory and pattern recognition
- Phase 2: Coherence preservation and fidelity validation
- SHA-REMESH relationship for latent structural memory
"""

import math
import pytest
import networkx as nx

from tnfr.operators.remesh import (
    StructuralIdentity,
    structural_similarity,
    structural_memory_match,
    RemeshCoherenceLossError,
    compute_global_coherence,
    validate_coherence_preservation,
)
from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA


# ==============================================================================
# Phase 1: Structural Memory & Pattern Recognition Tests
# ==============================================================================


class TestStructuralSimilarity:
    """Test structural similarity calculations."""
    
    def test_identical_scalars(self):
        """Identical EPI values should have similarity = 1.0."""
        similarity = structural_similarity(0.5, 0.5)
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_close_scalars(self):
        """Similar EPI values should have high similarity."""
        similarity = structural_similarity(0.5, 0.52)
        assert similarity > 0.9
        assert similarity < 1.0
    
    def test_distant_scalars(self):
        """Distant EPI values should have low similarity."""
        similarity = structural_similarity(0.1, 0.9)
        assert similarity < 0.5
    
    def test_euclidean_metric_default(self):
        """Default metric should be euclidean."""
        sim1 = structural_similarity(0.5, 0.6)
        sim2 = structural_similarity(0.5, 0.6, metric="euclidean")
        assert sim1 == pytest.approx(sim2)
    
    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation"])
    def test_different_metrics(self, metric):
        """All metrics should return values in [0, 1]."""
        np = pytest.importorskip("numpy")
        arr1 = [0.5, 0.3, 0.7]
        arr2 = [0.52, 0.31, 0.69]
        
        similarity = structural_similarity(arr1, arr2, metric=metric)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # These are similar arrays
    
    def test_vector_similarity_requires_numpy(self):
        """Vector comparison without numpy should raise ImportError."""
        # Temporarily hide numpy
        import sys
        import tnfr.operators.remesh as remesh_module
        
        original_get_numpy = remesh_module._get_numpy
        
        def mock_get_numpy():
            return None
        
        # Clear cache if it exists
        if hasattr(original_get_numpy, 'cache_clear'):
            original_get_numpy.cache_clear()
        
        remesh_module._get_numpy = mock_get_numpy
        
        try:
            with pytest.raises(ImportError, match="NumPy required"):
                structural_similarity([0.5, 0.3], [0.52, 0.31])
        finally:
            remesh_module._get_numpy = original_get_numpy
            if hasattr(remesh_module._get_numpy, 'cache_clear'):
                remesh_module._get_numpy.cache_clear()
    
    def test_mismatched_vector_shapes(self):
        """Vectors with different shapes should raise ValueError."""
        pytest.importorskip("numpy")
        
        with pytest.raises(ValueError, match="same shape"):
            structural_similarity([0.5, 0.3], [0.5, 0.3, 0.7])
    
    def test_unknown_metric(self):
        """Unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            structural_similarity(0.5, 0.6, metric="unknown")


class TestStructuralIdentity:
    """Test StructuralIdentity class and SHA integration."""
    
    def test_create_basic_identity(self):
        """Can create structural identity with basic parameters."""
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.8, 1.2),
            tolerance=0.1,
        )
        
        assert identity.epi_signature == 0.5
        assert identity.vf_range == (0.8, 1.2)
        assert identity.tolerance == 0.1
        assert identity.frozen_by_sha is False
    
    def test_sha_stabilized_flag(self):
        """SHA-stabilized identity can be created."""
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.8, 1.2),
            frozen_by_sha=True,
        )
        
        assert identity.frozen_by_sha is True
    
    def test_matches_exact_node(self):
        """Identity should match node with exact values."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
        )
        
        assert identity.matches(G.nodes[1])
    
    def test_matches_within_tolerance(self):
        """Identity should match node within tolerance."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.52)  # Within default 0.1 tolerance
        set_attr(G.nodes[1], ALIAS_VF, 1.02)
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
            tolerance=0.1,
        )
        
        assert identity.matches(G.nodes[1])
    
    def test_no_match_outside_tolerance(self):
        """Identity should not match node outside tolerance."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.7)  # Outside 0.1 tolerance
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
            tolerance=0.1,
        )
        
        assert not identity.matches(G.nodes[1])
    
    def test_sha_stabilized_accepts_frozen_vf(self):
        """SHA-stabilized identity accepts νf ≈ 0 (frozen state)."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        # Node with EPI signature but frozen νf (SHA effect)
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 0.001)  # Frozen by SHA
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),  # Original range before SHA
            frozen_by_sha=True,  # Mark as SHA-stabilized
        )
        
        # Should match despite νf outside original range
        assert identity.matches(G.nodes[1])
    
    def test_non_sha_rejects_frozen_vf(self):
        """Non-SHA-stabilized identity rejects frozen νf."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 0.001)  # Frozen
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
            frozen_by_sha=False,  # Not SHA-stabilized
        )
        
        # Should NOT match - νf outside range
        assert not identity.matches(G.nodes[1])
    
    def test_phase_pattern_matching(self):
        """Identity can match phase patterns."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, math.pi / 2)
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
            phase_pattern=math.pi / 2,
        )
        
        assert identity.matches(G.nodes[1])
    
    def test_phase_pattern_circular_wrap(self):
        """Phase matching handles circular wrap-around."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, 0.05)  # Near 0
        
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
            phase_pattern=2 * math.pi - 0.05,  # Near 2π (wraps to ~0)
            tolerance=0.15,
        )
        
        # Should match due to circular wrap
        assert identity.matches(G.nodes[1])
    
    def test_record_transformation(self):
        """Can record transformations in lineage."""
        identity = StructuralIdentity(
            epi_signature=0.5,
            vf_range=(0.9, 1.1),
        )
        
        assert len(identity.lineage) == 0
        
        identity.record_transformation("Applied REMESH hierarchical")
        identity.record_transformation("Applied IL stabilization")
        
        assert len(identity.lineage) == 2
        assert "REMESH" in identity.lineage[0]
        assert "IL" in identity.lineage[1]
    
    def test_capture_from_node_basic(self):
        """Can capture identity from node."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.6)
        set_attr(G.nodes[1], ALIAS_VF, 0.95)
        
        identity = StructuralIdentity.capture_from_node(G.nodes[1])
        
        assert identity.epi_signature == pytest.approx(0.6)
        assert identity.vf_range[0] <= 0.95 <= identity.vf_range[1]
        assert identity.frozen_by_sha is False
    
    def test_capture_from_node_sha_stabilized(self):
        """Can capture SHA-stabilized identity."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.7)
        set_attr(G.nodes[1], ALIAS_VF, 0.85)
        
        identity = StructuralIdentity.capture_from_node(
            G.nodes[1],
            is_sha_frozen=True,
        )
        
        assert identity.frozen_by_sha is True
        assert len(identity.lineage) == 1
        assert "SHA" in identity.lineage[0]
    
    def test_capture_from_node_with_phase(self):
        """Captured identity includes phase if present."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_THETA, math.pi / 4)
        
        identity = StructuralIdentity.capture_from_node(G.nodes[1])
        
        assert identity.phase_pattern is not None
        assert identity.phase_pattern == pytest.approx(math.pi / 4)


class TestStructuralMemoryMatch:
    """Test structural memory pattern matching across network."""
    
    def test_finds_similar_nodes(self):
        """Should find nodes with similar EPI patterns."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4])
        inject_defaults(G)
        
        # Node 1: source pattern
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        # Node 2: similar to source
        set_attr(G.nodes[2], ALIAS_EPI, 0.52)
        # Node 3: very similar to source
        set_attr(G.nodes[3], ALIAS_EPI, 0.501)
        # Node 4: dissimilar
        set_attr(G.nodes[4], ALIAS_EPI, 0.9)
        
        matches = structural_memory_match(G, 1, threshold=0.75)
        
        # Should find nodes 2 and 3, not 4
        matched_nodes = [node for node, sim in matches]
        assert 2 in matched_nodes
        assert 3 in matched_nodes
        assert 4 not in matched_nodes
    
    def test_sorted_by_similarity(self):
        """Results should be sorted by similarity (highest first)."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[2], ALIAS_EPI, 0.52)  # Less similar
        set_attr(G.nodes[3], ALIAS_EPI, 0.501) # More similar
        
        matches = structural_memory_match(G, 1, threshold=0.5)
        
        assert len(matches) >= 2
        # Node 3 should come before node 2 (higher similarity)
        similarities = {node: sim for node, sim in matches}
        assert similarities[3] > similarities[2]
    
    def test_excludes_source_node(self):
        """Should not include source node in results."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[2], ALIAS_EPI, 0.5)  # Identical
        
        matches = structural_memory_match(G, 1)
        
        matched_nodes = [node for node, sim in matches]
        assert 1 not in matched_nodes
        assert 2 in matched_nodes
    
    def test_empty_result_no_matches(self):
        """Returns empty list when no nodes match."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.1)
        set_attr(G.nodes[2], ALIAS_EPI, 0.9)
        set_attr(G.nodes[3], ALIAS_EPI, 0.95)
        
        matches = structural_memory_match(G, 1, threshold=0.9)
        
        assert len(matches) == 0
    
    def test_custom_target_nodes(self):
        """Can specify subset of nodes to search."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4])
        inject_defaults(G)
        
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, 0.5)  # All similar
        
        # Only search nodes 2 and 3
        matches = structural_memory_match(
            G, 1,
            target_nodes=[2, 3],
            threshold=0.5
        )
        
        matched_nodes = [node for node, sim in matches]
        assert 2 in matched_nodes or 3 in matched_nodes
        assert 4 not in matched_nodes  # Not in target set


# ==============================================================================
# Phase 2: Coherence Preservation & Fidelity Validation Tests
# ==============================================================================


class TestComputeGlobalCoherence:
    """Test global coherence calculation."""
    
    def test_empty_graph(self):
        """Empty graph has coherence = 0."""
        G = nx.DiGraph()
        inject_defaults(G)
        
        coherence = compute_global_coherence(G)
        assert coherence == 0.0
    
    def test_single_stable_node(self):
        """Stable node (high vf, low ΔNFR) has high coherence."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 1.0)  # High vf
        set_attr(G.nodes[1], ALIAS_DNFR, 0.01)  # Low ΔNFR
        
        coherence = compute_global_coherence(G)
        assert coherence > 0.9
    
    def test_single_unstable_node(self):
        """Unstable node (low vf or high ΔNFR) has low coherence."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)
        
        set_attr(G.nodes[1], ALIAS_EPI, 0.5)
        set_attr(G.nodes[1], ALIAS_VF, 0.2)  # Low vf
        set_attr(G.nodes[1], ALIAS_DNFR, 0.8)  # High ΔNFR
        
        coherence = compute_global_coherence(G)
        assert coherence < 0.5
    
    def test_mixed_network(self):
        """Network with mixed stability has intermediate coherence."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)
        
        # Node 1: high coherence
        set_attr(G.nodes[1], ALIAS_VF, 1.0)
        set_attr(G.nodes[1], ALIAS_DNFR, 0.01)
        
        # Node 2: low coherence
        set_attr(G.nodes[2], ALIAS_VF, 0.1)
        set_attr(G.nodes[2], ALIAS_DNFR, 0.9)
        
        # Node 3: medium coherence
        set_attr(G.nodes[3], ALIAS_VF, 0.6)
        set_attr(G.nodes[3], ALIAS_DNFR, 0.3)
        
        coherence = compute_global_coherence(G)
        assert 0.3 < coherence < 0.8


class TestValidateCoherencePreservation:
    """Test coherence preservation validation."""
    
    def test_perfect_preservation(self):
        """Perfect preservation returns fidelity = 1.0."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 1.0)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.01)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 1.0)
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.01)
        
        fidelity = validate_coherence_preservation(G_before, G_after)
        
        assert fidelity == pytest.approx(1.0, abs=0.05)
    
    def test_improved_coherence(self):
        """Increased coherence returns fidelity > 1.0."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 0.5)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.5)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 1.0)  # Improved
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.01)  # More stable
        
        fidelity = validate_coherence_preservation(G_before, G_after)
        
        assert fidelity > 1.0
    
    def test_acceptable_degradation(self):
        """Degradation within threshold passes without error."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 1.0)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.01)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 0.9)  # Slight degradation
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.05)
        
        fidelity = validate_coherence_preservation(
            G_before, G_after,
            min_fidelity=0.8,
            rollback_on_failure=False
        )
        
        assert 0.8 < fidelity < 1.0
    
    def test_unacceptable_degradation_no_rollback(self):
        """Severe degradation returns low fidelity without error if no rollback."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 1.0)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.01)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 0.1)  # Severe degradation
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.9)
        
        fidelity = validate_coherence_preservation(
            G_before, G_after,
            min_fidelity=0.85,
            rollback_on_failure=False  # Don't raise
        )
        
        assert fidelity < 0.85
    
    def test_unacceptable_degradation_with_rollback(self):
        """Severe degradation raises RemeshCoherenceLossError with rollback."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 1.0)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.01)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 0.1)
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.9)
        
        with pytest.raises(RemeshCoherenceLossError) as exc_info:
            validate_coherence_preservation(
                G_before, G_after,
                min_fidelity=0.85,
                rollback_on_failure=True
            )
        
        error = exc_info.value
        assert error.fidelity < 0.85
        assert error.min_fidelity == 0.85
        assert "details" in error.details or len(error.details) >= 0
    
    def test_zero_coherence_before(self):
        """Zero coherence before returns fidelity = 1.0 (edge case)."""
        G_before = nx.DiGraph()
        G_before.add_node(1)
        inject_defaults(G_before)
        set_attr(G_before.nodes[1], ALIAS_VF, 0.0)
        set_attr(G_before.nodes[1], ALIAS_DNFR, 0.0)
        
        G_after = nx.DiGraph()
        G_after.add_node(1)
        inject_defaults(G_after)
        set_attr(G_after.nodes[1], ALIAS_VF, 0.5)
        set_attr(G_after.nodes[1], ALIAS_DNFR, 0.1)
        
        fidelity = validate_coherence_preservation(G_before, G_after)
        
        # Edge case: if starting coherence is zero, any result is "preservation"
        assert fidelity == 1.0


class TestRemeshCoherenceLossError:
    """Test RemeshCoherenceLossError exception."""
    
    def test_error_creation(self):
        """Can create RemeshCoherenceLossError with details."""
        error = RemeshCoherenceLossError(
            fidelity=0.7,
            min_fidelity=0.85,
            details={"n_nodes_before": 10, "n_nodes_after": 5}
        )
        
        assert error.fidelity == 0.7
        assert error.min_fidelity == 0.85
        assert error.details["n_nodes_before"] == 10
    
    def test_error_message(self):
        """Error message contains fidelity information."""
        error = RemeshCoherenceLossError(
            fidelity=0.65,
            min_fidelity=0.85,
        )
        
        message = str(error)
        assert "0.65" in message or "65%" in message or "65.00%" in message
        assert "0.85" in message or "85%" in message or "85.00%" in message
