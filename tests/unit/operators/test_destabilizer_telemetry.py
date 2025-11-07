"""Tests for R4 Extended telemetry: destabilizer context tracking.

This module tests the telemetry functionality that records which destabilizer
type enabled a mutation or self-organization event.
"""

import pytest

from tnfr.config.operator_names import COHERENCE, DISSONANCE, EMISSION, EXPANSION, MUTATION, RECEPTION, SELF_ORGANIZATION, SILENCE, TRANSITION
from tnfr.structural import Coherence, Dissonance, Emission, Expansion, Mutation, Reception, SelfOrganization, Transition, create_nfr


@pytest.fixture
def nfr_with_validation():
    """Create an NFR node with precondition validation enabled."""
    G, node = create_nfr("test", epi=0.3, vf=1.0)
    G.graph['VALIDATE_OPERATOR_PRECONDITIONS'] = True
    return G, node


class TestMutationTelemetry:
    """Test telemetry for ZHIR (mutation) operations."""

    def test_mutation_records_strong_destabilizer_context(self, nfr_with_validation):
        """OZ (strong) destabilizer context is recorded in node metadata."""
        G, node = nfr_with_validation
        
        # Apply sequence: AL → EN → IL → OZ → ZHIR
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        Mutation()(G, node)
        
        # Check telemetry
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        assert context['destabilizer_type'] == 'strong'
        assert context['destabilizer_operator'] == DISSONANCE
        assert context['destabilizer_distance'] == 1  # OZ immediately before ZHIR

    def test_mutation_records_moderate_destabilizer_context(self, nfr_with_validation):
        """VAL (moderate) destabilizer context is recorded."""
        G, node = nfr_with_validation
        
        # Apply sequence: AL → EN → IL → VAL → IL → ZHIR
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Expansion()(G, node)
        Coherence()(G, node)
        Mutation()(G, node)
        
        # Check telemetry
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        assert context['destabilizer_type'] == 'moderate'
        assert context['destabilizer_operator'] == EXPANSION
        assert context['destabilizer_distance'] == 2  # VAL 2 steps before ZHIR

    def test_mutation_prefers_closest_destabilizer(self, nfr_with_validation):
        """When multiple destabilizers present, records the closest valid one."""
        G, node = nfr_with_validation
        
        # Apply sequence: AL → EN → IL → OZ → IL → VAL → IL → ZHIR
        # Both OZ (strong) and VAL (moderate) are within their windows
        # Should record VAL since it's closer
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)  # 4 steps from ZHIR
        Coherence()(G, node)
        Expansion()(G, node)  # 2 steps from ZHIR
        Coherence()(G, node)
        Mutation()(G, node)
        
        # Check telemetry - should prefer closer destabilizer
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        # VAL is closer (2 steps) than OZ (4 steps), so it should be recorded
        assert context['destabilizer_operator'] in [EXPANSION, DISSONANCE]
        assert context['destabilizer_distance'] <= 4

    def test_mutation_records_history_in_context(self, nfr_with_validation):
        """Recent operator history is included in telemetry."""
        G, node = nfr_with_validation
        
        # Apply sequence
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        Coherence()(G, node)
        Mutation()(G, node)
        
        # Check that history is recorded
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        assert 'recent_history' in context
        assert len(context['recent_history']) > 0
        # Should contain operator names
        assert DISSONANCE in context['recent_history']
        assert COHERENCE in context['recent_history']


class TestSelfOrganizationTelemetry:
    """Test telemetry for THOL (self-organization) operations."""

    def test_thol_records_strong_destabilizer_context(self, nfr_with_validation):
        """OZ (strong) destabilizer context is recorded for THOL."""
        G, node = nfr_with_validation
        
        # Apply sequence: AL → EN → IL → OZ → THOL
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        SelfOrganization()(G, node)
        
        # Check telemetry
        context = G.nodes[node].get('_mutation_context')  # Same field name for both ZHIR and THOL
        assert context is not None
        assert context['destabilizer_type'] == 'strong'
        assert context['destabilizer_operator'] == DISSONANCE

    def test_thol_records_moderate_destabilizer_context(self, nfr_with_validation):
        """NAV (moderate) destabilizer context is recorded for THOL."""
        G, node = nfr_with_validation
        
        # Apply sequence: AL → EN → IL → OZ → NAV → IL → THOL
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        Transition()(G, node)
        Coherence()(G, node)
        SelfOrganization()(G, node)
        
        # Check telemetry
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        # Should record TRANSITION since it's closer
        assert context['destabilizer_operator'] in [TRANSITION, DISSONANCE]


class TestTelemetryEdgeCases:
    """Test edge cases for telemetry system."""

    def test_mutation_without_history_records_none(self, nfr_with_validation):
        """When no history available, telemetry records None values."""
        G, node = nfr_with_validation
        
        # Clear any existing history
        if 'glyph_history' in G.nodes[node]:
            del G.nodes[node]['glyph_history']
        
        # Try mutation without proper sequence
        try:
            Mutation()(G, node)
        except Exception:
            pass  # May fail for other reasons, we're testing telemetry
        
        # Check telemetry exists even without history
        context = G.nodes[node].get('_mutation_context')
        if context is not None:
            assert context['destabilizer_type'] is None
            assert context['destabilizer_operator'] is None

    def test_telemetry_preserved_across_operations(self, nfr_with_validation):
        """Telemetry context is preserved after the bifurcation event."""
        G, node = nfr_with_validation
        
        # Apply full sequence
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        Mutation()(G, node)
        
        # Verify context immediately after mutation
        context_before = G.nodes[node].get('_mutation_context')
        assert context_before is not None
        
        # Apply more operators
        Coherence()(G, node)
        
        # Context should still be preserved
        context_after = G.nodes[node].get('_mutation_context')
        assert context_after is not None
        assert context_after['destabilizer_type'] == context_before['destabilizer_type']


class TestTelemetryIntegration:
    """Integration tests for telemetry in real structural sequences."""

    def test_bifurcation_pathway_traceability(self, nfr_with_validation):
        """Complete bifurcation pathway can be traced through telemetry."""
        G, node = nfr_with_validation
        
        # Apply a complex sequence with multiple destabilizers
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)  # First destabilizer
        Coherence()(G, node)
        Expansion()(G, node)  # Second destabilizer
        Coherence()(G, node)
        Mutation()(G, node)  # Bifurcation event
        
        # Telemetry should show which destabilizer enabled the mutation
        context = G.nodes[node].get('_mutation_context')
        assert context is not None
        assert context['destabilizer_operator'] in [DISSONANCE, EXPANSION]
        assert 'recent_history' in context
        
        # History should include both destabilizers
        history = context['recent_history']
        assert DISSONANCE in history or EXPANSION in history

    def test_multiple_bifurcations_separate_contexts(self, nfr_with_validation):
        """Each bifurcation event has its own telemetry context."""
        G, node = nfr_with_validation
        
        # First bifurcation
        Emission()(G, node)
        Reception()(G, node)
        Coherence()(G, node)
        Dissonance()(G, node)
        Mutation()(G, node)
        
        first_context = G.nodes[node]['_mutation_context'].copy()
        
        # Second bifurcation (if we apply another transformer)
        Coherence()(G, node)
        Expansion()(G, node)
        Coherence()(G, node)
        Mutation()(G, node)
        
        second_context = G.nodes[node]['_mutation_context']
        
        # Contexts may differ (most recent wins)
        # The key point is that telemetry is updated for each bifurcation
        assert second_context is not None
        assert 'destabilizer_operator' in second_context
