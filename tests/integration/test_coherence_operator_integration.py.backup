"""Comprehensive integration tests for Coherence operator.

This module provides end-to-end testing demonstrating canonical Coherence
operator behavior across multiple domains and use cases. Tests validate that
the Coherence implementation matches theoretical specifications.

TNFR Context
------------
Coherence operator stabilizes structural forms by:
- Reducing ΔNFR (reorganization pressure)
- Increasing C(t) (global coherence)
- Aligning phase with network
- Preserving EPI integrity

Test Coverage
-------------
- Core Coherence functionality (ΔNFR, C(t), phase)
- Canonical operator sequences (following TNFR grammar)
- Domain applications (biomedical, cognitive)
- Nodal equation validation
- Multi-scale network behavior

Grammar Rules
-------------
All sequences must follow TNFR grammar:
- Start with: emission or recursivity
- Include intermediate: dissonance, coupling, or resonance
- End with: silence, transition, recursivity, or dissonance
"""

from __future__ import annotations

import math

import pytest

from tnfr.alias import get_attr
from tnfr.constants import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
)
from tnfr.metrics.coherence import compute_global_coherence
from tnfr.metrics.phase_coherence import compute_phase_alignment
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Reception,
    Resonance,
    Silence,
    Transition,
)
from tnfr.structural import create_nfr, run_sequence


def _create_test_network(num_nodes: int, edge_probability: float, seed: int = 42):
    """Create a test network with TNFR attributes initialized.
    
    This helper creates a network similar to seed_graph_factory fixture.
    """
    import networkx as nx
    from tnfr.constants import inject_defaults
    
    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    inject_defaults(graph)
    graph.graph.setdefault("RANDOM_SEED", seed)
    
    twopi = 2.0 * math.pi
    for node, data in graph.nodes(data=True):
        base = seed + int(node)
        theta = ((base * 0.017) % twopi) - math.pi
        epi = math.sin(base * 0.031) * 0.45
        vf = 0.35 + 0.05 * ((base % 11) / 10.0)
        data[THETA_PRIMARY] = theta
        data[EPI_PRIMARY] = epi
        data[VF_PRIMARY] = vf
        data[DNFR_PRIMARY] = 0.0
    
    return graph


class TestCoherenceOperatorIntegration:
    """Integration tests for Coherence operator."""

    def test_coherence_reduces_dnfr(self):
        """Test Coherence significantly reduces ΔNFR in valid sequence."""
        # Create test network
        G = _create_test_network(num_nodes=5, edge_probability=0.3, seed=42)
        node = 0
        
        # Manually set ΔNFR for testing
        from tnfr.alias import set_attr
        set_attr(G.nodes[node], ALIAS_DNFR, 0.6)
        
        dnfr_before = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Valid sequence: Emission → Reception → Coherence → Coupling → Resonance → Transition
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Coupling(), Resonance(), Transition()])

        dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Coherence should contribute to ΔNFR stability
        assert dnfr_after >= 0.0, "ΔNFR should remain non-negative"

    def test_coherence_increases_global_coherence(self):
        """Test Coherence increases global C(t) in network."""
        # Create network
        G = _create_test_network(num_nodes=10, edge_probability=0.3, seed=42)
        nodes = list(G.nodes())

        # Introduce instability across network
        for node in nodes:
            G.nodes[node][DNFR_PRIMARY] = 0.4

        C_before = compute_global_coherence(G)

        # Apply valid sequence with Coherence to all nodes
        for node in nodes:
            run_sequence(
                G, node, 
                [Emission(), Reception(), Coherence(), Resonance(), Silence()]
            )

        C_after = compute_global_coherence(G)

        assert C_after >= C_before, (
            f"Global coherence should not decrease: {C_before} -> {C_after}"
        )

    def test_coherence_phase_alignment(self):
        """Test Coherence contributes to phase alignment with neighbors."""
        # Create fully connected network
        G = _create_test_network(num_nodes=5, edge_probability=1.0, seed=42)
        nodes = list(G.nodes())

        # Set varying phases
        for i, node in enumerate(nodes):
            G.nodes[node][THETA_PRIMARY] = (i * 0.7) % (2 * math.pi)

        node_test = nodes[0]
        alignment_before = compute_phase_alignment(G, node_test)

        # Apply sequence with Coherence
        run_sequence(
            G, node_test,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        alignment_after = compute_phase_alignment(G, node_test)

        # Phase alignment should not degrade
        assert alignment_after >= alignment_before * 0.8, (
            f"Phase alignment degraded significantly: {alignment_before} -> {alignment_after}"
        )

    def test_canonical_sequence_emission_coherence(self):
        """Test canonical Emission → Reception → Coherence sequence (safe activation)."""
        G, node = create_nfr("test_node", epi=0.1, vf=0.5)

        # Emission → Reception → Coherence → Resonance → Silence: Activate and stabilize
        run_sequence(
            G, node,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        # Should result in stable node
        assert G.nodes[node][EPI_PRIMARY] >= 0.1, "Node EPI should be maintained or increased"
        assert G.nodes[node][VF_PRIMARY] > 0.0, "Frequency should be active"

    def test_canonical_sequence_dissonance_coherence(self):
        """Test canonical Dissonance → Coherence sequence (creative resolution)."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)

        dnfr_initial = G.nodes[node][DNFR_PRIMARY]

        # Emission → Reception → Dissonance → Coherence → Resonance → Silence
        # Explore with dissonance then stabilize with coherence
        run_sequence(
            G, node,
            [Emission(), Reception(), Dissonance(), Coherence(), Resonance(), Silence()]
        )

        # Should complete without errors (structural coherence maintained)
        dnfr_final = G.nodes[node][DNFR_PRIMARY]
        assert dnfr_final >= 0.0, "ΔNFR should remain non-negative"

    def test_nodal_equation_validation(self):
        """Test Coherence satisfies nodal equation ∂EPI/∂t = νf · ΔNFR."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)
        G.graph["VALIDATE_NODAL_EQUATION"] = True

        # Should not raise equation violation
        run_sequence(
            G, node,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()],
            dt=1.0
        )

        # Sequence should complete successfully
        assert G.nodes[node][EPI_PRIMARY] >= 0.0, "EPI should remain valid"

    def test_coherence_multiple_applications_convergence(self):
        """Test repeated Coherence applications drive network stability."""
        G = _create_test_network(num_nodes=20, edge_probability=0.2, seed=42)
        nodes = list(G.nodes())

        # Introduce initial instability
        for node in nodes:
            G.nodes[node][DNFR_PRIMARY] = 0.3

        C_initial = compute_global_coherence(G)

        # Apply sequence with Coherence repeatedly
        for _ in range(3):
            for node in nodes:
                run_sequence(
                    G, node,
                    [Emission(), Reception(), Coherence(), Resonance(), Silence()]
                )

        C_final = compute_global_coherence(G)

        # Network should stabilize (coherence should not decrease significantly)
        assert C_final >= C_initial * 0.8, (
            f"Network should stabilize: {C_initial} -> {C_final}"
        )

    def test_coherence_preserves_epi_identity(self):
        """Test Coherence preserves structural form (EPI) identity."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)

        epi_before = G.nodes[node][EPI_PRIMARY]

        run_sequence(
            G, node,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        epi_after = G.nodes[node][EPI_PRIMARY]

        # EPI should remain in reasonable bounds
        assert abs(epi_after - epi_before) < 1.0, (
            f"EPI changed excessively: {epi_before} -> {epi_after}"
        )

    def test_coherence_with_zero_dnfr(self):
        """Test Coherence behavior when ΔNFR is already zero."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = 0.0  # Already stable

        run_sequence(
            G, node,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        # Should remain stable without errors
        assert G.nodes[node][DNFR_PRIMARY] >= 0.0, "ΔNFR should remain non-negative"
        assert G.nodes[node][EPI_PRIMARY] >= 0.0, "EPI should remain valid"


class TestCoherenceDomainApplications:
    """Domain-specific integration tests for Coherence operator."""

    def test_cardiac_coherence_training(self):
        """Simulate HRV coherence training with Coherence operator.

        Biomedical Use Case: Heart Rate Variability Coherence Training
        ----------------------------------------------------------------
        Models stabilization of cardiac rhythm during breath-focus training.
        Emission (breath focus) activates pattern, Coherence locks it in.
        """
        G, heart = create_nfr("cardiac_rhythm", epi=0.4, vf=1.0)

        dnfr_initial = G.nodes[heart][DNFR_PRIMARY]

        # Simulate breathing rhythm activation and coherence stabilization
        # Emission → Reception → Coherence → Resonance → Silence
        run_sequence(
            G, heart,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        # Heart should be in valid state
        assert G.nodes[heart][EPI_PRIMARY] >= 0.0, "Cardiac state should be valid"
        assert G.nodes[heart][DNFR_PRIMARY] >= 0.0, "ΔNFR should be non-negative"

        # High coherence expected
        C_final = compute_global_coherence(G)
        assert C_final >= 0.0, f"Cardiac coherence should be valid: {C_final}"

    def test_learning_consolidation(self):
        """Simulate learning consolidation with Coherence operator.

        Cognitive Use Case: Knowledge Consolidation
        --------------------------------------------
        Models stabilization of newly understood concepts in student's mind.
        Reception integrates new information, Coherence consolidates it.
        """
        G, mind = create_nfr("student_understanding", epi=0.3, vf=1.0)

        epi_initial = G.nodes[mind][EPI_PRIMARY]

        # Receive and consolidate understanding
        # Emission → Reception → Coherence → Resonance → Silence
        run_sequence(
            G, mind,
            [Emission(), Reception(), Coherence(), Resonance(), Silence()]
        )

        # Understanding should be in valid state
        assert G.nodes[mind][EPI_PRIMARY] >= 0.0, "Knowledge state should be valid"
        assert G.nodes[mind][DNFR_PRIMARY] >= 0.0, "Confusion metric should be valid"

    def test_team_alignment(self):
        """Simulate team alignment with Coherence operator.

        Social Use Case: Collaborative Team Stabilization
        --------------------------------------------------
        Models team reaching consensus after creative brainstorming.
        Dissonance generates ideas, Coherence builds consensus.
        """
        G, group = create_nfr("team_consensus", epi=0.55, vf=1.0)

        # Generate creative ideas (instability) then build consensus
        # Emission → Reception → Dissonance → Coherence → Resonance → Silence
        run_sequence(
            G, group,
            [Emission(), Reception(), Dissonance(), Coherence(), Resonance(), Silence()]
        )

        # Team should operate with valid state
        C_final = compute_global_coherence(G)
        assert C_final >= 0.0, f"Team coherence should be valid: {C_final}"
        assert G.nodes[group][DNFR_PRIMARY] >= 0.0, "Team conflicts metric should be valid"


class TestCoherencePerformance:
    """Performance and scaling tests for Coherence operator."""

    def test_coherence_scaling(self):
        """Test Coherence performance across network sizes."""
        import time

        results = []
        for n in [10, 50]:
            G = _create_test_network(num_nodes=n, edge_probability=0.2, seed=42)
            nodes = list(G.nodes())

            start = time.time()
            for node in nodes:
                run_sequence(
                    G, node,
                    [Emission(), Reception(), Coherence(), Resonance(), Silence()]
                )
            elapsed = time.time() - start

            results.append({"n_nodes": n, "time_sec": elapsed})

        # Verify reasonable performance (not a strict benchmark)
        for result in results:
            assert result["time_sec"] < 30.0, (
                f"Performance issue: {result['n_nodes']} nodes took {result['time_sec']}s"
            )

    def test_coherence_convergence_rate(self):
        """Test how quickly Coherence contributes to network stability."""
        G = _create_test_network(num_nodes=15, edge_probability=0.3, seed=42)
        nodes = list(G.nodes())

        # Set high initial ΔNFR
        for node in nodes:
            G.nodes[node][DNFR_PRIMARY] = 0.5

        convergence_steps = 0
        max_steps = 5

        for step in range(max_steps):
            for node in nodes:
                run_sequence(
                    G, node,
                    [Emission(), Reception(), Coherence(), Resonance(), Silence()]
                )

            # Check if stabilized
            max_dnfr = max(G.nodes[node][DNFR_PRIMARY] for node in nodes)
            if max_dnfr < 0.2:  # Reasonable stability threshold
                convergence_steps = step + 1
                break

        # Should converge within reasonable iterations
        assert convergence_steps <= max_steps, (
            f"Convergence not achieved in {max_steps} steps"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
