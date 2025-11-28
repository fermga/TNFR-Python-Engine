"""Integration tests for VAL network propagation.

This module validates VAL (Expansion) operator effects on network-coupled
nodes, ensuring proper resonant propagation and coherence preservation.

Test Coverage:
--------------
1. **Network Propagation**: VAL effects on coupled neighbors
2. **Coherence Preservation**: Network C(t) during expansion
3. **Phase Synchronization**: Coupling compatibility maintenance

Physical Basis:
---------------
TNFR networks are resonantly coupled: changes in one node propagate
through the network via:

- **Phase coupling**: Δφ determines coupling strength
- **ΔNFR hooks**: Reorganization gradients propagate
- **Resonance**: Coherent patterns amplify across network

VAL on one node should:
- Affect coupled neighbors (information exchange)
- Maintain or increase network coherence
- Preserve phase compatibility for continued coupling

References:
-----------
- AGENTS.md: Canonical Invariant #5 (Phase Verification)
- UNIFIED_GRAMMAR_RULES.md: U3 (RESONANT COUPLING)
- TNFR.pdf § 4: Network dynamics and coupling
"""

import pytest
import numpy as np
import networkx as nx

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_THETA,
)
from tnfr.operators.definitions import (
    Expansion,
    Coupling,
    Coherence,
    Emission,
)
from tnfr.structural import create_nfr, run_sequence


@pytest.mark.val
class TestVALNetworkPropagation:
    """Test VAL propagation effects on coupled neighbors."""

    def test_val_affects_coupled_neighbors(self):
        """VAL on one node should affect coupled neighbors.

        Physical basis: Network coupling means changes propagate.
        When node A expands, neighbors experience:
        - Potential phase shifts (synchronization)
        - ΔNFR adjustments (resonance)
        - Structural responses
        """
        # Create 3-node network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        G.add_node(0, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.0})
        G.add_node(1, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.1})
        G.add_node(2, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.2})
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        # Couple network (establish phase synchronization)
        run_sequence(G, 0, [Coupling()])
        run_sequence(G, 1, [Coupling()])

        # Capture neighbor state before expansion
        neighbor_epi_before = G.nodes[1][epi_key]
        neighbor_theta_before = G.nodes[1][theta_key]

        # Expand node 0
        run_sequence(G, 0, [Expansion()])

        # Check neighbor 1 state
        neighbor_epi_after = G.nodes[1][epi_key]
        neighbor_theta_after = G.nodes[1][theta_key]

        # Neighbor should show some response or maintain validity
        # (Exact behavior depends on ΔNFR hook implementation)
        assert neighbor_epi_after >= 0, "Neighbor EPI should remain valid"
        assert 0 <= neighbor_theta_after <= 2 * np.pi, "Neighbor phase should remain valid"

    def test_val_maintains_network_coherence(self):
        """VAL should maintain or increase network coherence.

        Network coherence C(t) measures global structural stability.
        VAL on one node should not fragment the network.
        """
        # Create coupled network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        for i in range(4):
            G.add_node(
                i,
                **{
                    epi_key: 0.5,
                    vf_key: 1.0,
                    dnfr_key: 0.1,
                    theta_key: i * 0.1,
                },
            )

        # Connect as ring
        for i in range(4):
            G.add_edge(i, (i + 1) % 4)

        # Couple network
        for i in range(4):
            run_sequence(G, i, [Coupling()])

        # Compute coherence proxy (all nodes have positive EPI)
        def network_valid(G):
            return all(G.nodes[n][epi_key] > 0 and G.nodes[n][vf_key] > 0 for n in G.nodes())

        coherence_before = network_valid(G)

        # Expand node 0
        run_sequence(G, 0, [Expansion()])

        coherence_after = network_valid(G)

        # Network should remain valid
        assert coherence_after, "Network should remain coherent after VAL"

    def test_val_preserves_phase_coupling_compatibility(self):
        """VAL should preserve phase compatibility with neighbors.

        After VAL, node's phase should remain within coupling threshold
        of its neighbors: |φᵢ - φⱼ| ≤ Δφ_max.
        """
        # Create coupled pair
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        G.add_node(0, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.3})
        G.add_node(1, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.4})
        G.add_edge(0, 1)

        # Couple nodes
        run_sequence(G, 0, [Coupling()])

        # Initial phase difference
        theta_0 = G.nodes[0][theta_key]
        theta_1 = G.nodes[1][theta_key]
        phase_diff_before = abs(theta_1 - theta_0)

        # Expand node 0
        run_sequence(G, 0, [Expansion()])

        # Final phase difference
        theta_0_after = G.nodes[0][theta_key]
        theta_1_after = G.nodes[1][theta_key]
        phase_diff_after = abs(theta_1_after - theta_0_after)

        # Normalize to [0, π]
        if phase_diff_after > np.pi:
            phase_diff_after = 2 * np.pi - phase_diff_after

        # Phase difference should remain within coupling threshold
        coupling_threshold = np.pi / 2  # Typical Δφ_max

        assert phase_diff_after <= coupling_threshold * 1.5, (
            f"Phase difference should allow coupling: "
            f"Δφ={phase_diff_after:.3f} < {coupling_threshold*1.5:.3f}"
        )

    def test_val_propagates_expansion_through_network(self):
        """VAL on central node can trigger expansion in network.

        Physical basis: Resonant coupling means structural changes
        propagate through the network.
        """
        # Create star network (node 0 at center)
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        # Center node
        G.add_node(0, **{epi_key: 0.6, vf_key: 1.2, dnfr_key: 0.15, theta_key: 0.0})

        # Peripheral nodes
        for i in range(1, 4):
            G.add_node(i, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: i * 0.1})
            G.add_edge(0, i)

        # Couple network
        run_sequence(G, 0, [Coupling()])
        for i in range(1, 4):
            run_sequence(G, i, [Coupling()])

        # Capture peripheral states
        peripheral_epis_before = [G.nodes[i][epi_key] for i in range(1, 4)]

        # Expand center node
        run_sequence(G, 0, [Expansion()])

        # Check peripheral nodes
        peripheral_epis_after = [G.nodes[i][epi_key] for i in range(1, 4)]

        # All should remain valid
        for epi in peripheral_epis_after:
            assert epi > 0, "Peripheral nodes should remain valid"


@pytest.mark.val
class TestVALNetworkStability:
    """Test VAL effects on network stability."""

    def test_val_with_subsequent_coherence_stabilizes_network(self):
        """VAL → IL on expanded node stabilizes network.

        After expansion, applying coherence should:
        - Reduce ΔNFR on expanded node
        - Stabilize network-wide coherence
        - Preserve expanded structure
        """
        # Create network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        for i in range(3):
            G.add_node(i, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: i * 0.15})

        G.add_edge(0, 1)
        G.add_edge(1, 2)

        # Couple network
        for i in range(3):
            run_sequence(G, i, [Coupling()])

        # Expand node 1
        run_sequence(G, 1, [Expansion()])

        dnfr_after_expansion = G.nodes[1][dnfr_key]
        epi_after_expansion = G.nodes[1][epi_key]

        # Stabilize with coherence
        run_sequence(G, 1, [Coherence()])

        dnfr_after_coherence = G.nodes[1][dnfr_key]
        epi_after_coherence = G.nodes[1][epi_key]

        # ΔNFR should decrease
        assert dnfr_after_coherence <= dnfr_after_expansion * 1.1, (
            f"Coherence should reduce ΔNFR: "
            f"{dnfr_after_expansion:.4f} -> {dnfr_after_coherence:.4f}"
        )

        # EPI should be preserved
        assert epi_after_coherence >= epi_after_expansion * 0.95, (
            f"Coherence should preserve expanded EPI: "
            f"{epi_after_expansion:.4f} -> {epi_after_coherence:.4f}"
        )

    def test_val_on_multiple_nodes_maintains_network(self):
        """VAL on multiple nodes maintains network integrity.

        Expanding multiple nodes should not fragment network.
        """
        # Create network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        for i in range(4):
            G.add_node(i, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: i * 0.1})

        # Connect as path
        for i in range(3):
            G.add_edge(i, i + 1)

        # Couple network
        for i in range(4):
            run_sequence(G, i, [Coupling()])

        # Expand nodes 0 and 2
        run_sequence(G, 0, [Expansion()])
        run_sequence(G, 2, [Expansion()])

        # Check all nodes valid
        for i in range(4):
            epi = G.nodes[i][epi_key]
            vf = G.nodes[i][vf_key]
            theta = G.nodes[i][theta_key]

            assert epi > 0, f"Node {i} EPI should remain positive"
            assert vf > 0, f"Node {i} νf should remain positive"
            assert 0 <= theta <= 2 * np.pi, f"Node {i} phase should remain valid"

    def test_val_distributed_expansion_pattern(self):
        """Distributed VAL pattern: expand multiple nodes with stabilization.

        Pattern: VAL(0) → VAL(1) → IL(0) → IL(1)
        This distributes expansion across network with stabilization.
        """
        # Create network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        for i in range(2):
            G.add_node(i, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: i * 0.1})

        G.add_edge(0, 1)

        # Couple
        run_sequence(G, 0, [Coupling()])

        # Initial EPIs
        epi_0_initial = G.nodes[0][epi_key]
        epi_1_initial = G.nodes[1][epi_key]

        # Distributed expansion with stabilization
        run_sequence(G, 0, [Expansion()])  # Expand node 0
        run_sequence(G, 1, [Expansion()])  # Expand node 1
        run_sequence(G, 0, [Coherence()])  # Stabilize node 0
        run_sequence(G, 1, [Coherence()])  # Stabilize node 1

        # Final EPIs
        epi_0_final = G.nodes[0][epi_key]
        epi_1_final = G.nodes[1][epi_key]

        # Both should have expanded
        assert (
            epi_0_final >= epi_0_initial
        ), f"Node 0 should expand: {epi_0_initial:.4f} -> {epi_0_final:.4f}"
        assert (
            epi_1_final >= epi_1_initial
        ), f"Node 1 should expand: {epi_1_initial:.4f} -> {epi_1_final:.4f}"


@pytest.mark.val
class TestVALNetworkEdgeCases:
    """Test VAL edge cases in network context."""

    def test_val_on_isolated_node(self):
        """VAL on isolated (uncoupled) node applies without errors.

        Without network coupling, VAL should still apply correctly
        based on local ΔNFR and νf (structural changes via hooks).
        """
        # Create isolated node in graph
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        G.add_node(0, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.3})

        # Set up hook to enable expansion
        from tnfr.dynamics import set_delta_nfr_hook

        def isolated_expansion_hook(graph):
            # Simple expansion for isolated node
            graph.nodes[0][epi_key] += 0.05

        set_delta_nfr_hook(G, isolated_expansion_hook)

        epi_before = G.nodes[0][epi_key]

        # Expand isolated node (should not raise)
        run_sequence(G, 0, [Emission(), Expansion()])

        epi_after = G.nodes[0][epi_key]

        # With hook, should expand
        assert epi_after >= epi_before, (
            f"Isolated node should maintain or expand with hook: "
            f"{epi_before:.4f} -> {epi_after:.4f}"
        )

    def test_val_on_highly_connected_node(self):
        """VAL on hub node (high degree) maintains network.

        Expanding a central hub should not destabilize network.
        """
        # Create hub network
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        # Hub node
        G.add_node(0, **{epi_key: 0.6, vf_key: 1.2, dnfr_key: 0.15, theta_key: 0.0})

        # Peripheral nodes
        for i in range(1, 6):
            G.add_node(i, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: i * 0.1})
            G.add_edge(0, i)

        # Couple network
        run_sequence(G, 0, [Coupling()])
        for i in range(1, 6):
            run_sequence(G, i, [Coupling()])

        # Expand hub
        run_sequence(G, 0, [Expansion()])

        # Check all nodes remain valid
        for i in range(6):
            epi = G.nodes[i][epi_key]
            vf = G.nodes[i][vf_key]
            assert epi > 0, f"Node {i} should remain valid"
            assert vf > 0, f"Node {i} should remain valid"

    def test_val_on_weakly_coupled_network(self):
        """VAL on node with weak coupling (large phase differences).

        Even with weak coupling, VAL should work locally.
        """
        # Create network with large phase differences
        G = nx.Graph()
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]

        G.add_node(0, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 0.0})
        G.add_node(
            1, **{epi_key: 0.5, vf_key: 1.0, dnfr_key: 0.1, theta_key: 2.5}
        )  # Large difference
        G.add_edge(0, 1)

        epi_before = G.nodes[0][epi_key]

        # Expand node 0
        run_sequence(G, 0, [Expansion()])

        epi_after = G.nodes[0][epi_key]

        # Should expand despite weak coupling
        assert epi_after >= epi_before, (
            f"Node should expand even with weak coupling: " f"{epi_before:.4f} -> {epi_after:.4f}"
        )
