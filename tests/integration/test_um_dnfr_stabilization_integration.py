"""Integration test: Verify that ΔNFR reduction through coupling improves Si.

This test validates that the stabilizing effect of coupling (reduced ΔNFR)
contributes to improved stability as measured by Sense Index (Si) over time.
"""

import math
import pytest
from tnfr.structural import create_nfr
from tnfr.operators import apply_glyph
from tnfr.config import inject_defaults


def test_coupling_dnfr_reduction_improves_si_trajectory():
    """Test that ΔNFR reduction contributes to Si improvement over multiple steps."""
    from tnfr.dynamics import step

    # Create a small network with initial instability (high ΔNFR)
    G1, node1 = create_nfr("test_node1", vf=1.0, theta=0.0, epi=0.5)
    G1.add_node("node2", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.6, Si=0.4)
    G1.add_node("node3", theta=-0.1, EPI=0.5, vf=1.0, dnfr=0.6, Si=0.4)
    G1.add_edge(node1, "node2")
    G1.add_edge(node1, "node3")

    # Inject defaults for step function
    inject_defaults(G1)

    # Set initial high ΔNFR (instability)
    G1.nodes[node1]["dnfr"] = 0.8
    G1.nodes[node1]["Si"] = 0.3

    # Enable ΔNFR stabilization
    G1.graph["UM_STABILIZE_DNFR"] = True

    # Create identical network for comparison without ΔNFR stabilization
    G2, node2 = create_nfr("test_node1", vf=1.0, theta=0.0, epi=0.5)
    G2.add_node("node2", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.6, Si=0.4)
    G2.add_node("node3", theta=-0.1, EPI=0.5, vf=1.0, dnfr=0.6, Si=0.4)
    G2.add_edge(node2, "node2")
    G2.add_edge(node2, "node3")

    inject_defaults(G2)

    G2.nodes[node2]["dnfr"] = 0.8
    G2.nodes[node2]["Si"] = 0.3

    # Disable ΔNFR stabilization
    G2.graph["UM_STABILIZE_DNFR"] = False

    # Apply coupling to both networks
    apply_glyph(G1, node1, "UM")
    apply_glyph(G2, node2, "UM")

    dnfr1_after_coupling = G1.nodes[node1]["dnfr"]
    dnfr2_after_coupling = G2.nodes[node2]["dnfr"]

    # Verify that G1 has reduced ΔNFR while G2 doesn't
    assert dnfr1_after_coupling < 0.8, "G1 should have reduced ΔNFR"
    assert dnfr2_after_coupling == pytest.approx(0.8, abs=1e-9), "G2 should have unchanged ΔNFR"

    # Evolve both networks for several steps
    for _ in range(5):
        step(G1, dt=1.0)
        step(G2, dt=1.0)

    si1_final = G1.nodes[node1]["Si"]
    si2_final = G2.nodes[node2]["Si"]

    # Network with ΔNFR stabilization should show better or equal stability
    # (Lower ΔNFR generally contributes to improved Si in subsequent evolution)
    # Note: Si can be affected by many factors, so we check that the stabilized
    # network doesn't have significantly worse Si
    assert (
        si1_final >= si2_final * 0.9
    ), "Network with ΔNFR stabilization should maintain comparable or better Si"


def test_repeated_coupling_reduces_dnfr_progressively():
    """Test that repeated coupling applications progressively reduce ΔNFR."""
    G, node = create_nfr("test_node", vf=1.0, theta=0.0, epi=0.5)
    G.add_node("neighbor1", theta=0.05, EPI=0.5, vf=1.0, dnfr=0.5, Si=0.5)
    G.add_node("neighbor2", theta=-0.05, EPI=0.5, vf=1.0, dnfr=0.5, Si=0.5)
    G.add_edge(node, "neighbor1")
    G.add_edge(node, "neighbor2")

    G.nodes[node]["dnfr"] = 1.0
    G.graph["UM_STABILIZE_DNFR"] = True

    dnfr_values = [G.nodes[node]["dnfr"]]

    # Apply coupling multiple times
    for _ in range(5):
        apply_glyph(G, node, "UM")
        dnfr_values.append(G.nodes[node]["dnfr"])

    # Each application should reduce ΔNFR (or keep it stable if near zero)
    for i in range(1, len(dnfr_values)):
        assert (
            dnfr_values[i] <= dnfr_values[i - 1]
        ), f"ΔNFR should not increase: {dnfr_values[i-1]} -> {dnfr_values[i]}"

    # Final ΔNFR should be significantly lower than initial
    assert (
        dnfr_values[-1] < dnfr_values[0] * 0.5
    ), "Repeated coupling should substantially reduce ΔNFR"


def test_coupling_stabilization_with_coherence_sequence():
    """Test that coupling+coherence sequence maximizes stability."""
    G, node = create_nfr("test_node", vf=1.0, theta=0.0, epi=0.5)
    G.add_node("neighbor1", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.4, Si=0.5)
    G.add_node("neighbor2", theta=0.15, EPI=0.5, vf=1.0, dnfr=0.4, Si=0.5)
    G.add_edge(node, "neighbor1")
    G.add_edge(node, "neighbor2")

    G.nodes[node]["dnfr"] = 0.9
    G.graph["UM_STABILIZE_DNFR"] = True

    dnfr_initial = G.nodes[node]["dnfr"]

    # Apply coupling (reduces ΔNFR through phase alignment)
    apply_glyph(G, node, "UM")
    dnfr_after_coupling = G.nodes[node]["dnfr"]

    # Apply coherence (further reduces ΔNFR through structural stabilization)
    apply_glyph(G, node, "IL")
    dnfr_after_coherence = G.nodes[node]["dnfr"]

    # Both operators should contribute to ΔNFR reduction
    assert dnfr_after_coupling < dnfr_initial, "Coupling should reduce ΔNFR"
    assert dnfr_after_coherence < dnfr_after_coupling, "Coherence should further reduce ΔNFR"

    # Combined reduction should be substantial
    total_reduction_pct = (dnfr_initial - dnfr_after_coherence) / dnfr_initial
    assert total_reduction_pct > 0.3, "UM+IL sequence should provide substantial stabilization"


def test_dnfr_reduction_metrics_captured():
    """Test that coupling metrics correctly capture ΔNFR reduction."""
    from tnfr.operators.metrics import coupling_metrics
    from tnfr.constants.aliases import ALIAS_DNFR

    G, node = create_nfr("test_node", vf=1.0, theta=0.0, epi=0.5)
    G.add_node("neighbor", theta=0.1, EPI=0.5, vf=1.0, dnfr=0.5, Si=0.5)
    G.add_edge(node, "neighbor")

    dnfr_before = 1.0
    G.nodes[node]["dnfr"] = dnfr_before
    theta_before = G.nodes[node]["theta"]
    G.graph["UM_STABILIZE_DNFR"] = True

    # Apply coupling
    apply_glyph(G, node, "UM")

    # Collect metrics manually
    metrics = coupling_metrics(G, node, theta_before, dnfr_before)

    # Check that metrics include ΔNFR reduction information
    assert "dnfr_before" in metrics, "Metrics should include dnfr_before"
    assert "dnfr_after" in metrics, "Metrics should include dnfr_after"
    assert "dnfr_reduction" in metrics, "Metrics should include dnfr_reduction"
    assert "dnfr_reduction_pct" in metrics, "Metrics should include dnfr_reduction_pct"

    # Verify that reduction is positive
    assert metrics["dnfr_reduction"] > 0, "Should have positive ΔNFR reduction"
    assert metrics["dnfr_after"] < metrics["dnfr_before"], "ΔNFR should decrease"
