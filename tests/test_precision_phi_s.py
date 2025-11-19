"""Tests for precision mode integration with Φ_s computation.

Validates that precision_mode affects ONLY numeric details, never physics.

CRITICAL INVARIANT: Grammar (U1-U6) decisions must be identical across
all precision modes. Only numeric accuracy changes, never semantics.
"""

import pytest
import networkx as nx
import numpy as np

from tnfr.config import (
    get_precision_mode,
    set_precision_mode,
)
from tnfr.physics.fields import compute_structural_potential
from tnfr.metrics.common import compute_coherence


@pytest.fixture(autouse=True)
def reset_precision_mode():
    """Reset precision mode to standard after each test."""
    yield
    set_precision_mode("standard")


def create_test_graph(n=10, seed=42):
    """Create small test graph with ΔNFR values."""
    np.random.seed(seed)
    G = nx.watts_strogatz_graph(n, 3, 0.3, seed=seed)
    
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] = np.random.uniform(-2.0, 2.0)
        G.nodes[node]['vf'] = np.random.uniform(0.5, 2.0)
        G.nodes[node]['phase'] = np.random.uniform(0, 2 * np.pi)
    
    return G


def test_precision_mode_defaults():
    """Verify precision mode defaults to standard."""
    assert get_precision_mode() == "standard"


def test_phi_s_standard_vs_high():
    """Φ_s results should be close but not identical (high vs standard)."""
    G = create_test_graph(n=15)
    
    set_precision_mode("standard")
    phi_s_standard = compute_structural_potential(G, alpha=2.0)
    
    # Change precision to high
    set_precision_mode("high")
    # Use slightly different alpha to bypass cache
    phi_s_high = compute_structural_potential(G, alpha=2.001)
    
    # Should be highly correlated but may differ in precision
    nodes = list(G.nodes())
    vals_standard = [phi_s_standard[n] for n in nodes]
    vals_high = [phi_s_high[n] for n in nodes]
    
    # Correlation should be very high
    corr = np.corrcoef(vals_standard, vals_high)[0, 1]
    assert corr > 0.99, f"Standard/high correlation too low: {corr}"
    
    # But exact values may differ slightly
    # (This is expected; high mode uses log-space for stability)


def test_phi_s_research_mode():
    """Research mode should use extended precision dtype."""
    G = create_test_graph(n=10)
    
    set_precision_mode("research")
    phi_s = compute_structural_potential(G, alpha=2.0)
    
    # Should complete without error
    assert len(phi_s) == G.number_of_nodes()
    
    # Values should be finite
    assert all(np.isfinite(v) for v in phi_s.values())


def test_u6_invariant_across_precision():
    """U6 decisions must be invariant to precision mode.
    
    Critical: Changing precision affects numeric accuracy, never
    whether U6 structural potential confinement criterion is met.
    """
    G = create_test_graph(n=12, seed=123)
    
    # Compute Φ_s drift in standard mode
    set_precision_mode("standard")
    phi_s_before_std = compute_structural_potential(G, alpha=2.0)
    
    # Perturb ΔNFR to create drift
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] *= 1.5
    
    phi_s_after_std = compute_structural_potential(G, alpha=2.001)
    
    drift_std = max(
        abs(phi_s_after_std[n] - phi_s_before_std[n])
        for n in G.nodes()
    )
    
    # Now repeat in high mode
    # Reset ΔNFR
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] /= 1.5
    
    set_precision_mode("high")
    phi_s_before_high = compute_structural_potential(G, alpha=2.002)
    
    for node in G.nodes():
        G.nodes[node]['delta_nfr'] *= 1.5
    
    phi_s_after_high = compute_structural_potential(G, alpha=2.003)
    
    drift_high = max(
        abs(phi_s_after_high[n] - phi_s_before_high[n])
        for n in G.nodes()
    )
    
    # U6 threshold is 2.0
    u6_threshold = 2.0
    
    # Decision should be identical
    violates_u6_std = drift_std >= u6_threshold
    violates_u6_high = drift_high >= u6_threshold
    
    assert violates_u6_std == violates_u6_high, (
        f"U6 decision differs across precision modes: "
        f"standard drift={drift_std:.3f}, high drift={drift_high:.3f}"
    )


def test_coherence_invariant_across_precision():
    """Coherence should be similar across precision modes.
    
    C(t) is computed from variance, which should be stable
    across precision changes.
    """
    G = create_test_graph(n=20, seed=456)
    
    set_precision_mode("standard")
    C_std = compute_coherence(G)
    
    set_precision_mode("high")
    C_high = compute_coherence(G)
    
    # Should agree to within 1%
    assert abs(C_std - C_high) < 0.01, (
        f"Coherence differs too much: std={C_std:.4f}, high={C_high:.4f}"
    )


def test_precision_mode_setter_validation():
    """Invalid precision modes should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid precision mode"):
        set_precision_mode("ultra")  # type: ignore
    
    with pytest.raises(ValueError, match="Invalid precision mode"):
        set_precision_mode("low")  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
