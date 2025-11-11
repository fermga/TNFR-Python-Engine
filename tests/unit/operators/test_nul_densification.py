"""Tests for NUL operator canonical ΔNFR densification dynamics.

This test module validates the canonical structural densification behavior
of the NUL (Contraction) operator as specified in TNFR theory.

According to the canonical implementation:
- Volume reduction: V' = V · λ, where λ < 1 (typically 0.85)
- Density increase: ρ_ΔNFR = ΔNFR / V'
- Result: ΔNFR' = ΔNFR · densification_factor

The densification_factor > 1.0 reflects structural pressure concentration
due to volume reduction, implementing the nodal equation ∂EPI/∂t = νf · ΔNFR(t).
"""

import pytest
import networkx as nx
from tnfr.structural import create_nfr
from tnfr.operators import apply_glyph
from tnfr.types import Glyph
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY


def test_nul_increases_dnfr():
    """Test that NUL operator increases ΔNFR due to densification.

    This is the core canonical behavior: when structure contracts,
    ΔNFR density must increase to preserve structural pressure conservation.
    """
    # Create a node with non-zero ΔNFR
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.1  # Initial ΔNFR

    dnfr_before = G.nodes[node][DNFR_PRIMARY]

    # Apply Contraction operator via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    dnfr_after = G.nodes[node][DNFR_PRIMARY]

    # CRITICAL: ΔNFR must increase (densification)
    assert dnfr_after > dnfr_before, (
        f"NUL must increase ΔNFR (densification). " f"Before: {dnfr_before}, After: {dnfr_after}"
    )


def test_nul_densification_factor():
    """Test that densification factor is applied correctly.

    The densification_factor should be > 1.0 and result in proportional
    ΔNFR increase. Default canonical value is 1.35.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    dnfr_before = G.nodes[node][DNFR_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    dnfr_after = G.nodes[node][DNFR_PRIMARY]

    # Calculate actual densification factor
    actual_factor = dnfr_after / dnfr_before if dnfr_before != 0 else float("inf")

    # Densification factor should be in canonical range [1.3, 1.5]
    assert (
        1.3 <= actual_factor <= 1.5
    ), f"Densification factor {actual_factor:.3f} outside canonical range [1.3, 1.5]"


def test_nul_densification_telemetry():
    """Test that densification telemetry is recorded correctly.

    The operator must track densification events for reproducibility
    and theoretical traceability.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.15

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    # Check telemetry was recorded
    assert "nul_densification_log" in G.graph, "Densification telemetry must be recorded in graph"

    log = G.graph["nul_densification_log"]
    assert len(log) > 0, "Densification log should not be empty"

    entry = log[-1]
    assert "dnfr_before" in entry, "Log must record ΔNFR before densification"
    assert "dnfr_after" in entry, "Log must record ΔNFR after densification"
    assert "densification_factor" in entry, "Log must record densification factor"
    assert "contraction_scale" in entry, "Log must record contraction scale"

    # Verify densification factor is applied correctly
    expected_dnfr_after = entry["dnfr_before"] * entry["densification_factor"]
    assert abs(entry["dnfr_after"] - expected_dnfr_after) < 1e-9, (
        f"ΔNFR after densification doesn't match expected value. "
        f"Expected: {expected_dnfr_after}, Got: {entry['dnfr_after']}"
    )


def test_nul_densification_with_zero_dnfr():
    """Test NUL behavior when ΔNFR = 0 (equilibrium state).

    When ΔNFR = 0, densification still applies but result remains 0.
    This preserves equilibrium states.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.0

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    dnfr_after = G.nodes[node][DNFR_PRIMARY]

    # ΔNFR should remain 0 (0 × densification_factor = 0)
    assert (
        abs(dnfr_after) < 1e-9
    ), f"NUL with ΔNFR=0 should preserve equilibrium. Got ΔNFR={dnfr_after}"


def test_nul_densification_with_negative_dnfr():
    """Test NUL densification with negative ΔNFR (contraction pressure).

    Densification should amplify magnitude while preserving sign.
    Negative ΔNFR represents contraction pressure, which intensifies.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = -0.1

    dnfr_before = G.nodes[node][DNFR_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    dnfr_after = G.nodes[node][DNFR_PRIMARY]

    # ΔNFR should become more negative (amplified contraction pressure)
    assert (
        dnfr_after < dnfr_before
    ), f"NUL should amplify negative ΔNFR. Before: {dnfr_before}, After: {dnfr_after}"

    # Magnitude should increase by densification factor
    magnitude_ratio = abs(dnfr_after) / abs(dnfr_before)
    assert (
        1.3 <= magnitude_ratio <= 1.5
    ), f"Magnitude amplification {magnitude_ratio:.3f} outside canonical range"


def test_nul_vf_reduction_preserved():
    """Test that existing νf reduction behavior is preserved.

    NUL should still reduce νf (structural frequency) as before.
    Densification is an addition, not a replacement.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)

    vf_before = G.nodes[node][VF_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    vf_after = G.nodes[node][VF_PRIMARY]

    # νf should decrease (existing behavior)
    assert vf_after < vf_before, f"NUL must reduce νf. Before: {vf_before}, After: {vf_after}"


def test_nul_epi_reduction_preserved():
    """Test that existing EPI reduction behavior is preserved.

    NUL should still reduce EPI as before.
    Densification is an addition, not a replacement.

    Note: EPI may be returned as dict with continuous/discrete components.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)

    epi_before = G.nodes[node][EPI_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    epi_after = G.nodes[node][EPI_PRIMARY]

    # EPI should decrease (existing behavior)
    # Handle both scalar and dict EPI formats
    if isinstance(epi_after, dict) and "continuous" in epi_after:
        # Complex EPI format - extract real part of continuous component
        epi_after_value = abs(epi_after["continuous"][0])
    else:
        epi_after_value = float(epi_after) if not isinstance(epi_after, complex) else abs(epi_after)

    epi_before_value = (
        float(epi_before) if not isinstance(epi_before, (dict, complex)) else abs(epi_before)
    )

    assert (
        epi_after_value < epi_before_value
    ), f"NUL must reduce EPI. Before: {epi_before_value}, After: {epi_after_value}"


def test_nul_densification_metrics():
    """Test that contraction_metrics includes densification data.

    Metrics collection must expose densification for telemetry and analysis.
    """
    from tnfr.operators.metrics import contraction_metrics

    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    # Collect metrics manually
    metrics = contraction_metrics(G, node, vf_before, epi_before)

    assert metrics["operator"] == "Contraction", "Should be Contraction metrics"
    assert metrics["glyph"] == "NUL", "Should be NUL glyph"

    # Check densification-specific metrics
    assert (
        "densification_factor" in metrics or "dnfr_increase" in metrics
    ), "Metrics should include densification tracking"


def test_nul_nodal_equation_compliance():
    """Test that densification is consistent with nodal equation.

    The nodal equation ∂EPI/∂t = νf · ΔNFR(t) must remain valid.
    When νf decreases and ΔNFR increases, the product should remain bounded.

    This is a sanity check that densification doesn't violate physics.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.1

    vf_before = G.nodes[node][VF_PRIMARY]
    dnfr_before = G.nodes[node][DNFR_PRIMARY]
    product_before = vf_before * dnfr_before

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    vf_after = G.nodes[node][VF_PRIMARY]
    dnfr_after = G.nodes[node][DNFR_PRIMARY]
    product_after = vf_after * dnfr_after

    # The product νf · ΔNFR should change but remain bounded
    # With NUL_scale=0.85 and densification=1.35, product ≈ 0.85 × 1.35 ≈ 1.15
    # This means slight increase in structural pressure despite contraction

    assert abs(product_after) < 10.0, f"Product νf·ΔNFR diverged: {product_after}"

    # Product should be positive if both were positive (directional consistency)
    if vf_before > 0 and dnfr_before > 0:
        assert product_after > 0, "Product sign should be preserved for positive inputs"


def test_nul_custom_densification_factor():
    """Test that custom densification_factor can be configured.

    Users should be able to override the default densification factor
    via GLYPH_FACTORS configuration.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.1

    # Set custom densification factor
    custom_factor = 1.5
    if "GLYPH_FACTORS" not in G.graph:
        from tnfr.config.defaults_core import CoreDefaults

        G.graph["GLYPH_FACTORS"] = CoreDefaults().GLYPH_FACTORS.copy()
    G.graph["GLYPH_FACTORS"]["NUL_densification_factor"] = custom_factor

    dnfr_before = G.nodes[node][DNFR_PRIMARY]

    # Apply Contraction via apply_glyph
    apply_glyph(G, node, Glyph.NUL)

    dnfr_after = G.nodes[node][DNFR_PRIMARY]
    actual_factor = dnfr_after / dnfr_before

    # Should use custom factor
    assert abs(actual_factor - custom_factor) < 0.01, (
        f"Should use custom densification factor {custom_factor}, " f"got {actual_factor:.3f}"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
