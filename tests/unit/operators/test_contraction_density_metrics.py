"""Tests for NUL operator structural density metrics (Issue #2728).

This test module validates the new structural density metrics added to the
contraction_metrics function, which enable:
- Validation of canonical NUL behavior
- Early warning for over-compression
- Research and analysis workflows
- Improved operator observability

Density metrics include:
- density_before: |ΔNFR| / max(EPI, ε) before contraction
- density_after: |ΔNFR| / max(EPI, ε) after contraction
- densification_ratio: density_after / density_before
- is_critical_density: Warning flag for over-compression
"""

import pytest
from tnfr.structural import create_nfr
from tnfr.operators import apply_glyph
from tnfr.types import Glyph
from tnfr.constants import DNFR_PRIMARY, VF_PRIMARY, EPI_PRIMARY
from tnfr.operators.metrics import contraction_metrics


def test_density_metrics_present():
    """Test that new density metrics are included in contraction_metrics output."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Verify all new density metrics are present
    assert "density_before" in metrics, "density_before metric must be present"
    assert "density_after" in metrics, "density_after metric must be present"
    assert "densification_ratio" in metrics, "densification_ratio metric must be present"
    assert "is_critical_density" in metrics, "is_critical_density metric must be present"


def test_density_calculation_correct():
    """Test that density is calculated correctly as |ΔNFR| / max(EPI, ε)."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]
    dnfr_before = G.nodes[node][DNFR_PRIMARY]

    # Calculate expected density before
    expected_density_before = abs(dnfr_before) / epi_before

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Verify density_before calculation
    assert abs(metrics["density_before"] - expected_density_before) < 1e-9, (
        f"density_before should be {expected_density_before}, " f"got {metrics['density_before']}"
    )

    # Verify density_after is positive and increased
    assert metrics["density_after"] > 0, "density_after should be positive"
    assert (
        metrics["density_after"] > metrics["density_before"]
    ), "density_after should be greater than density_before after contraction"


def test_densification_ratio_correct():
    """Test that densification_ratio is calculated as density_after / density_before."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Calculate expected ratio
    expected_ratio = metrics["density_after"] / metrics["density_before"]

    assert abs(metrics["densification_ratio"] - expected_ratio) < 1e-9, (
        f"densification_ratio should be {expected_ratio}, " f"got {metrics['densification_ratio']}"
    )

    # Ratio should be > 1.0 for canonical contraction
    assert (
        metrics["densification_ratio"] > 1.0
    ), "densification_ratio should exceed 1.0 (density increases during contraction)"


def test_densification_ratio_physical_consistency():
    """Test that densification_ratio aligns with canonical physics.

    The densification_ratio should approximately equal:
    densification_factor / contraction_factor

    This reflects the physics: ΔNFR increases by densification_factor,
    while EPI decreases by contraction_factor, so density increases
    by the ratio of these factors.
    """
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Get densification and contraction factors
    densification_factor = metrics.get("densification_factor", 1.35)
    contraction_factor = metrics["contraction_factor"]

    # Expected ratio from physics
    expected_ratio = densification_factor / contraction_factor

    # Allow some tolerance due to numerical effects
    assert abs(metrics["densification_ratio"] - expected_ratio) < 0.1, (
        f"densification_ratio ({metrics['densification_ratio']:.3f}) should be "
        f"approximately densification_factor / contraction_factor = "
        f"{expected_ratio:.3f}"
    )


def test_critical_density_warning_default_threshold():
    """Test that is_critical_density warns when density exceeds default threshold (5.0)."""
    # Create node with high ΔNFR and low EPI to approach critical density
    G, node = create_nfr("test_node", epi=0.3, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 1.5  # High ΔNFR

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Check if critical density is detected
    # With EPI ≈ 0.255 and ΔNFR ≈ 2.025, density ≈ 7.94 > 5.0
    if metrics["density_after"] > 5.0:
        assert (
            metrics["is_critical_density"] is True
        ), "is_critical_density should be True when density exceeds threshold"
    else:
        assert (
            metrics["is_critical_density"] is False
        ), "is_critical_density should be False when density is below threshold"


def test_critical_density_custom_threshold():
    """Test that custom CRITICAL_DENSITY_THRESHOLD can be configured."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.5  # Moderate ΔNFR

    # Set custom low threshold
    G.graph["CRITICAL_DENSITY_THRESHOLD"] = 1.0

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # With threshold = 1.0 and density ≈ 1.59, should trigger warning
    assert (
        metrics["is_critical_density"] is True
    ), "is_critical_density should respect custom threshold"


def test_density_with_zero_dnfr():
    """Test density metrics when ΔNFR = 0 (equilibrium state)."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.0

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Density should be 0 when ΔNFR = 0
    assert metrics["density_before"] == 0.0, "density_before should be 0 when ΔNFR = 0"
    assert metrics["density_after"] == 0.0, "density_after should be 0 when ΔNFR = 0"

    # Ratio should be inf (or handled gracefully)
    assert metrics["densification_ratio"] == float(
        "inf"
    ), "densification_ratio should be inf when density_before = 0"

    # Critical density should be False for equilibrium
    assert (
        metrics["is_critical_density"] is False
    ), "is_critical_density should be False for equilibrium state"


def test_density_with_negative_dnfr():
    """Test density metrics with negative ΔNFR (contraction pressure)."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = -0.2  # Negative ΔNFR

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Density should use absolute value of ΔNFR
    assert metrics["density_before"] > 0, "density_before should be positive (uses |ΔNFR|)"
    assert metrics["density_after"] > 0, "density_after should be positive (uses |ΔNFR|)"

    # Densification should still occur (magnitude increases)
    assert (
        metrics["density_after"] > metrics["density_before"]
    ), "density_after should exceed density_before even with negative ΔNFR"


def test_density_with_very_small_epi():
    """Test density calculation with very small EPI (near epsilon)."""
    G, node = create_nfr("test_node", epi=1e-10, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.1

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Should use epsilon (1e-9) as denominator, not zero
    # density ≈ 0.1 / 1e-9 = 1e8 (very high)
    assert (
        metrics["density_before"] > 1e7
    ), "density_before should be very high when EPI is near zero"

    # Critical density should definitely be triggered
    assert (
        metrics["is_critical_density"] is True
    ), "is_critical_density should be True for near-zero EPI"


def test_density_metrics_integration_with_existing():
    """Test that new density metrics integrate cleanly with existing metrics."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Verify all expected metrics are present
    expected_keys = {
        # Basic metrics
        "operator",
        "glyph",
        "vf_decrease",
        "vf_final",
        "delta_epi",
        "epi_final",
        "dnfr_final",
        "contraction_factor",
        # Densification metrics
        "densification_factor",
        "dnfr_densified",
        "dnfr_before",
        "dnfr_increase",
        # NEW density metrics
        "density_before",
        "density_after",
        "densification_ratio",
        "is_critical_density",
    }

    actual_keys = set(metrics.keys())
    assert expected_keys.issubset(
        actual_keys
    ), f"Missing expected metrics. Expected: {expected_keys - actual_keys}"

    # Verify metric types
    assert isinstance(metrics["density_before"], float)
    assert isinstance(metrics["density_after"], float)
    assert isinstance(metrics["densification_ratio"], float)
    assert isinstance(metrics["is_critical_density"], bool)


def test_density_metrics_observability():
    """Test that density metrics provide useful observability for analysis."""
    G, node = create_nfr("test_node", epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.3

    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]

    apply_glyph(G, node, Glyph.NUL)

    metrics = contraction_metrics(G, node, vf_before, epi_before)

    # Verify metrics enable validation of canonical NUL behavior
    # 1. Densification occurred
    assert metrics["dnfr_increase"] > 0, "ΔNFR should increase (densification)"

    # 2. Density increased proportionally
    assert metrics["densification_ratio"] > 1.0, "Density should increase"

    # 3. Can detect over-compression risk
    compression_safe = not metrics["is_critical_density"]
    # For this test case, should be safe
    assert compression_safe is True, "Compression should be safe for moderate ΔNFR"

    # 4. Metrics are traceable and reproducible
    assert all(
        k in metrics for k in ["density_before", "density_after", "densification_ratio"]
    ), "All density metrics should be traceable"


def test_density_metrics_support_research_workflows():
    """Test that density metrics support research and analysis use cases."""
    # Simulate a research workflow that tracks density evolution
    densities = []

    for initial_dnfr in [0.1, 0.2, 0.3, 0.4]:
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)
        G.nodes[node][DNFR_PRIMARY] = initial_dnfr

        epi_before = G.nodes[node][EPI_PRIMARY]
        vf_before = G.nodes[node][VF_PRIMARY]

        apply_glyph(G, node, Glyph.NUL)

        metrics = contraction_metrics(G, node, vf_before, epi_before)

        densities.append(
            {
                "initial_dnfr": initial_dnfr,
                "density_before": metrics["density_before"],
                "density_after": metrics["density_after"],
                "densification_ratio": metrics["densification_ratio"],
                "is_critical": metrics["is_critical_density"],
            }
        )

    # Verify we can analyze density evolution
    assert len(densities) == 4, "Should capture all density measurements"

    # Density should scale with initial ΔNFR
    for i in range(len(densities) - 1):
        assert (
            densities[i + 1]["density_before"] > densities[i]["density_before"]
        ), "Density should increase with higher initial ΔNFR"

    # Densification ratio should be relatively consistent (canonical behavior)
    ratios = [d["densification_ratio"] for d in densities]
    avg_ratio = sum(ratios) / len(ratios)
    for ratio in ratios:
        assert (
            abs(ratio - avg_ratio) < 0.5
        ), f"Densification ratio should be consistent across conditions (got {ratio:.3f}, avg {avg_ratio:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
