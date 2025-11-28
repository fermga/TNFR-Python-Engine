"""Tests for clip-aware nodal equation validation.

This module validates that the nodal equation validation correctly handles
structural_clip interventions through the clip-aware mode.

Related Issues:
- fermga/TNFR-Python-Engine#2665: Validation: Update nodal equation validation for clip-aware mode
- fermga/TNFR-Python-Engine#2661: structural_clip implementation
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.dynamics.structural_clip import structural_clip
from tnfr.operators.nodal_equation import (
    NodalEquationViolation,
    validate_nodal_equation,
)


def _create_test_nfr(
    epi: float = 0.0,
    vf: float = 1.0,
    dnfr: float = 0.0,
    **graph_config,
) -> tuple[nx.Graph, str]:
    """Create a test NFR node with optional graph configuration.

    Parameters
    ----------
    epi : float
        Initial EPI value for the node
    vf : float
        Initial structural frequency (νf) for the node
    dnfr : float
        Initial ΔNFR value for the node
    **graph_config
        Additional graph configuration parameters

    Returns
    -------
    tuple[nx.Graph, str]
        Graph and node identifier
    """
    G = nx.Graph()
    node = "test_node"
    G.add_node(node)
    inject_defaults(G)

    # Set initial structural parameters
    G.nodes[node][EPI_PRIMARY] = epi
    G.nodes[node][VF_PRIMARY] = vf
    G.nodes[node]["ΔNFR"] = dnfr
    G.nodes[node]["Si"] = 0.5

    # Apply graph configuration
    for key, value in graph_config.items():
        G.graph[key] = value

    return G, node


class TestClipAwareMode:
    """Test clip-aware validation mode."""

    def test_clip_aware_mode_passes_when_clip_intervenes(self):
        """Clip-aware mode should pass validation when structural_clip intervenes."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2)

        epi_before = 0.95
        dt = 1.0

        # Theoretical EPI: 0.95 + 1.0 * 0.2 * 1.0 = 1.15 (exceeds boundary)
        # Actual EPI after clip: 1.0
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Validation should pass in clip-aware mode
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Clip-aware mode should pass when clip intervenes"

    def test_clip_aware_mode_with_lower_boundary(self):
        """Clip-aware mode should handle lower boundary clipping."""
        G, node = _create_test_nfr(epi=-0.95, vf=1.0, dnfr=-0.3)

        epi_before = -0.95
        dt = 1.0

        # Theoretical EPI: -0.95 + 1.0 * -0.3 * 1.0 = -1.25 (below boundary)
        # Actual EPI after clip: -1.0
        epi_theoretical = epi_before + (1.0 * -0.3 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Validation should pass in clip-aware mode
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Clip-aware mode should handle lower boundary"

    def test_clip_aware_mode_without_clip_intervention(self):
        """Clip-aware mode should work correctly when no clip occurs."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1)

        epi_before = 0.5
        dt = 1.0

        # Theoretical EPI: 0.5 + 1.0 * 0.1 * 1.0 = 0.6 (within bounds)
        epi_actual = epi_before + (1.0 * 0.1 * dt)

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Validation should pass (no clip needed)
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Clip-aware mode should work when no clip occurs"

    def test_clip_aware_mode_with_soft_clipping(self):
        """Clip-aware mode should work with soft clipping mode."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2, CLIP_MODE="soft")

        epi_before = 0.95
        dt = 1.0

        # Theoretical EPI exceeds boundary
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="soft")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Validation should pass with soft clip mode
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Clip-aware mode should work with soft clipping"


class TestClassicMode:
    """Test classic validation mode (clip_aware=False)."""

    def test_classic_mode_detects_clip_intervention(self):
        """Classic mode should detect when clip intervenes."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2)

        epi_before = 0.95
        dt = 1.0

        # Theoretical EPI: 1.15, Actual after clip: 1.0
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Classic mode should fail (detects clip intervention)
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=False,
            tolerance=1e-9,
        )

        assert result is False, "Classic mode should detect clip intervention"

    def test_classic_mode_passes_without_clip(self):
        """Classic mode should pass when no clip occurs."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1)

        epi_before = 0.5
        dt = 1.0

        # No clip needed
        epi_actual = epi_before + (1.0 * 0.1 * dt)

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Classic mode should pass
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=False,
            tolerance=1e-9,
        )

        assert result is True, "Classic mode should pass without clip"

    def test_classic_mode_with_strict_raises_exception(self):
        """Classic mode with strict=True should raise exception on clip."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2)

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should raise exception in strict mode
        with pytest.raises(NodalEquationViolation) as exc_info:
            validate_nodal_equation(
                G,
                node,
                epi_before,
                epi_actual,
                dt,
                operator_name="test",
                clip_aware=False,
                strict=True,
                tolerance=1e-9,
            )

        # Verify exception details
        assert exc_info.value.operator == "test"
        assert exc_info.value.details["clip_aware"] is False


class TestGraphConfiguration:
    """Test validation using graph configuration."""

    def test_graph_config_clip_aware_true(self):
        """Graph configuration NODAL_EQUATION_CLIP_AWARE=True should enable clip-aware mode."""
        G, node = _create_test_nfr(
            epi=0.95,
            vf=1.0,
            dnfr=0.2,
            NODAL_EQUATION_CLIP_AWARE=True,
            NODAL_EQUATION_TOLERANCE=1e-9,
        )

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should use graph config (clip-aware mode)
        result = validate_nodal_equation(G, node, epi_before, epi_actual, dt, operator_name="test")

        assert result is True, "Graph config should enable clip-aware mode"

    def test_graph_config_clip_aware_false(self):
        """Graph configuration NODAL_EQUATION_CLIP_AWARE=False should use classic mode."""
        G, node = _create_test_nfr(
            epi=0.95,
            vf=1.0,
            dnfr=0.2,
            NODAL_EQUATION_CLIP_AWARE=False,
            NODAL_EQUATION_TOLERANCE=1e-9,
        )

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should use graph config (classic mode)
        result = validate_nodal_equation(G, node, epi_before, epi_actual, dt, operator_name="test")

        assert result is False, "Graph config should enforce classic mode"

    def test_explicit_parameter_overrides_graph_config(self):
        """Explicit clip_aware parameter should override graph configuration."""
        G, node = _create_test_nfr(
            epi=0.95,
            vf=1.0,
            dnfr=0.2,
            NODAL_EQUATION_CLIP_AWARE=False,  # Graph says classic mode
        )

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Explicit parameter overrides graph config
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,  # Override to clip-aware mode
        )

        assert result is True, "Explicit parameter should override graph config"


class TestToleranceLevels:
    """Test different tolerance levels."""

    def test_tight_tolerance_clip_aware(self):
        """Tight tolerance should work with clip-aware mode."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2)

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Very tight tolerance (1e-12)
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-12,
        )

        assert result is True, "Tight tolerance should work with clip-aware mode"

    def test_loose_tolerance_classic_mode(self):
        """Loose tolerance in classic mode should pass near-boundary cases."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.05)

        epi_before = 0.95
        dt = 1.0

        # No clip, but close to boundary
        epi_actual = epi_before + (1.0 * 0.05 * dt)  # = 1.0

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Loose tolerance
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=False,
            tolerance=1e-3,
        )

        assert result is True, "Loose tolerance should allow small errors"

    def test_tolerance_from_graph_config(self):
        """Tolerance should be read from graph configuration."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1, NODAL_EQUATION_TOLERANCE=1e-6)

        epi_before = 0.5
        dt = 1.0

        # Small error (within 1e-6)
        epi_actual = epi_before + (1.0 * 0.1 * dt) + 1e-8

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should use graph tolerance
        result = validate_nodal_equation(G, node, epi_before, epi_actual, dt, operator_name="test")

        assert result is True, "Should use tolerance from graph config"


class TestCustomBoundaries:
    """Test validation with custom EPI boundaries."""

    def test_custom_boundaries_upper(self):
        """Clip-aware mode should respect custom upper boundary."""
        G, node = _create_test_nfr(epi=0.45, vf=1.0, dnfr=0.2, EPI_MAX=0.5)  # Custom upper boundary

        epi_before = 0.45
        dt = 1.0

        # Theoretical: 0.45 + 0.2 = 0.65 (exceeds custom boundary 0.5)
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 0.5, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should pass with custom boundary
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Should respect custom upper boundary"
        assert epi_actual == 0.5, "EPI should be clipped to custom boundary"

    def test_custom_boundaries_lower(self):
        """Clip-aware mode should respect custom lower boundary."""
        G, node = _create_test_nfr(
            epi=-0.45, vf=1.0, dnfr=-0.2, EPI_MIN=-0.5  # Custom lower boundary
        )

        epi_before = -0.45
        dt = 1.0

        # Theoretical: -0.45 - 0.2 = -0.65 (below custom boundary -0.5)
        epi_theoretical = epi_before + (1.0 * -0.2 * dt)
        epi_actual = structural_clip(epi_theoretical, -0.5, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should pass with custom boundary
        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Should respect custom lower boundary"
        assert epi_actual == -0.5, "EPI should be clipped to custom boundary"


class TestEdgeCases:
    """Test edge cases and corner conditions."""

    def test_zero_dt(self):
        """Validation should handle dt=0 gracefully."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1)

        epi_before = 0.5
        epi_after = 0.5  # No change expected
        dt = 0.0

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_after

        result = validate_nodal_equation(
            G, node, epi_before, epi_after, dt, operator_name="test", clip_aware=True
        )

        assert result is True, "Should handle dt=0"

    def test_zero_dnfr(self):
        """Validation should handle ΔNFR=0 (no structural pressure)."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.0)

        epi_before = 0.5
        epi_after = 0.5  # No change expected
        dt = 1.0

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_after

        result = validate_nodal_equation(
            G, node, epi_before, epi_after, dt, operator_name="test", clip_aware=True
        )

        assert result is True, "Should handle ΔNFR=0"

    def test_at_exact_boundary(self):
        """Validation should handle exact boundary values."""
        G, node = _create_test_nfr(epi=1.0, vf=1.0, dnfr=0.1)

        epi_before = 1.0
        dt = 1.0

        # Already at boundary, should stay there
        epi_theoretical = epi_before + (1.0 * 0.1 * dt)
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        result = validate_nodal_equation(
            G,
            node,
            epi_before,
            epi_actual,
            dt,
            operator_name="test",
            clip_aware=True,
            tolerance=1e-9,
        )

        assert result is True, "Should handle exact boundary"
        assert epi_actual == 1.0, "Should remain at boundary"

    def test_invalid_clip_mode_falls_back_to_hard(self):
        """Invalid CLIP_MODE should fall back to 'hard' mode."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0, dnfr=0.2, CLIP_MODE="invalid_mode")

        epi_before = 0.95
        dt = 1.0

        # Clip intervenes
        epi_theoretical = epi_before + (1.0 * 0.2 * dt)
        # Should use 'hard' mode as fallback
        epi_actual = structural_clip(epi_theoretical, -1.0, 1.0, mode="hard")

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        # Should still validate correctly with fallback
        result = validate_nodal_equation(
            G, node, epi_before, epi_actual, dt, operator_name="test", clip_aware=True
        )

        assert result is True, "Should handle invalid clip_mode gracefully"


class TestExceptionDetails:
    """Test exception details in strict mode."""

    def test_clip_aware_exception_includes_details(self):
        """Clip-aware mode exception should include relevant details."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1)

        epi_before = 0.5
        # Intentionally wrong value to trigger exception
        epi_actual = 0.8  # Should be 0.6
        dt = 1.0

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        with pytest.raises(NodalEquationViolation) as exc_info:
            validate_nodal_equation(
                G,
                node,
                epi_before,
                epi_actual,
                dt,
                operator_name="test",
                clip_aware=True,
                strict=True,
                tolerance=1e-9,
            )

        # Check exception details
        details = exc_info.value.details
        assert "clip_aware" in details
        assert details["clip_aware"] is True
        assert "epi_theoretical" in details
        assert "epi_expected" in details
        assert "clip_intervened" in details

    def test_classic_mode_exception_includes_details(self):
        """Classic mode exception should include relevant details."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0, dnfr=0.1)

        epi_before = 0.5
        # Intentionally wrong value to trigger exception
        epi_actual = 0.8
        dt = 1.0

        # Update node
        G.nodes[node][EPI_PRIMARY] = epi_actual

        with pytest.raises(NodalEquationViolation) as exc_info:
            validate_nodal_equation(
                G,
                node,
                epi_before,
                epi_actual,
                dt,
                operator_name="test",
                clip_aware=False,
                strict=True,
                tolerance=1e-9,
            )

        # Check exception details
        details = exc_info.value.details
        assert "clip_aware" in details
        assert details["clip_aware"] is False
        # Classic mode should not include clip-specific fields in details
        assert "clip_intervened" not in details
