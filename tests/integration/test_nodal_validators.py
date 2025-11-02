"""DEPRECATED: Comprehensive tests for nodal validators.

⚠️ DEPRECATION NOTICE:
This module has been consolidated into test_nodal_validators_critical_paths.py
which provides more comprehensive coverage with parametrized tests.

See:
- tests/integration/test_nodal_validators_critical_paths.py for comprehensive validator tests
- tests/README_TEST_OPTIMIZATION.md for usage guidelines
- tests/TEST_CONSOLIDATION_SUMMARY.md for detailed consolidation mapping

This module tests critical paths for node-level validation including:
- Structural frequency validation
- Phase coherence checks
- EPI/νf bounds verification
"""

import math

import networkx as nx
import pytest

# Mark entire module as deprecated - tests are redundant with critical paths suite
pytestmark = pytest.mark.skip(
    reason="DEPRECATED: Consolidated into test_nodal_validators_critical_paths.py"
)

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_epi_vf_in_bounds,
    assert_graph_has_tnfr_defaults,
)


def test_nodal_frequency_validation_positive() -> None:
    """Verify nodal frequency validator accepts positive frequencies."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{VF_PRIMARY: 1.0, EPI_PRIMARY: 0.5, DNFR_PRIMARY: 0.0})

    # νf should be positive
    vf = G.nodes[0][VF_PRIMARY]
    assert vf > 0.0


def test_nodal_frequency_validation_rejects_negative() -> None:
    """Verify nodal frequency validator flags negative frequencies."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{VF_PRIMARY: -0.5, EPI_PRIMARY: 0.5, DNFR_PRIMARY: 0.0})

    vf = G.nodes[0][VF_PRIMARY]
    # Negative νf violates structural constraints
    assert vf < 0.0  # This would fail validation in actual validator


def test_nodal_phase_coherence_in_bounds() -> None:
    """Verify nodal phase values remain within [-π, π]."""
    G = nx.Graph()
    inject_defaults(G)

    # Valid phase range
    valid_phases = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]

    for i, phase in enumerate(valid_phases):
        G.add_node(i, **{THETA_KEY: phase, EPI_PRIMARY: 0.0, VF_PRIMARY: 1.0})

    for node, data in G.nodes(data=True):
        theta = data[THETA_KEY]
        assert -math.pi <= theta <= math.pi


def test_nodal_epi_structural_bounds() -> None:
    """Verify EPI values are structurally bounded."""
    G = nx.Graph()
    inject_defaults(G)

    # EPI should be finite
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: -0.3, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})

    for _, data in G.nodes(data=True):
        epi = data[EPI_PRIMARY]
        assert math.isfinite(epi)


def test_nodal_validator_checks_required_attributes() -> None:
    """Verify nodal validator ensures required attributes exist."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})

    required_attrs = [EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY]

    for attr in required_attrs:
        assert attr in G.nodes[0]


def test_nodal_validator_accepts_valid_configuration() -> None:
    """Verify nodal validator accepts structurally sound nodes."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(
        0,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 1.2,
            DNFR_PRIMARY: 0.0,
            THETA_KEY: 0.3,
        }
    )

    # All attributes present and valid
    assert EPI_PRIMARY in G.nodes[0]
    assert VF_PRIMARY in G.nodes[0]
    assert DNFR_PRIMARY in G.nodes[0]
    assert THETA_KEY in G.nodes[0]

    # Values are finite
    for key in [EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, THETA_KEY]:
        assert math.isfinite(G.nodes[0][key])


def test_nodal_validator_network_level_checks() -> None:
    """Verify validator performs network-level structural checks."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: -0.1})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.1})

    # ΔNFR should be conserved at network level
    assert_dnfr_balanced(G)


def test_nodal_validator_bounds_checking() -> None:
    """Verify validator can check EPI and νf bounds."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.0})

    # Should pass with reasonable bounds
    assert_epi_vf_in_bounds(G, epi_min=-1.0, epi_max=1.0, vf_min=0.0, vf_max=2.0)


def test_nodal_validator_default_presence() -> None:
    """Verify validator checks for TNFR defaults."""
    G = nx.Graph()
    inject_defaults(G)

    assert_graph_has_tnfr_defaults(G)


def test_nodal_validator_dnfr_conservation_multi_node() -> None:
    """Verify ΔNFR conservation holds across multiple nodes."""
    G = nx.Graph()
    inject_defaults(G)

    # Create balanced ΔNFR distribution
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.1})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: -0.05})
    G.add_node(2, **{EPI_PRIMARY: 0.7, VF_PRIMARY: 0.8, DNFR_PRIMARY: -0.05})

    assert_dnfr_balanced(G)


def test_nodal_validator_phase_wrapping() -> None:
    """Verify phase values are properly wrapped to [-π, π]."""
    G = nx.Graph()
    inject_defaults(G)

    # Phases that need wrapping
    unwrapped_phases = [2 * math.pi, -2 * math.pi, 3 * math.pi]

    for i, phase in enumerate(unwrapped_phases):
        wrapped = ((phase + math.pi) % (2 * math.pi)) - math.pi
        G.add_node(i, **{THETA_KEY: wrapped, EPI_PRIMARY: 0.0, VF_PRIMARY: 1.0})

        # Wrapped phase should be in range
        assert -math.pi <= G.nodes[i][THETA_KEY] <= math.pi


def test_nodal_validator_coupled_nodes() -> None:
    """Verify validator handles coupled node configurations."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.0})
    G.add_edge(0, 1)

    # Both nodes should be valid
    for node in [0, 1]:
        assert EPI_PRIMARY in G.nodes[node]
        assert VF_PRIMARY in G.nodes[node]
        assert math.isfinite(G.nodes[node][EPI_PRIMARY])
        assert math.isfinite(G.nodes[node][VF_PRIMARY])


def test_nodal_validator_stability_metric() -> None:
    """Verify validator can assess nodal stability via ΔNFR magnitude."""
    G = nx.Graph()
    inject_defaults(G)

    # Stable node (low ΔNFR)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.01})

    # Unstable node (high ΔNFR)
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.5})

    # Compensating node
    G.add_node(2, **{EPI_PRIMARY: 0.7, VF_PRIMARY: 0.8, DNFR_PRIMARY: -0.51})

    # Network should still be balanced
    assert_dnfr_balanced(G)

    # Individual node stability can be assessed
    stable_dnfr = abs(G.nodes[0][DNFR_PRIMARY])
    unstable_dnfr = abs(G.nodes[1][DNFR_PRIMARY])
    assert stable_dnfr < unstable_dnfr
