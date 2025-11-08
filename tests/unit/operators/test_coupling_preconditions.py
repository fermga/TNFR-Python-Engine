"""Tests for UM (Coupling) canonical precondition validation.

This module validates the implementation of TNFR canonical precondition
requirements for the Coupling (UM) operator:

1. Graph connectivity: At least one other node exists
2. Active EPI: Node has sufficient structural form (EPI > threshold)
3. Structural frequency: Node has capacity for synchronization (νf > threshold)
4. Phase compatibility (optional): Compatible neighbors when strict checking enabled

The validation can be enabled/disabled via the VALIDATE_OPERATOR_PRECONDITIONS
graph flag to maintain backward compatibility.
"""

import math

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coupling
from tnfr.operators.preconditions import OperatorPreconditionError, validate_coupling
from tnfr.structural import create_nfr


class TestCouplingPreconditions:
    """Test suite for UM canonical precondition validation."""

    def test_validate_coupling_success_basic(self):
        """Validation passes for valid node with sufficient EPI and νf."""
        G, node = create_nfr("active", epi=0.15, vf=0.50)
        # Add another node to make graph have multiple nodes
        G.add_node("other")

        # Should not raise
        validate_coupling(G, node)

    def test_validate_coupling_success_at_epi_threshold(self):
        """Validation passes for node just at EPI threshold."""
        G, node = create_nfr("threshold", epi=0.05, vf=0.50)
        G.add_node("other")

        # Should not raise (0.05 == default threshold)
        validate_coupling(G, node)

    def test_validate_coupling_success_at_vf_threshold(self):
        """Validation passes for node just at νf threshold."""
        G, node = create_nfr("threshold", epi=0.15, vf=0.01)
        G.add_node("other")

        # Should not raise (0.01 == default threshold)
        validate_coupling(G, node)

    def test_validate_coupling_fails_single_node_graph(self):
        """Validation fails when graph has only one node."""
        G, node = create_nfr("solo", epi=0.15, vf=0.50)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "Coupling" in error_msg
        assert "no other nodes" in error_msg

    def test_validate_coupling_fails_low_epi(self):
        """Validation fails when EPI below threshold."""
        G, node = create_nfr("inactive", epi=0.02, vf=0.50)
        G.add_node("other")

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "Coupling" in error_msg
        assert "EPI too low" in error_msg
        assert "0.020" in error_msg
        assert "0.050" in error_msg

    def test_validate_coupling_fails_low_vf(self):
        """Validation fails when νf below threshold."""
        G, node = create_nfr("frozen", epi=0.15, vf=0.005)
        G.add_node("other")

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "Coupling" in error_msg
        assert "Structural frequency too low" in error_msg
        assert "0.005" in error_msg
        assert "0.010" in error_msg

    def test_validate_coupling_custom_epi_threshold(self):
        """Validation respects custom EPI threshold."""
        G, node = create_nfr("custom", epi=0.08, vf=0.50)
        G.add_node("other")

        # Set custom higher threshold
        G.graph["UM_MIN_EPI"] = 0.10

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "0.080" in error_msg
        assert "0.100" in error_msg

    def test_validate_coupling_custom_vf_threshold(self):
        """Validation respects custom νf threshold."""
        G, node = create_nfr("custom", epi=0.15, vf=0.03)
        G.add_node("other")

        # Set custom higher threshold
        G.graph["UM_MIN_VF"] = 0.05

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "0.030" in error_msg
        assert "0.050" in error_msg

    def test_validate_coupling_negative_epi_uses_absolute_value(self):
        """Validation uses absolute value of EPI."""
        G, node = create_nfr("negative", epi=-0.15, vf=0.50)
        G.add_node("other")

        # Should pass: |−0.15| = 0.15 > 0.05
        validate_coupling(G, node)

    def test_validate_coupling_phase_check_disabled_by_default(self):
        """Phase compatibility not checked by default."""
        G, node = create_nfr("node1", epi=0.15, vf=0.50, theta=0.0)
        G.add_node("node2")
        set_attr(G.nodes["node2"], ALIAS_THETA, math.pi)  # Opposite phase
        set_attr(G.nodes["node2"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node2"], ALIAS_VF, 0.50)
        G.add_edge(node, "node2")

        # Should pass even though phase difference is π (max incompatibility)
        validate_coupling(G, node)

    def test_validate_coupling_phase_check_strict_enabled_compatible(self):
        """Strict phase check passes with compatible neighbor."""
        G, node = create_nfr("node1", epi=0.15, vf=0.50, theta=0.5)
        G.add_node("node2")
        set_attr(G.nodes["node2"], ALIAS_THETA, 0.7)  # Close phase
        set_attr(G.nodes["node2"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node2"], ALIAS_VF, 0.50)
        G.add_edge(node, "node2")

        # Enable strict phase checking
        G.graph["UM_STRICT_PHASE_CHECK"] = True

        # Should pass: |0.7 - 0.5| = 0.2 < π/2
        validate_coupling(G, node)

    def test_validate_coupling_phase_check_strict_enabled_incompatible(self):
        """Strict phase check fails with no compatible neighbors."""
        G, node = create_nfr("node1", epi=0.15, vf=0.50, theta=0.0)
        G.add_node("node2")
        set_attr(G.nodes["node2"], ALIAS_THETA, math.pi)  # Opposite phase
        set_attr(G.nodes["node2"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node2"], ALIAS_VF, 0.50)
        G.add_edge(node, "node2")

        # Enable strict phase checking
        G.graph["UM_STRICT_PHASE_CHECK"] = True

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "Coupling" in error_msg
        assert "No phase-compatible neighbors" in error_msg

    def test_validate_coupling_phase_check_strict_no_neighbors(self):
        """Strict phase check passes when node has no neighbors (can create new links)."""
        G, node = create_nfr("isolated", epi=0.15, vf=0.50, theta=0.0)
        G.add_node("other")  # Other node but not connected

        # Enable strict phase checking
        G.graph["UM_STRICT_PHASE_CHECK"] = True

        # Should pass: no neighbors to check, UM can create functional links
        validate_coupling(G, node)

    def test_validate_coupling_phase_check_custom_max_diff(self):
        """Strict phase check respects custom max phase difference."""
        G, node = create_nfr("node1", epi=0.15, vf=0.50, theta=0.0)
        G.add_node("node2")
        set_attr(G.nodes["node2"], ALIAS_THETA, 1.0)  # Phase diff = 1.0
        set_attr(G.nodes["node2"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node2"], ALIAS_VF, 0.50)
        G.add_edge(node, "node2")

        # Enable strict checking with tighter threshold
        G.graph["UM_STRICT_PHASE_CHECK"] = True
        G.graph["UM_MAX_PHASE_DIFF"] = 0.5  # Tighter than default π/2

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_coupling(G, node)

        error_msg = str(exc_info.value)
        assert "0.500" in error_msg  # Custom threshold shown

    def test_validate_coupling_phase_check_multiple_neighbors_one_compatible(self):
        """Strict phase check passes if at least one neighbor is compatible."""
        G, node = create_nfr("node1", epi=0.15, vf=0.50, theta=0.0)

        # Add incompatible neighbor
        G.add_node("node2")
        set_attr(G.nodes["node2"], ALIAS_THETA, math.pi)
        set_attr(G.nodes["node2"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node2"], ALIAS_VF, 0.50)
        G.add_edge(node, "node2")

        # Add compatible neighbor
        G.add_node("node3")
        set_attr(G.nodes["node3"], ALIAS_THETA, 0.3)
        set_attr(G.nodes["node3"], ALIAS_EPI, 0.15)
        set_attr(G.nodes["node3"], ALIAS_VF, 0.50)
        G.add_edge(node, "node3")

        # Enable strict phase checking
        G.graph["UM_STRICT_PHASE_CHECK"] = True

        # Should pass: node3 is compatible (|0.3 - 0.0| < π/2)
        validate_coupling(G, node)


class TestCouplingOperatorWithPreconditions:
    """Test Coupling operator with precondition validation enabled."""

    def test_coupling_operator_validates_when_enabled(self):
        """Coupling operator validates preconditions when flag enabled."""
        G, node = create_nfr("invalid", epi=0.02, vf=0.50)  # Too low EPI
        G.add_node("other")

        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        coupling = Coupling()
        with pytest.raises(OperatorPreconditionError):
            coupling(G, node)

    def test_coupling_operator_skips_validation_when_disabled(self):
        """Coupling operator skips validation when flag disabled."""
        G, node = create_nfr("invalid", epi=0.02, vf=0.50)  # Too low EPI
        G.add_node("other")

        # Disable precondition validation (default)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False

        coupling = Coupling()
        # Should not raise - validation skipped
        coupling(G, node)

    def test_coupling_operator_validates_with_explicit_flag(self):
        """Coupling operator validates when explicitly requested."""
        G, node = create_nfr("invalid", epi=0.02, vf=0.50)
        G.add_node("other")

        # Enable graph-level validation flag
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        coupling = Coupling()
        with pytest.raises(OperatorPreconditionError):
            coupling(G, node, validate_preconditions=True)
