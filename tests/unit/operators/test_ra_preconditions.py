"""Tests for RA (Resonance) operator preconditions and validation.

This module validates the enhanced RA precondition implementation that adds:

1. **Coherent source validation**: EPI >= threshold for propagation
2. **Network connectivity check**: Edges must exist for resonance
3. **Phase compatibility check**: Synchronization with neighbors
4. **Controlled dissonance**: ΔNFR must not be excessive
5. **Sufficient νf**: Structural frequency must support propagation

Tests verify canonical TNFR properties:
- RA requires coherent source (EPI >= threshold)
- RA requires network connectivity (edges)
- RA requires controlled dissonance (ΔNFR <= threshold)
- RA requires sufficient νf for propagation dynamics
- RA warns on phase misalignment (suboptimal resonance)
- Validation can be disabled for backward compatibility
"""

import math
import warnings

import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, THETA_PRIMARY
from tnfr.operators.definitions import (
    Resonance,
    Coupling,
    Emission,
    Reception,
    Coherence,
    Silence,
)
from tnfr.operators.preconditions import (
    validate_resonance,
    diagnose_resonance_readiness,
)
from tnfr.structural import create_nfr, run_sequence


def test_ra_requires_coherent_source():
    """RA should fail when EPI is below minimum threshold."""
    # Create weak source (EPI below threshold)
    G, node = create_nfr("weak_source", epi=0.05, vf=0.9)
    
    # Add neighbor so connectivity check passes
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.1,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # RA should fail due to low EPI
    with pytest.raises(ValueError, match="RA requires coherent source"):
        validate_resonance(G, node)


def test_ra_requires_network_connectivity():
    """RA should fail when node has no edges for propagation."""
    # Create isolated node with sufficient EPI
    G, node = create_nfr("isolated", epi=0.8, vf=0.9)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # RA should fail due to no connectivity
    with pytest.raises(ValueError, match="RA requires network connectivity"):
        validate_resonance(G, node)


def test_ra_requires_controlled_dissonance():
    """RA should fail when ΔNFR is too high (unstable state)."""
    # Create node with high dissonance
    G, node = create_nfr("chaotic", epi=0.8, vf=0.9)
    G.nodes[node][DNFR_PRIMARY] = 0.8  # High dissonance
    
    # Add neighbor
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.1,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # RA should fail due to high dissonance
    with pytest.raises(ValueError, match="RA requires controlled dissonance"):
        validate_resonance(G, node)


def test_ra_requires_sufficient_vf():
    """RA should fail when νf is too low for propagation dynamics."""
    # Create node with very low structural frequency
    G, node = create_nfr("low_vf", epi=0.8, vf=0.005)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Add neighbor
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.1,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # RA should fail due to low νf
    with pytest.raises(ValueError, match="RA requires sufficient structural frequency"):
        validate_resonance(G, node)


def test_ra_warns_phase_misalignment():
    """RA should warn when phase is misaligned with neighbors (suboptimal)."""
    # Create node with phase opposite to neighbor
    G, node = create_nfr("source", epi=0.8, vf=0.9, theta=0.0)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Add neighbor with opposite phase
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: math.pi,  # Opposite phase
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    G.graph["RA_MAX_PHASE_DIFF"] = 1.0  # ~60 degrees threshold
    
    # RA should warn about phase misalignment
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_resonance(G, node)
        
        assert len(w) > 0
        assert "phase misalignment" in str(w[0].message).lower()


def test_ra_passes_with_valid_preconditions():
    """RA should pass validation when all preconditions are met."""
    # Create valid source node
    G, node = create_nfr("source", epi=0.8, vf=0.9, theta=0.2)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Add neighbor with compatible phase
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.25,  # Similar phase
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # RA should pass - no exception raised
    validate_resonance(G, node)  # Should succeed


def test_um_ra_sequence_passes_validation():
    """UM → RA canonical sequence should satisfy all preconditions."""
    # Create source and target with latent state for AL
    G, source = create_nfr("source", epi=0.3, vf=1.0, theta=0.2)  # Lower EPI for AL
    G.nodes[source][DNFR_PRIMARY] = 0.05  # Lower DNFR
    
    target = "target"
    G.add_node(
        target,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.9,
            THETA_PRIMARY: 0.25,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(source, target)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # Valid full sequence: AL → EN → IL → UM → RA → SHA
    run_sequence(
        G,
        source,
        [Emission(), Reception(), Coherence(), Coupling(), Resonance(), Silence()],
    )


def test_al_ra_sequence_passes_validation():
    """AL → RA canonical sequence should satisfy all preconditions."""
    # Create latent node with neighbor  
    G, node = create_nfr("latent", epi=0.2, vf=0.6, theta=0.2)
    G.nodes[node][DNFR_PRIMARY] = 0.0  # Very low initial DNFR
    
    # Adjust EN threshold to allow for AL-induced DNFR increase
    G.graph["DNFR_RECEPTION_MAX"] = 0.3
    
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.25,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # Valid sequence: AL → EN → IL → RA → SHA
    run_sequence(
        G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    )


def test_il_ra_sequence_passes_validation():
    """IL → RA canonical sequence should satisfy all preconditions."""
    # Create active node with moderate dissonance
    G, node = create_nfr("active", epi=0.7, vf=0.9, theta=0.2)
    G.nodes[node][DNFR_PRIMARY] = 0.3
    
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.25,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Enable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # Valid sequence: AL → EN → IL → RA → SHA
    run_sequence(
        G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    )


def test_ra_validation_can_be_disabled():
    """RA validation should be skippable for backward compatibility."""
    # Create invalid node (no connectivity)
    G, node = create_nfr("isolated", epi=0.8, vf=0.9)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Disable validation
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    
    # Valid grammar sequence even though RA preconditions not ideal
    run_sequence(
        G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    )


def test_diagnose_resonance_readiness_reports_failures():
    """diagnose_resonance_readiness should identify all failing checks."""
    # Create node that fails multiple checks
    G, node = create_nfr("problematic", epi=0.05, vf=0.005)  # Low EPI and νf
    G.nodes[node][DNFR_PRIMARY] = 0.8  # High dissonance
    # No neighbors - connectivity fails
    
    diag = diagnose_resonance_readiness(G, node)
    
    assert not diag["ready"], "Node should not be ready"
    assert diag["checks"]["coherent_source"] == "failed"
    assert diag["checks"]["network_connectivity"] == "failed"
    assert diag["checks"]["structural_frequency"] == "failed"
    assert diag["checks"]["controlled_dissonance"] == "failed"
    assert len(diag["recommendations"]) > 0


def test_diagnose_resonance_readiness_reports_ready():
    """diagnose_resonance_readiness should report ready for valid nodes."""
    # Create fully valid node
    G, node = create_nfr("ready", epi=0.8, vf=0.9, theta=0.2)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.25,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    diag = diagnose_resonance_readiness(G, node)
    
    assert diag["ready"], "Node should be ready"
    assert diag["checks"]["coherent_source"] == "passed"
    assert diag["checks"]["network_connectivity"] == "passed"
    assert diag["checks"]["structural_frequency"] == "passed"
    assert diag["checks"]["controlled_dissonance"] == "passed"


def test_diagnose_resonance_readiness_has_canonical_sequences():
    """diagnose_resonance_readiness should provide canonical sequences."""
    G, node = create_nfr("test", epi=0.8, vf=0.9)
    
    diag = diagnose_resonance_readiness(G, node)
    
    assert "canonical_sequences" in diag
    assert len(diag["canonical_sequences"]) > 0
    assert any("UM → RA" in seq for seq in diag["canonical_sequences"])
    assert any("AL → RA" in seq for seq in diag["canonical_sequences"])


def test_ra_custom_thresholds_via_graph_metadata():
    """RA validation should respect custom thresholds in graph metadata."""
    # Create node with EPI = 0.15
    G, node = create_nfr("custom", epi=0.15, vf=0.9, theta=0.2)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            THETA_PRIMARY: 0.25,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Set strict threshold
    G.graph["RA_MIN_SOURCE_EPI"] = 0.2  # Above current EPI
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # Should fail with custom threshold
    with pytest.raises(ValueError, match="RA requires coherent source"):
        validate_resonance(G, node)
    
    # Lower threshold
    G.graph["RA_MIN_SOURCE_EPI"] = 0.1  # Below current EPI
    
    # Should pass now
    validate_resonance(G, node)


def test_ra_isolated_node_warning_when_coupling_not_required():
    """RA should warn about isolated nodes when require_coupling=False."""
    from tnfr.operators.preconditions.resonance import validate_resonance_strict
    
    # Create isolated node
    G, node = create_nfr("isolated", epi=0.8, vf=0.9)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Validate without requiring coupling
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_resonance_strict(G, node, require_coupling=False)
        
        assert len(w) > 0
        assert "isolated" in str(w[0].message).lower()


def test_ra_phase_check_optional():
    """RA phase validation should be optional and not break if unavailable."""
    # Create node without phase data
    G, node = create_nfr("no_phase", epi=0.8, vf=0.9)
    G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Remove phase attribute if it exists
    if THETA_PRIMARY in G.nodes[node]:
        del G.nodes[node][THETA_PRIMARY]
    
    neighbor = "neighbor"
    G.add_node(
        neighbor,
        **{
            EPI_PRIMARY: 0.5,
            VF_PRIMARY: 0.8,
            DNFR_PRIMARY: 0.05,
            "epi_kind": "seed",
        },
    )
    G.add_edge(node, neighbor)
    
    # Should not fail even if phase unavailable
    validate_resonance(G, node)


def test_ra_diagnostic_values_match_node_state():
    """diagnose_resonance_readiness values should match actual node state."""
    # Create node with specific values
    epi_val = 0.75
    vf_val = 0.85
    dnfr_val = 0.15
    theta_val = 0.3
    
    G, node = create_nfr("specific", epi=epi_val, vf=vf_val, theta=theta_val)
    G.nodes[node][DNFR_PRIMARY] = dnfr_val
    
    diag = diagnose_resonance_readiness(G, node)
    
    # Check values match (with tolerance for floating point)
    assert abs(diag["values"]["epi"] - epi_val) < 0.01
    assert abs(diag["values"]["vf"] - vf_val) < 0.01
    assert abs(diag["values"]["dnfr"] - dnfr_val) < 0.01
    assert abs(diag["values"]["theta"] - theta_val) < 0.01


def test_ra_diagnostic_thresholds_match_configuration():
    """diagnose_resonance_readiness should report configured thresholds."""
    G, node = create_nfr("test", epi=0.8, vf=0.9)
    
    # Set custom thresholds
    G.graph["RA_MIN_SOURCE_EPI"] = 0.15
    G.graph["RA_MAX_DISSONANCE"] = 0.6
    G.graph["RA_MIN_VF"] = 0.02
    G.graph["RA_MAX_PHASE_DIFF"] = 1.5
    
    diag = diagnose_resonance_readiness(G, node)
    
    assert diag["thresholds"]["min_epi"] == 0.15
    assert diag["thresholds"]["max_dissonance"] == 0.6
    assert diag["thresholds"]["min_vf"] == 0.02
    assert diag["thresholds"]["max_phase_diff"] == 1.5
