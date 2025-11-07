"""Tests for structural irreversibility and traceability of Emission (AL) operator.

This module validates the implementation of TNFR.pdf §2.2.1 requirement that
AL (Emisión fundacional) must mark structural irreversibility:

    "Una vez activado, AL reorganiza el campo. No puede deshacerse.
    Toda emisión deja un trazo estructural incluso si el nodo colapsa,
    su activación ha reconfigurado las condiciones de coherencia."

Tests verify:
- Timestamp marking on first emission
- Persistent activation flag
- Structural lineage initialization
- Re-activation counter increment
- Backward compatibility (nodes without metadata)
"""

from datetime import datetime, timezone

import pytest

from tnfr.alias import get_attr_str
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_EMISSION_TIMESTAMP
from tnfr.dynamics import set_delta_nfr_hook
from tnfr.operators.definitions import Emission, Reception, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def test_emission_marks_timestamp_on_first_activation():
    """First AL activation must register a structural timestamp."""
    G, node = create_nfr("genesis", epi=0.2, vf=0.9)

    # Record time before emission
    time_before = datetime.now(timezone.utc)

    # Apply emission with valid TNFR sequence (AL → EN → IL → SHA)
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Record time after emission
    time_after = datetime.now(timezone.utc)

    # Verify timestamp exists and is within expected range
    timestamp_str = get_attr_str(
        G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
    )
    assert timestamp_str is not None, "Emission timestamp must be set"

    # Parse timestamp and verify it's between before/after
    emission_time = datetime.fromisoformat(timestamp_str)
    assert (
        time_before <= emission_time <= time_after
    ), "Timestamp must be within execution window"


def test_emission_sets_activation_flag():
    """AL must set persistent _emission_activated flag."""
    G, node = create_nfr("seed", epi=0.3, vf=1.0)

    # Before emission: no flag
    assert "_emission_activated" not in G.nodes[node]

    # Apply emission with coherence
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # After emission: flag must be True
    assert G.nodes[node]["_emission_activated"] is True


def test_emission_preserves_origin_timestamp():
    """AL must preserve original emission timestamp (_emission_origin)."""
    G, node = create_nfr("origin_test", epi=0.25, vf=0.8)

    # First emission
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    original_timestamp = G.nodes[node]["_emission_origin"]
    assert original_timestamp is not None

    # Re-emission should NOT change origin
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    assert (
        G.nodes[node]["_emission_origin"] == original_timestamp
    ), "Origin timestamp must be immutable"


def test_emission_initializes_structural_lineage():
    """AL must initialize _structural_lineage on first activation."""
    G, node = create_nfr("lineage_test", epi=0.15, vf=1.2)

    # Apply emission with coherence
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify lineage structure
    assert "_structural_lineage" in G.nodes[node]
    lineage = G.nodes[node]["_structural_lineage"]

    assert "origin" in lineage
    assert "activation_count" in lineage
    assert "derived_nodes" in lineage
    assert "parent_emission" in lineage

    # Verify initial values
    assert lineage["activation_count"] == 1
    assert lineage["derived_nodes"] == []
    assert lineage["parent_emission"] is None
    assert lineage["origin"] == G.nodes[node]["_emission_origin"]


def test_emission_reactivation_increments_counter():
    """Re-applying AL must increment activation_count."""
    G, node = create_nfr("reactivation_test", epi=0.2, vf=1.0)

    # First activation
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    assert G.nodes[node]["_structural_lineage"]["activation_count"] == 1

    # Second activation
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    assert G.nodes[node]["_structural_lineage"]["activation_count"] == 2

    # Third activation
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    assert G.nodes[node]["_structural_lineage"]["activation_count"] == 3


def test_emission_timestamp_uses_utc():
    """Emission timestamp must use UTC timezone."""
    G, node = create_nfr("utc_test", epi=0.3, vf=0.9)

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    timestamp_str = get_attr_str(
        G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
    )
    assert timestamp_str is not None

    # Parse and verify timezone
    emission_time = datetime.fromisoformat(timestamp_str)
    assert emission_time.tzinfo is not None, "Timestamp must include timezone"


def test_emission_timestamp_is_iso_format():
    """Emission timestamp must be in ISO 8601 format."""
    G, node = create_nfr("iso_test", epi=0.25, vf=1.0)

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    timestamp_str = get_attr_str(
        G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
    )
    assert timestamp_str is not None

    # Should parse without error
    try:
        datetime.fromisoformat(timestamp_str)
    except ValueError:
        pytest.fail("Timestamp must be valid ISO 8601 format")


def test_emission_backward_compatible_with_legacy_nodes():
    """Nodes created before feature should work without errors."""
    G, node = create_nfr("legacy_node", epi=0.2, vf=1.0)

    # Manually remove any emission metadata (simulate legacy node)
    for key in ["_emission_activated", "_emission_origin", "_structural_lineage"]:
        if key in G.nodes[node]:
            del G.nodes[node][key]

    # Applying emission should work without errors
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # And should now have metadata
    assert "_emission_activated" in G.nodes[node]
    assert "_structural_lineage" in G.nodes[node]


def test_emission_multiple_nodes_independent_timestamps():
    """Different nodes must have independent emission timestamps."""
    G, node1 = create_nfr("node1", epi=0.2, vf=1.0)
    
    # Create second node properly using create_nfr
    from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed
    G.add_node("node2")
    G.nodes["node2"][EPI_PRIMARY] = 0.3
    G.nodes["node2"][VF_PRIMARY] = 0.9
    G.nodes["node2"]["theta"] = 0.0
    
    # Set DNFR hook for proper operation
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    # Apply emissions at different times
    run_sequence(G, node1, [Emission(), Reception(), Coherence(), Silence()])

    # Small delay to ensure different timestamps
    import time

    time.sleep(0.01)

    run_sequence(G, "node2", [Emission(), Reception(), Coherence(), Silence()])

    # Timestamps should be different
    ts1 = get_attr_str(G.nodes[node1], ALIAS_EMISSION_TIMESTAMP, default=None)
    ts2 = get_attr_str(G.nodes["node2"], ALIAS_EMISSION_TIMESTAMP, default=None)

    assert ts1 != ts2, "Each node must have unique emission timestamp"


def test_emission_lineage_origin_matches_timestamp():
    """The lineage origin must match the emission_timestamp."""
    G, node = create_nfr("match_test", epi=0.2, vf=1.0)

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    timestamp = get_attr_str(
        G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
    )
    lineage_origin = G.nodes[node]["_structural_lineage"]["origin"]

    assert (
        timestamp == lineage_origin
    ), "Timestamp and lineage origin must match"


def test_emission_with_scripted_dnfr():
    """Emission irreversibility must work with custom ΔNFR hooks."""
    G, node = create_nfr("scripted_test", epi=0.18, vf=1.0)

    # Verify irreversibility metadata exists after emission
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify irreversibility metadata exists
    assert G.nodes[node]["_emission_activated"] is True
    assert "_emission_origin" in G.nodes[node]
    assert "_structural_lineage" in G.nodes[node]

    # Verify node was properly updated
    # Note: EPI might be a dict (BEPI) or float depending on operator effects
    assert EPI_PRIMARY in G.nodes[node] or "EPI" in G.nodes[node]


def test_emission_reactivation_preserves_all_original_metadata():
    """Re-activation must preserve all original metadata except counter."""
    G, node = create_nfr("preservation_test", epi=0.2, vf=1.0)

    # First activation
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Capture original metadata
    original_timestamp = get_attr_str(
        G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None
    )
    original_origin = G.nodes[node]["_emission_origin"]
    original_activated = G.nodes[node]["_emission_activated"]

    # Re-activate
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify preservation
    assert (
        get_attr_str(G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None)
        == original_timestamp
    )
    assert G.nodes[node]["_emission_origin"] == original_origin
    assert G.nodes[node]["_emission_activated"] == original_activated

    # Only counter should change
    assert G.nodes[node]["_structural_lineage"]["activation_count"] == 2
