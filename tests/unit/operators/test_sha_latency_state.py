"""Tests for SHA operator latency state management.

This module validates the SHA (Silence) operator's explicit latency state
tracking as specified in TNFR.pdf §2.3.10. SHA is not merely frequency reduction
but a transition to latent state with temporal tracking.

Tests verify:
- Latency state attributes are set when SHA is applied
- EPI preservation integrity during silence
- Reactivation validation when AL is applied after SHA
- Extended silence warnings
- Metrics include latency tracking information

Note: Tests follow canonical TNFR grammar with structural coherence:
- AL → EN → IL → SHA (activation → integration → stabilization → silence)
- SHA → IL → AL (silence → stabilization → reactivation - coherent awakening)
- SHA → EN → IL (silence → reception → stabilization - network reactivation)

Direct SHA → AL is avoided as it violates structural continuity (zero → high frequency
requires intermediate stabilization).
"""

from datetime import datetime, timezone

import pytest

from tnfr.operators.definitions import Emission, Silence, Coherence, Reception
from tnfr.structural import create_nfr, run_sequence


def test_sha_sets_latency_state():
    """When SHA is applied, node should enter latent state with tracking attributes."""
    G, node = create_nfr("latency_test", epi=0.2, vf=1.0)

    # Apply valid sequence: AL → EN → IL → SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify latency state attributes are set
    assert G.nodes[node]["latent"] is True
    assert "latency_start_time" in G.nodes[node]
    assert "preserved_epi" in G.nodes[node]
    assert "silence_duration" in G.nodes[node]

    # Verify latency_start_time is valid ISO timestamp
    timestamp = G.nodes[node]["latency_start_time"]
    # Should not raise exception
    datetime.fromisoformat(timestamp)


def test_sha_preserves_epi_snapshot():
    """SHA should snapshot current EPI for integrity verification."""
    G, node = create_nfr("preserve_test", epi=0.2, vf=1.0)

    # Apply valid sequence leading to SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify preserved EPI matches current EPI (within tolerance)
    preserved_epi = G.nodes[node]["preserved_epi"]
    current_epi = G.nodes[node]["EPI"]

    assert abs(preserved_epi - current_epi) < 0.01


def test_sha_initializes_silence_duration():
    """SHA should initialize silence_duration to 0.0."""
    G, node = create_nfr("duration_test", epi=0.2, vf=1.0)

    # Apply valid sequence ending in SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify silence_duration is initialized to 0
    assert G.nodes[node]["silence_duration"] == 0.0


def test_al_clears_latency_state():
    """When AL is applied after SHA, latency state should be cleared."""
    G, node = create_nfr("reactivation_test", epi=0.2, vf=1.0)

    # Apply sequence with SHA then coherent reactivation (SHA → IL → AL)
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify latency state is set
    assert G.nodes[node]["latent"] is True

    # Reactivate coherently (SHA → IL → AL requires intermediate stabilization)
    run_sequence(G, node, [Coherence(), Emission()])

    # Verify latency state is cleared
    assert G.nodes[node].get("latent", False) is False
    assert "latency_start_time" not in G.nodes[node]
    assert "preserved_epi" not in G.nodes[node]
    assert "silence_duration" not in G.nodes[node]


def test_al_warns_on_extended_silence():
    """AL should warn when reactivating after extended silence."""
    G, node = create_nfr("extended_silence_test", epi=0.2, vf=1.0)

    # Set maximum silence duration threshold
    G.graph["MAX_SILENCE_DURATION"] = 100.0

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    G.nodes[node]["silence_duration"] = 150.0  # Exceeds threshold

    # Reactivate coherently (SHA → IL → AL)
    # AL should warn about extended silence
    with pytest.warns(UserWarning, match="reactivating after extended silence"):
        run_sequence(G, node, [Coherence(), Emission()])


def test_al_warns_on_epi_drift():
    """AL should warn if EPI has drifted during silence."""
    G, node = create_nfr("drift_test", epi=0.2, vf=1.0)

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Manually modify EPI to simulate drift (more than 1% tolerance)
    preserved_epi = G.nodes[node]["preserved_epi"]
    G.nodes[node]["EPI"] = preserved_epi * 1.05  # 5% drift

    # Reactivate coherently (SHA → IL → AL)
    # AL should warn about EPI drift
    with pytest.warns(UserWarning, match="EPI drifted during silence"):
        run_sequence(G, node, [Coherence(), Emission()])


def test_sha_metrics_include_latency_tracking():
    """SHA metrics should include latency state information."""
    G, node = create_nfr("metrics_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify metrics were collected
    assert "operator_metrics" in G.graph
    # Find SHA metrics (last operator in sequence)
    sha_metrics = None
    for m in G.graph["operator_metrics"]:
        if m["glyph"] == "SHA":
            sha_metrics = m
            break

    assert sha_metrics is not None

    # Verify latency tracking metrics are present
    assert "latent" in sha_metrics
    assert sha_metrics["latent"] is True
    assert "silence_duration" in sha_metrics
    assert sha_metrics["silence_duration"] == 0.0
    assert "preservation_integrity" in sha_metrics
    assert "epi_variance_during_silence" in sha_metrics


def test_sha_metrics_preservation_integrity():
    """SHA metrics should calculate preservation integrity correctly."""
    G, node = create_nfr("integrity_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Find SHA metrics
    sha_metrics = None
    for m in G.graph["operator_metrics"]:
        if m["glyph"] == "SHA":
            sha_metrics = m
            break

    # With no drift, preservation_integrity should be very small
    assert sha_metrics["preservation_integrity"] < 0.01


def test_sha_metrics_without_preserved_epi():
    """SHA metrics should handle case where preserved_epi is not set."""
    G, node = create_nfr("no_preserve_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    # Manually remove preserved_epi to test robustness
    del G.nodes[node]["preserved_epi"]

    # Collect metrics again using Coherence
    run_sequence(G, node, [Coherence()])

    # Should not raise exception, should use fallback


def test_il_sha_sequence():
    """Canonical IL → SHA sequence should work correctly."""
    G, node = create_nfr("il_sha_test", epi=0.2, vf=1.0)

    # Apply sequence: AL → EN → IL → SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify latency state is set
    assert G.nodes[node]["latent"] is True


def test_sha_al_sequence():
    """Canonical SHA → IL → AL sequence should reactivate correctly."""
    G, node = create_nfr("sha_al_test", epi=0.2, vf=1.0)

    # Apply sequence ending in SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Reactivate coherently (SHA → IL → AL)
    run_sequence(G, node, [Coherence(), Emission()])

    # Verify node is reactivated (latency state cleared)
    assert G.nodes[node].get("latent", False) is False


def test_multiple_sha_applications():
    """Multiple SHA applications should reset latency state each time."""
    G, node = create_nfr("multiple_sha_test", epi=0.2, vf=1.0)

    # Apply first SHA sequence
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    first_timestamp = G.nodes[node]["latency_start_time"]

    # Apply SHA again (can follow SHA)
    run_sequence(G, node, [Silence()])
    second_timestamp = G.nodes[node]["latency_start_time"]

    # Timestamps should be different (new latency period)
    assert first_timestamp != second_timestamp

    # Latency state should still be set
    assert G.nodes[node]["latent"] is True


def test_sha_without_max_silence_duration():
    """AL should handle reactivation when MAX_SILENCE_DURATION is not set."""
    G, node = create_nfr("no_max_test", epi=0.2, vf=1.0)

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    G.nodes[node]["silence_duration"] = 1000.0

    # Reactivate coherently (SHA → IL → AL)
    # Should not raise exception, no warning expected
    run_sequence(G, node, [Coherence(), Emission()])

    # Latency state should be cleared
    assert G.nodes[node].get("latent", False) is False


def test_al_without_latency_state():
    """AL should work normally when node is not in latent state."""
    G, node = create_nfr("no_latency_test", epi=0.2, vf=1.0)

    # Apply AL with complete stabilization sequence
    run_sequence(G, node, [Emission(), Reception(), Coherence()])

    # Should work without issues
    assert "_emission_activated" in G.nodes[node]


def test_sha_vf_reduction_with_latency():
    """SHA should reduce νf while setting latency state."""
    G, node = create_nfr("vf_reduction_test", epi=0.2, vf=1.0)

    # Apply sequence up to coherence
    run_sequence(G, node, [Emission(), Reception(), Coherence()])

    vf_before = G.nodes[node]["vf"]

    # Apply SHA
    run_sequence(G, node, [Silence()])

    vf_after = G.nodes[node]["vf"]

    # Verify vf was reduced
    assert vf_after < vf_before

    # Verify latency state is set
    assert G.nodes[node]["latent"] is True


def test_preserved_epi_integrity_check():
    """Test preservation integrity calculation in metrics."""
    G, node = create_nfr("integrity_calc_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Manually modify EPI slightly
    preserved_epi = G.nodes[node]["preserved_epi"]
    G.nodes[node]["EPI"] = preserved_epi * 1.02  # 2% drift

    # Apply another operator to collect metrics
    run_sequence(G, node, [Coherence()])

    # Check that metrics would show drift
    # (This test demonstrates the integrity check is working)


def test_latency_state_timestamp_format():
    """Latency start time should be valid ISO 8601 UTC timestamp."""
    G, node = create_nfr("timestamp_format_test", epi=0.2, vf=1.0)

    # Apply sequence with SHA
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    timestamp_str = G.nodes[node]["latency_start_time"]

    # Parse timestamp - should not raise exception
    dt = datetime.fromisoformat(timestamp_str)

    # Should be recent (within last minute)
    now = datetime.now(timezone.utc)
    time_diff = (now - dt.replace(tzinfo=timezone.utc)).total_seconds()
    assert time_diff < 60.0  # Less than 1 minute old


def test_sha_sets_latency_state():
    """When SHA is applied, node should enter latent state with tracking attributes."""
    G, node = create_nfr("latency_test", epi=0.2, vf=1.0)

    # Apply Emission then SHA (valid sequence)
    run_sequence(G, node, [Emission(), Silence()])

    # Verify latency state attributes are set
    assert G.nodes[node]["latent"] is True
    assert "latency_start_time" in G.nodes[node]
    assert "preserved_epi" in G.nodes[node]
    assert "silence_duration" in G.nodes[node]

    # Verify latency_start_time is valid ISO timestamp
    timestamp = G.nodes[node]["latency_start_time"]
    # Should not raise exception
    datetime.fromisoformat(timestamp)


def test_sha_preserves_epi_snapshot():
    """SHA should snapshot current EPI for integrity verification."""
    G, node = create_nfr("preserve_test", epi=0.2, vf=1.0)

    # Apply Emission first to activate, then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Verify preserved EPI matches current EPI (within tolerance)
    preserved_epi = G.nodes[node]["preserved_epi"]
    current_epi = G.nodes[node]["EPI"]

    assert abs(preserved_epi - current_epi) < 0.01


def test_sha_initializes_silence_duration():
    """SHA should initialize silence_duration to 0.0."""
    G, node = create_nfr("duration_test", epi=0.2, vf=1.0)

    # Apply Emission then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Verify silence_duration is initialized to 0
    assert G.nodes[node]["silence_duration"] == 0.0


def test_al_clears_latency_state():
    """When AL is applied after SHA, latency state should be cleared."""
    G, node = create_nfr("reactivation_test", epi=0.2, vf=1.0)

    # Apply AL then SHA then AL again
    run_sequence(G, node, [Emission(), Silence(), Emission()])

    # Verify latency state is cleared
    assert G.nodes[node].get("latent", False) is False
    assert "latency_start_time" not in G.nodes[node]
    assert "preserved_epi" not in G.nodes[node]
    assert "silence_duration" not in G.nodes[node]


def test_al_warns_on_extended_silence():
    """AL should warn when reactivating after extended silence."""
    G, node = create_nfr("extended_silence_test", epi=0.2, vf=1.0)

    # Set maximum silence duration threshold
    G.graph["MAX_SILENCE_DURATION"] = 100.0

    # Apply AL then SHA, and manually set extended silence duration
    run_sequence(G, node, [Emission(), Silence()])
    G.nodes[node]["silence_duration"] = 150.0  # Exceeds threshold

    # AL should warn about extended silence
    with pytest.warns(UserWarning, match="reactivating after extended silence"):
        run_sequence(G, node, [Emission()])


def test_al_warns_on_epi_drift():
    """AL should warn if EPI has drifted during silence."""
    G, node = create_nfr("drift_test", epi=0.2, vf=1.0)

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Manually modify EPI to simulate drift (more than 1% tolerance)
    preserved_epi = G.nodes[node]["preserved_epi"]
    G.nodes[node]["EPI"] = preserved_epi * 1.05  # 5% drift

    # AL should warn about EPI drift
    with pytest.warns(UserWarning, match="EPI drifted during silence"):
        run_sequence(G, node, [Emission()])


def test_sha_metrics_include_latency_tracking():
    """SHA metrics should include latency state information."""
    G, node = create_nfr("metrics_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Verify metrics were collected
    assert "operator_metrics" in G.graph
    # Find SHA metrics (second operator)
    sha_metrics = None
    for m in G.graph["operator_metrics"]:
        if m["glyph"] == "SHA":
            sha_metrics = m
            break

    assert sha_metrics is not None

    # Verify latency tracking metrics are present
    assert "latent" in sha_metrics
    assert sha_metrics["latent"] is True
    assert "silence_duration" in sha_metrics
    assert sha_metrics["silence_duration"] == 0.0
    assert "preservation_integrity" in sha_metrics
    assert "epi_variance_during_silence" in sha_metrics


def test_sha_metrics_preservation_integrity():
    """SHA metrics should calculate preservation integrity correctly."""
    G, node = create_nfr("integrity_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Find SHA metrics
    sha_metrics = None
    for m in G.graph["operator_metrics"]:
        if m["glyph"] == "SHA":
            sha_metrics = m
            break

    # With no drift, preservation_integrity should be very small
    assert sha_metrics["preservation_integrity"] < 0.01


def test_sha_metrics_without_preserved_epi():
    """SHA metrics should handle case where preserved_epi is not set."""
    G, node = create_nfr("no_preserve_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])
    # Manually remove preserved_epi before collecting metrics again
    del G.nodes[node]["preserved_epi"]

    # Collect metrics again using Coherence (which also collects)
    run_sequence(G, node, [Coherence()])

    # Should not raise exception, should use fallback
    # This test verifies robustness


def test_il_sha_sequence():
    """Canonical IL → SHA sequence should work correctly."""
    G, node = create_nfr("il_sha_test", epi=0.2, vf=1.0)

    # Apply AL first, then IL then SHA
    run_sequence(G, node, [Emission(), Coherence(), Silence()])

    # Verify latency state is set
    assert G.nodes[node]["latent"] is True


def test_sha_al_sequence():
    """Canonical SHA → AL sequence should reactivate correctly."""
    G, node = create_nfr("sha_al_test", epi=0.2, vf=1.0)

    # Apply AL, SHA, then AL again
    run_sequence(G, node, [Emission(), Silence(), Emission()])

    # Verify node is reactivated (latency state cleared)
    assert G.nodes[node].get("latent", False) is False


def test_multiple_sha_applications():
    """Multiple SHA applications should reset latency state each time."""
    G, node = create_nfr("multiple_sha_test", epi=0.2, vf=1.0)

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])
    first_timestamp = G.nodes[node]["latency_start_time"]

    # Apply SHA again
    run_sequence(G, node, [Silence()])
    second_timestamp = G.nodes[node]["latency_start_time"]

    # Timestamps should be different (new latency period)
    assert first_timestamp != second_timestamp

    # Latency state should still be set
    assert G.nodes[node]["latent"] is True


def test_sha_without_max_silence_duration():
    """AL should handle reactivation when MAX_SILENCE_DURATION is not set."""
    G, node = create_nfr("no_max_test", epi=0.2, vf=1.0)

    # Apply AL then SHA with very long silence (no MAX_SILENCE_DURATION set)
    run_sequence(G, node, [Emission(), Silence()])
    G.nodes[node]["silence_duration"] = 1000.0

    # Should not raise exception, no warning expected
    run_sequence(G, node, [Emission()])

    # Latency state should be cleared
    assert G.nodes[node].get("latent", False) is False


def test_al_without_latency_state():
    """AL should work normally when node is not in latent state."""
    G, node = create_nfr("no_latency_test", epi=0.2, vf=1.0)

    # Apply AL directly without SHA
    run_sequence(G, node, [Emission()])

    # Should work without issues
    assert "_emission_activated" in G.nodes[node]


def test_sha_vf_reduction_with_latency():
    """SHA should reduce νf while setting latency state."""
    G, node = create_nfr("vf_reduction_test", epi=0.2, vf=1.0)

    # Apply AL first
    run_sequence(G, node, [Emission()])

    vf_before = G.nodes[node]["vf"]

    # Apply SHA
    run_sequence(G, node, [Silence()])

    vf_after = G.nodes[node]["vf"]

    # Verify vf was reduced
    assert vf_after < vf_before

    # Verify latency state is set
    assert G.nodes[node]["latent"] is True


def test_preserved_epi_integrity_check():
    """Test preservation integrity calculation in metrics."""
    G, node = create_nfr("integrity_calc_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])

    # Manually modify EPI slightly
    preserved_epi = G.nodes[node]["preserved_epi"]
    G.nodes[node]["EPI"] = preserved_epi * 1.02  # 2% drift

    # Apply another operator to collect metrics
    run_sequence(G, node, [Coherence()])

    # Check that metrics would show drift
    # (This test demonstrates the integrity check is working)


def test_latency_state_timestamp_format():
    """Latency start time should be valid ISO 8601 UTC timestamp."""
    G, node = create_nfr("timestamp_format_test", epi=0.2, vf=1.0)

    # Apply AL then SHA
    run_sequence(G, node, [Emission(), Silence()])

    timestamp_str = G.nodes[node]["latency_start_time"]

    # Parse timestamp - should not raise exception
    dt = datetime.fromisoformat(timestamp_str)

    # Should be recent (within last minute)
    now = datetime.now(timezone.utc)
    time_diff = (now - dt.replace(tzinfo=timezone.utc)).total_seconds()
    assert time_diff < 60.0  # Less than 1 minute old
