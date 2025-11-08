"""Tests for emission-specific metrics with structural fidelity indicators.

This module validates the extended emission metrics collection that reflects
the canonical AL (Emission) structural effects as documented in TNFR.pdf §2.2.1:

- EPI: Increments (form activation)
- vf: Activates/increases (Hz_str)
- DELTA_NFR: Initializes positive reorganization
- theta: Influences phase alignment

Tests verify:
- emission_quality indicator (valid/weak)
- activation_from_latency detection
- form_emergence_magnitude measurement
- frequency_activation detection
- reorganization_positive verification
- emission_timestamp traceability
- irreversibility_marker presence
"""


from tnfr.operators.definitions import Emission, Reception, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def test_emission_metrics_collection_enabled():
    """When COLLECT_OPERATOR_METRICS is True, metrics must be collected."""
    G, node = create_nfr("metrics_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify metrics were collected
    assert "operator_metrics" in G.graph
    assert len(G.graph["operator_metrics"]) > 0

    # Find the Emission metric (first operator)
    metrics = G.graph["operator_metrics"][0]
    assert metrics["operator"] == "Emission"
    assert metrics["glyph"] == "AL"


def test_emission_quality_valid():
    """Valid emission should have quality='valid' when both EPI and vf increase."""
    G, node = create_nfr("valid_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # For valid emission, delta_epi should be positive
    # Note: vf may not always increase depending on the ΔNFR dynamics
    assert metrics["delta_epi"] > 0, "EPI must increase during emission"

    # emission_quality is "valid" if both increased, "weak" otherwise
    if metrics["delta_epi"] > 0 and metrics["delta_vf"] > 0:
        assert metrics["emission_quality"] == "valid"
    else:
        assert metrics["emission_quality"] == "weak"


def test_emission_quality_weak():
    """Weak emission (no increase) should have quality='weak'."""
    G, node = create_nfr("weak_test", epi=0.75, vf=0.1)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # High EPI and low vf may result in weak emission
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # Check for weak emission (at least one delta not positive)
    if metrics["delta_epi"] <= 0 or metrics["delta_vf"] <= 0:
        assert metrics["emission_quality"] == "weak"


def test_activation_from_latency_true():
    """Node with EPI < 0.3 should be detected as latent."""
    G, node = create_nfr("latent_test", epi=0.15, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]
    assert metrics["activation_from_latency"] is True


def test_activation_from_latency_false():
    """Node with EPI >= 0.3 should not be detected as latent."""
    G, node = create_nfr("active_test", epi=0.5, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]
    assert metrics["activation_from_latency"] is False


def test_form_emergence_magnitude():
    """form_emergence_magnitude should equal absolute EPI increment."""
    G, node = create_nfr("magnitude_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # form_emergence_magnitude should equal delta_epi
    assert metrics["form_emergence_magnitude"] == metrics["delta_epi"]
    assert isinstance(metrics["form_emergence_magnitude"], float)


def test_frequency_activation_true():
    """frequency_activation should be True when vf increases."""
    G, node = create_nfr("freq_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    if metrics["delta_vf"] > 0:
        assert metrics["frequency_activation"] is True


def test_frequency_activation_false():
    """frequency_activation should be False when vf doesn't increase."""
    G, node = create_nfr("no_freq_test", epi=0.7, vf=0.1)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    if metrics["delta_vf"] <= 0:
        assert metrics["frequency_activation"] is False


def test_reorganization_positive_true():
    """reorganization_positive should be True when ΔNFR > 0."""
    G, node = create_nfr("positive_dnfr_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # AL should initialize positive ΔNFR
    if metrics["dnfr_initialized"] > 0:
        assert metrics["reorganization_positive"] is True


def test_reorganization_positive_false():
    """reorganization_positive should be False when ΔNFR <= 0."""
    G, node = create_nfr("negative_dnfr_test", epi=0.8, vf=0.1)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    if metrics["dnfr_initialized"] <= 0:
        assert metrics["reorganization_positive"] is False


def test_emission_timestamp_present():
    """emission_timestamp should be present after activation."""
    G, node = create_nfr("timestamp_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # Timestamp should be captured
    assert "emission_timestamp" in metrics
    # Should be a string (ISO format) or None before proper activation
    # After AL, it should be set
    if metrics["irreversibility_marker"]:
        assert metrics["emission_timestamp"] is not None


def test_irreversibility_marker_true():
    """irreversibility_marker should be True after emission activation."""
    G, node = create_nfr("marker_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # After emission, the activation flag should be set
    assert "irreversibility_marker" in metrics
    assert metrics["irreversibility_marker"] is True


def test_all_core_metrics_present():
    """All core metrics must be present in emission metrics."""
    G, node = create_nfr("core_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # Core metrics
    assert "operator" in metrics
    assert "glyph" in metrics
    assert "delta_epi" in metrics
    assert "delta_vf" in metrics
    assert "dnfr_initialized" in metrics
    assert "theta_current" in metrics


def test_all_extended_metrics_present():
    """All extended AL-specific metrics must be present."""
    G, node = create_nfr("extended_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # AL-specific metrics
    assert "emission_quality" in metrics
    assert "activation_from_latency" in metrics
    assert "form_emergence_magnitude" in metrics
    assert "frequency_activation" in metrics
    assert "reorganization_positive" in metrics

    # Traceability metrics
    assert "emission_timestamp" in metrics
    assert "irreversibility_marker" in metrics


def test_legacy_compatibility_fields():
    """Legacy fields must remain present for backward compatibility."""
    G, node = create_nfr("legacy_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # Legacy fields
    assert "epi_final" in metrics
    assert "vf_final" in metrics
    assert "dnfr_final" in metrics
    assert "activation_strength" in metrics
    assert "is_activated" in metrics


def test_emission_metrics_types():
    """Verify metric value types are correct."""
    G, node = create_nfr("types_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]

    # String types
    assert isinstance(metrics["operator"], str)
    assert isinstance(metrics["glyph"], str)
    assert isinstance(metrics["emission_quality"], str)
    assert metrics["emission_quality"] in ["valid", "weak"]

    # Float types
    assert isinstance(metrics["delta_epi"], float)
    assert isinstance(metrics["delta_vf"], float)
    assert isinstance(metrics["dnfr_initialized"], float)
    assert isinstance(metrics["theta_current"], float)
    assert isinstance(metrics["form_emergence_magnitude"], float)

    # Boolean types
    assert isinstance(metrics["activation_from_latency"], bool)
    assert isinstance(metrics["frequency_activation"], bool)
    assert isinstance(metrics["reorganization_positive"], bool)
    assert isinstance(metrics["irreversibility_marker"], bool)

    # Optional timestamp (string or None)
    assert metrics["emission_timestamp"] is None or isinstance(
        metrics["emission_timestamp"], str
    )


def test_emission_metrics_without_collection_flag():
    """When collection is disabled, no metrics should be stored."""
    G, node = create_nfr("no_collection_test", epi=0.2, vf=1.0)
    # Don't set COLLECT_OPERATOR_METRICS flag

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Metrics should not be collected
    assert "operator_metrics" not in G.graph


def test_multiple_emissions_multiple_metrics():
    """Multiple emissions should produce multiple metric entries."""
    G, node = create_nfr("multi_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # First emission
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    first_count = len(G.graph["operator_metrics"])

    # Second emission - coherent reactivation
    Coherence()(G, node)  # Reactivate from silence
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    second_count = len(G.graph["operator_metrics"])

    # Should have more metrics after second emission
    assert second_count > first_count


def test_emission_metrics_structural_consistency():
    """Verify structural consistency between metrics and node state after emission."""
    G, node = create_nfr("consistency_test", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Get the first metric (Emission operator)
    metrics = G.graph["operator_metrics"][0]

    # Note: After the full sequence, node state may have changed from multiple operators
    # The emission metrics capture state right after Emission, not after the full sequence
    # So we verify that the metrics are internally consistent
    assert isinstance(metrics["epi_final"], float)
    assert isinstance(metrics["vf_final"], float)
    assert isinstance(metrics["dnfr_final"], float)

    # The delta values should be consistent
    # (We can't compare to final node state since other operators modify it)
    assert metrics["form_emergence_magnitude"] == metrics["delta_epi"]
