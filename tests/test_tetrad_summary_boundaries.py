"""Tetrad telemetry preserves complete samples and finite summary availability."""

import json
import math
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.metrics import tetrad


def test_large_uniform_field_keeps_its_finite_mean_and_zero_spread():
    stats = tetrad._field_statistics({0: 1e308, 1: 1e308}, "low", False)
    assert stats["available"] is True
    assert stats["mean"] == stats["min"] == stats["max"] == 1e308
    assert stats["std"] == 0.0


def test_extreme_optional_statistics_do_not_discard_available_basic_statistics():
    stats = tetrad._field_statistics({0: -1e308, 1: 1e308}, "high", True)
    assert stats["mean"] == 0.0
    assert stats["std"] == 1e308
    assert stats["available"] is True
    assert stats["p50"] is None
    assert "p50" in stats["unavailable_statistics"]
    assert stats["histogram"] is None
    assert "histogram" in stats["unavailable_statistics"]
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("invalid", [True, "2.0", None, math.nan, Fraction(1, 10**400)])
def test_one_invalid_node_makes_the_whole_field_summary_unavailable(invalid):
    stats = tetrad._field_statistics({0: 1.0, 1: invalid}, "high", True)
    assert stats["available"] is False
    assert stats["mean"] is stats["min"] is stats["max"] is stats["std"] is None
    assert "no nodes were filtered" in stats["error"]["message"]
    assert "histogram" not in stats


def test_empty_field_is_unavailable_without_a_zero_mean():
    stats = tetrad._field_statistics({}, "low", False)
    assert stats["available"] is False
    assert stats["mean"] is None
    assert stats["error"]["message"] == "Empty field"


def test_ordinary_percentile_and_histogram_contract_is_preserved():
    stats = tetrad._field_statistics(
        dict(enumerate((1.0, 2.0, 3.0, 4.0, 5.0))), "high", True
    )
    assert stats["mean"] == 3.0
    assert stats["std"] == pytest.approx(math.sqrt(2.0))
    assert (stats["p25"], stats["p50"], stats["p75"]) == (2.0, 3.0, 4.0)
    assert (stats["p10"], stats["p90"], stats["p99"]) == pytest.approx((1.4, 4.6, 4.96))
    histogram = stats["histogram"]
    assert len(histogram["counts"]) == 20
    assert len(histogram["edges"]) == 21
    assert sum(histogram["counts"]) == 5
    assert (histogram["edges"][0], histogram["edges"][-1]) == (1.0, 5.0)
    assert "unavailable_statistics" not in stats


@pytest.mark.parametrize("nodes", [1, 2])
def test_tetrad_retains_unavailable_or_spectral_coherence_length_provenance(nodes):
    snapshot = tetrad.collect_tetrad_snapshot(
        nx.path_graph(nodes), include_histograms=False
    )
    if nodes == 1:
        assert snapshot["xi_c"] is None
        assert snapshot["xi_c_available"] is False
        assert snapshot["xi_c_provenance"]["method"] == "unavailable"
        assert snapshot["xi_c_error"] is not None
    else:
        assert snapshot["xi_c"] == pytest.approx(1 / math.sqrt(2))
        assert snapshot["xi_c_available"] is True
        assert snapshot["xi_c_provenance"]["method"] == "spectral_gap"
        assert "dimensionless" in snapshot["xi_c_provenance"]["distance_weighting"]
        assert snapshot["xi_c_error"] is None
    json.dumps(snapshot, allow_nan=False)


def test_estimator_failure_keeps_its_reason_without_discarding_local_fields(
    monkeypatch,
):
    def refuse(graph):
        raise ValueError("unsupported estimator support")

    monkeypatch.setattr(tetrad, "estimate_coherence_length_with_provenance", refuse)
    snapshot = tetrad.collect_tetrad_snapshot(
        nx.path_graph(2), include_histograms=False
    )
    assert snapshot["phi_s"]["available"] is True
    assert snapshot["xi_c_available"] is False
    assert snapshot["xi_c_provenance"] is None
    assert snapshot["xi_c_error"] == {
        "type": "ValueError",
        "message": "unsupported estimator support",
    }
