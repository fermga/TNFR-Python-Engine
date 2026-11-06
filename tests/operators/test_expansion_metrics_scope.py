"""VAL telemetry distinguishes configured policies from missing evidence."""

import json
import pickle
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_EPI
from tnfr.operators.expansion import Expansion
from tnfr.operators.metrics import expansion_metrics


def _graph(*, epi=1.125, capacity=1.125, acceleration=0.1):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=epi, nu_f=capacity, theta=0.0, delta_nfr=0.1)
    if acceleration is not None:
        graph.nodes[0][ALIAS_D2EPI[0]] = acceleration
    return graph


def _state(graph):
    # The existing neighbor reader may rebuild private trigonometric caches.
    metadata = {
        key: value for key, value in graph.graph.items() if not key.startswith("_")
    }
    return pickle.dumps(
        (metadata, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


@pytest.mark.parametrize("sign", (-1, 1))
def test_signed_growth_and_compatibility_labels_are_configured_policies(sign):
    graph = _graph(epi=sign * 1.125)
    before = _state(graph)
    metrics = expansion_metrics(graph, 0, 1.0, sign * 1.0)
    assert metrics["epi_growth_rate"] == metrics["vf_growth_rate"] == 0.125
    assert metrics["growth_ratio"] == 1
    assert metrics["growth_ratio_within_policy"] is metrics["fractal_preserved"] is True
    assert (
        metrics["coherence_above_threshold"] is metrics["coherence_preserved"] is True
    )
    assert metrics["acceleration_source"] == "stored_unverified"
    assert metrics["assessment_status"] == "pass"
    assert metrics["expansion_healthy"] is True
    assert metrics["unavailable_policies"] == metrics["failed_policies"] == ()
    assert "not_fractality" in metrics["growth_policy_scope"]
    assert "without_pre_event_baseline" in metrics["coherence_scope"]
    assert _state(graph) == before
    json.dumps(metrics, allow_nan=False)


def test_negative_form_growth_cannot_bypass_the_ratio_policy():
    metrics = expansion_metrics(_graph(epi=-2.0), 0, 1.0, -1.0)
    assert metrics["epi_growth_rate"] == 1
    assert metrics["growth_ratio"] == 0.125
    assert metrics["fractal_preserved"] is False
    assert metrics["expansion_healthy"] is False
    assert metrics["failed_policies"] == ("growth_ratio_within_policy",)


def test_missing_acceleration_is_unavailable_not_a_healthy_zero():
    graph = _graph(acceleration=None)
    before = _state(graph)
    # Exercise the public operator's optional collector through its facade.
    metrics = Expansion()._collect_metrics(graph, 0, {"epi": 1.0, "vf": 1.0})
    for key in (
        "d2epi",
        "bifurcation_risk",
        "bifurcation_magnitude",
        "acceleration_source",
    ):
        assert metrics[key] is None
    assert metrics["expansion_healthy"] is None
    assert metrics["assessment_status"] == "unavailable"
    assert metrics["unavailable_policies"] == ("acceleration_below_threshold",)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "epi_before,epi_after,vf_before", ((0, 1, 1), (1, 1, 1), (1, 1.125, 0))
)
def test_undefined_growth_denominators_remain_unavailable(
    epi_before, epi_after, vf_before
):
    metrics = expansion_metrics(_graph(epi=epi_after), 0, vf_before, epi_before)
    assert metrics["growth_ratio"] is None
    assert metrics["fractal_preserved"] is None
    assert metrics["expansion_healthy"] is None
    assert metrics["assessment_status"] == "unavailable"
    if vf_before == 0:
        assert metrics["expansion_factor"] is None
        assert metrics["vf_growth_rate"] is None


def test_known_policy_failure_remains_false_when_another_input_is_missing():
    graph = _graph(acceleration=None)
    graph.nodes[0]["delta_nfr"] = -0.1
    metrics = expansion_metrics(graph, 0, 1, 1)
    assert metrics["assessment_status"] == "fail"
    assert metrics["expansion_healthy"] is False
    assert metrics["failed_policies"] == ("positive_stored_pressure",)
    assert metrics["unavailable_policies"] == ("acceleration_below_threshold",)


@pytest.mark.parametrize("invalid", (None, "invalid", True, float("nan"), float("inf")))
def test_invalid_authoritative_acceleration_cannot_fall_through_to_an_alias(invalid):
    graph = _graph()
    graph.nodes[0][ALIAS_D2EPI[0]] = invalid
    graph.nodes[0][ALIAS_D2EPI[1]] = 0.0
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        expansion_metrics(graph, 0, 1, 1)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "key,value",
    (
        ("VAL_BIFURCATION_THRESHOLD", float("nan")),
        ("VAL_BIFURCATION_THRESHOLD", 0),
        ("VAL_BIFURCATION_THRESHOLD", -1),
        ("VAL_BIFURCATION_THRESHOLD", True),
        ("VAL_MIN_COHERENCE", float("inf")),
        ("VAL_MIN_COHERENCE", 1.1),
        ("VAL_MIN_COHERENCE", -0.1),
        ("VAL_FRACTAL_RATIO_MIN", float("nan")),
        ("VAL_FRACTAL_RATIO_MIN", 2),
        ("VAL_FRACTAL_RATIO_MAX", 0.5),
        ("VAL_FRACTAL_RATIO_MAX", "2"),
    ),
)
def test_invalid_thresholds_cannot_certify_health(key, value):
    graph = _graph(acceleration=100)
    graph.graph[key] = value
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        expansion_metrics(graph, 0, 1, 1)
    assert _state(graph) == before


def test_exact_nonzero_baselines_and_finite_output_admission():
    tiny = 2.0**-500
    metrics = expansion_metrics(_graph(epi=-2 * tiny), 0, 1, -tiny)
    assert metrics["epi_growth_rate"] == 1
    assert metrics["growth_ratio"] == 0.125
    with pytest.raises((TypeError, ValueError)):
        expansion_metrics(_graph(), 0, 1, Fraction(1, 2**1100))
    with pytest.raises((TypeError, ValueError)):
        expansion_metrics(_graph(epi=1.0e308), 0, 1, -1.0e308)
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = True
    with pytest.raises((TypeError, ValueError)):
        expansion_metrics(graph, 0, 1, 1)


def test_boundary_policy_comparisons_are_explicit():
    graph = _graph(acceleration=0.3)
    graph.graph["VAL_FRACTAL_RATIO_MIN"] = 1
    metrics = expansion_metrics(graph, 0, 1, 1)
    assert metrics["bifurcation_risk"] is False  # Strict acceleration threshold.
    assert metrics["bifurcation_magnitude"] == 1
    assert metrics["fractal_preserved"] is False  # Strict ratio interval.
    graph.graph["VAL_FRACTAL_RATIO_MIN"] = 0.5
    graph.nodes[0][ALIAS_D2EPI[0]] = -100
    metrics = expansion_metrics(graph, 0, 1, 1)
    assert metrics["bifurcation_risk"] is True
    assert metrics["expansion_healthy"] is False
