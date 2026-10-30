"""Public report consumers admit physical scalars before calculating outputs."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_VF
from tnfr.sdk import Network, diagnose_network
from tnfr.sdk.fluent import NetworkResults
from tnfr.sdk.utils import compare_networks, compute_network_statistics


def _result(**changes):
    fields = dict(
        coherence=0.5,
        sense_indices={0: 0.25, 1: 0.75},
        delta_nfr={0: -2.0, 1: 4.0},
        graph=nx.path_graph(2),
        avg_vf=1.0,
        avg_phase=0.0,
    )
    fields.update(changes)
    return NetworkResults(**fields)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        math.nan,
        math.inf,
        "0.5",
        0.5j,
        Fraction(1, 10**400),
        Fraction(10**100 + 1, 10**100),
        -0.1,
    ],
)
@pytest.mark.parametrize("consumer", ["comparison", "statistics"])
def test_public_result_consumers_reject_invalid_coherence(value, consumer):
    result = _result(coherence=value)
    with pytest.raises((TypeError, ValueError)):
        if consumer == "comparison":
            compare_networks({"sample": result}, metrics=["coherence"])
        else:
            compute_network_statistics(result)
    assert result.coherence is value


@pytest.mark.parametrize("field", ["avg_vf", "avg_phase"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(False), math.nan, math.inf, "1", Fraction(1, 10**400)]
)
def test_optional_present_averages_must_be_valid(field, value):
    result = _result(**{field: value})
    with pytest.raises((TypeError, ValueError)):
        compute_network_statistics(result)
    assert getattr(result, field) is value


@pytest.mark.parametrize("value", [-1.0, Fraction(-1, 10**400)])
def test_negative_average_capacity_is_not_an_available_observation(value):
    with pytest.raises(ValueError):
        compute_network_statistics(_result(avg_vf=value))


def test_statistics_use_admitted_values_and_preserve_signed_pressure():
    result = _result(
        coherence=Fraction(1, 2),
        sense_indices={0: Fraction(1, 4), 1: Fraction(3, 4)},
        delta_nfr={0: Fraction(-2), 1: Fraction(4)},
        avg_vf=Fraction(0),
        avg_phase=Fraction(1, 3),
    )
    prior = deepcopy((result.sense_indices, result.delta_nfr))

    stats = compute_network_statistics(result)
    compared = compare_networks({"sample": result})["sample"]

    assert stats == {
        "coherence": 0.5,
        "node_count": 2,
        "avg_si": 0.5,
        "min_si": 0.25,
        "max_si": 0.75,
        "std_si": 0.25,
        "avg_delta_nfr": 1.0,
        "min_delta_nfr": -2.0,
        "max_delta_nfr": 4.0,
        "std_delta_nfr": 3.0,
        "avg_vf": 0.0,
        "avg_phase": 1.0 / 3.0,
    }
    assert compared == {
        "coherence": 0.5,
        "node_count": 2,
        "avg_si": 0.5,
        "avg_delta_nfr": 1.0,
    }
    assert type(stats["min_delta_nfr"]) is float
    assert type(stats["max_si"]) is float
    assert (result.sense_indices, result.delta_nfr) == prior


def test_optional_absence_and_unselected_values_are_not_invented():
    stats = compute_network_statistics(_result(avg_vf=None, avg_phase=None))
    assert "avg_vf" not in stats and "avg_phase" not in stats
    assert compare_networks(
        {"sample": _result(coherence=math.nan)}, metrics=["node_count"]
    ) == {"sample": {"node_count": 2}}


@pytest.mark.parametrize(
    "capacity", [-1.0, True, np.bool_(True), math.nan, Fraction(-1, 10**400)]
)
def test_diagnostic_projection_rejects_authoritative_invalid_capacity(capacity):
    graph = nx.Graph()
    graph.add_node(0, EPI=-1.0, theta=0.0, delta_nfr=2.0)
    graph.nodes[0][ALIAS_VF[0]] = capacity
    graph.nodes[0][ALIAS_VF[-1]] = 1.0
    before = deepcopy(dict(graph.nodes(data=True)))

    report = diagnose_network(Network(graph))

    projection = report["state"]["nodes"][0]
    assert projection["nu_f"]["available"] is False
    assert projection["nu_f"]["value"] is None
    assert projection["nu_f"]["error"]["type"] in {"TypeError", "ValueError"}
    assert projection["epi"]["value"] == -1.0
    assert projection["delta_nfr"]["value"] == 2.0
    assert report["nodal"][0]["available"] is False
    assert graph.nodes[0].keys() == before[0].keys()
    assert graph.nodes[0][ALIAS_VF[0]] is capacity
    assert graph.nodes[0][ALIAS_VF[-1]] == 1.0


def test_zero_capacity_with_nonzero_pressure_is_valid_and_freezes_only_form_rate():
    graph = nx.Graph()
    graph.add_node(0, EPI=-1.0, nu_f=0.0, theta=0.25, delta_nfr=2.0)
    before = deepcopy(dict(graph.nodes(data=True)))

    report = diagnose_network(Network(graph))

    assert report["state"]["nodes"][0]["nu_f"] == {
        "available": True,
        "value": 0.0,
        "error": None,
    }
    assert report["nodal"][0]["available"] is True
    assert report["nodal"][0]["value"]["expected_depi_dt"] == 0.0
    assert report["nodal"][0]["value"]["delta_nfr"] == 2.0
    assert dict(graph.nodes(data=True)) == before
