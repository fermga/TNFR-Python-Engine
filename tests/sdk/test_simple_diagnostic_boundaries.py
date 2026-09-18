"""SDK observations preserve graph state and the canonical scalar/phase domains."""

from copy import deepcopy
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_SI
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.metrics.sense_index import compute_Si
from tnfr.sdk import simple
from tnfr.sdk.simple import Network, Results, TetradSnapshot
from tnfr.types import ensure_bepi, serialize_bepi, serialize_bepi_json


def _network(order=(0, 1)):
    graph = nx.Graph()
    for node in order:
        graph.add_node(
            node, EPI=-1.0 - 2.0 * node, nu_f=1.0 + node,
            delta_nfr=0.1 + 0.7 * node, phase=0.2 * node,
        )
        graph.nodes[node][ALIAS_SI[0]] = 99.0 + node
    graph.add_edge(0, 1, weight=1.0)
    graph.graph["N_JOBS"] = 1
    return Network(graph)


def test_sense_index_means_all_nodes_without_writing_live_state():
    network = _network()
    before_nodes = deepcopy(dict(network.G.nodes(data=True)))
    before_graph = deepcopy(network.G.graph)
    expected = compute_Si(deepcopy(network.G), inplace=False)
    assert expected[0] != expected[1]

    value = network.sense_index()

    assert value == pytest.approx(sum(expected.values()) / 2)
    assert _network((1, 0)).sense_index() == pytest.approx(value)
    assert dict(network.G.nodes(data=True)) == before_nodes
    assert network.G.graph == before_graph


@pytest.mark.parametrize("array_result", [False, True])
def test_sense_index_mapping_array_parity_and_detached_cache_writes(
    monkeypatch, array_result,
):
    network = _network()

    def metric(graph, *, inplace):
        assert graph is not network.G
        assert inplace is False
        graph.graph["private_metric_cache"] = [1]
        graph.nodes[0][ALIAS_SI[0]] = -50.0
        return np.array([0.2, 0.8]) if array_result else {0: 0.2, 1: 0.8}

    monkeypatch.setattr(simple, "compute_Si", metric)
    assert network.sense_index() == pytest.approx(0.5)
    assert "private_metric_cache" not in network.G.graph
    assert network.G.nodes[0][ALIAS_SI[0]] == 99.0


def test_empty_sense_index_is_zero():
    assert Network(nx.Graph()).sense_index() == 0.0


@pytest.mark.parametrize("phases", [(0.1, 2 * math.pi - 0.1), (0.0, 2 * math.pi)])
def test_mean_phase_uses_the_circle_at_the_wrap_seam(phases):
    network = _network()
    network.G.nodes[0]["phase"], network.G.nodes[1]["phase"] = phases
    value = network.avg_phase()
    assert 0.0 <= value < 2 * math.pi
    assert math.atan2(math.sin(value), math.cos(value)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("rotation", [0.0, 0.1, 1.0])
def test_antipodal_phase_has_no_direction(rotation):
    network = _network()
    network.G.nodes[0]["phase"] = rotation
    network.G.nodes[1]["phase"] = rotation + math.pi
    assert network.avg_phase() is None


def test_empty_phase_and_results_serialize_as_unavailable():
    assert Network(nx.Graph()).avg_phase() is None
    report = Results(0.0, 0.0, 0, 0, 0.0, None)
    assert report.to_dict()["avg_phase"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_phase_is_an_error_not_degeneracy(value):
    network = _network()
    network.G.nodes[0]["phase"] = value
    with pytest.raises(TNFRValueError, match="finite"):
        network.avg_phase()


@pytest.mark.parametrize("representation", [ensure_bepi, serialize_bepi, serialize_bepi_json])
def test_nfr_preserves_signed_scalar_epi_across_live_and_serialized_forms(representation):
    network = _network()
    for node in network.G:
        network.G.nodes[node]["EPI"] = representation(-1.0 - 2.0 * node)
    report = network.nfr()
    assert report["triad_available"] is True
    assert report["triad"]["epi_mean"] == -2.0
    assert network.nodal_state(0).epi == -1.0


@pytest.mark.parametrize("serialized", [False, True])
def test_nonuniform_epi_cannot_be_reported_as_signed_scalar_magnitude(serialized):
    network = _network()
    value = BEPIElement((-0.8, -0.7), (-0.8, -0.8), (0.0, 1.0))
    network.G.nodes[0]["EPI"] = serialize_bepi(value) if serialized else value
    with pytest.raises(TNFRValueError, match="uniform-real"):
        network.nfr()
    with pytest.raises(TNFRValueError, match="uniform-real"):
        network.nodal_state(0)


@pytest.mark.parametrize("xi", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_unavailable_or_invalid_correlation_length_is_not_safe(xi):
    result = TetradSnapshot(
        phi_s={0: 0.0}, grad_phi={0: 0.0}, k_phi={0: 0.0}, xi_c=xi,
    ).is_safe()
    assert result["xi_c_safe"] is False
    assert result["overall"] is False


@pytest.mark.parametrize("xi", [0.0, 1.0])
def test_finite_nonnegative_correlation_length_passes_its_advisory_check(xi):
    result = TetradSnapshot(xi_c=xi).is_safe()
    assert result["xi_c_safe"] is True
    assert result["overall"] is True


def test_nonfinite_local_field_remains_unsafe_with_valid_correlation():
    result = TetradSnapshot(grad_phi={0: -float("inf")}, xi_c=1.0).is_safe()
    assert result["grad_phi_safe"] is False
    assert result["overall"] is False
