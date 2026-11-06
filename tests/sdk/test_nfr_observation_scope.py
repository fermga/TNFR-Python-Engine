"""NFR observations preserve diagnostic scope and independent availability."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DEPI
from tnfr.physics.fields import classify_nodal_topology
from tnfr.sdk.simple import Network


def _pair(*, capacity=1.0, pressure=0.0, epi=0.0):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=epi, nu_f=capacity, delta_nfr=pressure, phase=0.0)
    return Network(graph)


def _exponential_star():
    """Every C_i*C_j equals exp(-distance/2), independently of the fitter."""
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, phase=0.0, delta_nfr=0.0)
    for node in range(1, 13):
        distance = 1 + (node - 1) % 4
        graph.add_node(
            node,
            EPI=0.0,
            nu_f=1.0,
            phase=0.0,
            delta_nfr=math.expm1(distance / 2.0),
        )
        graph.add_edge(0, node, weight=1.0, length=float(distance))
    return Network(graph)


@pytest.mark.parametrize("size", [0, 1, 3])
def test_absent_geometric_information_does_not_invent_an_annular_nfr(size):
    graph = nx.empty_graph(size)
    topology = classify_nodal_topology(graph)
    report = Network(graph).nfr()

    assert topology["topology"] == report["topology"] == "unavailable"
    assert topology["available"] is report["topology_available"] is False
    assert topology["centers"] == []
    assert topology["status"] == (
        "empty_graph" if size == 0 else "no_positive_metric_pairs"
    )
    assert report["coherence_length_available"] is False
    assert report["coherence_length_provenance"]["method"] == "unavailable"


def test_zero_metric_pairs_do_not_supply_geometric_identity():
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 0.0, "length")
    report = classify_nodal_topology(graph)
    assert report["centrality"] == {0: 0.0, 1: 0.0}
    assert report["available"] is False


def test_dimensionless_profile_statistics_do_not_overflow_finite_centrality():
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 1e-154, "length")
    report = classify_nodal_topology(graph)

    assert report["available"] is True
    assert report["centrality"] == pytest.approx({0: 1e308, 1: 1e308})
    assert report["concentration"] == 1.0
    assert report["dispersion"] == 0.0


def test_nonfinite_materialized_centrality_cannot_receive_a_geometry_label():
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 1e-155, "length")
    with pytest.raises(ValueError, match="centrality must be finite"):
        classify_nodal_topology(graph)


def test_annular_is_a_centrality_policy_not_a_ring_topology_test():
    ring = classify_nodal_topology(nx.cycle_graph(6))
    complete = classify_nodal_topology(nx.complete_graph(6))
    assert ring["topology"] == complete["topology"] == "annular"
    assert ring["status"] == complete["status"] == "centrality_profile_policy"
    assert ring["centrality"] != complete["centrality"]


def test_nfr_uses_state_dependent_fit_and_retains_units_and_provenance():
    network = _exponential_star()
    before = deepcopy(network.G.graph), deepcopy(dict(network.G.nodes(data=True)))
    report = network.nfr()

    assert report["coherence_length"] == pytest.approx(2.0, rel=2e-14)
    assert report["coherence_length_available"] is True
    assert report["coherence_length_provenance"]["method"] == "autocorrelation_fit"
    assert (
        "path-length units"
        in report["coherence_length_provenance"]["distance_weighting"]
    )
    assert (network.G.graph, dict(network.G.nodes(data=True))) == before

    for _, _, data in network.G.edges(data=True):
        data["length"] *= 3.0
    scaled = network.nfr()
    assert scaled["coherence_length"] == pytest.approx(6.0, rel=2e-14)


def test_spectral_fallback_is_reported_as_a_dimensionless_comparison():
    report = _pair().nfr()
    assert report["coherence_length"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert report["coherence_length_provenance"]["method"] == "spectral_gap"
    assert (
        "dimensionless" in report["coherence_length_provenance"]["distance_weighting"]
    )


def test_pressure_observation_survives_missing_rate_and_capacity():
    network = _pair(pressure=2.0)
    for node in network.G:
        del network.G.nodes[node]["nu_f"]
    report = network.nfr()
    assert report["pressure_telemetry_available"] is True
    assert report["mean_abs_dnfr"] == 2.0
    assert report["zero_pressure_fraction"] == 0.0
    assert report["dynamic_telemetry_available"] is False
    assert report["coherence"] is None
    assert report["equilibrium_fraction"] is None


@pytest.mark.parametrize("capacity", [-1.0, True, np.bool_(True), float("inf")])
def test_invalid_capacity_cannot_establish_unforced_dynamic_equilibrium(capacity):
    report = _pair(capacity=capacity).nfr()
    assert report["triad_available"] is False
    assert report["dynamic_telemetry_available"] is False
    assert report["equilibrium_fraction"] is None
    assert report["zero_pressure_fraction"] == 1.0


def test_unrepresentable_nodal_product_does_not_discard_pressure_observation():
    report = _pair(capacity=1e308, pressure=2.0).nfr()
    assert report["triad_available"] is True
    assert report["dynamic_telemetry_available"] is False
    assert report["depi_dt_source"] is None
    assert report["mean_abs_dnfr"] == 2.0
    assert report["triad"]["vf_mean"] == 1e308


def test_underflowed_nodal_prediction_is_unavailable_without_discarding_triad():
    report = _pair(capacity=1e-200, pressure=1e-200).nfr()
    assert report["triad_available"] is True
    assert report["pressure_telemetry_available"] is True
    assert report["dynamic_telemetry_available"] is False
    assert report["depi_dt_status"] == "nodal_product_underflow"
    assert report["mean_abs_depi_dt"] is report["equilibrium_fraction"] is None


@pytest.mark.parametrize("capacity,pressure", [(0.0, 2.0), (2.0, 0.0)])
def test_exact_zero_factor_keeps_unforced_rate_available(capacity, pressure):
    report = _pair(capacity=capacity, pressure=pressure).nfr()
    assert report["dynamic_telemetry_available"] is True
    assert report["depi_dt_status"] == "available"
    assert report["mean_abs_depi_dt"] == 0.0


@pytest.mark.parametrize("channel", ["nu_f", "phase", ALIAS_DEPI[0]])
def test_unrepresentable_stored_channel_cannot_be_reported_as_zero(channel):
    network = _pair()
    nx.set_node_attributes(network.G, Fraction(1, 10**400), channel)
    report = network.nfr()
    if channel == "phase":
        assert report["triad"]["phase_sync"] is None
    elif channel == "nu_f":
        assert report["triad"]["vf_mean"] is None
        assert report["dynamic_telemetry_available"] is False
    else:
        assert report["dynamic_telemetry_available"] is False
        assert report["depi_dt_status"] == "recorded_rate_unavailable"


def test_finite_signed_form_mean_does_not_overflow_its_unscaled_sum():
    report = _pair(epi=1e308).nfr()
    assert report["triad"]["epi_mean"] == 1e308


@pytest.mark.parametrize(
    "recorded", [None, 0.0, True, np.bool_(True), float("nan"), float("inf"), "invalid"]
)
def test_partial_or_invalid_recorded_rates_are_not_replaced_by_an_equilibrium_prediction(
    recorded,
):
    network = _pair()
    network.G.nodes[0][ALIAS_DEPI[0]] = recorded
    report = network.nfr()
    assert report["pressure_telemetry_available"] is True
    assert report["zero_pressure_fraction"] == 1.0
    assert report["dynamic_telemetry_available"] is False
    assert report["depi_dt_source"] is None
    assert report["equilibrium_fraction"] is None


@pytest.mark.parametrize("channel", ["delta_nfr", "nu_f", "phase", ALIAS_DEPI[0]])
def test_textual_stored_channels_are_not_reconstructed_as_real_observations(channel):
    network = _pair()
    nx.set_node_attributes(network.G, "1.0", channel)
    report = network.nfr()
    if channel == "phase":
        assert report["triad"]["phase_sync"] is None
    elif channel == "nu_f":
        assert report["triad"]["vf_mean"] is None
    else:
        assert report["dynamic_telemetry_available"] is False


def test_complete_recorded_rate_remains_distinct_from_zero_unforced_prediction():
    network = _pair()
    nx.set_node_attributes(network.G, 2.0, ALIAS_DEPI[0])
    report = network.nfr()
    assert report["dynamic_telemetry_available"] is True
    assert report["depi_dt_source"] == "recorded"
    assert report["zero_pressure_fraction"] == 1.0
    assert report["equilibrium_fraction"] == 0.0
    assert report["mean_abs_depi_dt"] == 2.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_uniform_subnormal_form_and_capacity_means_retain_their_nonzero_values(sign):
    tiny = float.fromhex("0x0.0000000000001p-1022")
    report = _pair(epi=sign * tiny, capacity=tiny).nfr()
    assert report["triad"]["epi_mean"] == sign * tiny
    assert report["triad"]["vf_mean"] == tiny


@pytest.mark.parametrize(
    "values",
    [
        (1e308, -1e308, 1.0),
        (-1e308, 1.0, 1e308),
        (1e308, -1e308, 3 * float.fromhex("0x0.0000000000001p-1022")),
    ],
)
def test_signed_form_mean_preserves_small_cancellation_residual(values):
    network = _pair()
    network.G.add_node(2, EPI=0.0, nu_f=1.0, delta_nfr=0.0, phase=0.0)
    for node, value in zip(network.G, values):
        network.G.nodes[node]["EPI"] = value
    expected = float(sum(map(Fraction.from_float, values), Fraction()) / len(values))
    assert expected != 0.0
    assert network.nfr()["triad"]["epi_mean"] == expected
