"""SDK bridge for pure Mutation prediction and two-sample evidence."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators.nodal_equation import compute_d2epi_dt2
from tnfr.sdk.simple import Network, NodalStateReport


def _network(
    *,
    epi: object = 0.2,
    nu_f: object = 1.0,
    delta_nfr: object = 0.3,
    **history: object,
) -> Network:
    graph = nx.Graph()
    graph.add_node(
        "n",
        **{
            ALIAS_EPI[0]: epi,
            ALIAS_VF[0]: nu_f,
            ALIAS_DNFR[0]: delta_nfr,
            ALIAS_THETA[0]: 0.25,
            **history,
        },
    )
    return Network(graph)


def test_nodal_state_keeps_prediction_and_observation_distinct_and_is_read_only():
    network = _network(
        epi=0.4,
        _epi_history=[0.1, 0.2, 0.4],
        epi_time_history=[(0.0, 0.5), (1.0, 0.4)],
    )
    before_nodes = deepcopy(dict(network.G.nodes(data=True)))
    before_graph = deepcopy(dict(network.G.graph))

    report = network.nodal_state("n", bifurcation_threshold=0.1)

    assert report.expected_depi_dt == pytest.approx(0.3)
    assert report.predicted_crossed is True
    assert report.near_bifurcation is report.predicted_crossed
    assert report.observed_depi_dt == pytest.approx(-0.1)
    assert report.observed_crossed is False
    assert report.mutation_threshold_satisfied is False
    # The timestamped history is authoritative but has only two samples, so it
    # can certify velocity while acceleration remains unavailable (legacy API
    # represents unavailable acceleration as 0.0).
    assert report.d2epi_dt2 == pytest.approx(0.0)
    assert report.evidence_available is True
    assert report.evidence_valid is True
    assert report.source == "epi_time_history"
    assert report.time_basis == "physical_time"
    assert report.physical_time_resolved is True
    assert report.current_endpoint_matches_state is True
    assert report.rate_gap == pytest.approx(-0.4)
    assert dict(network.G.nodes(data=True)) == before_nodes
    assert dict(network.G.graph) == before_graph


def test_observed_crossing_can_hold_when_the_prediction_does_not():
    report = _network(
        epi=0.2,
        delta_nfr=0.01,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    ).nodal_state("n", bifurcation_threshold=0.1)

    assert report.predicted_crossed is False
    assert report.near_bifurcation is False
    assert report.observed_crossed is True
    assert report.observed_depi_dt == pytest.approx(0.2)
    assert report.mutation_threshold_satisfied is True
    assert report.active is True


def test_missing_history_is_not_reported_as_a_negative_observation():
    report = _network().nodal_state("n", bifurcation_threshold=0.1)

    assert report.predicted_crossed is True
    assert report.observed_depi_dt is None
    assert report.observed_crossed is None
    assert report.evidence_available is False
    assert report.evidence_valid is False
    assert report.source is None
    assert report.time_basis is None
    assert report.physical_time_resolved is False
    assert report.reason == "missing_history"
    assert report.rate_gap is None
    assert report.mutation_threshold_satisfied is False


def test_legacy_history_retains_unit_step_evidence_scope():
    report = _network(
        epi=999.0,
        epi_history=[0.0, 0.2],
    ).nodal_state("n", bifurcation_threshold=0.1)

    assert report.observed_depi_dt == pytest.approx(0.2)
    assert report.observed_crossed is True
    assert report.evidence_valid is True
    assert report.source == "epi_history"
    assert report.time_basis == "legacy_unit_operator_step"
    assert report.physical_time_resolved is False
    assert report.current_endpoint_matches_state is None
    assert report.rate_gap is None


def test_stale_physical_endpoint_stays_tri_state_invalid():
    report = _network(
        epi=0.3,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    ).nodal_state("n", bifurcation_threshold=0.1)

    assert report.observed_depi_dt == pytest.approx(0.2)
    assert report.observed_crossed is None
    assert report.evidence_available is True
    assert report.evidence_valid is False
    assert report.current_endpoint_matches_state is False
    assert report.reason == "stale_physical_endpoint"
    assert report.mutation_threshold_satisfied is False


@pytest.mark.parametrize(
    ("channel", "value"),
    [
        (ALIAS_EPI[0], True),
        (ALIAS_EPI[0], "0.2"),
        (ALIAS_VF[0], -0.1),
        (ALIAS_VF[0], float("inf")),
        (ALIAS_DNFR[0], float("nan")),
        (ALIAS_DNFR[0], "0.3"),
    ],
)
def test_invalid_nodal_channels_are_not_coerced_to_zero(channel, value):
    network = _network()
    network.G.nodes["n"][channel] = value
    before = deepcopy(dict(network.G.nodes["n"]))

    with pytest.raises(TNFRValueError, match="Invalid nodal Mutation-trigger input"):
        network.nodal_state("n")

    assert dict(network.G.nodes["n"]) == before


def test_missing_nodal_channel_is_an_input_error():
    network = _network()
    del network.G.nodes["n"][ALIAS_DNFR[0]]

    with pytest.raises(TNFRValueError, match="Invalid nodal Mutation-trigger input"):
        network.nodal_state("n")


@pytest.mark.parametrize("threshold", [True, -0.1, float("nan"), float("inf")])
def test_invalid_mutation_threshold_is_rejected_without_mutation(threshold):
    network = _network()
    before = deepcopy(dict(network.G.nodes["n"]))

    with pytest.raises(TNFRValueError, match="Invalid nodal Mutation-trigger input"):
        network.nodal_state("n", bifurcation_threshold=threshold)

    assert dict(network.G.nodes["n"]) == before


def test_empty_scan_still_validates_the_mutation_threshold():
    with pytest.raises(TNFRValueError, match="Invalid nodal Mutation-trigger input"):
        Network(nx.Graph()).nodal_scan(bifurcation_threshold=True)


def test_d2epi_store_switch_keeps_the_legacy_default_and_supports_pure_reads():
    network = _network(_epi_history=[0.1, 0.2, 0.4])
    node_data = network.G.nodes["n"]
    before = deepcopy(dict(node_data))

    assert compute_d2epi_dt2(network.G, "n", store=False) == pytest.approx(0.1)
    assert dict(node_data) == before

    assert compute_d2epi_dt2(network.G, "n") == pytest.approx(0.1)
    assert node_data[ALIAS_D2EPI[0]] == pytest.approx(0.1)


def test_old_report_construction_keeps_near_bifurcation_as_prediction_alias():
    report = NodalStateReport(
        node="n",
        epi=0.2,
        nu_f=1.0,
        delta_nfr=0.3,
        coherence=0.7,
        phase=0.0,
        expected_depi_dt=0.3,
        d2epi_dt2=0.0,
        degree=0,
        equilibrium=False,
        active=True,
        near_bifurcation=True,
    )

    assert report.predicted_crossed is True
    serialized = report.to_dict()
    assert serialized["near_bifurcation"] is True
    assert serialized["predicted_crossed"] is True
    assert serialized["observed_crossed"] is None
    assert serialized["mutation_threshold_satisfied"] is False


def test_explicit_prediction_is_canonical_for_the_legacy_alias():
    report = NodalStateReport(
        node="n",
        epi=0.2,
        nu_f=1.0,
        delta_nfr=0.3,
        coherence=0.7,
        phase=0.0,
        expected_depi_dt=0.3,
        d2epi_dt2=0.0,
        degree=0,
        equilibrium=False,
        active=True,
        near_bifurcation=True,
        predicted_crossed=False,
    )

    assert report.predicted_crossed is False
    assert report.near_bifurcation is False
    assert not hasattr(report, "execution_ready")


def test_creative_mutation_without_evidence_is_rejected_before_any_effect():
    network = _network()
    before_nodes = deepcopy(dict(network.G.nodes(data=True)))
    before_graph = deepcopy(dict(network.G.graph))
    before_edges = list(network.G.edges(data=True))

    with pytest.raises(TNFRValueError, match="Mutation sequence preflight failed"):
        network.evolve(steps=1, sequence="creative_mutation")

    assert dict(network.G.nodes(data=True)) == before_nodes
    assert dict(network.G.graph) == before_graph
    assert list(network.G.edges(data=True)) == before_edges
    assert not hasattr(network.G, "_last_operator_applied")


def test_creative_mutation_preflight_is_atomic_across_all_nodes():
    network = _network(_epi_history=[0.0, 0.2])
    network.G.add_node(
        "missing",
        **{
            ALIAS_EPI[0]: 0.2,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.3,
            ALIAS_THETA[0]: 0.0,
        },
    )
    before = deepcopy(dict(network.G.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="Mutation sequence preflight failed"):
        network.evolve(steps=1, sequence="creative_mutation")

    assert dict(network.G.nodes(data=True)) == before


def test_physical_evidence_rejects_a_prior_epi_channel_operator_atomically():
    network = _network(
        epi=0.2,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    )
    before = deepcopy(dict(network.G.nodes(data=True)))

    with pytest.raises(
        TNFRValueError, match="physical_mutation_evidence_would_be_stale"
    ):
        network.evolve(steps=1, sequence="creative_mutation")

    assert dict(network.G.nodes(data=True)) == before
    assert not hasattr(network.G, "_last_operator_applied")
