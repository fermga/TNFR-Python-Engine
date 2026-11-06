"""Optional provenance envelopes preserve observation semantics."""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import pytest

from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
from tnfr.metrics import (
    StructuralObservation,
    observe_arithmetic_nfr,
    observe_graph_tetrad,
)
from tnfr.metrics.common import is_structural_equilibrium, structural_coherence
from tnfr.metrics.tetrad import collect_tetrad_snapshot
from tnfr.sdk.simple import Network


def test_observation_metadata_is_detached_and_serializable():
    metadata = {"source": "unit-test"}
    observation = StructuralObservation(
        domain="arithmetic",
        pressure_realization="divisor_functions",
        aggregation="mean_local",
        derivative_kind="static_zero",
        equilibrium_tolerance=1e-12,
        scope="descriptive fixed-point readout",
        value=1.0,
        metadata=metadata,
    )
    metadata["source"] = "mutated"
    result = observation.as_dict()
    assert result["metadata"] == {"source": "unit-test"}
    result["metadata"]["source"] = "changed"
    assert observation.metadata["source"] == "unit-test"


def test_default_observation_metadata_is_empty_and_read_only():
    observation = StructuralObservation(
        domain="graph",
        pressure_realization="laplacian",
        aggregation="global",
        derivative_kind="read_only_snapshot",
        equilibrium_tolerance=1e-12,
        scope="test",
    )
    assert dict(observation.metadata) == {}
    with pytest.raises(TypeError):
        observation.metadata["source"] = "mutation"


def test_nested_observation_payloads_are_detached_on_input_and_export():
    value = {"field": {"node": [0.25]}}
    metadata = {"source": {"steps": [1]}}
    observation = StructuralObservation(
        "graph", "laplacian", "global", "snapshot", None, "test", value, metadata
    )
    value["field"]["node"].append(0.5)
    metadata["source"]["steps"].append(2)
    exported = observation.as_dict()
    exported["value"]["field"]["node"].append(1.0)
    exported["metadata"]["source"]["steps"].append(3)
    assert observation.value == {"field": {"node": [0.25]}}
    assert observation.metadata["source"] == {"steps": [1]}


def test_graph_observation_retains_opaque_node_labels_in_detached_fields():
    nodes = (object(), object())
    graph = nx.Graph([(nodes[0], nodes[1])])
    for node in graph:
        graph.nodes[node].update(EPI=0.25, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    observation = Network(graph).tetrad_observation()
    exported = observation.as_dict()
    for name in ("phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"):
        retained_field = getattr(observation.value, name)
        exported_field = getattr(exported["value"], name)
        assert set(retained_field) == set(exported_field) == set(nodes)
        exported_field[nodes[0]] = 999.0
        assert retained_field[nodes[0]] != 999.0


@pytest.mark.parametrize(
    "tolerance", [True, "0.1", float("nan"), float("inf"), Fraction(1, 2**2000)]
)
def test_invalid_observation_tolerance_cannot_be_published(tolerance):
    with pytest.raises((ValueError, TypeError)):
        StructuralObservation(
            "graph", "laplacian", "global", "snapshot", tolerance, "test"
        )


def test_observation_metadata_rejects_incomplete_or_negative_provenance():
    with pytest.raises(ValueError):
        StructuralObservation(
            domain="",
            pressure_realization="graph",
            aggregation="global",
            derivative_kind="rate",
            equilibrium_tolerance=0.1,
            scope="test",
        )

    with pytest.raises(ValueError):
        StructuralObservation(
            domain="graph",
            pressure_realization="laplacian",
            aggregation="global",
            derivative_kind="rate",
            equilibrium_tolerance=-1.0,
            scope="test",
        )


def test_domain_observation_adapters_preserve_distinct_semantics():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=float(node), **{"ΔNFR": 0.0, "θ": 0.1})
    graph_observation = observe_graph_tetrad(collect_tetrad_snapshot(graph))
    arithmetic_observation = observe_arithmetic_nfr({"coherence": 1.0})
    assert graph_observation.aggregation == "per_node_field_statistics_plus_global_xi_c"
    assert arithmetic_observation.aggregation == (
        "canonical_static_coherence_of_mean_absolute_pressure"
    )
    assert arithmetic_observation.metadata["canonical_field"] == "coherence"
    assert (
        arithmetic_observation.metadata["descriptive_field"] == "mean_local_coherence"
    )


def test_empty_arithmetic_observation_is_marked_unavailable():
    observation = observe_arithmetic_nfr(
        {"coherence": None, "readout_available": False, "n_nodes": 0}
    )

    assert observation.metadata["unavailable"] is True


def test_public_domain_readouts_expose_opt_in_observation_envelopes():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=float(node), nu_f=1.0, theta=0.1, delta_nfr=0.0)
    network = Network(graph)
    assert network.tetrad_observation().domain == "graph"
    assert network.nfr_observation().domain == "graph"
    assert ArithmeticTNFRNetwork(10).nfr_observation().domain == "arithmetic"


def test_summary_and_per_node_tetrads_keep_distinct_complete_metadata():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    per_node = Network(graph).tetrad_observation()
    summarized = observe_graph_tetrad(collect_tetrad_snapshot(graph))
    assert per_node.aggregation == "per_node_fields_plus_global_xi_c"
    assert summarized.aggregation == "per_node_field_statistics_plus_global_xi_c"
    for observation in (per_node, summarized):
        assert observation.metadata["complete"] is True
        assert observation.metadata["unavailable"] is False
        assert observation.metadata["field_availability"] == {
            "phi_s": True,
            "grad_phi": True,
            "k_phi": True,
            "xi_c": True,
        }


def test_partial_tetrad_is_useful_but_does_not_claim_complete_evidence():
    import math

    graph = nx.star_graph(4)
    for node, phase in zip(graph, (0.3, 0.0, 0.0, math.pi, -math.pi)):
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, phase=phase, delta_nfr=0.0)
    observation = observe_graph_tetrad(collect_tetrad_snapshot(graph))
    assert observation.metadata["unavailable"] is False
    assert observation.metadata["complete"] is False
    assert observation.metadata["field_availability"] == {
        "phi_s": True,
        "grad_phi": True,
        "k_phi": False,
        "xi_c": True,
    }


@pytest.mark.parametrize("snapshot", [None, {}, {"phi_s": {}, "xi_c": float("nan")}])
def test_empty_tetrad_payload_has_no_available_evidence(snapshot):
    observation = observe_graph_tetrad(snapshot)
    assert observation.metadata["unavailable"] is True
    assert observation.metadata["complete"] is False
    assert not any(observation.metadata["field_availability"].values())


def test_per_node_completeness_requires_matching_nonempty_field_support():
    from tnfr.sdk.simple import TetradSnapshot

    snapshot = TetradSnapshot(
        phi_s={0: 0.0}, grad_phi={0: 0.0}, k_phi={1: 0.0}, xi_c=1.0
    )
    observation = observe_graph_tetrad(snapshot)
    assert all(observation.metadata["field_availability"].values())
    assert observation.metadata["complete"] is False


@pytest.mark.parametrize("invalid", [True, "0", float("nan"), Fraction(1, 10**400)])
def test_invalid_summary_scalar_does_not_inherit_available_flag(invalid):
    snapshot = collect_tetrad_snapshot(nx.path_graph(3))
    snapshot["phi_s"]["mean"] = invalid
    snapshot["xi_c"] = invalid
    observation = observe_graph_tetrad(snapshot)
    assert observation.metadata["field_availability"]["phi_s"] is False
    assert observation.metadata["field_availability"]["xi_c"] is False
    assert observation.metadata["complete"] is False


def test_cross_domain_fixed_point_logic_and_frozen_state_remain_distinct():
    arithmetic = observe_arithmetic_nfr({"coherence": structural_coherence(0.0)})
    assert is_structural_equilibrium(0.0)
    assert arithmetic.value["coherence"] == 1.0
    assert not is_structural_equilibrium(2.0, 0.0)
    assert structural_coherence(2.0, 0.0) < 1.0
