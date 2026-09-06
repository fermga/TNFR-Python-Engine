"""Optional provenance envelopes preserve observation semantics."""

from __future__ import annotations

import pytest

import networkx as nx

from tnfr.metrics import (
    StructuralObservation,
    observe_arithmetic_nfr,
    observe_emergent_element,
    observe_graph_tetrad,
)
from tnfr.metrics.common import is_structural_equilibrium, structural_coherence
from tnfr.physics.emergent_chemistry import classify_element
from tnfr.metrics.tetrad import collect_tetrad_snapshot
from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
from tnfr.physics.emergent_chemistry import classify_element_observation
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
    chemical_observation = observe_emergent_element(classify_element(2))
    assert graph_observation.aggregation == "per_node_fields_plus_global_xi_c"
    assert arithmetic_observation.aggregation == (
        "mean_local_structural_coherence"
    )
    assert chemical_observation.pressure_realization == (
        "valence_shell_distance"
    )


def test_public_domain_readouts_expose_opt_in_observation_envelopes():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            EPI=float(node), nu_f=1.0, theta=0.1, delta_nfr=0.0
        )
    network = Network(graph)
    assert network.tetrad_observation().domain == "graph"
    assert network.nfr_observation().domain == "graph"
    assert ArithmeticTNFRNetwork(10).nfr_observation().domain == "arithmetic"
    assert classify_element_observation(2).domain == "chemical"


def test_cross_domain_fixed_point_logic_and_frozen_state_remain_distinct():
    arithmetic = observe_arithmetic_nfr(
        {"coherence": structural_coherence(0.0)}
    )
    chemical = classify_element_observation(2)
    assert is_structural_equilibrium(0.0)
    assert arithmetic.value["coherence"] == 1.0
    assert chemical.value["closed_shell"] is True
    assert not is_structural_equilibrium(2.0, 0.0)
    assert structural_coherence(2.0, 0.0) < 1.0
