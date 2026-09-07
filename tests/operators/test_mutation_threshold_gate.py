"""ZHIR executes only after a finite, signed, strict growth-threshold sample."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.node import NodeNX
from tnfr.operators import apply_glyph, apply_glyph_obj
from tnfr.operators.definitions import Mutation
from tnfr.operators._mutation_gate import (
    mutation_threshold_sample,
    validate_mutation_capacity,
    validate_mutation_runtime_gate,
    validate_mutation_threshold,
)
from tnfr.operators.metrics_structural import mutation_metrics
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.mutation import diagnose_mutation_readiness


def _node(history):
    return {"epi_history": history}


def _graph(history=(0.0, 0.2), *, xi=0.1) -> nx.Graph:
    graph = nx.Graph(ZHIR_THRESHOLD_XI=xi)
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: float(history[-1]) if history else 0.0,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.3,
            ALIAS_THETA[0]: 0.4,
            "epi_kind": "wave",
            "epi_history": list(history),
            "glyph_history": ["IL", "OZ"],
        },
    )
    return graph


def test_positive_signed_sample_strictly_above_xi_is_admissible():
    sample = validate_mutation_threshold(_node([0.1, 0.3]), {})

    assert sample.depi_dt == pytest.approx(0.2)
    assert sample.xi == 0.1
    assert sample.crossed


@pytest.mark.parametrize(
    ("history", "xi"),
    [
        ([0.1, 0.2], 0.1),  # equality is not a crossing
        ([0.5, 0.0], 0.1),  # a large contraction has the wrong sign
        ([0.1, 0.19], 0.1),
    ],
)
def test_equality_contraction_and_subthreshold_growth_are_rejected(history, xi):
    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        validate_mutation_threshold(_node(history), {"ZHIR_THRESHOLD_XI": xi})


@pytest.mark.parametrize("history", [[], [0.1]])
def test_unverifiable_history_is_rejected(history):
    with pytest.raises(OperatorPreconditionError, match="at least two"):
        validate_mutation_threshold(_node(history), {})


@pytest.mark.parametrize("xi", [True, "0.1", -0.1, float("nan"), float("inf")])
def test_threshold_domain_is_finite_real_and_nonnegative(xi):
    with pytest.raises(OperatorPreconditionError, match="ZHIR_THRESHOLD_XI"):
        validate_mutation_threshold(_node([0.0, 0.2]), {"ZHIR_THRESHOLD_XI": xi})


@pytest.mark.parametrize("nu_f", [0.0, -0.1, True, "1.0", float("nan"), float("inf")])
def test_runtime_gate_requires_finite_active_structural_capacity(nu_f):
    node_data = {"epi_history": [0.0, 0.2], ALIAS_VF[0]: nu_f}

    with pytest.raises(OperatorPreconditionError, match="structural frequency"):
        validate_mutation_runtime_gate(node_data, {})


def test_explicit_capacity_floor_tightens_the_runtime_gate():
    node_data = {"epi_history": [0.0, 0.2], ALIAS_VF[0]: 0.4}

    with pytest.raises(OperatorPreconditionError, match="configured minimum"):
        validate_mutation_capacity(node_data, {"ZHIR_MIN_VF": 0.5})

    gate = validate_mutation_runtime_gate(
        {**node_data, ALIAS_VF[0]: 0.6}, {"ZHIR_MIN_VF": 0.5}
    )
    assert gate.nu_f == 0.6
    assert gate.minimum_nu_f == 0.5
    assert gate.threshold.crossed


@pytest.mark.parametrize(
    "history",
    [[0.0, float("nan")], [float("-inf"), 0.2], [True, 0.2], ["0.0", 0.2]],
)
def test_history_samples_must_be_finite_real_scalars(history):
    with pytest.raises(OperatorPreconditionError, match="epi_history"):
        mutation_threshold_sample(_node(history), {})


def test_legacy_history_key_is_read_without_rewriting_storage():
    node_data = {"_epi_history": [0.0, 0.2]}
    before = deepcopy(node_data)

    sample = validate_mutation_threshold(node_data, {})

    assert sample.history_key == "_epi_history"
    assert node_data == before


def test_readiness_diagnostic_uses_same_signed_gate_and_is_read_only():
    graph = _graph([0.5, 0.0])
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    report = diagnose_mutation_readiness(graph, 0)

    assert not report["checks"]["threshold_crossing"]["passed"]
    assert report["checks"]["threshold_crossing"]["depi_dt"] == -0.5
    assert graph.nodes[0] == node_before
    assert graph.graph == graph_before


@pytest.mark.parametrize("history", [[0.0, 0.1], [0.5, 0.0], [0.0]])
def test_graph_dispatch_rejects_before_cache_or_state_mutation(history):
    graph = _graph(history)
    node_before = deepcopy(graph.nodes[0])
    metadata_before = deepcopy(graph.graph)

    with pytest.raises(OperatorPreconditionError):
        apply_glyph(graph, 0, "ZHIR")

    assert graph.nodes[0] == node_before
    assert graph.graph == metadata_before
    assert "_node_cache" not in graph.graph


def test_object_dispatch_rejects_without_partial_telemetry_or_provenance():
    graph = _graph([0.5, 0.0])
    node = NodeNX.from_graph(graph, 0)
    storage_before = deepcopy(graph.nodes[0])

    with pytest.raises(OperatorPreconditionError):
        apply_glyph_obj(node, "ZHIR")

    assert graph.nodes[0] == storage_before
    assert "source_glyph" not in graph.nodes[0]
    assert not any(key.startswith("_zhir_") for key in graph.nodes[0])


def test_graph_dispatch_rejects_inactive_node_despite_prior_growth():
    graph = _graph([0.0, 0.2])
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    node_before = deepcopy(graph.nodes[0])
    metadata_before = deepcopy(graph.graph)

    with pytest.raises(OperatorPreconditionError, match="active structural frequency"):
        apply_glyph(graph, 0, "ZHIR")

    assert graph.nodes[0] == node_before
    assert graph.graph == metadata_before


def test_successful_graph_dispatch_preserves_identity_and_records_provenance():
    graph = _graph([0.0, 0.2])
    theta_before = graph.nodes[0][ALIAS_THETA[0]]

    apply_glyph(graph, 0, "ZHIR")

    assert graph.nodes[0][ALIAS_THETA[0]] != theta_before
    assert graph.nodes[0]["epi_kind"] == "wave"
    assert graph.nodes[0]["source_glyph"] == "ZHIR"
    assert graph.nodes[0]["glyph_history"][-1] == "ZHIR"
    assert not any(
        key in graph.nodes[0]
        for key in (
            "_zhir_threshold_warning",
            "_zhir_threshold_met",
            "_zhir_threshold_unknown",
        )
    )


@pytest.mark.parametrize(
    ("history", "expected_rate", "expected_met"),
    [([0.0, 0.2], 0.2, True), ([0.0, 0.1], 0.1, False), ([0.5, 0.0], -0.5, False)],
)
def test_mutation_metrics_share_the_signed_strict_sample(
    history, expected_rate, expected_met
):
    graph = _graph(history)

    metrics = mutation_metrics(
        graph,
        0,
        theta_before=graph.nodes[0][ALIAS_THETA[0]],
        epi_before=graph.nodes[0][ALIAS_EPI[0]],
    )

    assert metrics["depi_dt"] == pytest.approx(expected_rate)
    assert metrics["threshold_met"] is expected_met
    assert metrics["threshold_validated"] is expected_met
    assert metrics["threshold_warning"] is (not expected_met)
    assert metrics["threshold_unknown"] is False


@pytest.mark.parametrize("tau", [True, -0.1, "0.1", float("nan"), float("inf")])
def test_public_mutation_rejects_invalid_acceleration_threshold_before_effect(tau):
    graph = _graph([0.0, 0.2])
    node_before = deepcopy(dict(graph.nodes[0]))
    graph_before = deepcopy(dict(graph.graph))

    with pytest.raises(OperatorPreconditionError, match="tau"):
        Mutation()(graph, 0, tau=tau)

    assert dict(graph.nodes[0]) == node_before
    assert dict(graph.graph) == graph_before


def test_public_mutation_rejects_invalid_bifurcation_sink_before_effect():
    graph = _graph([0.0, 0.05, 0.3])
    graph.graph["zhir_bifurcation_events"] = {"invalid": "sink"}
    node_before = deepcopy(dict(graph.nodes[0]))
    graph_before = deepcopy(dict(graph.graph))

    with pytest.raises(OperatorPreconditionError, match="zhir_bifurcation_events"):
        Mutation()(graph, 0, tau=0.1)

    assert dict(graph.nodes[0]) == node_before
    assert dict(graph.graph) == graph_before
