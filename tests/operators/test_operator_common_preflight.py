"""Common public-operator controls reject before structural writes."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.operators.definitions import Coherence, Emission
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.graph["RANDOM_SEED"] = 17
    graph.add_node(
        0,
        **{
            EPI_PRIMARY: 0.4,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.2,
            THETA_PRIMARY: 0.0,
            "glyph_history": [],
        },
    )
    return graph


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("validate_preconditions", 1),
        ("collect_metrics", "yes"),
        ("validate_nodal_equation", None),
    ],
)
def test_invalid_common_keyword_flags_reject_before_emission(keyword, value):
    graph = _graph()
    before_node = deepcopy(dict(graph.nodes[0]))
    before_graph = deepcopy(dict(graph.graph))

    with pytest.raises(OperatorPreconditionError, match="must be a boolean"):
        Emission()(graph, 0, **{keyword: value})

    assert dict(graph.nodes[0]) == before_node
    assert dict(graph.graph) == before_graph


@pytest.mark.parametrize("dt", [0.0, -1.0, True, float("nan"), float("inf")])
def test_invalid_active_nodal_time_step_rejects_before_emission(dt):
    graph = _graph()
    before_node = deepcopy(dict(graph.nodes[0]))

    with pytest.raises(OperatorPreconditionError, match="dt"):
        Emission()(graph, 0, validate_nodal_equation=True, dt=dt)

    assert dict(graph.nodes[0]) == before_node


def test_invalid_metrics_sink_rejects_before_emission():
    graph = _graph()
    graph.graph["operator_metrics"] = {"not": "append-only"}
    before_node = deepcopy(dict(graph.nodes[0]))
    before_graph = deepcopy(dict(graph.graph))

    with pytest.raises(OperatorPreconditionError, match="operator_metrics"):
        Emission()(graph, 0, collect_metrics=True)

    assert dict(graph.nodes[0]) == before_node
    assert dict(graph.graph) == before_graph


def test_invalid_integrity_monitor_rejects_before_emission():
    graph = _graph()
    graph.graph["integrity_monitor"] = object()
    before_node = deepcopy(dict(graph.nodes[0]))

    with pytest.raises(OperatorPreconditionError, match="integrity_monitor"):
        Emission()(graph, 0)

    assert dict(graph.nodes[0]) == before_node


@pytest.mark.parametrize(
    "field",
    [VF_PRIMARY, DNFR_PRIMARY, THETA_PRIMARY],
)
def test_consumed_nonzero_node_scalars_cannot_underflow_through_public_preflight(field):
    graph = _graph()
    graph.nodes[0][field] = -Fraction(1, 10**400)
    before_node = deepcopy(dict(graph.nodes[0]))
    before_graph = deepcopy(graph.graph)
    with pytest.raises(OperatorPreconditionError, match="underflows"):
        Coherence()(graph, 0)
    assert dict(graph.nodes[0]) == before_node
    assert graph.graph == before_graph


def test_nonzero_public_phase_coefficient_cannot_be_silently_disabled():
    graph = _graph()
    before_node = deepcopy(dict(graph.nodes[0]))
    before_graph = deepcopy(graph.graph)
    with pytest.raises(OperatorPreconditionError, match="underflows"):
        Coherence()(graph, 0, phase_locking_coefficient=Fraction(1, 10**400))
    assert dict(graph.nodes[0]) == before_node
    assert graph.graph == before_graph
