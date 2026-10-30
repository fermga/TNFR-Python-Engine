"""Strict Reception ceilings and authoritative stored-state admission."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import Reception
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.preconditions.reception import validate_reception_strict


def _graph():
    graph = nx.path_graph(2)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=0.5,
            theta=0.0,
            delta_nfr=-0.25,
            glyph_history=["AL"],
        )
    return graph


@pytest.mark.parametrize("threshold", ["EPI_SATURATION_MAX", "DNFR_RECEPTION_MAX"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        float("nan"),
        float("inf"),
        -float("inf"),
        "0.9",
        Fraction(1, 10**400),
    ],
)
def test_invalid_ceiling_rejects_read_only_and_atomic_stage(threshold, value):
    graph = _graph()
    graph.graph[threshold] = value
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRValueError, match=threshold):
        validate_reception_strict(graph, 0)
    with pytest.raises(TNFRValueError, match=threshold):
        execute_neighbor_stage(graph, Reception(), (0, 1), track_sources=False)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph[threshold] is value
    assert "_node_cache" not in graph.graph
    assert "_last_operator_applied" not in graph.__dict__


@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_DNFR])
@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), "0.1", Fraction(1, 10**400)]
)
def test_invalid_authoritative_coordinate_cannot_fall_through_to_valid_alias(
    aliases, value
):
    graph = _graph()
    graph.nodes[0][aliases[0]] = value
    graph.nodes[0][aliases[1]] = 0.0
    with pytest.raises(TNFRValueError, match="EN precondition failed"):
        validate_reception_strict(graph, 0)


@pytest.mark.parametrize("represented", [False, True])
def test_negative_form_and_pressure_keep_signed_upper_bound_semantics(represented):
    graph = _graph()
    form = BEPIElement((-0.5, -0.5), (-0.5, -0.5), (0.0, 1.0)) if represented else -0.5
    graph.nodes[0].update(EPI=form, delta_nfr=-4.0)
    graph.graph.update(EPI_SATURATION_MAX=-0.25, DNFR_RECEPTION_MAX=-2.0)
    validate_reception_strict(graph, 0)
    assert graph.nodes[0]["EPI"] is form


@pytest.mark.parametrize(
    "coordinate,threshold",
    [("EPI", "EPI_SATURATION_MAX"), ("delta_nfr", "DNFR_RECEPTION_MAX")],
)
def test_finite_ceiling_equality_still_rejects(coordinate, threshold):
    graph = _graph()
    graph.graph[threshold] = -0.25
    graph.nodes[0][coordinate] = -0.25
    with pytest.raises(TNFRValueError, match="EN precondition failed"):
        validate_reception_strict(graph, 0)


def test_nonuniform_form_is_not_admitted_via_magnitude_projection():
    graph = _graph()
    graph.nodes[0]["EPI"] = BEPIElement((-0.5, -0.4), (-0.5, -0.5), (0.0, 1.0))
    with pytest.raises(TNFRValueError, match="EN precondition failed"):
        validate_reception_strict(graph, 0)
