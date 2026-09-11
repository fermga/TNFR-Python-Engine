"""Network REMESH validates every delayed proposal before committing any node."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.remesh import apply_network_remesh
from tnfr.types import real_scalar_epi, serialize_bepi


def _graph(*, past_second: float = 0.1) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
    )
    graph.nodes[0]["EPI"] = 0.5
    graph.nodes[1]["EPI"] = 0.2
    graph.graph["_epi_hist"] = deque(
        [{0: 0.0, 1: past_second}, {0: 0.5, 1: 0.2}], maxlen=8
    )
    return graph


@pytest.mark.parametrize("key", ["REMESH_TAU_GLOBAL", "REMESH_TAU_LOCAL"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1"])
def test_delay_domain_rejects_without_graph_or_node_mutation(key, value):
    graph = _graph()
    graph.graph[key] = value
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    metadata_before = deepcopy(graph.graph)

    with pytest.raises(TNFRValueError, match="positive integer"):
        apply_network_remesh(graph)

    assert dict(graph.nodes(data=True)) == nodes_before
    assert graph.graph == metadata_before


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"EPI_MIN": float("nan")}, "EPI_MIN"),
        ({"EPI_MAX": float("inf")}, "EPI_MAX"),
        ({"EPI_MIN": 1.0, "EPI_MAX": -1.0}, "must not exceed"),
    ],
)
def test_invalid_bounds_reject_before_any_node_commit(updates, message):
    graph = _graph()
    graph.graph.update(updates)
    nodes_before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match=message):
        apply_network_remesh(graph)

    assert dict(graph.nodes(data=True)) == nodes_before
    assert "_REMESH_ALPHA_SRC" not in graph.graph


def test_invalid_later_node_proposal_does_not_commit_earlier_node():
    graph = _graph(past_second=float("inf"))
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    metadata_before = deepcopy(graph.graph)

    with pytest.raises(TNFRValueError, match="delayed EPI"):
        apply_network_remesh(graph)

    assert dict(graph.nodes(data=True)) == nodes_before
    assert graph.graph == metadata_before


def test_insufficient_history_is_a_pure_noop():
    graph = _graph()
    graph.graph["_epi_hist"] = deque([{0: 0.5, 1: 0.2}], maxlen=8)
    before = deepcopy(graph.graph)

    apply_network_remesh(graph)

    assert graph.graph == before
    assert "_REMESH_ALPHA_SRC" not in graph.graph


def test_valid_remesh_commits_all_prevalidated_proposals():
    graph = _graph()

    apply_network_remesh(graph)

    assert graph.nodes[0]["EPI"] == pytest.approx(0.125)
    assert graph.nodes[1]["EPI"] == pytest.approx(0.125)
    assert graph.graph["_REMESH_ALPHA_SRC"] == "REMESH_ALPHA"
    assert graph.graph["_REMESH_META"]["epi_mean_after"] == pytest.approx(0.125)


def test_remesh_accepts_uniform_real_bepi_in_state_and_delayed_history():
    graph = _graph()
    graph.nodes[0]["EPI"] = serialize_bepi(-0.5)
    graph.graph["_epi_hist"][0][0] = serialize_bepi(-0.25)

    apply_network_remesh(graph)

    assert real_scalar_epi(graph.nodes[0]["EPI"]) == pytest.approx(-0.3125)


def test_remesh_rejects_rich_bepi_before_any_node_commit():
    graph = _graph()
    rich = BEPIElement((-0.5, -0.4), (-0.5, -0.5), (0.0, 1.0))
    graph.graph["_epi_hist"][0][1] = serialize_bepi(rich)
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        apply_network_remesh(graph)

    assert dict(graph.nodes(data=True)) == before
    assert "_REMESH_ALPHA_SRC" not in graph.graph
