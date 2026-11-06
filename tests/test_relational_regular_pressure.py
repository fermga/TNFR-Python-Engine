"""Regular-domain pressure uses the captured relative Arg branch once."""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics import dnfr


def _graph():
    graph = nx.path_graph(2)
    for node, value in enumerate((2**52, 2**52 + 1)):
        graph.nodes[node].update(EPI=float(value), theta=0.0, nu_f=1.0, delta_nfr=99.0)
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0}
    return graph


def _arguments(graph, sources):
    nodes = tuple(graph)
    index = {node: position for position, node in enumerate(nodes)}
    return {
        "nodes": nodes,
        "phase": tuple(graph.nodes[node]["theta"] for node in nodes),
        "neighbors": tuple(
            tuple(index[other] for other in graph[node]) for node in nodes
        ),
        "phase_sources": sources,
    }


def _state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


@pytest.mark.parametrize("python_only", (False, True))
def test_captured_branch_and_stable_epi_share_one_pressure_mix(
    monkeypatch, python_only
):
    graph = _graph()
    if python_only:
        monkeypatch.setattr(dnfr, "np", None)
    source = math.nextafter(1.0, 0.0)
    sources = (source, -source)
    arguments = _arguments(graph, sources)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "captured relative sources must not reconstruct global angles"
        )

    monkeypatch.setattr(dnfr, "default_compute_delta_nfr", forbidden)
    profile = {}
    dnfr._compute_delta_nfr_from_relative_sources(graph, **arguments, profile=profile)
    expected = tuple(
        float(Q(1, 2) * (sign + Q(value))) for sign, value in zip((1, -1), sources)
    )
    assert tuple(graph.nodes[node]["delta_nfr"] for node in graph) == expected
    assert profile["dnfr_path"] == "relative_resultant_canonical"
    assert graph.graph["_DNFR_META"]["hook"] == "relative_resultant_canonical"
    assert graph.graph["_dnfr_hook_name"] == "relative_resultant_canonical"
    assert graph.graph["_DNFR_META"]["weights_effective"] == graph.graph["DNFR_WEIGHTS"]


@pytest.mark.parametrize(
    "defect",
    (
        "order",
        "phase",
        "support",
        "bool_index",
        "length",
        "bool",
        "nan",
        "cut",
        "weight",
        "channel",
    ),
)
def test_bad_capture_or_source_is_rejected_before_any_graph_write(defect):
    graph = _graph()
    arguments = _arguments(graph, (0.25, -0.25))
    if defect == "order":
        arguments["nodes"] = (1, 0)
    elif defect == "phase":
        arguments["phase"] = (0.0, 0.125)
    elif defect == "support":
        arguments["neighbors"] = ((1,), ())
    elif defect == "bool_index":
        arguments["neighbors"] = ((True,), (0,))
    elif defect == "length":
        arguments["phase_sources"] = (0.25,)
    elif defect == "weight":
        graph.edges[0, 1]["weight"] = 0.5
    elif defect == "channel":
        graph.graph["DNFR_WEIGHTS"]["vf"] = 1.0
    else:
        arguments["phase_sources"] = (
            {"bool": True, "nan": float("nan"), "cut": 1.0}[defect],
            -0.25,
        )
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        dnfr._compute_delta_nfr_from_relative_sources(graph, **arguments)
    assert _state(graph) == before


@pytest.mark.parametrize("channel", ("phase", "epi"))
def test_nonzero_pressure_cannot_disappear_during_materialization(channel):
    graph = _graph()
    for node in graph:
        graph.nodes[node]["EPI"] = 0.0
    tiny = math.ulp(0.0)
    sources = (tiny, 0.0) if channel == "phase" else (0.0, 0.0)
    if channel == "epi":
        graph.nodes[1]["EPI"] = tiny
    before = _state(graph)
    with pytest.raises(ValueError, match="underflows"):
        dnfr._compute_delta_nfr_from_relative_sources(
            graph, **_arguments(graph, sources)
        )
    assert _state(graph) == before
