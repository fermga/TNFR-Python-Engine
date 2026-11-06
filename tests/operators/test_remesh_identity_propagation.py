"""Explicit memory jumps validate the full proposal before modifying state."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.remesh import propagate_structural_identity
from tnfr.types import ensure_bepi


def _graph():
    graph = nx.path_graph(3)
    graph.graph.update(_t=7.0, _trig_version=3, _vfmax=99.0, _vfmax_node=2)
    for node, epi, capacity, theta in (
        (0, -0.75, 0.0, 0.1),
        (1, 0.25, 2.0, math.tau - 0.1),
        (2, 0.5, 4.0, math.pi),
    ):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=capacity,
            theta=theta,
            delta_nfr=0.125,
            structural_lineage=[],
        )
    return graph


def test_memory_event_uses_circular_phase_and_one_shared_origin_snapshot():
    graph = _graph()
    origin = deepcopy(graph.nodes[0])
    propagate_structural_identity(graph, 0, [1, 1, 2, 0], 0.5)

    assert graph.nodes[0] == origin
    assert [graph.nodes[n]["EPI"] for n in (1, 2)] == [-0.25, -0.125]
    assert [graph.nodes[n]["nu_f"] for n in (1, 2)] == [1.0, 2.0]
    assert min(graph.nodes[1]["theta"], math.tau - graph.nodes[1]["theta"]) < 1e-14
    assert graph.nodes[2]["theta"] == pytest.approx(math.pi / 2 + 0.05)
    assert graph.graph["_t"] == 7.0
    assert graph.graph["_trig_version"] == 5
    assert "_vfmax" not in graph.graph and "_vfmax_node" not in graph.graph
    assert graph.graph["_dnfr_prep_dirty"] is True
    for node in (1, 2):
        assert graph.nodes[node]["delta_nfr"] == 0.125
        records = graph.nodes[node]["structural_lineage"]
        assert len(records) == 1
        assert records[0]["scope"] == "auxiliary_structural_memory_event"
        assert records[0]["vf_after"] == graph.nodes[node]["nu_f"]
        assert records[0]["theta_after"] == graph.nodes[node]["theta"]


@pytest.mark.parametrize(
    "strength", [-1.0, 2.0, True, "0.5", math.inf, Fraction(1, 2**2000)]
)
def test_invalid_strength_rejects_without_changes(strength):
    graph = _graph()
    before = deepcopy(graph)
    with pytest.raises((TypeError, ValueError)):
        propagate_structural_identity(graph, 0, [1, 2], strength)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


@pytest.mark.parametrize(
    "field,value",
    [
        ("nu_f", None),
        ("nu_f", -1.0),
        ("theta", True),
        ("EPI", Fraction(1, 2**2000)),
        ("EPI", 1 + 0j),
        ("structural_lineage", ()),
    ],
)
def test_invalid_later_target_cannot_partially_commit(field, value):
    graph = _graph()
    graph.nodes[2][field] = value
    before = deepcopy(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        propagate_structural_identity(graph, 0, [1, 2])
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_missing_channels_and_invalid_clip_policy_reject_before_writes():
    for field in ("EPI", "nu_f", "theta"):
        graph = _graph()
        del graph.nodes[2][field]
        before = deepcopy(graph)
        with pytest.raises(ValueError, match="missing"):
            propagate_structural_identity(graph, 0, [1, 2])
        assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
        assert graph.graph == before.graph

    graph = _graph()
    graph.graph["CLIP_MODE"] = "unsupported"
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="mode"):
        propagate_structural_identity(graph, 0, [1, 2])
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_zero_is_noop_and_convex_capacity_handles_extreme_finite_values():
    graph = _graph()
    graph.nodes[0]["nu_f"] = graph.nodes[1]["nu_f"] = 1.7e308
    graph.nodes[1]["EPI"] = 2.0
    before = deepcopy(graph)
    propagate_structural_identity(graph, 0, [1], 0.0)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph

    graph.nodes[1]["nu_f"] = 1.6e308
    propagate_structural_identity(graph, 0, [1], 0.5)
    expected = float((Fraction.from_float(1.6e308) + Fraction.from_float(1.7e308)) / 2)
    assert graph.nodes[1]["nu_f"] == expected
    assert 1.6e308 <= expected <= 1.7e308
    assert graph.nodes[1]["EPI"] == 0.625

    graph = _graph()
    graph.nodes[1]["nu_f"] = math.ulp(0.0)
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="underflows"):
        propagate_structural_identity(graph, 0, [1], 0.5)
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_uniform_real_bepi_retains_sign_and_rich_state_is_rejected():
    graph = _graph()
    graph.nodes[0]["EPI"] = ensure_bepi(-0.75)
    graph.nodes[1]["EPI"] = ensure_bepi(-0.25)
    propagate_structural_identity(graph, 0, [1], 0.5)
    assert graph.nodes[1]["EPI"] == -0.5

    graph.nodes[2]["EPI"] = ensure_bepi(((1, 2), (1, 2), (0, 1)))
    before = deepcopy(graph.nodes[1])
    with pytest.raises(ValueError):
        propagate_structural_identity(graph, 0, [1, 2])
    assert graph.nodes[1] == before
