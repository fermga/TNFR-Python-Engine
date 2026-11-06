"""UM/RA raw-state admission and outgoing-support execution boundaries."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators import _op_UM, apply_glyph
from tnfr.operators.definitions import Coupling, Resonance
from tnfr.operators.network_stage import execute_coupling_stage, execute_neighbor_stage
from tnfr.utils.cache import cached_nodes_and_A


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.0, nu_f=1.0, theta=0.0, dnfr=0.0, glyph_history=["AL"]
        )
    graph.graph.update(UM_FUNCTIONAL_LINKS=False, RANDOM_SEED=17)
    return graph


@pytest.mark.parametrize(
    "glyph,field,value",
    [
        ("RA", "neighbor_phase", "0.1"),
        ("UM", "target_phase", Fraction(1, 2**2000)),
        ("RA", "DELTA_PHI_MAX", "1.0"),
        ("UM", "UM_MAX_PHASE_DIFF", Fraction(-1, 2**2000)),
    ],
)
def test_phase_gate_rejects_coercion_before_any_node_or_edge_write(glyph, field, value):
    graph = _graph()
    if field == "neighbor_phase":
        graph.nodes[1][ALIAS_THETA[0]] = value
    elif field == "target_phase":
        graph.nodes[0][ALIAS_THETA[0]] = value
    else:
        graph.graph[field] = value
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRValueError, match="phase gate"):
        apply_glyph(graph, 0, glyph)
    assert dict(graph.nodes(data=True)) == before
    assert list(graph.edges) == [(0, 1)]
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("capacity", [True, "2", -1.0, Fraction(-1, 2**2000)])
def test_ra_inactive_amplification_still_requires_raw_nonnegative_capacity(
    staged, capacity
):
    graph = _graph()
    graph.nodes[0][ALIAS_VF[0]] = capacity
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRValueError, match="capacity"):
        if staged:
            execute_neighbor_stage(graph, Resonance(), (0,))
        else:
            apply_glyph(graph, 0, "RA")
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("staged", [False, True])
def test_ra_zero_capacity_is_valid_even_when_amplification_is_active(staged):
    graph = _graph()
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    graph.nodes[1][ALIAS_EPI[0]] = 0.6
    graph.graph["GLYPH_FACTORS"] = {"RA_epi_diff": 0.5}
    if staged:
        execute_neighbor_stage(graph, Resonance(), (0,))
    else:
        apply_glyph(graph, 0, "RA")
    assert get_attr(graph.nodes[0], ALIAS_VF, None) == 0.0
    # The operator is a configured event, not zero-capacity continuous flow.
    assert get_attr(graph.nodes[0], ALIAS_EPI, None) == pytest.approx(0.3)


def test_um_rejects_underflowing_negative_neighbor_capacity_before_commit():
    graph = _graph()
    graph.nodes[1][ALIAS_VF[0]] = Fraction(-1, 2**2000)
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRValueError, match="underflows"):
        apply_glyph(graph, 0, "UM")
    assert dict(graph.nodes(data=True)) == before


def test_graphless_unidirectional_um_commits_only_the_target_phase():
    neighbor = SimpleNamespace(theta=0.4)
    target = SimpleNamespace(
        theta=0.0,
        graph={"UM_BIDIRECTIONAL": False},
        neighbors=lambda: [neighbor],
    )
    _op_UM(target, {"UM_theta_push": 0.5})
    assert target.theta == pytest.approx(0.2)
    assert neighbor.theta == 0.4


@pytest.mark.parametrize("glyph", ["UM", "RA"])
@pytest.mark.parametrize("staged", [False, True])
def test_coupled_channels_use_unique_outgoing_support_including_zero_weights(
    glyph, staged
):
    graph = nx.MultiDiGraph()
    for node, value in enumerate((0.2, 0.4, 0.8, 0.99)):
        graph.add_node(
            node, EPI=value, nu_f=value, theta=0.0, dnfr=0.0, glyph_history=["AL"]
        )
    graph.add_edge(0, 1, weight=1000.0)
    graph.add_edge(0, 1, weight=1000.0)
    graph.add_edge(0, 2, weight=0.0)
    graph.add_edge(3, 0, weight=1000.0)
    graph.graph.update(
        UM_FUNCTIONAL_LINKS=False,
        RANDOM_SEED=17,
        GLYPH_FACTORS={"UM_vf_sync": 1.0, "RA_epi_diff": 0.5},
    )
    before_edges = list(graph.edges(keys=True, data=True))
    if not staged:
        apply_glyph(graph, 0, glyph)
    elif glyph == "UM":
        execute_coupling_stage(graph, Coupling(), (0,))
    else:
        execute_neighbor_stage(graph, Resonance(), (0,))
    if glyph == "UM":
        assert get_attr(graph.nodes[0], ALIAS_VF, None) == pytest.approx(0.6)
    else:
        assert get_attr(graph.nodes[0], ALIAS_EPI, None) == pytest.approx(0.4)
    assert list(graph.edges(keys=True, data=True)) == before_edges


def test_um_functional_link_invalidates_prepared_outgoing_support():
    graph = _graph()
    graph.add_node(2, EPI=0.0, nu_f=1.0, theta=0.0, dnfr=0.0)
    graph.graph.update(UM_FUNCTIONAL_LINKS=True, UM_COMPAT_THRESHOLD=0.0)
    nodes_before, before = cached_nodes_and_A(graph, require_numpy=True)
    assert before[nodes_before.index(0), nodes_before.index(2)] == 0.0

    apply_glyph(graph, 0, "UM")

    nodes_after, after = cached_nodes_and_A(graph, require_numpy=True)
    assert graph.has_edge(0, 2)
    assert after[nodes_after.index(0), nodes_after.index(2)] == 1.0
    assert after is not before
