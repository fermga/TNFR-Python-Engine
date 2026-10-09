"""Exact full-Hessian controls for conditional bridge composition."""

import networkx as nx
import pytest

from tnfr.physics.phase_cycle_geometry import (
    classify_c5_sine_critical_set,
    compose_bridge_tree_hessian_inertia,
    derive_phase_cycle_geometry,
)
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads


def test_full_nodal_congruence_retains_internal_modes_and_arbitrary_ports():
    s = pytest.importorskip("sympy")
    # A flat triangle, an antipodal pair, a quarter-turn C4 and a singleton.
    # Their cosine Hessians include positive, negative and degenerate blocks.
    components = ((0, 1, 2), (3, 4), (5, 6, 7, 8), (9,))
    internal_edges = (
        ((0, 1, 1), (1, 2, 1), (0, 2, 1)),
        ((3, 4, -1),),
        ((5, 6, 0), (6, 7, 0), (7, 8, 0), (5, 8, 0)),
        (),
    )
    actual_bridges = ((1, 4, 1), (3, 7, -1), (2, 9, 1))
    u = s.symbols("u0:6")
    b = s.symbols("b0:3")
    local = (0, u[0], u[1], 0, u[2], 0, u[3], u[4], u[5], 0)
    # Solve bridge differences for component origins; all internal coordinates
    # survive and neither endpoint needs to be a component anchor.
    origins = (0, b[0] + local[1] - local[4], 0, b[2] + local[2])
    origins = (origins[0], origins[1], origins[1] + b[1] - local[7], origins[3])
    phase = s.Matrix(
        [
            local[node] + origins[index]
            for index, nodes in enumerate(components)
            for node in nodes
        ]
    )
    transform = phase.jacobian(u + b)
    assert transform.rank() == 9
    full = s.zeros(10)
    blocks = []
    for nodes, edges in zip(components, internal_edges):
        block = s.zeros(len(nodes))
        for i, j, cosine in edges:
            for matrix, p, q in ((full, i, j), (block, nodes.index(i), nodes.index(j))):
                matrix[p, p] += cosine
                matrix[q, q] += cosine
                matrix[p, q] -= cosine
                matrix[q, p] -= cosine
        blocks.append(block[1:, 1:])
    for i, j, cosine in actual_bridges:
        full[i, i] += cosine
        full[j, j] += cosine
        full[i, j] -= cosine
        full[j, i] -= cosine
    expected = s.diag(*blocks, 1, -1, 1)
    assert transform.T * full * transform == expected
    assert full * s.ones(10, 1) == s.zeros(10, 1)
    report = compose_bridge_tree_hessian_inertia(
        ((2, 0, 0), (0, 1, 0), (0, 0, 3), (0, 0, 0)),
        ((0, 1, 1), (1, 2, -1), (0, 3, 1)),
    )
    assert report.relative_inertia == (4, 2, 3)
    assert report.total_nodes == 10
    assert report.common_phase_nullity == 1


def test_tree_cut_forces_zero_current_and_each_antipodal_bridge_is_negative():
    s = pytest.importorskip("sympy")
    edges = ((0, 1), (1, 2), (1, 3), (3, 4))
    incidence = s.zeros(5, 4)
    for edge, (i, j) in enumerate(edges):
        incidence[i, edge], incidence[j, edge] = -1, 1
    assert incidence.rank() == 4
    # On a tree, zero nodal sine current forces every edge sine to vanish.
    assert incidence.nullspace() == []
    for signs in ((1, 1, 1, 1), (1, -1, 1, 1)):
        report = compose_bridge_tree_hessian_inertia(
            ((0, 0, 0),) * 5,
            tuple((*edge, sign) for edge, sign in zip(edges, signs)),
        )
        pinned = (incidence * s.diag(*signs) * incidence.T)[:4, :4]
        inverse = incidence[:4, :].inv()
        assert inverse * pinned * inverse.T == s.diag(*signs)
        assert report.relative_inertia == (signs.count(1), signs.count(-1), 0)


def test_c5_reader_uses_general_composition_without_replacing_its_critical_proof(
    monkeypatch,
):
    import tnfr.physics.phase_cycle_geometry as owner

    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 10), (5, 10)))
    family = classify_c5_sine_critical_set(
        derive_phase_cycle_geometry(graph),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
    )
    stable_choices = tuple(
        i for i, mask in enumerate(family.cycle_supplementary_masks) if mask == 0
    )
    assert len(stable_choices) == 3
    original = owner.compose_bridge_tree_hessian_inertia
    calls = []

    def capture(components, bridges):
        result = original(components, bridges)
        calls.append(result)
        return result

    monkeypatch.setattr(owner, "compose_bridge_tree_hessian_inertia", capture)
    for first in stable_choices:
        for second in stable_choices:
            report = family.phase_hessian_inertia(
                cycle_choices=(first, second), bridge_turns=(0, 0)
            )
            assert report.relative_inertia == (10, 0, 0)
            assert (
                report.state.sine_balance_status
                == "proved_by_period_reflection_cancellation"
            )
    assert len(calls) == 9
    assert all(item.total_nodes == 11 for item in calls)
    # A third supplied C5 adds its four internal directions and a bridge,
    # rather than extending a universal list of nine possible geometries.
    third = original(((4, 0, 0),) * 3, ((0, 1, 1), (1, 2, 1)))
    assert third.relative_inertia == (14, 0, 0)


@pytest.mark.parametrize(
    "components,bridges",
    (
        ((), ()),
        (((True, 0, 0),), ()),
        (((1.0, 0, 0),), ()),
        (((-1, 0, 0),), ()),
        (((1, 0),), ()),
        (((0, 0, 0),) * 2, ((False, 1, 1),)),
        (((0, 0, 0),) * 2, ((0, 1, True),)),
        (((0, 0, 0),) * 2, ((0, 1, 0),)),
        (((0, 0, 0),) * 2, ((0, 2, 1),)),
        (((0, 0, 0),) * 2, ((0, 0, 1),)),
        (((0, 0, 0),) * 3, ((0, 1, 1), (1, 0, -1))),
        (((0, 0, 0),) * 4, ((0, 1, 1), (1, 2, 1), (2, 0, 1))),
        (((0, 0, 0),) * 3, ((0, 1, 1),)),
    ),
)
def test_invalid_component_or_nonbridge_support_cannot_acquire_a_certificate(
    components, bridges
):
    with pytest.raises((ValueError, TypeError)):
        compose_bridge_tree_hessian_inertia(components, bridges)


def test_single_component_degeneracy_and_exact_export_keep_the_conditional_scope(
    tmp_path,
):
    single = compose_bridge_tree_hessian_inertia(((2, 1, 3),), ())
    assert single.relative_inertia == (2, 1, 3)
    assert single.total_nodes == 7
    assert not hasattr(single, "local_exponential_attraction_certified")
    assert not hasattr(single, "formation_certified")
    point = compose_bridge_tree_hessian_inertia(((0, 0, 0),), ())
    assert point.total_nodes == 1 and point.relative_inertia == (0, 0, 0)
    payload = single.to_dict()
    assert payload["schema"] == "tnfr.bridge-tree-hessian-inertia.v1"
    path = tmp_path / "bridge-inertia.json"
    export_to_json(payload, path)
    assert json_loads(path.read_text(encoding="utf-8")) == payload
