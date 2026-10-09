"""Exact cycle-space/phase reconstruction and separate production controls.

Rational coordinates are turns of mathematical 2*pi, not rationalized radians.
Numerical winding and source checks never certify the ideal symbolic state.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import networkx as nx
import pytest

from tests.joint_phase_helpers import configure
from tests.joint_phase_helpers import exact_phase_cycle_state as _state
from tests.joint_phase_helpers import execute_joint_step
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.phase_cycle_geometry import (
    derive_phase_cycle_geometry,
    reconstruct_circular_phase_state,
    reconstruct_phase_cycle_state,
)
from tnfr.physics.phase_response import observe_phase_lock_source
from tnfr.physics.winding_certificates import certify_phase_winding


def test_circular_reconstruction_proves_supplementary_balance_without_relaxing_acute():
    graph = nx.cycle_graph(5)
    geometry = derive_phase_cycle_geometry(graph)
    turns = tuple(F(i, 6) for i in range(5))
    gaps = tuple(turns[j] - turns[i] for i, j in geometry.edges)
    circular = reconstruct_circular_phase_state(geometry, edge_turns=gaps)
    assert circular.nodal_turns == turns
    assert not any(circular.symbolic_sine_coefficients)
    assert circular.sine_balance_status == "proved_by_period_reflection_cancellation"
    with pytest.raises(ValueError, match="strictly acute"):
        reconstruct_phase_cycle_state(geometry, edge_turns=gaps)
    # Integral edge-lift changes preserve the circular phases and sine proof,
    # while their recorded circulations and edge offsets remain distinct.
    lifted = reconstruct_circular_phase_state(
        geometry, edge_turns=tuple(value + 7 * (i + 1) for i, value in enumerate(gaps))
    )
    assert lifted.nodal_turns == circular.nodal_turns
    assert lifted.symbolic_sine_coefficients == circular.symbolic_sine_coefficients
    assert lifted.edge_integer_offsets != circular.edge_integer_offsets


def test_circular_half_turn_is_exact_zero_current_not_an_acute_or_winding_claim():
    geometry = derive_phase_cycle_geometry(nx.path_graph(2))
    result = reconstruct_circular_phase_state(geometry, edge_turns=(F(1, 2),))
    assert result.nodal_turns == (F(0), F(1, 2))
    assert result.symbolic_sine_coefficients == ((), ())
    assert result.cycle_periods == ()
    assert (
        "edge_lift_periods_are_not_principal_windings_at_antipodal_edges"
        in result.scope
    )
    assert (
        reconstruct_circular_phase_state(
            geometry, edge_turns=(F(1, 4),)
        ).sine_balance_status
        == "unresolved"
    )
    with pytest.raises(ValueError, match="strictly acute"):
        reconstruct_phase_cycle_state(geometry, edge_turns=(F(1, 2),))


@pytest.mark.parametrize("value", (True, float("nan"), complex(0, 1)))
def test_circular_reconstruction_retains_shared_real_turn_admission(value):
    geometry = derive_phase_cycle_geometry(nx.path_graph(2))
    with pytest.raises((TypeError, ValueError)):
        reconstruct_circular_phase_state(geometry, edge_turns=(value,))


def test_circular_reconstruction_rejects_nonintegral_period_and_tampered_support():
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    with pytest.raises(ValueError, match="integral"):
        reconstruct_circular_phase_state(geometry, edge_turns=(F(1, 7),) * 5)
    with pytest.raises(ValueError, match="derived fields"):
        reconstruct_circular_phase_state(
            replace(geometry, cycle_rank=4), edge_turns=(0,) * 5
        )


@pytest.mark.parametrize(
    "graph,rank,short_rank",
    [
        (nx.path_graph(6), 0, 0),
        (nx.cycle_graph(3), 1, 1),
        (nx.cycle_graph(4), 1, 1),
        (nx.cycle_graph(5), 1, 0),
        (nx.cycle_graph(6), 1, 0),
        (nx.complete_graph(5), 6, 6),
        (nx.barbell_graph(3, 0), 2, 2),
    ],
)
def test_integer_fundamental_basis_and_short_cycle_obstruction(graph, rank, short_rank):
    result = derive_phase_cycle_geometry(graph)
    n, m = len(result.nodes), len(result.edges)
    assert result.cycle_rank == m - n + 1 == rank
    assert result.short_cycle_span_rank == short_rank
    assert result.short_cycle_consensus_only == (rank == short_rank)
    assert (
        exact_rank(tuple(tuple(F(v) for v in row) for row in result.incidence)) == n - 1
    )
    assert (
        exact_rank(tuple(tuple(F(v) for v in row) for row in result.cycle_rows)) == rank
    )
    chords = [i for i in range(m) if i not in result.tree_edge_indices]
    assert tuple(tuple(row[j] for j in chords) for row in result.cycle_rows) == tuple(
        tuple(int(i == j) for j in range(rank)) for i in range(rank)
    )
    for cycle, row in zip(result.fundamental_cycles, result.cycle_rows, strict=True):
        assert len(set(cycle)) == len(cycle) >= 3
        assert all(
            sum(b * c for b, c in zip(node_row, row, strict=True)) == 0
            for node_row in result.incidence
        )
        assert sum(abs(value) for value in row) == len(cycle)
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    assert {
        frozenset((result.nodes[result.edges[e][0]], result.nodes[result.edges[e][1]]))
        for e in result.bridge_edge_indices
    } == bridges


def test_topology_readout_ignores_transport_weights_state_and_cache_and_is_frozen():
    graph = nx.cycle_graph(5)
    graph.graph["cache"] = {"sentinel": [1, 2]}
    nx.set_node_attributes(graph, "not a phase", "theta")
    for edge in graph.edges:
        graph.edges[edge]["weight"] = float("nan")
    graph.edges[0, 1]["weight"] = 0
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    metadata = deepcopy(graph.graph)
    result = derive_phase_cycle_geometry(graph)
    assert result == derive_phase_cycle_geometry(nx.cycle_graph(5))
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == metadata
    assert math.isnan(graph.edges[1, 2]["weight"])
    with pytest.raises(FrozenInstanceError):
        result.cycle_rank = 2


@pytest.mark.parametrize("size,winding", [(5, 1), (5, -1), (6, 1), (9, 2)])
def test_uniform_unicyclic_twists_reconstruct_exactly_and_balance_by_oddness(
    size, winding
):
    graph = nx.cycle_graph(size)
    graph.add_edges_from([(0, size), (size, size + 1)])
    phases = {node: F(winding * node, size) % 1 for node in range(size)}
    phases.update({size: F(0), size + 1: F(0)})
    result = _state(graph, phases)
    assert result.nodal_turns == tuple(phases[node] for node in result.geometry.nodes)
    assert tuple(map(abs, result.cycle_periods)) == (abs(winding),)
    assert result.sine_balance_status == "proved_by_odd_cancellation"
    assert result.symbolic_sine_coefficients == ((),) * (size + 2)
    assert all(result.edge_turns[e] == 0 for e in result.geometry.bridge_edge_indices)
    for (i, j), gap, offset in zip(
        result.geometry.edges,
        result.edge_turns,
        result.edge_integer_offsets,
        strict=True,
    ):
        assert result.nodal_turns[j] - result.nodal_turns[i] - gap == offset
        assert type(offset) is int


def test_two_independent_windings_survive_a_bridge_without_bridge_current():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(6))
    graph.add_edge(0, 5)
    phases = {i: F(i, 5) for i in range(5)}
    phases.update({i + 5: -F(i, 6) % 1 for i in range(6)})
    result = _state(graph, phases)
    assert result.geometry.cycle_rank == 2
    assert sorted(map(abs, result.cycle_periods)) == [1, 1]
    assert result.sine_balance_status == "proved_by_odd_cancellation"
    assert len(result.geometry.bridge_edge_indices) == 1
    assert result.edge_turns[result.geometry.bridge_edge_indices[0]] == 0


def test_integral_periods_do_not_certify_sine_balance_or_sector_existence():
    graph = nx.cycle_graph(6)
    phases = {i: F(i, 6) for i in graph}
    original = _state(graph, phases)
    phases[1] += F(1, 100)
    perturbed = _state(graph, phases)
    assert perturbed.cycle_periods == original.cycle_periods
    assert perturbed.sine_balance_status == "unresolved"
    assert any(perturbed.symbolic_sine_coefficients)
    # A nonconstant zero-period tree state also reconstructs, but cannot lock
    # under the separate equal-capacity acute-tree theorem.
    tree = _state(nx.path_graph(4), {i: F(i, 10) for i in range(4)})
    assert tree.cycle_periods == ()
    assert tree.geometry.short_cycle_consensus_only
    assert tree.sine_balance_status == "unresolved"


def test_a_true_multicycle_lock_can_be_unresolved_by_the_oddness_only_checker():
    # Three internally disjoint A->B paths have equal endpoint phases modulo
    # one turn. Their distinct sine currents balance by a pentagon identity,
    # not by opposite equal-angle pairs at A and B.
    graph = nx.Graph()
    graph.add_nodes_from((0, 1))
    phases = {0: F(0), 1: F(1, 2)}
    next_node = 2
    for length, turn in ((10, F(1, 20)), (6, F(1, 12)), (10, F(-3, 20))):
        interior = list(range(next_node, next_node + length - 1))
        next_node += length - 1
        path = [0, *interior, 1]
        graph.add_edges_from(zip(path, path[1:]))
        phases.update({node: k * turn % 1 for k, node in enumerate(interior, 1)})
        assert length * turn % 1 == F(1, 2)
    state = _state(graph, phases)
    assert state.geometry.cycle_rank == 2
    assert state.sine_balance_status == "unresolved"
    assert state.symbolic_sine_coefficients[0] == (
        (F(1, 20), 1),
        (F(1, 12), 1),
        (F(3, 20), -1),
    )
    assert all(not row for row in state.symbolic_sine_coefficients[2:])
    # Exact a+b*sqrt(5) coefficients: sin(pi/10)=(sqrt(5)-1)/4,
    # sin(pi/6)=1/2, sin(3*pi/10)=(sqrt(5)+1)/4.
    radicals = {
        F(1, 20): (F(-1, 4), F(1, 4)),
        F(1, 12): (F(1, 2), F(0)),
        F(3, 20): (F(1, 4), F(1, 4)),
    }
    for row in state.symbolic_sine_coefficients[:2]:
        assert tuple(sum(c * radicals[a][k] for a, c in row) for k in (0, 1)) == (0, 0)


def test_nonintegral_cycle_period_is_rejected_even_when_currents_would_cancel():
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    # Half a turn around C5: each vertex's formal sine currents cancel, but
    # no circle-valued node phases can realize these edge differences.
    gaps = tuple(F(sign, 10) for sign in geometry.cycle_rows[0])
    assert sum(a * c for a, c in zip(gaps, geometry.cycle_rows[0])) == F(1, 2)
    with pytest.raises(ValueError, match="integral|integer"):
        reconstruct_phase_cycle_state(geometry, edge_turns=gaps)


def test_doubled_real_cycle_basis_is_not_an_integer_cycle_lattice_certificate():
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    doubled = replace(
        geometry,
        cycle_rows=tuple(tuple(2 * c for c in row) for row in geometry.cycle_rows),
    )
    with pytest.raises(ValueError, match="rebuild|primitive|match|differ"):
        reconstruct_phase_cycle_state(doubled, edge_turns=(0,) * 5)


@pytest.mark.parametrize(
    "field,value",
    [("cycle_rank", 99), ("short_cycle_span_rank", 1), ("bridge_edge_indices", (0,))],
)
def test_detached_geometry_caches_cannot_spoof_admission(field, value):
    geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
    with pytest.raises(ValueError, match="rebuild|primitive|match|differ"):
        reconstruct_phase_cycle_state(
            replace(geometry, **{field: value}), edge_turns=(0,) * 5
        )


def test_common_rotation_relabeling_and_orientation_do_not_erase_signed_geometry():
    graph = nx.cycle_graph(5)
    phases = {i: F(i, 5) for i in graph}
    result = _state(graph, phases)
    rotated = _state(graph, {i: v + F(7, 13) for i, v in phases.items()})
    assert rotated == result
    reversed_state = _state(graph, {i: -v for i, v in phases.items()})
    assert reversed_state.cycle_periods == tuple(-w for w in result.cycle_periods)
    assert reversed_state.edge_turns == tuple(-a for a in result.edge_turns)
    mapping = {i: ("n", (3 * i) % 5) for i in graph}
    relabeled = _state(
        nx.relabel_nodes(graph, mapping), {mapping[i]: p for i, p in phases.items()}
    )
    assert relabeled.nodal_turns == result.nodal_turns
    assert relabeled.cycle_periods == result.cycle_periods
    # Reordering vertices changes the coordinate basis/root, not phase differences.
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(graph)))
    reordered.add_edges_from(graph.edges)
    changed = _state(reordered, phases)
    reconstructed = dict(zip(changed.geometry.nodes, changed.nodal_turns))
    assert all(
        (reconstructed[j] - reconstructed[i]) % 1 == (phases[j] - phases[i]) % 1
        for i, j in graph.edges
    )


def test_long_cycle_plus_chord_can_be_excluded_by_short_cycle_span():
    graph = nx.cycle_graph(5)
    graph.add_edge(0, 2)
    geometry = derive_phase_cycle_geometry(graph)
    assert geometry.cycle_rank == geometry.short_cycle_span_rank == 2
    assert geometry.short_cycle_consensus_only
    with pytest.raises(ValueError, match="acute|quarter|1/4"):
        _state(graph, {i: F(i, 5) for i in graph})


def test_exact_prepared_sector_has_separate_finite_engine_and_winding_observations():
    graph = nx.cycle_graph(5)
    prepared = _state(graph, {i: F(i, 5) for i in graph})
    configure(graph)
    for i, phase in enumerate(prepared.nodal_turns):
        graph.nodes[i].update(
            EPI=0.125, theta=float(phase) * math.tau, nu_f=1.0, delta_nfr=0.0, dEPI=0.0
        )
    default_compute_delta_nfr(graph)
    observed = observe_phase_lock_source(graph, coupling_strength=0.5)
    assert observed.full_u3_admission and observed.strict_acute_edges_estimate
    assert max(map(abs, observed.full_support_rate_residual)) < 3e-15
    assert max(map(abs, observed.capture.phase_gradient)) < 3e-15
    for cycle, period in zip(
        prepared.geometry.fundamental_cycles, prepared.cycle_periods
    ):
        certificate = certify_phase_winding(
            graph, [prepared.geometry.nodes[i] for i in cycle]
        )
        assert certificate.is_defined and certificate.winding == period
    step = execute_joint_step(graph, dt=0.125, coupling_strength=0.5, time=0)
    assert tuple(map(float, step.after_epi.epi)) == pytest.approx(
        (0.125,) * 5, abs=3e-15
    )
    assert certify_phase_winding(graph, range(5)).winding == 1
    # Exiting a stricter U3 gate is not crossing the winding branch.
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.19 * math.tau
    gated = observe_phase_lock_source(graph, coupling_strength=0.5)
    certificate = certify_phase_winding(graph, range(5), phase_gate=0.19 * math.tau)
    assert not gated.full_u3_admission
    assert certificate.is_defined and certificate.winding == 1
    assert not certificate.u3_admissible


@pytest.mark.parametrize(
    "value", [F(1, 4), F(-1, 4), F(1, 3), True, float("nan"), "0.1"]
)
def test_invalid_or_nonacute_exact_turns_are_rejected(value):
    geometry = derive_phase_cycle_geometry(nx.path_graph(2))
    with pytest.raises((TypeError, ValueError)):
        reconstruct_phase_cycle_state(geometry, edge_turns=(value,))


@pytest.mark.parametrize("values", [(), (0, 0), {0}, {0: 0}])
def test_ambiguous_or_wrong_size_gap_containers_are_rejected(values):
    geometry = derive_phase_cycle_geometry(nx.path_graph(2))
    with pytest.raises((TypeError, ValueError)):
        reconstruct_phase_cycle_state(geometry, edge_turns=values)


@pytest.mark.parametrize(
    "graph",
    [
        nx.Graph(),
        nx.empty_graph(1),
        nx.empty_graph(3),
        nx.cycle_graph(5, create_using=nx.DiGraph),
        nx.MultiGraph([(0, 1)]),
        nx.Graph([(0, 0), (0, 1)]),
    ],
)
def test_unsupported_phase_supports_are_rejected(graph):
    with pytest.raises((TypeError, ValueError)):
        derive_phase_cycle_geometry(graph)


@pytest.mark.parametrize("graph", [nx.path_graph(33), nx.complete_graph(11)])
def test_declared_computation_budget_is_enforced(graph):
    with pytest.raises(ValueError, match="32|50"):
        derive_phase_cycle_geometry(graph)
