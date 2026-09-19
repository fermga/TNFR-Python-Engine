"""Exact declared-source solvability and capacity-band controls on fixed support.

These detached identities select no phase/capacity law and authenticate no
runtime trajectory or correspondence between supplied phase data and angles.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics._cycle_algebra import dot
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.phase_response import (
    derive_phase_capacity_balance,
    derive_phase_response,
)
from tnfr.physics.support_transport import observe_support_transport


def _state(graph=None, *, epi=None, capacity=None):
    graph = nx.path_graph(2) if graph is None else graph
    count = len(graph)
    epi = (Q(1, 4),) * count if epi is None else epi
    capacity = (Q(1),) * count if capacity is None else capacity
    for node, x, nu in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=0.0, delta_nfr=0.0)
    return graph, observe_support_transport(graph)


def _balance(source, **overrides):
    options = dict(
        phase_gradient=(0,) * len(source.nodes),
        forcing=(0,) * len(source.nodes),
        phase_weight=Q(1, 4),
        capacity_weight=Q(1, 4),
    )
    options.update(overrides)
    return derive_phase_capacity_balance(source, **options)


@pytest.mark.parametrize("t", (Q(-1, 8), Q(0), Q(1, 8)))
def test_p2_joint_compensation_family_has_the_exact_centered_profile_and_band(t):
    capacity = (1 - t, Q(3, 2) + t)
    _, source = _state(epi=(Q(1, 4), 0), capacity=capacity)
    result = _balance(
        source,
        phase_gradient=(-2 * t, 2 * t),
        forcing=(Q(1, 8), Q(-1, 8)),
        capacity_lower=(Q(3, 4),) * 2,
        capacity_upper=(Q(7, 4),) * 2,
    )

    assert result.source_compatible
    assert result.support_degrees == (1, 1)
    assert result.source_difference == (Q(-1, 8) - t / 2, Q(1, 8) + t / 2)
    assert result.compatibility_residual == result.center_residual == 0
    assert result.centered_capacity == (Q(-1, 4) - t, Q(1, 4) + t)
    assert result.equation_residual == (0, 0)
    assert result.uniform_shift_lower == 1 + t
    assert result.uniform_shift_lower_inclusive
    assert result.uniform_shift_upper == Q(3, 2) - t
    assert result.has_admissible_capacity
    assert tuple(z + Q(5, 4) for z in result.centered_capacity) == capacity
    assert result.uniform_shift_lower <= Q(5, 4) <= result.uniform_shift_upper
    # The declared singleton phase source balances the existing capacity
    # channel exactly. The EPI channel contributes the opposite forcing.
    assert (
        tuple(
            result.phase_weight * g + result.capacity_weight * h
            for g, h in zip(
                result.phase_gradient, source.capacity_gradient, strict=True
            )
        )
        == result.forcing
    )
    assert tuple(-Q(1, 2) * value for value in source.epi_gradient) == result.forcing


def test_strict_u3_star_phase_source_has_an_exact_range_obstruction():
    _, source = _state(nx.star_graph(3))
    phase_over_pi = (0, Q(1, 3), Q(1, 3), Q(-1, 3))
    assert all(abs(phase_over_pi[j] - phase_over_pi[0]) < Q(1, 2) for j in (1, 2, 3))
    # The center resultant is 2*exp(i*pi/3)+exp(-i*pi/3), whose
    # direction is pi/6. Each leaf has the center as its sole neighbor.
    phase_source = (Q(1, 6), Q(-1, 3), Q(-1, 3), Q(1, 3))
    result = _balance(source, phase_gradient=phase_source, phase_weight=1)

    assert result.support_degrees == (3, 1, 1, 1)
    assert result.source_difference == phase_source
    assert result.compatibility_residual == Q(1, 6)
    assert not result.source_compatible
    assert not result.has_admissible_capacity
    assert result.centered_capacity is result.equation_residual is None
    assert result.center_residual is None
    assert result.uniform_shift_lower is result.uniform_shift_upper is None
    assert not result.uniform_shift_lower_inclusive


def test_topology_channel_uses_the_same_support_poisson_equation():
    _, source = _state(nx.star_graph(3))
    result = _balance(source, phase_weight=0, topology_weight=Q(1, 2))

    assert source.topology_gradient == (-2, 2, 2, 2)
    assert result.source_difference == (-1, 1, 1, 1)
    assert result.centered_capacity == (-2, 2, 2, 2)
    assert result.equation_residual == (0, 0, 0, 0)
    assert result.compatibility_residual == result.center_residual == 0
    assert result.uniform_shift_lower == 2
    assert not result.uniform_shift_lower_inclusive
    assert result.uniform_shift_upper is None
    assert result.has_admissible_capacity


def test_nonuniform_conductance_does_not_replace_the_unique_support_metric():
    graph = nx.path_graph(3)
    graph[0][1]["weight"] = 1.0
    graph[1][2]["weight"] = 7.0
    profile = (Q(1, 4), Q(0), Q(-1, 4))
    _, source = _state(graph, epi=profile)
    result = _balance(source, phase_gradient=profile, phase_weight=1, capacity_weight=1)

    assert result.support_degrees == (1, 2, 1)
    assert result.centered_capacity == profile
    assert result.equation_residual == (0, 0, 0)
    assert result.center_residual == 0
    # Weighted EPI transport produces a nonzero middle row for this profile;
    # the unweighted capacity mean instead has zero middle-row response.
    assert -source.epi_gradient[1] == Q(3, 16)
    assert result.source_difference[1] == 0


def test_self_loop_and_parallel_edges_each_enter_unique_support_once():
    graph = nx.MultiGraph()
    graph.add_edges_from(((0, 0), (0, 1), (0, 1)))
    _, source = _state(graph)
    result = _balance(
        source, phase_gradient=(Q(1, 2), -1), phase_weight=1, capacity_weight=1
    )

    assert graph.number_of_edges(0, 1) == 2
    assert source.support_neighbors == ((0, 1), (0,))
    assert source.conductance == ((0, 0, 1), (0, 1, 2), (1, 0, 2))
    assert result.support_degrees == (2, 1)
    assert result.centered_capacity == (Q(1, 3), Q(-2, 3))
    assert result.equation_residual == (0, 0)
    assert result.compatibility_residual == result.center_residual == 0


def test_reflected_p2_phases_share_the_gram_response_but_reverse_capacity_profile():
    _, source = _state()
    # cos(pi/3)=cos(-pi/3)=1/2. Reflection preserves this exact Gram,
    # while singleton mean displacements reverse in the regular U3 chart.
    gram = ((1, Q(1, 2)), (Q(1, 2), 1))
    references = []
    results = []
    for phase_over_pi in ((0, Q(1, 3)), (0, Q(-1, 3))):
        displacement = phase_over_pi[1] - phase_over_pi[0]
        assert abs(displacement) < Q(1, 2)
        reference = derive_phase_response(
            cosine_gram=gram,
            mean_neighbors=((1,), (0,)),
            receiver_sources=((0,), (1,)),
            phase_factor=1,
        )
        references.append(reference)
        results.append(
            _balance(
                source,
                phase_gradient=(displacement, -displacement),
                phase_weight=1,
                capacity_weight=1,
            )
        )

    assert references[0] == references[1]
    assert references[0].mean_response == ((0, 1), (1, 0))
    assert results[0].centered_capacity == (Q(1, 6), Q(-1, 6))
    assert results[1].centered_capacity == tuple(
        -z for z in results[0].centered_capacity
    )
    assert all(result.equation_residual == (0, 0) for result in results)
    # A Gram/response alone therefore cannot authenticate oriented source g.


@pytest.mark.parametrize("directed", (False, True))
def test_reciprocal_zero_conductance_support_still_determines_capacity(directed):
    graph = nx.path_graph(3)
    if directed:
        graph = graph.to_directed()
    nx.set_edge_attributes(graph, 0.0, "weight")
    _, source = _state(graph)
    result = _balance(
        source, phase_gradient=(Q(1, 4), 0, Q(-1, 4)), phase_weight=1, capacity_weight=1
    )

    assert source.conductance == ()
    assert source.epi_gradient == (0, 0, 0)
    assert result.source_compatible
    assert result.centered_capacity == (Q(1, 4), 0, Q(-1, 4))
    assert result.equation_residual == (0, 0, 0)


def test_singleton_self_loop_has_only_uniform_capacity_freedom():
    graph = nx.Graph()
    graph.add_edge("node", "node", weight=0.0)
    _, source = _state(graph)
    result = _balance(source)

    assert result.support_degrees == (1,)
    assert result.centered_capacity == result.equation_residual == (0,)
    assert result.uniform_shift_lower == 0
    assert not result.uniform_shift_lower_inclusive
    assert result.has_admissible_capacity
    obstruction = _balance(source, forcing=(1,))
    assert obstruction.compatibility_residual == -1
    assert not obstruction.source_compatible


def test_canonical_forcing_decomposition_reconstructs_centered_live_capacity_without_writes():
    graph = nx.star_graph(3)
    for edge, weight in zip(graph.edges, (2.0, 1.0, 0.0), strict=True):
        graph.edges[edge]["weight"] = weight
    _state(graph, epi=(0.0, 0.25, 0.5, 0.75), capacity=(1, 2, 3, 4))
    for node in graph:
        graph.nodes[node]["theta"] = 0.2 * node
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.25, "phase": 0.25, "vf": 0.25, "topo": 0.25}
    before = deepcopy(graph)
    captured = capture_non_epi_forcing(graph)
    channels = dict(decompose_non_epi_forcing(captured))
    assert (
        tuple(sum(row) for row in zip(*channels.values(), strict=True))
        == captured.forcing
    )
    weights = dict(captured.normalized_weights)
    result = derive_phase_capacity_balance(
        captured.snapshot,
        phase_gradient=captured.phase_gradient,
        forcing=captured.forcing,
        phase_weight=weights["phase"],
        capacity_weight=weights["vf"],
        topology_weight=weights["topo"],
    )
    degree_mean = dot(result.support_degrees, captured.snapshot.capacity) / sum(
        result.support_degrees
    )
    assert degree_mean == 2
    assert result.centered_capacity == tuple(
        value - degree_mean for value in captured.snapshot.capacity
    )
    assert result.centered_capacity == (-1, 0, 1, 2)
    assert result.equation_residual == (0, 0, 0, 0)
    assert result.center_residual == result.compatibility_residual == 0
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(graph.edges) == dict(before.edges)
    assert graph.graph == before.graph
    # The captured represented phase coefficients suffice for this arithmetic
    # identity; this is not authentication of an exact analytic phase chart.


@pytest.mark.parametrize(
    "lower,upper,shift,inclusive,admissible",
    (
        (None, None, Q(1, 4), False, True),
        ((1, Q(3, 2)), (1, Q(3, 2)), Q(5, 4), True, True),
        ((0, Q(1, 2)), (0, Q(1, 2)), Q(1, 4), False, False),
        (None, (Q(1, 8), Q(1, 8)), Q(1, 4), False, False),
    ),
)
def test_closed_bands_intersect_strict_positivity_at_the_correct_boundary(
    lower, upper, shift, inclusive, admissible
):
    _, source = _state()
    result = _balance(
        source, forcing=(Q(1, 8), Q(-1, 8)), capacity_lower=lower, capacity_upper=upper
    )
    assert result.source_compatible
    assert result.centered_capacity == (Q(-1, 4), Q(1, 4))
    assert result.capacity_lower == ((0, 0) if lower is None else lower)
    assert result.capacity_upper == upper
    assert result.uniform_shift_lower == shift
    assert result.uniform_shift_lower_inclusive is inclusive
    assert result.has_admissible_capacity is admissible
    assert result.uniform_shift_upper == (
        None
        if upper is None
        else min(
            bound - value
            for bound, value in zip(upper, result.centered_capacity, strict=True)
        )
    )


def test_relabeling_and_node_permutation_preserve_the_ordered_solution():
    graph, source = _state(nx.path_graph(3))
    gradient = (Q(1, 4), 0, Q(-1, 4))
    result = _balance(
        source, phase_gradient=gradient, phase_weight=1, capacity_weight=1
    )
    order = (2, 0, 1)
    labels = {0: "left", 1: ("middle",), 2: 17}
    permuted = nx.Graph()
    permuted.add_nodes_from((labels[i], dict(graph.nodes[i])) for i in order)
    permuted.add_edges_from((labels[i], labels[j]) for i, j in graph.edges)
    other = _balance(
        observe_support_transport(permuted),
        phase_gradient=tuple(gradient[i] for i in order),
        phase_weight=1,
        capacity_weight=1,
    )
    assert other.source.nodes == tuple(labels[i] for i in order)
    assert other.centered_capacity == tuple(result.centered_capacity[i] for i in order)
    assert other.support_degrees == tuple(result.support_degrees[i] for i in order)
    assert other.compatibility_residual == other.center_residual == 0
    assert other.uniform_shift_lower == result.uniform_shift_lower


def test_snapshot_derived_caches_are_rebuilt_and_result_is_immutable():
    _, source = _state(nx.star_graph(3))
    forged = replace(
        source,
        topology_gradient=(999,) * 4,
        capacity_gradient=(17,) * 4,
        rate=(23,) * 4,
        energy_rate=31,
    )
    expected = _balance(source, topology_weight=1)
    actual = _balance(forged, topology_weight=1)
    assert actual == expected
    assert actual.source == source
    with pytest.raises(FrozenInstanceError):
        actual.centered_capacity = (0,) * 4


@pytest.mark.parametrize(
    "graph",
    (
        nx.empty_graph(),
        nx.empty_graph(1),
        nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)),
    ),
)
def test_empty_isolated_and_disconnected_support_is_rejected(graph):
    _, source = _state(graph)
    with pytest.raises(ValueError):
        _balance(source)


def test_a_directed_zero_arc_cannot_bypass_reciprocal_support_validation():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(((0, 1, 0.0), (1, 1, 0.0)))
    _, source = _state(graph)
    assert source.conductance == ()
    assert all(source.support_neighbors)
    with pytest.raises(ValueError, match="reciprocal"):
        _balance(source)


@pytest.mark.parametrize(
    "field,value",
    (
        ("nodes", {0, 1}),
        ("nodes", {0: "a", 1: "b"}),
        ("support_neighbors", {0: (1,), 1: (0,)}),
        ("support_neighbors", ({1}, {0})),
    ),
)
def test_unordered_node_or_support_containers_are_rejected(field, value):
    _, source = _state()
    with pytest.raises(TypeError):
        _balance(replace(source, **{field: value}))


@pytest.mark.parametrize(
    "overrides,error",
    (
        ({"capacity_weight": 0}, ValueError),
        ({"capacity_weight": -1}, ValueError),
        ({"phase_weight": -1}, ValueError),
        ({"topology_weight": -1}, ValueError),
        ({"phase_weight": float("nan")}, ValueError),
        ({"capacity_weight": True}, TypeError),
        ({"phase_gradient": (0,)}, ValueError),
        ({"forcing": {0, 1}}, TypeError),
        ({"phase_gradient": (0, float("inf"))}, ValueError),
        ({"forcing": (0, False)}, TypeError),
        ({"capacity_lower": (Q(-1, 4), 0)}, ValueError),
        ({"capacity_upper": (0, -1)}, ValueError),
        ({"capacity_lower": (2, 2), "capacity_upper": (1, 3)}, ValueError),
        ({"capacity_upper": (1,)}, ValueError),
        ({"capacity_lower": {0, 1}}, TypeError),
        ({"capacity_upper": (1, float("nan"))}, ValueError),
    ),
)
def test_malformed_coefficients_vectors_and_bands_are_rejected(overrides, error):
    _, source = _state()
    with pytest.raises(error):
        _balance(source, **overrides)


def test_capacity_balance_requires_a_typed_snapshot():
    with pytest.raises(TypeError):
        derive_phase_capacity_balance(
            {}, phase_gradient=(0, 0), forcing=(0, 0), phase_weight=1, capacity_weight=1
        )
