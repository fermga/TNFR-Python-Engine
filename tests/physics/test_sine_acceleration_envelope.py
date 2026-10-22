"""Independent complete-field controls for conditional sine acceleration.

These static controls admit no temporal history or maintained source. Exact
centered-form balls are supplied hypotheses, not inferred future bounds.
"""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)


def _model(*, beta=2, exchange=Q(1)):
    model = RelationalExchangeModel(beta, epi_weight=0, phase_domain="regular")
    # The owner re-admits currently supplied coefficients without normalizing
    # a reconstructed model. This exercises that explicit existing contract.
    object.__setattr__(model, "phase_weight", exchange)
    return model


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(("a", ("b", 0), 7, "d"))
    nodes = tuple(graph)
    for left, right, weight in ((0, 1, 2), (1, 2, 0), (2, 3, Q(3, 4)), (3, 0, 1)):
        graph.add_edge(nodes[left], nodes[right], weight=weight)
    for node, form, phase, capacity in zip(
        nodes,
        (Q(1, 4), Q(-1, 2), Q(3, 4), Q(1, 8)),
        (Q(0), Q(1, 4), Q(-3, 8), Q(1, 2)),
        (Q(1, 2), Q(1), Q(3, 2), Q(2)),
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    return graph


@pytest.fixture(scope="module")
def sample():
    return bound_relational_sine_support_exchange(_graph(), reference_model=_model())


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _independent_acceleration_and_bound(mp, source, radius):
    positions = {node: index for index, node in enumerate(source.nodes)}
    neighbors = [[] for _ in source.nodes]
    gradient = [Q(0)] * len(source.nodes)
    strength = [Q(0)] * len(source.nodes)
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        neighbors[i].append(j)
        neighbors[j].append(i)
        difference = weight * (source.epi[i] - source.epi[j])
        gradient[i] += difference
        gradient[j] -= difference
        strength[i] += weight
        strength[j] += weight
    w = _mp(mp, source.reference_model.phase_weight)
    beta = _mp(mp, source.reference_model.storage_scale)
    mobility = tuple(
        w * _mp(mp, capacity) / (mp.pi * len(adjacent))
        for capacity, adjacent in zip(source.capacity, neighbors)
    )
    phase_rates = tuple(
        mobility[i] * _mp(mp, value) / beta for i, value in enumerate(gradient)
    )
    edge_radius = mp.sqrt(2) * _mp(mp, radius)
    phase_ceilings = tuple(
        mobility[i] * _mp(mp, value) * edge_radius / beta
        for i, value in enumerate(strength)
    )
    acceleration, bounds = [], []
    for i, adjacent in enumerate(neighbors):
        acceleration.append(
            mobility[i]
            * sum(
                mp.cos(_mp(mp, source.phase[j] - source.phase[i]))
                * (phase_rates[j] - phase_rates[i])
                for j in adjacent
            )
        )
        bounds.append(
            mobility[i] * sum(phase_ceilings[j] + phase_ceilings[i] for j in adjacent)
        )
    return acceleration, bounds


@pytest.mark.parametrize("beta,exchange", [(2, Q(1)), (4, Q(3, 4)), (Q(1, 2), Q(2))])
def test_complete_field_and_envelope_retain_actual_coefficients(
    sample, mp, beta, exchange
):
    source = replace(sample, reference_model=_model(beta=beta, exchange=exchange))
    radius = Q(2)
    result = source.conditional_form_acceleration_bounds(form_radius=radius)
    accelerations, expected = _independent_acceleration_and_bound(mp, source, radius)
    for bounds, actual, ceiling in zip(result, accelerations, expected):
        assert bounds.lo == -bounds.hi
        assert _mp(mp, bounds.lo) <= actual <= _mp(mp, bounds.hi)
        assert ceiling <= _mp(mp, bounds.hi)
        assert _mp(mp, bounds.hi) - ceiling < mp.mpf("1e-34")


def test_zero_conductance_neighbor_still_transmits_phase_acceleration(mp):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 0
    for node, form in enumerate((0, Q(1, 4), Q(-1, 4))):
        graph.nodes[node].update(EPI=form, theta=0, nu_f=1)
    source = bound_relational_sine_support_exchange(
        graph, reference_model=_model(beta=1)
    )
    result = source.conditional_form_acceleration_bounds(form_radius=Q(1, 2))
    actual, expected = _independent_acceleration_and_bound(mp, source, Q(1, 2))
    assert source.form_gradient[0] == 0
    assert abs(actual[0] - 1 / (4 * mp.pi**2)) < mp.mpf("1e-95")
    assert 0 < actual[0] < expected[0] <= _mp(mp, result[0].hi)


def test_zero_capacity_and_uniform_zero_radius_remain_exact(sample):
    inactive = replace(sample, capacity=(Q(0), *sample.capacity[1:]))
    assert inactive.conditional_form_acceleration_bounds(form_radius=2)[0] == I(0)
    uniform = replace(sample, epi=(Q(-7, 2),) * len(sample.nodes))
    assert uniform.conditional_form_acceleration_bounds(form_radius=0) == (I(0),) * 4
    all_inactive = replace(sample, capacity=(Q(0),) * 4)
    assert (
        all_inactive.conditional_form_acceleration_bounds(form_radius=2) == (I(0),) * 4
    )


def test_exact_boundary_and_common_form_translation(sample):
    boundary = replace(sample, epi=(Q(1, 2), Q(1, 2), Q(-1, 2), Q(-1, 2)))
    expected = boundary.conditional_form_acceleration_bounds(form_radius=1)
    shifted = replace(boundary, epi=tuple(value + 10**400 for value in boundary.epi))
    assert shifted.conditional_form_acceleration_bounds(form_radius=1) == expected
    with pytest.raises(ValueError, match="centered form exceeds"):
        boundary.conditional_form_acceleration_bounds(form_radius=Q(1) - Q(1, 2**60))


@pytest.mark.parametrize(
    "radius",
    [True, False, -1, Q(-1, 100), float("nan"), float("inf"), "1", 1j, Decimal("0.1")],
)
def test_invalid_radius_rejects_without_a_zero_fallback(sample, radius):
    with pytest.raises((TypeError, ValueError)):
        sample.conditional_form_acceleration_bounds(form_radius=radius)


def test_exact_small_radius_retains_a_nonzero_outward_bound(sample):
    uniform = replace(sample, epi=(Q(0),) * 4)
    result = uniform.conditional_form_acceleration_bounds(form_radius=Q(1, 10**400))
    assert all(bounds.lo < 0 < bounds.hi for bounds in result)


def test_cached_fields_cannot_change_an_envelope(sample):
    forged = replace(
        sample,
        degrees=(999,) * 4,
        form_gradient=(Q(999),) * 4,
        form_rates=(I(999),) * 4,
        phase_rates=(I(999),) * 4,
        storage=I(999),
    )
    assert forged.conditional_form_acceleration_bounds(
        form_radius=2
    ) == sample.conditional_form_acceleration_bounds(form_radius=2)


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True, Q(0), Q(0), Q(0))},
        {"phase": (False, Q(0), Q(0), Q(0))},
        {"capacity": (Q(-1), Q(1), Q(1), Q(1))},
        {"conductances": (Q(-1), Q(0), Q(1), Q(1))},
        {"edges": (("a", ("b", 0)), (7, "d")), "conductances": (Q(1), Q(1))},
        {"law": "unrelated_law"},
        {"reference_model": RelationalExchangeModel(1, phase_domain="regular")},
    ],
)
def test_invalid_source_primitives_cannot_use_cached_envelopes(sample, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(sample, **changes).conditional_form_acceleration_bounds(form_radius=2)


def test_frozen_isolate_report_does_not_inherit_connected_method():
    graph = _graph()
    graph.add_node("isolate", EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_frozen_isolates(graph, reference_model=_model())
    assert not hasattr(source, "conditional_form_acceleration_bounds")
    with pytest.raises(TypeError, match="SineSupportExchangeComparison"):
        SineSupportExchangeComparison.conditional_form_acceleration_bounds(
            source, form_radius=2
        )


def test_large_form_radius_does_not_create_a_universal_thol_obstruction(mp):
    graph = nx.path_graph(2)
    graph.nodes[0].update(EPI=3, theta=0, nu_f=1)
    graph.nodes[1].update(EPI=1, theta=Q(1, 3), nu_f=1)
    source = bound_relational_sine_support_exchange(
        graph, reference_model=_model(beta=1)
    )
    actual, _ = _independent_acceleration_and_bound(mp, source, Q(2))
    result = source.conditional_form_acceleration_bounds(form_radius=2)
    assert abs(actual[0]) > mp.mpf("0.1")
    assert _mp(mp, result[0].lo) <= actual[0] <= _mp(mp, result[0].hi)


def test_twenty_node_unit_comparison_has_the_declared_parent_envelope(mp):
    # This unit-weight comparison graph bounds every admitted <=1 weighted
    # member of the contact support. It is not regenerated UM output or a
    # prepared research source; uniform form only admits the supplied ball.
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    for offset in (0, 10):
        for pair in range(5):
            following = (pair + 1) % 5
            for left in (offset + 2 * pair, offset + 2 * pair + 1):
                for right in (offset + 2 * following, offset + 2 * following + 1):
                    graph.add_edge(left, right, weight=1)
    assert graph.number_of_edges() == 40
    graph.add_edge(0, 1, weight=1)
    graph.add_edges_from(
        (left, right, {"weight": 1}) for left in (0, 1) for right in (10, 11)
    )
    assert graph.number_of_edges() == 45
    for node in graph:
        graph.nodes[node].update(
            EPI=Q(5, 16),
            theta=Q(node, 128),
            nu_f=Q(1, 2) if node in (0, 1) else Q(1),
        )
    assert tuple(graph.degree(node) for node in (0, 1, 10, 11)) == (7, 7, 6, 6)
    assert all(graph.degree(node) == 4 for node in set(graph) - {0, 1, 10, 11})
    assert set(graph[0]) == {1, 2, 3, 8, 9, 10, 11}
    source = bound_relational_sine_support_exchange(
        graph, reference_model=_model(beta=1)
    )
    result = source.conditional_form_acceleration_bounds(form_radius=Q(1, 16))
    # Parent mobility factor 1/14 multiplies six unit-capacity neighbors,
    # one half-capacity twin, and seven copies of its own half capacity.
    coefficient = Q(1, 14) * (6 + Q(1, 2) + 7 * Q(1, 2))
    assert coefficient == Q(5, 7)
    expected = mp.sqrt(2) * _mp(mp, Q(1, 16) * coefficient) / mp.pi**2
    assert expected <= _mp(mp, result[0].hi)
    assert _mp(mp, result[0].hi) - expected < mp.mpf("1e-34")
    assert result[0] == result[1]
    assert result[0].hi < Q(1, 128)
    assert all(bounds.hi < Q(1, 48) for bounds in result)
