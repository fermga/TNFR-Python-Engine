"""Independent instantaneous acceleration controls for weighted sine exchange.

These are complete-field algebra and admission checks. No source trajectory,
sampled history, maintained organization or operator occurrence is produced.
"""

import pickle
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)


def _model(*, beta=2, exchange=Q(1)):
    model = RelationalExchangeModel(beta, epi_weight=0, phase_domain="regular")
    # Re-admission retains supplied effective coefficients, rather than
    # normalizing a newly reconstructed reference model.
    object.__setattr__(model, "phase_weight", exchange)
    return model


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(("a", ("b", 0), 7, "d"))
    nodes = tuple(graph)
    for i, j, weight in (
        (0, 1, Q(1, 2)),
        (1, 2, Q(2)),
        (2, 3, Q(0)),
        (3, 0, Q(3, 4)),
        (0, 2, Q(3, 2)),
    ):
        graph.add_edge(nodes[i], nodes[j], weight=weight, length=17)
    for node, form, phase, capacity in zip(
        nodes,
        (Q(1, 4), Q(-1, 2), Q(3, 4), Q(1, 8)),
        (Q(0), Q(1, 4), Q(-3, 8), Q(1, 2)),
        (Q(1, 2), Q(1), Q(3, 2), Q(2)),
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"list": [1, 2]})
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _independent_acceleration(mp, source):
    positions = {node: i for i, node in enumerate(source.nodes)}
    neighbors = [[] for _ in source.nodes]
    gradient = [Q(0)] * len(source.nodes)
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        neighbors[i].append(j)
        neighbors[j].append(i)
        flow = weight * (source.epi[i] - source.epi[j])
        gradient[i] += flow
        gradient[j] -= flow
    w = _mp(mp, source.reference_model.phase_weight)
    beta = _mp(mp, source.reference_model.storage_scale)
    mobility = tuple(
        w * _mp(mp, capacity) / (mp.pi * len(adjacent))
        for capacity, adjacent in zip(source.capacity, neighbors)
    )
    phase_velocity = tuple(
        mobility[i] * _mp(mp, value) / beta for i, value in enumerate(gradient)
    )
    return tuple(
        mobility[i]
        * sum(
            mp.cos(_mp(mp, source.phase[j] - source.phase[i]))
            * (phase_velocity[j] - phase_velocity[i])
            for j in adjacent
        )
        for i, adjacent in enumerate(neighbors)
    )


def _check_enclosure(mp, bounds, expected):
    for interval, value in zip(bounds, expected):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        assert _mp(mp, interval.width) < mp.mpf("1e-34") * max(1, abs(value))


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("instantaneous acceleration must not advance a trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def sample():
    return bound_relational_sine_support_exchange(_graph(), reference_model=_model())


@pytest.mark.parametrize("beta,exchange", [(2, Q(1)), (4, Q(3, 4)), (Q(1, 2), Q(2))])
def test_full_acceleration_retains_weighted_gradient_and_coefficients(
    sample, mp, beta, exchange
):
    source = replace(sample, reference_model=_model(beta=beta, exchange=exchange))
    actual = source.form_acceleration_bounds()
    expected = _independent_acceleration(mp, source)
    _check_enclosure(mp, actual, expected)
    baseline = _independent_acceleration(mp, sample)
    coefficient = 2 * exchange**2 / beta
    assert max(
        abs(value - _mp(mp, coefficient) * base)
        for value, base in zip(expected, baseline)
    ) < mp.mpf("1e-95")


@pytest.mark.parametrize("factor", [Q(1, 8), Q(3, 2), Q(8), Q(2**20)])
def test_common_capacity_has_quadratic_acceleration_at_the_same_state(
    sample, mp, factor
):
    scaled = replace(sample, capacity=tuple(factor * nu for nu in sample.capacity))
    baseline = _independent_acceleration(mp, sample)
    expected = tuple(_mp(mp, factor**2) * value for value in baseline)
    _check_enclosure(mp, scaled.form_acceleration_bounds(), expected)
    direct = _independent_acceleration(mp, scaled)
    assert max(abs(a - b) for a, b in zip(direct, expected)) < mp.mpf("1e-80")


def test_zero_conductance_edge_remains_in_the_phase_acceleration(mp):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 0
    for node, form in enumerate((0, Q(1, 4), Q(-1, 4))):
        graph.nodes[node].update(EPI=form, theta=0, nu_f=1)
    source = bound_relational_sine_support_exchange(
        graph, reference_model=_model(beta=1)
    )
    bounds = source.form_acceleration_bounds()
    assert source.form_gradient[0] == 0
    assert bounds[0].lo > 0
    assert _mp(mp, bounds[0].lo) <= 1 / (4 * mp.pi**2) <= _mp(mp, bounds[0].hi)
    _check_enclosure(mp, bounds, _independent_acceleration(mp, source))


def test_zero_capacity_target_and_zero_gradient_are_exact(sample, mp):
    inactive = replace(sample, capacity=(Q(0), *sample.capacity[1:]))
    bounds = inactive.form_acceleration_bounds()
    assert bounds[0] == I(0)
    _check_enclosure(mp, bounds, _independent_acceleration(mp, inactive))
    assert (
        replace(sample, capacity=(Q(0),) * 4).form_acceleration_bounds() == (I(0),) * 4
    )
    assert (
        replace(sample, epi=(Q(-7, 2),) * 4).form_acceleration_bounds() == (I(0),) * 4
    )


def test_common_form_and_phase_origin_do_not_change_acceleration(sample):
    shifted = replace(
        sample,
        epi=tuple(value + 10**400 for value in sample.epi),
        phase=tuple(value + 10**400 for value in sample.phase),
    )
    assert shifted.form_acceleration_bounds() == sample.form_acceleration_bounds()


def test_derived_fields_are_rebuilt_before_evaluation(sample):
    forged = replace(
        sample,
        degrees=(999,) * 4,
        form_gradient=(Q(999),) * 4,
        relative_resultant=((I(999), I(999)),) * 4,
        phase_sources=(I(999),) * 4,
        pressure=(I(999),) * 4,
        form_rates=(I(999),) * 4,
        phase_rates=(I(999),) * 4,
        storage=I(999),
    )
    assert forged.form_acceleration_bounds() == sample.form_acceleration_bounds()


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True, Q(0), Q(0), Q(0))},
        {"phase": (False, Q(0), Q(0), Q(0))},
        {"capacity": (Q(-1), Q(1), Q(1), Q(1))},
        {"capacity": (Decimal("0.1"), Q(1), Q(1), Q(1))},
        {"conductances": (Q(-1), Q(0), Q(1), Q(1), Q(1))},
        {"conductances": (True, Q(0), Q(1), Q(1), Q(1))},
        {"epi": (float("nan"), Q(0), Q(0), Q(0))},
        {"phase": (float("inf"), Q(0), Q(0), Q(0))},
        {"edges": (("a", ("b", 0)), (7, "d")), "conductances": (Q(1), Q(1))},
        {"law": "unrelated_law"},
        {"reference_model": RelationalExchangeModel(1, phase_domain="regular")},
        {"reference_model": RelationalExchangeModel(1, epi_weight=0)},
    ],
)
def test_invalid_primitive_state_or_law_rejects_cached_rates(sample, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(sample, **changes).form_acceleration_bounds()


def test_other_comparison_domains_do_not_inherit_the_method():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=Q(node, 8), theta=Q(node, 4), nu_f=1)
    ordinary = bound_relational_sine_exchange(graph, reference_model=_model())
    graph.add_node("isolate", EPI=0, theta=0, nu_f=1)
    frozen = bound_relational_sine_frozen_isolates(graph, reference_model=_model())
    for source in (ordinary, frozen):
        assert not hasattr(source, "form_acceleration_bounds")
        with pytest.raises(TypeError, match="SineSupportExchangeComparison"):
            SineSupportExchangeComparison.form_acceleration_bounds(source)


def test_reader_neither_mutates_source_nor_supplies_temporal_history(mp):
    graph = _graph()
    before = _snapshot(graph)
    source = bound_relational_sine_support_exchange(graph, reference_model=_model())
    bounds = source.form_acceleration_bounds()
    _check_enclosure(mp, bounds, _independent_acceleration(mp, source))
    assert _snapshot(graph) == before
    observation = observe_structural_acceleration(graph, "a")
    assert not observation.available
    assert observation.value is None
    assert _snapshot(graph) == before
