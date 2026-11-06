"""Exact legacy arithmetic and immutable-coefficient controls on fresh fields.

Only unrelated instantaneous states and Taylor jets are evaluated. No
formation, reserved trajectory or archived response producer is invoked.
"""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import sin as jet_sin
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics import _sine_flow
from tnfr.physics.phase_cycle_geometry import _derive
from tnfr.physics.relational_sine_comparison import _sine_rate_evaluator, _sine_rates
from tnfr.physics.relational_sine_two_port_readout import _full_sine_field


def _legacy_rates(model, degrees, gradient, capacity, phase_currents, factors=None):
    """The previous per-call operation order, independent of the new factory."""
    metric = tuple(1 / (pi_interval() * degree) for degree in degrees)
    if factors is not None:
        metric = tuple(factor * value for factor, value in zip(factors, metric))
    sources = tuple(value * mobility for value, mobility in zip(phase_currents, metric))
    loss, exchange = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)
    pressure = tuple(
        -loss * value / degree + exchange * source
        for value, degree, source in zip(gradient, degrees, sources)
    )
    form = tuple(capacity * value for capacity, value in zip(capacity, pressure))
    phase = tuple(
        (exchange / beta) * nu * value * mobility
        for nu, value, mobility in zip(capacity, gradient, metric)
    )
    return dict(
        inverse_phase_metric=metric,
        phase_sources=sources,
        pressure=pressure,
        form_rates=form,
        phase_rates=phase,
    )


def _geometry(size):
    if size == 4:
        edges = ((0, 1), (0, 2), (0, 3), (1, 2))
    else:
        contacts = ((0, 9), (1, 10)) if size == 18 else ((4, 13), (13, 22))
        edges = (
            tuple(
                (9 * part + local, 9 * part + (local + 1) % 9)
                for part in range(size // 9)
                for local in range(9)
            )
            + contacts
        )
        edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    geometry = _derive(tuple(range(size)), edges)
    degrees = tuple(sum(i in edge for edge in edges) for i in range(size))
    return geometry, degrees


def _state(size, order):
    values = tuple(
        I(Q((7 * i + 2) % 19 - 9, 23), Q((7 * i + 2) % 19 - 9, 23) + Q(1, 1000))
        for i in range(2 * size)
    )
    if order is None:
        return values
    return tuple(
        Jet(
            (value,)
            + tuple(
                I(Q((i + degree) % 5 - 2, 100**degree))
                for degree in range(1, order + 1)
            )
        )
        for i, value in enumerate(values)
    )


def _legacy_field(model, geometry, degrees, state):
    size = len(degrees)
    epi, phase = state[:size], state[size:]
    jet = isinstance(state[0], Jet)
    zero = Jet.constant(0, state[0].order) if jet else I(0)
    current = [zero for _ in range(size)]
    sine = jet_sin if jet else sin
    for i, j in geometry.edges:
        flow = sine(phase[j] - phase[i])
        current[i] += flow
        current[j] -= flow
    neighbors = tuple(
        tuple(j if i == node else i for i, j in geometry.edges if node in (i, j))
        for node in range(size)
    )
    gradient = tuple(
        sum((epi[i] - epi[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
    )
    rates = _legacy_rates(model, degrees, gradient, (Q(1),) * size, current)
    return tuple(
        value / Q(model.effective_weights[0])
        for value in rates["form_rates"] + rates["phase_rates"]
    )


@pytest.mark.parametrize("size", (4, 18, 27))
@pytest.mark.parametrize("order", (None, 0, 1, 4, 16))
def test_shared_field_is_exactly_equal_to_prior_arithmetic(size, order):
    geometry, degrees = _geometry(size)
    model = RelationalExchangeModel(
        7 / 5, epi_weight=2 / 7, phase_weight=5 / 7, phase_domain="regular"
    )
    state = _state(size, order)
    flow, domain = _sine_flow._full_sine_field(model, geometry, degrees)
    assert flow(state) == _legacy_field(model, geometry, degrees, state)
    assert domain(state) == (Q(1),)
    assert _full_sine_field is _sine_flow._full_sine_field


@pytest.mark.parametrize("order", (None, 2))
def test_prepared_rates_retain_held_capacities_and_exchange_factors(order):
    model = RelationalExchangeModel(Q(5, 4), phase_domain="regular")
    degrees = [1, 2, 3]
    capacities = [Q(0), Q(2, 3), Q(4)]
    factors = [I(Q(1, 2)), I(Q(2, 3), Q(3, 4)), I(2)]
    state = _state(3, order)
    gradient, currents = state[:3], state[3:]
    expected = _legacy_rates(model, degrees, gradient, capacities, currents, factors)
    prepared = _sine_rate_evaluator(
        model, degrees, capacities, exchange_factors=factors
    )
    assert prepared(gradient, currents) == expected
    assert (
        _sine_rates(
            model, degrees, gradient, capacities, currents, exchange_factors=factors
        )
        == expected
    )
    degrees[1], capacities[1], factors[1] = 9, Q(9), I(9)
    assert prepared(gradient, currents) == expected
    assert (
        _sine_rates(
            model, degrees, gradient, capacities, currents, exchange_factors=factors
        )
        != expected
    )


def test_field_prepares_fixed_rate_coefficients_only_once(monkeypatch):
    calls = []
    original = _sine_flow._sine_rate_evaluator

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(_sine_flow, "_sine_rate_evaluator", counted)
    geometry, degrees = _geometry(4)
    model = RelationalExchangeModel(1, phase_domain="regular")
    flow, _ = _sine_flow._full_sine_field(model, geometry, degrees)
    for order in (None, 1, 4):
        flow(_state(4, order))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="both full nodal rows"):
        flow(_state(4, None)[:-1])
    with pytest.raises(ValueError, match="degrees"):
        _sine_flow._full_sine_field(model, geometry, (2,) * 4)


def test_shared_field_normalizes_degree_iterators_once_and_rejects_equality_aliases():
    geometry, degrees = _geometry(4)
    model = RelationalExchangeModel(1, phase_domain="regular")
    source = _state(4, None)
    flow, _ = _sine_flow._full_sine_field(model, geometry, iter(degrees))
    assert flow(source) == _legacy_field(model, geometry, degrees, source)
    for invalid in (True, 1.0, Q(1), 0, -1):
        with pytest.raises(ValueError, match="positive ordinary integers"):
            _sine_flow._full_sine_field(model, geometry, degrees[:-1] + (invalid,))
    with pytest.raises(ValueError, match="complete"):
        _sine_flow._full_sine_field(model, geometry, iter(degrees[:-1]))
    with pytest.raises(ValueError, match="nonempty"):
        _sine_flow._full_sine_field(model, SimpleNamespace(nodes=(), edges=()), ())


def test_subgrid_clock_divisor_keeps_its_original_field_failure_boundary():
    geometry, degrees = _geometry(4)
    model = RelationalExchangeModel(
        1, epi_weight=Q(1, 2**200), phase_weight=1, phase_domain="regular"
    )
    flow, _ = _sine_flow._full_sine_field(model, geometry, degrees)
    state = _state(4, None)
    # The model has positive represented loss. Its interval reciprocal is
    # unavailable only when the field is evaluated, where the shared solver
    # can retain the failure; coefficient preparation must not move that event.
    with pytest.raises(ZeroDivisionError):
        _legacy_field(model, geometry, degrees, state)
    with pytest.raises(ZeroDivisionError):
        flow(state)
