"""Held interval coefficients, directed multiplicities and full-state jets.

Only independent instantaneous fields and formal derivatives are evaluated;
no reserved successor trajectory or historical producer is executed.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import sin as jet_sin
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics import _sine_flow

MODEL = RelationalExchangeModel(2, epi_weight=0, phase_weight=1, phase_domain="regular")
DEGREES = (2, 2)
CAPACITY = (I(Q(1, 2), Q(3, 4)), I(2, Q(5, 2)))
WEIGHT = I(1, Q(3, 2))
ROWS = (((1, 2, WEIGHT * 2),), ((0, 1, WEIGHT), (1, 1, I(5, 6))))
STATE = (Q(1, 4), Q(-3, 8), Q(-1, 8), Q(3, 16))
DIRECTION = (Q(2, 7), Q(-1, 9), Q(1, 5), Q(-2, 9))


def _number(value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    assert _number(interval.lo) <= value <= _number(interval.hi)


def test_interval_coefficients_enclose_independent_corner_laws_and_jet_derivatives():
    flow = _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, CAPACITY, ROWS)
    ordinary = flow(STATE)
    jets = flow(
        tuple(Jet((value, direction, 0)) for value, direction in zip(STATE, DIRECTION))
    )
    assert tuple(value.coeffs[0] for value in jets) == ordinary
    with mp.workdps(90):
        x0, x1, theta0, theta1 = map(_number, STATE)
        dx0, dx1, dtheta0, dtheta1 = map(_number, DIRECTION)
        angle, speed = theta1 - theta0, dtheta1 - dtheta0
        for weight, nu0, nu1 in product(
            (WEIGHT.lo, WEIGHT.hi),
            (CAPACITY[0].lo, CAPACITY[0].hi),
            (CAPACITY[1].lo, CAPACITY[1].hi),
        ):
            weight, nu0, nu1 = map(_number, (weight, nu0, nu1))
            # Independent fine equations for one root and two equal leaves.
            # Root sees two neighbors; each leaf also sees its identical peer.
            for i, (nu, count, orientation) in enumerate(((nu0, 2, 1), (nu1, 1, -1))):
                scale = nu * count * orientation / (2 * mp.pi)
                expected = (
                    scale * mp.sin(angle),
                    scale * mp.cos(angle) * speed,
                    -scale * mp.sin(angle) * speed**2 / 2,
                )
                for interval, value in zip(jets[i].coeffs, expected):
                    _contains(interval, value)
                for interval, value in zip(
                    jets[2 + i].coeffs,
                    (
                        scale * weight * (x0 - x1) / 2,
                        scale * weight * (dx0 - dx1) / 2,
                        mp.mpf(0),
                    ),
                ):
                    _contains(interval, value)


@pytest.mark.parametrize("order", (None, 2))
@pytest.mark.parametrize("reuse", (False, True))
def test_pair_cache_preserves_directed_counts_and_has_one_evaluation_per_pair(
    monkeypatch, order, reuse
):
    calls = []
    name = "sin" if order is None else "jet_sin"
    original = getattr(_sine_flow, name)

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(_sine_flow, name, counted)
    flow = _sine_flow._held_sine_neighbor_field(
        MODEL, DEGREES, CAPACITY, ROWS, reuse_pair_sines=reuse
    )
    state = (
        STATE if order is None else tuple(Jet.constant(value, order) for value in STATE)
    )
    flow(state)
    assert len(calls) == (1 if reuse else 2)


@pytest.mark.parametrize("order", (None, 0, 3))
def test_legacy_directed_policy_keeps_exact_rational_arithmetic(order):
    capacities = (Q(1, 2), Q(5, 3))
    rows = (((1, 2, Q(7, 3)),), ((0, 1, Q(7, 6)), (1, 1, Q(5))))
    model = RelationalExchangeModel(
        Q(3, 2), epi_weight=2, phase_weight=5, phase_domain="regular"
    )
    state = tuple(map(I, STATE))
    if order is not None:
        state = tuple(Jet((value,) + (I(Q(1, 17)),) * order) for value in state)
    sine = sin if order is None else jet_sin
    zero = I(0) if order is None else Jet.constant(0, order)
    gradient = tuple(
        sum((weight * (state[i] - state[j]) for j, _, weight in row if j != i), zero)
        for i, row in enumerate(rows)
    )
    currents = tuple(
        sum(
            (
                count * sine(state[2 + j] - state[2 + i])
                for j, count, _ in row
                if j != i
            ),
            zero,
        )
        for i, row in enumerate(rows)
    )
    e, w = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)
    metric = tuple(1 / (pi_interval() * degree) for degree in DEGREES)
    expected = tuple(
        nu * (-e * q / degree + w * (current * mobility))
        for nu, q, degree, current, mobility in zip(
            capacities, gradient, DEGREES, currents, metric
        )
    ) + tuple(
        (w / beta) * nu * q * mobility
        for nu, q, mobility in zip(capacities, gradient, metric)
    )
    flow = _sine_flow._held_sine_neighbor_field(
        model, DEGREES, capacities, rows, reuse_pair_sines=False
    )
    assert flow(state) == expected


def test_held_inputs_are_snapshotted_and_internal_neighbors_vanish_exactly():
    degrees, capacity = list(DEGREES), list(CAPACITY)
    rows = [[list(entry) for entry in row] for row in ROWS]
    flow = _sine_flow._held_sine_neighbor_field(MODEL, degrees, capacity, rows)
    expected = flow(STATE)
    degrees[0], capacity[0], rows[0][0][2] = 99, I(99), I(99)
    assert flow(STATE) == expected
    internal = _sine_flow._held_sine_neighbor_field(
        MODEL, (2,), (I(1, 2),), (((0, 2, I(5, 6)),),)
    )
    assert internal((I(-1, 1), I(-1, 1))) == (I(0), I(0))


def test_zero_conductance_retains_phase_current_and_zero_capacity_freezes_rows():
    rows = (((1, 1, 0),), ((0, 1, 0),))
    flow = _sine_flow._held_sine_neighbor_field(MODEL, (1, 1), (1, 0), rows)
    rates = flow(STATE)
    assert rates[0].lo > 0
    assert rates[1:] == (I(0), I(0), I(0))


@pytest.mark.parametrize("bad", (True, -1, I(-1, 0), 0.5, float("nan"), float("inf")))
@pytest.mark.parametrize("target", ("capacity", "weight"))
def test_invalid_coefficients_are_rejected_before_evaluation(bad, target):
    capacity = (bad, 1) if target == "capacity" else (1, 1)
    rows = (((1, 2, bad if target == "weight" else 1),), ((0, 2, 1),))
    with pytest.raises((TypeError, ValueError)):
        _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, capacity, rows)


@pytest.mark.parametrize(
    "rows",
    (
        (((1, True, 1),), ((0, 2, 1),)),
        (((1, 0, 1),), ((0, 2, 1),)),
        (((True, 2, 1),), ((0, 2, 1),)),
        (((2, 2, 1),), ((0, 2, 1),)),
        (((1, 1, 1), (1, 1, 1)), ((0, 2, 1),)),
        (((1, 1, 1),), ((0, 2, 1),)),
        (((1, 2),), ((0, 2, 1),)),
    ),
)
def test_invalid_neighbor_rows_cannot_hide_in_degree_or_index_equality(rows):
    with pytest.raises((TypeError, ValueError)):
        _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, (1, 1), rows)


def test_shape_scalar_and_jet_domain_admission():
    for degrees in ((), (True, 2), (2.0, 2), (0, 2)):
        with pytest.raises(ValueError, match="degrees"):
            _sine_flow._held_sine_neighbor_field(MODEL, degrees, CAPACITY, ROWS)
    with pytest.raises(ValueError, match="capacity"):
        _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, CAPACITY[:1], ROWS)
    with pytest.raises(ValueError, match="nodal row"):
        _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, CAPACITY, ROWS[:1])
    with pytest.raises(TypeError, match="Boolean"):
        _sine_flow._held_sine_neighbor_field(
            MODEL, DEGREES, CAPACITY, ROWS, reuse_pair_sines=1
        )
    flow = _sine_flow._held_sine_neighbor_field(MODEL, DEGREES, CAPACITY, ROWS)
    for state in (STATE[:-1], STATE + (0,)):
        with pytest.raises(ValueError, match="every reduced/full"):
            flow(state)
    for value in (True, 0.0, float("nan")):
        with pytest.raises(TypeError):
            flow((value,) + STATE[1:])
    for state in (
        (Jet.constant(0, 1),) + STATE[1:],
        STATE[:-1] + (Jet.constant(0, 1),),
        (Jet.constant(0, 1),) * 3 + (Jet.constant(0, 2),),
    ):
        with pytest.raises(ValueError, match="[Jj]et"):
            flow(state)
