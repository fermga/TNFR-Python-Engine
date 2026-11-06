"""Independent static controls for conditional child-asymmetry forgetting.

The prescribed preparations are never evolved. Derivatives use the complete
fine sine law, and the deadline is a proved allowance, not a sampled response.
"""

import math
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_child_relaxation as owner

EDGES = tuple(
    (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
) + tuple(
    (9 * c + j, 9 * d + j)
    for c, d in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for j in range(9)
)
NEIGHBORS = tuple(
    tuple(b if a == i else a for a, b in EDGES if i in (a, b)) for i in range(45)
)
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))
ZERO = (Q(0),) * 45
UNIT = (Q(1),) * 45
H = Q(1, 128)


def _state(a=None, b=None, capacities=UNIT):
    a = (H,) + (Q(0),) * 8 if a is None else a
    b = (Q(0), H) + (Q(0),) * 7 if b is None else b
    form, offsets = list(ZERO), list(ZERO)
    for j in range(9):
        form[9 + j], form[18 + j] = a[j], -a[j]
        offsets[9 + j], offsets[18 + j] = b[j], -b[j]
    return tuple(form), tuple(offsets), tuple(capacities)


def _bound(state=None, elapsed_time=Q(0), **overrides):
    form, offsets, capacities = _state() if state is None else state
    return owner._bound_organized_source_child_relaxation(
        **{
            "form": form,
            "phase_offsets": offsets,
            "capacities": capacities,
            "elapsed_time": elapsed_time,
            **overrides,
        }
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child relaxation controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(context, value):
    if isinstance(value, (Q, int)):
        return context.mpf(value.numerator) / int(value.denominator)
    return value


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _hidden(row):
    return tuple((row[9 + j] - row[18 + j]) / 2 for j in range(9))


def _fields(context, form, offsets, capacities):
    """Build both rows and storage from independent fine edge incidence."""
    gamma, alpha = 1 / (1023 * context.pi), 2 * context.pi / 9
    q, sine = [context.mpf(0)] * 45, [context.mpf(0)] * 45
    energy = context.mpf(0)
    for a, b in EDGES:
        difference = _mp(context, form[b]) - _mp(context, form[a])
        gap = (
            alpha * (b % 9 - a % 9)
            + _mp(context, offsets[b])
            - _mp(context, offsets[a])
        )
        q[a] -= difference
        q[b] += difference
        sine[a] += context.sin(gap)
        sine[b] -= context.sin(gap)
        energy += difference**2 / 2 + 1 - context.cos(gap)
    dx = tuple(
        _mp(context, nu) * (-gradient + gamma * current) / 4
        for nu, gradient, current in zip(capacities, q, sine)
    )
    dtheta = tuple(
        gamma * _mp(context, nu) * gradient / 4 for nu, gradient in zip(capacities, q)
    )
    return dx, dtheta, energy - 45 * (1 - context.cos(alpha))


def _metric(context, state):
    form, offsets, capacities = state
    a, b = _hidden(form), _hidden(offsets)
    gamma = 1 / (1023 * context.pi)
    return context.fsum(
        (_mp(context, x) ** 2 + _mp(context, y) ** 2 + 2 * gamma * _mp(context, x * y))
        / _mp(context, capacities[9 + j])
        for j, (x, y) in enumerate(zip(a, b))
    )


def _derivatives(context, state):
    form, offsets, capacities = state
    a, b = tuple(map(lambda x: _mp(context, x), _hidden(form))), tuple(
        map(lambda x: _mp(context, x), _hidden(offsets))
    )
    dx, dtheta, energy = _fields(context, *state)
    da, db = _hidden(dx), _hidden(dtheta)
    gamma = 1 / (1023 * context.pi)
    euclidean = 2 * context.fsum(x * u + y * v for x, y, u, v in zip(a, b, da, db))
    metric = 2 * context.fsum(
        (x * u + y * v + gamma * (u * y + x * v)) / _mp(context, capacities[9 + j])
        for j, (x, y, u, v) in enumerate(zip(a, b, da, db))
    )
    return metric, euclidean, energy


def _ideal_word(context, state):
    form, offsets, capacities = state
    factors = canonical_glyph_factor_defaults()
    beta = _mp(context, Q(factors["EN_mix"]))
    lam = _mp(context, Q(DEFAULT_PHASE_LOCKING_COEFFICIENT))
    kappa = Q(factors["SHA_vf_factor"])
    x, phase = tuple(_mp(context, x) for x in form), tuple(
        _mp(context, x) for x in offsets
    )
    result_x, result_phase, result_nu = list(x), list(phase), list(capacities)
    alpha = 2 * context.pi / 9
    for target in (0, 1):
        peers = NEIGHBORS[target]
        result_x[target] = (1 - beta) * x[target] + beta * context.fsum(
            x[j] for j in peers
        ) / 5
        resultant = context.fsum(
            context.exp(
                context.j * (alpha * (j % 9 - target) + phase[j] - phase[target])
            )
            for j in peers
        )
        assert resultant.real > 4
        result_phase[target] += lam * context.arg(resultant)
        result_nu[target] *= kappa
    return tuple(result_x), tuple(result_phase), tuple(result_nu)


@pytest.fixture(scope="module")
def initial():
    return _bound()


@pytest.fixture(scope="module")
def deadline(initial):
    return _bound(elapsed_time=10 / initial.decay_rate_lower)


def test_frozen_deadline_is_a_certified_allowance_below_the_fixed_tolerance(
    initial, deadline, mp
):
    gamma = 1 / (1023 * mp.pi)
    exact_metric = 2 * _mp(mp, H) ** 2
    assert initial.certified and deadline.certified
    assert initial.initial_hidden_norm_squared == 2 * H**2
    assert initial.initial_weighted_square == 2 * H**2
    assert initial.initial_weighted_cross == 0
    assert initial.initial_metric_bounds == (2 * H**2, 2 * H**2)
    assert initial.initial_metric_upper == 2 * H**2
    assert deadline.exponent == 10
    assert deadline.elapsed_time == 10 / initial.decay_rate_lower
    rate = initial.gamma_bounds.lo**2 / (4 * (1 + initial.gamma_bounds.hi))
    assert initial.decay_rate_lower == rate
    assert 0 < _mp(mp, rate) <= gamma**2 / (4 * (1 + gamma))
    assert _mp(mp, deadline.decay_bounds[0]) <= mp.exp(-10)
    assert mp.exp(-10) <= _mp(mp, deadline.decay_bounds[1])
    assert exact_metric * mp.exp(-10) <= _mp(mp, deadline.metric_upper_bound)
    assert exact_metric * mp.exp(-10) / (1 - gamma) <= _mp(
        mp, deadline.hidden_norm_squared_upper_bound
    )
    assert 0 < deadline.hidden_norm_squared_upper_bound < H**2 / 256
    assert deadline.projected_phase_defect_upper_bound == 0
    assert 0 < deadline.projected_form_defect_upper_bound
    assert 0 < deadline.coherence_argument_defect_upper_bound


def test_energy_collar_and_both_full_rows_are_admitted_independently(initial, mp):
    dx, phase_rate, excess = _fields(mp, *_state())
    expected = 4 * _mp(mp, H) ** 2 + 4 * (1 + mp.cos(2 * mp.pi / 9)) * (
        1 - mp.cos(_mp(mp, H))
    )
    assert mp.almosteq(excess, expected, abs_eps=mp.mpf("1e-95"))
    assert 0 < excess < mp.mpf(".01")
    _contains(mp, initial.excess_bounds, excess)
    assert initial.geometry_certified and initial.in_acute_cell
    assert initial.below_energy_ceiling and initial.domain_certified
    assert initial.boundary_ceiling_margin > 0 and initial.collar_margin > 0
    assert initial.cosine_lower_bound == Q(1, 2)
    assert initial.hidden_gap_lower_bound == 2
    assert mp.cos(_mp(mp, initial.collar_angle_upper)) > mp.mpf(".5")
    for expected_row, actual_row in (
        (dx, initial.orbit.fine.form_tau_rate_bounds),
        (phase_rate, initial.orbit.fine.phase_tau_rate_bounds),
    ):
        for expected_value, actual_value in zip(expected_row, actual_row):
            _contains(mp, actual_value, expected_value)
    assert any(value != 0 for value in phase_rate)


def test_frozen_euclidean_growth_does_not_defeat_common_metric_decay(mp):
    a = (H / 65536,) + (Q(0),) * 8
    b = (H,) + (Q(0),) * 8
    state = _state(a, b)
    report = _bound(state)
    derivative, euclidean, energy = _derivatives(mp, state)
    gamma, alpha, h, small = (
        1 / (1023 * mp.pi),
        2 * mp.pi / 9,
        _mp(mp, H),
        _mp(mp, H / 65536),
    )
    k00 = 2 * (1 + mp.cos(alpha)) * mp.sin(h) / h
    independent_euclidean = -2 * small**2 + gamma * small * h * (4 - k00) / 2
    independent_metric = (
        -2 * (1 - gamma**2) * small**2
        - gamma * small * h * k00 / 2
        - gamma**2 * h**2 * k00 / 2
    )
    assert gamma > mp.mpf(1) / 4096
    assert mp.cos(alpha) < mp.mpf(4) / 5 and mp.sin(h) / h < 1
    assert mp.almosteq(euclidean, independent_euclidean)
    assert mp.almosteq(derivative, independent_metric)
    assert euclidean > 0 > derivative
    assert 0 < energy < mp.mpf(".01") and report.certified
    _contains(mp, report.initial_euclidean_squared_derivative_bounds, euclidean)
    _contains(mp, report.initial_metric_derivative_bounds, derivative)
    assert report.initial_euclidean_squared_derivative_bounds.lo > 0
    assert report.initial_metric_derivative_bounds.hi < 0
    assert derivative <= -_mp(mp, report.decay_rate_lower) * _metric(mp, state)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_weighted_metric_derivative_uses_general_paired_capacities(heterogeneous, mp):
    capacities = list(UNIT)
    if heterogeneous:
        capacities = [Q(3 + i % 7, 4) for i in range(45)]
        for j in range(9):
            capacities[9 + j] = capacities[18 + j] = Q(j + 1, 3)
    a = tuple(Q((3 * j) % 7 - 3, 4096) for j in range(9))
    b = tuple(Q((5 * j) % 11 - 5, 4096) for j in range(9))
    state = _state(a, b, capacities)
    report = _bound(state)
    weighted_square = sum(
        ((x**2 + y**2) / capacities[9 + j] for j, (x, y) in enumerate(zip(a, b))),
        Q(0),
    )
    weighted_cross = sum(
        (x * y / capacities[9 + j] for j, (x, y) in enumerate(zip(a, b))), Q(0)
    )
    assert report.certified
    assert report.initial_weighted_square == weighted_square
    assert report.initial_weighted_cross == weighted_cross
    assert report.paired_capacity_lower == min(capacities[9:18])
    assert report.paired_capacity_upper == max(capacities[9:18])
    assert report.capacity_upper == max(capacities)
    exact_metric = _metric(mp, state)
    assert _mp(mp, report.initial_metric_bounds[0]) <= exact_metric
    assert exact_metric <= _mp(mp, report.initial_metric_bounds[1])
    derivative, euclidean, _ = _derivatives(mp, state)
    _contains(mp, report.initial_metric_derivative_bounds, derivative)
    _contains(mp, report.initial_euclidean_squared_derivative_bounds, euclidean)
    assert derivative <= -_mp(mp, report.decay_rate_lower) * exact_metric < 0
    assert report.decay_rate_lower == min(
        capacities[9:18]
    ) * report.gamma_bounds.lo**2 / (4 * (1 + report.gamma_bounds.hi))


def test_instantaneous_projected_law_and_argument_defects_obey_allowances(mp):
    state = _state()
    form, offsets, capacities = map(list, state)
    # This shared offset makes the target's even phase defect nonzero.
    offsets[10] += H
    offsets[19] += H
    state = tuple(form), tuple(offsets), tuple(capacities)
    report = _bound(state)
    projected = tuple(
        tuple((row[i] + row[SWAP[i]]) / 2 for i in range(45)) for row in state[:2]
    )
    fine_rows = _fields(mp, *state)[:2]
    core_rows = _fields(mp, *projected, state[2])[:2]
    for fine, core, bound in zip(
        fine_rows,
        core_rows,
        (
            report.projected_form_defect_upper_bound,
            report.projected_phase_defect_upper_bound,
        ),
    ):
        difference = max(
            abs((fine[i] + fine[SWAP[i]]) / 2 - core[i]) for i in range(45)
        )
        assert difference <= _mp(mp, bound) + mp.mpf("1e-95")
    after_fine = _ideal_word(mp, state)
    after_core = _ideal_word(mp, (*projected, state[2]))
    lam = _mp(mp, Q(DEFAULT_PHASE_LOCKING_COEFFICIENT))
    argument_defect = max(
        abs(after_fine[1][i] - after_core[1][i]) / lam for i in (0, 1)
    )
    assert 0 < argument_defect <= _mp(mp, report.coherence_argument_defect_upper_bound)


def test_marked_ideal_word_preserves_hidden_metric_and_paired_capacities(mp):
    form, offsets, capacities = _state()
    form, offsets = list(form), list(offsets)
    form[0], form[1] = H, -H
    offsets[9] += H
    offsets[18] += H
    state = tuple(form), tuple(offsets), capacities
    report = _bound(state)
    after = _ideal_word(mp, state)
    assert report.certified and abs(sum(form, Q(0)) / 45) < Q(1, 30)
    assert any(after[0][i] != _mp(mp, form[i]) for i in (0, 1))
    assert any(after[1][i] != _mp(mp, offsets[i]) for i in (0, 1))
    assert _hidden(after[0]) == tuple(_mp(mp, x) for x in _hidden(form))
    assert _hidden(after[1]) == tuple(_mp(mp, x) for x in _hidden(offsets))
    assert after[2][9:27] == UNIT[9:27]
    assert _metric(mp, after) == _metric(mp, state)
    assert _fields(mp, *after)[2] <= _fields(mp, *state)[2]
    assert (
        after[2][0]
        == after[2][1]
        == Q(canonical_glyph_factor_defaults()["SHA_vf_factor"])
    )


def test_exact_symmetry_and_tiny_rational_data_are_not_rounded_to_one_case(initial):
    zero = _bound((ZERO, ZERO, UNIT), elapsed_time=10 / initial.decay_rate_lower)
    scale = Q(1, 2**600)
    state = _state()
    tiny = _bound(
        (tuple(scale * x for x in state[0]), tuple(scale * x for x in state[1]), UNIT),
        elapsed_time=10 / initial.decay_rate_lower,
    )
    regular = _bound(elapsed_time=10 / initial.decay_rate_lower)
    assert zero.certified and tiny.certified
    assert zero.initial_hidden_norm_squared == zero.hidden_norm_squared_upper_bound == 0
    assert (
        zero.projected_form_defect_upper_bound
        == zero.coherence_argument_defect_upper_bound
        == 0
    )
    assert tiny.initial_hidden_norm_squared == 2 * (scale * H) ** 2 > 0
    assert tiny.initial_metric_upper == scale**2 * regular.initial_metric_upper
    assert tiny.metric_upper_bound == scale**2 * regular.metric_upper_bound > 0
    assert (
        tiny.hidden_norm_squared_upper_bound
        == scale**2 * regular.hidden_norm_squared_upper_bound
    )
    assert (
        tiny.projected_form_defect_upper_bound
        == scale**2 * regular.projected_form_defect_upper_bound
    )
    assert (
        tiny.coherence_argument_defect_upper_bound
        == scale**2 * regular.coherence_argument_defect_upper_bound
    )


def test_global_exchange_and_common_raw_phase_shift_preserve_certificate(initial):
    state = _state()
    exchanged = _bound(tuple(tuple(row[j] for j in SWAP) for row in state))
    shifted = _bound(phase_offsets=tuple(x + Q(5, 7) for x in state[1]))
    for report in (exchanged, shifted):
        assert report.certified
        assert report.initial_hidden_norm_squared == initial.initial_hidden_norm_squared
        assert report.initial_metric_bounds == initial.initial_metric_bounds
        assert report.decay_rate_lower == initial.decay_rate_lower
        assert (
            report.hidden_norm_squared_upper_bound
            == initial.hidden_norm_squared_upper_bound
        )
    assert shifted.phase_offsets == tuple(x + Q(5, 7) for x in state[1])


def test_low_physical_energy_does_not_admit_a_wrong_raw_phase_lift():
    offsets = list(_state()[1])
    offsets[9] += Q(2 * math.pi)
    report = _bound(phase_offsets=tuple(offsets))
    assert report.excess_bounds.hi < Q(1, 100)
    assert not report.in_acute_cell and not report.domain_certified
    assert not report.certified and report.unavailable_reasons
    assert report.decay_bounds is report.metric_upper_bound is None
    assert report.hidden_norm_squared_upper_bound is None


def test_acute_energy_boundary_is_unavailable_without_destroying_static_evidence():
    state = _state((Q(1, 20),) + (Q(0),) * 8, (Q(0),) * 9)
    report = _bound(state)
    assert report.in_acute_cell and not report.below_energy_ceiling
    assert not report.certified and report.unavailable_reasons
    assert report.hidden_norm_squared_upper_bound is None
    assert report.initial_hidden_norm_squared == Q(1, 400)
    assert report.initial_metric_derivative_bounds.hi < 0


def test_unsupported_exponential_work_is_unavailable_without_extrapolation(initial):
    report = _bound(elapsed_time=Q(4097) / initial.decay_rate_lower)
    assert report.domain_certified and report.exponent == 4097
    assert not report.exponential_work_admitted and not report.certified
    assert report.decay_bounds is report.metric_upper_bound is None
    assert report.hidden_norm_squared_upper_bound is None
    assert report.projected_form_defect_upper_bound is None
    assert report.coherence_argument_defect_upper_bound is None
    assert report.unavailable_reasons


@pytest.mark.parametrize("value", [True, math.nan, math.inf, Q(-1, 100)])
def test_invalid_elapsed_time_rejects_before_geometry(value, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid elapsed time reached a geometric or sine observer")

    monkeypatch.setattr(owner, "_encode_organized_source_child_orbit", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _bound(elapsed_time=value)


@pytest.mark.parametrize(
    "field,value", [("form", True), ("phase_offsets", math.nan), ("capacities", 0)]
)
def test_invalid_physical_primitives_reject(field, value):
    row = list(dict(zip(("form", "phase_offsets", "capacities"), _state()))[field])
    row[0] = value
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: row})


def test_tiny_paired_capacity_asymmetry_is_not_silently_averaged():
    capacities = list(UNIT)
    capacities[9] += Q(1, 2**600)
    with pytest.raises(ValueError, match="equal"):
        _bound(capacities=capacities)
