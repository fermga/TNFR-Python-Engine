"""Independent history-to-contact controls for actual coupled sine dynamics.

Closed modes evaluate the full piecewise tangent, never nonlinear trajectories.
Whole-channel hybrid witnesses concern exact reached projections; numerical
outer coordinate intervals are never identified with those projections.
"""

from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product, repeat
from math import factorial, prod

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as flow
from tnfr.physics import _sine_history_information as owner
from tnfr.physics import _sine_joint_information as direct


@dataclass(frozen=True)
class _B:
    lo: Q
    hi: Q | None = None

    def __post_init__(self):
        object.__setattr__(self, "lo", Q(self.lo))
        object.__setattr__(self, "hi", self.lo if self.hi is None else Q(self.hi))
        assert self.lo <= self.hi

    def __add__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        return _B(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _B(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, _B) else _B(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        endpoints = tuple(
            a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)
        )
        return _B(min(endpoints), max(endpoints))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / Q(scalar))

    def __pow__(self, exponent):
        result = _B(1)
        for _ in range(exponent):
            result *= self
        return result


def _decay(value):
    value = Q(value)
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(101)), Q(0))
    return _B(upper - value**101 / factorial(101), upper)


def _sine(value):
    value = Q(value)
    if value < 0:
        return -_sine(-value)
    assert value <= 1
    upper = sum(
        ((-1) ** k * value ** (2 * k + 1) / factorial(2 * k + 1) for k in range(51)),
        Q(0),
    )
    return _B(upper - value**103 / factorial(103), upper)


def _hyperbolic(time):
    square = Q(time) ** 2 / 8
    even = sum((square**k / factorial(2 * k) for k in range(49)), Q(0))
    odd = sum((square**k / factorial(2 * k + 1) for k in range(49)), Q(0))
    even_tail = square**49 / factorial(98) / (1 - square / (99 * 100))
    odd_tail = square**49 / factorial(99) / (1 - square / (100 * 101))
    return _B(even, even + even_tail), Q(time) / 4 * _B(odd, odd + odd_tail)


def _on_heat(time, initial):
    """Independent P4 modes and their first two heat-time derivatives."""
    x0, x1, x2, x3 = (v if isinstance(v, _B) else _B(v) for v in initial)
    a, b = (x0 + x3) / 2, (x1 + x2) / 2
    p, q = (x0 - x3) / 2, (x1 - x2) / 2
    mean, contrast = (a + b) / 2, (a - b) / 2
    e = _decay(time / 2)
    c, k = _hyperbolic(time)
    symmetric = (mean + contrast * e, mean - contrast * e)
    symmetric_rate = (-contrast * e / 2, contrast * e / 2)
    symmetric_acceleration = (contrast * e / 4, -contrast * e / 4)
    anti = (e * (c * p + k * (p + q)), e * (c * q + k * (p - q)))
    differentiated_mode = (
        e * (k * p / 2 + c * (p + q) / 4),
        e * (k * q / 2 + c * (p - q) / 4),
    )
    anti_rate = tuple(
        -value / 2 + derivative for value, derivative in zip(anti, differentiated_mode)
    )
    anti_acceleration = tuple(
        3 * value / 8 - derivative
        for value, derivative in zip(anti, differentiated_mode)
    )

    def lift(sym, antisym):
        return (
            sym[0] + antisym[0],
            sym[1] + antisym[1],
            sym[1] - antisym[1],
            sym[0] - antisym[0],
        )

    return (
        lift(symmetric, anti),
        lift(symmetric_rate, anti_rate),
        lift(symmetric_acceleration, anti_acceleration),
    )


def _off_heat(time, initial):
    x0, x1, x2, x3 = (v if isinstance(v, _B) else _B(v) for v in initial)
    mean = (x1 + x2 + x3) / 3
    odd = (x1 - x3) / 2 * _decay(time / 4)
    even = (x1 - 2 * x2 + x3) / 6 * _decay(3 * time / 4)
    return (
        (x0, mean + odd + even, mean - 2 * even, mean - odd + even),
        (_B(0), -odd / 4 - 3 * even / 4, 3 * even / 2, odd / 4 - 3 * even / 4),
        (_B(0), odd / 16 + 9 * even / 16, -9 * even / 8, -odd / 16 + 9 * even / 16),
    )


def _heat_rows(state, edges, capacity=(Q(1),) * 4):
    rows = [_B(0) for _ in state]
    for i, j in edges:
        difference = (state[j] - state[i]) / 4
        rows[i] += capacity[i] * difference
        rows[j] -= capacity[j] * difference
    return tuple(rows)


def _tangent_rows(state, edges, capacity=(Q(1),) * 4):
    form_heat = _heat_rows(state[:4], edges, capacity)
    phase_heat = _heat_rows(state[4:], edges, capacity)
    return tuple(x + theta / 2 for x, theta in zip(form_heat, phase_heat)) + tuple(
        -x / 2 for x in form_heat
    )


def _zero(value):
    assert value.lo <= 0 <= value.hi
    assert value.hi - value.lo < Q(1, 10**100)


def _contains(endpoints, oracle, tolerance=Q(1, 10**30)):
    assert endpoints[0] <= oracle.lo <= oracle.hi <= endpoints[1]
    assert endpoints[1] - endpoints[0] < tolerance


def _display(value):
    with localcontext() as context:
        context.prec = 25
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _tangent(time, initial_form, initial_phase, *, on):
    """Both critical tangent rows, including their physical-time derivatives."""
    t = Q(time) / 2
    solve = _on_heat if on else _off_heat
    form, form_rate, form_acc = solve(t, initial_form)
    phase, phase_rate, phase_acc = solve(t, initial_phase)
    summed_rate = tuple(x + p for x, p in zip(form_rate, phase_rate))
    summed_acc = tuple(x + p for x, p in zip(form_acc, phase_acc))
    state = tuple(x + t * r for x, r in zip(form, summed_rate)) + tuple(
        p - t * r for p, r in zip(phase, summed_rate)
    )
    rate = tuple(
        x + p / 2 + t * a / 2 for x, p, a in zip(form_rate, phase_rate, summed_acc)
    ) + tuple(-x / 2 - t * a / 2 for x, a in zip(form_rate, summed_acc))
    return state, rate


def _history(preparation_time, response_time, initial, *, on):
    reached, preparation_rate = _tangent(
        preparation_time, initial[:4], initial[4:], on=False
    )
    final, response_rate = _tangent(response_time, reached[:4], reached[4:], on=on)
    return reached, preparation_rate, final, response_rate


@lru_cache(maxsize=None)
def _history_coefficients(preparation_time, response_time):
    coefficients = []
    for j in range(8):
        initial = tuple(Q(i == j) for i in range(8))
        on = _history(preparation_time, response_time, initial, on=True)[2]
        off = _history(preparation_time, response_time, initial, on=False)[2]
        coefficients.append((on[0] + on[1] - off[0] - off[1]) / 2)
    return tuple(coefficients)


def _arguments(**changes):
    delta, epsilon = Q(1, 10000), Q(1, 10**8)
    active_rows, widths = (1, 2, 3, 5, 6, 7), (
        epsilon,
        delta,
        delta,
        epsilon,
        epsilon,
        epsilon,
    )
    generators = tuple(
        tuple(width if i == row else Q(0) for row, width in zip(active_rows, widths))
        for i in range(8)
    )
    values = dict(
        nodes=(0, 1, 2, 3),
        old_edges=((1, 2), (2, 3)),
        old_conductances=(Q(1), Q(1)),
        capacity=(Q(1),) * 4,
        contact=(0, 1),
        contact_weight=Q(1),
        retained_nodes=(0, 1),
        observation_nodes=(0, 1),
        phase_anchor=0,
        coordinate_labels=tuple(
            (i, channel) for channel in ("form", "relative_phase") for i in range(4)
        ),
        initial_center=(Q(0), Q(0), 2 * delta, Q(0), Q(0), Q(0), Q(0), Q(0)),
        initial_generators=generators,
        uncertainty_labels=("x1", "x2", "x3", "phi1", "phi2", "phi3"),
        preparation_relation_cost=0,
        preparation_time=Q(1, 4),
        elapsed_time=Q(1, 4),
        mean_kind="arithmetic",
        reading_error_bound=Q(0),
        tolerance=Q(1, 200000000),
    )
    values.update(changes)
    return values


def _report(**changes):
    return owner._prepared_sine_history_information(**_arguments(**changes))


def _source(arguments, latent):
    return tuple(
        center + sum((v * u for v, u in zip(row, latent)), Q(0))
        for center, row in zip(
            arguments["initial_center"], arguments["initial_generators"]
        )
    )


def _projected_oracle(arguments):
    physical = _history_coefficients(
        arguments["preparation_time"], arguments["elapsed_time"]
    )
    coefficients = tuple(
        physical[node + (0 if channel == "form" else 4)]
        for node, channel in arguments["coordinate_labels"]
    )
    center = sum(
        (a * x for a, x in zip(coefficients, arguments["initial_center"])), _B(0)
    )
    latent = tuple(
        sum(
            (
                a * row[j]
                for a, row in zip(coefficients, arguments["initial_generators"])
            ),
            _B(0),
        )
        for j in range(len(arguments["uncertainty_labels"]))
    )
    return center, latent


def _tube(form, phase, degree, duration):
    z = degree * duration / 4
    growth = 1 / (1 - z**2 / 2)
    ceiling = (phase + z * form) * growth
    return z, growth, ceiling, duration * degree * ceiling**3 * growth / 6


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_preparation_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("history information cannot integrate or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
            (external, "_derive_organized_external_interaction"),
            (external, "_assess_organized_external_interaction"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return _report()


@pytest.mark.parametrize(
    "preparation_time,response_time",
    [(Q(0), Q(1, 4)), (Q(1, 4), Q(0)), (Q(1, 100), Q(1, 100)), (Q(1, 4), Q(1, 4))],
)
@pytest.mark.parametrize(
    "initial",
    [
        (0, 0, Q(1, 5000), 0, 0, 0, 0, 0),
        (
            0,
            Q(1, 10**8),
            Q(3, 10000),
            Q(-1, 10000),
            0,
            Q(-1, 10**8),
            Q(1, 10**8),
            Q(1, 10**8),
        ),
    ],
)
def test_complete_closed_piecewise_tangent_keeps_both_rows_and_handoff_state(
    preparation_time, response_time, initial
):
    for on, edges in ((True, ((0, 1), (1, 2), (2, 3))), (False, ((1, 2), (2, 3)))):
        reached, pre_rate, final, final_rate = _history(
            preparation_time, response_time, initial, on=on
        )
        for actual, expected in zip(pre_rate, _tangent_rows(reached, ((1, 2), (2, 3)))):
            _zero(actual - expected)
        for actual, expected in zip(final_rate, _tangent_rows(final, edges)):
            _zero(actual - expected)
        for state in (reached, final):
            _zero(sum(state[:4], _B(0)) - sum(initial[:4]))
            _zero(sum(state[4:], _B(0)) - sum(initial[4:]))
        if response_time == 0:
            for first, second in zip(reached, final):
                _zero(first - second)
        if not on:
            collapsed, _ = _tangent(
                preparation_time + response_time, initial[:4], initial[4:], on=False
            )
            for actual, expected in zip(final, collapsed):
                _zero(actual - expected)


def test_response_order_is_not_replaced_by_commuting_supports():
    initial = (Q(0), Q(0), Q(1), Q(0)) + (Q(0),) * 4
    normal = _history(Q(1, 4), Q(1, 4), initial, on=True)[2]
    reverse_first, _ = _tangent(Q(1, 4), initial[:4], initial[4:], on=True)
    reverse, _ = _tangent(Q(1, 4), reverse_first[:4], reverse_first[4:], on=False)
    assert (normal[0] - reverse[0]).lo > 0


def test_source_is_independent_full_rank_six_and_retained_uncertainty_is_consumed(
    report,
):
    args = _arguments()
    active = (1, 2, 3, 5, 6, 7)
    diagonal = tuple(args["initial_generators"][i][j] for j, i in enumerate(active))
    assert prod(diagonal) == Q(1, 10000) ** 2 * Q(1, 10**8) ** 4 > 0
    assert all(
        args["initial_generators"][i][j] == (diagonal[k] if j == k else 0)
        for k, i in enumerate(active)
        for j in range(6)
    )
    corners = tuple(_source(args, corner) for corner in product((-1, 1), repeat=6))
    assert len(set(corners)) == 64
    assert report.coordinate_marginal_bounds == tuple(
        (min(state[i] for state in corners), max(state[i] for state in corners))
        for i in range(8)
    )
    assert report.uncertainty_dimension == 6
    assert (
        report.preparation_relation_cost
        == report.measurement_cost
        == report.evolved_coordinate_count
        == 0
    )
    assert report.static_coordinate_count == 8
    assert report.initial_generators[1][0] > 0
    assert report.initial_generators[5][3] > 0
    old = {key: value for key, value in args.items() if key != "preparation_time"}
    with pytest.raises(ValueError, match="retained"):
        direct._prepared_sine_joint_information(**old)


def test_history_coefficients_obey_frozen_analytical_bounds_and_signed_box_oracle(
    report, record_property
):
    physical = _history_coefficients(Q(1, 4), Q(1, 4))
    for actual, lower, upper in (
        (physical[2] - physical[6], Q(0), Q(1, 98304)),
        (physical[3] - physical[7], Q(0), Q(11, 50331648)),
        (physical[6], Q(0), Q(1, 32768)),
        (physical[7], Q(0), Q(11, 12582912)),
    ):
        assert lower < actual.lo <= actual.hi < upper
    assert physical[2].lo > Q(217, 6291456)
    assert (physical[2] + physical[3]).hi < Q(2103, 50331648)
    assert sum(max(abs(physical[i].lo), abs(physical[i].hi)) for i in (1, 5, 6, 7)) < Q(
        16139, 12582912
    )
    center, latent = _projected_oracle(_arguments())
    _contains(report.tangent_center_bounds, center)
    for actual, expected in zip(report.tangent_generator_coefficient_bounds, latent):
        _contains(actual, expected)
    radius = sum((-value if value.hi < 0 else value for value in latent), _B(0))
    assert all(value.lo > 0 or value.hi < 0 for value in latent)
    _contains(report.linear_information_error_bounds, radius)
    record_property(
        "history_tangent_radius",
        tuple(_display(x) for x in report.linear_information_error_bounds),
    )


def test_all_64_initial_corners_continue_without_reboxing_or_reset(report):
    args = _arguments()
    center, latent = _projected_oracle(args)
    physical = _history_coefficients(Q(1, 4), Q(1, 4))
    values = []
    for corner in product((-1, 1), repeat=6):
        initial = _source(args, corner)
        direct_response = sum((a * x for a, x in zip(physical, initial)), _B(0))
        affine_response = center + sum((a * u for a, u in zip(latent, corner)), _B(0))
        _zero(direct_response - affine_response)
        model = report.nonlinear_response_error_upper_bound
        assert report.forecast_response_bounds[0] <= direct_response.lo - model
        assert direct_response.hi + model <= report.forecast_response_bounds[1]
        values.append((corner, direct_response))
    maximum_corner = tuple(-1 if value.hi < 0 else 1 for value in latent)
    maximum = next(value for corner, value in values if corner == maximum_corner)
    minimum = next(
        value for corner, value in values if corner == tuple(-x for x in maximum_corner)
    )
    radius = sum((-value if value.hi < 0 else value for value in latent), _B(0))
    _zero((maximum - minimum) / 2 - radius)
    _zero((maximum + minimum) / 2 - center)


def test_global_piecewise_remainder_has_original_state_and_total_clock(report):
    b0, p0 = Q(3, 10000), Q(1, 10**8)
    z, growth, phase, branch = _tube(b0, p0, Q(2), Q(1, 2))
    assert z == Q(1, 4) and growth == Q(32, 31)
    assert report.initial_form_upper_bound == b0
    assert report.initial_phase_upper_bound == p0
    assert report.nonlinear_response_error_upper_bound == 2 * branch < Q(17, 10**14)
    # A future-only allowance from the original source loses the preparation.
    assert branch > _tube(b0, p0, Q(2), Q(1, 4))[3]
    assert phase > p0


@pytest.mark.parametrize("z", [Q(0), Q(1, 4), Q(1, 2)])
def test_independent_volterra_series_covers_full_piecewise_time(z):
    even = sum((z ** (2 * j) / factorial(2 * j) for j in range(49)), Q(0))
    odd = sum((z ** (2 * j) / factorial(2 * j + 1) for j in range(49)), Q(0))
    cosh_upper = even + z**98 / factorial(98) / (1 - z * z / (99 * 100))
    sinh_ratio_upper = odd + z**98 / factorial(99) / (1 - z * z / (100 * 101))
    growth = 1 / (1 - z * z / 2)
    assert cosh_upper <= growth and sinh_ratio_upper <= growth
    b0, p0 = Q(3, 10000), Q(1, 10**8)
    assert p0 * cosh_upper + z * b0 * sinh_ratio_upper <= (p0 + z * b0) * growth


def test_actual_whole_channel_hybrids_prove_lost_association_obstruction(
    record_property,
):
    delta, epsilon = Q(1, 10000), Q(1, 10**8)
    initial_plus = (0, 0, 3 * delta, 0, 0, 0, 0, 0)
    initial_minus = (0, 0, delta, 0, 0, 0, 0, 0)
    plus = _tangent(Q(1, 4), initial_plus[:4], initial_plus[4:], on=False)[0]
    minus = _tangent(Q(1, 4), initial_minus[:4], initial_minus[4:], on=False)[0]
    half = tuple((a - b) / 2 for a, b in zip(plus, minus))
    assert half[1].lo > Q(29, 512) * delta
    assert (-half[5]).lo > Q(29, 1024) * delta
    assert half[2].lo > Q(7, 8) * delta
    # Entire form and phase vectors are selected, preserving each channel's
    # spatial constraints and inventories. With the actual reached vectors
    # the same construction belongs to proj_x(S) x proj_phi(S) by definition;
    # it need not be a reachable joint state. No outer box is substituted.
    hybrid_plus, hybrid_minus = plus[:4] + minus[4:], minus[:4] + plus[4:]
    _zero(sum(hybrid_plus[:4], _B(0)) - 3 * delta)
    _zero(sum(hybrid_minus[:4], _B(0)) - delta)
    _zero(sum(hybrid_plus[4:], _B(0)))
    _zero(sum(hybrid_minus[4:], _B(0)))
    direct_coefficients = _history_coefficients(Q(0), Q(1, 4))
    assert direct_coefficients[1].lo > Q(705, 1048576)
    assert direct_coefficients[5].lo > Q(465, 1048576)
    assert direct_coefficients[2].lo > Q(435, 50331648)
    hybrid_half_gap = sum(
        (
            coefficient * (a - b) / 2
            for coefficient, a, b in zip(direct_coefficients, hybrid_plus, hybrid_minus)
        ),
        _B(0),
    )
    analytical_lower = delta * (
        Q(54375, 1073741824) + Q(3045, 402653184) - Q(1, 2097152)
    )
    assert hybrid_half_gap.lo > analytical_lower
    # The phase2 contribution is adverse in a whole-channel swap. It must
    # remain in the signed sum, even though selected other terms are positive.
    assert (-direct_coefficients[6] * half[6]).hi < 0
    zp, _, ph, ex = _tube(3 * delta, epsilon, Q(2), Q(1, 4))
    ephase = zp * ex
    bh = 3 * delta + zp * ph
    assert ex < Q(5, 10**15) and ephase == ex / 8
    assert bh < Q(305, 100) * delta and ph < Q(38, 100) * delta
    transfer = 2 * ((1 + Q(1, 8)) * ex + Q(1, 8) * ephase)
    assert transfer == Q(73, 32) * ex
    hybrid_model = 2 * _tube(Q(305, 100) * delta, Q(38, 100) * delta, Q(2), Q(1, 4))[3]
    assert hybrid_model < Q(8, 10**14)
    unavoidable = hybrid_half_gap.lo - transfer - hybrid_model
    assert unavoidable > Q(57, 10**10) > Q(1, 200000000)
    record_property(
        "whole_channel_hybrid_tangent_half_gap",
        tuple(_display(x) for x in (hybrid_half_gap.lo, hybrid_half_gap.hi)),
    )
    record_property("prehistory_form_error_upper", _display(ex))
    record_property("prehistory_phase_error_upper", _display(ephase))
    record_property("hybrid_initial_error_transfer", _display(transfer))
    record_property("hybrid_contact_nonlinear_error", _display(hybrid_model))
    record_property("actual_channel_product_unavoidable_lower", _display(unavoidable))
    record_property(
        "membership",
        "whole-channel selections from two actual reached states; exact projection product; no nonlinear endpoint evaluated",
    )


def test_frozen_useful_history_forecast_is_positive(report, record_property):
    assert report.linear_information_error_bounds[1] < Q(42, 10**10)
    assert report.nonlinear_response_error_upper_bound < Q(17, 10**14)
    assert report.forecast_error_upper_bound < Q(42, 10**10) < report.tolerance
    assert report.forecast_response_bounds[0] > Q(26, 10**10)
    assert report.computational_center_error_upper_bound < Q(1, 10**30)
    assert report.status == "certified" and report.response_sign == "positive"
    record_property("target", str(report.tolerance))
    record_property("history_forecast", _display(report.predicted_response))
    record_property("total_forecast_error", _display(report.forecast_error_upper_bound))
    record_property(
        "global_nonlinear_remainder",
        _display(report.nonlinear_response_error_upper_bound),
    )
    record_property(
        "actual_response_interval",
        tuple(_display(x) for x in report.forecast_response_bounds),
    )


@pytest.mark.parametrize("common_phase", [Q(0), Q(2, 7)])
def test_preparation_full_field_jet_has_phase_feedback_before_contact(common_phase):
    delta = Q(1, 10000)
    field = flow._held_quarter_sine_field(
        capacity=(1,) * 4, gamma=Q(1, 2), weighted_edges=((1, 2, 1), (2, 3, 1))
    )
    initial = (Q(0), Q(0), 2 * delta, Q(0)) + (common_phase,) * 4
    first = field(initial)
    expected = tuple(
        _B(x)
        for x in (0, delta / 2, -delta, delta / 2, 0, -delta / 4, delta / 2, -delta / 4)
    )
    for actual, oracle in zip(first, expected):
        _contains((actual.lo, actual.hi), oracle, Q(1, 10**35))
    jets = tuple(
        Jet((I(value), derivative)) for value, derivative in zip(initial, first)
    )
    second = field(jets)
    expected_second = tuple(
        _B(x)
        for x in (
            0,
            -9 * delta / 32,
            9 * delta / 16,
            -9 * delta / 32,
            0,
            3 * delta / 16,
            -3 * delta / 8,
            3 * delta / 16,
        )
    )
    for actual, oracle in zip(second, expected_second):
        _contains((actual.coeffs[1].lo, actual.coeffs[1].hi), oracle, Q(1, 10**35))
    assert second[1].coeffs[1].lo > -3 * delta / 8
    assert first[5].hi < 0 and second[5].coeffs[1].lo > 0


def test_preparation_report_encloses_full_reached_state_in_original_row_order(report):
    args = _arguments()
    reached, _ = _tangent(
        Q(1, 4), args["initial_center"][:4], args["initial_center"][4:], on=False
    )
    for actual, expected in zip(report.preparation_tangent_center_bounds, reached):
        _contains(actual, expected)
    for j in range(6):
        column = tuple(row[j] for row in args["initial_generators"])
        propagated, _ = _tangent(Q(1, 4), column[:4], column[4:], on=False)
        for i, expected in enumerate(propagated):
            _contains(report.preparation_tangent_generator_bounds[i][j], expected)
    zp, growth, ph, ex = _tube(Q(3, 10000), Q(1, 10**8), Q(2), Q(1, 4))
    assert report.preparation_phase_growth_bound == growth
    assert report.preparation_phase_tube_upper_bound == ph
    assert report.preparation_form_tube_upper_bound == Q(3, 10000) + zp * ph
    assert report.preparation_form_error_upper_bound == ex
    assert report.preparation_phase_error_upper_bound == zp * ex
    # These are computable outer bounds for the actual image, not its exact
    # projections. The independent 64 tangent corners plus the justified
    # prefix errors must fit them; no extrema of the nonlinear image claimed.
    for corner in product((-1, 1), repeat=6):
        state = _source(args, corner)
        reached, _ = _tangent(Q(1, 4), state[:4], state[4:], on=False)
        for i, (actual, expected) in enumerate(
            zip(report.preparation_marginal_bounds, reached)
        ):
            error = ex if i < 4 else zp * ex
            assert actual[0] <= expected.lo - error
            assert expected.hi + error <= actual[1]
    assert (
        report.preparation_marginal_bounds[1][0]
        < report.preparation_marginal_bounds[1][1]
    )
    assert (
        report.preparation_marginal_bounds[5][0]
        < report.preparation_marginal_bounds[5][1]
    )


def test_original_phase_anchor_is_not_reset_when_preparation_moves_it():
    report = _report(old_edges=((0, 2), (2, 3)))
    assert report.initial_center[4] == 0 and all(
        x == 0 for x in report.initial_generators[4]
    )
    expected, _ = _tangent(Q(1, 4), (0, 0, Q(1, 5000), 0), (0, 0, 0, 0), on=False)
    # Relabel old P3 node1 as physical0 and its isolated node0 as physical1.
    order = (1, 0, 2, 3)
    for i, j in enumerate(order):
        _contains(report.preparation_tangent_center_bounds[i], expected[j])
        _contains(report.preparation_tangent_center_bounds[i + 4], expected[j + 4])
    assert (
        report.preparation_tangent_center_bounds[4][1]
        + report.preparation_phase_error_upper_bound
        < 0
    )


@pytest.mark.parametrize("mean_kind", ["arithmetic", "inverse_capacity"])
def test_actual_capacities_and_observation_weights_share_the_same_clock(mean_kind):
    capacities = (Q(1), Q(2), Q(3), Q(4))
    report = _report(
        capacity=capacities,
        preparation_time=Q(1, 32),
        elapsed_time=Q(1, 32),
        mean_kind=mean_kind,
    )
    assert report.observation_weights == (
        (Q(1, 2), Q(1, 2), Q(0), Q(0))
        if mean_kind == "arithmetic"
        else (Q(2, 3), Q(1, 3), Q(0), Q(0))
    )
    for matrix, edges in (
        (report.off_generator, ((1, 2), (2, 3))),
        (report.on_generator, ((0, 1), (1, 2), (2, 3))),
    ):
        for j in range(4):
            basis = tuple(_B(Q(i == j)) for i in range(4))
            expected = _heat_rows(basis, edges, capacities)
            assert tuple(matrix[i][j] for i in range(4)) == tuple(
                row.lo for row in expected
            )
    assert (
        report.preparation_mobility_degree
        == report.off_mobility_degree
        == report.off_history_mobility_degree
    )
    assert report.on_history_mobility_degree == max(
        report.preparation_mobility_degree, report.on_mobility_degree
    )


def test_common_clock_scales_both_history_windows_and_each_dynamic_row(report):
    scaled = _report(
        capacity=(Q(3),) * 4, preparation_time=Q(1, 12), elapsed_time=Q(1, 12)
    )
    for name in (
        "tangent_center_bounds",
        "tangent_generator_coefficient_bounds",
        "preparation_tangent_center_bounds",
        "preparation_tangent_generator_bounds",
        "preparation_marginal_bounds",
        "preparation_form_error_upper_bound",
        "preparation_phase_error_upper_bound",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
        "status",
        "response_sign",
    ):
        assert getattr(scaled, name) == getattr(report, name)
    assert report.total_time == report.preparation_time + report.elapsed_time == Q(1, 2)


def test_node_row_and_latent_associations_remain_attached_across_handoff(report):
    args = _arguments()
    names = {0: "anchor", 1: "retained", 2: "near", 3: "far"}
    row_order = (7, 2, 4, 1, 6, 0, 3, 5)
    column_order = (5, 1, 3, 0, 4, 2)
    signs = (-1, 1, -1, 1, 1, -1)
    changed = _report(
        nodes=("far", "retained", "anchor", "near"),
        old_edges=(("near", "retained"), ("far", "near")),
        contact=("retained", "anchor"),
        retained_nodes=("retained", "anchor"),
        observation_nodes=("anchor", "retained"),
        phase_anchor="anchor",
        coordinate_labels=tuple(
            (names[args["coordinate_labels"][i][0]], args["coordinate_labels"][i][1])
            for i in row_order
        ),
        initial_center=tuple(args["initial_center"][i] for i in row_order),
        initial_generators=tuple(
            tuple(
                args["initial_generators"][i][j] * signs[k]
                for k, j in enumerate(column_order)
            )
            for i in row_order
        ),
        uncertainty_labels=tuple(args["uncertainty_labels"][j] for j in column_order),
    )
    for name in (
        "tangent_center_bounds",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
        "status",
    ):
        assert getattr(changed, name) == getattr(report, name)
    assert changed.preparation_tangent_center_bounds == tuple(
        report.preparation_tangent_center_bounds[i] for i in row_order
    )
    assert changed.preparation_marginal_bounds == tuple(
        report.preparation_marginal_bounds[i] for i in row_order
    )
    expected = tuple(
        tuple(
            signs[k] * x
            for x in (
                report.tangent_generator_coefficient_bounds[j]
                if signs[k] > 0
                else report.tangent_generator_coefficient_bounds[j][::-1]
            )
        )
        for k, j in enumerate(column_order)
    )
    assert changed.tangent_generator_coefficient_bounds == expected


def test_no_preparation_reduces_to_existing_known_retained_joint_contract():
    args = _arguments(preparation_time=Q(0))
    rows = list(args["initial_generators"])
    rows[1] = rows[5] = (Q(0),) * 6
    args["initial_generators"] = tuple(rows)
    history = owner._prepared_sine_history_information(**args)
    original = direct._prepared_sine_joint_information(
        **{k: v for k, v in args.items() if k != "preparation_time"}
    )
    # The two contractions enclose the same exact response but group signed
    # numerical intervals differently. Computational widths need not match.
    center, latent = _projected_oracle(args)
    radius = sum((-value if value.hi < 0 else value for value in latent), _B(0))
    for result in (history, original):
        _contains(result.tangent_center_bounds, center)
        for actual, expected in zip(
            result.tangent_generator_coefficient_bounds, latent
        ):
            _contains(actual, expected)
        _contains(result.linear_information_error_bounds, radius)
        assert result.computational_center_error_upper_bound < Q(1, 10**30)
    for name in (
        "nonlinear_response_error_upper_bound",
        "status",
        "response_sign",
    ):
        assert getattr(history, name) == getattr(original, name)
    assert (
        history.preparation_form_error_upper_bound
        == history.preparation_phase_error_upper_bound
        == 0
    )


@pytest.mark.parametrize("zero", ["response_time", "contact"])
def test_matched_zero_does_not_erase_nonzero_preparation(zero):
    report = _report(
        **(
            {"elapsed_time": Q(0)}
            if zero == "response_time"
            else {"contact_weight": Q(0)}
        )
    )
    assert report.preparation_kernel is not None
    assert report.preparation_form_error_upper_bound > 0
    assert report.tangent_center_bounds == (Q(0), Q(0))
    assert report.tangent_generator_coefficient_bounds == ((Q(0), Q(0)),) * 6
    assert report.forecast_response_bounds == (Q(0), Q(0))
    assert report.nonlinear_response_error_upper_bound == 0
    assert report.preparation_tangent_center_bounds[1][0] > 0
    assert report.status == "certified" and report.response_sign == "zero"


def test_point_zero_readout_and_declared_cost_have_distinct_meanings(report):
    point = _report(initial_generators=((),) * 8, uncertainty_labels=())
    assert (
        point.uncertainty_dimension == 0
        and point.linear_information_error_bounds == (Q(0), Q(0))
    )
    assert point.nonlinear_response_error_upper_bound > 0
    zero = _report(
        initial_center=(Q(0),) * 8, initial_generators=((),) * 8, uncertainty_labels=()
    )
    assert (
        zero.forecast_response_bounds == (Q(0), Q(0)) and zero.response_sign == "zero"
    )
    noisy = _report(reading_error_bound=Q(1, 10**7), preparation_relation_cost=13)
    assert noisy.forecast_error_upper_bound == report.forecast_error_upper_bound + Q(
        1, 10**7
    )
    assert noisy.information_error_lower_bound == report.information_error_lower_bound
    assert noisy.preparation_relation_cost == 13
    assert noisy.status == noisy.response_sign == "unresolved"


def test_closed_whole_clock_and_chart_boundaries_and_redundant_columns():
    report = _report(preparation_time=Q(1, 2), elapsed_time=Q(1, 2))
    assert (
        report.total_time == 1
        and report.off_phase_growth_bound == report.on_phase_growth_bound == Q(8, 7)
    )
    args = _arguments()
    point = _report(
        initial_center=(0, 0, 0, 0, 0, 0, 0, Q(1, 4)),
        initial_generators=((),) * 8,
        uncertainty_labels=(),
        preparation_time=Q(0),
        elapsed_time=Q(0),
    )
    assert point.initial_phase_upper_bound == Q(1, 4)
    extra = _report(
        initial_generators=tuple(
            row + (Q(0),) * 42 for row in args["initial_generators"]
        ),
        uncertainty_labels=tuple(range(48)),
    )
    assert extra.uncertainty_dimension == 48
    assert extra.forecast_response_bounds == _report().forecast_response_bounds


def _invalid_cases():
    args = _arguments()
    labels, center, rows = (
        args["coordinate_labels"],
        args["initial_center"],
        args["initial_generators"],
    )

    def row_change(index, row):
        return tuple(row if i == index else current for i, current in enumerate(rows))

    return [
        {"nodes": (0, 1, 1, 3)},
        {"nodes": {0, 1, 2, 3}},
        {"nodes": repeat(0)},
        {"old_edges": ((1, 2), (2, 1))},
        {"old_conductances": (True, 1)},
        {"capacity": (1, 0, 1, 1)},
        {"contact_weight": Q(-1)},
        {"retained_nodes": (0, 1, 2, 3)},
        {"retained_nodes": (0, 2)},
        {"observation_nodes": (2,)},
        {"phase_anchor": 2},
        {"phase_anchor": 9},
        {"coordinate_labels": labels[:-1]},
        {"coordinate_labels": labels[:-1] + (labels[0],)},
        {"coordinate_labels": labels[:-1] + ((9, "relative_phase"),)},
        {"coordinate_labels": labels[:-1] + ((3, "phase"),)},
        {"coordinate_labels": set(labels)},
        {"coordinate_labels": repeat((0, "form"))},
        {"initial_center": center[:-1]},
        {"initial_center": (True,) + center[1:]},
        {"initial_center": center[:2] + (float("nan"),) + center[3:]},
        {"initial_generators": rows[:-1]},
        {"initial_generators": repeat(rows[0])},
        {"initial_generators": row_change(2, (Q(0),) * 5)},
        {"initial_generators": row_change(2, repeat(Q(0)))},
        {"initial_generators": row_change(2, (True,) + (Q(0),) * 5)},
        {"initial_generators": row_change(2, (float("inf"),) + (Q(0),) * 5)},
        {"initial_center": center[:4] + (Q(1, 10**100),) + center[5:]},
        {"initial_generators": row_change(4, (Q(1, 10**100),) + (Q(0),) * 5)},
        {"initial_center": center[:7] + (Q(1, 4),)},
        {"initial_generators": row_change(7, (Q(1, 4), Q(1, 10**100)) + (Q(0),) * 4)},
        {"uncertainty_labels": ("a",) * 6},
        {"uncertainty_labels": tuple(range(5))},
        {"uncertainty_labels": set(range(6))},
        {"uncertainty_labels": repeat("u")},
        {"uncertainty_labels": tuple(range(49))},
        {"preparation_relation_cost": True},
        {"preparation_relation_cost": -1},
        {"preparation_relation_cost": 2.0},
        {"preparation_relation_cost": Q(2)},
        {"preparation_time": True},
        {"preparation_time": Q(-1)},
        {"elapsed_time": True},
        {"elapsed_time": Q(-1)},
        {"preparation_time": Q(3, 4) + Q(1, 10**100)},
        {"mean_kind": "arbitrary"},
        {"reading_error_bound": Q(-1)},
        {"tolerance": True},
    ]


@pytest.mark.parametrize("changes", _invalid_cases())
def test_every_primitive_and_global_budget_precedes_numerical_work(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid history reached a numerical kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    monkeypatch.setattr(owner, "exact_matrix_product", forbidden)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _report(**changes)
    if not {"preparation_time", "elapsed_time", "contact_weight"}.intersection(changes):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            _report(**{**changes, "elapsed_time": Q(0), "contact_weight": Q(0)})


def test_late_global_domain_is_admitted_before_preparation_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid response branch reached preparation kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    monkeypatch.setattr(owner, "exact_matrix_product", forbidden)
    # Prefix and each separate duration fit; the complete on-history does not.
    with pytest.raises(ValueError):
        _report(contact_weight=Q(4), preparation_time=Q(1, 4), elapsed_time=Q(1, 4))
    # Response-zero does not excuse an inadmissible preparation window.
    with pytest.raises(ValueError):
        _report(preparation_time=Q(2), elapsed_time=Q(0))


def test_report_is_frozen_and_ordered_inputs_normalize():
    args = _arguments()
    report = _report(
        nodes=[0, 1, 2, 3],
        coordinate_labels=[list(x) for x in args["coordinate_labels"]],
        initial_generators=[list(row) for row in args["initial_generators"]],
    )
    assert report.coordinate_labels == args["coordinate_labels"]
    assert report.initial_generators == args["initial_generators"]
    with pytest.raises(FrozenInstanceError):
        report.predicted_response = Q(0)
