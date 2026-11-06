"""Independent finite phase-information controls under the complete sine law.

Closed functions describe only the critically damped tangent. A separately
proved nonlinear remainder bounds the actual contact response; no numerical
trajectory or nonlinear endpoint is used as an oracle.
"""

from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import product, repeat
from math import factorial

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as flow
from tnfr.physics import _sine_phase_information as owner


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
    x0, x1, x2, x3 = map(Q, initial)
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
    x0, x1, x2, x3 = map(Q, initial)
    mean = (x1 + x2 + x3) / 3
    odd = (x1 - x3) / 2 * _decay(time / 4)
    even = (x1 - 2 * x2 + x3) / 6 * _decay(3 * time / 4)
    return (
        (_B(x0), mean + odd + even, mean - 2 * even, mean - odd + even),
        (_B(0), -odd / 4 - 3 * even / 4, 3 * even / 2, odd / 4 - 3 * even / 4),
        (_B(0), odd / 16 + 9 * even / 16, -9 * even / 8, -odd / 16 + 9 * even / 16),
    )


def _tangent(time, phase, *, on):
    heat_time = Q(time) / 2
    state, rate, acceleration = (_on_heat if on else _off_heat)(heat_time, phase)
    form = tuple(heat_time * value for value in rate)
    angle = tuple(
        value - heat_time * derivative for value, derivative in zip(state, rate)
    )
    form_rate = tuple(
        (derivative + heat_time * second) / 2
        for derivative, second in zip(rate, acceleration)
    )
    phase_rate = tuple(-heat_time * second / 2 for second in acceleration)
    return form + angle, form_rate + phase_rate


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


def _coefficients(time):
    result = []
    for j in range(4):
        initial = tuple(Q(i == j) for i in range(4))
        on, _ = _tangent(time, initial, on=True)
        off, _ = _tangent(time, initial, on=False)
        result.append((on[0] + on[1] - off[0] - off[1]) / 2)
    return tuple(result)


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


def _arguments(**changes):
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
        retained_relative_phase=(Q(0), Q(0)),
        hidden_prior_relative_phase_bounds=((Q(-1, 1000), Q(1, 1000)),) * 2,
        measured_nodes=(),
        phase_measurement_values=(),
        phase_measurement_error_bounds=(),
        elapsed_time=Q(1, 4),
        mean_kind="arithmetic",
        reading_error_bound=Q(0),
        tolerance=Q(1, 10**9),
    )
    values.update(changes)
    return values


def _report(**changes):
    return owner._prepared_sine_phase_information(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("phase information controls cannot integrate or execute events")

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
def p4_reports():
    common = dict(
        phase_measurement_values=(Q(1, 2000),),
        phase_measurement_error_bounds=(Q(1, 100000),),
    )
    return {
        "none": _report(),
        "near": _report(measured_nodes=(2,), **common),
        "far": _report(measured_nodes=(3,), **common),
    }


@pytest.mark.parametrize("time", [Q(0), Q(1, 100), Q(1, 4)])
@pytest.mark.parametrize(
    "initial", [(0, 0, 1, 0), (0, 0, 0, 1), (0, Q(1, 2000), Q(-1, 1000), Q(1, 1000))]
)
def test_independently_differentiated_modes_satisfy_both_coupled_tangent_rows(
    time, initial
):
    for on, edges in ((True, ((0, 1), (1, 2), (2, 3))), (False, ((1, 2), (2, 3)))):
        state, rate = _tangent(time, initial, on=on)
        for derivative, row in zip(rate, _tangent_rows(state, edges)):
            _zero(derivative - row)
        _zero(sum(state[:4], _B(0)))
        _zero(sum(state[4:], _B(0)) - sum(initial))
        if time == 0:
            assert state[:4] == (_B(0),) * 4
            assert state[4:] == tuple(_B(x) for x in initial)


def test_closed_tangent_covector_uses_heat_time_derivative_not_extra_chain_factor(
    p4_reports, record_property
):
    report = p4_reports["none"]
    coefficient = _coefficients(Q(1, 4))
    t = Q(1, 8)
    r, e = _decay(t / 4), _decay(t / 2)
    c, k = _hyperbolic(t)
    b2 = t / 8 * (e * (c - k) - r**3)
    b3 = t / 16 * (-r + r**3 + 2 * e * k)
    _zero(coefficient[2] - b2)
    _zero(coefficient[3] - b3)
    for actual, expected in zip(
        report.tangent_response_coefficient_bounds, coefficient
    ):
        _contains(actual, expected)
    assert Q(105, 16777216) < b2.lo <= b2.hi < Q(1, 131072)
    assert Q(105, 1610612736) < b3.lo <= b3.hi < Q(1, 12582912)
    assert report.heat_time == t
    record_property(
        "near_tangent_phase_coefficient", tuple(_display(x) for x in (b2.lo, b2.hi))
    )
    record_property(
        "far_tangent_phase_coefficient", tuple(_display(x) for x in (b3.lo, b3.hi))
    )


def test_frozen_nonlinear_accuracy_distinguishes_the_location_of_phase_information(
    p4_reports, record_property
):
    near, far, none = (p4_reports[name] for name in ("near", "far", "none"))
    assert near.status == "certified"
    assert near.forecast_error_upper_bound < Q(1, 10**9)
    assert near.response_sign == "positive"
    assert near.forecast_response_bounds[0] > Q(28, 10**10)
    assert none.status == far.status == "insufficient_information"
    assert none.information_error_lower_bound > Q(6, 10**9)
    assert far.information_error_lower_bound > Q(6, 10**9)
    assert near.measurement_cost == far.measurement_cost == 1
    assert none.measurement_cost == 0
    assert near.gamma == Q(1, 2)
    assert near.hidden_posterior_relative_phase_bounds == (
        (Q(49, 100000), Q(51, 100000)),
        (Q(-1, 1000), Q(1, 1000)),
    )
    assert near.computational_center_error_upper_bound < Q(1, 10**30)
    assert (
        near.forecast_error_upper_bound
        == near.linear_information_error_bounds[1]
        + near.nonlinear_response_error_upper_bound
        + near.computational_center_error_upper_bound
    )
    for report in (none, near, far):
        assert report.information_error_lower_bound == max(
            Q(0),
            report.linear_information_error_bounds[0]
            - report.nonlinear_response_error_upper_bound,
        )
    record_property("fixed_nonlinear_response_tolerance", "1/1000000000")
    record_property(
        "tangent_information_radius_bounds",
        tuple(_display(x) for x in near.linear_information_error_bounds),
    )
    record_property(
        "nonlinear_remainder", _display(near.nonlinear_response_error_upper_bound)
    )
    record_property("rational_forecast", _display(near.predicted_response))
    record_property("total_forecast_error", _display(near.forecast_error_upper_bound))
    record_property(
        "actual_nonlinear_response_certified_interval",
        tuple(_display(x) for x in near.forecast_response_bounds),
    )
    record_property(
        "no_phase_reading_unavoidable_error_lower",
        _display(none.information_error_lower_bound),
    )
    record_property(
        "remote_phase_reading_unavoidable_error_lower",
        _display(far.information_error_lower_bound),
    )
    record_property(
        "interpretation",
        "nonlinear response bounded by tangent plus rigorous remainder; no nonlinear endpoint evaluated",
    )


@pytest.mark.parametrize("remote_measured", [False, True])
def test_indistinguishable_complete_phase_preparations_have_finite_nonlinear_half_gap(
    p4_reports, remote_measured
):
    report = p4_reports["far" if remote_measured else "none"]
    far = Q(1, 2000) if remote_measured else Q(0)
    responses = []
    for near in (Q(-1, 1000), Q(1, 1000)):
        preparation = (Q(0), Q(0), near, far)
        assert preparation[:2] == (Q(0), Q(0))
        if remote_measured:
            assert preparation[3] == report.phase_measurement_values[0]
        on, _ = _tangent(Q(1, 4), preparation, on=True)
        off, _ = _tangent(Q(1, 4), preparation, on=False)
        responses.append((on[0] + on[1] - off[0] - off[1]) / 2)
    half_gap = (responses[1] - responses[0]) / 2
    _zero(half_gap - _coefficients(Q(1, 4))[2] / 1000)
    # Each complete nonlinear response is within E of its tangent response;
    # their half-separation is therefore at least the tangent half-gap minus E.
    # The nonlinear endpoints themselves have not been numerically evaluated.
    assert half_gap.lo - report.nonlinear_response_error_upper_bound > Q(6, 10**9)
    assert half_gap.lo - report.nonlinear_response_error_upper_bound > report.tolerance


def test_nonlinear_remainder_has_independent_scalar_comparison_and_original_prior(
    p4_reports,
):
    report = p4_reports["near"]
    delta, time, mobility = Q(1, 1000), Q(1, 4), Q(2)
    z = mobility * time / 4
    c = 1 / (1 - z**2 / 2)
    assert c == Q(128, 127)
    assert c - 1 - z**2 * c / 2 == 0
    # Volterra: Theta <= delta + z^2 Theta/2. Sine residual on any
    # edge is <=(2 delta C)^3/6, hence form forcing <=k delta^3 C^3/6.
    forcing = mobility * delta**3 * c**3 / 6
    branch = time * forcing * c
    assert report.initial_phase_upper_bound == delta
    assert report.off_mobility_degree == report.on_mobility_degree == mobility
    assert report.off_phase_growth_bound == report.on_phase_growth_bound == c
    assert (
        report.off_nonlinear_error_upper_bound
        == report.on_nonlinear_error_upper_bound
        == branch
    )
    assert (
        report.nonlinear_response_error_upper_bound == 2 * branch == delta**3 * c**4 / 6
    )
    assert 2 * branch < Q(173, 10**12)
    assert Q(1, 13107200000) + Q(1, 12582912000) + 2 * branch < Q(329, 10**12)
    known = _report(
        measured_nodes=(2, 3),
        phase_measurement_values=(Q(0), Q(0)),
        phase_measurement_error_bounds=(Q(0), Q(0)),
    )
    assert known.linear_information_error_bounds == (Q(0), Q(0))
    assert known.initial_phase_upper_bound == delta
    assert known.nonlinear_response_error_upper_bound == 2 * branch
    assert known.information_error_lower_bound == 0
    assert known.forecast_error_upper_bound == 2 * branch
    # The actual all-zero state is stationary; a positive sufficient
    # remainder is not an assertion of actual nonlinear error.
    assert known.response_sign == "unresolved"


@pytest.mark.parametrize("common_phase", [Q(0), Q(2, 7)])
def test_actual_sine_residual_and_phase_acceleration_are_nonzero(common_phase):
    delta = Q(1, 1000)
    sine = _sine(delta)
    field = flow._held_quarter_sine_field(
        capacity=(1,) * 4,
        gamma=Q(1, 2),
        weighted_edges=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
    )
    initial = (Q(0),) * 4 + (
        common_phase,
        common_phase,
        common_phase + delta,
        common_phase,
    )
    first = field(initial)
    expected_form = (_B(0), sine / 8, -sine / 4, sine / 8)
    for actual, expected in zip(first, expected_form + (_B(0),) * 4):
        _contains((actual.lo, actual.hi), expected, Q(1, 10**35))
    assert sine.hi < delta
    assert first[1].hi < delta / 8
    assert first[2].lo > -delta / 4
    jets = tuple(
        Jet((I(value), derivative)) for value, derivative in zip(initial, first)
    )
    second = field(jets)
    expected_form_acceleration = (sine / 32, -sine / 8, 3 * sine / 16, -3 * sine / 32)
    expected_phase_acceleration = (-sine / 64, sine / 16, -3 * sine / 32, 3 * sine / 64)
    for actual, expected in zip(
        second, expected_form_acceleration + expected_phase_acceleration
    ):
        _contains((actual.coeffs[1].lo, actual.coeffs[1].hi), expected, Q(1, 10**35))
    assert second[4].coeffs[1].hi < 0
    assert second[5].coeffs[1].lo > 0


def _signed_coefficients():
    order, time = (2, 0, 1, 3), Q(1, 8)
    e = _decay(time / 2)
    result = []
    for column in range(4):
        initial = tuple(Q(i == column) for i in range(4))
        _, rate, _ = _on_heat(time, tuple(initial[i] for i in order))
        off_rate0 = -(initial[0] - initial[2]) / 4 * e
        result.append(time * (rate[1] - off_rate0))
    return tuple(result)


def test_signed_tangent_box_corners_bound_nonlinear_error_without_claiming_exact_extrema():
    report = _report(
        old_edges=((0, 2), (1, 3)),
        observation_nodes=(0,),
        retained_relative_phase=(Q(0), Q(-1, 2000)),
        hidden_prior_relative_phase_bounds=(
            (Q(-1, 1000), Q(1, 2000)),
            (Q(0), Q(1, 1000)),
        ),
        measured_nodes=(2,),
        phase_measurement_values=(Q(-1, 1000),),
        phase_measurement_error_bounds=(Q(1, 1000),),
        tolerance=Q(1, 1000),
    )
    a = _signed_coefficients()
    assert a[2].hi < 0 < a[3].lo
    for actual, expected in zip(report.tangent_response_coefficient_bounds, a):
        _contains(actual, expected)
    assert report.hidden_posterior_relative_phase_bounds == (
        (Q(-1, 1000), Q(0)),
        (Q(0), Q(1, 1000)),
    )
    center = sum((v * x for v, x in zip(a, report.posterior_center)), _B(0))
    rho = (-a[2] + a[3]) / 2000
    assert (
        report.linear_information_error_bounds[0]
        <= rho.lo
        <= rho.hi
        <= report.linear_information_error_bounds[1]
    )
    errors = report.nonlinear_response_error_upper_bound
    for hidden in product((Q(-1, 1000), Q(0)), (Q(0), Q(1, 1000))):
        response = -a[1] / 2000 + a[2] * hidden[0] + a[3] * hidden[1]
        assert report.forecast_response_bounds[0] <= response.lo - errors
        assert response.hi + errors <= report.forecast_response_bounds[1]
    maximum = -a[1] / 2000 - a[2] / 1000 + a[3] / 1000
    minimum = -a[1] / 2000
    _zero((maximum + minimum) / 2 - center)
    _zero((maximum - minimum) / 2 - rho)
    assert (
        abs(report.predicted_response - center.lo)
        <= report.computational_center_error_upper_bound
    )
    assert (
        abs(report.predicted_response - center.hi)
        <= report.computational_center_error_upper_bound
    )


@pytest.mark.parametrize("mean_kind", ["arithmetic", "inverse_capacity"])
def test_actual_capacity_rows_and_signed_semigroup_projection(mean_kind):
    capacity = (Q(1), Q(2), Q(3), Q(4))
    report = _report(capacity=capacity, elapsed_time=Q(1, 16), mean_kind=mean_kind)
    weights = (
        (Q(1, 2), Q(1, 2), Q(0), Q(0))
        if mean_kind == "arithmetic"
        else (Q(2, 3), Q(1, 3), Q(0), Q(0))
    )
    assert report.observation_weights == weights
    projected = []
    for generator, kernel, edges in (
        (report.off_generator, report.off_kernel, ((1, 2), (2, 3))),
        (report.on_generator, report.on_kernel, ((0, 1), (1, 2), (2, 3))),
    ):
        for column in range(4):
            basis = tuple(_B(Q(i == column)) for i in range(4))
            expected = _heat_rows(basis, edges, capacity)
            assert tuple(row.lo for row in expected) == tuple(
                generator[i][column] for i in range(4)
            )
        signed_row = tuple(
            report.heat_time * sum(weights[i] * generator[i][j] for i in range(4))
            for j in range(4)
        )
        projected.append(
            tuple(
                sum(
                    (
                        _B(kernel.lower[i][j], kernel.upper[i][j]) * signed_row[i]
                        for i in range(4)
                    ),
                    _B(0),
                )
                for j in range(4)
            )
        )
    for interval, on, off in zip(
        report.tangent_response_coefficient_bounds, projected[1], projected[0]
    ):
        assert interval == ((on - off).lo, (on - off).hi)


def test_common_clock_and_source_relabeling_preserve_phase_association(p4_reports):
    baseline = p4_reports["near"]
    common = dict(
        measured_nodes=(2,),
        phase_measurement_values=(Q(1, 2000),),
        phase_measurement_error_bounds=(Q(1, 100000),),
    )
    scaled = _report(capacity=(Q(3),) * 4, elapsed_time=Q(1, 12), **common)
    fields = (
        "tangent_response_coefficient_bounds",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "predicted_response",
        "forecast_error_upper_bound",
        "status",
        "response_sign",
    )
    for name in fields:
        assert getattr(scaled, name) == getattr(baseline, name)
    relabeled = _report(
        nodes=("far", "right", "near", "left"),
        old_edges=(("near", "right"), ("far", "near")),
        contact=("right", "left"),
        retained_nodes=("right", "left"),
        observation_nodes=("left", "right"),
        phase_anchor="left",
        measured_nodes=("near",),
        phase_measurement_values=(Q(1, 2000),),
        phase_measurement_error_bounds=(Q(1, 100000),),
    )
    assert relabeled.tangent_response_coefficient_bounds == tuple(
        baseline.tangent_response_coefficient_bounds[i] for i in (3, 1, 2, 0)
    )
    for name in fields[1:]:
        assert getattr(relabeled, name) == getattr(baseline, name)


def test_zero_straddling_tangent_coefficient_does_not_create_false_information_lower_bound():
    report = _report(old_edges=((0, 2), (1, 2)))
    lower, upper = report.tangent_response_coefficient_bounds[2]
    assert lower < 0 < upper
    assert report.posterior_radii[2] == Q(1, 1000)
    assert (
        report.linear_information_error_bounds[0]
        == report.information_error_lower_bound
        == 0
    )
    assert report.linear_information_error_bounds[1] > 0


def test_tangent_insufficiency_does_not_transfer_when_nonlinear_remainder_dominates():
    report = _report(hidden_prior_relative_phase_bounds=((Q(-1, 4), Q(1, 4)),) * 2)
    assert report.linear_information_error_bounds[0] > report.tolerance
    assert (
        report.nonlinear_response_error_upper_bound
        > report.linear_information_error_bounds[1]
    )
    assert report.information_error_lower_bound == 0
    assert report.status == report.response_sign == "unresolved"


def test_prior_chart_boundary_clipping_and_anchor_order_are_explicit():
    report = _report(
        retained_nodes=(1, 0),
        retained_relative_phase=(Q(1, 4), Q(0)),
        hidden_prior_relative_phase_bounds=((Q(-1, 4), Q(1, 4)),) * 2,
        measured_nodes=(3, 2),
        phase_measurement_values=(Q(-1, 4), Q(1, 4)),
        phase_measurement_error_bounds=(Q(0), Q(0)),
        elapsed_time=Q(0),
    )
    assert report.initial_phase_upper_bound == Q(1, 4)
    assert report.hidden_posterior_relative_phase_bounds == (
        (Q(1, 4), Q(1, 4)),
        (Q(-1, 4), Q(-1, 4)),
    )
    assert report.posterior_center == (Q(0), Q(1, 4), Q(1, 4), Q(-1, 4))
    assert report.measurement_cost == 2


@pytest.mark.parametrize("zero", ["time", "contact"])
def test_identical_branch_shortcuts_follow_all_source_and_chart_admission(
    monkeypatch, zero
):
    def forbidden(*args, **kwargs):
        pytest.fail("admitted identical branches should not evaluate a matrix kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    report = _report(
        **({"elapsed_time": Q(0)} if zero == "time" else {"contact_weight": Q(0)})
    )
    assert report.off_kernel is report.on_kernel is None
    assert report.tangent_response_coefficient_bounds == ((Q(0), Q(0)),) * 4
    assert report.nonlinear_response_error_upper_bound == 0
    if zero == "contact":
        assert report.off_nonlinear_error_upper_bound > 0
        assert report.on_nonlinear_error_upper_bound > 0
    assert report.forecast_error_upper_bound == report.predicted_response == 0
    assert report.status == "certified" and report.response_sign == "zero"


def test_zero_prior_is_exact_stationary_and_final_readout_noise_remains_separate():
    zero = _report(hidden_prior_relative_phase_bounds=((Q(0), Q(0)),) * 2)
    assert (
        zero.initial_phase_upper_bound == zero.nonlinear_response_error_upper_bound == 0
    )
    assert zero.forecast_response_bounds == (Q(0), Q(0))
    assert zero.response_sign == "zero"
    near = dict(
        measured_nodes=(2,),
        phase_measurement_values=(Q(1, 2000),),
        phase_measurement_error_bounds=(Q(1, 100000),),
    )
    clean, noisy = _report(**near), _report(**near, reading_error_bound=Q(1, 10**8))
    assert noisy.forecast_error_upper_bound == clean.forecast_error_upper_bound + Q(
        1, 10**8
    )
    assert (
        noisy.nonlinear_response_error_upper_bound
        == clean.nonlinear_response_error_upper_bound
    )
    assert noisy.information_error_lower_bound == clean.information_error_lower_bound
    assert noisy.status == noisy.response_sign == "unresolved"


def test_nonlinear_comparison_domain_inclusive_boundary_and_additive_old_contact():
    report = _report(elapsed_time=Q(1))
    assert report.on_phase_growth_bound == Q(8, 7)
    assert report.off_phase_growth_bound == Q(8, 7)
    overlap = _report(
        old_edges=((0, 1), (1, 2), (2, 3)),
        old_conductances=(Q(2), Q(1), Q(1)),
        elapsed_time=Q(1, 16),
    )
    assert overlap.off_generator[0][1] == Q(1, 2)
    assert overlap.on_generator[0][1] == Q(3, 4)


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": (0, 1, 1, 3)},
        {"nodes": {0, 1, 2, 3}},
        {"nodes": repeat(0)},
        {"nodes": tuple(range(25))},
        {"old_edges": ((1, 2), (2, 1))},
        {"old_edges": ((1, 1), (2, 3))},
        {"old_edges": ((1, 4), (2, 3))},
        {"old_conductances": (Q(-1), Q(1))},
        {"old_conductances": (True, Q(1))},
        {"capacity": (Q(1), Q(0), Q(1), Q(1))},
        {"capacity": (Q(1), float("inf"), Q(1), Q(1))},
        {"contact": (0, 0)},
        {"contact_weight": Q(-1)},
        {"contact_weight": True},
        {"retained_nodes": (0, 1, 2, 3)},
        {"retained_nodes": (0, 2)},
        {"observation_nodes": ()},
        {"observation_nodes": (2,)},
        {"observation_nodes": (0, 0)},
        {"phase_anchor": 2},
        {"phase_anchor": 9},
        {"retained_relative_phase": (Q(1, 1000), Q(0))},
        {"retained_relative_phase": (Q(0), Q(1, 4) + Q(1, 10**100))},
        {"retained_relative_phase": (Q(0), True)},
        {"retained_relative_phase": (Q(0),)},
        {
            "hidden_prior_relative_phase_bounds": (
                (Q(1, 1000), Q(-1, 1000)),
                (Q(0), Q(0)),
            )
        },
        {
            "hidden_prior_relative_phase_bounds": (
                (Q(-1, 4) - Q(1, 10**100), Q(0)),
                (Q(0), Q(0)),
            )
        },
        {"hidden_prior_relative_phase_bounds": ((Q(-1, 1000), Q(1, 1000)),)},
        {"hidden_prior_relative_phase_bounds": ((float("nan"), Q(0)), (Q(0), Q(0)))},
        {
            "measured_nodes": (0,),
            "phase_measurement_values": (Q(0),),
            "phase_measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2, 2),
            "phase_measurement_values": (Q(0), Q(0)),
            "phase_measurement_error_bounds": (Q(0), Q(0)),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (),
            "phase_measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (Q(1, 10),),
            "phase_measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (Q(0),),
            "phase_measurement_error_bounds": (Q(-1),),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (True,),
            "phase_measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (Q(1, 5),),
            "phase_measurement_error_bounds": (Q(1, 10),),
        },
        {
            "measured_nodes": (2,),
            "phase_measurement_values": (Q(0),),
            "phase_measurement_error_bounds": (Q(1, 4) + Q(1, 10**100),),
        },
        {"elapsed_time": True},
        {"elapsed_time": Q(-1)},
        {"elapsed_time": Q(1) + Q(1, 10**100)},
        {"mean_kind": "phase_absolute"},
        {"reading_error_bound": Q(-1)},
        {"tolerance": True},
        {"tolerance": float("nan")},
    ],
)
def test_all_invalid_primitives_and_charts_reject_before_either_kernel(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid phase information reached an exponential")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _report(**changes)
    if "elapsed_time" not in changes and "contact_weight" not in changes:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            _report(**{**changes, "contact_weight": Q(0)})


def test_both_branch_domains_and_chart_validate_before_shortcuts(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("second branch or chart failed after first kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    with pytest.raises(ValueError):
        _report(contact_weight=Q(4), elapsed_time=Q(1, 2))
    with pytest.raises(ValueError):
        _report(contact_weight=Q(0), elapsed_time=Q(2))
    with pytest.raises(ValueError):
        _report(elapsed_time=Q(0), retained_relative_phase=(Q(0), Q(1, 3)))


def test_report_is_frozen_and_primitive_sequences_are_normalized():
    report = _report(nodes=[0, 1, 2, 3], old_edges=([1, 2], [2, 3]))
    assert report.nodes == (0, 1, 2, 3)
    assert report.old_edges == ((1, 2), (2, 3))
    with pytest.raises(FrozenInstanceError):
        report.predicted_response = Q(7)
