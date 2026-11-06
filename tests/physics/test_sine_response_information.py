"""Prospective information controls for an original finite contact response.

Independent closed heat solutions and rational series supply the oracles.
No trajectory integration, event execution or retained response replay is used.
"""

from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import product, repeat
from math import factorial

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_response_information as owner


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
            result = result * self
        return result


def _decay(value):
    value = Q(value)
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(101)), Q(0))
    return _B(upper - value**101 / factorial(101), upper)


def _hyperbolic(time):
    """C=cosh(sqrt(2)*t/4), K=sinh(sqrt(2)*t/4)/sqrt(2)."""
    square = Q(time) ** 2 / 8
    even = sum((square**k / factorial(2 * k) for k in range(49)), Q(0))
    odd = sum((square**k / factorial(2 * k + 1) for k in range(49)), Q(0))
    even_tail = square**49 / factorial(98) / (1 - square / (99 * 100))
    odd_tail = square**49 / factorial(99) / (1 - square / (100 * 101))
    return _B(even, even + even_tail), Q(time) / 4 * _B(odd, odd + odd_tail)


def _on_solution(time, initial):
    """Full P4 solution, differentiated before comparing with edge rows."""
    x0, x1, x2, x3 = map(Q, initial)
    a, b = (x0 + x3) / 2, (x1 + x2) / 2
    p, q = (x0 - x3) / 2, (x1 - x2) / 2
    mean, contrast = (a + b) / 2, (a - b) / 2
    e = _decay(time / 2)
    c, k = _hyperbolic(time)
    symmetric = (mean + contrast * e, mean - contrast * e)
    symmetric_rate = (-contrast * e / 2, contrast * e / 2)
    anti = (e * (c * p + k * (p + q)), e * (c * q + k * (p - q)))
    anti_rate = (
        -anti[0] / 2 + e * (k * p / 2 + c * (p + q) / 4),
        -anti[1] / 2 + e * (k * q / 2 + c * (p - q) / 4),
    )
    state = (
        symmetric[0] + anti[0],
        symmetric[1] + anti[1],
        symmetric[1] - anti[1],
        symmetric[0] - anti[0],
    )
    rate = (
        symmetric_rate[0] + anti_rate[0],
        symmetric_rate[1] + anti_rate[1],
        symmetric_rate[1] - anti_rate[1],
        symmetric_rate[0] - anti_rate[0],
    )
    return state, rate


def _off_solution(time, initial):
    x0, x1, x2, x3 = map(Q, initial)
    mean = (x1 + x2 + x3) / 3
    odd = (x1 - x3) / 2 * _decay(time / 4)
    even = (x1 - 2 * x2 + x3) / 6 * _decay(3 * time / 4)
    return (
        (_B(x0), mean + odd + even, mean - 2 * even, mean - odd + even),
        (_B(0), -odd / 4 - 3 * even / 4, 3 * even / 2, odd / 4 - 3 * even / 4),
    )


def _rows(state, edges, capacity=(Q(1),) * 4):
    rows = [_B(0) for _ in state]
    for a, b in edges:
        current = (state[b] - state[a]) / 4
        rows[a] += capacity[a] * current
        rows[b] -= capacity[b] * current
    return tuple(rows)


def _coefficients(time):
    result = []
    for j in range(4):
        initial = tuple(Q(i == j) for i in range(4))
        on, _ = _on_solution(time, initial)
        off, _ = _off_solution(time, initial)
        result.append((on[0] + on[1] - off[0] - off[1]) / 2)
    return tuple(result)


def _contains(endpoints, oracle):
    assert endpoints[0] <= oracle.lo <= oracle.hi <= endpoints[1]
    assert endpoints[1] - endpoints[0] < Q(1, 10**30)


def _zero(oracle):
    assert oracle.lo <= 0 <= oracle.hi
    assert oracle.hi - oracle.lo < Q(1, 10**100)


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
        retained_initial_form=(Q(0), Q(0)),
        hidden_prior_form_bounds=((Q(-1), Q(1)),) * 2,
        measured_nodes=(),
        measurement_values=(),
        measurement_error_bounds=(),
        elapsed_time=Q(1, 4),
        mean_kind="arithmetic",
        reading_error_bound=Q(0),
        tolerance=Q(1, 10**6),
    )
    values.update(changes)
    return values


def _report(**changes):
    return owner._prepared_sine_response_information(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("information controls cannot integrate or execute events")

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
    return {
        "none": _report(),
        "near": _report(
            measured_nodes=(2,),
            measurement_values=(Q(1, 2),),
            measurement_error_bounds=(Q(1, 100),),
        ),
        "far": _report(
            measured_nodes=(3,),
            measurement_values=(Q(1, 2),),
            measurement_error_bounds=(Q(1, 100),),
        ),
    }


@pytest.mark.parametrize("time", [Q(0), Q(1, 100), Q(1, 4)])
@pytest.mark.parametrize("initial", [(0, 0, 1, 0), (0, 0, 0, 1), (2, -1, 1, -2)])
def test_independent_closed_solutions_satisfy_complete_heat_rows(time, initial):
    for solver, edges in (
        (_on_solution, ((0, 1), (1, 2), (2, 3))),
        (_off_solution, ((1, 2), (2, 3))),
    ):
        state, rate = solver(time, initial)
        for derivative, row in zip(rate, _rows(state, edges)):
            _zero(derivative - row)
        _zero(sum(state, _B(0)) - sum(initial))
        if time == 0:
            for value, expected in zip(state, initial):
                assert value == _B(expected)
        # gamma=0 makes the complete phase row identically zero, whatever the lift.
        assert tuple(Q(0) * row.lo for row in _rows(state, edges)) == (Q(0),) * 4


def test_closed_covector_and_prospective_duhamel_brackets(p4_reports, record_property):
    report = p4_reports["none"]
    coefficient = _coefficients(Q(1, 4))
    r, e = _decay(Q(1, 16)), _decay(Q(1, 8))
    c, k = _hyperbolic(Q(1, 4))
    closed2 = Q(1, 12) + r**3 / 6 - e * c / 4
    closed3 = Q(1, 12) + r / 4 - r**3 / 12 - e * c / 4 - e * k / 2
    _zero(coefficient[2] - closed2)
    _zero(coefficient[3] - closed3)
    for actual, expected in zip(report.response_coefficient_bounds, coefficient):
        _contains(actual, expected)
    assert Q(7, 393216) < coefficient[2].lo <= coefficient[2].hi < Q(1, 49152)
    assert Q(7, 25165824) < coefficient[3].lo <= coefficient[3].hi < Q(1, 3145728)
    record_property(
        "near_hidden_response_coefficient",
        (_display(coefficient[2].lo), _display(coefficient[2].hi)),
    )
    record_property(
        "far_hidden_response_coefficient",
        (_display(coefficient[3].lo), _display(coefficient[3].hi)),
    )


def test_one_associated_coordinate_is_necessary_and_sufficient_in_declared_class(
    p4_reports, record_property
):
    none, near, far = (p4_reports[name] for name in ("none", "near", "far"))
    tolerance = Q(1, 10**6)
    assert none.status == far.status == "insufficient_information"
    assert none.information_error_bounds[0] > tolerance
    assert far.information_error_bounds[0] > tolerance
    assert near.status == "certified"
    assert near.forecast_error_upper_bound < tolerance
    assert near.information_error_bounds[1] < Q(41, 78643200)
    assert near.computational_center_error_upper_bound < Q(1, 10**30)
    assert near.response_sign == "positive"
    assert near.forecast_response_bounds[0] > Q(661, 78643200)
    assert none.measurement_cost == 0
    assert near.measurement_cost == far.measurement_cost == 1
    assert near.hidden_posterior_form_bounds == (
        (Q(49, 100), Q(51, 100)),
        (Q(-1), Q(1)),
    )
    assert far.hidden_posterior_form_bounds == ((Q(-1), Q(1)), (Q(49, 100), Q(51, 100)))
    record_property("fixed_response_tolerance", str(tolerance))
    record_property(
        "near_information_error_bounds",
        tuple(_display(x) for x in near.information_error_bounds),
    )
    record_property("near_implementable_forecast", _display(near.predicted_response))
    record_property(
        "near_total_forecast_error", _display(near.forecast_error_upper_bound)
    )
    record_property(
        "near_response_interval",
        tuple(_display(x) for x in near.forecast_response_bounds),
    )
    record_property(
        "no_measurement_lower_error", _display(none.information_error_bounds[0])
    )
    record_property(
        "far_measurement_lower_error", _display(far.information_error_bounds[0])
    )
    record_property(
        "evolved_coordinates",
        "0; full support and two propagators remain static preparation",
    )


def test_exact_hidden_inventory_does_not_resolve_associated_contrast(record_property):
    a = _coefficients(Q(1, 4))
    response = a[2] - a[3]
    assert response.lo > Q(55, 3145728) > Q(1, 10**6)
    witnesses = ((0, 0, 1, -1), (0, 0, -1, 1))
    readings = []
    for initial in witnesses:
        assert initial[:2] == (0, 0) and initial[2] + initial[3] == 0
        on, _ = _on_solution(Q(1, 4), initial)
        off, _ = _off_solution(Q(1, 4), initial)
        readings.append((on[0] + on[1] - off[0] - off[1]) / 2)
    _zero(readings[0] - response)
    _zero(readings[1] + response)
    assert readings[0].lo > 0 > readings[1].hi
    record_property(
        "equal_inventory_half_diameter", (_display(response.lo), _display(response.hi))
    )


def _signed_coefficients():
    """Path 2--0--1--3 on; two independent pairs 2--0 and 1--3 off."""
    order = (2, 0, 1, 3)
    result = []
    e = _decay(Q(1, 8))
    for column in range(4):
        initial = tuple(Q(i == column) for i in range(4))
        state, _ = _on_solution(Q(1, 4), tuple(initial[i] for i in order))
        off0 = (initial[0] + initial[2]) / 2 + (initial[0] - initial[2]) / 2 * e
        result.append(state[1] - off0)
    return tuple(result)


def test_signed_actual_covector_box_corners_attain_minimax_and_clipped_posterior():
    report = _report(
        old_edges=((0, 2), (1, 3)),
        observation_nodes=(0,),
        retained_initial_form=(Q(2), Q(-1)),
        hidden_prior_form_bounds=((Q(-2), Q(1)), (Q(1), Q(4))),
        measured_nodes=(2,),
        measurement_values=(Q(-3, 2),),
        measurement_error_bounds=(Q(1),),
        tolerance=Q(1),
    )
    a = _signed_coefficients()
    assert a[2].hi < 0 < a[3].lo
    for production, expected in zip(report.response_coefficient_bounds, a):
        _contains(production, expected)
    assert report.hidden_posterior_form_bounds == ((Q(-2), Q(-1, 2)), (Q(1), Q(4)))
    center = sum(
        (coefficient * value for coefficient, value in zip(a, report.posterior_center)),
        _B(0),
    )
    rho = -a[2] * Q(3, 4) + a[3] * Q(3, 2)
    assert (
        report.information_error_bounds[0]
        <= rho.lo
        <= rho.hi
        <= report.information_error_bounds[1]
    )
    for hidden in product((Q(-2), Q(-1, 2)), (Q(1), Q(4))):
        response = sum(
            (
                coefficient * value
                for coefficient, value in zip(a, (Q(2), Q(-1), *hidden))
            ),
            _B(0),
        )
        assert (
            report.forecast_response_bounds[0]
            <= response.lo
            <= response.hi
            <= report.forecast_response_bounds[1]
        )
    maximum = 2 * a[0] - a[1] - 2 * a[2] + 4 * a[3]
    minimum = 2 * a[0] - a[1] - a[2] / 2 + a[3]
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
def test_heterogeneous_capacity_generator_and_actual_mean_projection(mean_kind):
    capacity = (Q(1), Q(2), Q(3), Q(4))
    report = _report(capacity=capacity, elapsed_time=Q(1, 16), mean_kind=mean_kind)
    expected_weight = (
        (Q(1, 2), Q(1, 2), Q(0), Q(0))
        if mean_kind == "arithmetic"
        else (Q(2, 3), Q(1, 3), Q(0), Q(0))
    )
    assert report.observation_weights == expected_weight
    for generator, edges in (
        (report.off_generator, ((1, 2), (2, 3))),
        (report.on_generator, ((0, 1), (1, 2), (2, 3))),
    ):
        for column in range(4):
            basis = tuple(_B(Q(i == column)) for i in range(4))
            rows = _rows(basis, edges, capacity)
            assert tuple(row.lo for row in rows) == tuple(
                generator[i][column] for i in range(4)
            )
        assert all(sum(row) == 0 for row in generator)
        assert all(
            sum(generator[i][j] / capacity[i] for i in range(4)) == 0 for j in range(4)
        )
    for j, interval in enumerate(report.response_coefficient_bounds):
        lower = sum(
            expected_weight[i]
            * (report.on_kernel.lower[i][j] - report.off_kernel.upper[i][j])
            for i in range(4)
        )
        upper = sum(
            expected_weight[i]
            * (report.on_kernel.upper[i][j] - report.off_kernel.lower[i][j])
            for i in range(4)
        )
        assert interval == (lower, upper)


def test_clock_and_source_relabeling_preserve_associated_information(p4_reports):
    original = p4_reports["near"]
    scaled = _report(
        capacity=(Q(3),) * 4,
        elapsed_time=Q(1, 12),
        measured_nodes=(2,),
        measurement_values=(Q(1, 2),),
        measurement_error_bounds=(Q(1, 100),),
    )
    for name in (
        "response_coefficient_bounds",
        "information_error_bounds",
        "predicted_response",
        "forecast_error_upper_bound",
        "status",
        "response_sign",
    ):
        assert getattr(scaled, name) == getattr(original, name)
    # Reorder both full and retained populations; each primitive follows its own association.
    relabeled = _report(
        nodes=("hidden_far", "right", "hidden_near", "left"),
        old_edges=(("hidden_near", "right"), ("hidden_far", "hidden_near")),
        contact=("right", "left"),
        retained_nodes=("right", "left"),
        observation_nodes=("left", "right"),
        measured_nodes=("hidden_near",),
        measurement_values=(Q(1, 2),),
        measurement_error_bounds=(Q(1, 100),),
    )
    assert relabeled.response_coefficient_bounds == tuple(
        original.response_coefficient_bounds[i] for i in (3, 1, 2, 0)
    )
    for name in (
        "information_error_bounds",
        "predicted_response",
        "forecast_error_upper_bound",
        "status",
        "response_sign",
    ):
        assert getattr(relabeled, name) == getattr(original, name)


@pytest.mark.parametrize("zero", ["time", "contact"])
def test_structural_zero_shortcuts_follow_full_admission(monkeypatch, zero):
    def forbidden(*args, **kwargs):
        pytest.fail("admitted structural zero should not calculate an exponential")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    report = _report(
        **({"elapsed_time": Q(0)} if zero == "time" else {"contact_weight": Q(0)})
    )
    assert report.off_kernel is report.on_kernel is None
    assert report.response_coefficient_bounds == ((Q(0), Q(0)),) * 4
    assert report.information_error_bounds == (Q(0), Q(0))
    assert report.predicted_response == report.forecast_error_upper_bound == 0
    assert report.status == "certified" and report.response_sign == "zero"


def test_disconnected_hidden_population_has_zero_information_radius():
    report = _report(
        old_edges=((2, 3),),
        old_conductances=(Q(1),),
        retained_initial_form=(Q(1), Q(-1)),
    )
    assert report.response_coefficient_bounds[2:] == ((Q(0), Q(0)),) * 2
    assert report.information_error_bounds == (Q(0), Q(0))
    assert report.status == "certified"


def test_known_posterior_separates_computation_and_future_noise():
    changes = dict(
        measured_nodes=(2, 3),
        measurement_values=(Q(1, 2), Q(-1, 4)),
        measurement_error_bounds=(Q(0), Q(0)),
    )
    precise = _report(**changes)
    noisy = _report(**changes, reading_error_bound=Q(1, 1000))
    assert (
        precise.information_error_bounds
        == noisy.information_error_bounds
        == (Q(0), Q(0))
    )
    assert precise.status == "certified"
    assert noisy.status == "unresolved"
    assert noisy.forecast_error_upper_bound == precise.forecast_error_upper_bound + Q(
        1, 1000
    )
    assert noisy.predicted_response == precise.predicted_response
    assert noisy.response_sign == "unresolved"


def test_sign_crossing_coefficient_uses_zero_lower_absolute_value():
    # Hidden node2 feeds retained nodes symmetrically. Contact01 acts only on
    # their antisymmetric mode, so this hidden coefficient is exactly zero.
    # Independent branch enclosures cannot declare exact rank from midpoints.
    report = _report(old_edges=((0, 2), (1, 2)))
    lower, upper = report.response_coefficient_bounds[2]
    assert lower < 0 < upper
    assert report.posterior_radii[2] == 1
    assert report.information_error_bounds[0] == 0
    assert report.information_error_bounds[1] > 0
    assert report.status == "certified"


def test_touching_measurement_intersection_and_input_order_keep_node_association():
    report = _report(
        measured_nodes=(3, 2),
        measurement_values=(Q(-2), Q(2)),
        measurement_error_bounds=(Q(1), Q(1)),
    )
    assert report.hidden_posterior_form_bounds == ((Q(1), Q(1)), (Q(-1), Q(-1)))
    assert report.posterior_center == (Q(0), Q(0), Q(1), Q(-1))
    assert report.posterior_radii == (Q(0),) * 4
    assert report.measurement_cost == 2
    assert report.information_error_bounds == (Q(0), Q(0))
    assert report.response_sign == "positive"


def test_contact_adds_to_an_old_edge_without_silently_replacing_it():
    report = _report(
        old_edges=((0, 1), (1, 2), (2, 3)),
        old_conductances=(Q(2), Q(1), Q(1)),
        elapsed_time=Q(1, 16),
    )
    assert report.off_generator[0][1] == Q(1, 2)
    assert report.on_generator[0][1] == Q(3, 4)
    assert report.on_generator[0][0] == -Q(3, 4)


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
        {"old_conductances": (Q(1),)},
        {"capacity": (Q(1), Q(0), Q(1), Q(1))},
        {"capacity": (Q(1), float("inf"), Q(1), Q(1))},
        {"contact": (0, 0)},
        {"contact": (0, 9)},
        {"contact_weight": Q(-1)},
        {"contact_weight": True},
        {"retained_nodes": (0, 1, 2, 3)},
        {"retained_nodes": (0, 2)},
        {"retained_nodes": (0, 1, 1)},
        {"observation_nodes": ()},
        {"observation_nodes": (2,)},
        {"observation_nodes": (0, 0)},
        {"retained_initial_form": (Q(0),)},
        {"retained_initial_form": (True, Q(0))},
        {"hidden_prior_form_bounds": ((Q(1), Q(-1)), (Q(-1), Q(1)))},
        {"hidden_prior_form_bounds": ((Q(-1), Q(1)),)},
        {"hidden_prior_form_bounds": ((float("nan"), Q(1)), (Q(-1), Q(1)))},
        {
            "measured_nodes": (0,),
            "measurement_values": (Q(0),),
            "measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2, 2),
            "measurement_values": (Q(0), Q(0)),
            "measurement_error_bounds": (Q(0), Q(0)),
        },
        {
            "measured_nodes": (2,),
            "measurement_values": (),
            "measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2,),
            "measurement_values": (Q(2),),
            "measurement_error_bounds": (Q(0),),
        },
        {
            "measured_nodes": (2,),
            "measurement_values": (Q(0),),
            "measurement_error_bounds": (Q(-1),),
        },
        {
            "measured_nodes": (2,),
            "measurement_values": (True,),
            "measurement_error_bounds": (Q(0),),
        },
        {"elapsed_time": Q(-1)},
        {"elapsed_time": True},
        {"elapsed_time": Q(3)},
        {"mean_kind": "arbitrary_response_weighted"},
        {"reading_error_bound": Q(-1)},
        {"tolerance": True},
        {"tolerance": Q(-1)},
        {"tolerance": float("nan")},
    ],
)
def test_invalid_primitives_reject_before_either_kernel_even_at_zero_contact(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached a response kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _report(**changes)
    if "elapsed_time" not in changes and "contact_weight" not in changes:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            _report(**{**changes, "contact_weight": Q(0)})


def test_source_and_observation_domains_are_admitted_before_zero_shortcut(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inadmissible branch reached exponential")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    with pytest.raises(ValueError):
        _report(contact_weight=Q(0), elapsed_time=Q(3))
    # Off branch is admitted; on branch exceeds the shift budget. Neither
    # kernel may run until both domains have passed.
    with pytest.raises(ValueError):
        _report(contact_weight=Q(5), elapsed_time=Q(1))
    with pytest.raises(ValueError):
        _report(
            elapsed_time=Q(0),
            measured_nodes=(3,),
            measurement_values=(Q(4),),
            measurement_error_bounds=(Q(1),),
        )


def test_report_is_immutable_and_primitives_normalize_ordered_iterables():
    report = _report(
        nodes=[0, 1, 2, 3], old_edges=([1, 2], [2, 3]), retained_nodes=[0, 1]
    )
    assert report.nodes == (0, 1, 2, 3)
    assert report.old_edges == ((1, 2), (2, 3))
    with pytest.raises(FrozenInstanceError):
        report.predicted_response = Q(100)
