"""Independent four-branch controls for a two-contact composition bound.

Closed linear solutions and static row identities require no numerical
trajectory integration. The symmetric certificate bounds magnitude;
analytical signs are separate.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product
from math import factorial

import mpmath
import pytest

from tests.physics.test_sine_bridge_exchange import _mp
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_preparation as owner


def _arguments(**changes):
    values = dict(
        first_contact_weight_bounds=(Q(1, 2), Q(3, 4)),
        second_contact_weight_bounds=(Q(1, 8), Q(1, 4)),
        gamma_bounds=(Q(1, 4), Q(1, 3)),
        capacity_bounds=(Q(949, 1000), Q(1051, 1000)),
        initial_form_upper_bound=Q(1, 2),
        weighted_degree_upper_bound=3,
        elapsed_time=Q(1, 100),
        reading_error_bounds=(Q(1, 10**7),) * 4,
        inverse_capacity_reading_error_bounds=(Q(1, 10**7),) * 4,
        old_edge_chart_margins=(Q(1, 4),) * 3,
    )
    values.update(changes)
    return values


def _bound(**changes):
    return owner._prepared_two_contact_composition(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("composition controls cannot execute events or flows")

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
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _linear_rows(state, edges, capacity, *, context=None):
    result = [0] * len(state)
    for a, b, weight in edges:
        if context is not None:
            weight = _mp(context, weight)
        current = weight * (state[b] - state[a]) / 4
        left = capacity[a] if context is None else _mp(context, capacity[a])
        right = capacity[b] if context is None else _mp(context, capacity[b])
        result[a] += left * current
        result[b] -= right * current
    return tuple(result)


def _support(layout):
    old = ((0, 1, Q(1)),) if layout == "internal_path" else ()
    first = ((0, 2, Q(1)),)
    second = ((0 if layout == "shared" else 1, 3, Q(1)),)
    return old, old + first, old + second, old + first + second


def _linear_states(context, layout, time, *, heterogeneous=False):
    """Exact heat eigenmode formulas and their time derivatives, branchwise."""
    zero, one = context.mpf(0), context.mpf(1)
    initial, stationary = (zero, zero, one, one), (zero,) * 4
    if heterogeneous:
        assert layout == "shared"
        q, s = context.exp(-3 * time / 4), context.exp(-5 * time / 4)
        first = (2 * (1 - q) / 3, zero, (2 + q) / 3, one)
        second = (first[0], zero, one, first[2])
        joined = (4 * (1 - s) / 5, zero, (4 + s) / 5, (4 + s) / 5)
        return (
            (initial, first, second, joined),
            (
                stationary,
                (q / 2, zero, -q / 4, zero),
                (q / 2, zero, zero, -q / 4),
                (s, zero, -s / 4, -s / 4),
            ),
        )
    r, q, s = context.exp(-time / 4), context.exp(-time / 2), context.exp(-3 * time / 4)
    pair, pair_rate = (1 - q) / 2, q / 4
    first = (pair, zero, 1 - pair, one)
    first_rate = (pair_rate, zero, -pair_rate, zero)
    if layout == "shared":
        second = (pair, zero, one, 1 - pair)
        second_rate = (pair_rate, zero, zero, -pair_rate)
        joined = (2 * (1 - s) / 3, zero, (2 + s) / 3, (2 + s) / 3)
        joined_rate = (s / 2, zero, -s / 4, -s / 4)
    else:
        second = (zero, pair, one, 1 - pair)
        second_rate = (zero, pair_rate, zero, -pair_rate)
        joined = (pair, pair, 1 - pair, 1 - pair)
        joined_rate = (pair_rate, pair_rate, -pair_rate, -pair_rate)
        if layout == "internal_path":
            middle, leaf, source = (
                (1 - s) / 3,
                (2 - 3 * r + s) / 6,
                (2 + 3 * r + s) / 6,
            )
            middle_rate, leaf_rate, source_rate = s / 4, (r - s) / 8, -(r + s) / 8
            first, second = (middle, leaf, source, one), (leaf, middle, one, source)
            first_rate, second_rate = (middle_rate, leaf_rate, source_rate, zero), (
                leaf_rate,
                middle_rate,
                zero,
                source_rate,
            )
    return (initial, first, second, joined), (
        stationary,
        first_rate,
        second_rate,
        joined_rate,
    )


@pytest.mark.parametrize("layout", ("shared", "separate", "internal_path"))
@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_exact_linear_layouts_separate_port_overlap_from_internal_communication(
    layout, duration, mp, record_property
):
    report = _bound(
        first_contact_weight_bounds=(1, 1),
        second_contact_weight_bounds=(1, 1),
        gamma_bounds=(0, 0),
        capacity_bounds=(1, 1),
        initial_form_upper_bound=1,
        weighted_degree_upper_bound=2,
        elapsed_time=duration,
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
        old_edge_chart_margins=None,
    )
    supports = _support(layout)
    states, rates = _linear_states(mp, layout, _mp(mp, duration))
    tolerance = mp.mpf("1e-130")
    for edges, state, rate in zip(supports, states, rates):
        degree = [Q(0)] * 4
        for a, b, weight in edges:
            degree[a] += weight
            degree[b] += weight
        assert max(degree) <= 2
        actual = _linear_rows(state, edges, (Q(1),) * 4, context=mp)
        assert max(abs(a - b) for a, b in zip(actual, rate)) < tolerance
        assert abs(sum(state) - 2) < tolerance
    means = tuple((state[0] + state[1]) / 2 for state in states)
    defect = means[3] - means[1] - means[2] + means[0]
    r = mp.exp(-_mp(mp, duration) / 4)
    expected = {
        "shared": -((1 - r) ** 2) * (1 + 2 * r) / 6,
        "separate": mp.mpf(0),
        "internal_path": -((1 - r) ** 3) / 6,
    }[layout]
    assert abs(defect - expected) < tolerance
    assert report.mixed_state_defect_upper_bound == duration**2 / 4
    lower, upper = report.exact_true_defect_bounds
    assert _mp(mp, lower) <= defect <= _mp(mp, upper)
    full_defect = tuple(
        states[3][i] - states[1][i] - states[2][i] + states[0][i] for i in range(4)
    )
    assert (
        max(map(abs, full_defect))
        <= _mp(mp, report.mixed_state_defect_upper_bound) + tolerance
    )
    assert (
        report.inverse_capacity_exact_true_defect_bounds
        == report.exact_true_defect_bounds
    )
    assert lower <= 0 <= upper
    if duration and layout != "separate":
        assert defect < 0
    else:
        assert abs(defect) < tolerance
    # A zero actual defect can lie in a nonzero sufficient envelope. The flag
    # records only a proved zero envelope, never an inferred observed zero.
    assert report.exact_mixed_zero == (duration == 0)
    record_property("layout", layout)
    record_property("duration", str(duration))
    record_property("analytic_defect", mp.nstr(defect, 24))
    record_property("certified_absolute_bound", str(upper))


def test_rational_generator_products_expose_the_first_mixed_coefficients():
    initial, capacities = (Q(0), Q(0), Q(1), Q(1)), (Q(1),) * 4
    coefficients = {}
    for layout in ("shared", "separate", "internal_path"):
        branches = _support(layout)
        derivatives = [initial] * 4
        values = []
        for order in range(1, 4):
            derivatives = [
                _linear_rows(state, edges, capacities)
                for state, edges in zip(derivatives, branches)
            ]
            means = tuple(sum(state[:2]) / 2 for state in derivatives)
            values.append(
                (means[3] - means[1] - means[2] + means[0]) / factorial(order)
            )
        coefficients[layout] = tuple(values)
    assert coefficients["shared"][:2] == (0, Q(-1, 32))
    assert coefficients["separate"] == (0, 0, 0)
    assert coefficients["internal_path"] == (0, 0, Q(-1, 384))
    # Empty-background separate contacts annihilate each other's entire
    # coordinate subspaces, not only this particular initial condition.
    first, second = _support("separate")[1:3]
    for index in range(4):
        basis = tuple(Q(i == index) for i in range(4))
        assert (
            _linear_rows(_linear_rows(basis, first, capacities), second, capacities)
            == (0,) * 4
        )
        assert (
            _linear_rows(_linear_rows(basis, second, capacities), first, capacities)
            == (0,) * 4
        )


def test_equal_separate_contact_readings_do_not_determine_joint_response(mp):
    observations = {}
    for layout in ("shared", "separate", "internal_path"):
        states, _ = _linear_states(mp, layout, mp.mpf(1) / 4)
        observations[layout] = tuple(sum(state[:2]) / 2 for state in states)
    shared, separate, path = (
        observations[name] for name in ("shared", "separate", "internal_path")
    )
    assert shared[:3] == separate[:3]
    assert shared[3] < separate[3]
    # An internal path retains the initial scalar summaries but also changes
    # finite single-contact readings; only the first two layouts match those.
    assert path[0] == shared[0] == 0
    assert path[1] != shared[1] and path[2] != shared[2]


@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_heterogeneous_star_distinguishes_arithmetic_and_fixed_inverse_capacity_mean(
    duration, mp
):
    capacities = (Q(2), Q(1), Q(1), Q(1))
    mass = 1 / capacities[0] + 1 / capacities[1]
    assert mass == Q(3, 2)
    report = _bound(
        first_contact_weight_bounds=(1, 1),
        second_contact_weight_bounds=(1, 1),
        gamma_bounds=(0, 0),
        capacity_bounds=(1, 2),
        initial_form_upper_bound=1,
        weighted_degree_upper_bound=2,
        elapsed_time=duration,
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
        old_edge_chart_margins=None,
    )
    states, rates = _linear_states(mp, "shared", _mp(mp, duration), heterogeneous=True)
    for edges, state, rate in zip(_support("shared"), states, rates):
        actual = _linear_rows(state, edges, capacities, context=mp)
        assert max(abs(a - b) for a, b in zip(actual, rate)) < mp.mpf("1e-130")
        assert abs(
            sum(value / _mp(mp, capacity) for value, capacity in zip(state, capacities))
            - 2
        ) < mp.mpf("1e-130")
    ordinary = tuple(sum(state[:2]) / 2 for state in states)
    weighted = tuple((state[0] / 2 + state[1]) / _mp(mp, mass) for state in states)
    arithmetic = ordinary[3] - ordinary[1] - ordinary[2] + ordinary[0]
    inverse = weighted[3] - weighted[1] - weighted[2] + weighted[0]
    time = _mp(mp, duration)
    expected = 2 * (1 - mp.exp(-5 * time / 4)) / 5 - 2 * (1 - mp.exp(-3 * time / 4)) / 3
    assert abs(arithmetic - expected) < mp.mpf("1e-130")
    assert abs(inverse - 2 * arithmetic / 3) < mp.mpf("1e-130")
    assert report.mixed_state_defect_upper_bound == duration**2
    for value in (arithmetic, inverse):
        assert (
            _mp(mp, report.exact_true_defect_bounds[0])
            <= value
            <= _mp(mp, report.exact_true_defect_bounds[1])
        )


@pytest.mark.parametrize("gamma", (Q(0), Q(1), Q(2)))
def test_complete_row_jacobians_retain_form_phase_feedback_at_shared_ports(gamma):
    # At zero phase, sin'=1. This is the actual full eight-coordinate
    # edge Jacobian, independently summed from the two declared nodal rows.
    def jacobian(vector, edge):
        a, b = edge
        output = [Q(0)] * 8
        dx, dtheta = (vector[b] - vector[a]) / 4, (vector[4 + b] - vector[4 + a]) / 4
        output[a], output[b] = dx + gamma * dtheta, -dx - gamma * dtheta
        output[4 + a], output[4 + b] = -gamma * dx, gamma * dx
        return tuple(output)

    initial = (Q(0), Q(0), Q(1), Q(1), Q(0), Q(0), Q(0), Q(0))
    left = jacobian(jacobian(initial, (0, 2)), (0, 3))
    right = jacobian(jacobian(initial, (0, 3)), (0, 2))
    mixed = tuple(a + b for a, b in zip(left, right))
    assert sum(mixed[:2]) / 2 == (gamma**2 - 1) / 16
    assert sum(mixed[4:6]) / 2 == gamma / 16
    # A cancelling second derivative at gamma1 is not a finite zero defect.


def test_sine_hessian_bound_keeps_the_complete_weighted_star_population(mp):
    gamma, maximum, degree = Q(2), Q(2), Q(2)
    phase = (mp.pi / 2, mp.mpf(0), mp.mpf(0))
    q = r = (-1, 1, 1)
    curvature = [mp.mpf(0)] * 3
    for a, b in ((0, 1), (0, 2)):
        contribution = (
            -_mp(mp, gamma * maximum)
            * mp.sin(phase[b] - phase[a])
            * (q[b] - q[a])
            * (r[b] - r[a])
            / 4
        )
        curvature[a] += contribution
        curvature[b] -= contribution
    assert abs(max(map(abs, curvature)) - _mp(mp, gamma * maximum * degree)) < mp.mpf(
        "1e-130"
    )


@pytest.mark.parametrize("gamma", ((0, 0), (Q(1, 4), Q(1, 3)), (1, 2)))
def test_first_sensitivity_polynomial_and_its_square_give_independent_integrals(
    gamma, mp
):
    report = _bound(gamma_bounds=gamma)
    maximum, g, degree, initial, h = (
        Q(1051, 1000),
        Q(gamma[1]),
        Q(3),
        Q(1, 2),
        Q(1, 100),
    )
    # Sum the bridge's form and phase forcing rows before integrating.
    form_ceiling = (initial, maximum * degree * g / 4)
    form_forcing = (maximum * (2 * initial + g) / 4, maximum * form_ceiling[1] / 2)
    phase_forcing = tuple(maximum * g * value / 2 for value in form_ceiling)
    polynomial = (
        Q(0),
        form_forcing[0] + phase_forcing[0],
        (form_forcing[1] + phase_forcing[1]) / 2,
    )
    square = tuple(
        sum(
            (
                a * b
                for i, a in enumerate(polynomial)
                for j, b in enumerate(polynomial)
                if i + j == power
            ),
            Q(0),
        )
        for power in range(5)
    )

    def integral(coefficients):
        return sum(
            (
                value * h ** (power + 1) / (power + 1)
                for power, value in enumerate(coefficients)
            ),
            Q(0),
        )

    growth = report.difference_growth_upper_bound
    assert report.difference_forcing_constant_upper_bound == polynomial[1]
    assert report.difference_forcing_linear_upper_bound == 2 * polynomial[2]
    assert report.difference_growth_exponent == g * maximum * degree * h / 2
    exp_reference = mp.exp(_mp(mp, report.difference_growth_exponent))
    assert _mp(mp, growth) >= exp_reference
    assert _mp(mp, growth) - exp_reference < mp.mpf("1e-80")
    assert report.parameter_sensitivity_upper_bound == growth * sum(
        value * h**power for power, value in enumerate(polynomial)
    )
    assert report.integrated_parameter_sensitivity_upper_bound == growth * integral(
        polynomial
    )
    assert (
        report.integrated_squared_parameter_sensitivity_upper_bound
        == growth**2 * integral(square)
    )
    contact = maximum * (1 + g) * growth**2 * integral(polynomial)
    curvature = g * maximum * degree * growth**3 * integral(square)
    assert report.contact_derivative_mixed_coefficient_upper_bound == contact
    assert report.sine_curvature_mixed_coefficient_upper_bound == curvature
    assert report.mixed_sensitivity_upper_bound == contact + curvature
    assert report.mixed_state_defect_upper_bound == Q(3, 4) * Q(1, 4) * (
        contact + curvature
    )


@pytest.mark.parametrize("first,second", ((0, 0), (0, Q(1, 2)), (Q(1, 8), 0)))
def test_zero_either_contact_removes_true_defect_but_retains_each_noise_floor(
    first, second
):
    arithmetic, weighted = (Q(1, 1000), Q(2, 1000), Q(3, 1000), Q(4, 1000)), (
        Q(4, 1000),
        Q(3, 1000),
        Q(2, 1000),
        Q(2, 1000),
    )
    report = _bound(
        first_contact_weight_bounds=(0, first),
        second_contact_weight_bounds=(0, second),
        reading_error_bounds=arithmetic,
        inverse_capacity_reading_error_bounds=weighted,
    )
    assert (
        report.form_displacement_upper_bound > 0
        and report.phase_displacement_upper_bound > 0
    )
    assert report.mixed_sensitivity_upper_bound > 0
    assert (
        report.contact_weight_product_upper_bound
        == report.mixed_state_defect_upper_bound
        == 0
    )
    assert report.exact_mixed_zero is True
    for prefix, radii in (("", arithmetic), ("inverse_capacity_", weighted)):
        assert getattr(report, prefix + "exact_true_defect_bounds") == (0, 0)
        assert getattr(report, prefix + "true_defect_bounds") == I(0)
        assert getattr(report, prefix + "exact_observed_defect_bounds") == (
            -sum(radii),
            sum(radii),
        )


@pytest.mark.parametrize(
    "first,second",
    (
        (Q(1, 2), Q(1, 8)),
        (Q(1, 8), Q(1, 8)),
        (Q(1, 2**200), Q(1, 8)),
        (Q(1, 2**200), Q(1, 2**200)),
    ),
)
def test_exact_product_scaling_survives_subgrid_weights_and_display_rounding(
    first, second
):
    options = dict(
        first_contact_weight_bounds=(0, first),
        second_contact_weight_bounds=(0, second),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    report = _bound(**options)
    one_half = _bound(**dict(options, first_contact_weight_bounds=(0, first / 2)))
    both_half = _bound(
        **dict(
            options,
            first_contact_weight_bounds=(0, first / 2),
            second_contact_weight_bounds=(0, second / 2),
        )
    )
    assert report.mixed_state_defect_upper_bound > 0
    assert report.contact_weight_product_upper_bound == first * second
    assert (
        2 * one_half.mixed_state_defect_upper_bound
        == 4 * both_half.mixed_state_defect_upper_bound
        == report.mixed_state_defect_upper_bound
    )
    assert (
        tuple(2 * value for value in one_half.exact_true_defect_bounds)
        == report.exact_true_defect_bounds
    )
    assert (
        tuple(4 * value for value in both_half.exact_true_defect_bounds)
        == report.exact_true_defect_bounds
    )
    assert report.exact_true_defect_bounds[0] < 0 < report.exact_true_defect_bounds[1]
    assert not report.exact_mixed_zero
    if first < Q(1, 2**128):
        assert report.exact_true_defect_bounds[1] < Q(1, 2**128)
        assert report.true_defect_bounds.width >= Q(2, 2**128)
    assert report.true_defect_bounds.contains(I(*report.exact_true_defect_bounds))


def test_four_final_time_error_coefficients_and_missing_weighted_reading_are_explicit():
    arithmetic = (Q(1, 10**7), Q(2, 10**7), Q(3, 10**7), Q(4, 10**7))
    weighted = tuple(reversed(arithmetic))
    report = _bound(
        reading_error_bounds=arithmetic, inverse_capacity_reading_error_bounds=weighted
    )
    for prefix, radii in (("", arithmetic), ("inverse_capacity_", weighted)):
        corners = tuple(
            signs[0] * radii[0]
            - signs[1] * radii[1]
            - signs[2] * radii[2]
            + signs[3] * radii[3]
            for signs in product((-1, 1), repeat=4)
        )
        assert min(corners) == -sum(radii) and max(corners) == sum(radii)
        low, high = getattr(report, prefix + "exact_true_defect_bounds")
        assert getattr(report, prefix + "reading_error_upper_bound") == sum(radii)
        assert getattr(report, prefix + "exact_observed_defect_bounds") == (
            low + min(corners),
            high + max(corners),
        )
    absent = _bound(inverse_capacity_reading_error_bounds=None)
    assert (
        absent.inverse_capacity_exact_true_defect_bounds
        == absent.exact_true_defect_bounds
    )
    assert absent.inverse_capacity_exact_observed_defect_bounds is None
    assert absent.inverse_capacity_observed_defect_bounds is None
    assert absent.inverse_capacity_reading_error_upper_bound is None
    tiny = _bound(first_contact_weight_bounds=(Q(1, 2**200),) * 2)
    assert tiny.exact_true_defect_bounds[1] < tiny.reading_error_upper_bound
    assert tiny.exact_observed_defect_bounds[1] > tiny.reading_error_upper_bound


@pytest.mark.parametrize("common", (Q(1), Q(1, 4), Q(3, 2)))
@pytest.mark.parametrize("gamma", ((0, 0), (Q(1, 4), Q(1, 3)), (1, 2)))
def test_common_capacity_rescales_both_rows_under_one_clock(common, gamma):
    report = _bound(capacity_bounds=(common, common), gamma_bounds=gamma)
    unit = _bound(capacity_bounds=(1, 1), gamma_bounds=gamma, elapsed_time=common / 100)
    for field in (
        "form_upper_bound",
        "form_displacement_upper_bound",
        "phase_displacement_upper_bound",
        "difference_growth_exponent",
        "difference_growth_upper_bound",
        "parameter_sensitivity_upper_bound",
        "mixed_sensitivity_upper_bound",
        "mixed_state_defect_upper_bound",
        "exact_true_defect_bounds",
        "exact_observed_defect_bounds",
        "retention_margins",
    ):
        assert getattr(report, field) == getattr(unit, field)


def test_disjoint_contact_degree_needs_actual_association_not_a_sum_rule():
    report = _bound(
        first_contact_weight_bounds=(1, 1),
        second_contact_weight_bounds=(1, 1),
        weighted_degree_upper_bound=1,
        capacity_bounds=(1, 1),
        gamma_bounds=(0, 0),
        initial_form_upper_bound=1,
        old_edge_chart_margins=None,
    )
    joined = _support("separate")[-1]
    degree = [Q(0)] * 4
    for a, b, weight in joined:
        degree[a] += weight
        degree[b] += weight
    assert max(degree) == report.weighted_degree_upper_bound == 1
    assert (
        report.first_contact_weight_bounds[1] + report.second_contact_weight_bounds[1]
        > 1
    )


def test_zero_form_pure_diffusion_certifies_zero_even_with_two_positive_contacts():
    report = _bound(initial_form_upper_bound=0, gamma_bounds=(0, 0))
    assert report.contact_weight_product_upper_bound > 0
    assert report.mixed_state_defect_upper_bound == 0
    assert report.exact_true_defect_bounds == (0, 0)
    assert report.exact_mixed_zero


def test_chart_retention_uses_absolute_motion_separately_from_mixed_accuracy():
    baseline = _bound()
    drift = 2 * baseline.phase_displacement_upper_bound
    for margins, status in (
        (None, "not_requested"),
        ((0,), "indeterminate"),
        ((drift,), "indeterminate"),
        ((drift / 2,), "indeterminate"),
        ((drift + Q(1, 10**6),), "certified"),
    ):
        report = _bound(old_edge_chart_margins=margins)
        assert report.retention_status == status
        assert report.exact_true_defect_bounds == baseline.exact_true_defect_bounds
        assert (
            report.exact_observed_defect_bounds == baseline.exact_observed_defect_bounds
        )
    disconnected = _bound(
        first_contact_weight_bounds=(0, 0), old_edge_chart_margins=(0,)
    )
    assert (
        disconnected.exact_mixed_zero
        and disconnected.retention_status == "indeterminate"
    )


INVALID = (
    ("first_contact_weight_bounds", (-1, 1)),
    ("second_contact_weight_bounds", (-1, 0)),
    ("first_contact_weight_bounds", (2, 1)),
    ("second_contact_weight_bounds", (True, 1)),
    ("first_contact_weight_bounds", (0, float("nan"))),
    ("second_contact_weight_bounds", (0, float("inf"))),
    ("first_contact_weight_bounds", (0, 1, 2)),
    ("second_contact_weight_bounds", (0,)),
    ("first_contact_weight_bounds", {0, 1}),
    ("second_contact_weight_bounds", ("0", 1)),
    ("gamma_bounds", (-1, 1)),
    ("gamma_bounds", (2, 1)),
    ("gamma_bounds", (False, 1)),
    ("gamma_bounds", (0, float("nan"))),
    ("gamma_bounds", (0, 1j)),
    ("capacity_bounds", (0, 1)),
    ("capacity_bounds", (-1, 1)),
    ("capacity_bounds", (2, 1)),
    ("capacity_bounds", (1, True)),
    ("capacity_bounds", (1, float("inf"))),
    ("capacity_bounds", ("1", 2)),
    ("capacity_bounds", (1,)),
    ("initial_form_upper_bound", -1),
    ("initial_form_upper_bound", True),
    ("initial_form_upper_bound", float("nan")),
    ("weighted_degree_upper_bound", Q(1, 8)),
    ("weighted_degree_upper_bound", False),
    ("elapsed_time", -1),
    ("elapsed_time", True),
    ("elapsed_time", float("inf")),
    ("reading_error_bounds", (0, 0, 0)),
    ("reading_error_bounds", (0, 0, 0, -1)),
    ("reading_error_bounds", (0, 0, 0, True)),
    ("inverse_capacity_reading_error_bounds", (0,) * 5),
    ("inverse_capacity_reading_error_bounds", (0, 0, 0, float("inf"))),
    ("old_edge_chart_margins", ()),
    ("old_edge_chart_margins", (-1,)),
    ("old_edge_chart_margins", (False,)),
    ("old_edge_chart_margins", (0,) * 4097),
)


@pytest.mark.parametrize("name,value", INVALID)
def test_invalid_raw_premises_reject_before_interval_or_exponential_work(
    name, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid composition premise reached numerical work")

    monkeypatch.setattr(owner, "I", forbidden)
    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _bound(**{name: value})


def test_zero_product_does_not_bypass_growth_work_domain(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget zero-product request reached numerical work")

    monkeypatch.setattr(owner, "I", forbidden)
    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    with pytest.raises(ValueError, match="4096"):
        _bound(first_contact_weight_bounds=(0, 0), elapsed_time=10000)


def test_growth_work_boundary_is_inclusive_without_large_exponential_evaluation(
    monkeypatch,
):
    class BoundaryReached(Exception):
        pass

    def boundary(exponent):
        assert exponent == 4096
        raise BoundaryReached

    monkeypatch.setattr(owner, "_negative_exp_bounds", boundary)
    with pytest.raises(BoundaryReached):
        _bound(
            capacity_bounds=(1, 1),
            gamma_bounds=(1, 1),
            weighted_degree_upper_bound=2,
            elapsed_time=4096,
            first_contact_weight_bounds=(0, 0),
        )


def test_positive_subgrid_capacity_and_fixed_width_fraction_admission():
    tiny = Q(1, 2**200)
    report = _bound(capacity_bounds=(tiny, tiny), gamma_bounds=(0, 0))
    assert report.capacity_bounds == (tiny, tiny)
    assert report.mixed_state_defect_upper_bound > 0
    assert report.mixed_state_defect_upper_bound < Q(1, 2**400)
    np = pytest.importorskip("numpy")
    narrow = Q(np.int64(2**62), np.int64(2**62 + 1))
    exact = Q(2**62, 2**62 + 1)
    assert _bound(capacity_bounds=(narrow, narrow)) == _bound(
        capacity_bounds=(exact, exact)
    )


def test_inputs_are_copied_results_frozen_and_cached_verdicts_rejected():
    first, errors = [Q(1, 2), Q(3, 4)], [Q(1, 10**7)] * 4
    report = _bound(first_contact_weight_bounds=first, reading_error_bounds=errors)
    first[0], errors[-1] = -1, 100
    assert report.first_contact_weight_bounds == (Q(1, 2), Q(3, 4))
    assert report.reading_error_bounds == (Q(1, 10**7),) * 4
    with pytest.raises(FrozenInstanceError):
        report.exact_mixed_zero = True
    for invalid in (report, I(0, 1), {"lo": 0, "hi": 1, "certified": True}):
        with pytest.raises((TypeError, ValueError)):
            _bound(first_contact_weight_bounds=invalid)
    assert _bound(reading_error_bounds=errors).reading_error_upper_bound > 100


def test_ordered_iterables_stop_at_declared_work_boundaries():
    consumed = []

    def endless():
        for index in range(10000):
            consumed.append(index)
            yield Q(1)

    with pytest.raises(ValueError):
        _bound(first_contact_weight_bounds=endless())
    assert len(consumed) == 3
    consumed.clear()
    with pytest.raises(ValueError):
        _bound(reading_error_bounds=endless())
    assert len(consumed) == 5


@pytest.mark.parametrize(
    "missing",
    (
        "first_contact_weight_bounds",
        "second_contact_weight_bounds",
        "capacity_bounds",
        "gamma_bounds",
        "reading_error_bounds",
        "elapsed_time",
    ),
)
def test_required_law_contact_and_reading_premises_have_no_inferred_defaults(missing):
    values = _arguments()
    del values[missing]
    with pytest.raises(TypeError):
        owner._prepared_two_contact_composition(**values)
