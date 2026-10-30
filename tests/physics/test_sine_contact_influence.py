"""Independent support-associated finite contact controls without integration.

Complete closed linear states and their derivatives validate the analytic
oracles against the supplied D/4 row. Positive-gamma controls use exact local
directional algebra; they do not infer finite signs from an onset coefficient.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics import _comparison_flow as comparison
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_contact_support as owner
from tnfr.physics import _sine_preparation as scalar_owner


def _arguments(**changes):
    values = dict(
        nodes=(0, 1, 2, 3),
        old_edges=(),
        old_conductances=(),
        capacity=(Q(1),) * 4,
        first_contact=(0, 2),
        second_contact=(0, 3),
        observation_nodes=(0, 1),
        first_contact_weight_bounds=(Q(1), Q(1)),
        second_contact_weight_bounds=(Q(1), Q(1)),
        gamma_bounds=(Q(0), Q(0)),
        initial_form_upper_bound=Q(1),
        elapsed_time=Q(1, 4),
        reading_error_bounds=(Q(0),) * 4,
        inverse_capacity_reading_error_bounds=(Q(0),) * 4,
        old_edge_chart_margins=None,
    )
    values.update(changes)
    return values


def _bound(**changes):
    return owner._prepared_support_contact_composition(**_arguments(**changes))


def _mp(context, value):
    exact = Q(value)
    return context.mpf(int(exact.numerator)) / int(exact.denominator)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("support controls cannot execute events or flows")

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


def _linear_rows(state, edges, capacity, context):
    result = [context.mpf(0)] * len(state)
    for a, b, weight in edges:
        current = _mp(context, weight) * (state[b] - state[a]) / 4
        result[a] += _mp(context, capacity[a]) * current
        result[b] -= _mp(context, capacity[b]) * current
    return tuple(result)


def _branches(layout):
    old = ((0, 1, Q(1)),) if layout == "internal_path" else ()
    first, second = ((0, 2, Q(1)),), ((0 if layout == "shared" else 1, 3, Q(1)),)
    return old, old + first, old + second, old + first + second


def _closed_states(context, layout, time, heterogeneous=False):
    """Pair, star, and three-node path eigenmodes; no matrix exponential."""
    zero, one = context.mpf(0), context.mpf(1)
    initial, initial_rate = (zero, zero, one, one), (zero,) * 4
    if heterogeneous:
        q, s = context.exp(-3 * time / 4), context.exp(-5 * time / 4)
        first = (2 * (1 - q) / 3, zero, (2 + q) / 3, one)
        second = (first[0], zero, one, first[2])
        joint = (4 * (1 - s) / 5, zero, (4 + s) / 5, (4 + s) / 5)
        return (initial, first, second, joint), (
            initial_rate,
            (q / 2, zero, -q / 4, zero),
            (q / 2, zero, zero, -q / 4),
            (s, zero, -s / 4, -s / 4),
        )
    r = context.exp(-time / 4)
    q, s = context.exp(-time / 2), context.exp(-3 * time / 4)
    pair, rate = (1 - q) / 2, q / 4
    first, first_rate = (pair, zero, 1 - pair, one), (rate, zero, -rate, zero)
    if layout == "shared":
        second, second_rate = (pair, zero, one, 1 - pair), (rate, zero, zero, -rate)
        joint = (2 * (1 - s) / 3, zero, (2 + s) / 3, (2 + s) / 3)
        joint_rate = (s / 2, zero, -s / 4, -s / 4)
    else:
        second, second_rate = (zero, pair, one, 1 - pair), (zero, rate, zero, -rate)
        joint, joint_rate = (pair, pair, 1 - pair, 1 - pair), (rate, rate, -rate, -rate)
        if layout == "internal_path":
            middle = (1 - s) / 3
            leaf, source = (2 - 3 * r + s) / 6, (2 + 3 * r + s) / 6
            first, second = (middle, leaf, source, one), (leaf, middle, one, source)
            first_rate = (s / 4, (r - s) / 8, -(r + s) / 8, zero)
            second_rate = ((r - s) / 8, s / 4, zero, -(r + s) / 8)
    return (initial, first, second, joint), (
        initial_rate,
        first_rate,
        second_rate,
        joint_rate,
    )


def _layout_arguments(layout, **changes):
    old = ((0, 1),) if layout == "internal_path" else ()
    return _arguments(
        old_edges=old,
        old_conductances=(Q(1),) if old else (),
        second_contact=(0 if layout == "shared" else 1, 3),
        **changes,
    )


@pytest.mark.parametrize("layout", ("shared", "separate", "internal_path"))
@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_closed_linear_layouts_validate_full_rows_and_support_specific_bounds(
    layout, duration, mp, record_property
):
    report = owner._prepared_support_contact_composition(
        **_layout_arguments(layout, elapsed_time=duration)
    )
    states, derivatives = _closed_states(mp, layout, _mp(mp, duration))
    tolerance = mp.mpf("1e-130")
    for state, derivative, edges in zip(states, derivatives, _branches(layout)):
        rows = _linear_rows(state, edges, (Q(1),) * 4, mp)
        assert max(abs(x - y) for x, y in zip(rows, derivative)) < tolerance
        assert abs(sum(state) - 2) < tolerance
    means = tuple(sum(state[:2]) / 2 for state in states)
    defect = means[3] - means[1] - means[2] + means[0]
    r = mp.exp(-_mp(mp, duration) / 4)
    expected = {
        "shared": -((1 - r) ** 2) * (1 + 2 * r) / 6,
        "separate": mp.mpf(0),
        "internal_path": -((1 - r) ** 3) / 6,
    }[layout]
    assert abs(defect - expected) < tolerance
    lower, upper = report.exact_true_defect_bounds
    assert _mp(mp, lower) - tolerance <= defect <= _mp(mp, upper) + tolerance
    assert upper <= report.universal.mixed_state_defect_upper_bound
    for i, bound in enumerate(report.nodal_mixed_defect_upper_bounds):
        actual = states[3][i] - states[1][i] - states[2][i] + states[0][i]
        assert abs(actual) <= _mp(mp, bound) + tolerance
    assert report.independent_contact_blocks == (layout == "separate")
    assert report.exact_mixed_zero == (not duration or layout == "separate")
    assert (
        report.arithmetic_exact_zero
        == report.inverse_capacity_exact_zero
        == report.exact_mixed_zero
    )
    assert (
        report.inverse_capacity_exact_true_defect_bounds
        == report.exact_true_defect_bounds
    )
    if duration and layout != "separate":
        assert defect < 0 and upper > 0
    record_property("layout", layout)
    record_property("duration", str(duration))
    record_property("analytic_defect", mp.nstr(defect, 24))
    record_property("support_absolute_bound", str(upper))
    record_property(
        "universal_absolute_bound", str(report.universal.mixed_state_defect_upper_bound)
    )


def test_equal_individual_finite_readings_do_not_determine_joint_reading(mp):
    readings = {}
    for layout in ("shared", "separate"):
        states, _ = _closed_states(mp, layout, mp.mpf(1) / 4)
        readings[layout] = tuple(sum(state[:2]) / 2 for state in states)
    assert readings["shared"][:3] == readings["separate"][:3]
    assert readings["shared"][3] < readings["separate"][3]
    shared = owner._prepared_support_contact_composition(**_layout_arguments("shared"))
    separate = owner._prepared_support_contact_composition(
        **_layout_arguments("separate")
    )
    assert shared.universal.weighted_degree_upper_bound == 2
    assert separate.universal.weighted_degree_upper_bound == 1
    assert (
        shared.universal.mixed_sensitivity_upper_bound
        == separate.universal.mixed_sensitivity_upper_bound
    )
    assert shared.arithmetic_defect_upper_bound > 0
    assert separate.arithmetic_defect_upper_bound == 0


@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_heterogeneous_star_uses_two_actual_normalized_observations(duration, mp):
    capacity = (Q(2), Q(1), Q(1), Q(1))
    report = _bound(capacity=capacity, elapsed_time=duration)
    states, derivatives = _closed_states(
        mp, "shared", _mp(mp, duration), heterogeneous=True
    )
    tolerance = mp.mpf("1e-130")
    for state, derivative, edges in zip(states, derivatives, _branches("shared")):
        rows = _linear_rows(state, edges, capacity, mp)
        assert max(abs(x - y) for x, y in zip(rows, derivative)) < tolerance
        assert abs(sum(x / _mp(mp, c) for x, c in zip(state, capacity)) - 2) < tolerance
    ordinary = tuple((x[0] + x[1]) / 2 for x in states)
    weighted = tuple((x[0] / 2 + x[1]) / _mp(mp, Q(3, 2)) for x in states)
    arithmetic = ordinary[3] - ordinary[1] - ordinary[2] + ordinary[0]
    inverse = weighted[3] - weighted[1] - weighted[2] + weighted[0]
    assert abs(inverse - 2 * arithmetic / 3) < tolerance
    assert report.arithmetic_observation_weights == (Q(1, 2), Q(1, 2), 0, 0)
    assert report.inverse_capacity_observation_weights == (Q(1, 3), Q(2, 3), 0, 0)
    assert report.nodal_mixed_defect_upper_bounds[1] == 0
    assert (
        report.inverse_capacity_defect_upper_bound
        == 2 * report.arithmetic_defect_upper_bound / 3
    )
    assert abs(arithmetic) <= _mp(mp, report.arithmetic_defect_upper_bound) + tolerance
    assert (
        abs(inverse) <= _mp(mp, report.inverse_capacity_defect_upper_bound) + tolerance
    )


def test_zero_declared_old_edge_and_absent_edge_have_identical_positive_support():
    absent = owner._prepared_support_contact_composition(
        **_layout_arguments("separate")
    )
    zero = _bound(old_edges=((0, 1),), old_conductances=(Q(0),), second_contact=(1, 3))
    assert zero.old_edges == ((0, 1),) and zero.old_conductances == (0,)
    assert absent.old_edges == ()
    assert zero.joined_adjacency == absent.joined_adjacency
    assert zero.positive_support_components == absent.positive_support_components
    assert (
        zero.nodal_mixed_defect_upper_bounds
        == absent.nodal_mixed_defect_upper_bounds
        == (0,) * 4
    )
    assert zero.independent_contact_blocks


def test_tiny_internal_path_has_positive_continuous_influence_without_a_tail_floor():
    epsilon = Q(1, 1 << 200)
    report = _bound(
        old_edges=((0, 1),), old_conductances=(epsilon,), second_contact=(1, 3)
    )
    assert not report.independent_contact_blocks
    assert not report.exact_mixed_zero
    assert 0 < report.arithmetic_defect_upper_bound < epsilon
    assert report.arithmetic_defect_upper_bound < Q(1, 1 << 128)
    assert report.joined_adjacency[0][1] == epsilon
    assert report.kernel.upper[0][1] > 0


def test_observation_on_unaffected_isolate_is_exact_zero_without_global_independence():
    report = _bound(nodes=(0, 1, 2, 3, 4), capacity=(1,) * 5, observation_nodes=(4,))
    assert not report.independent_contact_blocks
    assert not report.exact_mixed_zero
    assert report.nodal_mixed_defect_upper_bounds[4] == 0
    assert report.arithmetic_exact_zero and report.inverse_capacity_exact_zero
    assert report.exact_true_defect_bounds == (0, 0)


@pytest.mark.parametrize("old_weight", (Q(0), Q(1, 2)))
def test_coincident_marked_contacts_and_old_edge_add_as_independent_channels(
    old_weight, mp
):
    old = ((0, 1),) if old_weight else ()
    report = _bound(
        nodes=(0, 1),
        old_edges=old,
        old_conductances=(old_weight,) if old else (),
        capacity=(1, 1),
        first_contact=(0, 1),
        second_contact=(1, 0),
        observation_nodes=(0,),
    )
    assert report.joined_adjacency == ((0, old_weight + 2), (old_weight + 2, 0))
    assert report.weighted_degrees == (old_weight + 2,) * 2
    time = _mp(mp, report.universal.elapsed_time)
    # x0(t;w)=(1-exp(-w*t/2))/2, x1=1-x0. Direct differentiation
    # gives x0'=w*(x1-x0)/4 and validates all four complete branches.
    readings = []
    for weight in (old_weight, old_weight + 1, old_weight + 1, old_weight + 2):
        e = mp.exp(-_mp(mp, weight) * time / 2)
        state, rate = ((1 - e) / 2, (1 + e) / 2), _mp(mp, weight) * e / 4
        rows = _linear_rows(state, ((0, 1, weight),), (1, 1), mp)
        assert abs(rows[0] - rate) < mp.mpf("1e-130")
        assert abs(rows[1] + rate) < mp.mpf("1e-130")
        readings.append(state[0])
    actual = readings[3] - readings[1] - readings[2] + readings[0]
    expected = (
        -mp.exp(-_mp(mp, old_weight) * time / 2) * (1 - mp.exp(-time / 2)) ** 2 / 2
    )
    assert abs(actual - expected) < mp.mpf("1e-130")
    assert -_mp(mp, report.arithmetic_defect_upper_bound) <= actual < 0


def _matvec(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


@pytest.mark.parametrize("gamma", (Q(0), Q(1), Q(2)))
def test_full_two_row_directional_bounds_and_endpoint_forcing(gamma):
    capacities, b0 = (Q(2), Q(1), Q(3, 2), Q(1, 2)), Q(1, 2)
    report = _bound(
        capacity=capacities, gamma_bounds=(gamma, gamma), initial_form_upper_bound=b0
    )
    adjacency, degree = report.joined_adjacency, report.weighted_degrees
    expected_matrix = tuple(
        tuple(
            (1 + gamma) * capacities[i] * (degree[i] if i == j else adjacency[i][j]) / 4
            for j in range(4)
        )
        for i in range(4)
    )
    assert report.comparison_matrix == expected_matrix
    # Linearization of BOTH rows at common phase: x'=D(-Lx+g*S)/4,
    # theta'=g*D*Lx/4. Form and phase directions are independent.
    for coordinate in range(8):
        dx = tuple(Q(i == coordinate) for i in range(4))
        dt = tuple(Q(4 + i == coordinate) for i in range(4))
        out_x, out_t = [Q(0)] * 4, [Q(0)] * 4
        for a, b in ((0, 2), (0, 3)):
            form, phase = (dx[b] - dx[a]) / 4, (dt[b] - dt[a]) / 4
            out_x[a] += capacities[a] * (form + gamma * phase)
            out_x[b] -= capacities[b] * (form + gamma * phase)
            out_t[a] -= gamma * capacities[a] * form
            out_t[b] += gamma * capacities[b] * form
        q = tuple(abs(x) + abs(t) for x, t in zip(dx, dt))
        assert all(
            abs(x) + abs(t) <= bound
            for x, t, bound in zip(out_x, out_t, _matvec(expected_matrix, q))
        )
    for contact, forcing, linear, derivative in (
        (
            (0, 2),
            report.first_forcing_constant,
            report.first_forcing_linear,
            report.first_contact_derivative_matrix,
        ),
        (
            (0, 3),
            report.second_forcing_constant,
            report.second_forcing_linear,
            report.second_contact_derivative_matrix,
        ),
    ):
        for i in range(4):
            expected = (
                capacities[i] * ((1 + gamma) * b0 / 2 + gamma / 4)
                if i in contact
                else 0
            )
            assert forcing[i] == expected
            assert linear[i] == (
                capacities[i]
                * (1 + gamma)
                * report.universal.mobility_degree_upper_bound
                * gamma
                / 8
                if i in contact
                else 0
            )
            assert derivative[i] == tuple(
                (1 + gamma) * capacities[i] / 4 if i in contact and j in contact else 0
                for j in range(4)
            )
    # Phases theta0=pi/2, theta2=theta3=0 realize |sin|=1 on
    # both edges. Opposite phase directions attain the center Hessian bound.
    u, v = (
        report.first_sensitivity_slope_upper_bounds,
        report.second_sensitivity_slope_upper_bounds,
    )
    expected_hessian = tuple(
        gamma
        * capacities[i]
        * sum((adjacency[i][j] * (u[i] + u[j]) * (v[i] + v[j]) for j in range(4)), Q(0))
        / 4
        for i in range(4)
    )
    assert report.mixed_hessian_upper_bounds == expected_hessian
    eta = tuple(value if i < 2 else -value for i, value in enumerate(u))
    zeta = tuple(value if i < 2 else -value for i, value in enumerate(v))
    form_hessian, phase_hessian = [Q(0)] * 4, (Q(0),) * 4
    for a, b in ((0, 2), (0, 3)):
        sine_gap = Q(-1)  # theta_b-theta_a=-pi/2 on these oriented edges.
        current_hessian = (
            -gamma * sine_gap * (eta[b] - eta[a]) * (zeta[b] - zeta[a]) / 4
        )
        form_hessian[a] += capacities[a] * current_hessian
        form_hessian[b] -= capacities[b] * current_hessian
    # The phase row is linear in form and has no phase Hessian. These signed
    # edge derivatives attain the nonnegative bound at every active node.
    assert (
        tuple(abs(x) + abs(t) for x, t in zip(form_hessian, phase_hessian))
        == expected_hessian
    )
    assert report.mixed_contact_forcing_upper_bounds == tuple(
        a + b
        for a, b in zip(
            _matvec(report.first_contact_derivative_matrix, v),
            _matvec(report.second_contact_derivative_matrix, u),
        )
    )
    if gamma:
        assert expected_hessian[0] > 0
    else:
        assert expected_hessian == (0,) * 4


@pytest.mark.parametrize("gamma", (Q(0), Q(2)))
def test_positive_phase_feedback_preserves_exact_separated_invariant_blocks(gamma):
    report = _bound(
        second_contact=(1, 3),
        gamma_bounds=(gamma, gamma),
        capacity=(1, 2, Q(3, 2), Q(1, 2)),
    )
    assert report.independent_contact_blocks and report.exact_mixed_zero
    assert report.mixed_contact_forcing_upper_bounds == (0,) * 4
    assert report.mixed_hessian_upper_bounds == (0,) * 4
    assert (
        report.exact_true_defect_bounds
        == report.inverse_capacity_exact_true_defect_bounds
        == (0, 0)
    )


def test_contact_exchange_and_endpoint_orientation_preserve_bound():
    args = dict(
        first_contact_weight_bounds=(Q(1, 4), Q(1, 2)),
        second_contact_weight_bounds=(Q(1, 8), Q(3, 4)),
        gamma_bounds=(Q(1, 4), Q(1, 2)),
        capacity=(2, 1, Q(3, 2), 1),
    )
    first = _bound(**args)
    reverse = _bound(**args, first_contact=(2, 0), second_contact=(3, 0))
    exchange = _bound(
        **dict(
            args,
            first_contact_weight_bounds=args["second_contact_weight_bounds"],
            second_contact_weight_bounds=args["first_contact_weight_bounds"],
        ),
        first_contact=(0, 3),
        second_contact=(0, 2),
    )
    assert (
        first.nodal_mixed_defect_upper_bounds
        == reverse.nodal_mixed_defect_upper_bounds
        == exchange.nodal_mixed_defect_upper_bounds
    )
    assert (
        first.first_sensitivity_slope_upper_bounds
        == exchange.second_sensitivity_slope_upper_bounds
    )
    assert (
        first.second_sensitivity_slope_upper_bounds
        == exchange.first_sensitivity_slope_upper_bounds
    )


def test_relabeling_and_node_order_transform_all_primitive_associations():
    capacity, order = (Q(2), Q(1), Q(3, 2), Q(1, 2)), (3, 1, 0, 2)
    labels = ("donor-a", "donor-b", "host-a", "host-b")
    base = _bound(
        old_edges=((0, 1),),
        old_conductances=(Q(1, 3),),
        capacity=capacity,
        gamma_bounds=(Q(1, 2), Q(1, 2)),
    )
    transformed = _bound(
        nodes=tuple(labels[i] for i in order),
        old_edges=((labels[1], labels[0]),),
        old_conductances=(Q(1, 3),),
        capacity=tuple(capacity[i] for i in order),
        first_contact=(labels[2], labels[0]),
        second_contact=(labels[0], labels[3]),
        observation_nodes=(labels[1], labels[0]),
        gamma_bounds=(Q(1, 2), Q(1, 2)),
    )
    assert transformed.nodal_mixed_defect_upper_bounds == tuple(
        base.nodal_mixed_defect_upper_bounds[i] for i in order
    )
    assert transformed.comparison_matrix == tuple(
        tuple(base.comparison_matrix[i][j] for j in order) for i in order
    )
    assert (
        transformed.arithmetic_defect_upper_bound == base.arithmetic_defect_upper_bound
    )
    assert (
        transformed.inverse_capacity_defect_upper_bound
        == base.inverse_capacity_defect_upper_bound
    )


@pytest.mark.parametrize("scale", (Q(1, 4), Q(1), Q(3, 2)))
def test_common_heterogeneous_capacity_scale_is_a_clock_change_for_both_rows(scale):
    capacity, h = (Q(2), Q(1), Q(3, 2), Q(1, 2)), Q(1, 100)
    shared = dict(capacity=capacity, gamma_bounds=(Q(2), Q(2)), elapsed_time=scale * h)
    clock = _bound(**shared)
    scaled = _bound(
        **dict(shared, capacity=tuple(scale * c for c in capacity), elapsed_time=h)
    )
    assert scaled.kernel.upper == clock.kernel.upper
    assert (
        scaled.nodal_mixed_defect_upper_bounds == clock.nodal_mixed_defect_upper_bounds
    )
    assert scaled.arithmetic_defect_upper_bound == clock.arithmetic_defect_upper_bound
    assert (
        scaled.inverse_capacity_defect_upper_bound
        == clock.inverse_capacity_defect_upper_bound
    )
    assert (
        scaled.inverse_capacity_observation_weights
        == clock.inverse_capacity_observation_weights
    )


@pytest.mark.parametrize(
    "side", ("first_contact_weight_bounds", "second_contact_weight_bounds")
)
def test_zero_contact_product_preserves_own_observation_noise_floors(side):
    report = _bound(
        **{side: (0, 0)},
        reading_error_bounds=(Q(1, 10), Q(1, 20), Q(1, 30), Q(1, 40)),
        inverse_capacity_reading_error_bounds=(Q(1, 70),) * 4,
    )
    error, weighted_error = Q(5, 24), Q(2, 35)
    assert report.nodal_mixed_defect_upper_bounds == (0,) * 4
    assert report.exact_mixed_zero
    assert report.exact_true_defect_bounds == (0, 0)
    assert report.exact_observed_defect_bounds == (-error, error)
    assert report.inverse_capacity_exact_observed_defect_bounds == (
        -weighted_error,
        weighted_error,
    )


def test_absent_weighted_error_budget_keeps_weighted_true_bound_available():
    report = _bound(inverse_capacity_reading_error_bounds=None)
    assert report.inverse_capacity_exact_true_defect_bounds is not None
    assert report.inverse_capacity_exact_observed_defect_bounds is None
    assert report.inverse_capacity_observed_defect_bounds is None


def test_subgrid_contact_product_survives_exact_projection_before_display():
    epsilon = Q(1, 1 << 200)
    report = _bound(
        first_contact_weight_bounds=(0, epsilon),
        second_contact_weight_bounds=(epsilon, epsilon),
    )
    assert report.universal.contact_weight_product_upper_bound == epsilon**2
    bound = report.arithmetic_defect_upper_bound
    assert 0 < bound < epsilon**2
    assert report.exact_true_defect_bounds == (-bound, bound)
    assert (
        report.true_defect_bounds.lo
        <= -bound
        < 0
        < bound
        <= report.true_defect_bounds.hi
    )
    assert report.true_defect_bounds.hi > bound
    assert not report.arithmetic_exact_zero


def test_old_chart_population_is_associated_and_independent_of_influence_zero():
    absent, empty = _bound(), _bound(old_edge_chart_margins=())
    assert (
        absent.retention_status == "not_requested" and absent.retention_margins is None
    )
    assert empty.old_edge_chart_margins == empty.retention_margins == ()
    assert empty.retention_status == "certified"
    report = _bound(
        old_edges=((0, 1),),
        old_conductances=(0,),
        second_contact=(1, 3),
        gamma_bounds=(1, 1),
        old_edge_chart_margins=(Q(1, 100),),
    )
    assert report.exact_mixed_zero
    assert report.retention_status == "indeterminate"
    assert report.retention_margins[0] <= 0


def test_actual_degrees_are_derived_from_all_three_contributions():
    report = _bound(
        old_edges=((0, 2), (1, 3)),
        old_conductances=(Q(2), Q(3)),
        first_contact_weight_bounds=(0, Q(1, 2)),
        second_contact_weight_bounds=(0, Q(3, 2)),
    )
    assert report.weighted_degrees == (Q(4), Q(3), Q(5, 2), Q(9, 2))
    assert report.universal.weighted_degree_upper_bound == Q(9, 2)
    assert report.joined_adjacency[0][2] == Q(5, 2)


def test_report_is_immutable_and_does_not_accept_cached_degrees_or_reports():
    report = _bound()
    with pytest.raises(FrozenInstanceError):
        report.capacity = (2,) * 4
    with pytest.raises(TypeError):
        _bound(weighted_degree_upper_bound=1)
    with pytest.raises(TypeError):
        _bound(source=report)


def test_finite_ordered_iterables_normalize_to_immutable_source_associations():
    baseline = _bound(old_edges=((0, 1),), old_conductances=(Q(1, 3),))
    report = _bound(
        nodes=iter((0, 1, 2, 3)),
        old_edges=([0, 1],),
        old_conductances=iter((Q(1, 3),)),
        capacity=iter((1, 1, 1, 1)),
        first_contact=[0, 2],
        second_contact=iter((0, 3)),
        observation_nodes=[0, 1],
    )
    assert report == baseline


@pytest.mark.parametrize(
    "changes",
    (
        {"nodes": ()},
        {"nodes": (0,)},
        {"nodes": (0, 0, 2, 3)},
        {"nodes": (0, [], 2, 3)},
        {"nodes": {0, 1, 2, 3}},
        {"nodes": tuple(range(25)), "capacity": (1,) * 25},
        {"old_edges": ((0, 0),), "old_conductances": (1,)},
        {"old_edges": ((0, 4),), "old_conductances": (1,)},
        {"old_edges": ((0, 1), (1, 0)), "old_conductances": (1, 2)},
        {"old_edges": ({0, 1},), "old_conductances": (1,)},
        {"old_edges": ((0, 1),), "old_conductances": ()},
        {"old_edges": (), "old_conductances": (1,)},
        {"old_edges": ((0, 1),), "old_conductances": (-1,)},
        {"old_edges": ((0, 1),), "old_conductances": (True,)},
        {"old_edges": ((0, 1),), "old_conductances": (float("nan"),)},
        {"capacity": (1, 0, 1, 1)},
        {"capacity": (1, -1, 1, 1)},
        {"capacity": (1, True, 1, 1)},
        {"capacity": (1, float("inf"), 1, 1)},
        {"capacity": (1, 1, 1)},
        {"capacity": {0: 1, 1: 1, 2: 1, 3: 1}},
        {"first_contact": (0, 0)},
        {"first_contact": (0, 4)},
        {"second_contact": (0, 1, 2)},
        {"second_contact": {0, 3}},
        {"observation_nodes": ()},
        {"observation_nodes": (0, 0)},
        {"observation_nodes": (0, 4)},
        {"observation_nodes": {0, 1}},
        {"first_contact_weight_bounds": (-1, 0)},
        {"second_contact_weight_bounds": (1, 0)},
        {"gamma_bounds": (0, True)},
        {"elapsed_time": True},
        {"initial_form_upper_bound": -1},
        {"reading_error_bounds": (0, 0, 0)},
        {"inverse_capacity_reading_error_bounds": (0, 0, True, 0)},
        {"old_edge_chart_margins": (1,)},
        {
            "old_edges": ((0, 1),),
            "old_conductances": (0,),
            "old_edge_chart_margins": (),
        },
        {
            "old_edges": ((0, 1),),
            "old_conductances": (0,),
            "old_edge_chart_margins": (-1,),
        },
        {"elapsed_time": Q(65)},
        # Actual comparison rows of this isolate vanish. The separate legacy
        # resource bound alpha*h still rejects maxcapacity*maxdegree inflation.
        {"capacity": (1, 10**9, 1, 1), "gamma_bounds": (1, 1)},
    ),
)
def test_invalid_raw_primitives_and_both_resource_caps_reject_before_work(
    changes, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid support primitives reached numerical work")

    for object_, name in (
        (comparison, "nonnegative_exponential_upper"),
        (scalar_owner, "_negative_exp_bounds"),
        (scalar_owner, "I"),
    ):
        monkeypatch.setattr(object_, name, forbidden)
    for name in ("nonnegative_exponential_upper", "I"):
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _bound(**changes)


@pytest.mark.parametrize("field", ("capacity", "old_conductances", "observation_nodes"))
def test_unbounded_primitive_iterables_are_rejected_after_bounded_admission(field):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: repeat(1)})


def test_invalid_physical_values_are_not_bypassed_by_zero_contact_product(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid zero-product input reached exponential work")

    monkeypatch.setattr(scalar_owner, "_negative_exp_bounds", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _bound(first_contact_weight_bounds=(0, 0), capacity=(1, True, 1, 1))
    with pytest.raises((TypeError, ValueError)):
        _bound(
            first_contact_weight_bounds=(0, 0),
            reading_error_bounds=(0, 0, float("inf"), 0),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"elapsed_time": Q(32)},
        {"capacity": (1, 16384, 1, 1), "gamma_bounds": (1, 1)},
    ),
)
def test_both_exact_resource_boundaries_are_admitted_before_numerical_work(
    changes, monkeypatch
):
    class AdmittedBoundary(Exception):
        pass

    def reached_numerical_work(**kwargs):
        raise AdmittedBoundary

    # The shared reader is invoked only after both support resource checks.
    # This checks admission without evaluating a large exponent or resolvent.
    monkeypatch.setattr(
        owner, "_prepared_two_contact_composition", reached_numerical_work
    )
    with pytest.raises(AdmittedBoundary):
        _bound(**changes)


def test_scalar_bound_remains_available_without_a_support_or_matrix_domain(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("old scalar API acquired the new matrix kernel")

    monkeypatch.setattr(comparison, "nonnegative_exponential_upper", forbidden)
    result = scalar_owner._prepared_two_contact_composition(
        first_contact_weight_bounds=(1, 1),
        second_contact_weight_bounds=(1, 1),
        gamma_bounds=(0, 0),
        capacity_bounds=(1, 1),
        initial_form_upper_bound=1,
        weighted_degree_upper_bound=2,
        elapsed_time=100,
        reading_error_bounds=(0,) * 4,
    )
    assert result.mixed_state_defect_upper_bound == 2500
