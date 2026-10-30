"""Independent finite bridge controls for positive held capacity families.

Arithmetic and inverse-capacity readings have different normalizations and
separate error budgets. Static boxes and analytic solutions are not trajectories.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product

import mpmath
import pytest

# Only pure setup/reference helpers; no previous fixtures or evaluated reports.
from tests.physics.test_sine_bridge_exchange import _contains, _mp, _rows, _witness
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_preparation as owner

CAPACITY = (Q(949, 1000), Q(1051, 1000))
PORT_CAPACITY = (Q(1049, 1000), Q(1051, 1000))
MASS = (Q(3000, 1051) + Q(2000, 951), Q(3000, 1049) + Q(2000, 949))
CONTRAST = Q(411, 2000)


def _arguments(**changes):
    values = dict(
        initial_form_difference_bounds=(Q(-1, 100), Q(1, 100)),
        initial_phase_gap_bounds=(Q(1, 2), Q(3, 4)),
        bridge_weight_bounds=(Q(1, 2), Q(3, 4)),
        gamma_bounds=(Q(1, 4), Q(1, 3)),
        initial_form_upper_bound=Q(1, 2),
        weighted_degree_upper_bound=3,
        elapsed_time=Q(1, 100),
        donor_node_count=5,
        reading_error_bounds=(Q(1, 10**7),) * 4,
        old_edge_chart_margins=(Q(1, 4),) * 12,
        capacity_bounds=CAPACITY,
        donor_port_capacity_bounds=PORT_CAPACITY,
        donor_internal_capacity_contrast_upper_bound=CONTRAST,
        inverse_capacity_mass_bounds=MASS,
        inverse_capacity_reading_error_bounds=(Q(1, 10**7),) * 4,
    )
    values.update(changes)
    return values


def _bound(**changes):
    return owner._prepared_capacity_bridge_exchange(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("capacity bridge controls cannot execute events or flows")

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


def _capacity_boxes(witness):
    return {
        node: (
            1 + Q((-1) ** index, 20) - Q(1, 1000),
            1 + Q((-1) ** index, 20) + Q(1, 1000),
        )
        for index, node in enumerate(witness.nodes)
    }


def _capacity_rows(context, witness, edges, form, phase, capacity):
    # `_rows` is an independent explicit edge sum from the prior test owner,
    # not a production evaluator. Both complete rows receive the same D.
    dx, dt, degree = _rows(context, witness, edges, form, phase)
    return (
        {node: _mp(context, capacity[node]) * dx[node] for node in witness.nodes},
        {node: _mp(context, capacity[node]) * dt[node] for node in witness.nodes},
        degree,
    )


@pytest.mark.parametrize("which", (0, 1))
def test_full_capacity_state_boxes_transfer_both_readings_across_supports(
    which, mp, record_property
):
    witness = _witness(which, mp)
    report = _bound(old_edge_chart_margins=(Q(1, 4),) * (11 + which))
    assert report.observed_sign == report.inverse_capacity_observed_sign == "negative"
    assert report.retention_status == "certified"
    boxes = _capacity_boxes(witness)
    assert all(CAPACITY[0] <= lo < hi <= CAPACITY[1] for lo, hi in boxes.values())
    assert boxes[witness.donor[0]] == PORT_CAPACITY
    inverse_mass = (
        sum((1 / boxes[node][1] for node in witness.donor), Q(0)),
        sum((1 / boxes[node][0] for node in witness.donor), Q(0)),
    )
    assert inverse_mass == MASS
    # Independent endpoint admission covers every point of each capacity box,
    # including correlations; the selected static representatives below do not
    # establish this universal association.
    donor_edges = tuple((a, b, w) for a, b, w in witness.edges if a in witness.donor)
    variation = sum(
        (
            w * max(abs(boxes[a][0] - boxes[b][1]), abs(boxes[a][1] - boxes[b][0]))
            for a, b, w in donor_edges
        ),
        Q(0),
    )
    assert variation == Q(41, 200) + which * Q(1, 2000) <= CONTRAST
    radius = witness.node_radius
    assert all(abs(value) + radius < Q(1, 2) for value in witness.form.values())
    a0 = witness.form[witness.host[0]] - witness.form[witness.donor[0]]
    assert Q(-1, 100) <= a0 - 2 * radius < a0 + 2 * radius <= Q(1, 100)
    assert Q(1, 2) < Q(3, 5) - 2 * radius < Q(3, 5) + 2 * radius < Q(3, 4)
    for cycle in witness.cycles:
        collar_lower = min(Q(6, len(cycle)), 3 * (Q(1, 2) - Q(2, len(cycle))))
        assert Q(1, 4) + 2 * radius < collar_lower
    tolerance = mp.mpf("1e-130")
    for capacity_sign, form_sign, phase_sign in product((-1, 1), repeat=3):
        capacity = {
            node: 1 + Q((-1) ** index, 20) + capacity_sign * Q((-1) ** index, 1000)
            for index, node in enumerate(witness.nodes)
        }
        initial_form = {
            node: _mp(mp, witness.form[node] + form_sign * (-1) ** index * radius)
            for index, node in enumerate(witness.nodes)
        }
        initial_phase = {
            node: witness.phase[node] + _mp(mp, phase_sign * (-1) ** index * radius)
            for index, node in enumerate(witness.nodes)
        }
        rho_mass = sum((1 / capacity[node] for node in witness.donor), Q(0))
        assert MASS[0] <= rho_mass <= MASS[1]
        # Each pair starts at the very same full unknown state and capacities.
        # Later static points separately exercise the allowed motion boxes;
        # they are not asserted to be samples of the actual continuations.
        for moved in (False, True):
            totals = []
            for branch, edges in enumerate((witness.edges, witness.joined)):
                orientation = -1 if branch == 0 else 1
                form = dict(initial_form)
                phase = dict(initial_phase)
                if moved:
                    for index, node in enumerate(witness.nodes):
                        sign = orientation * (-1) ** index
                        form[node] += _mp(
                            mp, sign * report.form_displacement_upper_bound
                        )
                        phase[node] += _mp(
                            mp, sign * report.phase_displacement_upper_bound
                        )
                assert max(abs(value) for value in form.values()) < _mp(
                    mp, report.form_upper_bound
                )
                dx, dt, degrees = _capacity_rows(
                    mp, witness, edges, form, phase, capacity
                )
                assert max(degrees.values()) <= 3
                assert max(abs(value) for value in dx.values()) < _mp(
                    mp, report.form_rate_upper_bound
                )
                assert max(abs(value) for value in dt.values()) < _mp(
                    mp, report.phase_rate_upper_bound
                )
                weighted_x = sum(
                    dx[node] / _mp(mp, capacity[node]) for node in witness.donor
                )
                weighted_t = sum(
                    dt[node] / _mp(mp, capacity[node]) for node in witness.donor
                )
                host_x = sum(
                    dx[node] / _mp(mp, capacity[node]) for node in witness.host
                )
                host_t = sum(
                    dt[node] / _mp(mp, capacity[node]) for node in witness.host
                )
                assert abs(weighted_x + host_x) < tolerance
                assert abs(weighted_t + host_t) < tolerance
                ordinary_mean = sum(dx[node] for node in witness.donor) / 5
                totals.append((ordinary_mean, weighted_x / _mp(mp, rho_mass)))
                if branch == 0:
                    assert abs(weighted_x) < tolerance and abs(weighted_t) < tolerance
                else:
                    a = form[witness.host[0]] - form[witness.donor[0]]
                    d = phase[witness.donor[0]] - phase[witness.host[0]]
                    current = (
                        _mp(mp, witness.weight)
                        * (a - _mp(mp, witness.gamma) * mp.sin(d))
                        / 4
                    )
                    assert abs(weighted_x - current) < tolerance
                    assert (
                        abs(
                            weighted_t + _mp(mp, witness.gamma * witness.weight) * a / 4
                        )
                        < tolerance
                    )
                    for bounds, value in (
                        (
                            report.inverse_capacity_donor_total_form_rate_bounds,
                            weighted_x,
                        ),
                        (report.inverse_capacity_host_total_form_rate_bounds, host_x),
                        (
                            report.inverse_capacity_donor_mean_form_rate_bounds,
                            weighted_x / _mp(mp, rho_mass),
                        ),
                        (
                            report.inverse_capacity_donor_total_phase_rate_bounds,
                            weighted_t,
                        ),
                        (report.inverse_capacity_host_total_phase_rate_bounds, host_t),
                        (
                            report.inverse_capacity_donor_mean_phase_rate_bounds,
                            weighted_t / _mp(mp, rho_mass),
                        ),
                    ):
                        _contains(mp, bounds, value)
                for cycle in witness.cycles:
                    gaps = tuple(
                        phase[b]
                        - phase[a]
                        + (2 * mp.pi if index == len(cycle) - 1 else 0)
                        for index, (a, b) in enumerate(
                            zip(cycle, cycle[1:] + cycle[:1])
                        )
                    )
                    assert all(0 < gap < mp.pi / 2 for gap in gaps)
                    assert abs(sum(gaps) - 2 * mp.pi) < tolerance
            arithmetic_rate = totals[1][0] - totals[0][0]
            _contains(mp, report.donor_mean_form_rate_contrast_bounds, arithmetic_rate)
            if not moved:
                _contains(
                    mp, report.initial_donor_mean_form_rate_bounds, arithmetic_rate
                )
                _contains(
                    mp,
                    report.inverse_capacity_initial_donor_mean_form_rate_bounds,
                    totals[1][1] - totals[0][1],
                )
    record_property("donor_count", 5)
    record_property("host_count", len(witness.host))
    record_property("marked_edge_population", 11 + which)
    record_property("capacity_contrast_upper", str(variation))
    record_property(
        "arithmetic_observed_upper", str(report.observed_increment_bounds.hi)
    )
    record_property(
        "weighted_observed_upper",
        str(report.inverse_capacity_observed_increment_bounds.hi),
    )
    record_property("retention_minimum", str(min(report.retention_margins)))


def test_disconnected_arithmetic_motion_is_not_a_weighted_conservation_law(mp):
    witness = _witness(0, mp)
    capacity = {
        node: 1 + Q((-1) ** index, 20) for index, node in enumerate(witness.nodes)
    }
    form = {node: _mp(mp, value) for node, value in witness.form.items()}
    dx, dt, _ = _capacity_rows(
        mp, witness, witness.edges, form, witness.phase, capacity
    )
    # Exact C5 sine currents cancel at every carrier node. Independent rational
    # edge gradients give (-.004,0,.0035,-.0035,.004) on the donor.
    gradient = {node: Q(0) for node in witness.donor}
    for a, b, w in witness.edges:
        if a not in witness.donor:
            continue
        gradient[a] += w * (witness.form[a] - witness.form[b])
        gradient[b] -= w * (witness.form[a] - witness.form[b])
    assert tuple(gradient.values()) == (
        Q(-1, 250),
        0,
        Q(7, 2000),
        Q(-7, 2000),
        Q(1, 250),
    )
    exact = -sum((capacity[node] * gradient[node] for node in witness.donor), Q(0)) / 20
    assert exact == Q(-7, 400000)
    assert abs(sum(dx[node] for node in witness.donor) / 5 - _mp(mp, exact)) < mp.mpf(
        "1e-130"
    )
    assert abs(
        sum(dx[node] / _mp(mp, capacity[node]) for node in witness.donor)
    ) < mp.mpf("1e-130")
    assert abs(
        sum(dt[node] / _mp(mp, capacity[node]) for node in witness.donor)
    ) < mp.mpf("1e-130")
    joined_x, _, _ = _capacity_rows(
        mp, witness, witness.joined, form, witness.phase, capacity
    )
    # Unequal port capacities also prevent ordinary host/donor reciprocity.
    ordinary_reaction_sum = sum(joined_x.values()) - sum(dx.values())
    a = form[witness.host[0]] - form[witness.donor[0]]
    d = witness.phase[witness.donor[0]] - witness.phase[witness.host[0]]
    current = _mp(mp, witness.weight) * (a - _mp(mp, witness.gamma) * mp.sin(d)) / 4
    expected = _mp(mp, capacity[witness.donor[0]] - capacity[witness.host[0]]) * current
    assert abs(ordinary_reaction_sum - expected) < mp.mpf("1e-130")
    assert ordinary_reaction_sum < 0


def _three_node_arguments(**changes):
    values = dict(
        initial_form_difference_bounds=(1, 1),
        initial_phase_gap_bounds=(0, 0),
        bridge_weight_bounds=(1, 1),
        gamma_bounds=(0, 0),
        initial_form_upper_bound=1,
        weighted_degree_upper_bound=2,
        elapsed_time=Q(1, 100),
        donor_node_count=2,
        reading_error_bounds=(0,) * 4,
        old_edge_chart_margins=None,
        capacity_bounds=(1, 2),
        donor_port_capacity_bounds=(2, 2),
        donor_internal_capacity_contrast_upper_bound=1,
        inverse_capacity_mass_bounds=(Q(3, 2), Q(3, 2)),
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    values.update(changes)
    return values


@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_exact_three_node_solution_retains_both_branch_arithmetic_motion(duration, mp):
    report = owner._prepared_capacity_bridge_exchange(
        **_three_node_arguments(elapsed_time=duration)
    )
    # Full linear rows: connected symmetry x1=x2 gives a'=-5a/4,
    # disconnected donor difference gives a'=-3a/4. These closed solutions
    # retain control motion; no numerical integrator or trajectory is invoked.
    time = _mp(mp, duration)
    on_decay, off_decay = mp.exp(-5 * time / 4), mp.exp(-3 * time / 4)
    connected = (4 * (1 - on_decay) / 5, (4 + on_decay) / 5, (4 + on_decay) / 5)
    disconnected = (2 * (1 - off_decay) / 3, (2 + off_decay) / 3, mp.mpf(1))
    arithmetic = sum(connected[:2]) / 2 - sum(disconnected[:2]) / 2
    weighted = (
        connected[0] / 2 + connected[1] - disconnected[0] / 2 - disconnected[1]
    ) / mp.mpf("1.5")
    expected_a = 3 * (1 - on_decay) / 10 - (1 - off_decay) / 6
    expected_w = 2 * (1 - on_decay) / 15
    assert abs(arithmetic - expected_a) < mp.mpf("1e-130")
    assert abs(weighted - expected_w) < mp.mpf("1e-130")
    _contains(mp, report.true_increment_bounds, arithmetic)
    _contains(mp, report.inverse_capacity_true_increment_bounds, weighted)
    assert report.increment_remainder_upper_bound == duration**2
    assert report.inverse_capacity_increment_remainder_upper_bound == duration**2 / 3
    assert report.retention_status == "not_requested"
    if duration == Q(1, 100):
        assert (
            report.observed_sign == report.inverse_capacity_observed_sign == "positive"
        )
    if duration == Q(1, 4):
        assert report.true_increment_bounds == I(0, Q(1, 8))
        assert report.observed_sign == "indeterminate"
        assert report.inverse_capacity_observed_sign == "positive"
    if duration == 1:
        assert arithmetic > 0 and weighted > 0
        assert (
            report.observed_sign
            == report.inverse_capacity_observed_sign
            == "indeterminate"
        )


def test_three_node_exact_matrix_rows_expose_the_omitted_internal_remainder():
    capacity, initial = (Q(2), Q(1), Q(1)), (Q(0), Q(1), Q(1))

    def derivative(state, edges):
        result = [Q(0)] * 3
        for a, b in edges:
            current = (state[b] - state[a]) / 4
            result[a] += capacity[a] * current
            result[b] -= capacity[b] * current
        return tuple(result)

    old, joined = ((0, 1),), ((0, 1), (0, 2))
    off_rate, on_rate = derivative(initial, old), derivative(initial, joined)
    off_second, on_second = derivative(off_rate, old), derivative(on_rate, joined)
    assert off_rate == (Q(1, 2), Q(-1, 4), 0)
    assert on_rate == (1, Q(-1, 4), Q(-1, 4))
    assert (sum(on_rate[:2]) - sum(off_rate[:2])) / 2 == Q(1, 4)
    actual_second = (sum(on_second[:2]) - sum(off_second[:2])) / 2
    bridge_second = capacity[0] * (on_rate[2] - on_rate[0]) / 8
    assert actual_second == Q(-3, 8)
    assert bridge_second == Q(-5, 16)
    assert actual_second - bridge_second == Q(-1, 16)


def test_complete_row_majorants_integrate_both_internal_branch_remainders():
    report = _bound()
    kappa, b, g, h = CAPACITY[1] * 3, Q(1, 2), Q(1, 3), Q(1, 100)
    form_speed = (kappa * (2 * b + g) / 4, kappa**2 * g / 8)
    phase_speed = (g * kappa * b / 2, kappa**2 * g**2 / 8)

    def integrate(coefficients):
        return sum(
            (
                value * h ** (index + 1) / (index + 1)
                for index, value in enumerate(coefficients)
            ),
            Q(0),
        )

    x, t = integrate(form_speed), integrate(phase_speed)
    # Integrate 2X+2gT directly from the two complete nodal speed polynomials.
    integral = sum(
        (
            2 * (a + g * p) * h ** (index + 2) / ((index + 1) * (index + 2))
            for index, (a, p) in enumerate(zip(form_speed, phase_speed))
        ),
        Q(0),
    )
    assert report.mobility_degree_upper_bound == kappa
    assert report.form_upper_bound == b + kappa * g * h / 4
    assert report.form_displacement_upper_bound == x
    assert report.phase_displacement_upper_bound == t
    assert report.rate_variation_integral_upper_bound == integral
    assert (
        report.increment_remainder_upper_bound
        == (PORT_CAPACITY[1] * Q(3, 4) + 2 * CONTRAST) * integral / 20
    )
    assert report.inverse_capacity_increment_remainder_upper_bound == Q(
        3, 4
    ) * integral / (4 * MASS[0])
    assert report.internal_mean_rate_contrast_upper_bound == CONTRAST * (x + g * t) / 5
    assert report.retention_margins == (Q(1, 4) - 2 * t,) * 12
    # This allowance accounts for the internal contribution identified by the
    # exact three-node solution. Its numerical size is not claimed optimal,
    # nor does omitting it necessarily make every other loose bound fail.
    assert (
        report.increment_remainder_upper_bound
        > PORT_CAPACITY[1] * Q(3, 4) * integral / 20
    )


@pytest.mark.parametrize("common", (Q(1), Q(1, 4), Q(3, 2)))
def test_common_capacity_reduces_to_unit_law_in_the_declared_rescaled_clock(common):
    values = _arguments(
        capacity_bounds=(common, common),
        donor_port_capacity_bounds=(common, common),
        donor_internal_capacity_contrast_upper_bound=0,
        inverse_capacity_mass_bounds=(5 / common, 5 / common),
    )
    capacity = owner._prepared_capacity_bridge_exchange(**values)
    unit_values = {
        name: value
        for name, value in values.items()
        if name
        not in (
            "capacity_bounds",
            "donor_port_capacity_bounds",
            "donor_internal_capacity_contrast_upper_bound",
            "inverse_capacity_mass_bounds",
            "inverse_capacity_reading_error_bounds",
        )
    }
    unit_values["elapsed_time"] = common * values["elapsed_time"]
    unit = owner._prepared_bridge_exchange(**unit_values)
    for field in (
        "form_upper_bound",
        "form_displacement_upper_bound",
        "phase_displacement_upper_bound",
        "increment_remainder_upper_bound",
        "retention_margins",
    ):
        assert getattr(capacity, field) == getattr(unit, field)
    assert capacity.internal_mean_rate_contrast_upper_bound == 0
    assert (
        capacity.inverse_capacity_increment_remainder_upper_bound
        == unit.increment_remainder_upper_bound
    )
    # The equivalent expressions perform a different number/order of outward
    # dyadic operations. The exact theorem is the same, not necessarily every
    # endpoint bit. Fewer than32 bounded operations here make2^-110 a generous
    # independent grid allowance (coefficients and intervals are all<16).
    for field in ("true_increment_bounds", "observed_increment_bounds"):
        reference = getattr(unit, field)
        for candidate in (
            getattr(capacity, field),
            getattr(capacity, "inverse_capacity_" + field),
        ):
            assert abs(candidate.lo - reference.lo) < Q(1, 2**110)
            assert abs(candidate.hi - reference.hi) < Q(1, 2**110)
    assert (
        capacity.observed_sign
        == capacity.inverse_capacity_observed_sign
        == unit.observed_sign
    )
    assert capacity.retention_status == unit.retention_status


def test_distinct_reading_errors_and_missing_weighted_request_preserve_independence():
    baseline = _bound()
    absent = _bound(inverse_capacity_reading_error_bounds=None)
    assert absent.inverse_capacity_observed_increment_bounds is None
    assert absent.inverse_capacity_observed_sign == "not_requested"
    assert (
        absent.inverse_capacity_true_increment_bounds
        == baseline.inverse_capacity_true_increment_bounds
    )
    assert absent.observed_increment_bounds == baseline.observed_increment_bounds
    for reading in ("reading_error_bounds", "inverse_capacity_reading_error_bounds"):
        uncertain = _bound(**{reading: (1,) * 4})
        assert uncertain.retention_status == "certified"
        if reading == "reading_error_bounds":
            assert uncertain.observed_sign == "indeterminate"
            assert (
                uncertain.inverse_capacity_observed_increment_bounds
                == baseline.inverse_capacity_observed_increment_bounds
            )
        else:
            assert uncertain.inverse_capacity_observed_sign == "indeterminate"
            assert (
                uncertain.observed_increment_bounds
                == baseline.observed_increment_bounds
            )


def test_arbitrarily_correlated_four_errors_are_kept_for_each_observation():
    arithmetic = (Q(1, 10**7), Q(2, 10**7), Q(3, 10**7), Q(4, 10**7))
    weighted = (Q(4, 10**7), Q(3, 10**7), Q(2, 10**7), Q(2, 10**7))
    report = _bound(
        reading_error_bounds=arithmetic, inverse_capacity_reading_error_bounds=weighted
    )
    for radii, prefix in ((arithmetic, ""), (weighted, "inverse_capacity_")):
        corners = tuple(
            signs[0] * radii[0]
            - signs[1] * radii[1]
            - signs[2] * radii[2]
            + signs[3] * radii[3]
            for signs in product((-1, 1), repeat=4)
        )
        assert max(corners) == -min(corners) == sum(radii)
        assert getattr(report, prefix + "reading_error_upper_bound") == sum(radii)
        assert getattr(report, prefix + "observed_increment_bounds").contains(
            getattr(report, prefix + "true_increment_bounds")
            + I(min(corners), max(corners))
        )


def test_background_fallbacks_and_tighter_primitives_remain_explicit():
    fallback = _bound(
        donor_internal_capacity_contrast_upper_bound=None,
        inverse_capacity_mass_bounds=None,
    )
    universal = (CAPACITY[1] - CAPACITY[0]) * 5 * 3 / 2
    physical_mass = (5 / CAPACITY[1], 5 / CAPACITY[0])
    assert fallback.donor_internal_capacity_contrast_upper_bound is None
    assert fallback.inverse_capacity_mass_bounds is None
    assert fallback.fallback_donor_internal_capacity_contrast_upper_bound == universal
    assert fallback.effective_donor_internal_capacity_contrast_upper_bound == universal
    assert fallback.effective_inverse_capacity_mass_bounds == physical_mass
    tighter = _bound()
    assert (
        tighter.effective_donor_internal_capacity_contrast_upper_bound
        == CONTRAST
        < universal
    )
    assert tighter.effective_inverse_capacity_mass_bounds == MASS
    assert (
        tighter.increment_remainder_upper_bound
        < fallback.increment_remainder_upper_bound
    )
    loose = _bound(
        donor_internal_capacity_contrast_upper_bound=10,
        inverse_capacity_mass_bounds=(1, 100),
    )
    assert loose.donor_internal_capacity_contrast_upper_bound == 10
    assert loose.inverse_capacity_mass_bounds == (1, 100)
    assert loose.effective_donor_internal_capacity_contrast_upper_bound == universal
    assert loose.effective_inverse_capacity_mass_bounds == physical_mass
    # A supplied tighter contrast is additional conditional source information;
    # its validity cannot be inferred or refuted from broad marginal capacities.
    assert (
        _bound(
            donor_internal_capacity_contrast_upper_bound=0
        ).internal_mean_rate_contrast_upper_bound
        == 0
    )


def test_counts_and_inverse_mass_retain_their_distinct_normalizations():
    first = _bound(
        reading_error_bounds=(0,) * 4, inverse_capacity_reading_error_bounds=(0,) * 4
    )
    second = _bound(
        donor_node_count=10,
        inverse_capacity_mass_bounds=tuple(2 * value for value in MASS),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    assert (
        second.inverse_capacity_donor_total_form_rate_bounds
        == first.inverse_capacity_donor_total_form_rate_bounds
    )
    assert (
        second.inverse_capacity_host_total_form_rate_bounds
        == first.inverse_capacity_host_total_form_rate_bounds
    )
    for prefix in ("", "inverse_capacity_"):
        assert 2 * getattr(
            second, prefix + "increment_remainder_upper_bound"
        ) == getattr(first, prefix + "increment_remainder_upper_bound")
        left, right = getattr(first, prefix + "true_increment_bounds"), getattr(
            second, prefix + "true_increment_bounds"
        )
        assert abs(left.lo - 2 * right.lo) < Q(1, 2**110)
        assert abs(left.hi - 2 * right.hi) < Q(1, 2**110)


def test_zero_horizon_zero_response_and_strict_retention_boundaries():
    baseline = _bound()
    zero_time = _bound(
        elapsed_time=0,
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    assert zero_time.gamma_bounds[0] > 0
    assert (
        zero_time.true_increment_bounds
        == zero_time.inverse_capacity_true_increment_bounds
        == I(0)
    )
    assert (
        zero_time.observed_sign
        == zero_time.inverse_capacity_observed_sign
        == "indeterminate"
    )
    assert zero_time.retention_margins == zero_time.old_edge_chart_margins
    exact_zero = _bound(
        initial_form_difference_bounds=(0, 0),
        initial_form_upper_bound=0,
        gamma_bounds=(0, 0),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    assert (
        exact_zero.true_increment_bounds
        == exact_zero.inverse_capacity_true_increment_bounds
        == I(0)
    )
    assert (
        exact_zero.observed_sign
        == exact_zero.inverse_capacity_observed_sign
        == "indeterminate"
    )
    drift = 2 * baseline.phase_displacement_upper_bound
    for margins, expected in (
        (None, "not_requested"),
        ((0,), "indeterminate"),
        ((drift,), "indeterminate"),
        ((drift / 2,), "indeterminate"),
        ((drift + Q(1, 10**6),), "certified"),
    ):
        report = _bound(old_edge_chart_margins=margins)
        assert report.retention_status == expected
        assert report.observed_increment_bounds == baseline.observed_increment_bounds
        assert (
            report.inverse_capacity_observed_increment_bounds
            == baseline.inverse_capacity_observed_increment_bounds
        )


def test_positive_subgrid_inverse_mass_is_inverted_before_outward_rounding():
    capacity = Q(2**200)
    values = _three_node_arguments(
        weighted_degree_upper_bound=1,
        capacity_bounds=(capacity, capacity),
        donor_port_capacity_bounds=(capacity, capacity),
        donor_node_count=1,
        donor_internal_capacity_contrast_upper_bound=0,
        inverse_capacity_mass_bounds=(1 / capacity, 1 / capacity),
        elapsed_time=0,
    )
    report = owner._prepared_capacity_bridge_exchange(**values)
    assert report.effective_inverse_capacity_mass_bounds == (Q(1, 2**200),) * 2
    assert report.inverse_capacity_initial_donor_mean_form_rate_bounds == I(Q(2**198))
    assert (
        report.true_increment_bounds
        == report.inverse_capacity_true_increment_bounds
        == I(0)
    )


def test_fixed_width_capacity_fraction_components_are_normalized_exactly():
    np = pytest.importorskip("numpy")
    exact = Q(2**62, 2**62 + 1)
    narrow = Q(np.int64(2**62), np.int64(2**62 + 1))
    expected = _bound(
        capacity_bounds=(exact, exact),
        donor_port_capacity_bounds=(exact, exact),
        donor_internal_capacity_contrast_upper_bound=0,
        inverse_capacity_mass_bounds=None,
    )
    actual = _bound(
        capacity_bounds=(narrow, narrow),
        donor_port_capacity_bounds=(narrow, narrow),
        donor_internal_capacity_contrast_upper_bound=0,
        inverse_capacity_mass_bounds=None,
    )
    assert actual == expected
    for value in actual.capacity_bounds + actual.effective_inverse_capacity_mass_bounds:
        assert type(value.numerator) is type(value.denominator) is int


def test_inputs_are_copied_and_observations_rebuilt_without_cached_verdicts():
    capacity, errors = list(CAPACITY), [Q(1, 10**7)] * 4
    report = _bound(
        capacity_bounds=capacity, inverse_capacity_reading_error_bounds=errors
    )
    capacity[0], errors[-1] = 0, 100
    assert report.capacity_bounds == CAPACITY
    assert report.inverse_capacity_reading_error_bounds == (Q(1, 10**7),) * 4
    with pytest.raises(FrozenInstanceError):
        report.observed_sign = "positive"
    assert (
        _bound(
            inverse_capacity_reading_error_bounds=errors
        ).inverse_capacity_observed_sign
        == "indeterminate"
    )
    for bad in (report, I(*CAPACITY), {"lo": 1, "hi": 2, "certified": True}):
        with pytest.raises((TypeError, ValueError)):
            _bound(capacity_bounds=bad)


INVALID = (
    ("capacity_bounds", (0, 1)),
    ("capacity_bounds", (-1, 1)),
    ("capacity_bounds", (2, 1)),
    ("capacity_bounds", (True, 2)),
    ("capacity_bounds", (1, float("nan"))),
    ("capacity_bounds", (1, float("inf"))),
    ("capacity_bounds", ("1", 2)),
    ("capacity_bounds", (1 + 0j, 2)),
    ("capacity_bounds", (1,)),
    ("capacity_bounds", (1, 2, 3)),
    ("capacity_bounds", {1, 2}),
    ("donor_port_capacity_bounds", (0, 1)),
    ("donor_port_capacity_bounds", (1, False)),
    ("donor_port_capacity_bounds", (1, float("inf"))),
    ("donor_port_capacity_bounds", (Q(1, 2), 1)),
    ("donor_port_capacity_bounds", (1, Q(3, 2))),
    ("donor_internal_capacity_contrast_upper_bound", -1),
    ("donor_internal_capacity_contrast_upper_bound", True),
    ("donor_internal_capacity_contrast_upper_bound", float("nan")),
    ("inverse_capacity_mass_bounds", (0, 10)),
    ("inverse_capacity_mass_bounds", (-1, 10)),
    ("inverse_capacity_mass_bounds", (2, 1)),
    ("inverse_capacity_mass_bounds", (1, 2)),
    ("inverse_capacity_mass_bounds", (True, 10)),
    ("inverse_capacity_mass_bounds", (1, float("inf"))),
    ("inverse_capacity_mass_bounds", (1, 2, 3)),
    ("inverse_capacity_reading_error_bounds", (0, 0, 0, True)),
    ("inverse_capacity_reading_error_bounds", (0, 0, 0, -1)),
    ("inverse_capacity_reading_error_bounds", (0, 0, 0, float("nan"))),
    ("inverse_capacity_reading_error_bounds", (0, 0, 0)),
    ("inverse_capacity_reading_error_bounds", (0,) * 5),
    ("initial_form_difference_bounds", (True, 1)),
    ("gamma_bounds", (0, float("inf"))),
    ("elapsed_time", True),
    ("donor_node_count", 1.0),
    ("reading_error_bounds", (0, 0, 0, True)),
    ("old_edge_chart_margins", (Q(1, 4), False)),
    ("old_edge_chart_margins", (0,) * 4097),
)


@pytest.mark.parametrize("name,value", INVALID)
def test_all_invalid_primitives_reject_before_interval_or_trigonometry(
    name, value, monkeypatch
):
    def numerical_work(*args, **kwargs):
        pytest.fail("invalid capacity input reached interval or trigonometric work")

    monkeypatch.setattr(owner, "I", numerical_work)
    monkeypatch.setattr(owner, "sin", numerical_work)
    with pytest.raises((TypeError, ValueError)):
        _bound(**{name: value})


@pytest.mark.parametrize(
    "missing",
    (
        "capacity_bounds",
        "donor_port_capacity_bounds",
        "reading_error_bounds",
        "donor_node_count",
    ),
)
def test_required_capacity_and_observation_premises_have_no_implicit_defaults(missing):
    values = _arguments()
    del values[missing]
    with pytest.raises(TypeError):
        owner._prepared_capacity_bridge_exchange(**values)


def test_ordered_iterables_stop_at_the_declared_work_boundary():
    consumed = []

    def endless():
        for index in range(10000):
            consumed.append(index)
            yield Q(1)

    with pytest.raises(ValueError):
        _bound(capacity_bounds=endless())
    assert len(consumed) == 3
    consumed.clear()
    with pytest.raises(ValueError):
        _bound(inverse_capacity_reading_error_bounds=endless())
    assert len(consumed) == 5
