"""Independent controls for a conductance-continuous matched bridge bound.

Exact endpoint tuples carry the small-conductance theorem. Dyadic intervals
are outward display enclosures; no state trajectory or event is executed.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product

import mpmath
import pytest

# Pure raw setup and explicit edge sums, never old fixtures or response records.
from tests.physics.test_sine_bridge_exchange import _contains, _mp, _witness
from tests.physics.test_sine_capacity_bridge_exchange import (
    CAPACITY,
    CONTRAST,
    INVALID,
    MASS,
    PORT_CAPACITY,
    _arguments,
    _capacity_boxes,
    _capacity_rows,
    _three_node_arguments,
)
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_preparation as owner


def _bound(**changes):
    return owner._prepared_matched_capacity_bridge_exchange(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("matched bridge controls cannot execute events or flows")

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


def _contains_exact(context, endpoints, value):
    assert _mp(context, endpoints[0]) <= value <= _mp(context, endpoints[1])


@pytest.mark.parametrize("which", (0, 1))
def test_fixed_positive_capacity_families_keep_their_independent_association(
    which, mp, record_property
):
    witness = _witness(which, mp)
    values = _arguments(old_edge_chart_margins=(Q(1, 4),) * (11 + which))
    report = owner._prepared_matched_capacity_bridge_exchange(**values)
    prior = owner._prepared_capacity_bridge_exchange(**values)
    boxes = _capacity_boxes(witness)
    assert all(CAPACITY[0] <= lo < hi <= CAPACITY[1] for lo, hi in boxes.values())
    assert boxes[witness.donor[0]] == PORT_CAPACITY
    assert (
        sum((1 / boxes[node][1] for node in witness.donor), Q(0)),
        sum((1 / boxes[node][0] for node in witness.donor), Q(0)),
    ) == MASS
    donor_edges = tuple(edge for edge in witness.edges if edge[0] in witness.donor)
    variation = sum(
        (
            weight * max(abs(boxes[a][0] - boxes[b][1]), abs(boxes[a][1] - boxes[b][0]))
            for a, b, weight in donor_edges
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
        collar = min(Q(6, len(cycle)), 3 * (Q(1, 2) - Q(2, len(cycle))))
        assert Q(1, 4) + 2 * radius < collar
    assert report.observed_sign == report.inverse_capacity_observed_sign == "negative"
    assert report.exact_observed_increment_bounds[1] < Q(-1, 50000)
    assert report.inverse_capacity_exact_observed_increment_bounds[1] < Q(-1, 50000)
    assert (
        report.increment_remainder_upper_bound < prior.increment_remainder_upper_bound
    )
    assert report.retention_margins == prior.retention_margins
    assert report.retention_status == "certified"
    assert (
        report.inverse_capacity_increment_remainder_upper_bound
        == prior.inverse_capacity_increment_remainder_upper_bound
    )
    # Direct initial edge sums preserve the whole matched state, rather than
    # subtracting independent boxes as if their members were the same source.
    tolerance = mp.mpf("1e-130")
    for orientation in (-1, 1):
        capacity = {
            node: 1 + Q((-1) ** index, 20) + orientation * Q((-1) ** index, 1000)
            for index, node in enumerate(witness.nodes)
        }
        form = {
            node: _mp(mp, value + orientation * (-1) ** index * radius)
            for index, (node, value) in enumerate(witness.form.items())
        }
        phase = {
            node: value + _mp(mp, orientation * (-1) ** index * radius)
            for index, (node, value) in enumerate(witness.phase.items())
        }
        off, _, _ = _capacity_rows(mp, witness, witness.edges, form, phase, capacity)
        on, _, _ = _capacity_rows(mp, witness, witness.joined, form, phase, capacity)
        arithmetic = sum(on[node] - off[node] for node in witness.donor) / 5
        mass = sum((1 / capacity[node] for node in witness.donor), Q(0))
        weighted = sum(
            (on[node] - off[node]) / _mp(mp, capacity[node]) for node in witness.donor
        ) / _mp(mp, mass)
        _contains(mp, report.initial_donor_mean_form_rate_bounds, arithmetic)
        _contains(
            mp, report.inverse_capacity_initial_donor_mean_form_rate_bounds, weighted
        )
        assert (
            abs(sum(off[node] / _mp(mp, capacity[node]) for node in witness.donor))
            < tolerance
        )
    record_property(
        "arithmetic_observed_upper", str(report.exact_observed_increment_bounds[1])
    )
    record_property(
        "weighted_observed_upper",
        str(report.inverse_capacity_exact_observed_increment_bounds[1]),
    )
    record_property("arithmetic_remainder", str(report.increment_remainder_upper_bound))
    record_property(
        "previous_arithmetic_remainder", str(prior.increment_remainder_upper_bound)
    )


@pytest.mark.parametrize("duration", (Q(0), Q(1, 100), Q(1, 4), Q(1)))
def test_exact_three_node_solution_retains_control_motion_and_improves_resolution(
    duration, mp, record_property
):
    values = _three_node_arguments(elapsed_time=duration)
    report = owner._prepared_matched_capacity_bridge_exchange(**values)
    prior = owner._prepared_capacity_bridge_exchange(**values)
    time = _mp(mp, duration)
    on_decay, off_decay = mp.exp(-5 * time / 4), mp.exp(-3 * time / 4)
    on = (4 * (1 - on_decay) / 5, (4 + on_decay) / 5, (4 + on_decay) / 5)
    off = (2 * (1 - off_decay) / 3, (2 + off_decay) / 3, mp.mpf(1))
    arithmetic = sum(on[:2]) / 2 - sum(off[:2]) / 2
    weighted = (on[0] / 2 + on[1] - off[0] / 2 - off[1]) / mp.mpf("1.5")
    assert abs(arithmetic - (3 * (1 - on_decay) / 10 - (1 - off_decay) / 6)) < mp.mpf(
        "1e-130"
    )
    assert abs(weighted - 2 * (1 - on_decay) / 15) < mp.mpf("1e-130")
    _contains_exact(mp, report.exact_true_increment_bounds, arithmetic)
    _contains_exact(mp, report.inverse_capacity_exact_true_increment_bounds, weighted)
    assert max(abs(a - b) for a, b in zip(on, off)) <= _mp(
        mp, report.matched_state_difference_upper_bound
    )
    assert report.difference_growth_upper_bound == 1
    assert report.matched_state_difference_upper_bound == duration
    assert report.integrated_matched_state_difference_upper_bound == duration**2 / 2
    assert report.matched_internal_increment_remainder_upper_bound == duration**2 / 8
    assert report.increment_remainder_upper_bound == 5 * duration**2 / 8
    assert prior.increment_remainder_upper_bound == duration**2
    assert report.inverse_capacity_increment_remainder_upper_bound == duration**2 / 3
    assert report.retention_status == "not_requested"
    if duration == 0:
        assert report.exact_true_increment_bounds == (0, 0)
    elif duration <= Q(1, 4):
        assert (
            report.observed_sign == report.inverse_capacity_observed_sign == "positive"
        )
    else:
        assert arithmetic > 0 and weighted > 0
        assert (
            report.observed_sign
            == report.inverse_capacity_observed_sign
            == "indeterminate"
        )
    if duration == Q(1, 4):
        assert prior.observed_sign == "indeterminate"
        assert report.exact_true_increment_bounds == (Q(3, 128), Q(13, 128))
    record_property("duration", str(duration))
    record_property("arithmetic_sign", report.observed_sign)
    record_property("previous_arithmetic_sign", prior.observed_sign)


def test_exact_zero_bridge_preserves_each_readings_noise_and_optional_request():
    arithmetic = (Q(1, 1000), Q(2, 1000), Q(3, 1000), Q(4, 1000))
    weighted = (Q(4, 1000), Q(3, 1000), Q(2, 1000), Q(2, 1000))
    report = _bound(
        bridge_weight_bounds=(0, 0),
        reading_error_bounds=arithmetic,
        inverse_capacity_reading_error_bounds=weighted,
    )
    assert report.capacity_bounds[0] < report.capacity_bounds[1]
    assert report.form_displacement_upper_bound > 0
    assert report.phase_displacement_upper_bound > 0
    assert report.matched_state_difference_upper_bound == 0
    assert report.integrated_matched_state_difference_upper_bound == 0
    assert report.matched_internal_mean_rate_upper_bound == 0
    for prefix, radii in (("", arithmetic), ("inverse_capacity_", weighted)):
        assert getattr(report, prefix + "exact_true_increment_bounds") == (0, 0)
        assert getattr(report, prefix + "true_increment_bounds") == I(0)
        assert getattr(report, prefix + "increment_remainder_upper_bound") == 0
        assert getattr(report, prefix + "exact_observed_increment_bounds") == (
            -sum(radii),
            sum(radii),
        )
        assert getattr(report, prefix + "observed_sign") == "indeterminate"
    absent = _bound(
        bridge_weight_bounds=(0, 0), inverse_capacity_reading_error_bounds=None
    )
    assert absent.inverse_capacity_exact_true_increment_bounds == (0, 0)
    assert absent.inverse_capacity_exact_observed_increment_bounds is None
    assert absent.inverse_capacity_observed_increment_bounds is None
    assert absent.inverse_capacity_observed_sign == "not_requested"
    for legacy in (
        owner._prepared_bridge_exchange,
        owner._prepared_capacity_bridge_exchange,
    ):
        values = _arguments(bridge_weight_bounds=(0, 0))
        if legacy is owner._prepared_bridge_exchange:
            values = {
                name: value for name, value in values.items() if "capacity" not in name
            }
        with pytest.raises(ValueError, match="strictly positive"):
            legacy(**values)


@pytest.mark.parametrize("epsilon", (Q(1, 2), Q(1, 8), Q(1, 32), Q(1, 2**200)))
def test_zero_containing_families_shrink_exactly_without_a_uniform_strict_sign(epsilon):
    report = _bound(
        bridge_weight_bounds=(0, epsilon),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    half = _bound(
        bridge_weight_bounds=(0, epsilon / 2),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    for prefix in ("", "inverse_capacity_"):
        full_bounds = getattr(report, prefix + "exact_true_increment_bounds")
        half_bounds = getattr(half, prefix + "exact_true_increment_bounds")
        assert full_bounds[0] < 0 < full_bounds[1]
        assert tuple(2 * value for value in half_bounds) == full_bounds
        assert getattr(report, prefix + "observed_sign") == "indeterminate"
        assert getattr(half, prefix + "observed_sign") == "indeterminate"
        assert getattr(report, prefix + "true_increment_bounds").contains(
            I(*full_bounds)
        )


def test_subgrid_positive_signal_is_distinct_from_display_roundoff_and_noise():
    tiny = Q(1, 2**200)
    clean = _bound(
        bridge_weight_bounds=(tiny, tiny),
        reading_error_bounds=(0,) * 4,
        inverse_capacity_reading_error_bounds=(0,) * 4,
    )
    noisy = _bound(bridge_weight_bounds=(tiny, tiny))
    for prefix in ("", "inverse_capacity_"):
        endpoints = getattr(clean, prefix + "exact_true_increment_bounds")
        assert endpoints[0] < endpoints[1] < 0
        assert getattr(clean, prefix + "observed_sign") == "negative"
        assert getattr(clean, prefix + "true_increment_bounds").contains(0)
        assert getattr(noisy, prefix + "observed_sign") == "indeterminate"
        assert getattr(noisy, prefix + "exact_true_increment_bounds") == endpoints
        assert max(map(abs, endpoints)) < Q(1, 2**128)


@pytest.mark.parametrize("common", (Q(1), Q(1, 4), Q(3, 2)))
@pytest.mark.parametrize("gamma", ((0, 0), (Q(1, 4), Q(1, 3))))
def test_common_capacity_reduces_in_both_rows_under_one_clock(common, gamma):
    values = _arguments(
        capacity_bounds=(common, common),
        donor_port_capacity_bounds=(common, common),
        donor_internal_capacity_contrast_upper_bound=0,
        inverse_capacity_mass_bounds=(5 / common, 5 / common),
        gamma_bounds=gamma,
    )
    report = owner._prepared_matched_capacity_bridge_exchange(**values)
    unit_values = dict(values)
    unit_values.update(
        capacity_bounds=(1, 1),
        donor_port_capacity_bounds=(1, 1),
        inverse_capacity_mass_bounds=(5, 5),
        elapsed_time=common * values["elapsed_time"],
    )
    unit = owner._prepared_matched_capacity_bridge_exchange(**unit_values)
    for field in (
        "form_upper_bound",
        "form_displacement_upper_bound",
        "phase_displacement_upper_bound",
        "increment_remainder_upper_bound",
        "inverse_capacity_increment_remainder_upper_bound",
        "retention_margins",
        "difference_growth_exponent",
        "difference_growth_upper_bound",
        "matched_state_difference_upper_bound",
    ):
        assert getattr(report, field) == getattr(unit, field)
    # Coefficient interval construction can differ in arithmetic order; the
    # exact conductance-last scaling retains only that bounded grid difference.
    for prefix in ("", "inverse_capacity_"):
        for field in ("exact_true_increment_bounds", "exact_observed_increment_bounds"):
            candidate, reference = getattr(report, prefix + field), getattr(
                unit, prefix + field
            )
            assert max(abs(a - b) for a, b in zip(candidate, reference)) < Q(1, 2**110)
        assert getattr(report, prefix + "observed_sign") == getattr(
            unit, prefix + "observed_sign"
        )


def test_correlated_four_readings_have_separate_exact_error_extrema():
    arithmetic = (Q(1, 10**7), Q(2, 10**7), Q(3, 10**7), Q(4, 10**7))
    weighted = (Q(4, 10**7), Q(3, 10**7), Q(2, 10**7), Q(2, 10**7))
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
        low, high = getattr(report, prefix + "exact_true_increment_bounds")
        assert getattr(report, prefix + "exact_observed_increment_bounds") == (
            low + min(corners),
            high + max(corners),
        )
    absent = _bound(inverse_capacity_reading_error_bounds=None)
    baseline = _bound()
    assert (
        absent.exact_observed_increment_bounds
        == baseline.exact_observed_increment_bounds
    )
    assert (
        absent.inverse_capacity_exact_true_increment_bounds
        == baseline.inverse_capacity_exact_true_increment_bounds
    )
    assert absent.inverse_capacity_exact_observed_increment_bounds is None


def test_chart_retention_is_independent_of_both_observations():
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
        assert (
            report.exact_observed_increment_bounds
            == baseline.exact_observed_increment_bounds
        )
        assert (
            report.inverse_capacity_exact_observed_increment_bounds
            == baseline.inverse_capacity_exact_observed_increment_bounds
        )
    noisy = _bound(
        reading_error_bounds=(1,) * 4, inverse_capacity_reading_error_bounds=(1,) * 4
    )
    assert (
        noisy.observed_sign == noisy.inverse_capacity_observed_sign == "indeterminate"
    )
    assert noisy.retention_status == "certified"


@pytest.mark.parametrize(
    "name,value",
    INVALID
    + (
        ("bridge_weight_bounds", (-1, 0)),
        ("bridge_weight_bounds", (False, 1)),
        ("bridge_weight_bounds", (1, 0)),
        ("bridge_weight_bounds", (0, float("inf"))),
        ("bridge_weight_bounds", (0, 4)),
    ),
)
def test_invalid_raw_premises_reject_before_numerical_work(name, value, monkeypatch):
    def numerical_work(*args, **kwargs):
        pytest.fail("invalid matched premise reached interval, sine or exponential")

    for field in ("I", "sin", "_negative_exp_bounds"):
        monkeypatch.setattr(owner, field, numerical_work)
    with pytest.raises((TypeError, ValueError)):
        _bound(**{name: value})


def test_zero_response_does_not_bypass_the_exponential_work_domain(monkeypatch):
    def numerical_work(*args, **kwargs):
        pytest.fail("over-budget matched request reached numerical work")

    for field in ("I", "sin", "_negative_exp_bounds"):
        monkeypatch.setattr(owner, field, numerical_work)
    with pytest.raises(ValueError, match="4096"):
        _bound(bridge_weight_bounds=(0, 0), elapsed_time=10000)


def test_exponential_work_boundary_is_inclusive_without_evaluating_its_large_bound(
    monkeypatch,
):
    class BoundaryReached(Exception):
        pass

    def stop_at_boundary(exponent):
        assert exponent == 4096
        raise BoundaryReached

    monkeypatch.setattr(owner, "_negative_exp_bounds", stop_at_boundary)
    with pytest.raises(BoundaryReached):
        _bound(
            bridge_weight_bounds=(0, 0),
            capacity_bounds=(1, 1),
            donor_port_capacity_bounds=(1, 1),
            donor_internal_capacity_contrast_upper_bound=0,
            inverse_capacity_mass_bounds=None,
            gamma_bounds=(1, 1),
            weighted_degree_upper_bound=2,
            elapsed_time=4096,
        )


def test_previous_positive_reader_keeps_its_domain_without_new_exponential_work(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("previous independent-motion reader acquired exponential work")

    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    report = owner._prepared_capacity_bridge_exchange(**_arguments(elapsed_time=10000))
    assert report.elapsed_time == 10000
    assert report.observed_sign == "indeterminate"


@pytest.mark.parametrize("gamma", ((Q(1, 4), Q(1, 3)), (Q(1), Q(2))))
def test_positive_gamma_coefficients_retain_both_rows_and_rigorous_growth(gamma, mp):
    report = _bound(gamma_bounds=gamma)
    maximum, degree, g, initial, h = CAPACITY[1], Q(3), gamma[1], Q(1, 2), Q(1, 100)
    # Derive forcing coefficients by summing the independent complete bridge
    # row ceilings: M*(2B+g)/4 and M*g*B/2. Integrate their powers of time.
    b = (initial, maximum * degree * g / 4)
    form_forcing = (maximum * (2 * b[0] + g) / 4, maximum * b[1] / 2)
    phase_forcing = tuple(maximum * g * value / 2 for value in b)
    combined = tuple(a + p for a, p in zip(form_forcing, phase_forcing))
    assert report.difference_forcing_constant_upper_bound == combined[0]
    assert report.difference_forcing_linear_upper_bound == combined[1]
    exponent = g * maximum * degree * h / 2
    assert report.difference_growth_exponent == exponent
    exponential = mp.exp(_mp(mp, exponent))
    assert _mp(mp, report.difference_growth_upper_bound) >= exponential
    assert _mp(mp, report.difference_growth_upper_bound) - exponential < mp.mpf("1e-80")
    integral = sum(
        (
            coefficient * h ** (power + 1) / (power + 1)
            for power, coefficient in enumerate(combined)
        ),
        Q(0),
    )
    double_integral = sum(
        (
            coefficient * h ** (power + 2) / ((power + 1) * (power + 2))
            for power, coefficient in enumerate(combined)
        ),
        Q(0),
    )
    assert (
        report.matched_state_difference_upper_bound
        == Q(3, 4) * report.difference_growth_upper_bound * integral
    )
    assert (
        report.integrated_matched_state_difference_upper_bound
        == Q(3, 4) * report.difference_growth_upper_bound * double_integral
    )
    assert (
        report.matched_internal_increment_remainder_upper_bound
        == CONTRAST
        * max(Q(1), g)
        * report.integrated_matched_state_difference_upper_bound
        / 10
    )
    assert report.increment_remainder_upper_bound == min(
        report.matched_increment_remainder_upper_bound,
        report.independent_increment_remainder_upper_bound,
    )


def test_exact_subgrid_mass_is_admitted_before_reciprocal_display():
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
    report = owner._prepared_matched_capacity_bridge_exchange(**values)
    assert report.effective_inverse_capacity_mass_bounds == (Q(1, 2**200),) * 2
    assert report.exact_true_increment_bounds == (0, 0)
    assert report.inverse_capacity_exact_true_increment_bounds == (0, 0)


def test_frozen_output_and_copied_primitives_do_not_accept_a_cached_verdict():
    weights, errors = [Q(1, 2), Q(3, 4)], [Q(1, 10**7)] * 4
    report = _bound(bridge_weight_bounds=weights, reading_error_bounds=errors)
    weights[0], errors[-1] = -1, 100
    assert report.bridge_weight_bounds == (Q(1, 2), Q(3, 4))
    assert report.reading_error_bounds == (Q(1, 10**7),) * 4
    with pytest.raises(FrozenInstanceError):
        report.observed_sign = "positive"
    for bad in (report, I(0, 1), {"lo": 0, "hi": 1, "certified": True}):
        with pytest.raises((TypeError, ValueError)):
            _bound(bridge_weight_bounds=bad)
    assert _bound(reading_error_bounds=errors).observed_sign == "indeterminate"


def test_ordered_iterables_stop_at_the_raw_work_boundary():
    consumed = []

    def endless():
        for index in range(10000):
            consumed.append(index)
            yield Q(1)

    with pytest.raises(ValueError):
        _bound(bridge_weight_bounds=endless())
    assert len(consumed) == 3
    consumed.clear()
    with pytest.raises(ValueError):
        _bound(inverse_capacity_reading_error_bounds=endless())
    assert len(consumed) == 5


@pytest.mark.parametrize(
    "missing",
    (
        "capacity_bounds",
        "donor_port_capacity_bounds",
        "reading_error_bounds",
        "donor_node_count",
    ),
)
def test_required_matched_premises_have_no_inferred_defaults(missing):
    values = _arguments()
    del values[missing]
    with pytest.raises(TypeError):
        owner._prepared_matched_capacity_bridge_exchange(**values)
