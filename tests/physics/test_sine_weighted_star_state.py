"""Full-state sufficiency and hidden-association controls without evolution."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_weighted_state import (
    assess_sine_weighted_star_retention,
    assess_sine_weighted_star_state,
    reconstruct_sine_weighted_star_state,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
WEIGHTS = (3, 3, 3, 2, 2, 2, 2)
DEGREES = (4, 2, 2, 2, 4)
FORMS = (Q(3, 7), Q(-1, 9), Q(2, 5), Q(-4, 11), Q(1, 13))
TURNS = (Q(1, 7), Q(-2, 9), Q(3, 11), Q(1, 13), Q(-2, 15))


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    exact_decimal = Q(mp.nstr(value, 150))
    assert interval.lo <= exact_decimal <= interval.hi


def _direct_edge_values(forms, turns, scale):
    """Independent complete edge sums and differentiated complex definition."""
    capacities = (4 * scale, 2 * scale, 2 * scale, 2 * scale, Q(8))
    sine_sums = [mp.mpf(0) for _ in forms]
    gradient = [Q(0) for _ in forms]
    form_storage, phase_storage = Q(0), mp.mpf(0)
    for (i, j), weight in zip(EDGES, WEIGHTS):
        delta = 2 * mp.pi * _mp(turns[j] - turns[i])
        sine_sums[i] += mp.sin(delta)
        sine_sums[j] -= mp.sin(delta)
        difference = forms[i] - forms[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        form_storage += weight * difference**2 / 2
        phase_storage += 1 - mp.cos(delta)
    pressure = [value / (mp.pi * degree) for value, degree in zip(sine_sums, DEGREES)]
    form_rates = [_mp(nu) * value for nu, value in zip(capacities, pressure)]
    phase_numerators = [
        nu * value / degree for nu, value, degree in zip(capacities, gradient, DEGREES)
    ]
    phase_rates = [_mp(value) / mp.pi for value in phase_numerators]
    mean_form = sum(forms[:4]) / 4
    mean_phase_turns = sum(turns[:4]) / 4
    internal_form = tuple(value - mean_form for value in forms[:4])
    internal_phase_turns = tuple(value - mean_phase_turns for value in turns[:4])
    exponents = [mp.exp(2j * mp.pi * _mp(value)) for value in internal_phase_turns]
    moment = sum(exponents) / 4
    internal_phase_rates = [
        value - sum(phase_rates[:4]) / 4 for value in phase_rates[:4]
    ]
    moment_rate = (
        1j
        * sum(
            value * exponential
            for value, exponential in zip(internal_phase_rates, exponents)
        )
        / 4
    )
    contact_mean = (
        sum(mp.sin(2 * mp.pi * _mp(turns[4] - value)) for value in turns[:4]) / 4
    )
    internal_form_storage = sum(value**2 for value in internal_form) + Q(3, 2) * sum(
        (internal_form[0] - internal_form[j]) ** 2 for j in (1, 2, 3)
    )
    internal_phase_storage = sum(
        1 - mp.cos(2 * mp.pi * _mp(turns[j] - turns[0])) for j in (1, 2, 3)
    )
    reference_storage = 4 * _mp(mean_form - forms[4]) ** 2 + 4 * (
        1 - mp.cos(2 * mp.pi * _mp(turns[4] - mean_phase_turns))
    )
    return {
        "pressure": pressure,
        "form_rates": form_rates,
        "phase_numerators": tuple(phase_numerators),
        "phase_rates": phase_rates,
        "moment": moment,
        "moment_rate": moment_rate,
        "contact_mean": contact_mean,
        "form_storage": form_storage,
        "phase_storage": phase_storage,
        "internal_form_storage": internal_form_storage,
        "internal_phase_storage": internal_phase_storage,
        "reference_storage": reference_storage,
    }


@pytest.fixture(scope="module")
def report():
    return assess_sine_weighted_star_state(
        capacity_scale=Q(23, 25), form=FORMS, phase_turns=TURNS
    )


def test_mean_internal_coordinates_reconstruct_every_signed_form_and_declared_lift(
    report,
):
    assert report.form == FORMS
    assert report.phase_turns == TURNS
    assert report.collective_form == (sum(FORMS[:4]) / 4, FORMS[4])
    assert report.collective_phase_turns == (sum(TURNS[:4]) / 4, TURNS[4])
    assert sum(report.internal_form) == sum(report.internal_phase_turns) == 0
    assert (
        tuple(report.collective_form[0] + value for value in report.internal_form)
        + (report.collective_form[1],)
        == FORMS
    )
    assert (
        tuple(
            report.collective_phase_turns[0] + value
            for value in report.internal_phase_turns
        )
        + (report.collective_phase_turns[1],)
        == TURNS
    )
    reconstructed = reconstruct_sine_weighted_star_state(
        capacity_scale=report.capacity_scale,
        collective_form=report.collective_form,
        collective_phase_turns=report.collective_phase_turns,
        internal_form=report.internal_form,
        internal_phase_turns=report.internal_phase_turns,
    )
    assert reconstructed == report
    assert report.exact_row_reconstruction_certified
    with pytest.raises(FrozenInstanceError):
        report.form = (0,) * 5


@pytest.mark.parametrize("scale", (Q(23, 25), Q(1), Q(7, 5)))
def test_nonlinear_rows_pressure_moments_and_storage_match_independent_edges(scale):
    report = assess_sine_weighted_star_state(
        capacity_scale=scale, form=FORMS, phase_turns=TURNS
    )
    assert report.capacity == (4 * scale, 2 * scale, 2 * scale, 2 * scale, 8)
    with mp.workdps(180):
        expected = _direct_edge_values(FORMS, TURNS, scale)
        assert report.phase_rate_numerators == expected["phase_numerators"]
        assert report.form_storage == expected["form_storage"]
        assert report.internal_form_storage == expected["internal_form_storage"]
        for actual, values in (
            (report.pressure_bounds, expected["pressure"]),
            (report.form_rate_bounds, expected["form_rates"]),
            (report.phase_rate_bounds, expected["phase_rates"]),
        ):
            for enclosure, value in zip(actual, values):
                _contains(enclosure, value)
        for key, actual in (
            ("phase_storage", report.phase_storage_bounds),
            ("internal_phase_storage", report.internal_phase_storage_bounds),
            ("reference_storage", report.collective_reference_storage_bounds),
        ):
            _contains(actual, expected[key])
        total = _mp(expected["form_storage"]) + expected["phase_storage"]
        _contains(report.full_storage_bounds, total)
        _contains(
            report.storage_remainder_bounds, total - expected["reference_storage"]
        )
        _contains(report.moment_real_bounds, expected["moment"].real)
        _contains(report.moment_imag_bounds, expected["moment"].imag)
        _contains(report.moment_real_rate_bounds, expected["moment_rate"].real)
        _contains(report.moment_imag_rate_bounds, expected["moment_rate"].imag)
        _contains(report.contact_sine_mean_bounds, expected["contact_mean"])
        mean_dx = sum(expected["form_rates"][:4]) / 4
        mean_dtheta = sum(expected["phase_rates"][:4]) / 4
        form_gap = _mp(sum(FORMS[:4]) / 4 - FORMS[4])
        phase_gap = 2 * mp.pi * _mp(TURNS[4] - sum(TURNS[:4]) / 4)
        gap_rate = mean_dx - expected["form_rates"][4]
        phase_gap_rate = expected["phase_rates"][4] - mean_dtheta
        reference_storage_rate = (
            8 * form_gap * gap_rate + 4 * mp.sin(phase_gap) * phase_gap_rate
        )
        _contains(
            report.collective_reference_storage_rate_bounds, reference_storage_rate
        )
        _contains(report.storage_remainder_rate_bounds, -reference_storage_rate)
        for index in range(4):
            _contains(
                report.internal_form_rate_bounds[index],
                expected["form_rates"][index] - mean_dx,
            )
            _contains(
                report.internal_phase_rate_bounds[index],
                expected["phase_rates"][index] - mean_dtheta,
            )
        for enclosures, values in (
            (report.collective_form_rate_bounds, (mean_dx, expected["form_rates"][4])),
            (
                report.collective_phase_rate_bounds,
                (mean_dtheta, expected["phase_rates"][4]),
            ),
        ):
            for enclosure, value in zip(enclosures, values):
                _contains(enclosure, value)
    assert report.exact_row_reconstruction_certified
    assert report.storage_rate_bounds == I(0)
    assert (
        report.storage_remainder_rate_bounds
        == -report.collective_reference_storage_rate_bounds
    )
    assert report.weighted_form_mean_rate_bounds == I(0)
    assert report.weighted_phase_mean_rate_bounds == I(0)
    weights = (1 / scale,) * 4 + (Q(1, 2),)
    total_weight = sum(weights)
    assert (
        report.weighted_form_mean
        == sum(weight * value for weight, value in zip(weights, FORMS)) / total_weight
    )
    assert (
        report.weighted_phase_mean_turns
        == sum(weight * value for weight, value in zip(weights, TURNS)) / total_weight
    )


def test_family_limit_is_the_previous_collective_law_with_no_internal_state():
    scale = Q(23, 25)
    report = assess_sine_weighted_star_state(
        capacity_scale=scale,
        form=(Q(1, 4),) * 4 + (Q(-1, 8),),
        phase_turns=(Q(1, 16),) * 4 + (Q(3, 16),),
    )
    assert report.internal_form == (0,) * 4
    assert report.internal_phase_turns == (0,) * 4
    assert report.moment_real_bounds == I(1)
    assert report.moment_imag_bounds == I(0)
    assert report.moment_real_rate_bounds == report.moment_imag_rate_bounds == I(0)
    assert report.internal_form_storage == 0
    assert report.internal_phase_storage_bounds == I(0)
    assert report.internal_phase_rate_numerators == (0,) * 4
    assert report.phase_rate_numerators == (2 * scale * Q(3, 8),) * 4 + (-6,)
    with mp.workdps(180):
        _contains(
            report.collective_form_rate_bounds[0], _mp(scale) * mp.sqrt(2) / (2 * mp.pi)
        )
        _contains(report.collective_form_rate_bounds[1], -4 * mp.sqrt(2) / mp.pi)
        _contains(report.full_storage_bounds, mp.mpf(9) / 16 + 4 - 2 * mp.sqrt(2))
    assert report.storage_remainder_bounds.contains(0)


def test_same_coarse_coordinates_and_storage_can_have_different_coarse_derivatives():
    phase = (Q(1, 4), Q(-1, 12), Q(-1, 12), Q(-1, 12), 0)
    plus = assess_sine_weighted_star_state(form=(0,) * 5, phase_turns=phase)
    minus = assess_sine_weighted_star_state(
        form=(0,) * 5, phase_turns=tuple(-value for value in phase)
    )
    assert plus.collective_form == minus.collective_form == (0, 0)
    assert plus.collective_phase_turns == minus.collective_phase_turns == (0, 0)
    assert plus.weighted_form_mean == minus.weighted_form_mean == 0
    assert plus.weighted_phase_mean_turns == minus.weighted_phase_mean_turns == 0
    # Every edge difference changes sign, so all cosine costs agree exactly.
    assert plus.form_storage == minus.form_storage == 0
    assert plus.phase_storage_bounds == minus.phase_storage_bounds
    assert plus.moment_imag_bounds.hi < 0 < minus.moment_imag_bounds.lo
    assert plus.collective_form_rate_bounds[0].lo > 0
    assert minus.collective_form_rate_bounds[0].hi < 0
    with mp.workdps(180):
        for state in (plus, minus):
            _contains(state.full_storage_bounds, mp.mpf(17) / 2 - 3 * mp.sqrt(3) / 2)
        _contains(plus.collective_form_rate_bounds[0], 1 / (8 * mp.pi))
        _contains(minus.collective_form_rate_bounds[0], -1 / (8 * mp.pi))


def test_same_coarse_moment_and_storage_omit_the_form_phase_association():
    scale = Q(23, 25)
    phase = (0, Q(1, 4), Q(-1, 4), 0, Q(1, 4))
    form = (0, Q(1, 8), Q(-1, 8), 0, 0)
    plus = assess_sine_weighted_star_state(
        capacity_scale=scale, form=form, phase_turns=phase
    )
    minus = assess_sine_weighted_star_state(
        capacity_scale=scale, form=tuple(-value for value in form), phase_turns=phase
    )
    for field in (
        "collective_form",
        "collective_phase_turns",
        "weighted_form_mean",
        "weighted_phase_mean_turns",
        "moment_real_bounds",
        "moment_imag_bounds",
        "full_storage_bounds",
        "collective_form_rate_bounds",
        "collective_phase_rate_bounds",
    ):
        assert getattr(plus, field) == getattr(minus, field)
    assert plus.moment_real_bounds == I(Q(1, 2))
    assert plus.moment_imag_bounds == I(0)
    assert plus.full_storage_bounds == I(Q(389, 64))
    assert plus.moment_real_rate_bounds.hi < 0 < minus.moment_real_rate_bounds.lo
    assert plus.moment_imag_rate_bounds == minus.moment_imag_rate_bounds == I(0)
    with mp.workdps(180):
        _contains(plus.moment_real_rate_bounds, -5 * _mp(scale) / (16 * mp.pi))
        _contains(minus.moment_real_rate_bounds, 5 * _mp(scale) / (16 * mp.pi))
    # An autonomous derivative of the augmented observation (coarse,M,H)
    # would have to agree on this fiber. The disjoint M-dot intervals refute it.


def test_individual_full_turn_lift_change_preserves_circular_rows_but_transports_mean(
    report,
):
    turns = (TURNS[0] + 1,) + TURNS[1:]
    changed = assess_sine_weighted_star_state(
        capacity_scale=report.capacity_scale, form=FORMS, phase_turns=turns
    )
    assert changed.collective_phase_turns[0] == report.collective_phase_turns[0] + Q(
        1, 4
    )
    assert changed.collective_phase_turns[1] == report.collective_phase_turns[1]
    assert changed.internal_phase_turns == (
        report.internal_phase_turns[0] + Q(3, 4),
    ) + tuple(value - Q(1, 4) for value in report.internal_phase_turns[1:])
    for field in (
        "pressure_bounds",
        "form_rate_bounds",
        "phase_rate_bounds",
        "full_storage_bounds",
        "contact_sine_mean_bounds",
    ):
        assert getattr(changed, field) == getattr(report, field)
    assert changed.moment_real_bounds == report.moment_imag_bounds
    assert changed.moment_imag_bounds == -report.moment_real_bounds
    assert changed.weighted_phase_mean_turns - report.weighted_phase_mean_turns == (
        (1 / report.capacity_scale) / (4 / report.capacity_scale + Q(1, 2))
    )


def test_declared_lift_storage_remainder_can_be_strictly_negative():
    report = assess_sine_weighted_star_state(
        form=(0,) * 5,
        phase_turns=(Q(3, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4)),
    )
    assert report.collective_phase_turns == (0, Q(-1, 4))
    assert report.moment_real_bounds == I(0)
    assert report.moment_imag_bounds == I(-1)
    assert report.form_rate_bounds == report.phase_rate_bounds == (I(0),) * 5
    assert report.full_storage_bounds == report.internal_phase_storage_bounds == I(0)
    assert report.collective_reference_storage_bounds == I(4)
    assert report.storage_remainder_bounds == I(-4)


def test_zero_complex_moment_is_valid_without_inventing_a_resultant_angle():
    phase = (0, Q(1, 4), Q(1, 2), Q(3, 4), Q(1, 7))
    report = assess_sine_weighted_star_state(form=FORMS, phase_turns=phase)
    assert report.moment_real_bounds == report.moment_imag_bounds == I(0)
    assert report.contact_sine_mean_bounds == I(0)
    assert report.collective_form_rate_bounds == (I(0), I(0))
    # The four root phases cancel at B exactly; a transcendental floating
    # evaluation of that zero can leave a residual far below its precision.
    assert report.pressure_bounds[4] == I(0)
    assert report.collective_phase_turns == (Q(3, 8), Q(1, 7))
    assert report.exact_row_reconstruction_certified
    with mp.workdps(180):
        expected = _direct_edge_values(FORMS, phase, Q(1))
        _contains(report.moment_real_rate_bounds, expected["moment_rate"].real)
        _contains(report.moment_imag_rate_bounds, expected["moment_rate"].imag)
        for enclosure, value in zip(
            report.pressure_bounds[:4], expected["pressure"][:4]
        ):
            _contains(enclosure, value)


@pytest.mark.parametrize("field", ("form", "phase_turns"))
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), complex(1, 2)))
def test_invalid_primitive_coordinates_cannot_be_coerced(field, value):
    arguments = {"form": FORMS, "phase_turns": TURNS}
    arguments[field] = (value,) + arguments[field][1:]
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_state(**arguments)


@pytest.mark.parametrize(
    "scale", (0, -1, True, float("nan"), float("inf"), complex(1, 0))
)
def test_capacity_scale_is_finite_strictly_positive_and_nonboolean(scale):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_state(
            capacity_scale=scale, form=FORMS, phase_turns=TURNS
        )


@pytest.mark.parametrize(
    "forms,turns",
    (
        (FORMS[:-1], TURNS),
        (FORMS + (0,), TURNS),
        (FORMS, TURNS[:-1]),
        (FORMS, (0.0,) * 5),
    ),
)
def test_coordinate_order_length_and_exact_turn_domain(forms, turns):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_state(form=forms, phase_turns=turns)


@pytest.mark.parametrize(
    "mutation",
    (
        {"collective_form": (0,)},
        {"collective_phase_turns": (0,)},
        {"internal_form": (1, 0, 0, 0)},
        {"internal_phase_turns": (0, 0, 0, Q(1, 4))},
        {"internal_form": (0.1, 0.2, -0.3, 0)},
        {"internal_form": (False, 0, 0, 0)},
        {"internal_phase_turns": (0.0,) * 4},
    ),
)
def test_reconstruction_requires_exact_constrained_internal_coordinates(mutation):
    arguments = {
        "collective_form": (0, 0),
        "collective_phase_turns": (0, 0),
        "internal_form": (0,) * 4,
        "internal_phase_turns": (0,) * 4,
    }
    with pytest.raises((TypeError, ValueError)):
        reconstruct_sine_weighted_star_state(**(arguments | mutation))


def test_signed_exact_and_represented_forms_are_kept_without_magnitude_projection():
    forms = (Q(-1, 3), 0.1, -0.1, 0, -2)
    report = assess_sine_weighted_star_state(form=forms, phase_turns=(0,) * 5)
    assert report.form == (Q(-1, 3), Q.from_float(0.1), Q.from_float(-0.1), 0, -2)
    assert report.collective_form[0] == Q(-1, 12)
    assert report.phase_rate_numerators != (0,) * 5


def test_sdk_export_rebuilds_the_full_state_from_primitives_before_atomic_write(
    report, tmp_path
):
    poisoned = replace(
        report,
        collective_form=(),
        internal_form=(),
        collective_phase_turns=(),
        internal_phase_turns=(),
        form_rate_bounds=(I(999),),
        moment_real_rate_bounds=I(-999),
        full_storage_bounds=I(-999),
        exact_row_reconstruction_certified=False,
    )
    generic = relational_report_to_dict(poisoned)
    assert generic["report"] == report.to_dict()["report"]
    assert generic["report_type"] == type(report).__name__
    destination = tmp_path / "weighted-star-state.json"
    export_to_json(generic, destination)
    assert json_loads(destination.read_text(encoding="utf-8")) == generic
    assert generic["report"]["form"][0] == {"numerator": 3, "denominator": 7}
    assert generic["report"]["phase_turns"][0] == {"numerator": 1, "denominator": 7}
    original = destination.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(replace(report, capacity_scale=False)),
            destination,
        )
    assert destination.read_bytes() == original


def _dirichlet_storage(forms):
    return sum(
        (
            Q(weight, 2) * (forms[i] - forms[j]) ** 2
            for (i, j), weight in zip(EDGES, WEIGHTS)
        ),
        Q(0),
    )


@pytest.mark.parametrize("leaf", (1, 2, 3))
def test_sharp_form_boundary_minimizer_is_not_a_strict_retention_certificate(leaf):
    forms = [Q(1, 2), Q(7, 16), Q(7, 16), Q(7, 16), Q(11, 32)]
    forms[leaf] = Q(0)
    gradient = [Q(0)] * 5
    laplacian = [[Q(0)] * 5 for _ in range(5)]
    for (i, j), weight in zip(EDGES, WEIGHTS):
        gradient[i] += weight * (forms[i] - forms[j])
        gradient[j] -= weight * (forms[i] - forms[j])
        laplacian[i][i] += weight
        laplacian[j][j] += weight
        laplacian[i][j] -= weight
        laplacian[j][i] -= weight
    free = tuple(index for index in range(5) if index not in (0, leaf))
    assert all(gradient[index] == 0 for index in free)
    assert gradient[0] == -gradient[leaf]
    effective_conductance = gradient[0] / (forms[0] - forms[leaf])
    assert effective_conductance == Q(35, 8)
    hessian = tuple(tuple(laplacian[i][j] for j in free) for i in free)
    assert hessian == ((5, 0, -2), (0, 5, -2), (-2, -2, 8))
    a, b, c = hessian
    minors = (
        a[0],
        a[0] * b[1] - a[1] * b[0],
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]),
    )
    assert minors == (5, 25, 160)
    assert all(value > 0 for value in minors)
    # Harmonic free rows and positive Dirichlet Hessian prove this is the
    # global constrained minimum, rather than just one sampled boundary.
    energy = _dirichlet_storage(forms)
    assert energy == effective_conductance * (forms[0] - forms[leaf]) ** 2 / 2
    state = assess_sine_weighted_star_state(form=forms, phase_turns=(0,) * 5)
    retention = assess_sine_weighted_star_retention(state)
    assert energy == retention.form_boundary_storage == Q(35, 64)
    assert retention.storage_margin == 0
    assert retention.internal_form_gaps[leaf - 1] == Q(1, 2)
    assert retention.internal_form_gap_upper_bound == I(Q(1, 2))
    assert not retention.source_identity_admitted
    assert not retention.all_time_identity_retention_certified
    assert retention.status == "unavailable"


def test_below_boundary_state_has_a_strict_full_energy_certificate():
    forms = (Q(1, 4), 0, Q(7, 32), Q(7, 32), Q(11, 64))
    state = assess_sine_weighted_star_state(
        capacity_scale=Q(23, 25), form=forms, phase_turns=(0,) * 5
    )
    retention = assess_sine_weighted_star_retention(state)
    assert state.form_storage == Q(35, 256)
    assert retention.source_identity_admitted
    assert retention.all_time_identity_retention_certified
    assert retention.storage_margin == Q(105, 256)
    assert retention.internal_form_gap_upper_bound == I(Q(1, 4))
    assert retention.internal_phase_cosine_lower_bound == Q(221, 256)
    assert retention.phase_boundary_storage_lower_bound == 1
    assert retention.unavailable_reasons == ()
    assert (
        retention.temporal_scope
        == "all_real_times_under_the_unchanged_complete_held_law"
    )
    # The state is genuinely off the zero-internal-coordinate family.
    assert state.internal_form != (0,) * 4


def test_phase_boundary_is_circular_and_retention_uses_full_storage():
    equilibrium = assess_sine_weighted_star_state(
        form=(0,) * 5,
        phase_turns=(Q(3, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4)),
    )
    retained = assess_sine_weighted_star_retention(equilibrium)
    assert equilibrium.storage_remainder_bounds == I(-4)
    assert retained.internal_phase_gaps_turns == (0,) * 3
    assert retained.all_time_identity_retention_certified
    assert retained.storage_margin == Q(35, 64)
    boundary = assess_sine_weighted_star_retention(
        assess_sine_weighted_star_state(
            form=(0,) * 5, phase_turns=(0, Q(1, 4), 0, 0, 0)
        )
    )
    assert boundary.internal_phase_gaps_turns[0] == Q(1, 4)
    assert boundary.state.full_storage_bounds == I(2)
    assert not boundary.source_identity_admitted
    assert boundary.status == "unavailable"


def test_large_energy_can_keep_initial_identity_without_this_sufficient_certificate():
    # This even lies in the exact invariant family. Failure of the stronger
    # full-state sublevel test cannot be called identity loss or instability.
    state = assess_sine_weighted_star_state(form=(0, 0, 0, 0, 1), phase_turns=(0,) * 5)
    retention = assess_sine_weighted_star_retention(state)
    assert state.internal_form == state.internal_phase_turns == (0,) * 4
    assert retention.source_identity_admitted
    assert state.full_storage_bounds == I(4)
    assert not retention.all_time_identity_retention_certified
    assert retention.status == "unavailable"
    assert retention.unavailable_reasons == (
        "strict_full_storage_margin_not_certified",
    )


def test_finite_neighborhood_budget_and_off_family_point_near_exact_maximal_reference():
    epsilon = Q(1, 256)
    # There are seventeen conductance units and seven bare edges. Squared
    # endpoint-error bounds in the two storage channels give these constants.
    assert 2 * sum(WEIGHTS) == 34
    assert 2 * len(EDGES) == 14
    perturbation_squared = (2 * sum(WEIGHTS) + 2 * len(EDGES)) * epsilon**2
    assert perturbation_squared == Q(3, 4096)
    # (sqrt(1/2)+sqrt(3)/64)^2 < 1/2+5/128+3/4096,
    # using sqrt(6)<5/2, without replacing that radical by a float.
    assert 6 < Q(5, 2) ** 2
    ceiling = Q(1, 2) + Q(5, 128) + perturbation_squared
    assert ceiling == Q(2211, 4096)
    assert Q(35, 64) - ceiling == Q(29, 4096) > 0
    center = Q(725, 2048)
    forms = (
        center + Q(1, 2048),
        center - Q(1, 2048),
        center + Q(1, 4096),
        center - Q(1, 4096),
        0,
    )
    turns = (Q(1, 4096), Q(-1, 4096), 0, Q(1, 8192), Q(-1, 8192))
    # The exact reference is X_A=sqrt(2)/4, X_B=0, all phases0,
    # with H*=1/2. Rational square brackets certify every finite form error.
    lower, upper = max(value - epsilon for value in forms[:4]), min(
        value + epsilon for value in forms[:4]
    )
    assert 0 < lower and lower**2 < Q(1, 8) < upper**2
    assert max(abs(value) for value in turns) < Q(1, 2048)
    assert 2 * 4 * max(abs(value) for value in turns) < epsilon  # pi<4
    state = assess_sine_weighted_star_state(form=forms, phase_turns=turns)
    retention = assess_sine_weighted_star_retention(state)
    assert state.full_storage_bounds.lo > Q(1, 2)
    assert state.full_storage_bounds.hi < ceiling
    assert state.internal_form != (0,) * 4 and state.internal_phase_turns != (0,) * 4
    assert retention.all_time_identity_retention_certified


def test_retention_readmits_primitive_state_and_sdk_export_ignores_cached_verdicts(
    tmp_path,
):
    boundary = assess_sine_weighted_star_state(
        form=(Q(1, 2), 0, Q(7, 16), Q(7, 16), Q(11, 32)), phase_turns=(0,) * 5
    )
    poisoned_state = replace(boundary, full_storage_bounds=I(0), form_storage=Q(0))
    retention = assess_sine_weighted_star_retention(poisoned_state)
    assert retention.state.full_storage_bounds == I(Q(35, 64))
    assert not retention.all_time_identity_retention_certified
    forged = replace(
        retention,
        storage_margin=Q(999),
        source_identity_admitted=True,
        all_time_identity_retention_certified=True,
        status="certified_identity_retention",
        internal_form_gap_upper_bound=I(0),
    )
    generic = relational_report_to_dict(forged)
    assert generic["report"] == retention.to_dict()["report"]
    assert generic["report"]["all_time_identity_retention_certified"] is False
    assert generic["report"]["form_boundary_storage"] == {
        "numerator": 35,
        "denominator": 64,
    }
    destination = tmp_path / "weighted-star-retention.json"
    export_to_json(generic, destination)
    assert json_loads(destination.read_text(encoding="utf-8")) == generic
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_retention(replace(boundary, capacity_scale=False))
    with pytest.raises((TypeError, ValueError)):
        replace(retention, state=replace(boundary, phase_turns=(False,) * 5)).to_dict()
    with pytest.raises(TypeError):
        assess_sine_weighted_star_retention(object())
