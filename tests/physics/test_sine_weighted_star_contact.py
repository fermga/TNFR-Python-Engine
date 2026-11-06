"""Independent joined-law and maintenance controls without evolution or events."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_weighted_contact import (
    assess_sine_weighted_star_contact,
)
from tnfr.physics.relational_sine_weighted_state import assess_sine_weighted_star_state
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

STAR_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
EDGES = STAR_EDGES + tuple((i + 5, j + 5) for i, j in STAR_EDGES) + ((4, 9),)
WEIGHTS = (3, 3, 3, 2, 2, 2, 2) * 2 + (1,)
DEGREES = (4, 2, 2, 2, 5) * 2
CAPACITIES = (4, 2, 2, 2, 8) * 2
MEAN_WEIGHTS = (Q(1), Q(1), Q(1), Q(1), Q(5, 8)) * 2
FORMS = (
    Q(3, 7),
    Q(-1, 9),
    Q(2, 5),
    Q(-4, 11),
    Q(1, 13),
    Q(-2, 7),
    Q(3, 8),
    Q(-1, 5),
    Q(2, 9),
    Q(-3, 11),
)
TURNS = (
    Q(1, 7),
    Q(-2, 9),
    Q(3, 11),
    Q(1, 13),
    Q(-2, 15),
    Q(-2, 11),
    Q(1, 8),
    Q(2, 13),
    Q(-1, 17),
    Q(3, 16),
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    exact_decimal = Q(mp.nstr(value, 150))
    assert interval.lo <= exact_decimal <= interval.hi


def _state(forms, turns=(0,) * 5):
    return assess_sine_weighted_star_state(form=forms, phase_turns=turns)


def _contact(forms, turns=(0,) * 10):
    return assess_sine_weighted_star_contact(
        _state(forms[:5], turns[:5]), _state(forms[5:], turns[5:])
    )


@pytest.fixture(scope="module")
def generic_contact():
    return _contact(FORMS, TURNS)


@pytest.fixture(scope="module")
def exchanging_contact():
    forms = (Q(1, 16),) * 4 + (Q(0),)
    return assess_sine_weighted_star_contact(
        _state(forms), _state(forms, (Q(1, 16),) * 5)
    )


def _direct_field(forms, turns):
    """Assemble ten independent bare-sine and weighted-form incidence rows."""
    currents = [mp.mpf(0) for _ in forms]
    gradient = [Q(0) for _ in forms]
    form_stores, phase_stores = [], []
    for (i, j), weight in zip(EDGES, WEIGHTS):
        phase = 2 * mp.pi * _mp(turns[j] - turns[i])
        current = mp.sin(phase)
        difference = forms[i] - forms[j]
        currents[i] += current
        currents[j] -= current
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        form_stores.append(Q(weight, 2) * difference**2)
        phase_stores.append(1 - mp.cos(phase))
    pressure = [value / (mp.pi * degree) for value, degree in zip(currents, DEGREES)]
    form_rates = [value * nu for value, nu in zip(pressure, CAPACITIES)]
    phase_numerators = tuple(
        Q(nu, degree) * value
        for value, nu, degree in zip(gradient, CAPACITIES, DEGREES)
    )
    phase_rates = [_mp(value) / mp.pi for value in phase_numerators]
    edge_work = []
    for (i, j), weight in zip(EDGES, WEIGHTS):
        difference = _mp(forms[i] - forms[j])
        phase = 2 * mp.pi * _mp(turns[j] - turns[i])
        edge_work.append(
            weight * difference * (form_rates[i] - form_rates[j])
            + mp.sin(phase) * (phase_rates[j] - phase_rates[i])
        )
    return {
        "pressure": pressure,
        "form_rates": form_rates,
        "phase_rates": phase_rates,
        "phase_numerators": phase_numerators,
        "form_stores": form_stores,
        "phase_stores": phase_stores,
        "edge_work": edge_work,
    }


def test_complete_joined_rows_and_storage_match_independent_ten_node_sums(
    generic_contact,
):
    report = generic_contact
    assert report.nodes == tuple(range(10))
    assert report.edges == EDGES and report.conductances == WEIGHTS
    assert report.bare_degrees == DEGREES
    assert report.capacity == CAPACITIES
    assert report.form == FORMS and report.phase_turns == TURNS
    assert report.independent_continuous_dimension == 20
    assert report.exact_chart_reconstruction_certified
    with mp.workdps(180):
        expected = _direct_field(FORMS, TURNS)
        assert report.phase_rate_numerators == expected["phase_numerators"]
        for actual, values in (
            (report.pressure_bounds, expected["pressure"]),
            (report.form_rate_bounds, expected["form_rates"]),
            (report.phase_rate_bounds, expected["phase_rates"]),
        ):
            for enclosure, value in zip(actual, values):
                _contains(enclosure, value)
        form_store = sum(expected["form_stores"])
        phase_store = sum(expected["phase_stores"])
        assert report.form_storage == form_store
        assert report.bridge_form_storage == expected["form_stores"][-1]
        _contains(report.phase_storage_bounds, phase_store)
        _contains(report.full_storage_bounds, _mp(form_store) + phase_store)
        _contains(report.bridge_phase_storage_bounds, expected["phase_stores"][-1])
        _contains(
            report.support_work_bounds,
            _mp(expected["form_stores"][-1]) + expected["phase_stores"][-1],
        )
        for block, selected in enumerate((slice(0, 7), slice(7, 14))):
            _contains(
                report.component_storage_bounds[block],
                _mp(sum(expected["form_stores"][selected]))
                + sum(expected["phase_stores"][selected]),
            )
            _contains(
                report.component_storage_rate_bounds[block],
                sum(expected["edge_work"][selected]),
            )
        _contains(report.bridge_storage_rate_bounds, expected["edge_work"][-1])
        assert abs(sum(expected["edge_work"])) < mp.mpf("1e-170")
    assert report.storage_rate_bounds == I(0)
    assert report.weighted_form_mean_rate_bounds == I(0)
    assert report.weighted_phase_mean_rate_bounds == I(0)
    with pytest.raises(FrozenInstanceError):
        report.form = (0,) * 10


def test_actual_degree_means_and_component_charge_fluxes(generic_contact):
    report = generic_contact
    assert report.postcontact_mean_weights == MEAN_WEIGHTS
    assert sum(MEAN_WEIGHTS) == Q(37, 4)
    assert report.weighted_form_mean == sum(
        weight * value for weight, value in zip(MEAN_WEIGHTS, FORMS)
    ) / Q(37, 4)
    assert report.weighted_phase_mean_turns == sum(
        weight * value for weight, value in zip(MEAN_WEIGHTS, TURNS)
    ) / Q(37, 4)
    with mp.workdps(180):
        expected = _direct_field(FORMS, TURNS)
        current = mp.sin(2 * mp.pi * _mp(TURNS[9] - TURNS[4]))
        difference = FORMS[4] - FORMS[9]
        for indices, sign in ((range(5), 1), (range(5, 10), -1)):
            # Exact incidence cancellation leaves only the oriented bridge.
            measured_form_charge_rate = sum(
                _mp(MEAN_WEIGHTS[i]) * expected["form_rates"][i] for i in indices
            )
            assert abs(measured_form_charge_rate - sign * current / mp.pi) < mp.mpf(
                "1e-170"
            )
            assert (
                sum(MEAN_WEIGHTS[i] * report.phase_rate_numerators[i] for i in indices)
                == sign * difference
            )
        _contains(report.bridge_current_bounds, current)
    assert report.bridge_form_difference == FORMS[4] - FORMS[9]
    assert report.bridge_phase_gap_turns == TURNS[9] - TURNS[4]


def test_nonzero_exchange_has_positive_joint_margin_and_opposite_component_work(
    exchanging_contact,
):
    report = exchanging_contact
    assert report.source_individual_retention_certified == (True, True)
    assert report.component_storage_bounds == (I(Q(1, 64)),) * 2
    assert report.form_storage == Q(1, 32)
    assert report.bridge_form_storage == 0
    assert report.support_work_bounds.lo > 0
    assert report.source_joint_identity_admitted
    assert report.all_time_joint_identity_retention_certified
    assert report.status == "certified_joint_identity_retention"
    assert report.joint_storage_margin > 0 and report.unavailable_reasons == ()
    assert report.phase_rate_numerators == (Q(1, 8),) * 4 + (Q(-4, 5),) + (
        Q(1, 8),
    ) * 4 + (
        Q(-4, 5),
    )
    assert report.form_rate_bounds[:4] == report.form_rate_bounds[5:9] == (I(0),) * 4
    assert report.form_rate_bounds[4].lo > 0 > report.form_rate_bounds[9].hi
    assert report.component_storage_rate_bounds[0].hi < 0
    assert report.component_storage_rate_bounds[1].lo > 0
    assert report.bridge_storage_rate_bounds == report.storage_rate_bounds == I(0)
    with mp.workdps(180):
        current = mp.sin(mp.pi / 8)
        _contains(report.full_storage_bounds, mp.mpf(33) / 32 - mp.cos(mp.pi / 8))
        _contains(report.form_rate_bounds[4], 8 * current / (5 * mp.pi))
        _contains(report.form_rate_bounds[9], -8 * current / (5 * mp.pi))
        _contains(report.component_storage_rate_bounds[0], -4 * current / (5 * mp.pi))
        _contains(report.component_storage_rate_bounds[1], 4 * current / (5 * mp.pi))
    # The isolated reference B row is -1/pi. The joined B row is -4/(5pi),
    # even at zero bridge form gradient: the old four-neighbor denominator
    # cannot be reused for the internal contribution after contact.
    assert report.left.phase_rate_numerators[4] == -1
    assert report.phase_rate_numerators[4] == Q(-4, 5)
    assert report.left.form_rate_bounds == report.right.form_rate_bounds == (I(0),) * 5


def test_reversing_relative_phase_reverses_exchange_without_changing_work_budget(
    exchanging_contact,
):
    before = exchanging_contact
    opposite = assess_sine_weighted_star_contact(
        before.left, _state(before.right.form, (Q(-1, 16),) * 5)
    )
    assert opposite.full_storage_bounds == before.full_storage_bounds
    assert opposite.support_work_bounds == before.support_work_bounds
    assert opposite.phase_rate_numerators == before.phase_rate_numerators
    assert opposite.form_rate_bounds == tuple(
        -value for value in before.form_rate_bounds
    )
    assert opposite.component_storage_rate_bounds == tuple(
        -value for value in before.component_storage_rate_bounds
    )
    assert opposite.all_time_joint_identity_retention_certified
    quarter_turn = assess_sine_weighted_star_contact(
        before.left, _state(before.right.form, (Q(1, 4),) * 5)
    )
    assert quarter_turn.component_storage_bounds == (I(Q(1, 64)),) * 2
    assert quarter_turn.source_individual_retention_certified == (True, True)
    assert quarter_turn.support_work_bounds == I(1)
    assert quarter_turn.full_storage_bounds == I(Q(33, 32))
    assert quarter_turn.source_joint_identity_admitted
    assert not quarter_turn.all_time_joint_identity_retention_certified
    assert quarter_turn.status == "unavailable"


def test_insertion_work_is_an_exact_added_edge_cost_not_old_component_energy():
    left = _state((0,) * 5)
    right = _state((Q(1, 2),) * 5, (Q(1, 4),) * 5)
    report = assess_sine_weighted_star_contact(left, right)
    assert report.component_storage_bounds == (I(0), I(0))
    assert report.bridge_form_storage == Q(1, 8)
    assert report.bridge_phase_storage_bounds == I(1)
    assert report.support_work_bounds == report.full_storage_bounds == I(Q(9, 8))
    assert report.source_individual_retention_certified == (True, True)
    assert report.source_joint_identity_admitted
    assert not report.all_time_joint_identity_retention_certified
    assert report.status == "unavailable"
    assert report.joint_storage_margin == Q(-37, 64)
    assert report.phase_rate_numerators[4] == Q(-4, 5)
    assert report.phase_rate_numerators[9] == Q(4, 5)


def _dirichlet_data(forms, fixed):
    gradient = [Q(0)] * 10
    laplacian = [[Q(0)] * 10 for _ in range(10)]
    energy = Q(0)
    for (i, j), weight in zip(EDGES, WEIGHTS):
        difference = forms[i] - forms[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        energy += Q(weight, 2) * difference**2
        laplacian[i][i] += weight
        laplacian[j][j] += weight
        laplacian[i][j] -= weight
        laplacian[j][i] -= weight
    free = tuple(index for index in range(10) if index not in fixed)
    # Exact symmetric elimination checks the complete eight-dimensional
    # Dirichlet Hessian, rather than only the isolated three-coordinate one.
    reduced = [[laplacian[i][j] for j in free] for i in free]
    for k in range(len(free)):
        pivot = reduced[k][k]
        assert pivot > 0
        for i in range(k + 1, len(free)):
            for j in range(k + 1, len(free)):
                reduced[i][j] -= reduced[i][k] * reduced[k][j] / pivot
    return gradient, free, energy


@pytest.mark.parametrize("leaf", (1, 2, 3))
def test_sharp_joint_boundary_minimizer_extends_harmonically_through_second_star(leaf):
    first = [Q(1, 2), Q(7, 16), Q(7, 16), Q(7, 16), Q(11, 32)]
    first[leaf] = Q(0)
    forms = tuple(first) + (first[4],) * 5
    gradient, free, energy = _dirichlet_data(forms, (0, leaf))
    assert all(gradient[i] == 0 for i in free)
    assert gradient[0] == -gradient[leaf] == Q(35, 16)
    effective_conductance = gradient[0] / (forms[0] - forms[leaf])
    assert effective_conductance == Q(35, 8)
    assert energy == effective_conductance * (forms[0] - forms[leaf]) ** 2 / 2
    report = _contact(forms)
    assert report.form_storage == energy == report.joint_identity_storage_boundary
    assert energy == Q(35, 64)
    assert report.component_storage_bounds == (I(energy), I(0))
    assert report.support_work_bounds == I(0)
    assert report.joint_storage_margin == 0
    assert report.internal_form_gap_upper_bound == I(Q(1, 2))
    assert not report.source_joint_identity_admitted
    assert not report.all_time_joint_identity_retention_certified
    assert report.status == "unavailable"


def test_strictly_inside_full_joined_barrier_certifies_off_family_state():
    first = (Q(1, 4), 0, Q(7, 32), Q(7, 32), Q(11, 64))
    report = _contact(first + (first[4],) * 5)
    assert report.full_storage_bounds == I(Q(35, 256))
    assert report.joint_storage_margin == Q(105, 256)
    assert report.internal_form_gap_upper_bound == I(Q(1, 4))
    assert report.internal_phase_cosine_lower_bound == Q(221, 256)
    assert report.all_time_joint_identity_retention_certified
    assert report.left.internal_form != (0,) * 4


def test_individual_certificates_do_not_compose_by_conjunction_even_at_zero_work():
    source = _state((Q(1, 3),) * 4 + (0,))
    report = assess_sine_weighted_star_contact(source, source)
    assert report.component_storage_bounds == (I(Q(4, 9)),) * 2
    assert report.source_individual_retention_certified == (True, True)
    assert report.support_work_bounds == I(0)
    assert report.full_storage_bounds == I(Q(8, 9))
    assert report.source_joint_identity_admitted
    assert not report.all_time_joint_identity_retention_certified
    assert report.status == "unavailable"
    assert report.unavailable_reasons == ("strict_joint_storage_margin_not_certified",)
    # Identical coherent copies retain zero internal coordinates on the exact
    # invariant family. This failed sufficient bound does not establish escape.
    for offset in (0, 5):
        assert len(set(report.phase_rate_numerators[offset : offset + 4])) == 1
        assert report.form_rate_bounds[offset : offset + 4] == (I(0),) * 4
    assert report.phase_rate_numerators[4] == report.phase_rate_numerators[9]


def test_component_swap_and_individual_full_turn_lift_covariance(generic_contact):
    report = generic_contact
    swapped = assess_sine_weighted_star_contact(report.right, report.left)
    assert swapped.full_storage_bounds == report.full_storage_bounds
    assert swapped.support_work_bounds == report.support_work_bounds
    assert (
        swapped.form_rate_bounds
        == report.form_rate_bounds[5:] + report.form_rate_bounds[:5]
    )
    assert (
        swapped.component_storage_rate_bounds
        == report.component_storage_rate_bounds[::-1]
    )
    assert swapped.bridge_current_bounds == -report.bridge_current_bounds
    turns = TURNS[:4] + (TURNS[4] + 1,) + TURNS[5:]
    lifted = _contact(FORMS, turns)
    for field in (
        "full_storage_bounds",
        "support_work_bounds",
        "pressure_bounds",
        "form_rate_bounds",
        "phase_rate_bounds",
        "component_storage_rate_bounds",
    ):
        assert getattr(lifted, field) == getattr(report, field)
    assert lifted.weighted_phase_mean_turns - report.weighted_phase_mean_turns == Q(
        5, 74
    )
    assert lifted.bridge_phase_gap_turns == report.bridge_phase_gap_turns - 1


def test_phase_identity_uses_principal_gaps_and_rejects_the_strict_boundary():
    phases = (Q(3, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4), Q(-1, 4))
    source = _state((0,) * 5, phases)
    report = assess_sine_weighted_star_contact(source, source)
    assert report.full_storage_bounds == I(0)
    assert report.all_time_joint_identity_retention_certified
    boundary = assess_sine_weighted_star_contact(
        _state((0,) * 5, (0, Q(1, 4), 0, 0, 0)), _state((0,) * 5)
    )
    assert boundary.full_storage_bounds == I(2)
    assert not boundary.source_joint_identity_admitted
    assert boundary.status == "unavailable"


def test_open_twenty_coordinate_budget_and_off_family_exchange_control():
    epsilon = Q(1, 256)
    assert 2 * sum(WEIGHTS) == 70
    assert 2 * len(EDGES) == 30
    perturbation_squared = (70 + 30) * epsilon**2
    assert perturbation_squared == Q(5, 128) ** 2
    # H*=33/32-cos(pi/8)<1/8 follows from the exact half-angle
    # identity and sqrt(2)>7/5. No rounded source energy is a premise.
    assert Q(7, 5) ** 2 < 2
    assert (2 + Q(7, 5)) / 4 > Q(29, 32) ** 2
    assert Q(1, 8) < Q(3, 8) ** 2
    ceiling = (Q(3, 8) + Q(5, 128)) ** 2
    assert ceiling == Q(2809, 16384) < Q(35, 64)
    # Each node has an independent coordinate budget. The bridge angle
    # exceeds 47/128 radians (pi>3), and its sine exceeds 1/3.
    lower_angle = Q(3, 8) - 2 * epsilon
    assert lower_angle == Q(47, 128)
    assert lower_angle - lower_angle**3 / 6 > Q(1, 3)
    lower_form_gap = Q(1, 16) - 2 * epsilon
    current_error = bridge_form_error = 2 * epsilon
    supply_bracket_upper = (
        -8 * lower_form_gap / 3 + 4 * current_error * bridge_form_error
    )
    assert supply_bracket_upper == Q(-7, 48) + Q(1, 4096) < Q(-1, 8)
    # A full rational off-family point with all twenty coordinates perturbed
    # checks the API on this open domain; it does not replace the box proof.
    form_reference = ((Q(1, 16),) * 4 + (Q(0),)) * 2
    turn_reference = (Q(0),) * 5 + (Q(1, 16),) * 5
    form_errors = tuple(
        Q(value, 4096) for value in (1, -2, 3, -4, 5, -6, 7, -8, 9, -10)
    )
    turn_errors = tuple(
        Q(value, 32768) for value in (10, -9, 8, -7, 6, -5, 4, -3, 2, -1)
    )
    assert max(abs(value) for value in form_errors) < epsilon
    assert max(abs(value) for value in turn_errors) < Q(1, 2048)
    assert 2 * 4 * max(abs(value) for value in turn_errors) < epsilon  # pi<4
    forms = tuple(a + b for a, b in zip(form_reference, form_errors))
    turns = tuple(a + b for a, b in zip(turn_reference, turn_errors))
    report = _contact(forms, turns)
    assert report.left.internal_form != report.right.internal_form != (0,) * 4
    assert report.left.internal_phase_turns != (0,) * 4
    assert report.right.internal_phase_turns != (0,) * 4
    assert report.full_storage_bounds.hi < ceiling
    assert report.all_time_joint_identity_retention_certified
    # These stronger rational inequalities imply the proved bounds
    # H_left'<-1/(5pi), H_right'>1/(5pi), using pi>3.
    assert report.component_storage_rate_bounds[0].hi < Q(-1, 15)
    assert report.component_storage_rate_bounds[1].lo > Q(1, 15)


@pytest.mark.parametrize(
    "field,value",
    (
        ("form", (True,) * 5),
        ("form", (float("inf"),) * 5),
        ("form", (float("nan"),) * 5),
        ("form", (complex(1, 0),) * 5),
        ("form", (0,) * 4),
        ("phase_turns", (False,) * 5),
        ("phase_turns", (0.0,) * 5),
        ("phase_turns", (float("nan"),) * 5),
        ("phase_turns", (Q(1, 4),) * 6),
        ("capacity_scale", True),
        ("capacity_scale", 0),
        ("capacity_scale", Q(23, 25)),
        ("capacity_scale", float("inf")),
    ),
)
def test_consumed_primitive_state_is_readmitted_even_with_favorable_caches(
    exchanging_contact,
    field,
    value,
):
    source = replace(
        exchanging_contact.left,
        **{
            field: value,
            "full_storage_bounds": I(0),
            "exact_row_reconstruction_certified": True,
        },
    )
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_contact(source, exchanging_contact.right)
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_contact(exchanging_contact.right, source)


def test_state_type_and_report_type_are_not_duck_dispatched(exchanging_contact):
    class Fake:
        def to_dict(self):
            raise AssertionError("untrusted object dispatch")

    with pytest.raises(TypeError):
        assess_sine_weighted_star_contact(Fake(), exchanging_contact.right)
    with pytest.raises(TypeError):
        type(exchanging_contact).to_dict(Fake())


def test_fresh_primitive_reconstruction_and_sdk_atomic_export(
    generic_contact, tmp_path
):
    report = generic_contact
    poisoned_source = replace(
        report.left,
        capacity=(0,) * 5,
        full_storage_bounds=I(0),
        internal_form=(0,) * 4,
        form_rate_numerator_expressions=((),) * 5,
    )
    readmitted = assess_sine_weighted_star_contact(poisoned_source, report.right)
    assert readmitted == report
    forged = replace(
        report,
        left=poisoned_source,
        full_storage_bounds=I(0),
        phase_rate_numerators=(0,) * 10,
        source_individual_retention_certified=(True, True),
        all_time_joint_identity_retention_certified=True,
        joint_storage_margin=Q(999),
        status="certified_joint_identity_retention",
        bare_degrees=(4,) * 10,
    )
    generic = relational_report_to_dict(forged)
    assert generic["report"] == report.to_dict()["report"]
    assert generic["report"]["all_time_joint_identity_retention_certified"] is False
    assert generic["report"]["joint_identity_storage_boundary"] == {
        "numerator": 35,
        "denominator": 64,
    }
    assert generic["report"]["bare_degrees"] == list(DEGREES)
    destination = tmp_path / "weighted-star-contact.json"
    export_to_json(generic, destination)
    assert json_loads(destination.read_text(encoding="utf-8")) == generic
    prior = destination.read_bytes()
    invalid = replace(forged, right=replace(report.right, capacity_scale=False))
    with pytest.raises((TypeError, ValueError)):
        export_to_json(relational_report_to_dict(invalid), destination)
    assert destination.read_bytes() == prior
