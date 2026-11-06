"""Independent full-node controls for the global unordered sine-pair state."""

import json
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.physics.relational_sine_scale import (
    derive_sine_global_pair_state,
    evaluate_sine_global_pair_state,
)
from tnfr.sdk import export_to_json, relational_report_to_dict

PAIRS = tuple((2 * i, 2 * i + 1) for i in range(5))
BASE = tuple((i, (i + 1) % 5) for i in range(5))
EDGES = tuple((i, j) for a, b in BASE for i in PAIRS[a] for j in PAIRS[b])
ZERO = (Q(0), Q(0))
PRIMITIVES = (
    "form_means",
    "resultants",
    "phase_products",
    "internal_form_squared",
    "form_phase_moments",
)
RATE_FIELDS = (
    "form_mean_rate_pi_numerators",
    "resultant_rate_pi_numerators",
    "phase_product_rate_pi_numerators",
    "internal_form_squared_rate_pi_numerators",
    "form_phase_moment_rate_pi_numerators",
)


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _scale(a, coefficient):
    return a[0] * coefficient, a[1] * coefficient


def _mul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _conj(a):
    return a[0], -a[1]


def _mean(a, b):
    return _scale(_add(a, b), Q(1, 2))


def _state():
    forms = tuple(Q(value, 16) for value in (3, -1, 5, 1, -2, -2, 4, 4, 0, 0))
    phasors = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, 1),
        (Q(3, 5), Q(4, 5)),
        (Q(4, 5), Q(-3, 5)),
        (1, 0),
        (1, 0),
        (Q(3, 5), Q(4, 5)),
        (Q(-3, 5), Q(-4, 5)),
    )
    return forms, tuple(tuple(Q(v) for v in z) for z in phasors)


def _fine_rows(forms, phasors):
    """All twenty original rows multiplied by pi, from actual fine edges."""
    neighbors = [[] for _ in forms]
    for i, j in EDGES:
        neighbors[i].append(j)
        neighbors[j].append(i)
    dx = tuple(
        sum((_mul(_conj(phasors[i]), phasors[j])[1] for j in row), Q(0)) / 4
        for i, row in enumerate(neighbors)
    )
    dtheta = tuple(
        sum((forms[i] - forms[j] for j in row), Q(0)) / 4
        for i, row in enumerate(neighbors)
    )
    dz = tuple(_mul((Q(0), rate), z) for rate, z in zip(dtheta, phasors))
    return dx, dtheta, dz


def _pushforward(forms, phasors):
    dx, _, dz = _fine_rows(forms, phasors)
    coordinates, rates = [[] for _ in range(5)], [[] for _ in range(5)]
    for i, j in PAIRS:
        u, du = (forms[i] - forms[j]) / 2, (dx[i] - dx[j]) / 2
        difference = _scale(_add(phasors[i], _scale(phasors[j], -1)), Q(1, 2))
        d_difference = _scale(_add(dz[i], _scale(dz[j], -1)), Q(1, 2))
        values = (
            (forms[i] + forms[j]) / 2,
            _mean(phasors[i], phasors[j]),
            _mul(phasors[i], phasors[j]),
            u * u,
            _scale(difference, u),
        )
        derivatives = (
            (dx[i] + dx[j]) / 2,
            _mean(dz[i], dz[j]),
            _add(_mul(dz[i], phasors[j]), _mul(phasors[i], dz[j])),
            2 * u * du,
            _add(_scale(difference, du), _scale(d_difference, u)),
        )
        for target, value in zip(coordinates, values):
            target.append(value)
        for target, value in zip(rates, derivatives):
            target.append(value)
    return tuple(map(tuple, coordinates)), tuple(map(tuple, rates))


def _inputs(report):
    return {name: getattr(report, name) for name in PRIMITIVES}


def test_all_global_coordinates_and_rates_are_full_fine_pushforwards():
    forms, phasors = _state()
    report = derive_sine_global_pair_state(forms, phasors)
    coordinates, rates = _pushforward(forms, phasors)
    for field, expected in zip(PRIMITIVES, coordinates):
        assert getattr(report, field) == expected
    for field, expected in zip(RATE_FIELDS, rates):
        assert getattr(report, field) == expected
    assert report.pairs == PAIRS
    assert report.base_edges == BASE
    assert report.clock == "structural_t"
    assert report.pair_strata == (
        "antipodal_phase",
        "coincident_phase",
        "split_phase",
        "coincident_phase",
        "antipodal_phase",
    )
    assert evaluate_sine_global_pair_state(**_inputs(report)) == report


def test_block_current_and_its_motion_retain_actual_fine_boundary_normalization():
    forms, phasors = _state()
    report = derive_sine_global_pair_state(forms, phasors)
    _, phase_rates, _ = _fine_rows(forms, phasors)
    for (a, b), current, rate in zip(
        report.directed_block_edges,
        report.block_current_pi_numerators,
        report.block_current_rate_pi_squared_numerators,
    ):
        expected = (
            sum(
                (
                    _mul(_conj(phasors[i]), phasors[j])[1]
                    for i in PAIRS[a]
                    for j in PAIRS[b]
                ),
                Q(0),
            )
            / 8
        )
        derivative = (
            sum(
                (
                    _mul(_conj(phasors[i]), phasors[j])[0]
                    * (phase_rates[j] - phase_rates[i])
                    for i in PAIRS[a]
                    for j in PAIRS[b]
                ),
                Q(0),
            )
            / 8
        )
        assert current == expected
        assert rate == derivative
    assert sum(report.form_mean_rate_pi_numerators) == 0


def test_storage_and_work_match_all_twenty_edges_without_discarded_terms():
    forms, phasors = _state()
    report = derive_sine_global_pair_state(forms, phasors)
    dx, rates, _ = _fine_rows(forms, phasors)
    form_storage = sum(((forms[i] - forms[j]) ** 2 / 2 for i, j in EDGES), Q(0))
    phase_storage = sum(
        (1 - _mul(_conj(phasors[i]), phasors[j])[0] for i, j in EDGES), Q(0)
    )
    form_work = sum(((forms[i] - forms[j]) * (dx[i] - dx[j]) for i, j in EDGES), Q(0))
    phase_work = sum(
        (
            _mul(_conj(phasors[i]), phasors[j])[1] * (rates[j] - rates[i])
            for i, j in EDGES
        ),
        Q(0),
    )
    assert report.form_storage == form_storage
    assert report.phase_storage == phase_storage
    assert report.storage == form_storage + phase_storage
    assert form_work != 0
    assert report.form_storage_rate_pi_numerator == form_work
    assert report.phase_storage_rate_pi_numerator == phase_work
    assert report.full_storage_rate_pi_numerator == form_work + phase_work == 0


@pytest.mark.parametrize("mask", (1, 2, 4, 8, 16, 31))
def test_independent_whole_member_swaps_preserve_the_complete_quotient(mask):
    forms, phasors = _state()
    expected = derive_sine_global_pair_state(forms, phasors)
    forms, phasors = list(forms), list(phasors)
    for k, (i, j) in enumerate(PAIRS):
        if mask & (1 << k):
            forms[i], forms[j] = forms[j], forms[i]
            phasors[i], phasors[j] = phasors[j], phasors[i]
    assert derive_sine_global_pair_state(forms, phasors) == expected


def test_common_form_origin_and_common_circular_rotation_are_covariant():
    forms, phasors = _state()
    baseline = derive_sine_global_pair_state(forms, phasors)
    shift, rotation = Q(3, 7), (Q(3, 5), Q(4, 5))
    moved = derive_sine_global_pair_state(
        tuple(x + shift for x in forms), tuple(_mul(rotation, z) for z in phasors)
    )
    assert moved.form_means == tuple(x + shift for x in baseline.form_means)
    for name in ("resultants", "form_phase_moments", RATE_FIELDS[1], RATE_FIELDS[4]):
        assert getattr(moved, name) == tuple(
            _mul(rotation, z) for z in getattr(baseline, name)
        )
    squared_rotation = _mul(rotation, rotation)
    for name in ("phase_products", RATE_FIELDS[2]):
        assert getattr(moved, name) == tuple(
            _mul(squared_rotation, z) for z in getattr(baseline, name)
        )
    for name in (RATE_FIELDS[0], RATE_FIELDS[3], "internal_form_squared", "storage"):
        assert getattr(moved, name) == getattr(baseline, name)


def test_zero_resultant_and_zero_variance_still_require_phase_product():
    forms = (Q(0),) * 10
    real = ((Q(1), Q(0)), (Q(-1), Q(0))) + ((Q(1), Q(0)),) * 8
    imaginary = ((Q(0), Q(1)), (Q(0), Q(-1))) + real[2:]
    left = derive_sine_global_pair_state(forms, real)
    right = derive_sine_global_pair_state(forms, imaginary)
    for field in (
        "form_means",
        "resultants",
        "internal_form_squared",
        "form_phase_moments",
    ):
        assert getattr(left, field) == getattr(right, field)
    assert left.phase_products[0] != right.phase_products[0]
    assert left.form_phase_moment_rate_pi_numerators[0] == ZERO
    assert right.form_phase_moment_rate_pi_numerators[0] == (0, -1)
    assert left.form_phase_moment_rate_pi_numerators == _pushforward(forms, real)[1][4]
    assert (
        right.form_phase_moment_rate_pi_numerators
        == _pushforward(forms, imaginary)[1][4]
    )


def test_coincident_phase_can_hide_form_variance_until_the_second_phase_response():
    phase = ((Q(1), Q(0)),) * 10
    forms = (Q(1, 7), Q(-1, 7)) + (Q(0),) * 8
    moving = derive_sine_global_pair_state(forms, phase)
    stationary = derive_sine_global_pair_state((Q(0),) * 10, phase)
    for name in ("form_means", "resultants", "phase_products", "form_phase_moments"):
        assert getattr(moving, name) == getattr(stationary, name)
    assert (
        moving.resultant_rate_pi_numerators == stationary.resultant_rate_pi_numerators
    )
    assert moving.form_phase_moment_rate_pi_numerators[0] == (0, Q(1, 49))
    # At this all-aligned state x'=0, hence theta''=0. Compute Z'' from
    # individual phasors instead of differentiating the proposed quotient.
    _, theta_rate, _ = _fine_rows(forms, phase)
    second = -(theta_rate[0] ** 2 + theta_rate[1] ** 2) / 2
    assert second == -moving.internal_form_squared[0] == Q(-1, 49)


def test_rational_realizable_invariants_need_not_have_rational_fine_lifts():
    report = evaluate_sine_global_pair_state(
        form_means=(Q(0),) * 5,
        resultants=((Q(1, 2), Q(0)),) * 5,
        phase_products=((Q(1), Q(0)),) * 5,
        internal_form_squared=(Q(1, 3),) * 5,
        form_phase_moments=((Q(0), Q(1, 2)),) * 5,
    )
    with mp.workdps(80):
        u, delta = 1 / mp.sqrt(3), mp.pi / 3
        forms = tuple(v for _ in PAIRS for v in (u, -u))
        phasors = tuple(
            z
            for _ in PAIRS
            for z in ((mp.cos(delta), mp.sin(delta)), (mp.cos(delta), -mp.sin(delta)))
        )
        _, derivatives = _pushforward(forms, phasors)
        for field, expected in zip(RATE_FIELDS, derivatives):
            for actual, exact in zip(getattr(report, field), expected):
                aa = actual if isinstance(actual, tuple) else (actual,)
                ee = exact if isinstance(exact, tuple) else (exact,)
                for a, e in zip(aa, ee):
                    rational = mp.mpf(a.numerator) / a.denominator
                    assert abs(rational - e) < mp.mpf("1e-75")


def test_tiny_exact_form_and_paired_phase_information_are_not_materialized_away():
    tiny = Q(1, 2**1100)
    forms = (tiny, -tiny) + (Q(0),) * 8
    phasors = ((Q(1), Q(0)), (Q(-1), Q(0))) + ((Q(1), Q(0)),) * 8
    report = derive_sine_global_pair_state(forms, phasors)
    assert report.form_phase_moments[0] == (tiny, 0)
    assert report.resultant_rate_pi_numerators[0] == (0, tiny)
    assert report.internal_form_squared[0] == tiny**2 > 0


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("resultants", ((Q(2), Q(0)),) * 5),
        ("resultants", ((Q(0), Q(1, 2)),) * 5),
        ("phase_products", ((Q(1, 2), Q(0)),) * 5),
        ("internal_form_squared", (Q(-1),) * 5),
        ("form_phase_moments", ((Q(1), Q(0)),) * 5),
    ),
)
def test_algebraic_candidates_must_be_realizable_not_merely_complex_coordinates(
    field, replacement
):
    inputs = dict(
        form_means=(Q(0),) * 5,
        resultants=((Q(0), Q(0)),) * 5,
        phase_products=((Q(1), Q(0)),) * 5,
        internal_form_squared=(Q(1),) * 5,
        form_phase_moments=((Q(0), Q(1)),) * 5,
    )
    inputs[field] = replacement
    with pytest.raises((TypeError, ValueError)):
        evaluate_sine_global_pair_state(**inputs)


def test_zero_variance_candidate_with_resultant_outside_unit_disk_is_rejected():
    # All polynomial equalities can hold at U=W=0, even for |Z|>1.
    with pytest.raises((TypeError, ValueError)):
        evaluate_sine_global_pair_state(
            form_means=(0,) * 5,
            resultants=((2, 0),) * 5,
            phase_products=((1, 0),) * 5,
            internal_form_squared=(0,) * 5,
            form_phase_moments=((0, 0),) * 5,
        )


@pytest.mark.parametrize("bad", (True, float("nan"), float("inf")))
def test_nonphysical_scalar_admission_in_both_paths(bad):
    forms, phasors = _state()
    with pytest.raises((TypeError, ValueError)):
        derive_sine_global_pair_state((bad,) + forms[1:], phasors)
    with pytest.raises((TypeError, ValueError)):
        derive_sine_global_pair_state(forms, ((bad, 0),) + phasors[1:])
    report = derive_sine_global_pair_state(forms, phasors)
    inputs = _inputs(report)
    inputs["form_means"] = (bad,) + report.form_means[1:]
    with pytest.raises((TypeError, ValueError)):
        evaluate_sine_global_pair_state(**inputs)


def test_shape_order_and_exact_unit_circle_admission_are_not_silent_coercions():
    forms, phasors = _state()
    for x, z in ((forms[:-1], phasors), (forms, phasors[:-1]), (set(forms), phasors)):
        with pytest.raises((TypeError, ValueError)):
            derive_sine_global_pair_state(x, z)
    with pytest.raises((TypeError, ValueError)):
        derive_sine_global_pair_state(forms, ((0.6, 0.8),) + phasors[1:])
    report = derive_sine_global_pair_state(forms, phasors)
    for field in PRIMITIVES:
        inputs = _inputs(report)
        inputs[field] = inputs[field][:-1]
        with pytest.raises((TypeError, ValueError)):
            evaluate_sine_global_pair_state(**inputs)
    assert derive_sine_global_pair_state(iter(forms), iter(phasors)) == report


def test_primitive_reevaluation_never_uses_edited_derived_fields_and_sdk_is_exact(
    tmp_path,
):
    report = derive_sine_global_pair_state(*_state())
    poisoned = replace(
        report,
        storage=Q(-100),
        form_mean_rate_pi_numerators=(Q(987),) * 5,
        resultant_rate_pi_numerators=((Q(123), Q(456)),) * 5,
    )
    # There is no report-consuming solver: explicitly submit only declared
    # state coordinates. Export alone is not proof authentication.
    rebuilt = evaluate_sine_global_pair_state(**_inputs(poisoned))
    assert rebuilt == report
    direct = report.to_dict()
    assert relational_report_to_dict(report)["report"] == direct["report"]
    target = tmp_path / "global-pair-state.json"
    export_to_json(direct, target)
    assert json.loads(target.read_text()) == direct
    assert direct["report"]["form_means"][0] == {"numerator": 1, "denominator": 16}
