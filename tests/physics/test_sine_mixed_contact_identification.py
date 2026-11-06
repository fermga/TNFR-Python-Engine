"""Independent nuisance-contact sets for one absolute receiver observation.

The order-12 series is an analytical constant-force proxy, with its factorial
tail and live-phase allowance kept separate. No source, nonlinear trajectory,
calibration, measured record or classifier is produced by these controls.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_mixed_contact_identification as owner

WIDTH = Q(1, 1024)
BOX = ((1 - WIDTH, 1 + WIDTH),) + ((Q(5, 4) - WIDTH, Q(5, 4)),) * 8
ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
REGISTRATION_NOISE = Q(1, 10**6)
FORM_NOISE = Q(1, 10**13)
TIME = Q(1, 32)
TARGET = Q(5, 10**13)
FEEDBACK = Q(13, 9 * 3000**2)


def _assess(**changes):
    values = dict(
        contact_conductance_bounds=BOX,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        registration_error_bound=REGISTRATION_NOISE,
        form_readout_error_bound=FORM_NOISE,
    )
    values.update(changes)
    return owner._assess_sine_mixed_contact_identification(**values)


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, lower, upper=None):
    upper = lower if upper is None else upper
    assert _mp(context, interval[0]) <= lower <= upper <= _mp(context, interval[1])


def _point(weights):
    return tuple((w, w) for w in weights)


def _coefficient(weights, law):
    if law == "degree":
        return sum(weights) / 27
    return sum(w / (2 + w) for w in weights) / 9


def _transport(weights, law):
    # Solve the phase-registration congruence independently of production p(j).
    matching = tuple(
        next(k for k in range(9) if (2 * (k - 4) - (j - 4)) % 9 == 0) for j in range(9)
    )
    edges = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in (0, 1) for j in range(9)
    )
    edges += tuple((j, 9 + matching[j], w) for j, w in enumerate(weights))
    matrix = [[Q(0) for _ in range(18)] for _ in range(18)]
    for a, b, w in edges:
        matrix[a][a] += w
        matrix[b][b] += w
        matrix[a][b] -= w
        matrix[b][a] -= w
    denominators = (
        (Q(3),) * 18 if law == "degree" else tuple(matrix[i][i] for i in range(18))
    )
    return tuple(
        tuple(value / denominator for value in row)
        for row, denominator in zip(matrix, denominators)
    )


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _proxy(matrix, norm):
    z = (Q(1),) * 9 + (Q(-1),) * 9
    power, total = z, [Q(0)] * 18
    for n in range(13):
        factor = (-1) ** n * TIME ** (n + 1) / factorial(n + 1)
        total = [old + factor * new for old, new in zip(total, power)]
        power = _apply(matrix, power)
    tail = norm**13 * TIME**14 / factorial(14) / (1 - norm * TIME / 15)
    return sum(total[:9]) / 9, tail


@pytest.fixture(scope="module", autouse=True)
def no_evolution_events_or_prior_response_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("contact-set controls cannot evolve, attach or replay a source")

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
        ):
            patch.setattr(object_, name, forbidden)
        # This reader needs no field factory, so no static exception is granted.
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_raw_marked_box_and_fresh_reference_retain_the_universal_identity(report):
    assert report.contact_conductance_bounds == BOX
    reference = report.lower_endpoint_reference
    assert reference.contact_conductances == tuple(pair[0] for pair in BOX)
    assert not reference.early_profile_supported and not reference.certified
    assert report.contact_box_within_protocol
    assert report.source_certified and report.isolated_waiting_certified
    assert report.geometry_certified and report.both_joint_identities_certified
    assert report.registration_response_certified
    assert reference.strength_common_form_limit_bounds == (-100 * ETA, 100 * ETA)
    assert reference.recovery_duration == 2 * 10**10
    assert (
        reference.bare_degree_reference.precontact_reference.geometry.class_order
        == (1, 2)
    )


def test_endpoint_reduction_has_exact_monotone_factorization(report):
    low, high = tuple(a for a, _ in BOX), tuple(b for _, b in BOX)
    for a, b in BOX:
        # This identity supplies the nonnegative factor for every u<=v in the
        # whole admitted box; the theorem does not interpolate sampled corners.
        assert b / (2 + b) - a / (2 + a) == 2 * (b - a) / ((2 + a) * (2 + b)) > 0
    assert report.degree_coefficient_bounds == (
        _coefficient(low, "degree"),
        _coefficient(high, "degree"),
    )
    assert report.strength_coefficient_bounds == (
        _coefficient(low, "strength"),
        _coefficient(high, "strength"),
    )
    assert report.degree_coefficient_bounds[0] == Q(11, 27) - WIDTH / 3
    assert report.strength_coefficient_bounds[1] == Q(133, 351) + 2 * WIDTH / (
        27 * (3 + WIDTH)
    )
    effective = (
        report.degree_coefficient_bounds[0] - report.strength_coefficient_bounds[1]
    )
    assert effective > Q(10, 351) - 29 * WIDTH / 81 > Q(1, 36)
    assert 29 * 1404 < 81 * 1024


def test_independent_order12_full_node_proxies_fit_each_absolute_envelope(
    mp, report, record_property
):
    low, high = tuple(a for a, _ in BOX), tuple(b for _, b in BOX)
    alternating = tuple(pair[i % 2] for i, pair in enumerate(BOX))
    forcing = mp.sin(_mp(mp, Q(1, 1024))) / (3069 * mp.pi)
    cases = 0
    for weights in (low, high, alternating):
        for law, norm, az, a2z, cubic in (
            ("degree", Q(13, 6), Q(5, 6), Q(65, 36), Q(65, 216)),
            ("strength", Q(2), Q(10, 13), Q(20, 13), Q(10, 39)),
        ):
            matrix = _transport(weights, law)
            z = (Q(1),) * 9 + (Q(-1),) * 9
            assert all(sum(row) == 0 for row in matrix)
            assert all(
                matrix[i][j] <= 0 for i in range(18) for j in range(18) if i != j
            )
            assert max(sum(abs(v) for v in row) for row in matrix) <= norm
            assert max(map(abs, _apply(matrix, z))) <= az
            assert max(map(abs, _apply(matrix, _apply(matrix, z)))) <= a2z
            coefficient = sum(_apply(matrix, z)[:9]) / 18
            assert coefficient == _coefficient(weights, law)
            center, tail = _proxy(matrix, norm)
            leading = TIME - coefficient * TIME**2
            assert leading - cubic * TIME**3 <= center - tail
            assert center + tail <= leading + cubic * TIME**3
            full_allowance = tail + FEEDBACK * TIME**3
            _contains(
                mp,
                getattr(report, law + "_nominal_response_bounds"),
                forcing * _mp(mp, center - full_allowance),
                forcing * _mp(mp, center + full_allowance),
            )
            cases += 1
    record_property("fixed_proxy_profiles", "lower,upper,alternating")
    record_property("constant_force_series_order", "12")
    record_property("individual_law_proxy_checks", str(cases))


def test_separate_live_phase_source_and_reading_envelopes_preserve_the_gap(
    mp, report, record_property
):
    forcing = mp.sin(_mp(mp, Q(1, 1024))) / (3069 * mp.pi)
    c = report.forcing_bounds
    _contains(mp, (c.lo, c.hi), forcing)
    assert c.lo >= Q(1, 10240000)
    assert report.forcing_width == c.hi - c.lo <= Q(1, 10**12)
    assert report.forcing_width_margin >= 0 and report.forcing_lower_margin >= 0
    assert report.numerical_enclosure_certified
    assert report.degree_remainder_coefficient == Q(65, 216) + FEEDBACK
    assert report.strength_remainder_coefficient == Q(10, 39) + FEEDBACK
    assert (
        report.degree_remainder_coefficient + report.strength_remainder_coefficient
        < Q(3, 5)
    )
    assert report.per_hypothesis_source_error_upper_bound == 2 * ETA
    for law in ("degree", "strength"):
        rlo, rhi = getattr(report, law + "_coefficient_bounds")
        beta = getattr(report, law + "_remainder_coefficient")
        a, b = (
            TIME - TIME**2 * rhi - TIME**3 * beta,
            TIME - TIME**2 * rlo + TIME**3 * beta,
        )
        assert 0 < a <= b <= TIME
        nominal = getattr(report, law + "_nominal_response_bounds")
        assert nominal == (c.lo * a, c.hi * b)
        actual = (nominal[0] - 2 * ETA, nominal[1] + 2 * ETA)
        assert getattr(report, law + "_actual_response_bounds") == actual
        assert getattr(report, law + "_recorded_response_bounds") == (
            actual[0] - FORM_NOISE,
            actual[1] + FORM_NOISE,
        )
    gap = (
        report.strength_recorded_response_bounds[0]
        - report.degree_recorded_response_bounds[1]
    )
    arithmetic_floor = Q(8, 10**13) - TIME * Q(1, 10**12) - 4 * ETA - 2 * FORM_NOISE
    assert gap == report.recorded_gap > arithmetic_floor > TARGET
    assert report.separation_margin == gap - TARGET > 0
    assert (
        report.disjoint_response_sets_certified and report.target_separation_certified
    )
    assert report.status == "certified_ordered_target_separation"
    assert (
        report.reverse_recorded_gap
        == report.degree_recorded_response_bounds[0]
        - report.strength_recorded_response_bounds[1]
        < 0
    )
    assert (
        not report.contains_unit_profile
        and not report.unit_profile_obstruction_certified
    )
    record_property(
        "degree_recorded_absolute_bounds", str(report.degree_recorded_response_bounds)
    )
    record_property(
        "strength_recorded_absolute_bounds",
        str(report.strength_recorded_response_bounds),
    )
    record_property("independent_fit_recorded_gap", str(gap))
    record_property("independent_fit_recorded_gap_display", mp.nstr(_mp(mp, gap), 24))
    record_property("gap_target", str(TARGET))


def test_cross_fit_gap_uses_distinct_nuisance_weights_not_a_paired_contrast(report):
    low = _assess(contact_conductance_bounds=_point(tuple(a for a, _ in BOX)))
    high = _assess(contact_conductance_bounds=_point(tuple(b for _, b in BOX)))
    assert (
        report.degree_recorded_response_bounds[1]
        == low.degree_recorded_response_bounds[1]
    )
    assert (
        report.strength_recorded_response_bounds[0]
        == high.strength_recorded_response_bounds[0]
    )
    assert (
        report.recorded_gap
        == high.strength_recorded_response_bounds[0]
        - low.degree_recorded_response_bounds[1]
    )
    assert (
        report.recorded_gap < low.recorded_gap
        and report.recorded_gap < high.recorded_gap
    )
    assert (
        low.lower_endpoint_reference.contact_conductances
        != high.lower_endpoint_reference.contact_conductances
    )
    # Their residual sets are also independently enlarged by 2*eta per law;
    # no same-state or same-work cancellation is used in this gap.
    nominal_gap = (
        report.strength_nominal_response_bounds[0]
        - report.degree_nominal_response_bounds[1]
    )
    assert report.recorded_gap == nominal_gap - 4 * ETA - 2 * FORM_NOISE


def test_nested_and_singleton_domains_shrink_both_necessary_sets(report):
    nested_box = tuple(((3 * a + b) / 4, (a + 3 * b) / 4) for a, b in BOX)
    singleton_box = _point(tuple((a + b) / 2 for a, b in BOX))
    nested = _assess(contact_conductance_bounds=nested_box)
    singleton = _assess(contact_conductance_bounds=singleton_box)
    for law in ("degree", "strength"):
        broad = getattr(report, law + "_recorded_response_bounds")
        middle = getattr(nested, law + "_recorded_response_bounds")
        point = getattr(singleton, law + "_recorded_response_bounds")
        assert broad[0] < middle[0] < point[0] < point[1] < middle[1] < broad[1]
    assert report.recorded_gap < nested.recorded_gap < singleton.recorded_gap
    assert nested.target_separation_certified and singleton.target_separation_certified


def test_identity_and_registration_noise_do_not_replace_form_identification(report):
    zero = _assess(registration_error_bound=0, form_readout_error_bound=0)
    assert zero.degree_recorded_response_bounds == report.degree_actual_response_bounds
    assert (
        zero.strength_recorded_response_bounds == report.strength_actual_response_bounds
    )
    assert zero.recorded_gap == report.recorded_gap + 2 * FORM_NOISE
    bad_registration = _assess(
        registration_error_bound=REGISTRATION_NOISE + Q(1, 10**100)
    )
    assert bad_registration.both_joint_identities_certified
    assert not bad_registration.registration_response_certified
    assert bad_registration.target_separation_certified
    assert bad_registration.recorded_gap == report.recorded_gap
    bad_form = _assess(form_readout_error_bound=FORM_NOISE + Q(1, 10**100))
    assert (
        bad_form.both_joint_identities_certified
        and bad_form.registration_response_certified
    )
    assert not bad_form.target_separation_certified
    assert (
        bad_form.degree_recorded_response_bounds[1]
        > report.degree_recorded_response_bounds[1]
    )


def test_admitted_unit_member_is_an_existential_obstruction_not_all_record_equality():
    broad = _assess(contact_conductance_bounds=((Q(3, 4), Q(5, 4)),) * 9)
    unit = _assess(contact_conductance_bounds=((Q(1), Q(1)),) * 9)
    for result in (broad, unit):
        assert (
            result.contains_unit_profile and result.unit_profile_obstruction_certified
        )
        assert result.both_joint_identities_certified
        assert (
            not result.disjoint_response_sets_certified
            and not result.target_separation_certified
        )
        assert result.recorded_gap < 0
        assert result.status == "certified_unit_profile_obstruction"
    assert unit.lower_endpoint_reference.exact_law_equivalence
    assert unit.lower_endpoint_reference.early_actual_contrast_bounds == (Q(0), Q(0))
    assert not broad.lower_endpoint_reference.exact_law_equivalence
    # The same unit source/error choice is a witness in both hypothesis sets.
    # The sets from all independently chosen broad-box states need not coincide.
    assert (
        broad.degree_recorded_response_bounds != broad.strength_recorded_response_bounds
    )


def test_nonunit_outer_overlap_is_unresolved_not_a_claim_of_equivalence():
    result = _assess(contact_conductance_bounds=((Q(101, 100), Q(101, 100)),) * 9)
    assert (
        not result.contains_unit_profile
        and not result.unit_profile_obstruction_certified
    )
    assert result.both_joint_identities_certified
    assert result.recorded_gap < 0
    degree, strength = (
        result.degree_recorded_response_bounds,
        result.strength_recorded_response_bounds,
    )
    assert max(degree[0], strength[0]) < min(degree[1], strength[1])
    assert (
        not result.disjoint_response_sets_certified
        and not result.target_separation_certified
    )
    assert not result.lower_endpoint_reference.exact_law_equivalence
    assert result.status == "unresolved_response_envelopes"


def test_reverse_disjointness_does_not_claim_the_ordered_primary_target():
    result = _assess(contact_conductance_bounds=((Q(3, 4), Q(3, 4)),) * 9)
    assert result.strength_coefficient_bounds[0] - result.degree_coefficient_bounds[
        1
    ] == Q(1, 44)
    analytical_floor = Q(1, 10240000) * TIME**2 * (Q(1, 44) - TIME * Q(3, 5))
    assert analytical_floor - TIME * Q(1, 10**12) - 4 * ETA - 2 * FORM_NOISE > 0
    assert (
        result.reverse_recorded_gap
        == result.degree_recorded_response_bounds[0]
        - result.strength_recorded_response_bounds[1]
        > 0
    )
    assert result.recorded_gap < 0
    assert (
        result.disjoint_response_sets_certified
        and not result.target_separation_certified
    )
    assert (
        not result.contains_unit_profile
        and not result.unit_profile_obstruction_certified
    )
    assert result.status == "certified_disjoint_without_ordered_target"


def test_failed_source_geometry_or_numerical_budget_cannot_certify(monkeypatch):
    original = owner._assess_sine_mixed_normalization

    def corrupt_geometry(**kwargs):
        report = original(**kwargs)
        previous = report.bare_degree_reference.precontact_reference
        return replace(
            report,
            bare_degree_reference=replace(
                report.bare_degree_reference,
                precontact_reference=replace(
                    previous,
                    geometry=replace(
                        previous.geometry, matching_permutation=tuple(range(9))
                    ),
                ),
            ),
        )

    replacements = (
        lambda **kwargs: replace(original(**kwargs), source_certified=False),
        lambda **kwargs: replace(original(**kwargs), isolated_waiting_certified=False),
        lambda **kwargs: replace(
            original(**kwargs), both_joint_identities_certified=False
        ),
        lambda **kwargs: replace(original(**kwargs), contact_conductances=(Q(1),) * 9),
        corrupt_geometry,
    )
    for replacement in replacements:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_assess_sine_mixed_normalization", replacement)
            result = _assess()
            assert not result.target_separation_certified
            assert result.unresolved_conditions
    with monkeypatch.context() as patch:
        delta = Q(1, 1024)
        patch.setattr(
            owner, "sin", lambda value: I(delta - delta**3 / 6, delta + delta / 100)
        )
        result = _assess()
        assert result.forcing_lower_margin > 0 and result.forcing_width_margin < 0
        assert not result.numerical_enclosure_certified
        assert (
            not result.target_separation_certified
            and not result.disjoint_response_sets_certified
        )
        unit = _assess(contact_conductance_bounds=((Q(1), Q(1)),) * 9)
        assert not unit.numerical_enclosure_certified
        assert unit.unit_profile_obstruction_certified
    with monkeypatch.context() as patch:
        patch.setattr(owner, "sin", lambda value: I(0))
        result = _assess()
        assert result.forcing_lower_margin < 0 and result.forcing_width_margin > 0
        assert not result.numerical_enclosure_certified
    for field in ("form_error_bound", "phase_error_bound"):
        result = _assess(**{field: 2 * ERROR})
        assert not result.source_certified and not result.target_separation_certified
    unavailable_unit = _assess(
        contact_conductance_bounds=((Q(1), Q(1)),) * 9, form_error_bound=2 * ERROR
    )
    assert unavailable_unit.contains_unit_profile
    assert not unavailable_unit.unit_profile_obstruction_certified


def test_only_box_uniform_reference_obligations_are_consumed(monkeypatch, report):
    original = owner._assess_sine_mixed_normalization
    calls = []

    def unrelated_paired_response(**kwargs):
        calls.append(kwargs)
        return replace(
            original(**kwargs),
            certified=False,
            early_response_certified=False,
            early_nominal_contrast_bounds=(Q(-(10**100)), Q(10**100)),
            matched_attachment_work_certified=False,
            status="unavailable",
        )

    monkeypatch.setattr(
        owner, "_assess_sine_mixed_normalization", unrelated_paired_response
    )
    result = _assess()
    assert len(calls) == 1
    assert calls[0]["contact_conductances"] == tuple(a for a, _ in BOX)
    assert result.target_separation_certified and result.both_joint_identities_certified
    assert result.recorded_gap == report.recorded_gap
    assert not result.lower_endpoint_reference.matched_attachment_work_certified


def test_valid_nonnegative_outside_domain_stays_unavailable():
    for box in (
        ((Q(0), Q(0)),) * 9,
        ((Q(3, 4) - Q(1, 10**100), Q(1)),) + BOX[1:],
        ((Q(1), Q(5, 4) + Q(1, 10**100)),) + BOX[1:],
    ):
        result = _assess(contact_conductance_bounds=box)
        assert not result.contact_box_within_protocol
        assert (
            not result.both_joint_identities_certified
            and not result.target_separation_certified
        )
        assert result.unresolved_conditions
        assert result.status == "unavailable"
        for name in (
            "degree_nominal_response_bounds",
            "strength_nominal_response_bounds",
            "degree_actual_response_bounds",
            "strength_actual_response_bounds",
            "degree_recorded_response_bounds",
            "strength_recorded_response_bounds",
            "recorded_gap",
            "reverse_recorded_gap",
            "separation_margin",
        ):
            assert getattr(result, name) is None


def test_all_nested_raw_inputs_admit_before_any_reference_or_numerical_work(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source, geometry or numerical work")

    for name in (
        "_assess_sine_mixed_normalization",
        "_derive_mixed_constitutive_pair",
        "sin",
        "I",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    malformed = (
        (),
        BOX[:-1],
        BOX + (BOX[0],),
        repeat((Q(1), Q(1))),
        {0: BOX[0]},
        "123456789",
        ((Q(1),),) + BOX[1:],
        ((Q(1), Q(1), Q(1)),) + BOX[1:],
        (repeat(Q(1)),) + BOX[1:],
        ({1, 2},) + BOX[1:],
        ((Q(2), Q(1)),) + BOX[1:],
        ((Q(-1), Q(1)),) + BOX[1:],
    )
    for value in (True, np.bool_(False), float("nan"), float("inf"), 1j, None):
        malformed += (((value, Q(1)),) + BOX[1:], ((Q(0), value),) + BOX[1:])
    for box in malformed:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(contact_conductance_bounds=box)
    for field in (
        "form_error_bound",
        "phase_error_bound",
        "registration_error_bound",
        "form_readout_error_bound",
    ):
        for value in (
            True,
            np.bool_(False),
            Q(-1),
            float("nan"),
            float("inf"),
            1j,
            None,
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{field: value})


def test_one_shot_fixed_width_admission_and_no_fitted_response_or_cached_input(report):
    def pairs():
        for a, b in BOX:
            yield iter(
                (
                    Q(np.int64(a.numerator), np.int64(a.denominator)),
                    Q(np.int64(b.numerator), np.int64(b.denominator)),
                )
            )

    actual = _assess(
        contact_conductance_bounds=pairs(),
        form_error_bound=Q(np.int64(1), np.int64(10**10)),
        phase_error_bound=Q(np.int64(1), np.int64(10**10)),
        registration_error_bound=Q(np.int64(1), np.int64(10**6)),
        form_readout_error_bound=Q(np.int64(1), np.int64(10**13)),
    )
    assert actual == report
    with pytest.raises(FrozenInstanceError):
        report.target_separation_certified = False
    for extra in (
        dict(lower_endpoint_reference=report.lower_endpoint_reference),
        dict(recorded_form=Q(0)),
        dict(elapsed_time=TIME),
        dict(gamma=Q(1, 3000)),
        dict(calibrated=True),
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
