"""Independent origin, finite-envelope and source controls for the contact.

Static states test identities and quantifiers, not acquisition of those states.
No contacted trajectory or registration scan evaluates the frozen response.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _organized_external_registration as owner


_ERROR = Q(1, 10**10)
_ZERO = (Q(0),) * 54
_FORMS = tuple(Q((7 * i) % 19 - 9, 64) for i in range(54))
_OFFSETS = tuple(Q((11 * i) % 23 - 11, 256) for i in range(54))
_EDGES = (
    tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
    + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    + tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _observe(**changes):
    return owner._observe_organized_external_registration(
        **(
            dict(relative_phase_origin=Q(2, 5), form=_FORMS, phase_offsets=_OFFSETS)
            | changes
        )
    )


def _assess(**changes):
    return owner._assess_organized_external_registration(
        **(
            dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
            | changes
        )
    )


@pytest.fixture(scope="module")
def report():
    return _assess()


def _independent_boundary(mp, offsets):
    return (
        sum(
            (
                mp.exp(1j * _mp(mp, offsets[45 + j] - offsets[9 * c + j]))
                for c in range(5)
                for j in range(9)
            ),
            mp.mpc(0),
        )
        / 9
    )


@pytest.mark.parametrize("origin", (Q(-7, 3), Q(0), Q(2, 5), Q(4)))
def test_arbitrary_origin_complete_rows_and_boundary_quadratures(mp, origin):
    observed = _observe(relative_phase_origin=origin)
    state = observed.interaction
    forms = tuple(_mp(mp, v) for v in _FORMS)
    phases = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, v) + (_mp(mp, origin) if i >= 45 else 0)
        for i, v in enumerate(_OFFSETS)
    )
    gradients, currents = [mp.mpf(0)] * 54, [mp.mpf(0)] * 54
    storage, contact_work = mp.mpf(0), mp.mpf(0)
    for index, (left, right) in enumerate(_EDGES):
        difference = forms[left] - forms[right]
        angle = phases[right] - phases[left]
        gradients[left] += difference
        gradients[right] -= difference
        currents[left] += mp.sin(angle)
        currents[right] -= mp.sin(angle)
        edge_storage = difference**2 / 2 + 1 - mp.cos(angle)
        storage += edge_storage
        if index >= 99:
            contact_work += edge_storage
    gamma = 1 / (1023 * mp.pi)
    for i in range(54):
        _contains(mp, state.phase_lift_bounds[i], phases[i])
        _contains(
            mp, state.form_tau_rate_bounds[i], (-gradients[i] + gamma * currents[i]) / 4
        )
        _contains(mp, state.phase_tau_rate_bounds[i], gamma * gradients[i] / 4)
    _contains(mp, state.storage_bounds, storage)
    _contains(mp, state.attachment_work_bounds, contact_work)
    _contains(mp, state.balance_residual_bounds, 0)
    assert abs(
        _mp(mp, state.storage_tau_rate) + sum(g**2 for g in gradients) / 4
    ) < mp.mpf("1e-160")
    boundary = _independent_boundary(mp, _OFFSETS)
    _contains(mp, observed.boundary_cosine_bounds, boundary.real)
    _contains(mp, observed.boundary_sine_bounds, boundary.imag)
    _contains(mp, observed.boundary_amplitude_bounds, abs(boundary))
    moment = (mp.exp(1j * _mp(mp, origin)) * boundary).imag
    donor_current = sum(currents[45:]) / 9
    assert abs(donor_current + moment) < mp.mpf("1e-160")
    assert observed.relative_phase_origin == origin
    assert all("pi_over_two" not in item for item in state.geometry.scope)
    assert all("pi_over_two" not in item for item in state.scope)
    assert "external_origin_supplied_from_primitive_preparation" in state.geometry.scope
    assert (
        "raw_rational_offsets_relative_to_carrier_and_declared_external_origin"
        in state.scope
    )


def test_excluded_origin_changes_interaction_but_not_boundary_information():
    left = _observe(relative_phase_origin=Q(-2, 3))
    right = _observe(relative_phase_origin=Q(5, 4))
    assert left.boundary_cosine_bounds == right.boundary_cosine_bounds
    assert left.boundary_sine_bounds == right.boundary_sine_bounds
    assert left.boundary_amplitude_bounds == right.boundary_amplitude_bounds
    assert (
        left.interaction.form_tau_rate_bounds != right.interaction.form_tau_rate_bounds
    )
    assert (
        left.interaction.phase_tau_rate_bounds
        == right.interaction.phase_tau_rate_bounds
    )
    assert left.interaction.storage_tau_rate == right.interaction.storage_tau_rate


def test_moving_a_rational_origin_between_preparation_and_offset_preserves_state():
    origin = Q(7, 16)
    explicit = _observe(relative_phase_origin=origin)
    offsets = _OFFSETS[:45] + tuple(value + origin for value in _OFFSETS[45:])
    absorbed = _observe(relative_phase_origin=0, phase_offsets=offsets)
    assert (
        explicit.interaction.phase_lift_bounds == absorbed.interaction.phase_lift_bounds
    )
    assert (
        explicit.interaction.form_tau_rate_bounds
        == absorbed.interaction.form_tau_rate_bounds
    )
    assert explicit.interaction.storage_bounds == absorbed.interaction.storage_bounds


def test_nonzero_capability_does_not_mean_nonzero_response_at_every_origin(mp):
    # A static, exact flat outer carrier is a blind alignment, not a claim
    # about all acquired residual members or a finite source reconstruction.
    blind = _observe(relative_phase_origin=0, form=_ZERO, phase_offsets=_ZERO)
    _contains(mp, blind.boundary_amplitude_bounds, mp.mpf(5))
    for interval in blind.interaction.form_tau_rate_bounds:
        _contains(mp, interval, 0)
    minus = _observe(relative_phase_origin=Q(-1, 3), form=_ZERO, phase_offsets=_ZERO)
    plus = _observe(relative_phase_origin=Q(1, 3), form=_ZERO, phase_offsets=_ZERO)
    assert minus.interaction.child_form_tau_rate_bounds[5].lo > 0
    assert plus.interaction.child_form_tau_rate_bounds[5].hi < 0


def test_retained_quadratures_need_both_components_and_fine_matching(mp):
    offsets = list(_ZERO)
    offsets[0], offsets[1], offsets[45] = Q(1, 4), Q(-1, 4), Q(1, 3)
    observed = _observe(form=_ZERO, phase_offsets=offsets)
    boundary = _independent_boundary(mp, offsets)
    assert abs(boundary.imag) > mp.mpf("0.1")
    assert abs(boundary) > abs(boundary.real)
    _contains(mp, observed.boundary_amplitude_bounds, abs(boundary))
    # Swapping just donor slots changes fine matching without changing means.
    offsets[45], offsets[46] = offsets[46], offsets[45]
    changed = _observe(form=_ZERO, phase_offsets=offsets)
    assert (
        observed.interaction.child_phase_offset_means
        == changed.interaction.child_phase_offset_means
    )
    changed_boundary = _independent_boundary(mp, offsets)
    assert abs(boundary - changed_boundary) > mp.mpf("0.01")
    _contains(mp, changed.boundary_sine_bounds, changed_boundary.imag)


@pytest.mark.parametrize(
    "invalid", (True, "0", None, float("nan"), float("inf"), 1j, Decimal("0"))
)
def test_origin_admission_precedes_field_evaluation(monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid origin reached the full-state field kernel")

    monkeypatch.setattr(owner, "_observe_admitted_external_interaction", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _observe(relative_phase_origin=invalid)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf")))
def test_last_coordinate_is_admitted_before_field_evaluation(
    monkeypatch, field, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid coordinate reached the full-state field kernel")

    monkeypatch.setattr(owner, "_observe_admitted_external_interaction", forbidden)
    values = list(_ZERO)
    values[-1] = invalid
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: values})


def test_represented_and_subfloat_origins_remain_exact():
    represented = _observe(relative_phase_origin=0.1, form=_ZERO, phase_offsets=_ZERO)
    assert represented.relative_phase_origin == Q.from_float(0.1)
    tiny = Q(1, 10**400)
    exact = _observe(relative_phase_origin=tiny, form=_ZERO, phase_offsets=_ZERO)
    assert exact.relative_phase_origin == tiny > 0
    with pytest.raises(FrozenInstanceError):
        exact.relative_phase_origin = 0


def _forbidden_rebuild(*args, **kwargs):
    pytest.fail("invalid primitive source reached acquisition or recovery")


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid", (True, "1e-10", None, float("nan"), float("inf"), -_ERROR, 2 * _ERROR)
)
def test_all_source_allowances_are_admitted_before_rebuilding(
    monkeypatch, field, invalid
):
    monkeypatch.setattr(external, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(external, "_assess_constitutive_recovery", _forbidden_rebuild)
    values = [_ERROR] * 6
    values[-1] = invalid
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid", ((), (_ERROR,) * 5, (_ERROR,) * 7, {_ERROR}, {0: _ERROR})
)
def test_source_requires_exactly_six_ordered_allowances(monkeypatch, field, invalid):
    monkeypatch.setattr(external, "_assess_organized_outer_winding", _forbidden_rebuild)
    monkeypatch.setattr(external, "_assess_constitutive_recovery", _forbidden_rebuild)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: invalid})


def test_excess_source_generator_has_bounded_consumption(monkeypatch):
    monkeypatch.setattr(external, "_assess_organized_outer_winding", _forbidden_rebuild)
    consumed = []

    def values():
        for index in range(100):
            consumed.append(index)
            yield _ERROR

    with pytest.raises(ValueError):
        _assess(phase_error_bounds=values())
    assert consumed == list(range(7))


@pytest.mark.parametrize(
    "keyword",
    (
        "external_interaction",
        "source_report",
        "relative_phase_origin",
        "readout_error",
        "horizon",
    ),
)
def test_frozen_assessor_rejects_report_shortcuts_and_protocol_overrides(keyword):
    with pytest.raises(TypeError):
        _assess(**{keyword: 0})


def test_fresh_source_chain_is_rebuilt_once(monkeypatch, report):
    base = report.external_interaction
    calls = []

    def host(**kwargs):
        calls.append(("host", kwargs))
        return base.host_acquisition

    def donor(**kwargs):
        calls.append(("donor", kwargs))
        return base.donor_recovery

    def observation():
        calls.append(("observation", {}))
        return base.observation_protocol

    monkeypatch.setattr(external, "_assess_organized_outer_winding", host)
    monkeypatch.setattr(external, "_assess_constitutive_recovery", donor)
    monkeypatch.setattr(
        external, "_assess_organized_sequential_observation_protocol", observation
    )
    rebuilt = _assess()
    assert [name for name, _ in calls] == ["host", "donor", "observation"]
    assert calls[0][1] == dict(
        form_error_bounds=(_ERROR,) * 5, phase_error_bounds=(_ERROR,) * 5
    )
    assert calls[1][1] == dict(
        deformation=0, form_error_bound=_ERROR, phase_error_bound=_ERROR
    )
    assert rebuilt.external_interaction is not base
    assert rebuilt.observed_capability_bounds == report.observed_capability_bounds
    assert (
        rebuilt.boundary_cosine_bounds[0]
        is rebuilt.external_interaction.outer_one_boundary_moment_bounds
    )
    assert (
        rebuilt.boundary_cosine_bounds[1]
        is rebuilt.external_interaction.outer_zero_boundary_moment_bounds
    )


def test_legacy_pi_over_two_comparison_verdict_is_not_a_registration_premise(
    monkeypatch, report
):
    irrelevant = replace(
        report.external_interaction,
        comparison_certified=False,
        work_certified=False,
        identity_certified=False,
        admitted=False,
        status="unavailable",
    )
    calls = []

    def supply(**kwargs):
        calls.append(kwargs)
        return irrelevant

    monkeypatch.setattr(owner, "_assess_organized_external_interaction", supply)
    current = _assess()
    assert len(calls) == 1
    assert current.external_interaction is irrelevant
    assert current.observed_capability_bounds == report.observed_capability_bounds
    assert current.comparison_certified and current.admitted


@pytest.mark.parametrize(
    "failed,flag_index,condition",
    (
        ("source_certified", 0, "host_acquisition_not_certified"),
        ("source_certified", 1, "paired_host_control_not_certified"),
        ("source_certified", 2, "external_recovery_not_certified"),
        ("source_certified", 3, "preceding_sequential_observation_not_certified"),
        ("source_certified", 4, "actual_hosts_outside_observation_class"),
        ("work_certified", None, "attachment_work"),
        ("identity_certified", None, "internal_positive_gap[0]"),
    ),
)
def test_missing_actual_source_or_retention_premises_block_admission(
    monkeypatch, report, failed, flag_index, condition
):
    base = report.external_interaction
    if failed == "source_certified":
        flags = list(base.source_admission_flags)
        flags[flag_index] = False
        unavailable = replace(
            base,
            source_certified=False,
            source_admission_flags=tuple(flags),
        )
    elif failed == "work_certified":
        unavailable = replace(
            base, contact_work_upper_bound=base.attachment_work_budget
        )
    else:
        unavailable = replace(
            base,
            identity_margins=(("internal_positive_gap[0]", Q(0)),)
            + base.identity_margins[1:],
        )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **kwargs: unavailable
    )
    current = _assess()
    assert not getattr(current, failed)
    assert not current.admitted
    assert current.status == "unavailable"
    assert condition in current.unresolved_conditions
    if failed == "source_certified":
        assert not current.work_certified
        assert not current.identity_certified
        assert not current.comparison_certified


def test_original_source_and_finite_error_history_are_preserved(report):
    base = report.external_interaction
    assert report.protocol_revision == "d2f6f3df"
    assert base.protocol_revision == "7edac2ab"
    assert base.relative_preparation_dimension == 96
    assert base.form_error_bounds == base.phase_error_bounds == (_ERROR,) * 6
    assert base.observation_protocol.total_elapsed_scaled_time == Q(1, 5000)
    assert base.elapsed_scaled_time == Q(1, 1000)
    assert base.finite_response_remainder_upper_bound > 0
    assert base.contact_work_upper_bound < 100
    assert max(base.joined_initial_storage_upper_bounds) < 200
    assert len(base.identity_margins) == 21
    assert all(margin > 0 for _, margin in base.identity_margins)
    assert (
        report.source_certified and report.work_certified and report.identity_certified
    )


def test_acquired_quadratures_and_norms_include_independent_carrier_points(report, mp):
    nominal = (
        sum((mp.exp(-4j * (c - 2) / mp.pi) for c in range(5)), mp.mpc(0)),
        mp.mpc(5),
    )
    for boundary, real, imag, amplitude in zip(
        nominal,
        report.boundary_cosine_bounds,
        report.boundary_sine_bounds,
        report.boundary_amplitude_bounds,
    ):
        _contains(mp, real, boundary.real)
        _contains(mp, imag, boundary.imag)
        _contains(mp, amplitude, abs(boundary))
        assert amplitude.lo >= 0
        # Complex rotation attains both extremal imaginary projections;
        # the amplitude is not merely the real component at pi/2.
        witness = mp.pi / 2 - mp.arg(boundary)
        assert abs((mp.exp(1j * witness) * boundary).imag - abs(boundary)) < mp.mpf(
            "1e-160"
        )
    assert report.boundary_cosine_bounds[0].hi < 0
    assert report.boundary_cosine_bounds[1].lo > 0


def test_capability_encloses_finite_error_and_adversarial_two_reading_noise(report, mp):
    base = report.external_interaction
    factor = _mp(mp, base.elapsed_scaled_time) / (4 * 1023 * mp.pi)
    uncertainty = _mp(mp, base.finite_response_remainder_upper_bound + 2 * _ERROR)
    for amplitude, capability in zip(
        report.boundary_amplitude_bounds, report.observed_capability_bounds
    ):
        minimum = max(mp.mpf(0), factor * _mp(mp, amplitude.lo) - uncertainty)
        maximum = factor * _mp(mp, amplitude.hi) + uncertainty
        assert 0 <= _mp(mp, capability.lo) <= minimum
        assert _mp(mp, capability.hi) >= maximum
    # The zero family's lower capability is existential in beta and uniform
    # in bounded noise. The one's upper is uniform in both beta and noise.
    one, zero = report.observed_capability_bounds
    assert one.hi < Q(1, 10**8)
    assert zero.lo > Q(3, 10**7)
    assert report.outer_one_upper_margin == Q(1, 10**8) - one.hi
    assert report.outer_zero_lower_margin == zero.lo - Q(3, 10**7)
    assert report.paired_absolute_response_gap_lower_bound == zero.lo - one.hi > 0
    assert report.comparison_certified and report.admitted
    assert report.unresolved_conditions == ()
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


@pytest.mark.parametrize(
    "cosine,sine",
    (
        (I(3), I(4)),
        (I(-4, -3), I(1, 2)),
        (I(-2, 3), I(-4, 1)),
        (I(-2, 3), I(4, 5)),
        (I(0), I(0)),
    ),
)
def test_quadrature_norm_encloses_rectangle_extrema(mp, cosine, sine):
    norm = owner._quadrature_norm(cosine, sine)
    re = tuple(_mp(mp, v) for v in (cosine.lo, cosine.hi))
    im = tuple(_mp(mp, v) for v in (sine.lo, sine.hi))
    nearest_re = 0 if re[0] <= 0 <= re[1] else min(abs(v) for v in re)
    nearest_im = 0 if im[0] <= 0 <= im[1] else min(abs(v) for v in im)
    lower = mp.sqrt(nearest_re**2 + nearest_im**2)
    upper = max(mp.sqrt(x**2 + y**2) for x in re for y in im)
    _contains(mp, norm, lower)
    _contains(mp, norm, upper)
    assert norm.lo >= 0
    if lower == 0:
        assert norm.lo == 0
