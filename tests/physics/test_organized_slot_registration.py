"""Independent static fields on generic shifted contacts, without source runs."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_slot_registration as owner
from tnfr.physics._organized_external_registration import (
    _observe_organized_external_registration,
)


_SLOTS = (1, 2, 0, 3, 8)
_FORM = tuple(Q((7 * index) % 19 - 9, 64) for index in range(54))
_PHASE = tuple(Q((11 * index) % 23 - 11, 256) for index in range(54))
_ZERO = (Q(0),) * 54
_BETA = Q(2, 5)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _observe(**changes):
    return owner._observe_organized_slot_registration(
        **(
            dict(
                slot_offsets=_SLOTS,
                relative_phase_origin=_BETA,
                form=_FORM,
                phase_offsets=_PHASE,
            )
            | changes
        )
    )


def _edges(slots):
    internal = tuple(
        (9 * child + j, 9 * child + (j + 1) % 9) for child in range(6) for j in range(9)
    )
    outer = tuple(
        (9 * child + j, 9 * ((child + 1) % 5) + j)
        for j in range(9)
        for child in range(5)
    )
    contact = tuple(
        (9 * child + j, 45 + (j + shift) % 9)
        for child, shift in enumerate(slots)
        for j in range(9)
    )
    return internal + outer + contact, contact


def _fields(mp, edges, phase):
    gradient, current = [Q(0)] * 54, [mp.mpf(0)] * 54
    form_cost, phase_cost = Q(0), mp.mpf(0)
    for left, right in edges:
        difference = _FORM[left] - _FORM[right]
        angle = phase[right] - phase[left]
        gradient[left] += difference
        gradient[right] -= difference
        current[left] += mp.sin(angle)
        current[right] -= mp.sin(angle)
        form_cost += difference**2 / 2
        phase_cost += 1 - mp.cos(angle)
    return gradient, current, form_cost, phase_cost


def test_complete_rows_means_clocks_and_work_match_independent_shifted_edges(mp):
    report = _observe()
    state, geometry = report.interaction, report.interaction.geometry
    edges, contact = _edges(_SLOTS)
    assert geometry.weighted_edges == tuple((a, b, Q(1)) for a, b in edges)
    assert len(set(edges)) == 144 and len(set(contact)) == 45
    degrees = tuple(sum(node in edge for edge in edges) for node in range(54))
    assert degrees == geometry.support_degrees == (5,) * 45 + (7,) * 9
    assert geometry.slot_offsets == _SLOTS
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    phase = tuple(
        alpha * (node % 9 - 4) + _mp(mp, offset) + (_mp(mp, _BETA) if node >= 45 else 0)
        for node, offset in enumerate(_PHASE)
    )
    gradient, current, form_cost, phase_cost = _fields(mp, edges, phase)
    boundary_gradient, boundary_current, boundary_form, boundary_phase = _fields(
        mp, contact, phase
    )
    assert state.form_gradient == tuple(gradient)
    assert state.contact_form_gradient == tuple(boundary_gradient)
    assert state.form_storage == form_cost
    _contains(mp, state.phase_storage_bounds, phase_cost)
    _contains(mp, state.storage_bounds, _mp(mp, form_cost) + phase_cost)
    _contains(mp, state.attachment_work_bounds, _mp(mp, boundary_form) + boundary_phase)
    expected_loss = -sum(value**2 for value in gradient) / 4
    assert state.storage_tau_rate == expected_loss
    _contains(mp, state.storage_rate_from_rows_bounds, _mp(mp, expected_loss))
    assert state.balance_residual_bounds.contains(0)
    for prefix, gradients, currents in (
        ("", gradient, current),
        ("contact_", boundary_gradient, boundary_current),
    ):
        dx = tuple((-_mp(mp, a) + gamma * b) / 4 for a, b in zip(gradients, currents))
        dt = tuple(gamma * _mp(mp, value) / 4 for value in gradients)
        for boxes, exact in (
            (getattr(state, prefix + "phase_current_bounds"), currents),
            (getattr(state, prefix + "form_tau_rate_bounds"), dx),
            (getattr(state, prefix + "phase_tau_rate_bounds"), dt),
        ):
            for box, value in zip(boxes, exact):
                _contains(mp, box, value)
        for child in range(6):
            span = slice(9 * child, 9 * child + 9)
            _contains(
                mp,
                getattr(state, prefix + "child_form_tau_rate_bounds")[child],
                sum(dx[span]) / 9,
            )
            _contains(
                mp,
                getattr(state, prefix + "child_phase_tau_rate_bounds")[child],
                sum(dt[span]) / 9,
            )
        if not prefix:
            for box, value in zip(state.form_original_rate_bounds, dx):
                _contains(mp, box, value * mp.mpf(1023) / 1024)
            for box, value in zip(state.phase_original_rate_bounds, dt):
                _contains(mp, box, value * mp.mpf(1023) / 1024)
    for box, value in zip(state.phase_lift_bounds, phase):
        _contains(mp, box, value)
    assert sum(state.form_gradient) == sum(state.contact_form_gradient) == 0
    assert state.child_form_means == tuple(
        sum(_FORM[9 * c : 9 * c + 9]) / 9 for c in range(6)
    )
    assert state.child_phase_offset_means == tuple(
        sum(_PHASE[9 * c : 9 * c + 9]) / 9 for c in range(6)
    )


def test_quadratures_include_fine_matching_and_exclude_only_beta(mp):
    report = _observe()
    _, contact = _edges(_SLOTS)
    alpha = 2 * mp.pi / 9
    moment = (
        sum(
            (
                mp.exp(
                    1j * (alpha * ((b % 9) - (a % 9)) + _mp(mp, _PHASE[b] - _PHASE[a]))
                )
                for a, b in contact
            ),
            mp.mpc(0),
        )
        / 9
    )
    _contains(mp, report.boundary_cosine_bounds, moment.real)
    _contains(mp, report.boundary_sine_bounds, moment.imag)
    _contains(mp, report.boundary_amplitude_bounds, abs(moment))
    changed_origin = _observe(relative_phase_origin=Q(-7, 3))
    assert report.boundary_cosine_bounds == changed_origin.boundary_cosine_bounds
    assert report.boundary_sine_bounds == changed_origin.boundary_sine_bounds
    helper = owner._bounded_slot_registration_moments(
        slot_offsets=_SLOTS, phase_offsets=_PHASE
    )
    assert helper == (report.boundary_cosine_bounds, report.boundary_sine_bounds)
    assert (
        report.interaction.contact_phase_current_bounds
        != changed_origin.interaction.contact_phase_current_bounds
    )


def test_carrier_slot_wraps_have_positive_donor_minus_host_sign(mp):
    cosine, sine = owner._bounded_slot_registration_moments(
        slot_offsets=_SLOTS, phase_offsets=_ZERO
    )
    exact = sum(mp.exp(2j * mp.pi * shift / 9) for shift in _SLOTS)
    _contains(mp, cosine, exact.real)
    _contains(mp, sine, exact.imag)
    assert sine.lo > 0 and exact.imag > 0
    shifted = list(_ZERO)
    shifted[45:] = (2 * pi_interval(),) * 9
    periodic = owner._bounded_slot_registration_moments(
        slot_offsets=_SLOTS, phase_offsets=shifted
    )
    _contains(mp, periodic[0], exact.real)
    _contains(mp, periodic[1], exact.imag)


def test_interval_offsets_retain_interior_fine_states(mp):
    width = Q(1, 4096)
    boxes = tuple(I(value - width, value + width) for value in _PHASE)
    cosine, sine = owner._bounded_slot_registration_moments(
        slot_offsets=_SLOTS, phase_offsets=boxes
    )
    selected = tuple(
        value + (width if index % 2 else -width) for index, value in enumerate(_PHASE)
    )
    _, contact = _edges(_SLOTS)
    exact = (
        sum(
            (
                mp.exp(
                    1j
                    * (
                        2 * mp.pi * ((b % 9) - (a % 9)) / 9
                        + _mp(mp, selected[b] - selected[a])
                    )
                )
                for a, b in contact
            ),
            mp.mpc(0),
        )
        / 9
    )
    _contains(mp, cosine, exact.real)
    _contains(mp, sine, exact.imag)


def test_zero_matching_preserves_all_existing_numerical_observations():
    previous = _observe_organized_external_registration(
        relative_phase_origin=_BETA, form=_FORM, phase_offsets=_PHASE
    )
    current = _observe(slot_offsets=(0,) * 5)
    for name, value in vars(previous).items():
        if name != "interaction":
            assert getattr(current, name) == value
    for name, value in vars(previous.interaction).items():
        if name != "geometry":
            assert getattr(current.interaction, name) == value
    for name, value in vars(previous.interaction.geometry).items():
        if name != "scope":
            assert getattr(current.interaction.geometry, name) == value


@pytest.mark.parametrize("bad", (True, Q(1), 1.0, "1", -1, 9, None))
def test_slot_labels_reject_coercion_before_geometry(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_derive_organized_external_interaction",
        lambda: pytest.fail("invalid slot reached geometry"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._derive_organized_slot_registration(slot_offsets=(0, 0, 0, 0, bad))


@pytest.mark.parametrize("bad", ((), (0,) * 4, (0,) * 6, {0}, "00000", None))
def test_slot_shape_requires_exactly_five_ordered_entries(bad):
    with pytest.raises((TypeError, ValueError)):
        _observe(slot_offsets=bad)


@pytest.mark.parametrize("field", ("form", "phase_offsets", "relative_phase_origin"))
@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), complex(0), None))
def test_state_admission_precedes_geometry_and_fields(field, bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_derive_organized_slot_registration",
        lambda **kwargs: pytest.fail("invalid state reached geometry"),
    )
    value = bad if field == "relative_phase_origin" else (_ZERO[:-1] + (bad,))
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: value})


@pytest.mark.parametrize("bad", (True, float("nan"), complex(0), None, "0"))
def test_interval_consumer_admits_every_scalar_before_geometry(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_derive_organized_slot_registration",
        lambda **kwargs: pytest.fail("invalid interval row reached geometry"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._bounded_slot_registration_moments(
            slot_offsets=_SLOTS, phase_offsets=_ZERO[:-1] + (bad,)
        )


def test_normalized_geometry_preserves_old_marked_cycles_and_is_immutable():
    slots = list(_SLOTS)
    geometry = owner._derive_organized_slot_registration(slot_offsets=slots)
    slots[-1] = 0
    assert geometry.slot_offsets == _SLOTS
    assert geometry.internal_cycles == tuple(
        tuple(9 * c + j for j in range(9)) for c in range(6)
    )
    assert geometry.outer_cycles == tuple(
        tuple(9 * c + j for c in range(5)) for j in range(9)
    )
    with pytest.raises(FrozenInstanceError):
        geometry.slot_offsets = (0,) * 5
