"""Static algebra, retained arithmetic and controlled mediation wiring.

The reserved assessment is read once; formation and trajectory producers are
forbidden. Controlled handoffs test branches without supplying new scientific
evidence or replaying the frozen family acquisition.
"""

import json
import subprocess
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_mediation as owner
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    # An unrelated primitive record used only to exercise rejection paths.
    return dict(
        formation_time=Q(7),
        relaxation_duration=Q(13),
        contact_duration=Q(1, 17),
        probe_amplitude=Q(1, 19),
        form_error_bound=Q(1, 23),
        phase_error_bound=Q(1, 29),
        endpoint_radius=Q(1, 31),
        radius=Q(1, 37),
        decay_power=3,
        readout_error_bound=Q(1, 41),
        contact_work_allowance=Q(1, 43),
        probe_work_allowance=Q(1, 47),
    )


@pytest.fixture(autouse=True)
def no_scientific_acquisition(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import _sine_formed_contact, relational_sine_formed_classes

    def forbidden(*args, **kwargs):
        pytest.fail("mediation controls must not acquire a source or replay a worker")

    monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
    monkeypatch.setattr(_sine_formed_contact, "_unprobed_handoff", forbidden)
    monkeypatch.setattr(
        _sine_formed_contact, "assess_sine_formed_class_pair", forbidden
    )
    monkeypatch.setattr(
        relational_sine_formed_classes, "assess_sine_formed_class_pair", forbidden
    )
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in ("flow_jets", "picard_tube", "validated_box_taylor_step"):
        monkeypatch.setattr(_validated_taylor, name, forbidden)


def test_twelve_mandatory_primitives_accept_no_cached_state_or_verdict():
    parameters = signature(owner.assess_sine_class_mediation).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 12
    assert all(
        item.kind == Parameter.KEYWORD_ONLY and item.default == Parameter.empty
        for item in parameters.values()
    )
    for name in (
        "source_handoff",
        "response",
        "classes",
        "contacts",
        "phase_origins",
        "initial_forms",
        "baseline",
        "response_threshold",
    ):
        with pytest.raises(TypeError):
            owner.assess_sine_class_mediation(**_arguments(), **{name: None})


@pytest.mark.parametrize("name", tuple(_arguments()))
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_authoritative_primitives_reject_before_handoff(name, invalid):
    values = _arguments()
    values[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_class_mediation(**values)


@pytest.mark.parametrize("name", tuple(_arguments()))
def test_negative_primitive_rejects_before_handoff(name):
    values = _arguments()
    values[name] = -1
    with pytest.raises(ValueError):
        owner.assess_sine_class_mediation(**values)


@pytest.mark.parametrize(
    "name,invalid",
    (
        ("contact_duration", Q(1, 4) + Q(1, 2**300)),
        ("radius", Q(1, 12) + Q(1, 2**300)),
        ("radius", 0),
        ("endpoint_radius", 0),
        ("formation_time", Q(20480) + Q(1, 2**300)),
        ("decay_power", 4097),
        ("decay_power", Q(2)),
        ("decay_power", 2.0),
        ("decay_power", np.int64(2)),
        ("probe_amplitude", Decimal("1e-400")),
        ("contact_duration", complex(1, 0)),
        ("probe_work_allowance", "1"),
    ),
)
def test_exact_domain_and_representation_boundaries_reject(name, invalid):
    values = _arguments()
    values[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_class_mediation(**values)


def test_rebuilt_walk_depends_on_the_two_actual_contacts():
    geometry = owner._central_port_geometry(3, ((0, 1), (1, 2)))
    assert owner._mediation_walks(geometry) == (Q(1, 12), Q(1, 24))
    # Removing either contact destroys donor-to-receiver transmission. This
    # checks the coefficient is consumed from the rebuilt matrices.
    for contacts in (((0, 1),), ((1, 2),), ()):
        disconnected = owner._central_port_geometry(3, contacts)
        assert owner._mediation_walks(disconnected) == (0, 0)


def test_support_clock_and_observation_are_declared_without_live_state():
    fields = owner.SineClassMediation.__dataclass_fields__
    edges = fields["edges"].default
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))
    assert len(edges) == 29 and sum(degrees) == 58
    assert tuple(degrees[node] for node in (4, 13, 22)) == (3, 4, 3)
    assert all(degrees[node] == 2 for node in range(27) if node not in (4, 13, 22))
    assert fields["classes"].default == ((1, 1, 1), (1, 2, 1))
    assert fields["phase_origins"].default == (0, 0, 0)
    assert fields["clock"].default == "tau=e*t; e=1023/1024"
    assert fields["phase_blind_law"].default == "x'=-KLx; theta'=gamma*KLx"
    assert "x_unprobed[22](h)" in fields["response_definition"].default


def _decode(value):
    """Read projected exact arithmetic, without admitting a report as evidence."""
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return Q(value["numerator"], value["denominator"])
        if set(value) == {"lo", "hi"}:
            return I(_decode(value["lo"]), _decode(value["hi"]))
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    return value


@pytest.fixture(scope="module")
def retained_report():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/sine_formed_classes/class-mediation-v1.json"
    )
    packet = json_loads(path.read_text(encoding="utf-8"))
    assert packet["schema"] == "tnfr.sine-class-mediation-assessment.v1"
    return _decode(packet["response"]["report"])


def test_retained_response_encloses_rebuilt_bound_and_strict_separation(
    retained_report,
):
    r = retained_report
    a, h, eps = r["probe_amplitude"], r["contact_duration"], r["endpoint_radius"]
    gamma, eta, difference = (
        r["gamma_bounds"],
        r["eta_bounds"],
        r["mediator_cosine_difference_bounds"],
    )
    # Only rational arithmetic on retained elementary-function enclosures; no
    # formation, response assessor or trigonometric producer is reconstructed.
    lead = a * eta * difference * h**3 / 144
    tail = 9 * a * eta.hi * difference.hi * h**4 / (1 - 3 * h / 4)
    bootstrap = 1 - 2 * eta.hi * h**2
    nonlinear = 16 * gamma.hi**3 * a**2 * h**3 / (3 * bootstrap**2 * (1 - 3 * h))
    preparation, noise = 4 * eps / (1 - 3 * h), 4 * r["readout_error_bound"]
    assert r["leading_contrast_bounds"].contains(lead)
    assert r["linear_tail_upper_bound"] == tail
    assert r["nonlinear_contrast_error_upper_bound"] == nonlinear
    assert r["preparation_contrast_error_upper_bound"] == preparation
    assert r["readout_contrast_error_upper_bound"] == noise
    expected = lead + I(-tail - nonlinear, tail + nonlinear)
    expected += I(-preparation, preparation)
    expected += I(-noise, noise)
    assert r["recorded_contrast_bounds"].contains(expected)
    assert r["recorded_contrast_bounds"].lo > Q(1, 10**25)
    assert r["recorded_contrast_bounds"].lo > preparation + noise


def test_retained_work_and_mean_leaf_follow_actual_event_primitives(retained_report):
    r = retained_report
    a, eps, radius = r["probe_amplitude"], r["endpoint_radius"], r["radius"]
    before = r["joined_bounds"]
    z2 = 6 * eps**2 + 2 * a * eps + Q(26, 27) * a**2
    energy = 22 * eps**2 + 6 * a * eps + Q(3, 2) * a**2
    assert r["contact_work_bounds"].contains(I(0, 8 * eps**2))
    assert r["probe_work_bounds"].contains(
        I(Q(3, 2) * a**2 - 6 * a * eps, Q(3, 2) * a**2 + 6 * a * eps)
    )
    assert r["post_probe_radius_squared_upper_bound"] == z2 < radius**2
    assert (
        r["post_probe_excess_storage_upper_bound"]
        == energy
        < before["joined_barrier_lower_bound"]
    )
    assert r["post_probe_form_mean_bounds"].contains(
        I(3 * a / 58 - 4 * eps / 58, 3 * a / 58 + 4 * eps / 58)
    )
    assert r["post_probe_phase_mean_bounds"].contains(I(-4 * eps / 58, 4 * eps / 58))
    source = r["source_handoff"]
    for key in (
        "endpoint_form_norm_squared_upper_bounds",
        "endpoint_phase_norm_squared_upper_bounds",
    ):
        assert all(value <= eps**2 for value in source[key])


@dataclass(frozen=True)
class _ControlledHandoff:
    """Test-only branch selector; never a scientific source certificate."""

    handoff_certified_by_class: tuple[bool, bool]


def _controlled_arguments():
    values = _arguments()
    values.update(
        contact_duration=Q(1, 8192),
        probe_amplitude=Q(1, 2048),
        endpoint_radius=Q(1, 2**100),
        radius=Q(1, 12),
        readout_error_bound=Q(0),
        contact_work_allowance=Q(1, 1000),
        probe_work_allowance=Q(1, 1000),
    )
    return values


@pytest.fixture
def controlled_assessor(monkeypatch, no_scientific_acquisition):
    calls = []

    def assess(*, flags=(True, True), **changes):
        def handoff(**primitives):
            calls.append(primitives)
            return _ControlledHandoff(flags)

        monkeypatch.setattr(owner, "_unprobed_handoff", handoff)
        values = _controlled_arguments()
        values.update(changes)
        return owner.assess_sine_class_mediation(**values)

    return assess, calls


@pytest.mark.parametrize("flags", ((False, False), (True, False), (False, True)))
def test_unavailable_source_suppresses_actual_channels(controlled_assessor, flags):
    assess, calls = controlled_assessor
    report = assess(flags=flags)
    assert len(calls) == 1 and not report.source_handoff_certified
    assert report.leading_contrast_bounds.lo > 0
    for name in (
        "preparation_contrast_error_upper_bound",
        "readout_contrast_error_upper_bound",
        "actual_contrast_bounds",
        "recorded_contrast_bounds",
        "phase_blind_recorded_contrast_bounds",
        "phase_blind_exclusion_margin_bounds",
        "contact_work_bounds",
        "probe_work_bounds",
        "probe_work_margin",
        "post_probe_radius_squared_upper_bound",
        "post_probe_excess_storage_upper_bound",
        "post_probe_radius_margin",
        "post_probe_storage_margin",
        "post_probe_form_mean_bounds",
        "post_probe_phase_mean_bounds",
    ):
        assert getattr(report, name) is None
    for name in (
        "response_certified",
        "phase_blind_alternative_excluded",
        "baseline_identity_certified",
        "probe_identity_certified",
        "identity_certified",
        "contact_work_within_allowance",
        "probe_work_within_allowance",
        "work_within_allowances",
    ):
        assert getattr(report, name) is False
    assert report.status == "unavailable"


@pytest.mark.parametrize("name", ("probe_amplitude", "contact_duration"))
def test_zero_input_or_horizon_has_no_positive_response(controlled_assessor, name):
    assess, _ = controlled_assessor
    report = assess(**{name: 0})
    assert report.source_handoff_certified
    assert report.leading_contrast_bounds == I(0)
    assert (
        report.linear_tail_upper_bound
        == report.nonlinear_contrast_error_upper_bound
        == 0
    )
    assert report.recorded_contrast_bounds.contains(0)
    assert not report.response_certified and not report.phase_blind_alternative_excluded
    assert report.identity_certified and report.work_within_allowances


def test_strict_response_equality_and_four_reading_error_count(controlled_assessor):
    assess, _ = controlled_assessor
    zero_noise = assess()
    assert zero_noise.status == "certified_class_mediation"
    delta = zero_noise.actual_contrast_bounds.lo / 4
    report = assess(readout_error_bound=delta)
    assert report.readout_contrast_error_upper_bound == 4 * delta
    assert report.recorded_contrast_bounds.lo == 0
    assert not report.response_certified
    assert report.identity_certified and report.work_within_allowances


def test_positive_response_alone_does_not_exclude_phase_blind_band(controlled_assessor):
    assess, _ = controlled_assessor
    zero_noise = assess()
    report = assess(readout_error_bound=zero_noise.actual_contrast_bounds.lo / 6)
    assert report.recorded_contrast_bounds.lo > 0 and report.response_certified
    assert report.phase_blind_exclusion_margin_bounds.lo < 0
    assert not report.phase_blind_alternative_excluded
    assert report.status == "unavailable"


def test_identity_storage_equality_is_strict_and_independent(
    controlled_assessor, monkeypatch
):
    assess, _ = controlled_assessor
    original = owner._joined_port_bounds
    values = _controlled_arguments()
    a, eps = values["probe_amplitude"], values["endpoint_radius"]
    post_energy = 22 * eps**2 + 6 * a * eps + Q(3, 2) * a**2

    def boundary(**kwargs):
        return replace(original(**kwargs), joined_barrier_lower_bound=post_energy)

    monkeypatch.setattr(owner, "_joined_port_bounds", boundary)
    report = assess()
    assert report.post_probe_storage_margin == 0
    assert report.baseline_identity_certified and not report.probe_identity_certified
    assert not report.identity_certified and report.response_certified
    assert report.work_within_allowances and report.status == "unavailable"


def test_large_probe_can_fail_radius_without_erasing_response_bound(
    controlled_assessor,
):
    assess, _ = controlled_assessor
    report = assess(probe_amplitude=Q(1, 2), probe_work_allowance=1)
    assert report.post_probe_radius_margin < 0
    assert not report.probe_identity_certified
    assert report.actual_contrast_bounds is not None
    assert report.baseline_identity_certified


def test_work_allowances_include_equality_but_not_smaller_exact_values(
    controlled_assessor,
):
    assess, _ = controlled_assessor
    values = _controlled_arguments()
    a, eps = values["probe_amplitude"], values["endpoint_radius"]
    contact, probe = 8 * eps**2, Q(3, 2) * a**2 + 6 * a * eps
    report = assess(contact_work_allowance=contact, probe_work_allowance=probe)
    assert report.contact_work_within_allowance and report.probe_work_within_allowance
    assert report.joined_bounds.work_margin_bounds == I(0)
    assert report.probe_work_margin == 0
    assert report.status == "certified_class_mediation"
    smaller = Q(1, 2**500)
    for key, allowance in (
        ("contact_work_allowance", contact),
        ("probe_work_allowance", probe),
    ):
        failed = assess(**{key: allowance - smaller})
        assert not failed.work_within_allowances
        assert failed.identity_certified and failed.response_certified


def test_probe_work_retains_signed_cross_term_and_nonzero_mean(controlled_assessor):
    assess, _ = controlled_assessor
    a, eps = Q(1, 16384), Q(1, 1024)
    report = assess(probe_amplitude=a, endpoint_radius=eps)
    assert report.probe_work_bounds.lo < 0 < report.probe_work_bounds.hi
    assert report.probe_work_bounds.contains(Q(3, 2) * a**2 - 6 * a * eps)
    assert report.probe_form_mean_shift == 3 * a / 58
    assert report.post_probe_form_mean_bounds.contains(3 * a / 58)
    assert report.post_probe_phase_mean_bounds.contains(0)


def test_normalized_primitive_handoff_is_fresh_once_per_call(controlled_assessor):
    assess, calls = controlled_assessor
    exact_tiny = Q(1, 10**400)
    report = assess(
        formation_time=0.125, form_error_bound=exact_tiny, phase_error_bound=0.0
    )
    assert len(calls) == 1
    assert calls[0] == {
        "formation_time": Q(1, 8),
        "relaxation_duration": Q(13),
        "form_error_bound": exact_tiny,
        "phase_error_bound": Q(0),
        "endpoint_radius": report.endpoint_radius,
        "radius": report.radius,
        "decay_power": 3,
    }
    assert all(
        isinstance(value, Q) for key, value in calls[0].items() if key != "decay_power"
    )
    assert report.form_error_bound == exact_tiny


def test_subgrid_positive_impulse_is_preserved_but_not_coerced_to_a_certificate(
    controlled_assessor,
):
    assess, _ = controlled_assessor
    amplitude = Q(1, 10**400)
    report = assess(probe_amplitude=amplitude)
    assert report.probe_amplitude == amplitude > 0
    assert report.probe_form_mean_shift == 3 * amplitude / 58 > 0
    assert report.leading_contrast_bounds.lo <= 0 <= report.leading_contrast_bounds.hi
    assert not report.response_certified


def test_maximum_horizon_is_admitted_without_promising_positive_contrast(
    controlled_assessor,
):
    assess, _ = controlled_assessor
    report = assess(contact_duration=Q(1, 4))
    assert report.nonlinear_bootstrap_margin > 0
    assert report.recorded_contrast_bounds.lo < 0 and not report.response_certified


@pytest.mark.parametrize("flags", ((True, True), (True, False)))
def test_sdk_projection_preserves_exact_fractions_and_unavailable_nulls(
    controlled_assessor, flags
):
    assess, _ = controlled_assessor
    report = assess(flags=flags)
    direct = report.to_dict()
    projected = relational_report_to_dict(report)
    assert direct["schema"] == "tnfr.sine-class-mediation.v1"
    assert projected["report_type"] == "SineClassMediation"
    assert projected["report"] == direct["report"]
    encoded = json.dumps(projected, allow_nan=False)
    restored = json_loads(encoded)
    assert restored == projected
    assert restored["report"]["probe_amplitude"] == {
        "numerator": 1,
        "denominator": 2048,
    }
    assert (restored["report"]["recorded_contrast_bounds"] is None) == (
        flags != (True, True)
    )
