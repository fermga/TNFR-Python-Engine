"""Static and synthetic own-law contact controls; no actual gate evaluation."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.physics import _organized_alternative_contact as owner
from tnfr.physics import _organized_external_interaction as sine_contact
from tnfr.physics import _organized_external_registration as sine_registration
from tnfr.physics import _organized_outer_winding as sine_source
from tnfr.physics import _organized_sequential_observation as word_owner


_ERROR = Q(1, 10**10)
_SYNTHETIC_Q = Q(1, 10**12)
_INTERNAL = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
_OUTER = tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for j in range(9) for c in range(5))
_CONTACT = tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
_EDGES = _INTERNAL + _OUTER + _CONTACT


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    def forbidden(*args, **kwargs):
        pytest.fail("static contact controls attempted source or response assessment")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, attribute in (
            (owner, "_assess_organized_alternative_outer_acquisition"),
            (owner, "_assess_constitutive_recovery"),
            (sine_contact, "_assess_organized_external_interaction"),
            (sine_registration, "_assess_organized_external_registration"),
            (sine_source, "_assess_organized_outer_winding"),
            (sine_source, "_derive_organized_outer_resource_bounds"),
            (sine_source, "certify_sine_prepared_entry"),
            (word_owner, "_assess_organized_sequential_observation_protocol"),
        ):
            patch.setattr(module, attribute, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def bounds():
    # This is a conditional, deliberately different residual allowance.
    # It does not evaluate any actual original host or donor preparation.
    return owner._derive_organized_alternative_contact_bounds(
        recovered_radius=_SYNTHETIC_Q
    )


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, box, value):
    assert _mp(mp, box.lo) <= value <= _mp(mp, box.hi)


def _potential(mp, angle):
    return 1 - mp.cos(angle) + (1 - mp.cos(9 * angle)) ** 2 / (6 * 9**4)


def _current(mp, angle):
    return mp.sin(angle) + mp.sin(9 * angle) * (1 - mp.cos(9 * angle)) / 2187


def _fields(mp, form, phase, edges=_EDGES):
    gradient, current = [mp.mpf(0)] * 54, [mp.mpf(0)] * 54
    storage = mp.mpf(0)
    for a, b in edges:
        difference, angle = form[a] - form[b], phase[b] - phase[a]
        gradient[a] += difference
        gradient[b] -= difference
        value = _current(mp, angle)
        current[a] += value
        current[b] -= value
        storage += difference**2 / 2 + _potential(mp, angle)
    return gradient, current, storage


@pytest.fixture(scope="module")
def state():
    forms = tuple(Q((7 * i) % 19 - 9, 64) for i in range(54))
    mean = sum(forms, Q(0)) / 54
    return (
        tuple(value - mean for value in forms),
        tuple(Q((11 * i) % 23 - 11, 256) for i in range(54)),
        Q(7, 11),
    )


@pytest.fixture(scope="module")
def observation(state):
    form, offsets, beta = state
    return owner._observe_organized_alternative_contact(
        form=form, phase_offsets=offsets, relative_phase_origin=beta
    )


def test_full_support_is_not_degree_normalized_and_origin_is_supplied(
    observation, state
):
    assert len(observation.geometry.weighted_edges) == 144
    assert tuple((a, b) for a, b, _ in observation.geometry.weighted_edges) == _EDGES
    assert observation.geometry.support_degrees == (5,) * 45 + (7,) * 9
    assert observation.geometry.clock_factor == Q(1, 4)
    assert observation.relative_phase_origin == state[2]
    assert observation.geometry.external_phase_origin_bounds.contains(state[2])
    assert all("pi_over_two" not in item for item in observation.geometry.scope)
    assert observation.geometry.held_capacities == (Q(1),) * 54


def test_complete_rows_reaction_and_storage_use_the_actual_edge_law(
    mp, observation, state
):
    form, offsets, beta = state
    x = tuple(_mp(mp, value) for value in form)
    phase = tuple(
        2 * mp.pi * (i % 9 - 4) / 9
        + _mp(mp, offset)
        + (_mp(mp, beta) if i >= 45 else 0)
        for i, offset in enumerate(offsets)
    )
    gradient, current, storage = _fields(mp, x, phase)
    boundary_gradient, boundary_current, work = _fields(mp, x, phase, _CONTACT)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dt = tuple(gamma * a / 4 for a in gradient)
    loss = sum(value**2 for value in gradient) / 4
    _contains(mp, observation.storage_bounds, storage)
    _contains(mp, observation.attachment_work_bounds, work)
    assert abs(_mp(mp, observation.storage_tau_rate) + loss) < mp.mpf("1e-80")
    assert observation.balance_residual_bounds.contains(0)
    for i in range(54):
        _contains(mp, observation.form_tau_rate_bounds[i], dx[i])
        _contains(mp, observation.phase_tau_rate_bounds[i], dt[i])
        _contains(mp, observation.form_original_rate_bounds[i], 1023 * dx[i] / 1024)
        _contains(mp, observation.phase_original_rate_bounds[i], 1023 * dt[i] / 1024)
        _contains(mp, observation.contact_phase_current_bounds[i], boundary_current[i])
        assert abs(
            _mp(mp, observation.contact_form_gradient[i]) - boundary_gradient[i]
        ) < mp.mpf("1e-80")
    assert abs(sum(boundary_current[:45]) + sum(boundary_current[45:])) < mp.mpf(
        "1e-80"
    )
    assert abs(sum(dx)) < mp.mpf("1e-80")
    donor_mean = sum(x[45:]) / 9
    mean_current = -sum(boundary_current[45:]) / 9
    donor_dx = sum(dx[45:]) / 9
    assert abs(
        donor_dx + mp.mpf(3) * donor_mean / 2 + gamma * mean_current / 4
    ) < mp.mpf("1e-80")
    assert abs(sum(dt[45:]) / 9 - 3 * gamma * donor_mean / 2) < mp.mpf("1e-80")
    epsilon = mp.mpf("1e-35")
    derivative = (
        mp.im(
            _fields(
                mp,
                tuple(a + mp.j * epsilon * b for a, b in zip(x, dx)),
                tuple(a + mp.j * epsilon * b for a, b in zip(phase, dt)),
            )[2]
        )
        / epsilon
    )
    assert abs(derivative + loss) < mp.mpf("1e-55")


def test_three_harmonics_reconstruct_the_same_marked_contact_current(
    mp, observation, state
):
    _, offsets, beta = state
    moments = {}
    for harmonic, cosine, sine in observation.harmonic_moment_bounds:
        value = (
            sum(
                mp.exp(mp.j * harmonic * _mp(mp, offsets[b] - offsets[a]))
                for a, b in _CONTACT
            )
            / 9
        )
        moments[harmonic] = value
        _contains(mp, cosine, mp.re(value))
        _contains(mp, sine, mp.im(value))
    origin = _mp(mp, beta)
    fundamental = mp.im(mp.exp(mp.j * origin) * moments[1])
    actual = (
        fundamental
        + (
            mp.im(mp.exp(9 * mp.j * origin) * moments[9])
            - mp.im(mp.exp(18 * mp.j * origin) * moments[18]) / 2
        )
        / 2187
    )
    assert abs(actual - fundamental) > mp.mpf("1e-5")
    _contains(mp, observation.contact_mean_current_bounds, actual)
    _contains(mp, observation.contact_current_from_moments_bounds, actual)
    assert observation.current_reconstruction_residual_bounds.contains(0)


def test_sequential_word_preserves_forms_and_internal_gaps_but_counts_every_event(mp):
    forms = tuple(Q((3 * i) % 13 - 6, 1000) for i in range(45))
    offsets = tuple(Q(i // 9 - 2, 9) for i in range(45))
    initial_offsets = offsets
    for marked in range(5):
        for event, multiplier in enumerate((1, -2, 1)):
            previous = offsets
            returned_forms, offsets = word_owner._apply_sequential_observation_event(
                form=forms,
                phase_offsets=offsets,
                marked_child=marked,
                event_index=event,
            )
            assert returned_forms == forms
            assert sum(offsets) == sum(previous)
            for c in range(5):
                assert (
                    offsets[9 * c + 1] - offsets[9 * c]
                    == previous[9 * c + 1] - previous[9 * c]
                )
            work = sum(
                _potential(mp, _mp(mp, offsets[b] - offsets[a]))
                - _potential(mp, _mp(mp, previous[b] - previous[a]))
                for a, b in _OUTER
            )
            upper = 18 * (1 + mp.mpf(1) / 1458) * abs(multiplier) / 10
            assert abs(work) < upper
        # Algebraic restoration of commands is distinct from any actual
        # evolution; this test does not insert or discard a drift interval.
        assert offsets == initial_offsets


def test_conditional_bounds_rebuild_own_source_and_keep_fine_errors(bounds):
    assert bounds.recovered_norm_bound == _SYNTHETIC_Q
    assert bounds.host_bounds.recovered_child_norm_bound == _SYNTHETIC_Q
    assert not hasattr(bounds, "source_certified")
    assert bounds.source_form_upper_bounds == (
        bounds.host_bounds.primitive_node_form_radius_upper_bound
        + bounds.host_bounds.full_form_error_upper_bound,
        bounds.host_bounds.full_form_error_upper_bound,
    )
    assert bounds.boundary_phase_error_upper_bound == (
        bounds.host_bounds.full_phase_error_upper_bound
        + bounds.carried_phase_drift_upper_bound
        + _SYNTHETIC_Q
    )
    assert all("pi_over_two" not in item for item in bounds.geometry.scope)
    assert (
        "all_external_origins_in_circle_are_separately_supplied_preparations"
        in bounds.geometry.scope
    )


def test_word_contact_and_finite_remainder_retain_own_j_and_k(bounds):
    g = bounds.geometry.gamma_bounds.hi
    j, k, time, h = Q(1459, 1458), Q(245, 243), Q(1, 5000), Q(1, 1000)
    bo = Q(1, 100) + g * j * time
    b = bo + Q(7, 4) * g * j * h
    assert bounds.current_absolute_upper_bound == j
    assert bounds.current_lipschitz_upper_bound == k
    assert bounds.observation_form_upper_bound == bo
    assert bounds.joined_form_upper_bound == b
    assert bounds.word_edge_drift_upper_bound == 4 * g * bo * time
    assert bounds.contact_edge_drift_upper_bound == 7 * g * b * h
    assert bounds.per_block_absolute_event_work_upper_bounds == (
        Q(9, 5) * j,
        Q(18, 5) * j,
        Q(9, 5) * j,
    )
    assert bounds.total_absolute_event_work_upper_bound == 36 * j
    expected = Q(15, 16) * g * j * h**2 + Q(15, 4) * g**2 * k * b * h**2
    assert bounds.finite_response_remainder_upper_bound == expected
    assert expected > Q(15, 16) * g * h**2 + Q(15, 4) * g**2 * b * h**2
    assert bounds.total_increment_error_upper_bound == expected + Q(2, 10**10)


def test_own_potential_resources_do_not_spend_prior_dissipation(bounds):
    q, k = _SYNTHETIC_Q, bounds.current_lipschitz_upper_bound
    work = 45 * ((bounds.observation_form_upper_bound + q) ** 2 / 2 + 2 + Q(2, 19683))
    assert bounds.contact_work_upper_bound == work
    assert work > 90
    assert (
        bounds.donor_storage_upper_bound
        == bounds.host_bounds.internal_cycle_minimum_bounds.hi + 2 * (1 + k) * q**2
    )
    assert (
        bounds.host_preword_storage_upper_bounds[1]
        == 5 * bounds.host_bounds.internal_cycle_minimum_bounds.hi + 20 * (1 + k) * q**2
    )
    for before, joined in zip(
        bounds.host_preword_storage_upper_bounds,
        bounds.joined_initial_storage_upper_bounds,
    ):
        assert (
            joined
            == before
            + bounds.total_absolute_event_work_upper_bound
            + bounds.donor_storage_upper_bound
            + work
        )


def test_observed_capability_keeps_higher_harmonics_and_two_errors_per_increment(
    bounds,
):
    assert bounds.higher_harmonic_current_upper_bound == Q(5, 1458)
    factor = bounds.geometry.gamma_bounds * Q(1, 4000)
    error = bounds.total_increment_error_upper_bound
    tail = Q(5, 1458)
    for amplitude, capability in zip(
        bounds.fundamental_amplitude_bounds, bounds.observed_capability_bounds
    ):
        lower = max(Q(0), factor.lo * max(Q(0), amplitude.lo - tail) - error)
        upper = factor.hi * (amplitude.hi + tail) + error
        assert capability.contains(lower) and capability.contains(upper)
        assert capability.hi > factor.hi * amplitude.hi + error
    assert (
        bounds.outer_one_upper_margin
        == Q(1, 10**8) - bounds.observed_capability_bounds[0].hi
    )
    assert bounds.outer_zero_lower_margin == bounds.observed_capability_bounds[
        1
    ].lo - Q(3, 10**7)
    assert (
        bounds.paired_absolute_response_gap_lower_bound
        == bounds.observed_capability_bounds[1].lo
        - bounds.observed_capability_bounds[0].hi
    )


def test_identity_checks_keep_word_commands_and_all_marked_contact_cycles(bounds):
    word, contact = dict(bounds.word_identity_margins), dict(
        bounds.contact_identity_margins
    )
    assert len(contact) == 12 + 18
    for family in ("outer_one", "outer_zero"):
        assert (
            contact[f"contact_{family}_outer_acute_gap[0]"]
            - word[f"word_{family}_outer_acute_gap"]
            == Q(1, 10) - bounds.contact_edge_drift_upper_bound
        )
        assert all(
            contact[f"contact_{family}_outer_acute_gap[{j}]"]
            == contact[f"contact_{family}_outer_acute_gap[0]"]
            for j in range(9)
        )
    assert all(
        contact[f"contact_internal_positive_gap[{c}]"]
        == contact["contact_internal_positive_gap[0]"]
        for c in range(5)
    )
    assert (
        contact["contact_internal_positive_gap[5]"]
        != contact["contact_internal_positive_gap[0]"]
    )


def _fake_host(**kwargs):
    rx, rt = kwargs["form_error_bounds"], kwargs["phase_error_bounds"]
    return SimpleNamespace(
        deformation=Q(1),
        form_error_bounds=rx,
        phase_error_bounds=rt,
        admitted=True,
        capture_certified=True,
        outer_winding_retained=True,
        child_winding_retained=True,
        control_capture_certified=True,
        control_outer_zero_winding_retained=True,
        control_child_winding_retained=True,
        source_formation_time=Q(200),
        source_relaxation_duration=Q(2 * 10**13),
        elapsed_scaled_time=Q(200),
        recovered_child_norm_bound=Q(1, 10**70),
        relative_preparation_dimension=8 * sum(v > 0 for v in rx + rt),
    )


def _fake_donor(**kwargs):
    rx, rt = kwargs["form_error_bound"], kwargs["phase_error_bound"]
    return SimpleNamespace(
        deformation=Q(1),
        form_error_bound=rx,
        phase_error_bound=rt,
        recovery_certified=True,
        formation_time=Q(100),
        relaxation_duration=Q(10**13),
        source=SimpleNamespace(preparation_dimension=8 * (int(rx > 0) + int(rt > 0))),
        endpoint_form_norm_squared_upper_bound=Q(0),
        endpoint_phase_norm_squared_upper_bound=Q(0),
    )


@pytest.mark.parametrize(
    "which,field,value",
    (
        ("host", "deformation", Q(0)),
        ("host", "control_capture_certified", False),
        ("host", "outer_winding_retained", False),
        ("host", "source_formation_time", Q(100)),
        ("host", "source_relaxation_duration", Q(10**13)),
        ("host", "relative_preparation_dimension", 79),
        ("donor", "deformation", Q(0)),
        ("donor", "formation_time", Q(200)),
        ("donor", "relaxation_duration", Q(2 * 10**13)),
        ("donor", "endpoint_form_norm_squared_upper_bound", Q(1, 10**140)),
        ("donor", "endpoint_phase_norm_squared_upper_bound", None),
    ),
)
def test_synthetic_source_failures_suppress_all_history_and_contact_claims(
    monkeypatch, bounds, which, field, value
):
    calls = []

    def host(**kwargs):
        calls.append(("host", kwargs))
        result = _fake_host(**kwargs)
        if which == "host":
            setattr(result, field, value)
        return result

    def donor(**kwargs):
        calls.append(("donor", kwargs))
        result = _fake_donor(**kwargs)
        if which == "donor":
            setattr(result, field, value)
        return result

    monkeypatch.setattr(owner, "_assess_organized_alternative_outer_acquisition", host)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", donor)
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_contact_bounds", lambda **kwargs: bounds
    )
    report = owner._assess_organized_alternative_contact(
        form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6
    )
    assert calls == [
        (
            "host",
            dict(form_error_bounds=(_ERROR,) * 5, phase_error_bounds=(_ERROR,) * 5),
        ),
        (
            "donor",
            dict(deformation=Q(1), form_error_bound=_ERROR, phase_error_bound=_ERROR),
        ),
    ]
    assert not report.source_certified and not report.admitted
    assert not report.word_certified and not report.work_certified
    assert not report.identity_certified and not report.response_certified
    assert report.unresolved_conditions
    with pytest.raises(FrozenInstanceError):
        report.admitted = True


@pytest.mark.parametrize(
    "flag",
    (
        "word_identity_certified",
        "word_work_certified",
        "contact_work_certified",
        "contact_identity_certified",
        "response_certified",
    ),
)
def test_synthetic_failed_resources_never_inherit_a_source_admission(
    monkeypatch, bounds, flag
):
    monkeypatch.setattr(
        owner, "_assess_organized_alternative_outer_acquisition", _fake_host
    )
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", _fake_donor)
    changed = replace(bounds, **{flag: False})
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_contact_bounds", lambda **kwargs: changed
    )
    report = owner._assess_organized_alternative_contact(
        form_error_bounds=(0,) * 6, phase_error_bounds=(0,) * 6
    )
    assert report.source_certified
    assert not report.admitted


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "bad",
    (
        True,
        "0",
        None,
        complex(0),
        float("nan"),
        float("inf"),
        -Q(1, 10**1000),
        _ERROR + Q(1, 10**1000),
    ),
)
def test_all_twelve_allowances_are_admitted_before_any_source(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid twelfth primitive reached source or geometry")

    monkeypatch.setattr(
        owner, "_assess_organized_alternative_outer_acquisition", forbidden
    )
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_contact_bounds", forbidden
    )
    kwargs = dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
    kwargs[field] = (_ERROR,) * 5 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_alternative_contact(**kwargs)


@pytest.mark.parametrize(
    "bad", ((), (0,) * 5, (0,) * 7, "000000", {0}, {i: 0 for i in range(6)})
)
def test_source_shapes_are_not_coerced(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_alternative_contact(
            form_error_bounds=bad, phase_error_bounds=(0,) * 6
        )


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "host",
        "donor",
        "deformation",
        "horizon",
        "relative_phase_origin",
        "word",
        "readout_error",
        "budget",
        "recovered_radius",
    ),
)
def test_cached_reports_and_retuned_protocol_are_not_arguments(extra):
    with pytest.raises(TypeError):
        owner._assess_organized_alternative_contact(
            form_error_bounds=(0,) * 6, phase_error_bounds=(0,) * 6, **{extra: 0}
        )


@pytest.mark.parametrize(
    "bad", (True, "0", None, complex(0), float("nan"), float("inf"), -Q(1, 10**1000))
)
def test_conditional_radius_admission_precedes_any_source_bound(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_derive_organized_alternative_outer_bounds",
        lambda **kwargs: pytest.fail("bad radius reached host arithmetic"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._derive_organized_alternative_contact_bounds(recovered_radius=bad)


@pytest.mark.parametrize("field", ("relative_phase_origin", "form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), "0"))
def test_static_observer_admits_every_coordinate_before_geometry(
    field, bad, monkeypatch
):
    monkeypatch.setattr(
        owner,
        "_geometry",
        lambda *args, **kwargs: pytest.fail("invalid state reached geometry"),
    )
    kwargs = dict(
        relative_phase_origin=Q(0), form=(Q(0),) * 54, phase_offsets=(Q(0),) * 54
    )
    kwargs[field] = bad if field == "relative_phase_origin" else (Q(0),) * 53 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_alternative_contact(**kwargs)
