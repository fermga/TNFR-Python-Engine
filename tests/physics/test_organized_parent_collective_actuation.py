"""Static controls for eight-observation, four-channel parent actuation."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_collective_actuation as owner
from tnfr.physics import _organized_parent_constructive_actuation as complete

_ERROR = Q(1, 10**10)
_MEANS = (Q(1, 70000), -Q(1, 30000), Q(1, 60000), Q(1, 90000))
_PHASES = (Q(1, 17), -Q(1, 19), Q(1, 23), -Q(1, 29))
_DELTA = (-Q(1, 4), Q(1, 4), Q(0), Q(0))


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _laplacian(values):
    return tuple(2 * values[i] - values[i ^ 1] - values[i ^ 2] for i in range(4))


def _broadcast(values):
    return tuple(value for value in values for _ in range(9))


def _means(values):
    return tuple(sum(values[start : start + 9], Q(0)) / 9 for start in range(0, 36, 9))


def _input(**changes):
    return owner._evaluate_collective_parent_input(
        **(
            dict(
                child_form_means=_MEANS,
                child_phase_offset_means=_PHASES,
                path_fraction=Q(1, 3),
            )
            | changes
        )
    )


def _assess(**changes):
    return owner._assess_organized_parent_collective_actuation(
        **(
            dict(form_error_bounds=(_ERROR,) * 4, phase_error_bounds=(_ERROR,) * 4)
            | changes
        )
    )


@pytest.fixture(scope="module")
def realization():
    return _assess()


@pytest.mark.parametrize("fraction", (Q(0), Q(1, 6), Q(1, 2), Q(5, 6), Q(1)))
def test_four_channel_input_has_exact_square_law_and_original_clock(fraction, mp):
    report = _input(path_fraction=fraction)
    reference = report.reference_path
    assert reference.form == _broadcast(_MEANS)
    assert reference.phase_offsets == _broadcast(_PHASES)
    assert report.child_phase_displacement == _DELTA
    assert report.predictor_initial_form_means == _MEANS
    assert report.predictor_initial_phase_offset_means == tuple(
        a + b for a, b in zip(_PHASES, _DELTA)
    )
    gamma, h, e = 1 / (1023 * mp.pi), mp.mpf(4), mp.mpf(1023) / 1024
    v = _mp(mp, fraction)
    x0 = tuple(_mp(mp, value) for value in _MEANS)
    t0 = tuple(_mp(mp, value) for value in _PHASES)
    displacement = tuple(_mp(mp, value) for value in _DELTA)
    inverse = (-mp.mpf(3) / 32, mp.mpf(3) / 32, -mp.mpf(1) / 32, mp.mpf(1) / 32)
    mean = sum(x0) / 4
    b = tuple(h * value / (2 * gamma) for value in _laplacian(x0))
    p, q = 3 * v**2 - 2 * v**3, v - 3 * v**2 + 2 * v**3
    dp, dq = 6 * v - 6 * v**2, 1 - 6 * v + 6 * v**2
    ddp, ddq = 6 - 12 * v, -6 + 12 * v
    x = tuple(
        mean + 2 * gamma * inv * dp / h + (initial - mean) * dq
        for inv, initial in zip(inverse, x0)
    )
    theta = tuple(
        initial + delta * p + slope * q
        for initial, delta, slope in zip(t0, displacement, b)
    )
    dx_tau = tuple(
        gamma**2 / (2 * h) * (2 * gamma * inv * ddp / h + (initial - mean) * ddq)
        for inv, initial in zip(inverse, x0)
    )
    dt_tau = tuple(
        gamma**2 / (2 * h) * (delta * dp + slope * dq)
        for delta, slope in zip(displacement, b)
    )
    lap = _laplacian(x)
    currents = tuple(
        mp.sin(theta[i ^ 1] - theta[i]) + mp.sin(theta[i ^ 2] - theta[i])
        for i in range(4)
    )
    for i in range(4):
        forcing = dx_tau[i] + (lap[i] - gamma * currents[i]) / 4
        _contains(mp, report.child_phase_current_bounds[i], currents[i])
        _contains(mp, report.child_form_input_tau_bounds[i], forcing)
        _contains(mp, report.child_form_input_slow_bounds[i], 2 * forcing / gamma**2)
        _contains(mp, report.child_form_input_original_bounds[i], e * forcing)
        _contains(mp, reference.form_bounds[9 * i], x[i])
        _contains(mp, reference.phase_offset_bounds[9 * i], theta[i])
        assert abs(dt_tau[i] - gamma * lap[i] / 4) < mp.mpf("1e-165")
        # The complete degree is four; the quotient's two neighbors do not
        # license changing either denominator to two.
        if fraction == Q(1, 6):
            assert abs(dt_tau[i] - gamma * lap[i] / 2) > mp.mpf("1e-12")
        for clock in ("tau", "slow", "original"):
            child = getattr(report, f"child_form_input_{clock}_bounds")[i]
            assert (
                getattr(report, f"form_input_{clock}_bounds")[9 * i : 9 * i + 9]
                == (child,) * 9
            )


def test_hidden_fine_states_share_one_input_without_becoming_the_reference(mp):
    initial = _input(path_fraction=Q(0))
    uniform_x, uniform_theta = _broadcast(_MEANS), _broadcast(_PHASES)
    fine_x = tuple(
        value + Q((i // 9 + 1) * (i % 9 - 4), 10**6)
        for i, value in enumerate(uniform_x)
    )
    fine_theta = tuple(
        value + Q((i // 9 + 1) * (3 * (i % 9) ** 2 - 68), 3 * 10**7)
        for i, value in enumerate(uniform_theta)
    )
    assert _means(fine_x) == _MEANS and _means(fine_theta) == _PHASES
    assert fine_x != uniform_x and fine_theta != uniform_theta
    same_input = _input(
        child_form_means=_means(fine_x),
        child_phase_offset_means=_means(fine_theta),
        path_fraction=Q(0),
    )
    assert same_input.form_input_tau_bounds == initial.form_input_tau_bounds
    # The older all-node recipe would use the unresolved fine coordinates
    # and is observably different. It cannot be substituted for this input.
    fine_recipe = complete._evaluate_parent_actuation_path(
        form=fine_x, phase_offsets=fine_theta, path_fraction=Q(0)
    )
    assert any(
        a.hi < b.lo or b.hi < a.lo
        for a, b in zip(
            fine_recipe.form_input_tau_bounds, initial.form_input_tau_bounds
        )
    )
    gamma = 1 / (1023 * mp.pi)
    fine_lap = []
    for i in range(36):
        child, j = divmod(i, 9)
        neighbors = (
            9 * child + (j - 1) % 9,
            9 * child + (j + 1) % 9,
            9 * (child ^ 1) + j,
            9 * (child ^ 2) + j,
        )
        fine_lap.append(sum(fine_x[i] - fine_x[k] for k in neighbors))
    # Same input does not erase the actual fine phase velocity.
    assert any(
        abs(
            gamma * _mp(mp, value) / 4 - gamma * _mp(mp, _laplacian(_MEANS)[i // 9]) / 4
        )
        > mp.mpf("1e-12")
        for i, value in enumerate(fine_lap)
    )


def test_same_input_error_endpoint_prediction_and_actual_work_bounds(realization):
    report, source = realization, realization.acquisition
    r, z, gamma = (
        source.full_form_error_upper_bound,
        source.full_phase_error_upper_bound,
        source.geometry.gamma_bounds.hi,
    )
    for horizon, prefix in ((4, "forced"), (5, "readout"), (8, "full_interval")):
        phase = (z + gamma * r) * 3 ** (4 * horizon)
        form = r + 24 * gamma * phase
        assert getattr(report, f"{prefix}_phase_error_upper_bound") == phase
        assert getattr(report, f"{prefix}_form_error_upper_bound") == form
    assert (
        report.endpoint_form_error_upper_bound
        == report.forced_form_error_upper_bound + r
    )
    assert (
        report.endpoint_phase_error_upper_bound
        == report.forced_phase_error_upper_bound + z
    )
    assert (
        report.child_form_mean_prediction_error_upper_bound
        == report.readout_form_error_upper_bound / 3
    )
    assert (
        report.child_phase_mean_prediction_error_upper_bound
        == report.readout_phase_error_upper_bound / 3
    )
    # Each nine-node mean has Euclidean operator norm 1/3. The forecast
    # is all eight coordinates, not the previous outer-gap-only observation.
    assert sum(Q(1, 9) ** 2 for _ in range(9)) == Q(1, 9)
    rel_z = (
        report.endpoint_phase_error_upper_bound
        + gamma * report.endpoint_form_error_upper_bound
    ) * 3**16
    assert report.release_phase_error_upper_bound == rel_z
    assert (
        report.release_form_error_upper_bound
        == report.endpoint_form_error_upper_bound + 24 * gamma * rel_z
    )
    assert (
        report.parent_gap_at_recovery_upper_bound
        == report.restoring_response.parent_gap_at_recovery_upper_bound + 2 * rel_z
    )
    assert (
        report.input_work_error_upper_bound
        == 64
        * report.forced_form_error_upper_bound
        * report.reference_input_norm_upper_bound
        / source.geometry.gamma_bounds.lo**2
    )
    assert report.input_work_error_upper_bound > 0
    assert (
        report.input_work_upper_bound
        == report.reference_input_work_upper_bound + report.input_work_error_upper_bound
    )
    assert report.input_work_upper_bound < 3 and report.work_budget_margin > 0
    assert report.endpoint_form_error_upper_bound <= report.endpoint_error_budget
    assert report.endpoint_phase_error_upper_bound <= report.endpoint_error_budget
    assert (
        report.child_form_mean_prediction_error_upper_bound
        <= report.prediction_error_budget
    )
    assert (
        report.child_phase_mean_prediction_error_upper_bound
        <= report.prediction_error_budget
    )
    assert report.parent_gap_at_recovery_upper_bound < Q(1, 4)
    assert report.child_positive_gap_margin > 0 and report.child_acute_gap_margin > 0
    assert report.post_actuation_disruption_margin > 0


def test_original_source_and_finite_scope_are_retained(realization):
    report = realization
    assert report.source_certified and report.conditional_realization_certified
    assert (
        report.endpoint_certified
        and report.work_certified
        and report.child_identities_retained
    )
    assert (
        report.disruption_certified
        and report.prediction_certified
        and report.recovery_certified
    )
    assert report.realization_certified and report.release_certified and report.admitted
    assert not report.unresolved_conditions
    assert report.source_slow_time == 4 and report.slow_duration == 4
    assert (
        report.release_slow_time == 8
        and report.readout_slow_time == 9
        and report.recovery_slow_time == 12
    )
    assert report.endpoint_error_budget == Q(1, 10**7)
    assert report.prediction_error_budget == Q(1, 10**6)
    assert report.acquisition.relative_preparation_dimension == 64
    for child in report.acquisition.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR
    with pytest.raises(FrozenInstanceError):
        report.prediction_certified = False


@pytest.mark.parametrize("field", ("child_form_means", "child_phase_offset_means"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_every_observation_is_admitted_before_broadcast_recipe(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed observation reached broadcast recipe")

    monkeypatch.setattr(owner, "_evaluate_parent_actuation_path", forbidden)
    values = list(_MEANS)
    values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _input(**{field: values})


@pytest.mark.parametrize("field", ("child_form_means", "child_phase_offset_means"))
@pytest.mark.parametrize("bad", ((), (Q(0),) * 3, (Q(0),) * 5, "0000", {Q(0)}))
def test_observation_shapes_reject_before_broadcast_recipe(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed observations reached broadcast recipe")

    monkeypatch.setattr(owner, "_evaluate_parent_actuation_path", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _input(**{field: bad})


@pytest.mark.parametrize(
    "bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100), 1 + Q(1, 10**100))
)
def test_path_clock_rejects_before_broadcast_recipe(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed clock reached broadcast recipe")

    monkeypatch.setattr(owner, "_evaluate_parent_actuation_path", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _input(path_fraction=bad)


def test_subgrid_observations_remain_nonzero_exact_primitives():
    tiny = Q(1, 10**1000)
    report = _input(
        child_form_means=(tiny, -tiny, Q(0), Q(0)),
        child_phase_offset_means=(Q(0), Q(0), tiny, -tiny),
    )
    assert report.child_form_means[0] == tiny != 0
    assert report.child_phase_offset_means[2] == tiny != 0
    assert report.predictor_initial_form_means == (tiny, -tiny, Q(0), Q(0))
    assert report.predictor_initial_phase_offset_means[2:] == (tiny, -tiny)


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100)))
def test_source_primitives_reject_before_reconstruction(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed original source reached reconstruction")

    monkeypatch.setattr(
        owner, "_assess_organized_parent_constructive_actuation", forbidden
    )
    values = [_ERROR] * 4
    values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize(
    "dependency", ("sources_certified", "entry_certified", "child_identities_retained")
)
def test_missing_source_cannot_be_replaced_by_prior_verdicts(
    dependency, realization, monkeypatch
):
    source = replace(realization.acquisition, **{dependency: False})
    construction = replace(realization.construction, acquisition=source)
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_constructive_actuation",
        lambda **kwargs: construction,
    )
    report = _assess()
    assert report.conditional_realization_certified
    assert (
        not report.source_certified
        and not report.realization_certified
        and not report.admitted
    )
    assert report.status == "unavailable" and report.unresolved_conditions


def test_old_path_and_forecast_flags_do_not_certify_or_block_new_result(
    realization, monkeypatch
):
    old = replace(
        realization.restoring_response,
        prediction_certified=False,
        contrast_certified=False,
        child_identities_retained=False,
        admitted=False,
        status="unavailable",
    )
    construction = replace(
        realization.construction,
        restoring_response=old,
        source_certified=False,
        realization_certified=False,
        release_certified=False,
        admitted=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_constructive_actuation",
        lambda **kwargs: construction,
    )
    report = _assess()
    assert (
        report.source_certified
        and report.realization_certified
        and report.release_certified
    )
    assert report.prediction_certified and report.child_identities_retained


def test_old_recovery_premise_controls_only_the_transferred_recovery(
    realization, monkeypatch
):
    old = replace(realization.restoring_response, recovery_certified=False)
    construction = replace(realization.construction, restoring_response=old)
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_constructive_actuation",
        lambda **kwargs: construction,
    )
    report = _assess()
    assert report.source_certified and report.realization_certified
    assert report.prediction_certified and report.child_identities_retained
    assert (
        not report.recovery_certified
        and not report.release_certified
        and not report.admitted
    )


@pytest.mark.parametrize(
    "extra",
    (
        "form",
        "phase_offsets",
        "fine_state",
        "release_observations",
        "Gamma",
        "duration",
        "trajectory",
    ),
)
def test_input_cannot_consume_hidden_or_later_observations(extra):
    with pytest.raises(TypeError):
        _input(**{extra: 0})


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "construction",
        "release_observations",
        "work_budget",
        "duration",
        "target",
        "trajectory",
    ),
)
def test_protocol_cannot_be_replaced_by_cached_reports_or_alternate_targets(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_shared_input_power_error_is_signed_and_may_have_nonzero_child_means():
    # This static identity holds for every common input, not just the cubic.
    # A later residual need not retain its initially zero child means.
    forcing = _broadcast((Q(1, 9), -Q(1, 11), Q(1, 13), -Q(1, 17)))
    initial = _broadcast(_MEANS)

    def fine_laplacian(values):
        result = []
        for i in range(36):
            child, j = divmod(i, 9)
            neighbors = (
                9 * child + (j - 1) % 9,
                9 * child + (j + 1) % 9,
                9 * (child ^ 1) + j,
                9 * (child ^ 2) + j,
            )
            result.append(sum(values[i] - values[k] for k in neighbors))
        return tuple(result)

    fine = tuple(Q((i // 9 + 1) * (i % 9 - 4), 10**6) for i in range(36))
    collective = _broadcast((Q(1, 1000), -Q(1, 1000), Q(1, 3000), -Q(1, 3000)))
    residual = tuple(a + b for a, b in zip(fine, collective))
    assert sum(residual) == 0 and _means(residual) != (Q(0),) * 4
    assert sum(a * b for a, b in zip(fine_laplacian(fine), forcing)) == 0
    reference_power = sum(a * b for a, b in zip(fine_laplacian(initial), forcing))
    for sign in (-1, 1):
        actual = tuple(a + sign * b for a, b in zip(initial, residual))
        actual_power = sum(a * b for a, b in zip(fine_laplacian(actual), forcing))
        correction = sign * sum(
            a * b for a, b in zip(fine_laplacian(residual), forcing)
        )
        assert actual_power - reference_power == correction != 0
        assert correction**2 <= 64 * sum(value**2 for value in residual) * sum(
            value**2 for value in forcing
        )
        assert (correction > 0) is (sign > 0)


@pytest.mark.parametrize(
    "margin,allowed", ((-Q(1, 10**100), False), (Q(0), True), (Q(1, 10**100), True))
)
@pytest.mark.parametrize(
    "constant,flag,bounds",
    (
        (
            "_ENDPOINT_BUDGET",
            "endpoint_certified",
            ("endpoint_form_error_upper_bound", "endpoint_phase_error_upper_bound"),
        ),
        (
            "_PREDICTION_BUDGET",
            "prediction_certified",
            (
                "child_form_mean_prediction_error_upper_bound",
                "child_phase_mean_prediction_error_upper_bound",
            ),
        ),
        ("_WORK_BUDGET", "work_certified", ("input_work_upper_bound",)),
    ),
)
def test_endpoint_prediction_and_work_budget_equalities_are_admitted(
    constant, flag, bounds, margin, allowed, realization, monkeypatch
):
    # Arithmetic boundary control only, not a replacement scientific protocol.
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_constructive_actuation",
        lambda **kwargs: realization.construction,
    )
    monkeypatch.setattr(
        owner, constant, max(getattr(realization, name) for name in bounds) + margin
    )
    report = _assess()
    assert getattr(report, flag) is allowed
    if not allowed:
        assert report.status == "unavailable" and report.unresolved_conditions


@pytest.mark.parametrize(
    "margin,allowed", ((-Q(1, 10**100), False), (Q(0), False), (Q(1, 10**100), True))
)
@pytest.mark.parametrize("which", ("disruption", "recovery"))
def test_parent_identity_boundaries_remain_strict(
    which, margin, allowed, realization, monkeypatch
):
    # Substituted margins exercise the boundary, without claiming a new source.
    if which == "disruption":
        old = replace(
            realization.restoring_response,
            post_event_disruption_margin=2
            * realization.endpoint_phase_error_upper_bound
            + margin,
        )
    else:
        old = replace(
            realization.restoring_response,
            parent_gap_at_recovery_upper_bound=realization.acquisition.parent_identity_gap
            - 2 * realization.release_phase_error_upper_bound
            - margin,
        )
    construction = replace(realization.construction, restoring_response=old)
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_constructive_actuation",
        lambda **kwargs: construction,
    )
    report = _assess()
    assert getattr(report, f"{which}_certified") is allowed
    if not allowed:
        assert not report.admitted and report.status == "unavailable"
