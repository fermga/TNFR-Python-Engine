"""Primitive admission and coupled bounds without new response evaluations.

The baseline uses the previously frozen preparation and reading allowances.
Damaged fresh dependencies exercise wiring; they are not alternative sources.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_joint_resources as owner

_SOURCE = Q(1, 10**10)
_RADIUS = Q(1, 10**70)
_DWELL = Q(2 * 10**13)
_SEQUENTIAL_NOISE = Q(1, 10**15)
_CONTACT_NOISE = Q(1, 10**10)


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(_SOURCE,) * 6,
        phase_error_bounds=(_SOURCE,) * 6,
        quarter_dwell=_DWELL,
        recovered_radius=_RADIUS,
        decay_power=512,
        sequential_readout_error=_SEQUENTIAL_NOISE,
        contact_readout_error=_CONTACT_NOISE,
    )
    return owner._assess_organized_joint_resources(**(arguments | changes))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _dependencies(monkeypatch, report, *, recovery=None, outer=None, observation=None):
    """Inject named damaged dependencies only for consumer-wiring checks."""
    monkeypatch.setattr(
        owner,
        "_assess_sine_recovery_resources",
        lambda **kwargs: report.recoveries[0] if recovery is None else recovery,
    )
    monkeypatch.setattr(
        owner,
        "_derive_organized_outer_resource_bounds",
        lambda **kwargs: report.outer_resources if outer is None else outer,
    )
    monkeypatch.setattr(
        owner,
        "_assess_organized_sequential_observation_resources",
        lambda **kwargs: report.observation if observation is None else observation,
    )


def test_original_parameter_domain_rebuilds_all_stages_and_ledgers(report):
    assert report.admitted and report.source_certified
    assert report.unresolved_conditions == ()
    assert report.relative_preparation_dimension == 6 * 2 * (9 - 1)
    assert all(value > 0 for _, value in report.sufficient_margins)
    assert all(source.recovery_certified for source in report.recoveries)
    assert report.outer_resources.conditional_capture_certified
    assert report.observation.uniform_identification_certified
    assert report.capability_bounds[0].hi < Q(1, 10**8)
    assert report.capability_bounds[1].lo > Q(3, 10**7)
    assert report.fixed_origin_contrast_bounds.lo > Q(1, 10**7)
    primitive_cost = sum(
        (source.source.initial_storage_upper_bounds[0] for source in report.recoveries),
        Q(0),
    )
    paid_events = (
        report.outer_resources.attachment_work_upper_bound
        + report.observation.total_absolute_event_work_upper_bound
        + report.contact_work_upper_bound
    )
    assert report.primitive_storage_upper_bound == primitive_cost
    assert report.historical_resource_upper_bound == primitive_cost + paid_events
    assert report.historical_resource_upper_bound > max(
        report.joined_storage_upper_bounds
    )
    assert report.contact_work_upper_bound < 100
    assert max(report.joined_storage_upper_bounds) < 200


def test_work_and_time_lower_bounds_have_independent_analytic_values(report, mp):
    origins = tuple(Q(4092 * (index - 2)) for index in range(5))
    direct_work = 9 * sum(
        ((origins[(index + 1) % 5] - origins[index]) ** 2 / 2 for index in range(5)),
        Q(0),
    )
    assert report.fixed_outer_work_lower_bound == direct_work
    assert report.outer_resources.attachment_work_upper_bound >= direct_work
    gamma = 1 / (1023 * mp.pi)
    q = _mp(mp, _RADIUS)
    expected_work_time = (
        45 * (1 - mp.cos(2 * mp.pi / 5))
        + 18 * (mp.pi - 2 * q) ** 2 / (200 * gamma**2)
        - 20 * q**2
    )
    _contains(mp, report.winding_work_time_lower_bound, expected_work_time)
    compression = _mp(mp, report.phase_compression_time_lower_bound)
    exact_time_lower_bound = mp.log(2 - q / _mp(mp, _SOURCE)) / gamma**2
    assert 0 < compression < exact_time_lower_bound
    assert compression < _mp(mp, _DWELL + 200)


def test_necessary_contact_noise_and_signal_account_for_both_readings(report, mp):
    noise = _CONTACT_NOISE
    assert report.necessary_contact_noise_margin == Q(1, 10**8) - 2 * noise
    gamma = 1 / (1023 * mp.pi)
    exact_margin = 5 * gamma / 4000 - _mp(mp, Q(3, 10**7) + 2 * noise)
    assert (
        0 < exact_margin <= _mp(mp, report.necessary_contact_signal_margin_upper_bound)
    )
    # The damping-kernel limit is stricter than its linear-time upper bound.
    damped = 5 * gamma * (1 - mp.exp(-mp.mpf(3) / 2000)) / 6
    assert damped < 5 * gamma / 4000


def test_equal_error_budgets_reuse_only_fresh_bounds_within_one_assessment(
    report, monkeypatch
):
    _dependencies(monkeypatch, report)
    calls = []

    def recovery(**kwargs):
        calls.append(kwargs)
        return report.recoveries[0]

    monkeypatch.setattr(owner, "_assess_sine_recovery_resources", recovery)
    first, second = _assess(), _assess()
    assert first.admitted and second.admitted
    assert len(calls) == 2
    assert all(
        value
        == dict(
            form_error_bound=_SOURCE,
            phase_error_bound=_SOURCE,
            quarter_dwell=_DWELL,
            endpoint_radius=_RADIUS,
            decay_power=512,
        )
        for value in calls
    )
    assert len(first.recoveries) == 6


@pytest.mark.parametrize(
    "target,field,label",
    (
        ("recovery", "recovery_certified", "recovered_sources"),
        ("outer", "conditional_capture_certified", "outer_full_fine_capture"),
        (
            "outer",
            "control_conditional_capture_certified",
            "outer_control_capture_domain",
        ),
        ("outer", "attachment_budget_certified", "outer_attachment_budget"),
        ("observation", "admitted", "sequential_observation"),
    ),
)
def test_every_fresh_stage_admission_remains_required(
    report, monkeypatch, target, field, label
):
    dependency = {
        "recovery": report.recoveries[0],
        "outer": report.outer_resources,
        "observation": report.observation,
    }[target]
    _dependencies(
        monkeypatch, report, **{target: replace(dependency, **{field: False})}
    )
    failed = _assess()
    assert not failed.admitted
    assert label in failed.unresolved_conditions
    assert failed.source_certified is (target != "recovery")
    assert all(value > 0 for _, value in failed.sufficient_margins)


def test_passing_flags_do_not_replace_consumed_numeric_margins(report, monkeypatch):
    outer = replace(
        report.outer_resources,
        full_phase_error_upper_bound=Q(1, 1000),
        capture_margin=Q(0),
        attachment_work_margin=Q(-1),
    )
    _dependencies(monkeypatch, report, outer=outer)
    failed = _assess()
    assert failed.source_certified
    assert not failed.admitted
    assert {
        "reprojected_fine_phase",
        "outer_capture",
        "outer_attachment_work",
    }.issubset(failed.unresolved_conditions)


def test_missing_recovery_and_readout_resolution_remain_distinct_failures(
    report, monkeypatch
):
    recovery = replace(report.recoveries[0], recovery_certified=False)
    observation = replace(
        report.observation,
        admitted=False,
        sufficient_margins=report.observation.sufficient_margins
        + (("test_resolution", Q(0)),),
    )
    _dependencies(monkeypatch, report, recovery=recovery, observation=observation)
    failed = _assess()
    assert not failed.source_certified and not failed.admitted
    assert {"recovered_sources", "sequential_observation", "test_resolution"}.issubset(
        failed.unresolved_conditions
    )


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -_SOURCE, Q(1))
)
def test_each_original_source_scalar_is_admitted_before_dependencies(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid source allowance reached a proof dependency")

    monkeypatch.setattr(owner, "_assess_sine_recovery_resources", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_outer_resource_bounds", forbidden)
    monkeypatch.setattr(
        owner, "_assess_organized_sequential_observation_resources", forbidden
    )
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: (_SOURCE,) * 5 + (bad,)})


@pytest.mark.parametrize(
    "field",
    (
        "quarter_dwell",
        "recovered_radius",
        "sequential_readout_error",
        "contact_readout_error",
    ),
)
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1))
)
def test_resource_scalars_are_admitted_before_dependencies(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid resource allowance reached a proof dependency")

    monkeypatch.setattr(owner, "_assess_sine_recovery_resources", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("bad", (True, Q(512), 512.0, 0, -1, 4097, "512"))
def test_decay_power_requires_the_declared_discrete_budget_before_dependencies(
    bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid power reached a proof dependency")

    monkeypatch.setattr(owner, "_assess_sine_recovery_resources", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(decay_power=bad)


@pytest.mark.parametrize("radius", (Q(0), Q(1, 12) + Q(1, 10**100)))
def test_radius_domain_precedes_source_execution(radius, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("unsupported radius reached a proof dependency")

    monkeypatch.setattr(owner, "_assess_sine_recovery_resources", forbidden)
    with pytest.raises(ValueError):
        _assess(recovered_radius=radius)


def test_reports_keep_immutable_normalized_primitive_budgets(report, monkeypatch):
    _dependencies(monkeypatch, report)
    values = [_SOURCE] * 6
    fresh = _assess(form_error_bounds=values)
    values[-1] = Q(999)
    assert fresh.form_error_bounds == (_SOURCE,) * 6
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "recoveries",
        "outer_resources",
        "observation",
        "trajectory",
        "horizon",
        "gamma",
    ),
)
def test_cached_reports_or_changed_law_cannot_replace_primitives(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: None})
