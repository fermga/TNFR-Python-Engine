"""Candidate admission and wiring controls; no declared trajectory is run."""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.physics import _organized_observed_forecast as owner


@pytest.mark.parametrize("bad", (True, float("nan"), float("inf"), Q(101, 10000), 0.01))
def test_candidate_prior_rejects_before_any_integration(monkeypatch, bad):
    monkeypatch.setattr(
        owner, "_forecast_leg", lambda *a, **k: pytest.fail("integration")
    )
    with pytest.raises((ValueError, TypeError)):
        owner.validate_observed_candidate(
            owner._sample_times(), (0,) * 150, (bad,) + (0,) * 9, rebuilt_gain=4
        )


def test_candidate_keeps_common_form_and_initial_only_phase_gauge():
    candidate = (Q(1, 100),) * 5 + (0, 1, 2, 3, 4)
    assert owner._admit_candidate(candidate) == candidate
    with pytest.raises(ValueError, match="gauge"):
        owner._admit_candidate((0,) * 5 + (1,) * 5)


@pytest.mark.parametrize(
    "kind", ("missing", "shifted", "reordered", "boolean", "nonfinite")
)
def test_joint_record_admission_precedes_integration(monkeypatch, kind):
    monkeypatch.setattr(
        owner, "_forecast_leg", lambda *a, **k: pytest.fail("integration")
    )
    times, values = owner._sample_times(), (0,) * 150
    if kind == "missing":
        values = values[:-1]
    elif kind == "shifted":
        times = (times[0] + Q(1, 10**20),) + times[1:]
    elif kind == "reordered":
        times = tuple(reversed(times))
    else:
        values = (True if kind == "boolean" else float("nan"),) + values[1:]
    with pytest.raises((ValueError, TypeError)):
        owner.validate_observed_candidate(times, values, (0,) * 10, rebuilt_gain=4)


def _solver_double(candidate, radius):
    times = owner._sample_times()
    points = times[1:] + (Q(216),)
    previous = times[0]
    steps = []
    for time in points:
        # This is only a wire double; it asserts no sine trajectory or target.
        steps.append(
            SimpleNamespace(
                time=previous,
                duration=time - previous,
                endpoint_center=candidate,
                endpoint_radius=radius,
            )
        )
        previous = time
    return SimpleNamespace(
        start=times[0],
        initial_center=candidate,
        initial_radius=Q(0),
        steps=tuple(steps),
        endpoint_center=candidate,
        endpoint_radius=radius,
        admitted=True,
        reason=None,
    )


def test_one_carried_candidate_and_error_budget_wiring(monkeypatch):
    candidate = (Q(1, 256),) * 5 + (0,) * 5
    calls = []

    def integrate(name, stages, initial, radius, **kwargs):
        calls.append((name, stages, initial, radius, kwargs))
        return _solver_double(initial, Q(1, 10**20))

    monkeypatch.setattr(owner, "_forecast_leg", integrate)
    report = owner.validate_observed_candidate(
        owner._sample_times(), candidate[:5] * 30, candidate, rebuilt_gain=4
    )
    assert report.admitted and report.reason is None
    assert len(calls) == 1 and calls[0][2:] == (candidate, 0, {"omit_word": False})
    assert calls[0][1][0].start == 200 and calls[0][1][-1].name == "released"
    assert report.error.maximum_true_output_radius < Q(1, 10**13)
    assert all(
        box.contains(value)
        for box, value in zip(report.predicted_true_form_bounds, candidate[:5])
    )


def test_bad_fit_abstains_without_retry_or_feasibility_claim(monkeypatch):
    calls = []

    def integrate(*args, **kwargs):
        calls.append(1)
        return _solver_double(args[2], Q(0))

    monkeypatch.setattr(owner, "_forecast_leg", integrate)
    report = owner.validate_observed_candidate(
        owner._sample_times(), (1,) * 150, (0,) * 10, rebuilt_gain=4
    )
    assert not report.admitted
    assert report.reason == "candidate_forecast_radius_exceeds_declared_policy"
    assert report.predicted_true_form_bounds is None and len(calls) == 1


def test_failed_enclosure_has_no_forecast(monkeypatch):
    monkeypatch.setattr(
        owner,
        "_forecast_leg",
        lambda *a, **k: SimpleNamespace(
            admitted=False, reason="synthetic_enclosure_failure"
        ),
    )
    report = owner.validate_observed_candidate(
        owner._sample_times(), (0,) * 150, (0,) * 10, rebuilt_gain=4
    )
    assert not report.admitted and report.reason == "synthetic_enclosure_failure"
    assert report.error is None and report.predicted_reading_bounds is None


def test_assessment_passes_only_readings_and_design_to_proposal(monkeypatch, tmp_path):
    from tnfr.physics import _organized_forecast_proposal as proposer
    from tnfr.research import _organized_forecast_inputs as inputs

    times = owner._sample_times()
    values = tuple((Q(index, 100000),) * 5 for index in range(30))
    record = SimpleNamespace(times=times, values=values)
    matrix = object()
    gain = SimpleNamespace(
        observation_matrix=matrix,
        carrier_future_gain=Q(4),
        source_error=Q(1, 10**18),
        provenance=object(),
    )
    monkeypatch.setattr(inputs, "load_archived_prediction_gain", lambda root: gain)
    monkeypatch.setattr(inputs, "extract_archived_form_record", lambda root: record)
    calls = []

    def propose(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(initial_candidate=None, reason="synthetic_fit_failure")

    monkeypatch.setattr(proposer, "propose_candidate", propose)
    monkeypatch.setattr(
        owner, "validate_observed_candidate", lambda *a, **k: pytest.fail("validation")
    )
    result = owner._assess_organized_observed_forecast(tmp_path)
    assert calls == [((times, tuple(x for row in values for x in row), matrix), {})]
    assert not result.admitted and result.forecast is None
    assert result.reason == "synthetic_fit_failure"


def test_benchmark_wires_one_fixed_assessment_without_execution(monkeypatch, tmp_path):
    from benchmarks import organized_observed_forecast as driver

    captured = {}

    def record(**kwargs):
        captured.update(kwargs)
        return "synthetic"

    monkeypatch.setattr(driver, "_run_single_analytic_assessment", record)
    assert driver.main(["--output", str(tmp_path)]) == 0
    assert captured["schema_name"] == "organized_observed_forecast"
    assert captured["assess"] is driver._assess
    assert captured["declaration_path"].endswith("ORGANIZED_OBSERVED_FORECAST.md")
    assert captured["admitted_verdict"] == "conditional_record_only_forecast_certified"
