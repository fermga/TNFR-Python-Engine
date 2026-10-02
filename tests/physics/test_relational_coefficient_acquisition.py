"""P2 acquisition bounds, short wiring controls and read-only retained evidence.

No test regenerates the full frozen temporal response. The two-step wiring
fixture has a different horizon; the full record audit is optional if absent.
"""

import hashlib
import zipfile
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath as mp
import pytest
import sympy as sp

from benchmarks import relational_coefficient_acquisition as acquisition
from tnfr.research.relational_acquisition import (
    audit_relational_coefficient_acquisition,
)
from tnfr.utils.io import json_loads


def _q(value):
    if isinstance(value, dict):
        return Q(value["numerator"], value["denominator"])
    return Q(value)


def _mp(value):
    value = _q(value)
    return mp.mpf(value.numerator) / value.denominator


def test_prior_bounds_follow_the_nonlinear_derivative_chain():
    u, phase, beta = sp.symbols("u phase beta", real=True, nonzero=True)
    q = phase / sp.sin(phase)
    field = sp.Matrix((-u - phase / sp.pi, u * q / (beta * sp.pi)))
    second = field.jacobian((u, phase)) * field
    third_form = (sp.Matrix([second[0]]).jacobian((u, phase)) * field)[0]
    assert sp.simplify(third_form + second[0] + second[1] / sp.pi) == 0
    assert (
        sp.simplify(
            second[1]
            - (field[0] * q + u * sp.diff(q, phase) * field[1]) / (beta * sp.pi)
        )
        == 0
    )
    energy = u**2 / 2 + beta * (1 - sp.cos(phase))
    assert sp.simplify((sp.Matrix([energy]).jacobian((u, phase)) * field)[0]) == -(u**2)
    bounds = acquisition.prior_window_bounds()
    assert acquisition.HORIZON * bounds["phase_speed_upper"] == Q(64, 393215)
    assert acquisition.HORIZON * bounds["phase_speed_upper"] < acquisition.PHASE_LIMIT
    assert bounds["growth_factor_upper"] == Q(256, 255)
    assert bounds["second_derivative_upper"] > Q(15, 100)
    assert Q(18, 100) < bounds["third_form_derivative_upper"] < Q(19, 100)


@pytest.mark.parametrize("phase", (0, Q(1, 256), Q(-1, 256), Q(1, 2**80)))
@pytest.mark.parametrize("beta", (Q(1, 2), Q(2)))
def test_field_interval_contains_an_independent_high_precision_reference(phase, beta):
    state = (Q(-1, 16), phase)
    intervals = acquisition.enclose_p2_field(state, beta)
    with mp.workdps(90):
        u, angle = map(_mp, state)
        ratio = angle / mp.sin(angle) if angle else mp.mpf(1)
        expected = (-u - angle / mp.pi, u * ratio / (_mp(beta) * mp.pi))
        for interval, value in zip(intervals, expected, strict=True):
            assert _mp(interval.lo) <= value <= _mp(interval.hi)


@pytest.mark.parametrize(
    "state,beta",
    (
        ((Q(1, 4), 0), 1),
        ((0, Q(1, 128)), 1),
        ((0, 0), Q(1, 4)),
        ((0, True), 1),
        ((0, 0), True),
        ((0, float("nan")), 1),
    ),
)
def test_field_enclosure_rejects_unsupported_domain(state, beta):
    with pytest.raises((ValueError, TypeError)):
        acquisition.enclose_p2_field(state, beta)


@pytest.fixture
def small_source(monkeypatch):
    files = {"benchmarks/fixture.py": b"# Source-archive plumbing only.\n"}
    monkeypatch.setattr(acquisition, "_source_files", lambda: dict(files))
    return files


@pytest.mark.parametrize(
    "key", ("source_beta", "sample_step", "initial_phase", "gamma")
)
def test_changed_protocol_rejects_before_any_field_or_step(
    monkeypatch, small_source, key
):
    protocol = deepcopy(acquisition.prepare_protocol())
    protocol[key] = "unfrozen"

    def forbidden(*args, **kwargs):
        pytest.fail("a mismatched protocol must not execute a scientific field")

    monkeypatch.setattr(acquisition, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(acquisition, "step_relational_exchange", forbidden)
    with pytest.raises(ValueError, match="protocol/source mismatch"):
        acquisition.evaluate_protocol(protocol)


def test_imported_source_must_belong_to_the_frozen_checkout(monkeypatch, tmp_path):
    monkeypatch.setitem(
        acquisition.sys.modules,
        "tnfr.foreign_source_control",
        SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    with pytest.raises(ValueError, match="outside the archived checkout"):
        acquisition.prepare_protocol()


@pytest.fixture
def two_step_protocol(monkeypatch, small_source):
    # This short wiring test is not the reserved 0,h,2h acquisition window.
    monkeypatch.setattr(acquisition, "STEP_COUNT", 2)
    monkeypatch.setattr(acquisition, "SAMPLE_STEP", acquisition.EULER_STEP)
    monkeypatch.setattr(acquisition, "HORIZON", 2 * acquisition.EULER_STEP)
    monkeypatch.setattr(acquisition, "SAMPLE_INDICES", (0, 1, 2))
    return acquisition.prepare_protocol()


def test_two_native_steps_retain_actual_clock_rates_and_update_defects(
    two_step_protocol,
):
    result = acquisition.evaluate_protocol(two_step_protocol)
    assert result["completed"], result["stopped"]
    assert len(result["frames"]) == 3
    previous = result["frames"][0]
    for index, frame in enumerate(result["frames"][1:], 1):
        assert frame["time"] == index * acquisition.EULER_STEP
        assert frame["clock_defect"] == 0
        for axis, name in enumerate(("epi", "phase")):
            old = Q(previous[name][0]) - Q(previous[name][1])
            new = Q(frame[name][0]) - Q(frame[name][1])
            expected = (
                new - old - acquisition.EULER_STEP * frame["contrast_rates"][axis]
            )
            assert frame["update_defects"][axis] == expected
        assert frame["accumulated_truncation_error"] > 0
        previous = frame
    assert (
        result["sample_analysis"]["report_type"] == "RelationalCoefficientSampleBounds"
    )
    assert result["analysis_error"] is None


def test_analysis_failure_preserves_the_acquired_frames(monkeypatch, two_step_protocol):
    def fail(*args, **kwargs):
        raise ValueError("injected analysis failure")

    monkeypatch.setattr(acquisition, "bound_relational_coefficient_from_samples", fail)
    result = acquisition.evaluate_protocol(two_step_protocol)
    assert result["completed"] and not result["passed"]
    assert len(result["frames"]) == 3
    assert result["analysis_error"]["error"] == "injected analysis failure"
    assert result["sample_analysis"] is None
    assert result["contains_predeclared_truth"] is None
    assert result["width_within_budget"] is None


def test_clock_failure_retains_partial_evidence_and_unavailable_estimate(
    monkeypatch, two_step_protocol
):
    native_step = acquisition.step_relational_exchange

    def bad_clock(*args, **kwargs):
        step = native_step(*args, **kwargs)
        return replace(step, clock_defect=Q(1, 2**80))

    monkeypatch.setattr(acquisition, "step_relational_exchange", bad_clock)
    result = acquisition.evaluate_protocol(two_step_protocol)
    assert not result["completed"] and not result["passed"]
    assert result["stopped"]["attempted_step"] == 1
    assert "clock" in result["stopped"]["error"]
    assert len(result["frames"]) == 1
    assert result["sample_analysis"] is None
    assert result["contains_predeclared_truth"] is None
    assert result["width_within_budget"] is None


def test_cli_freezes_without_execution_and_preserves_a_failed_response(
    monkeypatch, small_source, tmp_path
):
    calls = []

    def negative(protocol):
        calls.append(protocol)
        return dict(
            protocol=protocol,
            completed=True,
            passed=False,
            contains_predeclared_truth=None,
            width_within_budget=None,
        )

    monkeypatch.setattr(acquisition, "evaluate_protocol", negative)
    output = tmp_path / "result.json"
    arguments = ["--output", str(output)]
    assert acquisition.main(["--prepare", *arguments]) == 0
    assert calls == [] and not output.exists()
    frozen = output.with_suffix(".protocol.json")
    saved_protocol = frozen.read_bytes()
    with pytest.raises(FileExistsError):
        acquisition.main(["--prepare", *arguments])
    assert acquisition.main(arguments) == 1
    saved_response = output.read_bytes()
    payload = json_loads(saved_response)
    assert payload["passed"] is False
    assert payload["contains_predeclared_truth"] is None
    assert payload["protocol_sha256"] == hashlib.sha256(saved_protocol).hexdigest()
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        acquisition.main(arguments)
    assert (
        output.read_bytes() == saved_response and frozen.read_bytes() == saved_protocol
    )


def test_cli_preserves_provenance_for_preacquisition_error(
    monkeypatch, small_source, tmp_path
):
    output = tmp_path / "result.json"
    arguments = ["--output", str(output)]
    acquisition.main(["--prepare", *arguments])

    def fail(*args, **kwargs):
        raise ValueError("injected preparation failure")

    monkeypatch.setattr(acquisition, "evaluate_protocol", fail)
    with pytest.raises(ValueError, match="preparation failure"):
        acquisition.main(arguments)
    record = json_loads(output.read_bytes())
    assert not record["completed"] and not record["passed"]
    assert record["frames"] == [] and record["sample_analysis"] is None
    assert record["contains_predeclared_truth"] is None
    assert record["width_within_budget"] is None
    assert record["protocol_sha256"] and record["source_archive_sha256"]


def test_tampered_archive_rejects_before_response(monkeypatch, small_source, tmp_path):
    output = tmp_path / "result.json"
    arguments = ["--output", str(output)]
    acquisition.main(["--prepare", *arguments])
    with zipfile.ZipFile(output.with_suffix(".source.zip"), "w") as archive:
        archive.writestr(next(iter(small_source)), b"different source")

    def forbidden(*args, **kwargs):
        pytest.fail("an inconsistent archive must not execute a response")

    monkeypatch.setattr(acquisition, "evaluate_protocol", forbidden)
    with pytest.raises(ValueError, match="archive content differs"):
        acquisition.main(arguments)
    assert not output.exists()


@pytest.fixture(scope="module")
def retained_record():
    path = acquisition.DEFAULT_OUTPUT
    if not path.is_file():
        pytest.skip("local frozen acquisition is absent; never regenerate it in a test")
    return json_loads(path.read_bytes())


def test_retained_response_reconstructs_every_error_and_the_sample_decision(
    retained_record,
):
    audit = audit_relational_coefficient_acquisition(acquisition.DEFAULT_OUTPUT)
    assert audit.consistent, audit.unavailable_reasons
    assert audit.acquisition_complete
    assert audit.completed_steps == retained_record["protocol"]["step_count"]
    assert (
        audit.recorded_passed is audit.reconstructed_passed is retained_record["passed"]
    )
    assert audit.sample_bounds.jet.coefficient_bounds is not None
