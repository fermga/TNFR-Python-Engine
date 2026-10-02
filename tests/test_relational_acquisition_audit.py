"""Read-only record admission without importing or executing a research producer."""

import hashlib
import zipfile
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_observations import (
    bound_relational_coefficient_from_samples,
)
from tnfr.research import relational_acquisition as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def synthetic_record():
    # An internally bounded but uninformative record, not a native trajectory.
    # Constant retained states and zero declared rates have a large field defect;
    # the independently reconstructed error budget therefore cannot resolve chi.
    u, r, step = Q(1, 8), Q(1, 256), Q(1, 2**21)
    horizon = 2 * step
    lower = 1 - r**2 / 6
    q0, q1 = 1 / lower, r / (3 * lower**2)
    p, speed = u + r / 3, Q(2, 3) * u * q0
    u2, phase2 = p + speed / 3, Q(2, 3) * (p * q0 + u * q1 * speed)
    bounds = {
        "sinc_lower": lower,
        "inverse_sinc_upper": q0,
        "inverse_sinc_derivative_upper": q1,
        "form_speed_upper": p,
        "phase_speed_upper": speed,
        "second_derivative_upper": max(u2, phase2),
        "third_form_derivative_upper": u2 + phase2 / 3,
        "lipschitz_upper": Q(2),
        "growth_factor_upper": 1 / (1 - 2 * horizon),
    }
    protocol = dict(
        protocol="relational-p2-coefficient-acquisition-v1",
        nodes=(0, 1),
        edges=((0, 1),),
        capacity=(Q(1), Q(1)),
        effective_weights=(Q(1, 2), Q(1, 2)),
        initial_phase=(Q(0), Q(0)),
        gamma="absent",
        events="none; support and capacity held; no clipping or controller",
        observation="exact represented EPI[0]-EPI[1]; gain one; baseline zero",
        clock="relative structural time k*euler_step; exact dyadic grid",
        pressure_refresh="native field at every step and endpoint",
        prior_rectangle=(u, r),
        prior_beta=(Q(1, 2), Q(2)),
        source_beta=Q(1),
        initial_form=(u / 2, -u / 2),
        euler_step=step,
        sample_step=step,
        horizon=horizon,
        step_count=2,
        sample_indices=(0, 1, 2),
        precision_limit=Q(1, 8),
        window_bounds=bounds,
        runtime={
            "pressure_path": "fused_canonical",
            "integrator": "native simultaneous explicit Euler",
            "interval_method": "rational_outward_dyadic128_machin_trigonometry_v1",
            "precision": (
                "binary64 native states; exact Fraction defects; "
                "outward dyadic128 oracle"
            ),
        },
        source_sha256={
            "fixture.py": hashlib.sha256(b"# synthetic source\n").hexdigest()
        },
    )
    frames = []
    for index in range(3):
        field_sum = index * step * u
        truncation = index * bounds["second_derivative_upper"] * step**2 / 2
        frame = dict(
            step=index,
            time=index * step,
            epi=(0.0625, -0.0625),
            phase=(0.0, 0.0),
            pressure_path="fused_canonical",
            continuous_contrast_error_upper=(field_sum + truncation)
            / (1 - 2 * horizon),
        )
        if index:
            frame.update(
                contrast_rates=(Q(0), Q(0)),
                start_form_rate=(0.0, 0.0),
                start_phase_rate=(0.0, 0.0),
                field_error_upper=u,
                update_defects=(Q(0), Q(0)),
                clock_defect=Q(0),
                accumulated_field_error=field_sum,
                accumulated_update_error=Q(0),
                accumulated_truncation_error=truncation,
            )
        frames.append(frame)
    report = bound_relational_coefficient_from_samples(
        (u, u, u),
        sample_step=step,
        sample_error_bound=frames[-1]["continuous_contrast_error_upper"],
        third_derivative_bound=bounds["third_form_derivative_upper"],
    )
    assert report.jet.coefficient_bounds is None
    return dict(
        protocol=protocol,
        frames=frames,
        completed=True,
        stopped=None,
        analysis_error=None,
        sample_analysis=relational_report_to_dict(report),
        contains_predeclared_truth=None,
        width_within_budget=None,
        passed=False,
    )


def test_consistency_does_not_promote_an_unresolved_record_to_success(synthetic_record):
    result = owner.audit_relational_coefficient_record(
        synthetic_record, protocol=synthetic_record["protocol"]
    )
    assert result.consistent and result.acquisition_complete
    assert result.completed_steps == 2
    assert result.recorded_passed is result.reconstructed_passed is False
    assert result.sample_bounds.jet.coefficient_bounds is None
    assert "unresolved_restoring_gap" in result.unavailable_reasons
    payload = result.to_dict()
    payload["sample_bounds"]["report"]["samples"][0]["numerator"] = 999
    assert result.sample_bounds.samples[0] == Q(1, 8)
    with pytest.raises(FrozenInstanceError):
        result.status = "passed"


@pytest.mark.parametrize(
    "field,value",
    (
        ("time", Q(7)),
        ("clock_defect", Q(1, 2**80)),
        ("field_error_upper", Q(0)),
        ("update_defects", (Q(1), Q(0))),
        ("start_form_rate", (True, 0.0)),
        ("start_form_rate", (10**400, 10**400)),
        ("start_phase_rate", (2**53 + 1, 2**53 + 1)),
        ("accumulated_truncation_error", Q(0)),
        ("accumulated_update_error", Q(1)),
        ("continuous_contrast_error_upper", Q(0)),
        ("step", True),
        ("pressure_path", "unrecorded_path"),
    ),
)
def test_corrupt_step_evidence_rejects(synthetic_record, field, value):
    record = deepcopy(synthetic_record)
    record["frames"][1][field] = value
    with pytest.raises((ValueError, TypeError)):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


def test_embedded_precision_change_cannot_rewrite_the_frozen_protocol(synthetic_record):
    record = deepcopy(synthetic_record)
    record["protocol"]["precision_limit"] = Q(10)
    with pytest.raises(ValueError, match="frozen protocol differs"):
        owner.audit_relational_coefficient_record(
            record, protocol=synthetic_record["protocol"]
        )


@pytest.mark.parametrize(
    "field,value",
    (("passed", True), ("completed", 1), ("width_within_budget", False)),
)
def test_decision_flags_cannot_be_coerced_or_invented(synthetic_record, field, value):
    record = deepcopy(synthetic_record)
    record[field] = value
    with pytest.raises(ValueError):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


def test_partial_record_keeps_estimate_unavailable(synthetic_record):
    record = deepcopy(synthetic_record)
    record.update(
        frames=record["frames"][:2],
        completed=False,
        stopped={"attempted_step": 2, "error_type": "ValueError", "error": "fixture"},
        sample_analysis=None,
    )
    result = owner.audit_relational_coefficient_record(
        record, protocol=record["protocol"]
    )
    assert result.consistent and result.acquisition_complete is False
    assert result.completed_steps == 1 and result.sample_bounds is None
    assert result.unavailable_reasons == ("acquisition_incomplete",)


@pytest.mark.parametrize("field", ("stopped", "analysis_error"))
def test_missing_failure_fields_reject(synthetic_record, field):
    record = deepcopy(synthetic_record)
    del record[field]
    with pytest.raises(ValueError, match="missing failure metadata"):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


@pytest.mark.parametrize("failure", (False, True, {"error_type": 1, "error": "bad"}))
def test_failure_markers_require_explicit_structured_metadata(
    synthetic_record, failure
):
    record = deepcopy(synthetic_record)
    record.update(analysis_error=failure, sample_analysis=None)
    with pytest.raises(ValueError, match="failure metadata"):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


def test_stop_ordinal_and_analysis_stage_must_match_the_retained_prefix(
    synthetic_record,
):
    record = deepcopy(synthetic_record)
    record.update(
        frames=record["frames"][:2],
        completed=False,
        sample_analysis=None,
        stopped={"attempted_step": 1, "error_type": "ValueError", "error": "fixture"},
    )
    with pytest.raises(ValueError, match="stop ordinal"):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])
    record["stopped"]["attempted_step"] = 2
    record["analysis_error"] = {"error_type": "ValueError", "error": "fixture"}
    with pytest.raises(ValueError, match="cannot have analysis"):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


@pytest.mark.parametrize(
    "path",
    (
        ("clock",),
        ("observation",),
        ("pressure_refresh",),
        ("runtime", "integrator"),
        ("runtime", "precision"),
        ("runtime", "interval_method"),
    ),
)
def test_matching_but_unsupported_protocol_declarations_reject(synthetic_record, path):
    record = deepcopy(synthetic_record)
    item = record["protocol"]
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = "unsupported"
    with pytest.raises(ValueError, match="differs"):
        owner.audit_relational_coefficient_record(record, protocol=record["protocol"])


@pytest.mark.parametrize("before_acquisition", (False, True))
def test_valid_recorded_failures_preserve_unavailable_analysis(
    synthetic_record, before_acquisition
):
    record = deepcopy(synthetic_record)
    record["sample_analysis"] = None
    failure = {"error_type": "ValueError", "error": "fixture failure"}
    if before_acquisition:
        record.update(
            frames=[],
            completed=False,
            stopped={"stage": "before_acquisition", **failure},
        )
    else:
        record["analysis_error"] = failure
    result = owner.audit_relational_coefficient_record(
        record, protocol=record["protocol"]
    )
    assert result.consistent and result.sample_bounds is None
    assert result.recorded_passed is result.reconstructed_passed is False
    assert result.acquisition_complete is (not before_acquisition)


def _write_bundle(tmp_path, record):
    response = tmp_path / "result.json"
    frozen, archive = response.with_suffix(".protocol.json"), response.with_suffix(
        ".source.zip"
    )
    export_to_json(json_loads(owner._encoded(record["protocol"])), frozen)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("fixture.py", b"# synthetic source\n")
    payload = deepcopy(record)
    payload["protocol_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    payload["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    export_to_json(json_loads(owner._encoded(payload)), response)
    return response, frozen, archive


def test_file_audit_checks_bytes_without_writing_or_producer_import(
    synthetic_record, tmp_path
):
    paths = _write_bundle(tmp_path, synthetic_record)
    before = tuple(path.read_bytes() for path in paths)
    result = owner.audit_relational_coefficient_acquisition(paths[0])
    assert result.consistent, result.unavailable_reasons
    assert result.recorded_passed is False
    assert tuple(path.read_bytes() for path in paths) == before


def test_missing_files_are_unavailable_not_a_negative_measurement(tmp_path):
    result = owner.audit_relational_coefficient_acquisition(tmp_path / "missing.json")
    assert result.status == "unavailable" and not result.consistent
    assert result.recorded_passed is result.reconstructed_passed is None
    assert result.sample_bounds is None


def test_duplicate_json_keys_and_changed_archive_are_inconsistent(
    synthetic_record, tmp_path
):
    response, frozen, archive = _write_bundle(tmp_path, synthetic_record)
    original = response.read_bytes()
    response.write_bytes(b'{"protocol":1,"protocol":2}')
    assert (
        owner.audit_relational_coefficient_acquisition(response).status
        == "inconsistent"
    )
    response.write_bytes(original)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("fixture.py", b"modified source")
    result = owner.audit_relational_coefficient_acquisition(response)
    assert result.status == "inconsistent"
    assert result.unavailable_reasons == ("archive digest differs",)
    assert frozen.is_file()


@pytest.mark.parametrize("change", ("member", "inventory", "compression"))
def test_archive_contents_reject_even_with_an_updated_outer_digest(
    synthetic_record, tmp_path, change
):
    response, _, archive = _write_bundle(tmp_path, synthetic_record)
    compression = zipfile.ZIP_BZIP2 if change == "compression" else zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w", compression=compression) as stream:
        stream.writestr(
            "extra.py" if change == "inventory" else "fixture.py",
            b"changed member" if change == "member" else b"# synthetic source\n",
        )
    record = json_loads(response.read_bytes())
    record["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    export_to_json(record, response)
    result = owner.audit_relational_coefficient_acquisition(response)
    assert result.status == "inconsistent"
    assert result.recorded_passed is None


def test_cli_uses_only_the_read_only_owner(monkeypatch, tmp_path, capsys):
    path = tmp_path / "missing.json"
    calls = []

    def audit(value):
        calls.append(value)
        return owner.RelationalAcquisitionAudit(
            "unavailable", unavailable_reasons=("missing_artifacts",)
        )

    monkeypatch.setattr(owner, "audit_relational_coefficient_acquisition", audit)
    assert owner.main([str(path)]) == 2
    assert calls == [path]
    assert '"status":"unavailable"' in capsys.readouterr().out
