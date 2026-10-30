"""Read-only audit of the fixed-input reuse pilot and retained polynomial.

Byte associations bind this implementation pilot to its frozen source and
previous candidate; they do not authenticate execution or establish every
sample enclosure. The complete signed polynomial defect is rebuilt once.
Neither canonical input sampling nor any candidate producer is executed.
"""

from dataclasses import asdict
from fractions import Fraction as Q
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._interval_chebyshev import _norm_upper_admitted
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _localized_variational_candidate as candidate_owner
from tnfr.physics import _localized_variational_inputs as input_owner
from tnfr.physics import _localized_variational_panel_inputs as panel_input_owner
from tnfr.physics._localized_variational_panels import _bound_cycle_variational_panel
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/localized_variational_reuse"
PRIOR = ROOT / "docs/assets/localized_variational_panel"
JSON_LIMIT = 16 * 1024**2
MEMBER_LIMIT = 64 * 1024**2
ARCHIVE_LIMIT = 128 * 1024**2
PRIOR_NAMES = {
    "protocol.json",
    "source-manifest.json",
    "source.zip",
    "evaluate.py",
    "freeze.json",
    "attempt.json",
    "report.json",
    "response.zip",
}


def _read(directory, name):
    validate_artifact_path(name)
    return decode_exact_tree(
        json_loads(read_bytes_bounded(directory / name, max_bytes=JSON_LIMIT))
    )


def _receipts(items):
    assert type(items) is tuple and items
    result = {}
    for item in items:
        assert type(item) is dict and set(item) == {"path", "bytes", "sha256"}
        name = validate_artifact_path(item["path"])
        assert name not in result
        assert type(item["bytes"]) is int and 0 <= item["bytes"] <= ARCHIVE_LIMIT
        result[name] = item
    return result


def _archive_members(directory, report, expected, selected):
    assert (
        file_receipt(directory / "response.zip", label="response.zip")
        == report["response"]
    )
    receipts = _receipts(report["response_members"])
    assert set(receipts) == expected and selected <= expected
    verify_archive_members(
        directory / "response.zip",
        {name: receipt["sha256"] for name, receipt in receipts.items()},
        max_bytes=ARCHIVE_LIMIT,
    )
    members = {}
    with ZipFile(directory / "response.zip") as archive:
        for name, receipt in receipts.items():
            assert archive.getinfo(name).file_size == receipt["bytes"] <= MEMBER_LIMIT
            if name in selected:
                with archive.open(name) as stream:
                    data = stream.read(MEMBER_LIMIT + 1)
                assert len(data) == receipt["bytes"] <= MEMBER_LIMIT
                assert sha256_bytes(data) == receipt["sha256"]
                members[name] = decode_exact_tree(json_loads(data))
    return members


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    lower, upper = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(lower, upper)
    assert (result.lo, result.hi) == (lower, upper)
    return result


def _table(rows, *, intervals):
    assert type(rows) is tuple and len(rows) == 1024
    assert all(type(row) is tuple and len(row) == 97 for row in rows)
    admit = _interval if intervals else exact_record
    return tuple(tuple(admit(value) for value in row) for row in rows)


@pytest.fixture(scope="module")
def retained():
    def forbidden(*args, **kwargs):
        pytest.fail("read-only reuse evidence audit attempted a scientific producer")

    paths = tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    before = tuple(file_receipt(path, max_bytes=ARCHIVE_LIMIT) for path in paths)
    with pytest.MonkeyPatch.context() as patch:
        for owner, names in (
            (
                input_owner,
                ("_prepare_localized_variational_inputs", "_make_input_evaluator"),
            ),
            (
                panel_input_owner,
                (
                    "_prepare_localized_variational_inputs",
                    "_build_localized_variational_panel_inputs",
                    "_sample_panel_inputs",
                ),
            ),
            (
                candidate_owner,
                (
                    "_construct_cycle_candidate",
                    "_construct_zero_initial_cycle_candidate",
                ),
            ),
        ):
            for name in names:
                patch.setattr(owner, name, forbidden)
        with forbid_organized_response_evaluation():
            records = {
                name: _read(EVIDENCE, name)
                for name in (
                    "protocol.json",
                    "source-manifest.json",
                    "freeze.json",
                    "attempt.json",
                    "report.json",
                )
            }
            records["prior_protocol"] = _read(PRIOR, "protocol.json")
            prior_report = _read(PRIOR, "report.json")
            members = _archive_members(
                EVIDENCE,
                records["report.json"],
                {"inputs.json", "validation.json"},
                {"inputs.json", "validation.json"},
            )
            old = _archive_members(
                PRIOR,
                prior_report,
                {"inputs.json", "candidate.json", "validation.json"},
                {"candidate.json"},
            )
            yield records, members, old["candidate.json"], prior_report
    after_paths = tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    assert after_paths == paths
    assert (
        tuple(file_receipt(path, max_bytes=ARCHIVE_LIMIT) for path in paths) == before
    )


@pytest.fixture(scope="module")
def reconstructed(retained):
    _, members, candidate, _ = retained
    inputs = members["inputs.json"]
    form = _table(candidate["form_coefficients"], intervals=False)
    phase = _table(candidate["phase_coefficients"], intervals=False)
    source_form = _table(inputs["source_form_coefficients"], intervals=True)
    source_phase = _table(inputs["source_phase_coefficients"], intervals=True)
    edge = _table(inputs["edge_coefficient_polynomials"], intervals=True)
    sigma = _interval(inputs["sigma_bounds"])
    assert sigma == pi_interval() / (1024**2 - 1)
    duration = exact_record(inputs["duration"])
    generator_error = exact_record(inputs["generator_error"])
    source_error = exact_record(inputs["source_error"])
    assert duration == 16
    assert generator_error == Q(1, 2**114) and source_error == Q(1, 2**109)
    bounds = _bound_cycle_variational_panel(
        form,
        phase,
        source_form,
        source_phase,
        edge,
        sigma=sigma,
        duration=duration,
        generator_error=generator_error,
        source_error=source_error,
    )
    initial = _norm_upper_admitted(bounds.initial_form + bounds.initial_phase)
    exact_zero = all(
        sum(((-1) ** k * value for k, value in enumerate(row)), Q(0)) == 0
        for table in (form, phase)
        for row in table
    )
    return bounds, initial, exact_zero


def test_frozen_source_prior_candidate_and_attempt_are_byte_associated(retained):
    records, _, _, _ = retained
    freeze, manifest = records["freeze.json"], records["source-manifest.json"]
    assert freeze["schema"] == "tnfr.localized-variational-input-reuse-freeze.v1"
    assert manifest["schema"] == "tnfr.localized-variational-input-reuse-source.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert freeze["byte_associations_are_not_authentication"] is True
    artifacts = _receipts(freeze["artifacts"])
    assert set(artifacts) == {
        "protocol.json",
        "source-manifest.json",
        "source.zip",
        "evaluate.py",
    }
    for name, receipt in artifacts.items():
        assert file_receipt(EVIDENCE / name, label=name) == receipt
    old = _receipts(freeze["prior_artifacts"])
    assert set(old) == PRIOR_NAMES
    for name, receipt in old.items():
        assert file_receipt(PRIOR / name, label=name) == receipt
    assert records["report.json"]["prior_report"] == old["report.json"]
    sources, proofs = _receipts(manifest["files"]), _receipts(manifest["proof_members"])
    assert not set(sources) & set(proofs)
    assert set(proofs) == {"theory/nodal/LOCALIZED_RETENTION_MECHANISMS.md"}
    evaluator_path = "docs/assets/localized_variational_reuse/evaluate.py"
    assert {
        evaluator_path,
        "pyproject.toml",
        "src/tnfr/physics/_localized_variational_inputs.py",
    } <= set(sources)
    assert all(
        name.startswith("src/tnfr/") or name in {evaluator_path, "pyproject.toml"}
        for name in sources
    )
    archive_receipts = sources | proofs
    verify_archive_members(
        EVIDENCE / "source.zip",
        {name: receipt["sha256"] for name, receipt in archive_receipts.items()},
        max_bytes=ARCHIVE_LIMIT,
    )
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert all(
            archive.getinfo(name).file_size == receipt["bytes"]
            for name, receipt in archive_receipts.items()
        )
    assert sources[evaluator_path]["sha256"] == artifacts["evaluate.py"]["sha256"]
    assert sources[evaluator_path]["bytes"] == artifacts["evaluate.py"]["bytes"]
    attempt = records["attempt.json"]
    assert attempt["freeze_sha256"] == file_receipt(EVIDENCE / "freeze.json")["sha256"]
    assert attempt["maximum_evaluations"] == 1 and attempt["numeric_threads"] == 1
    assert not (EVIDENCE / "failure.json").exists()


def test_protocol_and_inputs_keep_same_panel_and_separate_resource_policy(retained):
    records, members, candidate, _ = retained
    protocol = records["protocol.json"]
    assert protocol["schema"] == "tnfr.localized-variational-input-reuse-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["model"] == records["prior_protocol"]["model"]
    panel = protocol["panel"]
    assert panel == records["prior_protocol"]["panel"]
    assert exact_record(panel["start"]) == 0 and exact_record(panel["duration"]) == 16
    assert type(panel["degree"]) is int and panel["degree"] == 96
    assert exact_record(panel["generator_interpolation_error"]) == Q(1, 2**114)
    assert exact_record(panel["source_interpolation_error"]) == Q(1, 2**109)
    assert protocol["method"]["candidate_generator_executed"] is False
    assert protocol["method"]["new_variational_response_evaluated"] is False
    assert (
        protocol["method"]["retained_candidate"] == "prior response.zip/candidate.json"
    )
    assert protocol["method"]["numeric_threads"] == 1
    resources = protocol["resources"]
    for name, expected in (
        ("input_seconds_assessment_threshold", 180),
        ("validation_seconds_assessment_threshold", 120),
        ("full_horizon_seconds_scenario_policy", 86400),
        ("full_horizon_archive_bytes_scenario_policy", 8 * 1024**3),
    ):
        assert type(resources[name]) is int and resources[name] == expected
    assert resources["time_thresholds_are_not_abort_timers"] is True
    assert resources["extrapolation_is_not_a_bound"] is True
    assert protocol["evidence"]["byte_associations_are_not_authentication"] is True
    assert protocol["evidence"]["prior_artifacts_immutable"] is True
    acceptance = protocol["acceptance"]
    assert exact_record(acceptance["candidate_norm_upper"]) == 2**35
    assert exact_record(acceptance["polynomial_defect_norm_upper"]) == Q(1, 2**54)
    assert exact_record(acceptance["initial_enclosure_norm_upper"]) == Q(1, 2**70)
    assert acceptance["exact_initial_zero"] is True
    assert exact_record(acceptance["propagator_gain"]) == 512
    assert protocol["failure_policy"] == {
        "maximum_evaluations": 1,
        "retain_first_outcome": True,
        "automatic_retry_or_parameter_search": False,
    }
    inputs = members["inputs.json"]
    assert exact_record(inputs["start"]) == 0 and exact_record(inputs["end"]) == 16
    assert inputs["node_order"] == tuple(range(1024))
    assert inputs["edges"] == tuple((j, (j + 1) % 1024) for j in range(1024))
    assert inputs["degree"] == candidate["degree"] == 96
    assert candidate["node_count"] == 1024 and candidate["feedback_iterations"] == 20
    times = tuple(_interval(value) for value in inputs["sample_time_bounds"])
    assert len(times) == 97 and (times[0], times[48], times[-1]) == (I(16), I(8), I(0))
    assert all(0 <= time.lo <= time.hi <= 16 for time in times)
    assert all(left.lo >= right.hi for left, right in zip(times, times[1:]))
    assert len(inputs["sample_seconds"]) == 97
    assert all(
        type(value) is float and isfinite(value) and value >= 0
        for value in inputs["sample_seconds"]
    )


def test_same_archived_polynomial_has_rebuilt_complete_signed_defect(
    retained, reconstructed
):
    records, members, _, _ = retained
    bounds, initial, exact_zero = reconstructed
    assert asdict(bounds) == members["validation.json"]
    assert bounds.residual_degree == 192
    report = records["report.json"]
    assert report["schema"] == "tnfr.localized-variational-input-reuse-result.v1"
    assert report["panel"] == (0, 16)
    for name in (
        "candidate_norm_upper",
        "polynomial_defect_norm_upper",
        "conditional_integrated_defect_upper",
    ):
        assert exact_record(report[name]) == getattr(bounds, name)
    assert exact_record(report["initial_enclosure_norm_upper"]) == initial
    assert exact_record(report["local_finite_input_correction_error_upper"]) == 512 * (
        initial + bounds.conditional_integrated_defect_upper
    )
    assert exact_zero
    for name in (
        "candidate_regenerated",
        "new_variational_response_evaluated",
        "whole_interval_evaluated",
        "nonlinear_maintenance_established",
    ):
        assert report[name] is False


def test_report_recomputes_local_checks_and_labels_extrapolation_as_scenario(
    retained, reconstructed
):
    records, members, _, prior = retained
    report = records["report.json"]
    bounds, initial, exact_zero = reconstructed
    seconds = report["seconds"]
    acceptance = records["protocol.json"]["acceptance"]
    resources = records["protocol.json"]["resources"]
    assert all(
        type(value) is float and isfinite(value) and value >= 0
        for value in seconds.values()
    )
    assert (
        seconds["inputs"] > 0
        and seconds["total_with_artifact_io"]
        >= seconds["inputs"] + seconds["validation"]
    )
    expected_checks = {
        "candidate_bound": bounds.candidate_norm_upper
        <= exact_record(acceptance["candidate_norm_upper"]),
        "polynomial_defect": bounds.polynomial_defect_norm_upper
        <= exact_record(acceptance["polynomial_defect_norm_upper"]),
        "initial_enclosure": initial
        <= exact_record(acceptance["initial_enclosure_norm_upper"]),
        "exact_initial_zero": exact_zero,
        "input_cost": seconds["inputs"]
        <= resources["input_seconds_assessment_threshold"],
        "validation_cost": seconds["validation"]
        <= resources["validation_seconds_assessment_threshold"],
    }
    assert all(type(value) is bool for value in report["checks"].values())
    assert report["checks"] == expected_checks
    assert (
        report["input_speed_ratio_to_retained_attempt"]
        == prior["seconds"]["inputs"] / seconds["inputs"]
    )
    horizon = _interval(members["inputs.json"]["horizon_bounds"])
    count = -(-horizon.hi // 16)
    scenario = report["resource_scenario"]
    assert scenario["panel_count"] == count == 8988
    expected_seconds = count * (
        seconds["inputs"] + seconds["validation"] + prior["seconds"]["proposal"]
    )
    expected_bytes = count * prior["response"]["bytes"]
    assert scenario["sequential_seconds_excluding_io"] == expected_seconds
    assert scenario["archive_bytes_at_prior_panel_size"] == expected_bytes
    assert scenario["time_policy_passed"] is (
        expected_seconds <= resources["full_horizon_seconds_scenario_policy"]
    )
    assert scenario["storage_policy_passed"] is (
        expected_bytes <= resources["full_horizon_archive_bytes_scenario_policy"]
    )
    assert (
        scenario["scope"]
        == "Mixed measured-cost extrapolation, not a bound or full-run admission"
    )
