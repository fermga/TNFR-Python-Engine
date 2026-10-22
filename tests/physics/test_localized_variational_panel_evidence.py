"""Read-only first-panel evidence association and conditional reconstruction.

Retained input coefficient boxes are finite numerical evidence. These tests
check their byte associations, primitive domains and complete polynomial
defect, without regenerating the canonical samples or candidate. Receipts
do not authenticate chronology/execution or independently prove every
retained sample enclosure. No archived producer is imported or executed.
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
EVIDENCE = ROOT / "docs/assets/localized_variational_panel"
MEMBER_LIMIT = 128 * 1024**2
JSON_LIMIT = 16 * 1024**2


def _read(name):
    validate_artifact_path(name)
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=JSON_LIMIT))
    )


def _interval(record):
    assert type(record) is dict and set(record) == {"lo", "hi"}
    endpoints = exact_record(record["lo"]), exact_record(record["hi"])
    result = I(*endpoints)
    assert (result.lo, result.hi) == endpoints
    return result


def _table(rows, *, intervals):
    assert type(rows) is tuple and len(rows) == 1024
    assert all(type(row) is tuple and len(row) == 97 for row in rows)
    admit = _interval if intervals else exact_record
    return tuple(tuple(admit(value) for value in row) for row in rows)


def _receipt_map(receipts):
    assert type(receipts) is tuple and receipts
    result = {}
    for receipt in receipts:
        assert type(receipt) is dict and set(receipt) == {"path", "bytes", "sha256"}
        path = validate_artifact_path(receipt["path"])
        assert path not in result
        assert type(receipt["bytes"]) is int and receipt["bytes"] >= 0
        assert type(receipt["sha256"]) is str
        result[path] = receipt
    return result


@pytest.fixture(scope="module")
def retained():
    def forbidden(*args, **kwargs):
        pytest.fail("retained panel audit attempted a canonical producer")

    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    before = tuple(file_receipt(path, max_bytes=3 * MEMBER_LIMIT) for path in paths)
    with pytest.MonkeyPatch.context() as patch:
        for owner, names in (
            (input_owner, ("_prepare_localized_variational_inputs",)),
            (
                panel_input_owner,
                (
                    "_prepare_localized_variational_inputs",
                    "_build_localized_variational_panel_inputs",
                ),
            ),
            (candidate_owner, ("_construct_zero_initial_cycle_candidate",)),
        ):
            for name in names:
                patch.setattr(owner, name, forbidden)
        with forbid_organized_response_evaluation():
            records = {
                name: _read(name)
                for name in (
                    "protocol.json",
                    "source-manifest.json",
                    "freeze.json",
                    "attempt.json",
                    "report.json",
                )
            }
            report = records["report.json"]
            assert (
                file_receipt(EVIDENCE / "response.zip", label="response.zip")
                == report["response"]
            )
            receipts = _receipt_map(report["response_members"])
            assert set(receipts) == {"inputs.json", "candidate.json", "validation.json"}
            verify_archive_members(
                EVIDENCE / "response.zip",
                {name: receipt["sha256"] for name, receipt in receipts.items()},
                max_bytes=3 * MEMBER_LIMIT,
            )
            members = {}
            with ZipFile(EVIDENCE / "response.zip") as archive:
                for name, receipt in receipts.items():
                    assert (
                        archive.getinfo(name).file_size
                        == receipt["bytes"]
                        <= MEMBER_LIMIT
                    )
                    with archive.open(name) as stream:
                        data = stream.read(MEMBER_LIMIT + 1)
                    assert len(data) == receipt["bytes"] <= MEMBER_LIMIT
                    assert sha256_bytes(data) == receipt["sha256"]
                    members[name] = decode_exact_tree(json_loads(data))
            yield records, members
    assert tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file())) == paths
    assert (
        tuple(file_receipt(path, max_bytes=3 * MEMBER_LIMIT) for path in paths)
        == before
    )


@pytest.fixture(scope="module")
def reconstructed(retained):
    _, members = retained
    inputs, candidate = members["inputs.json"], members["candidate.json"]
    source_form = _table(inputs["source_form_coefficients"], intervals=True)
    source_phase = _table(inputs["source_phase_coefficients"], intervals=True)
    edge = _table(inputs["edge_coefficient_polynomials"], intervals=True)
    form = _table(candidate["form_coefficients"], intervals=False)
    phase = _table(candidate["phase_coefficients"], intervals=False)
    sigma = _interval(inputs["sigma_bounds"])
    assert sigma == pi_interval() / (1024**2 - 1)
    assert exact_record(inputs["start"]) == 0
    assert exact_record(inputs["duration"]) == exact_record(inputs["end"]) == 16
    assert exact_record(inputs["generator_error"]) == Q(1, 2**114)
    assert exact_record(inputs["source_error"]) == Q(1, 2**109)
    source_radius, edge_radius = panel_input_owner._coefficient_radius_bounds(
        source_form, source_phase, edge
    )
    assert exact_record(inputs["source_coefficient_radius_norm_sum"]) == source_radius
    assert exact_record(inputs["edge_coefficient_radius_operator_upper"]) == edge_radius
    bounds = _bound_cycle_variational_panel(
        form,
        phase,
        source_form,
        source_phase,
        edge,
        sigma=sigma,
        duration=16,
        generator_error=Q(1, 2**114),
        source_error=Q(1, 2**109),
    )
    exact_zero = all(
        sum(((-1) ** index * value for index, value in enumerate(row)), Q(0)) == 0
        for table in (form, phase)
        for row in table
    )
    initial = _norm_upper_admitted(bounds.initial_form + bounds.initial_phase)
    return bounds, initial, exact_zero


def test_frozen_source_response_and_attempt_associations(retained):
    records, _ = retained
    freeze, manifest = records["freeze.json"], records["source-manifest.json"]
    freeze_artifacts = _receipt_map(freeze["artifacts"])
    assert set(freeze_artifacts) == {
        "protocol.json",
        "source-manifest.json",
        "source.zip",
        "evaluate.py",
    }
    for name, receipt in freeze_artifacts.items():
        assert file_receipt(EVIDENCE / name, label=name) == receipt
    source_receipts = _receipt_map(manifest["files"])
    proof_receipts = _receipt_map(manifest["proof_members"])
    assert not set(source_receipts) & set(proof_receipts)
    all_receipts = source_receipts | proof_receipts
    verify_archive_members(
        EVIDENCE / "source.zip",
        {name: receipt["sha256"] for name, receipt in all_receipts.items()},
        max_bytes=128 * 1024**2,
    )
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert all(
            archive.getinfo(name).file_size == receipt["bytes"]
            for name, receipt in all_receipts.items()
        )
    assert "docs/assets/localized_variational_panel/evaluate.py" in source_receipts
    assert "pyproject.toml" in source_receipts
    assert "src/tnfr/physics/_localized_variational_panels.py" in source_receipts
    assert set(proof_receipts) == {"theory/nodal/LOCALIZED_RETENTION_MECHANISMS.md"}
    evaluator = source_receipts["docs/assets/localized_variational_panel/evaluate.py"]
    assert evaluator["sha256"] == freeze_artifacts["evaluate.py"]["sha256"]
    assert evaluator["bytes"] == freeze_artifacts["evaluate.py"]["bytes"]
    assert freeze["schema"] == "tnfr.localized-variational-first-panel-freeze.v1"
    assert manifest["schema"] == "tnfr.localized-variational-first-panel-source.v1"
    assert freeze["byte_associations_are_not_authentication"] is True
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert (
        records["attempt.json"]["freeze_sha256"]
        == file_receipt(EVIDENCE / "freeze.json")["sha256"]
    )
    assert records["attempt.json"]["maximum_evaluations"] == 1
    assert records["attempt.json"]["numeric_threads"] == 1
    assert not (EVIDENCE / "failure.json").exists()


def test_frozen_protocol_keeps_model_clock_prediction_and_failure_policy(retained):
    records, _ = retained
    protocol = records["protocol.json"]
    assert protocol["schema"] == "tnfr.localized-variational-first-panel-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    model, panel = protocol["model"], protocol["panel"]
    assert type(model["cycle_nodes"]) is int and model["cycle_nodes"] == 1024
    assert model["node_order"] == tuple(range(1024))
    assert exact_record(model["form_loss"]) == Q(1, 2**20)
    assert exact_record(model["exchange"]) == 1 - Q(1, 2**20)
    assert exact_record(model["capacity"]) == exact_record(model["storage_scale"]) == 1
    assert exact_record(model["profile_parameter"]) == Q(1, 2)
    assert (model["profile_cutoff"], model["sine_degree"], model["cosine_degree"]) == (
        24,
        17,
        16,
    )
    assert model["inputs_and_events"] == "absent"
    assert exact_record(panel["start"]) == 0 and exact_record(panel["duration"]) == 16
    assert panel["degree"] == 96
    assert exact_record(panel["generator_interpolation_error"]) == Q(1, 2**114)
    assert exact_record(panel["source_interpolation_error"]) == Q(1, 2**109)
    assert protocol["proposal"]["feedback_updates"] == 20
    assert protocol["proposal"]["feedback_product_degree"] == 192
    assert protocol["proposal"]["iteration_convergence_claim"] is False
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
    assert protocol["resources"]["time_thresholds_are_not_abort_timers"] is True
    assert protocol["resources"]["input_seconds_assessment_threshold"] == 600
    assert protocol["resources"]["proposal_seconds_assessment_threshold"] == 180
    assert protocol["resources"]["validation_seconds_assessment_threshold"] == 120
    assert protocol["evidence"]["byte_associations_are_not_authentication"] is True


def test_retained_primitive_shape_clock_and_separate_input_tails(retained):
    records, members = retained
    inputs, candidate = members["inputs.json"], members["candidate.json"]
    assert inputs["node_order"] == tuple(range(1024))
    assert inputs["edges"] == tuple(
        (index, (index + 1) % 1024) for index in range(1024)
    )
    assert type(inputs["degree"]) is int and inputs["degree"] == 96
    assert type(candidate["degree"]) is int and candidate["degree"] == 96
    assert type(candidate["node_count"]) is int and candidate["node_count"] == 1024
    assert candidate["feedback_iterations"] == 20
    assert candidate["factor_count"] == 512
    times = tuple(_interval(value) for value in inputs["sample_time_bounds"])
    assert len(times) == 97 and times[0] == I(16) and times[-1] == I(0)
    assert times[48] == I(8)
    assert all(0 <= value.lo <= value.hi <= 16 for value in times)
    assert all(left.lo >= right.hi for left, right in zip(times, times[1:]))
    assert len(inputs["sample_seconds"]) == 97
    assert all(
        type(value) is float and isfinite(value) and value >= 0
        for value in inputs["sample_seconds"]
    )
    assert records["report.json"]["whole_interval_evaluated"] is False
    assert records["report.json"]["nonlinear_maintenance_established"] is False


def test_full_signed_defect_rebuild_matches_retained_bounds(retained, reconstructed):
    _, members = retained
    bounds, _, _ = reconstructed
    assert asdict(bounds) == members["validation.json"]
    assert bounds.residual_degree == 192
    assert bounds.generator_error_bounds == I(Q(1, 2**114))
    assert bounds.source_error_bounds == I(Q(1, 2**109))


def test_report_assessment_reduces_rebuilt_primitives_not_cached_verdicts(
    retained, reconstructed
):
    records, members = retained
    report, candidate = records["report.json"], members["candidate.json"]
    bounds, initial, exact_zero = reconstructed
    assert report["schema"] == "tnfr.localized-variational-first-panel-result.v1"
    assert report["panel"] == (0, 16)
    assert all(
        type(value) is float and isfinite(value) and value >= 0
        for value in report["seconds"].values()
    )
    assert all(type(value) is bool for value in report["checks"].values())
    acceptance = records["protocol.json"]["acceptance"]
    resources = records["protocol.json"]["resources"]
    expected = {
        "candidate_bound": bounds.candidate_norm_upper
        <= exact_record(acceptance["candidate_norm_upper"]),
        "polynomial_defect": bounds.polynomial_defect_norm_upper
        <= exact_record(acceptance["polynomial_defect_norm_upper"]),
        "initial_enclosure": initial
        <= exact_record(acceptance["initial_enclosure_norm_upper"]),
        "exact_initial_zero": exact_zero,
        "input_cost": report["seconds"]["inputs"]
        <= resources["input_seconds_assessment_threshold"],
        "proposal_cost": candidate["total_seconds"]
        <= resources["proposal_seconds_assessment_threshold"],
        "validation_cost": report["seconds"]["validation"]
        <= resources["validation_seconds_assessment_threshold"],
    }
    assert report["checks"] == expected
    assert exact_zero
    assert report["panel_numerical_policy_passed"] is all(
        expected[name]
        for name in (
            "candidate_bound",
            "polynomial_defect",
            "initial_enclosure",
            "exact_initial_zero",
        )
    )
    assert report["local_resource_policy_passed"] is all(
        expected[name] for name in ("input_cost", "proposal_cost", "validation_cost")
    )
    assert report["candidate_norm_upper"] == bounds.candidate_norm_upper
    assert report["polynomial_defect_norm_upper"] == bounds.polynomial_defect_norm_upper
    assert (
        report["conditional_integrated_defect_upper"]
        == bounds.conditional_integrated_defect_upper
    )
    assert report["initial_enclosure_norm_upper"] == initial
    assert report["local_finite_input_correction_error_upper"] == exact_record(
        acceptance["propagator_gain"]
    ) * (initial + bounds.conditional_integrated_defect_upper)
    assert report["seconds"]["proposal"] == candidate["total_seconds"]
    for name in (
        "source_coefficient_radius_norm_sum",
        "edge_coefficient_radius_operator_upper",
    ):
        assert exact_record(report[name]) == exact_record(members["inputs.json"][name])
