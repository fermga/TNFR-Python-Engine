"""Read-only spatial projection of already retained complete coefficients.

The old coefficient-generation execution remains a stated premise. Shared audit
fixtures rebuild every retained polynomial endpoint, time tail and event carry;
this file reconstructs the new observation and its own complete error budget.
Compact byte references establish association, not an unseen response or proof
of acquisition, execution authenticity, or physical measurement.
"""

import io
import zipfile
from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.physics import test_sine_class_cubic_evidence as cubic_audit
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (
    ROOT / "docs/assets/sine_formed_classes/class-spatial-observation-v1.evidence.zip"
)
TRANSPORT_BYTES = 72744
TRANSPORT_SHA256 = "6dd3633441a21004f5f7299a0806a4336bc8cca4befbe0164a32dd465215975b"
INVENTORY = {
    "policy-v1.json": "3ac58f780e7d65db29f38367c9e22414af50cdad78948ddfaae05b44bc873e69",
    "attempt-v1.json": "2a22f0bdb25ce9cce4e72f819cdc6a22f81f856ec10659852eae67c5d3141ec2",
    "outcome-v1.json": "64e770071f83fad454607d243feec434dc9615ccf783dc40adff14df921ae148",
    "evaluated-source-v1.zip": "d3db4cce394f8bc5e470a5d69f02bcf2d42a634a3ac13da07a0316bdf9dd7632",
    "evaluate_once.py": "5654b316e450de155e5de5e068007b190937d33b5b83b0dc352ec0f835314bf8",
}

# Explicit fixture sharing: no old test functions are collected into this owner.
# The inherited evidence fixture reconstructs all three complete levels once.
retained = cubic_audit.retained
evidence = cubic_audit.evidence
prior_no_execution = cubic_audit.no_coefficient_response_or_worker_execution


@pytest.fixture(scope="module", autouse=True)
def no_spatial_calculation():
    from tnfr.physics import relational_sine_class_spatial_observation as owner

    def forbidden(*args, **kwargs):
        pytest.fail("retained spatial evidence must not regenerate a projection")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "bound_sine_class_spatial_observation", forbidden)
        patch.setattr(owner, "_class_cubic_coefficients", forbidden)
        yield


@pytest.fixture(scope="module")
def spatial_retained():
    receipt = file_receipt(ARCHIVE, max_bytes=2**20)
    assert receipt["bytes"] == TRANSPORT_BYTES and receipt["sha256"] == TRANSPORT_SHA256
    verify_archive_members(ARCHIVE, INVENTORY, max_bytes=2**20)
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = {name: archive.read(name) for name in INVENTORY}
    decoded = {
        name: decode_exact_tree(json_loads(value))
        for name, value in raw.items()
        if name.endswith(".json")
    }
    return raw, decoded


def test_exact_archive_attempt_policy_and_reference_association(
    spatial_retained, retained
):
    raw, decoded = spatial_retained
    prior_raw, _ = retained
    policy, attempt, outcome = (
        decoded[name]
        for name in ("policy-v1.json", "attempt-v1.json", "outcome-v1.json")
    )
    assert policy["schema"] == "tnfr.sine-class-spatial-observation-policy.v1"
    assert policy["status_at_declaration"] == "projection_not_evaluated"
    assert (
        policy["base_revision_before_implementation"]
        == "366fd72aca1f2858b5460e45af96800a115ec01e"
    )
    assert policy["adaptive_retries"] is False
    for key, expected in (
        ("reading_count", 16),
        ("amplitude_degree", 2),
        ("gamma_power", 3),
        ("time_polynomial_order", 64),
        ("interval_bits", 128),
        ("full_coordinates_per_level", 54),
        ("retained_amplitude_levels", 3),
        ("segment_count", 8),
    ):
        assert cubic_audit._integer(policy[key]) == expected
    assert tuple(map(cubic_audit._integer, policy["tail_first_omitted_indices"])) == (
        65,
        64,
        63,
    )
    assert exact_record(policy["linear_norm_bound"]) == Q(201, 100)
    assert exact_record(policy["cauchy_gamma_upper_bound"]) == Q(1, 3000)
    assert attempt["schema"] == "tnfr.sine-class-spatial-observation-attempt.v1"
    assert datetime.fromisoformat(attempt["started_utc"]).tzinfo is not None
    assert attempt["python"] and attempt["platform"]
    assert outcome["schema"] == "tnfr.sine-class-spatial-observation-outcome.v1"
    assert outcome["error"] is None
    assert outcome["reused_field_equality"] == {
        "class_parameters": True,
        "class_segments": True,
    }
    assert all(
        type(value) is bool for value in outcome["reused_field_equality"].values()
    )
    for receipt, name in (
        (attempt["policy"], "policy-v1.json"),
        (attempt["evaluated_source_archive"], "evaluated-source-v1.zip"),
        (outcome["attempt"], "attempt-v1.json"),
    ):
        assert cubic_audit._integer(receipt["bytes"]) == len(raw[name])
        assert receipt["sha256"] == sha256_bytes(raw[name])
    expected = {
        name: receipt["sha256"]
        for name, receipt in attempt["evaluated_sources"].items()
    }
    expected.update(
        {
            name: sha256_bytes(raw[name])
            for name in ("policy-v1.json", "evaluate_once.py")
        }
    )
    verify_archive_members(
        io.BytesIO(raw["evaluated-source-v1.zip"]), expected, max_bytes=2**20
    )
    with zipfile.ZipFile(io.BytesIO(raw["evaluated-source-v1.zip"])) as archive:
        for name, receipt in attempt["evaluated_sources"].items():
            assert cubic_audit._integer(receipt["bytes"]) == len(archive.read(name))
        name = "theory/nodal/SINE_CLASS_SPATIAL_OBSERVATION.md"
        # Only maintained-owner newline conventions may vary after checkout.
        prospective = archive.read(name).replace(b"\r\n", b"\n")
        assert (
            (ROOT / name).read_bytes().replace(b"\r\n", b"\n").startswith(prospective)
        )
    reference = policy["coefficient_reference"]
    assert reference == outcome["output"]["coefficient_evidence_reference"]
    assert (
        reference["path"]
        == "docs/assets/sine_formed_classes/class-cubic-response-v1.evidence.zip"
    )
    assert reference["sha256"] == cubic_audit.TRANSPORT_SHA256
    assert reference["outcome_member"] == "outcome-v1.json"
    assert reference["outcome_sha256"] == sha256_bytes(prior_raw["outcome-v1.json"])
    assert reference["reused_report_fields"] == ("class_parameters", "class_segments")
    assert attempt["coefficient_reference"]["sha256"] == cubic_audit.TRANSPORT_SHA256
    assert (
        cubic_audit._integer(attempt["coefficient_reference"]["bytes"])
        == cubic_audit.TRANSPORT_BYTES
    )


@pytest.fixture(scope="module")
def projection(spatial_retained, evidence):
    _, decoded = spatial_retained
    policy = decoded["policy-v1.json"]
    output = decoded["outcome-v1.json"]["output"]
    assert output["schema"] == "tnfr.sine-class-spatial-observation-compact.v1"
    report = output["report"]
    assert "class_parameters" not in report and "class_segments" not in report
    inputs, prior_report, gamma, _ = evidence
    assert {
        key: exact_record(value) for key, value in policy["inputs"].items()
    } == inputs
    assert {key: exact_record(report[key]) for key in inputs} == inputs
    assert cubic_audit._interval(report["gamma_bounds"]) == gamma
    weights = tuple(tuple(edge) for edge in report["observation_node_weights"])
    assert all(type(value) is int for pair in weights for value in pair)
    assert weights == ((23, 1), (21, -1)) == policy["observation_node_weights"]
    for key, value in (
        ("reading_count", 16),
        ("coefficient_amplitude_degree", 2),
        ("coefficient_scale_gamma_power", 3),
        ("time_polynomial_order", 64),
    ):
        assert cubic_audit._integer(report[key]) == value
    for key in (
        "compared_classes",
        "nodes",
        "edges",
        "contacts",
        "clock",
        "interval_method",
        "linear_majorant_rate",
        "cauchy_gamma_upper_bound",
    ):
        assert report[key] == prior_report[key]
    assert (
        report["method"]
        == "receiver_odd_quadratic_projection_of_complete_amplitude_time_polynomial64_v1"
    )
    class_values = []
    for _, first, both, second in prior_report["class_segments"]:

        def observation(segment):
            row = segment["endpoint_levels"][1]
            return cubic_audit._interval(row[23]) - cubic_audit._interval(row[21])

        class_values.append(
            observation(both) - observation(first) - observation(second)
        )

    def scale(interval):
        products = tuple(
            g**3 * v for g in (gamma.lo, gamma.hi) for v in (interval.lo, interval.hi)
        )
        return min(products), max(products)

    classes = tuple(map(scale, class_values))
    contrast = scale(class_values[0] - class_values[1])
    assert (
        tuple(map(cubic_audit._pair, report["scaled_class_quadratic_bounds"]))
        == classes
    )
    assert cubic_audit._pair(report["complete_quadratic_contrast_bounds"]) == contrast
    return inputs, report, gamma, contrast, prior_report


def test_complete_prior_coordinates_rebuild_the_new_projection(projection):
    _, report, _, (lower, upper), _ = projection
    assert Q(7951, 10**33) < lower <= upper < Q(7952, 10**33)
    assert upper - lower < Q(1, 10**40)
    assert report["coefficient_evaluated"] is True
    assert report["exact_contrast_zero"] is False


def test_cross_class_tail_source_and_sixteen_reading_decisions(projection):
    inputs, report, gamma, contrast, _ = projection
    a, b, t = (
        inputs[key]
        for key in ("first_probe_amplitude", "second_probe_amplitude", "total_duration")
    )
    eps, delta, g = inputs["endpoint_radius"], inputs["readout_error_bound"], Q(1, 3000)
    assert exact_record(report["cauchy_bootstrap_margin"]) == 1 - 16 * g * g * t * t > 0
    assert (
        exact_record(report["cauchy_radius_margin"])
        == 1 - 2 * g * (abs(a) + abs(b))
        > 0
    )
    assert report["cauchy_admitted"] is True
    remainder = tuple(
        Q(4096, 3) * g**7 * A**4 * t**3 / (1 - 4 * g * g * A * A)
        for A in (Q(0), abs(a), abs(b), abs(a) + abs(b))
    )
    higher, source = sum(remainder), 16 * eps / (1 - 2 * gamma.hi * t)
    assert (
        tuple(
            map(
                exact_record,
                report["per_history_higher_amplitude_contrast_remainder_upper_bounds"],
            )
        )
        == remainder
    )
    assert exact_record(report["higher_amplitude_contrast_error_upper_bound"]) == higher
    assert exact_record(report["source_contrast_error_upper_bound"]) == source
    true = contrast[0] - higher - source, contrast[1] + higher + source
    recorded = true[0] - 16 * delta, true[1] + 16 * delta
    decision = report["decision"]
    assert cubic_audit._pair(decision["true_bounds"]) == true
    assert cubic_audit._pair(decision["recorded_bounds"]) == recorded
    lower = true[0]
    assert type(decision["orientation"]) is int and decision["orientation"] == 1
    for key, value in (
        ("oriented_lower", lower),
        ("recorded_sign_margin", lower - 16 * delta),
        ("null_separation_margin", lower - 32 * delta),
        ("noise_ceiling", lower / 16),
        ("cancellation_margin", 16 * delta - max(map(abs, true))),
    ):
        assert exact_record(decision[key]) == value
    for key, value in (
        ("true_sign", lower > 0),
        ("recorded_sign", lower > 16 * delta),
        ("null_excluded", lower > 32 * delta),
        ("scalar_cancellation", max(map(abs, true)) <= 16 * delta),
    ):
        assert decision[key] is value
    assert report["status"] == decision["status"] == "true_sign_certified"
    assert (
        report["response_bound_available"] is True
        and report["unavailable_reasons"] == ()
    )
    assert Q(7785, 10**33) < true[0] <= true[1] < Q(8117, 10**33)
    assert recorded[0] < 0 < recorded[1]
    assert max(map(abs, true)) / 16 <= delta


def test_uniform_event_ledger_is_unchanged_and_independently_readmitted(projection):
    inputs, report, gamma, _, prior_report = projection
    assert report["uniform_history_bounds"] == prior_report["uniform_history_bounds"]
    cubic_audit._audit_work_and_identity(inputs, report, gamma)
