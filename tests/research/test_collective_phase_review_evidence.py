"""Read-only verification of the separate retained IL phase-accuracy review.

No producer or corrective evaluator is imported. The old response remains the
only state input, and the original 8/9 report remains an unsuccessful frozen
certificate even when its numerical subclaim is resolved by separate evidence.
"""

from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import pytest

from tests.research import test_collective_pressure_handoff_evidence as original
from tnfr.mathematics._rational_interval import I, arg, cos, sin
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.phase_reset_certificate import bound_phase_reset
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/assets/collective_pressure_handoff"
REVIEW = BASE / "phase_review_v1"
BUDGET = Q(1, 2**46)


def _read(name):
    validate_artifact_path(name)
    return decode_exact_tree(
        json_loads(read_bytes_bounded(REVIEW / name, max_bytes=32 * 1024**2))
    )


@pytest.fixture(scope="module")
def review():
    paths = tuple(sorted(path for path in BASE.rglob("*") if path.is_file()))
    before = tuple(file_receipt(path) for path in paths)
    records = {
        name: _read(name)
        for name in (
            "protocol.json",
            "freeze.json",
            "source-manifest.json",
            "attempt.json",
            "response.json",
            "report.json",
        )
    }
    old_response = original._read("response.json")
    old = original._rebuild(old_response["captures"]["initial"])
    new = original._rebuild(old_response["captures"]["IL_raw"])
    neighbors = tuple(
        tuple(
            right if left == i else left
            for left, right in old.edges
            if i in (left, right)
        )
        for i in old.nodes
    )
    certificates = bound_phase_reset(
        old.phase,
        new.phase,
        neighbors,
        (0, 1),
        coefficient=exact_record(old_response["settings"]["phase_coefficient"]),
        budget=BUDGET,
    )
    yield records, old_response, old, new, neighbors, certificates
    assert paths == tuple(sorted(path for path in BASE.rglob("*") if path.is_file()))
    assert before == tuple(file_receipt(path) for path in paths)


def test_original_assets_and_corrective_freeze_retain_exact_byte_associations(review):
    records, *_ = review
    freeze = records["freeze.json"]
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert len(freeze["prior_artifacts"]) == 22
    for label, directory in (("prior_artifacts", BASE), ("artifacts", REVIEW)):
        for receipt in freeze[label]:
            name = validate_artifact_path(receipt["path"])
            assert file_receipt(directory / name, label=name) == receipt
    manifest = records["source-manifest.json"]["files"]
    verify_archive_members(
        REVIEW / "source.zip", {row["path"]: row["sha256"] for row in manifest}
    )
    assert records["report.json"]["response"] == file_receipt(
        REVIEW / "response.json", label="response.json"
    )
    freeze_hash = file_receipt(REVIEW / "freeze.json")["sha256"]
    assert records["attempt.json"]["freeze_sha256"] == freeze_hash
    assert records["report.json"]["freeze_sha256"] == freeze_hash


def test_protocol_keeps_original_budget_and_does_not_claim_a_new_prediction(review):
    records, *_ = review
    protocol = records["protocol.json"]
    assert protocol["status_at_protocol_creation"] == "not_evaluated"
    assert protocol["interval_bits"] == 128
    assert protocol["cross_check_decimal_digits"] == 100
    assert exact_record(protocol["absolute_residual_allowance"]) == BUDGET
    assert protocol["original_response"] == file_receipt(
        BASE / "correction_v2/response.json", label="correction_v2/response.json"
    )
    result = records["response.json"]
    assert (
        result["operator_words_executed"] == result["numerical_trajectory_steps"] == 0
    )
    assert result["original_report_reclassified"] is False
    old_report = original._read("report.json")
    assert sum(row["passed"] for row in old_report["checks"]) == 8
    assert len(old_report["checks"]) == 9
    assert old_report["all_checks_passed"] is False


def test_corrective_bounds_rebuild_from_primitives_and_retain_nearest_lifts(review):
    records, _, old, new, neighbors, certificates = review
    response = records["response.json"]
    assert response["before_phase"] == old.phase
    assert response["after_phase"] == new.phase
    assert response["neighbors"] == neighbors
    assert response["edges"] == old.edges == new.edges
    for record, certificate in zip(response["certificates"], certificates):
        assert record["target"] == certificate.target
        for field in (
            "relative_real",
            "relative_imaginary",
            "ideal_angle",
            "ideal_displacement",
            "represented_displacement",
            "defect",
        ):
            value = record[field]
            assert I(exact_record(value["lo"]), exact_record(value["hi"])) == getattr(
                certificate, field
            )
        assert (
            certificate.nearest_lift_certified
            is record["nearest_lift_certified"]
            is True
        )
        assert certificate.defect.abs_max <= certificate.budget == BUDGET
        assert (
            certificate.classification
            == record["classification"]
            == "certified_within_budget"
        )
        assert certificate.defect.width < Q(1, 2**120)
    assert records["report.json"]["phase_accuracy_certified"] is True


def test_lower_precision_inputs_explain_old_unresolved_boxes_without_changing_them(
    review,
):
    _, response, old, new, _, certificates = review
    coefficient = exact_record(response["settings"]["phase_coefficient"])
    for i, certificate in enumerate(certificates):
        change = arg(
            cos(I(new.phase[i] - old.phase[i])), sin(I(new.phase[i] - old.phase[i]))
        )
        inherited = change - coefficient * arg(*old.relative_resultant[i])
        assert inherited == original._interval(response["IL"]["phase_defects"][i])
        assert inherited.contains(0) and inherited.abs_max > BUDGET
        assert inherited.contains(certificate.defect)
        assert not certificate.defect.contains(0)


def test_independent_100_digit_sine_cosine_reduction_lies_inside_each_new_box(review):
    records, response, old, new, neighbors, certificates = review
    coefficient = exact_record(response["settings"]["phase_coefficient"])
    with mp.workdps(100):

        def number(value):
            return mp.mpf(value.numerator) / value.denominator

        for certificate, saved in zip(
            certificates, records["response.json"]["cross_checks"]
        ):
            i = certificate.target
            gaps = tuple(number(old.phase[j] - old.phase[i]) for j in neighbors[i])
            ideal = number(coefficient) * mp.atan2(
                mp.fsum(mp.sin(gap) for gap in gaps),
                mp.fsum(mp.cos(gap) for gap in gaps),
            )
            change = number(new.phase[i] - old.phase[i])
            point = Q(mp.nstr(mp.atan2(mp.sin(change), mp.cos(change)) - ideal, 100))
            assert certificate.defect.contains(point)
            assert certificate.defect.contains(Q(saved["defect_decimal"]))
            assert saved["inside_rigorous_box"] is True


def test_later_claims_remain_distinct_and_rebuild_from_the_original_response(review):
    records, response, *_ = review
    fields = {
        name: original._rebuild(capture)
        for name, capture in response["captures"].items()
    }
    retained = {"response.json": response}
    original.test_il_work_and_identity_follow_from_all_retained_edges(retained, fields)
    original.test_oz_projection_and_all_candidate_products_retain_their_separate_roles(
        retained, fields
    )
    assert all(
        row["passed"]
        for row in records["response.json"]["independent_retained_state_checks"]
    )
    assert all(records["response.json"]["later_original_compound_boxes"].values())
