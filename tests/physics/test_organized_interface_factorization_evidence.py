"""Reconstruct retained support and finite rank arithmetic without assessment.

Exact archive associations do not authenticate execution or replace the
source, counterfactual-preparation and complete-law hypotheses of the proof.
"""

import hashlib
import subprocess
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_interface_factorization"
REVISION = "9548b8e178d499be4a4cd9a8ae1bd8ff96ae9a02"
REGISTRATIONS = ((0,) * 5, (5, 7, 0, 2, 4))


@pytest.fixture(scope="module")
def records():
    with forbid_sine_regeneration():
        return tuple(
            decode_exact_tree(json_loads((EVIDENCE / name).read_text(encoding="utf-8")))
            for name in ("attempt.json", "result.json")
        )


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


def _interval(value):
    return I(**value)


def _record(value):
    return dict(lo=value.lo, hi=value.hi)


def test_exact_receipt_source_inventory_and_committed_declaration(records):
    attempt, result = records
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    raw = (EVIDENCE / "attempt.json").read_bytes()
    assert result["attempt"] == dict(
        path="attempt.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert (
        attempt["parameters"] == {}
        and attempt["evaluation_status_at_start"] == "not_evaluated"
    )
    assert not (EVIDENCE / "incomplete.json").exists()
    try:
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("frozen Git revision is unavailable in this checkout")
    listing = _git(
        "ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *attempt["source_scope"]
    )
    assert hashlib.sha256(listing).hexdigest() == attempt["source_inventory_sha256"]
    inventory = {}
    for entry in listing.split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, kind, blob = metadata.decode("ascii").split()
            assert kind == "blob"
            inventory[path.decode("utf-8")] = dict(mode=mode, blob=blob)
    assert inventory == attempt["source_inventory"]
    assert set(result["imported_sources"].values()) <= set(inventory)
    assert (
        _git("rev-parse", REVISION + ":src").decode().strip() == attempt["source_tree"]
    )
    declaration = _git("show", REVISION + ":" + attempt["declaration"]).decode("utf-8")
    for fragment in (
        "(5,7,0,2,4)",
        "h=1e-3",
        "eta=1e-9",
        "[-4*eta,4*eta]",
        "threshold exactly zero",
    ):
        assert fragment in declaration


def test_both_recorded_supports_have_the_declared_bijections_and_old_cycles(records):
    report = records[1]["report"]
    assert report["slot_registrations"] == REGISTRATIONS
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(6) for j in range(9)
    )
    outer = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    for geometry, slots in zip(report["geometries"], REGISTRATIONS):
        contact = tuple(
            (9 * c + j, 45 + (j + slots[c]) % 9, Q(1))
            for c in range(5)
            for j in range(9)
        )
        edges = internal + outer + contact
        assert geometry["weighted_edges"] == edges and len(set(edges)) == 144
        assert geometry["contact_edges"] == contact and len(contact) == 45
        for child in range(5):
            assert {b for a, b, _ in contact if a // 9 == child} == set(range(45, 54))
        degrees = tuple(sum(node in (a, b) for a, b, _ in edges) for node in range(54))
        assert degrees == geometry["support_degrees"] == (5,) * 45 + (7,) * 9
        assert geometry["held_capacities"] == (Q(1),) * 54
        assert (
            geometry["clock_factor"] == Q(1, 4)
            and geometry["phase_domain"] == "regular"
        )
        assert geometry["internal_cycles"] == tuple(
            tuple(9 * c + j for j in range(9)) for c in range(6)
        )
        assert geometry["outer_cycles"] == tuple(
            tuple(9 * c + j for c in range(5)) for j in range(9)
        )


def test_phase_boxes_and_signed_moments_rebuild_from_retained_source(records):
    report = records[1]["report"]
    source = report["source"]
    z = (
        source["outer_resources"]["full_phase_error_upper_bound"]
        + source["observation_phase_drift_upper_bound"]
    )
    error, donor = I(-z, z), I(-source["recovered_radius"], source["recovered_radius"])
    origins = (
        (I(0),) * 5,
        tuple(
            map(
                _interval,
                source["outer_resources"]["primitive_entry"]["endpoint_phase_bounds"],
            )
        ),
    )
    alpha = 2 * pi_interval() / 9
    for row, origin in enumerate(origins):
        phase = (
            tuple(value + error for value in origin for _ in range(9)) + (donor,) * 9
        )
        assert report["source_phase_offset_bounds"][row] == tuple(map(_record, phase))
        for column, slots in enumerate(REGISTRATIONS):
            gaps = tuple(
                alpha * (((j + slots[c]) % 9) - j)
                + phase[45 + (j + slots[c]) % 9]
                - phase[9 * c + j]
                for c in range(5)
                for j in range(9)
            )
            cosine = (45 - sum((1 - cos(gap) for gap in gaps), I(0))) / 9
            sine = sum((sin(gap) for gap in gaps), I(0)) / 9
            assert report["cosine_moment_matrix"][row][column] == _record(cosine)
            assert report["sine_moment_matrix"][row][column] == _record(sine)


def test_true_observed_and_error_corrected_determinants_rebuild_separately(records):
    report = records[1]["report"]
    source, eta, h = report["source"], Q(1, 10**9), Q(1, 1000)
    gamma = Q(1, 1023) / pi_interval()
    b = (
        max(Q(1, 100) + gamma.hi / 5000, source["recovered_radius"])
        + Q(7, 4) * gamma.hi * h
    )
    remainder = Q(15, 16) * gamma.hi * h**2 + Q(15, 4) * gamma.hi**2 * b * h**2
    assert report["finite_response_remainder_upper_bound"] == remainder
    assert report["raw_readout_error_upper_bound"] == eta
    assert report["increment_readout_error_upper_bound"] == 2 * eta
    assert report["rectification_allowance_upper_bound"] == 4 * eta
    factor = gamma * h / 4
    true = tuple(
        tuple(-factor * _interval(value) + I(-remainder, remainder) for value in row)
        for row in report["cosine_moment_matrix"]
    )
    for extra, matrix_name, determinant_name in (
        (Q(0), "true_response_matrix", "true_determinant_bounds"),
        (2 * eta, "observed_response_matrix", "observed_determinant_bounds"),
        (4 * eta, "rectification_matrix", "determinant_bounds"),
    ):
        matrix = tuple(tuple(value + I(-extra, extra) for value in row) for row in true)
        assert report[matrix_name] == tuple(tuple(map(_record, row)) for row in matrix)
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        assert report[determinant_name] == _record(determinant)
        assert not determinant.contains(0)
        if extra != 2 * eta:
            label = "true_rank_margin" if extra == 0 else "rectification_rank_margin"
            assert report[label] == max(determinant.lo, -determinant.hi) > 0
    flags = dict(report["prerequisite_flags"])
    assert set(flags) == {
        "six_actual_recoveries",
        "outer_one_capture",
        "outer_zero_capture",
        "outer_attachment_budget",
        "sequential_observation",
        "complete_resource_margin_inventory",
    }
    assert all(flags.values()) and all(report["support_admission_flags"])
    assert flags["six_actual_recoveries"] == all(
        entry["recovery_certified"] for entry in source["recoveries"]
    )
    assert (
        flags["outer_one_capture"]
        == source["outer_resources"]["conditional_capture_certified"]
    )
    assert (
        flags["outer_zero_capture"]
        == source["outer_resources"]["control_conditional_capture_certified"]
    )
    assert (
        flags["outer_attachment_budget"]
        == source["outer_resources"]["attachment_budget_certified"]
    )
    assert flags["sequential_observation"] == source["observation"]["admitted"]
    previous_responses = {
        "outer_one_all_origin_response",
        "outer_zero_witness_response",
        "fixed_origin_response_contrast",
    }
    assert report["prerequisite_margins"] == tuple(
        (name, value)
        for name, value in source["sufficient_margins"]
        if name not in previous_responses
    )
    assert all(value > 0 for _, value in report["prerequisite_margins"])
    assert report["admitted"] and not report["unresolved_conditions"]
    assert records[1]["verdict"] == "finite_scalar_factorization_excluded"
