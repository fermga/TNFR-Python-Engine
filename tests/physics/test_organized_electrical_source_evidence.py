"""Read-only reconstruction of the retained collective electrical comparison.

Archive and Git associations preserve declared provenance; they neither
authenticate execution nor replace the model hypotheses. No assessor or
scientific producer is imported or invoked by these tests.
"""

import hashlib
import re
import subprocess
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_electrical_source"
DECLARATION = "theory/nodal/ORGANIZED_ELECTRICAL_SOURCE.md"
EVALUATOR = "benchmarks/organized_electrical_source.py"
SOURCE = "src/tnfr/physics/_organized_electrical_source.py"
REVISION = "10bf3e1003d2bb864d8bbbb9f2e2900957923c39"


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


def _laplacian(size, edges):
    matrix = [[0] * size for _ in range(size)]
    for left, right in edges:
        matrix[left][left] += 1
        matrix[right][right] += 1
        matrix[left][right] -= 1
        matrix[right][left] -= 1
    return matrix


def test_exact_receipts_inventory_and_prospective_committed_source(records):
    attempt, result = records
    revision = attempt["source_revision"]
    assert re.fullmatch("[0-9a-f]{40}", revision)
    assert result["source_revision"] == revision == REVISION
    assert attempt["schema"] == "organized_electrical_source_attempt_v1"
    assert result["schema"] == "organized_electrical_source_result_v1"
    assert attempt["declaration"] == DECLARATION and attempt["evaluator"] == EVALUATOR
    assert attempt["source_scope"] == (
        "src/tnfr",
        EVALUATOR,
        DECLARATION,
        "pyproject.toml",
    )
    raw = (EVIDENCE / "attempt.json").read_bytes()
    assert result["attempt"] == dict(
        path="attempt.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["interval_bits"] == 128
    assert (
        attempt["arithmetic_method"]
        == "rational_outward_dyadic128_machin_trigonometry_v1"
    )
    assert {path.name for path in EVIDENCE.glob("*.json")} == {
        "attempt.json",
        "result.json",
    }
    try:
        _git("cat-file", "-e", revision + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("retained source commit unavailable in this checkout")
    listing = _git(
        "ls-tree", "-r", "-z", "--full-tree", revision, "--", *attempt["source_scope"]
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
    assert {SOURCE, DECLARATION, EVALUATOR}.issubset(inventory)
    assert set(attempt["imported_sources"].values()) <= set(inventory)
    assert set(result["imported_sources"].values()) <= set(inventory)
    assert (
        result["imported_sources"]["tnfr.physics._organized_electrical_source"]
        == SOURCE
    )
    assert (
        _git("rev-parse", revision + ":src").decode().strip() == attempt["source_tree"]
    )
    declaration = _git("show", revision + ":" + DECLARATION).decode("utf-8")
    for fragment in (
        "Prospective source and decision declaration",
        "q=1e-8",
        "eta=1e-9",
        "h=1e-3",
        "decay power 80",
        "One analytical assessment",
        "No trajectory",
        "initialized at those actual collective means",
    ):
        assert fragment in declaration
    assessor = _git("show", revision + ":" + SOURCE).decode("utf-8")
    assert "def _assess_organized_electrical_source()" in assessor
    assert "_assess_organized_joint_resources(" in assessor


def test_original_preparation_word_support_and_spectral_budget_are_retained(records):
    report = records[1]["report"]
    source, q, h = report["source"], Q(1, 10**8), Q(1, 1000)
    assert (
        source["form_error_bounds"]
        == source["phase_error_bounds"]
        == (Q(1, 10**6),) * 6
    )
    assert source["quarter_dwell"] == 8 * 10**10 and source["decay_power"] == 80
    assert source["recovered_radius"] == q
    assert source["sequential_readout_error"] == Q(1, 10**14)
    assert source["contact_readout_error"] == Q(1, 10**9)
    assert source["contact_horizon"] == report["contact_horizon"] == h
    assert report["source_families"] == ("outer_zero", "outer_one")
    assert report["slot_offsets"] == (0,) * 5
    assert report["node_order"] == tuple(range(54))
    assert report["phase_template"] == "2*pi*(j-4)/9"
    observation = source["observation"]
    assert observation["marked_child_order"] == tuple(range(5))
    assert observation["receiver_child_order"] == (1, 2, 3, 4, 0)
    assert observation["event_multipliers"] == (1, -2, 1)
    assert observation["window_command_multipliers"] == (0, 1, -1, 0)
    assert observation["probe_amplitude"] == Q(1, 10)
    assert observation["elapsed_window_scaled_time"] == Q(1, 10**5)
    assert observation["elapsed_block_scaled_time"] == Q(1, 25000)
    assert observation["total_elapsed_scaled_time"] == Q(1, 5000)
    host = source["outer_resources"]["geometry"]
    host_edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    host_edges += tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    assert host["weighted_edges"] == host_edges
    assert host["held_capacities"] == (Q(1),) * 45
    assert host["clock_factor"] == Q(1, 4) and host["phase_domain"] == "regular"
    assert host["internal_cycles"] == tuple(
        tuple(9 * c + j for j in range(9)) for c in range(5)
    )
    assert host["outer_cycles"] == tuple(
        tuple(9 * c + j for c in range(5)) for j in range(9)
    )
    gamma = Q(1, 1023) / pi_interval()
    assert I(**host["gamma_bounds"]) == I(**observation["gamma_bounds"]) == gamma
    cycle = tuple((j, (j + 1) % 9) for j in range(9))
    wheel = tuple((c, (c + 1) % 5) for c in range(5)) + tuple((c, 5) for c in range(5))
    l9, lw = _laplacian(9, cycle), _laplacian(6, wheel)
    complement = tuple(
        (a, b)
        for a in range(6)
        for b in range(a + 1, 6)
        if (a, b) not in wheel and (b, a) not in wheel
    )
    lc = _laplacian(6, complement)
    signless = [[0] * 9 for _ in range(9)]
    for a, b in cycle:
        for i in (a, b):
            for j in (a, b):
                signless[i][j] += 1
    # Exact PSD witnesses: 4I-L_C9 is a sum of (e_a+e_b)(...)^T;
    # 6I-L_W is J plus the complement Laplacian. Their tensor sum
    # proves the declared contact upper bound ten without numerical spectra.
    assert all(
        4 * int(i == j) - l9[i][j] == signless[i][j] for i in range(9) for j in range(9)
    )
    assert all(
        6 * int(i == j) - lw[i][j] == 1 + lc[i][j] for i in range(6) for j in range(6)
    )
    assert report["observation_hidden"]["laplacian_upper"] == 8
    assert report["contact_hidden"]["laplacian_upper"] == 10


def test_hidden_bounds_reconstruct_from_source_and_satisfy_coupled_inequalities(
    records,
):
    report = records[1]["report"]
    source = report["source"]
    q = source["recovered_radius"]
    g = (Q(1, 1023) / pi_interval()).hi
    z0, a0 = 6 * q * (1 + g), 3 * q + 24 * g * (6 * q * (1 + g))
    outer = source["outer_resources"]
    assert outer["full_form_error_upper_bound"] == a0
    assert outer["full_phase_error_upper_bound"] == z0
    for name, duration, laplacian in (
        ("observation_hidden", Q(1, 5000), 8),
        ("contact_hidden", Q(1, 1000), 10),
    ):
        stage = report[name]
        assert stage["initial_form"] == a0 and stage["initial_phase"] == z0
        assert stage["gamma_upper"] == g
        assert stage["duration"] == duration and stage["laplacian_upper"] == laplacian
        coupling = g * laplacian * duration / 4
        assert stage["coupling"] == coupling < 1
        assert stage["domain_margin"] == 1 - coupling**2 > 0
        a, z = stage["form_upper"], stage["phase_upper"]
        assert a >= a0 and z >= z0
        assert a - coupling * z == a0
        assert z - coupling * a == z0
        a0, z0 = a + q, z + q


def test_circular_collar_covariance_and_uniform_comparison_rebuild(records):
    report = records[1]["report"]
    g, h = (Q(1, 1023) / pi_interval()).hi, report["contact_horizon"]
    z = report["contact_hidden"]["phase_upper"]
    assert report["collar_margin"] == pi_interval().lo / 2 - 2 * z > 0
    assert 9 * (1 - z**2 / 2) > 0  # Independent nonzero-resultant lower witness.
    rho = g * z**2 / 6
    assert report["closure_l1_upper"] == rho > 0
    assert report["regional_closure_upper"] == rho / 2
    coupling = 3 * g * h / 2
    assert report["comparison_domain_margin"] == 1 - coupling**2 > 0
    ex, ep = report["mean_form_error_upper"], report["mean_phase_error_upper"]
    assert ex - coupling * ep == h * rho
    assert ep == coupling * ex
    assert 0 < ep < ex < Q(1, 10**9)
    assert report["form_resolution_margin"] == Q(1, 10**9) - ex > 0


def test_signed_storage_hidden_loss_and_work_remain_separate(records):
    report = records[1]["report"]
    source, h = report["source"], report["contact_horizon"]
    a, z = (
        report["contact_hidden"]["form_upper"],
        report["contact_hidden"]["phase_upper"],
    )
    g = (Q(1, 1023) / pi_interval()).hi
    form_bound = max(Q(1, 100) + g / 5000, source["recovered_radius"]) + 7 * g * h / 4
    assert source["contact_form_upper_bound"] == form_bound
    assert report["storage_remainder_lower"] == -5 * z**2 < 0
    assert report["storage_remainder_upper"] == 5 * (a**2 + z**2) > 0
    loss = 25 * a**2
    work = 10 * form_bound * report["closure_l1_upper"]
    assert report["hidden_loss_rate_upper"] == loss > 0
    assert report["integrated_hidden_loss_upper"] == h * loss
    assert report["collective_hidden_work_rate_upper"] == work > 0
    assert report["integrated_collective_hidden_work_upper"] == h * work
    assert source["contact_work_upper_bound"] < 100
    assert all(value < 200 for value in source["joined_storage_upper_bounds"])
    assert "signed_hidden_storage_and_separate_hidden_loss_and_work" in report["scope"]
    assert (
        "fixed_postcontact_support_only_no_electrical_switching_certificate"
        in report["scope"]
    )


def test_admission_keeps_every_actual_history_premise_and_excludes_old_response_verdict(
    records,
):
    report = records[1]["report"]
    source, flags = report["source"], dict(report["prerequisite_flags"])
    assert len(flags) == len(report["prerequisite_flags"])
    assert set(flags) == {
        "six_actual_recoveries",
        "outer_one_capture",
        "outer_zero_capture",
        "outer_attachment_budget",
        "sequential_observation",
        "same_slot_product_support",
        "complete_observation_support",
        "complete_contact_law",
        "complete_observation_law",
        "complete_observation_word",
        "complete_history_margin_inventory",
    }
    assert all(value is True for value in flags.values())
    assert len(source["recoveries"]) == 6
    assert all(
        recovery["recovery_certified"] is True for recovery in source["recoveries"]
    )
    outer = source["outer_resources"]
    assert outer["conditional_capture_certified"] is True
    assert outer["control_conditional_capture_certified"] is True
    assert outer["attachment_budget_certified"] is True
    assert source["observation"]["admitted"] is True
    old_responses = {
        "outer_one_all_origin_response",
        "outer_zero_witness_response",
        "fixed_origin_response_contrast",
    }
    margins = tuple(
        (name, value)
        for name, value in source["sufficient_margins"]
        if name not in old_responses
    )
    assert report["prerequisite_margins"] == margins
    assert len(dict(margins)) == len(margins) == 25
    assert all(type(value) is Q and value > 0 for _, value in margins)
    assert report["admitted"] is True and report["unresolved_conditions"] == ()
    assert records[1]["verdict"] == "finite_collective_electrical_source_admitted"
