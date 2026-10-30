"""Read-only reconstruction of the retained finite-input assessment.

Receipt and Git associations preserve provenance, not execution authentication
or mathematical truth. No source assessor or scientific producer is imported.
"""

import hashlib
import subprocess
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_electrical_actuation"
REVISION = "9488f528226679b649724634ff950a917c57fb5f"
DECLARATION = "theory/nodal/ORGANIZED_ELECTRICAL_ACTUATION.md"
EVALUATOR = "benchmarks/organized_electrical_actuation.py"
SOURCE = "src/tnfr/physics/_organized_electrical_actuation.py"


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


def test_single_attempt_receipt_and_prospective_git_inventory(records):
    attempt, result = records
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_electrical_actuation_attempt_v1"
    assert result["schema"] == "organized_electrical_actuation_result_v1"
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
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("retained source commit unavailable in this checkout")
    listing = _git(
        "ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *attempt["source_scope"]
    )
    assert hashlib.sha256(listing).hexdigest() == attempt["source_inventory_sha256"]
    inventory = {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = dict(mode=mode, blob=blob)
    assert inventory == attempt["source_inventory"]
    assert {DECLARATION, EVALUATOR, SOURCE}.issubset(inventory)
    for record in records:
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert (
        result["imported_sources"]["tnfr.physics._organized_electrical_actuation"]
        == SOURCE
    )
    assert (
        _git("rev-parse", REVISION + ":src").decode().strip() == attempt["source_tree"]
    )
    declaration = _git("show", REVISION + ":" + DECLARATION).decode("utf-8")
    for fragment in (
        "Prospective declaration",
        "h0=1e-5",
        "d=1",
        "a=1/10",
        "below `2e8`",
        "below `1e9`",
        "below `3e8`",
        "No trajectory",
        "No donor, later phase event",
        "No actual form or phase reading",
    ):
        assert fragment in declaration


def test_actual_five_child_source_and_only_first_unforced_window(records):
    report = records[1]["report"]
    recovery, outer = report["recovery"], report["outer"]
    assert report["independent_child_count"] == 5
    assert report["source_families"] == ("outer_zero", "outer_one")
    assert report["marked_child"] == 0 and report["amplitude"] == Q(1, 10)
    assert report["duration"] == 1 and report["prefix_duration"] == Q(1, 10**5)
    assert recovery["form_error_bound"] == recovery["phase_error_bound"] == Q(1, 10**6)
    assert recovery["quarter_formation_time"] == 200
    assert recovery["quarter_dwell"] == 8 * 10**10 and recovery["decay_power"] == 80
    assert (
        recovery["endpoint_radius"]
        == outer["recovered_child_norm_bound"]
        == Q(1, 10**8)
    )
    assert recovery["recovery_certified"] is True
    geometry = outer["geometry"]
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    external = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    assert geometry["weighted_edges"] == internal + external
    assert len(geometry["weighted_edges"]) == 90
    assert geometry["node_order"] == tuple(range(45))
    assert geometry["support_degrees"] == (4,) * 45
    assert geometry["held_capacities"] == (Q(1),) * 45
    assert geometry["clock_factor"] == Q(1, 4) and geometry["phase_domain"] == "regular"
    assert geometry["internal_cycles"] == tuple(
        tuple(9 * c + j for j in range(9)) for c in range(5)
    )
    assert geometry["outer_cycles"] == tuple(
        tuple(9 * c + j for c in range(5)) for j in range(9)
    )
    assert (
        "no_remaining_observation_word_inverse_or_support_switching_admission"
        in report["scope"]
    )


def test_prefix_and_pulse_hidden_norms_rebuild_from_original_recovery(records):
    report, gamma = records[1]["report"], Q(1, 1023) / pi_interval()
    outer, pulse = report["outer"], report["pulse"]
    q, g = Q(1, 10**8), gamma.hi
    phase = 6 * q * (1 + g)
    form = 3 * q + 24 * g * phase
    assert outer["full_form_error_upper_bound"] == form
    assert outer["full_phase_error_upper_bound"] == phase
    assert I(**outer["geometry"]["gamma_bounds"]) == I(**pulse["gamma_bounds"]) == gamma
    nominal = max(
        I(**box).abs_max for box in outer["primitive_entry"]["endpoint_form_bounds"]
    )
    mean_gap = max(
        I(**box).abs_max
        for box in outer["primitive_entry"]["capture"]["phase_edge_gap_bounds"]
    )
    entry = nominal + form + g * report["prefix_duration"]
    assert report["nominal_form_upper"] == nominal
    assert report["entry_form_upper"] == pulse["initial_form_bound"] == entry
    assert (
        report["entry_mean_gap_upper"]
        == pulse["outer_gap"]
        == mean_gap + 2 * phase + 4 * g * entry * report["prefix_duration"]
    )
    for stage, duration in (
        (report["prefix_hidden"], report["prefix_duration"]),
        (pulse["hidden"], report["duration"]),
    ):
        assert stage["initial_form"] == form and stage["initial_phase"] == phase
        assert stage["laplacian_upper"] == 8 and stage["gamma_upper"] == g
        assert stage["duration"] == duration
        coupling = 2 * g * duration
        assert stage["coupling"] == coupling < 1
        assert stage["domain_margin"] == 1 - coupling**2 > 0
        next_form, next_phase = stage["form_upper"], stage["phase_upper"]
        # Reconstruct as the unique solution of the coupled majorant equations.
        assert next_form - coupling * next_phase == form
        assert next_phase - coupling * next_form == phase
        form, phase = next_form, next_phase
    assert pulse["initial_hidden_form"] == report["prefix_hidden"]["form_upper"]
    assert pulse["initial_hidden_phase"] == report["prefix_hidden"]["phase_upper"]


def test_live_row_endpoint_identity_and_same_source_comparison(records):
    pulse = records[1]["report"]["pulse"]
    pi, gamma = pi_interval(), I(**pulse["gamma_bounds"])
    a, d, g = abs(pulse["amplitude"]), pulse["duration"], gamma.hi
    residual = pulse["initial_form_bound"] + g * d
    drift = 2 * g * residual * d
    assert pulse["residual_form_upper"] == pulse["endpoint_form_upper"] == residual
    assert pulse["phase_residual_upper"] == drift
    assert pulse["form_change_upper"] == (2 * residual + g) * d
    hidden = pulse["hidden"]["phase_upper"]
    assert pulse["internal_positive_margin"] == 2 * pi.lo / 9 - 2 * hidden > 0
    assert pulse["internal_acute_margin"] == pi.lo / 2 - 2 * pi.hi / 9 - 2 * hidden > 0
    assert pulse["collar_margin"] == pi.lo / 2 - 2 * hidden > 0
    assert (
        pulse["outer_acute_margin"]
        == pi.lo / 2
        - pulse["outer_gap"]
        - a
        - 2 * pulse["initial_hidden_phase"]
        - 2 * drift
        > 0
    )
    ex, ep = (
        pulse["unforced_form_comparison_upper"],
        pulse["unforced_phase_comparison_upper"],
    )
    assert ex - 2 * g * d * ep == g * a * d / 4
    assert ep == 2 * g * d * ex
    assert 0 < ep < ex


def test_full_cost_bounds_reconstruct_from_cubic_extrema_and_actual_phase(records):
    pulse = records[1]["report"]["pulse"]
    gamma = I(**pulse["gamma_bounds"])
    a, d, residual = (
        abs(pulse["amplitude"]),
        pulse["duration"],
        pulse["residual_form_upper"],
    )
    k, inverse = (Q(4, 5),) + (Q(-1, 5),) * 4, (Q(2, 5), 0, Q(-1, 5), Q(-1, 5), 0)

    def action(v):
        return tuple(2 * v[c] - v[(c - 1) % 5] - v[(c + 1) % 5] for c in range(5))

    assert action(inverse) == k and sum(inverse) == sum(k) == 0
    first_max, second_max = Q(3, 2), Q(6)  # s_r'=6r(1-r), s_r''=6-12r.
    offset = 4 * a * max(map(abs, inverse)) * first_max / (gamma.lo * d)
    gradient = 4 * a * max(map(abs, k)) * first_max / (gamma.lo * d) + 8 * residual
    forcing = 4 * a * max(map(abs, inverse)) * second_max / (
        gamma.lo * d**2
    ) + a * max(map(abs, k)) * first_max / (gamma.lo * d)
    assert pulse["form_offset_upper"] == offset
    assert pulse["form_gradient_upper"] == gradient
    assert pulse["input_upper"] == forcing
    assert pulse["peak_storage_upper"] == 90 * ((2 * (offset + residual)) ** 2 / 2 + 2)
    assert pulse["continuous_loss_upper"] == 45 * d * gradient**2 / 4
    assert pulse["absolute_work_upper"] == 45 * d * gradient * forcing
    contrast = max(Q(0), a - 2 * pulse["phase_residual_upper"])
    assert pulse["actual_phase_contrast_lower"] == contrast > 0
    direction = (1, -1, 0, 0, 0)
    norm_squared = sum(v * v for v in direction)
    gradient_squared = sum(v * w for v, w in zip(direction, action(direction)))
    assert norm_squared == 2 and gradient_squared == 6
    assert (
        pulse["continuous_loss_lower"]
        == 36 * contrast**2 / (norm_squared * gamma.hi**2 * d)
        > 0
    )
    assert (
        pulse["peak_form_storage_lower"]
        == 72 * contrast**2 / (gradient_squared * gamma.hi**2 * d**2)
        > 0
    )
    assert pulse["continuous_loss_lower"] < pulse["continuous_loss_upper"]
    assert pulse["peak_form_storage_lower"] < pulse["peak_storage_upper"]
    # H_end >= 0 and W_plus >= W = H_end-H_entry+Loss force new supply;
    # neither prior loss nor signed hidden storage pays this pulse.
    entry_storage_upper = 180 * pulse["initial_form_bound"] ** 2 + 180
    assert entry_storage_upper < 181
    assert pulse["continuous_loss_lower"] - entry_storage_upper > 1858000


def test_every_source_premise_and_frozen_decision_is_retained(records):
    result, report = records[1], records[1]["report"]
    flags = dict(report["prerequisite_flags"])
    assert len(flags) == len(report["prerequisite_flags"]) == 12
    assert set(flags) == {
        "five_independent_child_recoveries",
        "recovered_radius_matches",
        "primitive_law_and_clock",
        "full_fine_comparison",
        "child_transit_identity",
        "outer_one_cell",
        "outer_one_capture",
        "outer_zero_capture",
        "outer_attachment_budget",
        "precontact_product_support",
        "complete_quarter_law",
        "complete_numeric_source_margins",
    }
    assert all(value is True for value in flags.values())
    recovery, outer, pulse = report["recovery"], report["outer"], report["pulse"]
    expected = {
        "actual_recovery_form": recovery["endpoint_form_margin"],
        "actual_recovery_phase": recovery["endpoint_phase_margin"],
        "outer_one_capture": outer["capture_margin"],
        "outer_zero_capture": outer["control_capture_margin"],
        "outer_attachment_work": outer["attachment_work_margin"],
        "outer_initial_storage": outer["initial_storage_margin"],
    }
    assert dict(report["prerequisite_margins"]) == expected
    assert len(report["prerequisite_margins"]) == 6
    assert all(type(value) is Q and value > 0 for value in expected.values())
    policies = (
        ("phase_residual", Q(1, 500000), "phase_residual_upper"),
        ("endpoint_form", Q(1, 500), "endpoint_form_upper"),
        ("form_change", Q(1, 250), "form_change_upper"),
        ("peak_storage", Q(2 * 10**8), "peak_storage_upper"),
        ("continuous_loss", Q(10**9), "continuous_loss_upper"),
        ("absolute_work", Q(3 * 10**8), "absolute_work_upper"),
    )
    expected = {}
    for name, budget, field in policies:
        assert report[name + "_budget"] == budget
        expected["absolute_supplied_work" if name == "absolute_work" else name] = (
            budget - pulse[field]
        )
    for name in ("internal_positive", "internal_acute", "outer_acute"):
        expected[name] = pulse[name + "_margin"]
    expected["circular_collar"] = pulse["collar_margin"]
    assert dict(report["sufficient_margins"]) == expected
    assert len(report["sufficient_margins"]) == 10
    assert all(type(value) is Q and value > 0 for value in expected.values())
    assert report["admitted"] is True and report["unresolved_conditions"] == ()
    assert result["verdict"] == "finite_organized_electrical_actuation_admitted"
