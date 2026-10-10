"""Read-only reconstruction of one frozen finite-input observation decision.

Source receipts retain provenance associations, not execution authentication
or physical evidence. No observation/source assessor or producer is imported.
"""

from fractions import Fraction as Q
import hashlib
from math import prod
from pathlib import Path
import subprocess

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_finite_observation"
REVISION = "0d46c82f46e722618920b6b8ebce0db475f13c64"
BASE = "8f5521be"
DECLARATION = "theory/nodal/ORGANIZED_FINITE_OBSERVATION.md"
EVALUATOR = "benchmarks/organized_finite_observation.py"
SOURCE = "src/tnfr/physics/_organized_finite_observation_source.py"
PURE = "src/tnfr/physics/_organized_finite_observation.py"


@pytest.fixture(scope="module")
def records():
    return tuple(
        decode_exact_tree(json_loads((EVIDENCE / name).read_text(encoding="utf-8")))
        for name in ("attempt.json", "result.json")
    )


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


def test_one_frozen_attempt_and_complete_source_receipt(records):
    attempt, result = records
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_finite_observation_attempt_v1"
    assert result["schema"] == "organized_finite_observation_result_v1"
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert (
        attempt["arithmetic_method"]
        == "rational_outward_dyadic128_machin_trigonometry_v1"
    )
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
    assert {path.name for path in EVIDENCE.glob("*.json")} == {
        "attempt.json",
        "result.json",
    }
    try:
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("retained source revision unavailable in this checkout")
    listing = _git(
        "ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *attempt["source_scope"]
    )
    inventory = {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = dict(mode=mode, blob=blob)
    assert inventory == attempt["source_inventory"]
    assert hashlib.sha256(listing).hexdigest() == attempt["source_inventory_sha256"]
    assert (
        _git("rev-parse", REVISION + ":src").decode().strip() == attempt["source_tree"]
    )
    assert {DECLARATION, EVALUATOR, SOURCE, PURE}.issubset(inventory)
    for record in records:
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert (
        result["imported_sources"]["tnfr.physics._organized_finite_observation_source"]
        == SOURCE
    )
    # The new inverse consumes their algebra; earlier producers remain frozen.
    unchanged = (
        "src/tnfr/physics/_organized_outer_observation.py",
        "src/tnfr/physics/_organized_sequential_observation.py",
        "src/tnfr/physics/_organized_electrical_actuation.py",
        "src/tnfr/physics/_organized_electrical_source.py",
        "benchmarks/organized_electrical_actuation.py",
        "benchmarks/organized_electrical_source.py",
    )
    assert not _git("diff", "--name-only", BASE, REVISION, "--", *unchanged)
    declaration = _git("show", REVISION + ":" + DECLARATION).decode("utf-8")
    assert "150 scalar readings" in declaration
    assert "No trajectory, raw physical data" in declaration
    assert "source-specific electrical error" in declaration


def test_actual_source_inclusion_and_fixed_complete_observation_word(records):
    report = records[1]["report"]
    recovery, outer, protocol = report["recovery"], report["outer"], report["protocol"]
    assert (
        report["source_stage"] == "actual_end_of_outer_formation_before_first_plateau"
    )
    assert report["source_families"] == ("outer_zero", "outer_one")
    assert report["independent_child_count"] == 5
    assert recovery["form_error_bound"] == recovery["phase_error_bound"] == Q(1, 10**6)
    assert recovery["quarter_formation_time"] == 200
    assert recovery["quarter_dwell"] == 8 * 10**10 and recovery["decay_power"] == 80
    q, gamma = Q(1, 10**8), Q(1, 1023) / pi_interval()
    assert recovery["endpoint_radius"] == outer["recovered_child_norm_bound"] == q
    z = 6 * q * (1 + gamma.hi)
    r = 3 * q + 24 * gamma.hi * z
    nominal = max(
        I(**box).abs_max for box in outer["primitive_entry"]["endpoint_form_bounds"]
    )
    gap = max(
        I(**box).abs_max
        for box in outer["primitive_entry"]["capture"]["phase_edge_gap_bounds"]
    )
    assert (
        report["initial_hidden_form_upper"] == outer["full_form_error_upper_bound"] == r
    )
    assert (
        report["initial_hidden_phase_upper"]
        == outer["full_phase_error_upper_bound"]
        == z
    )
    assert report["initial_form_upper"] == nominal + r < protocol["form_upper_bound"]
    assert z < protocol["fine_phase_upper_bound"]
    assert (
        report["initial_mean_gap_upper"]
        == gap + 2 * z
        < protocol["child_origin_gap_upper_bound"]
    )
    assert (
        protocol["reading_shape"] == (5, 3, 2, 5)
        and prod(protocol["reading_shape"]) == 150
    )
    assert protocol["marked_child_order"] == tuple(range(5))
    assert protocol["receiver_child_order"] == (1, 2, 3, 4, 0)
    assert protocol["plateau_command_multipliers"] == (0, 1, -1)
    assert protocol["pulse_multipliers"] == (1, -2, 1)
    assert protocol["pulse_duration"] == 1 and protocol["total_input_scaled_time"] == 15
    assert protocol["plateau_duration"] == Q(1, 10**5)
    assert protocol["total_elapsed_scaled_time"] == 15 * (
        1 + protocol["plateau_duration"]
    )
    assert (
        protocol["total_elapsed_original_time"]
        == Q(1024, 1023) * protocol["total_elapsed_scaled_time"]
    )
    assert protocol["permitted_windings"] == (-1, 0, 1)
    assert protocol["raw_readout_error_upper_bound"] == Q(1, 10**14)
    geometry = outer["geometry"]
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    ) + tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    assert geometry["weighted_edges"] == edges and len(edges) == 90
    for state in (geometry, protocol):
        assert state["node_order"] == tuple(range(45))
        assert state["held_capacities"] == (Q(1),) * 45
        assert state["clock_factor"] == Q(1, 4) and state["phase_domain"] == "regular"
        assert I(**state["gamma_bounds"]) == gamma


def test_live_phase_corrected_plateau_errors_and_cycle_resolution(records):
    protocol = records[1]["report"]["protocol"]
    pi, gamma = pi_interval(), Q(1, 1023) / pi_interval()
    g, h, total = gamma.hi, Q(1, 10**5), 15 * (1 + Q(1, 10**5))
    x, zeta, angle, epsilon = Q(1, 100), Q(1, 10**6), Q(7, 5), Q(1, 10)
    form = x + g * total
    phase = g * (x + form) + g**2 * total
    second = 2 * (2 * form + g) + g * 8 * (2 * g * form) / 4
    coefficients = (-1 + h / 4, 1 + h / 4) + (-h / 8,) * 4
    reading = sum(map(abs, coefficients)) * Q(1, 10**14)
    transport, trapezoid = g * h * phase, second * h**3 / 12
    assert protocol["observation_form_upper_bound"] == form
    assert protocol["phase_residual_upper_bound"] == phase
    assert protocol["plateau_second_derivative_upper_bound"] == second
    assert protocol["plateau_transport_error_upper_bound"] == transport
    assert protocol["plateau_trapezoid_error_upper_bound"] == trapezoid
    assert protocol["plateau_readout_error_upper_bound"] == reading
    b = transport + trapezoid + reading
    assert protocol["plateau_total_error_upper_bound"] == b
    assert protocol["odd_total_error_upper_bound"] == 2 * b
    assert protocol["even_total_error_upper_bound"] == 4 * b
    sine, remainder = sin(I(epsilon)), 1 - cos(I(epsilon))
    widths, rounding = [], []
    for key, trig, error in (("cosine", sine, 2 * b), ("sine", remainder, 4 * b)):
        inverse = 2 / (gamma * h * trig)
        assert I(**protocol[key + "_inverse_factor_bounds"]) == inverse
        allowance = (2 * inverse.hi + inverse.width + 2) * Q(1, 2**128)
        width = (
            2 * inverse.hi * error
            + inverse.width * (g * h * trig.hi / 2 + 2 * error)
            + allowance
        )
        assert protocol[key + "_width_upper_bound"] == width
        widths.append(width)
        rounding.append(allowance)
    c0 = cos(I(angle + 2 * zeta)).lo
    angle_width = widths[1] / c0 + widths[0] / c0**2 + 4 * zeta + Q(1, 10**20)
    assert protocol["cosine_prior_lower_bound"] == c0 > 0
    assert protocol["quadrature_interval_rounding_allowances"] == tuple(rounding)
    assert protocol["child_origin_angle_width_upper_bound"] == angle_width
    assert protocol["cycle_sum_width_upper_bound"] == 5 * angle_width < 2 * pi.lo
    assert protocol["winding_resolution_margin"] == 2 * pi.lo - 5 * angle_width > 0
    expected = (
        ("internal_positive_gap_margin", 2 * pi.lo / 9 - 2 * zeta - 2 * phase),
        ("internal_acute_gap_margin", pi.lo / 2 - 2 * pi.hi / 9 - 2 * zeta - 2 * phase),
        ("collar_margin", pi.lo / 2 - 2 * zeta - 4 * phase),
        ("outer_acute_gap_margin", pi.lo / 2 - angle - epsilon - 2 * zeta - 2 * phase),
    )
    for name, margin in expected:
        assert protocol[name] == margin > 0


def test_full_word_costs_include_all_fifteen_ramps_and_remaining_loss(records):
    p = records[1]["report"]["protocol"]
    gamma = I(**p["gamma_bounds"])
    amplitude, form, phase = (
        Q(1, 5),
        p["observation_form_upper_bound"],
        p["phase_residual_upper_bound"],
    )
    # Independent cubic extrema and the C5 inverse-column/direction norms.
    offset = 4 * amplitude * Q(2, 5) * Q(3, 2) / gamma.lo
    gradient = 4 * amplitude * Q(4, 5) * Q(3, 2) / gamma.lo + 8 * form
    forcing = (
        4 * amplitude * Q(2, 5) * 6 / gamma.lo
        + amplitude * Q(4, 5) * Q(3, 2) / gamma.lo
    )
    assert p["known_offset_peak_upper"] == offset
    assert p["full_gradient_peak_upper"] == gradient
    assert p["input_peak_upper"] == forcing
    assert p["peak_storage_upper"] == 90 * ((2 * (offset + form)) ** 2 / 2 + 2)
    assert (
        p["continuous_loss_upper"]
        == 45 * p["total_elapsed_scaled_time"] * gradient**2 / 4
    )
    assert p["absolute_work_upper"] == 45 * 15 * gradient * forcing
    increments = (Q(1, 10), Q(-1, 5), Q(1, 10)) * 5
    lower = sum(
        18 * max(Q(0), abs(a) - 4 * phase) ** 2 / gamma.hi**2 for a in increments
    )
    peak_lower = max(
        12 * max(Q(0), abs(a) - 4 * phase) ** 2 / gamma.hi**2 for a in increments
    )
    assert p["continuous_loss_lower"] == lower > 0
    assert p["peak_form_storage_lower"] == peak_lower > 0
    assert p["initial_storage_upper"] == 180 * Q(1, 100) ** 2 + 180
    assert (
        p["positive_supplied_work_lower"]
        == max(Q(0), lower - p["initial_storage_upper"])
        > 0
    )
    assert lower < p["continuous_loss_upper"]
    assert peak_lower < p["peak_storage_upper"]


def test_all_source_and_protocol_decisions_are_preserved(records):
    result, report = records[1], records[1]["report"]
    p, recovery, outer = report["protocol"], report["recovery"], report["outer"]
    flags = dict(report["prerequisite_flags"])
    assert len(flags) == len(report["prerequisite_flags"]) == 14
    assert all(flag is True for flag in flags.values())
    expected = dict(
        actual_recovery_form=recovery["endpoint_form_margin"],
        actual_recovery_phase=recovery["endpoint_phase_margin"],
        outer_one_capture=outer["capture_margin"],
        outer_zero_capture=outer["control_capture_margin"],
        outer_attachment_work=outer["attachment_work_margin"],
        outer_initial_storage=outer["initial_storage_margin"],
        outer_one_form_in_prior=Q(1, 100) - report["initial_form_upper"],
        outer_zero_form_in_prior=Q(1, 100) - report["initial_hidden_form_upper"],
        orthogonal_hidden_phase_in_prior=Q(1, 10**6)
        - report["initial_hidden_phase_upper"],
        outer_one_mean_gap_in_prior=Q(7, 5) - report["initial_mean_gap_upper"],
        outer_zero_mean_gap_in_prior=Q(7, 5) - 2 * report["initial_hidden_phase_upper"],
    )
    assert dict(report["prerequisite_margins"]) == expected
    assert len(report["prerequisite_margins"]) == 11
    assert all(type(value) is Q and value > 0 for value in expected.values())
    expected = dict(
        internal_positive_gap=p["internal_positive_gap_margin"],
        internal_acute_gap=p["internal_acute_gap_margin"],
        circular_collar=p["collar_margin"],
        outer_acute_gap=p["outer_acute_gap_margin"],
        peak_storage=5 * 10**8 - p["peak_storage_upper"],
        continuous_loss=2 * 10**9 - p["continuous_loss_upper"],
        cumulative_absolute_supplied_work=2 * 10**10 - p["absolute_work_upper"],
        positive_cosine_chart=p["cosine_prior_lower_bound"],
        complete_cycle_winding_resolution=p["winding_resolution_margin"],
    )
    assert dict(p["sufficient_margins"]) == expected
    assert len(p["sufficient_margins"]) == 9
    assert all(type(value) is Q and value > 0 for value in expected.values())
    assert report["admitted"] is p["admitted"] is True
    assert report["unresolved_conditions"] == p["unresolved_conditions"] == ()
    assert result["verdict"] == "finite_input_observation_admitted"
