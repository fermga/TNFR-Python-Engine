"""Read-only audit of the retained own-law finite contact assessment.

The original source, carrier, recovery and response are never regenerated.
Receipts associate retained bytes and source; they do not authenticate an
experiment or independently establish the conditional theorem.
"""

from fractions import Fraction as Q
import hashlib
import importlib
from pathlib import Path
import subprocess

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research.artifact_io import decode_exact_tree, exact_record, file_receipt
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
NAME = "organized_alternative_contact"
EVIDENCE = ROOT / "docs/assets" / NAME
DECLARATION = "theory/nodal/ORGANIZED_ALTERNATIVE_CONTACT.md"
EVALUATOR = "benchmarks/" + NAME + ".py"
SOURCE = "src/tnfr/physics/_" + NAME + ".py"
METHOD_REVISION = "pending_committed_method"
ERROR, RADIUS = Q(1, 10**10), Q(1, 10**70)
CURRENT, STIFFNESS = 1 + Q(1, 1458), 1 + Q(2, 243)


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    endpoints = exact_record(value["lo"]), exact_record(value["hi"])
    interval = I(*endpoints)
    assert (interval.lo, interval.hi) == endpoints
    return interval


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="module")
def records():
    raw = tuple(
        (EVIDENCE / name).read_bytes() for name in ("attempt.json", "result.json")
    )
    return (*tuple(decode_exact_tree(json_loads(value)) for value in raw), *raw)


@pytest.fixture(scope="module")
def source_snapshot(records):
    attempt = records[0]
    assert attempt["source_revision"] == METHOD_REVISION
    try:
        return {
            "listing": _git(
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                METHOD_REVISION,
                "--",
                *attempt["source_scope"],
            ),
            "src": _git("rev-parse", METHOD_REVISION + ":src").decode().strip(),
            "declaration": _git("show", METHOD_REVISION + ":" + DECLARATION),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.fixture(scope="module", autouse=True)
def preserve_evidence_and_exclude_execution(records, source_snapshot):
    # Resolve read-only Git and imports before the no-subprocess guard.
    targets = (
        (
            "tnfr.physics._organized_alternative_contact",
            (
                "_assess_organized_alternative_contact",
                "_derive_organized_alternative_contact_bounds",
                "_assess_organized_alternative_outer_acquisition",
                "_derive_organized_alternative_outer_bounds",
                "_assess_constitutive_recovery",
            ),
        ),
        (
            "tnfr.physics._organized_alternative_outer_acquisition",
            (
                "_assess_organized_alternative_outer_acquisition",
                "_derive_organized_alternative_outer_bounds",
                "_assess_constitutive_recovery",
            ),
        ),
        (
            "tnfr.physics._sine_constitutive_recovery",
            ("_assess_constitutive_recovery", "_assess_constitutive_class_pair"),
        ),
        (
            "tnfr.physics._organized_external_interaction",
            ("_assess_organized_external_interaction",),
        ),
        (
            "tnfr.physics._organized_external_registration",
            ("_assess_organized_external_registration",),
        ),
        (
            "tnfr.research._single_analytic_assessment",
            ("_run_single_analytic_assessment",),
        ),
    )
    imported = tuple(
        (importlib.import_module(name), attributes) for name, attributes in targets
    )
    paths = tuple(EVIDENCE / name for name in ("attempt.json", "result.json"))
    before = tuple(file_receipt(path) for path in paths)

    def forbidden(*args, **kwargs):
        pytest.fail("read-only contact evidence audit attempted scientific evaluation")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, attributes in imported:
            for attribute in attributes:
                patch.setattr(module, attribute, forbidden)
        yield
    assert tuple(file_receipt(path) for path in paths) == before


def test_single_attempt_receipts_and_pinned_committed_source(records, source_snapshot):
    attempt, result, raw_attempt, _ = records
    assert attempt["schema"] == NAME + "_attempt_v1"
    assert result["schema"] == NAME + "_result_v1"
    assert result["source_revision"] == attempt["source_revision"] == METHOD_REVISION
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert attempt["arithmetic_method"] == (
        "rational_outward_dyadic128_machin_trigonometry_v1"
    )
    assert (attempt["declaration"], attempt["evaluator"]) == (DECLARATION, EVALUATOR)
    assert attempt["source_scope"] == (
        "src/tnfr",
        EVALUATOR,
        DECLARATION,
        "pyproject.toml",
    )
    assert result["attempt"] == {
        "path": "attempt.json",
        "bytes": len(raw_attempt),
        "sha256": hashlib.sha256(raw_attempt).hexdigest(),
    }
    assert {path.name for path in EVIDENCE.iterdir()} == {"attempt.json", "result.json"}
    if source_snapshot is None:
        pytest.skip("retained method revision unavailable in this checkout")
    listing, inventory = source_snapshot["listing"], {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = {"mode": mode, "blob": blob}
    assert inventory == attempt["source_inventory"]
    assert hashlib.sha256(listing).hexdigest() == attempt["source_inventory_sha256"]
    assert source_snapshot["src"] == attempt["source_tree"]
    assert {DECLARATION, EVALUATOR, SOURCE} <= set(inventory)
    assert all(
        set(record["imported_sources"].values()) <= set(inventory)
        for record in (attempt, result)
    )
    declaration = source_snapshot["declaration"]
    payload = b"blob " + str(len(declaration)).encode() + b"\0" + declaration
    assert hashlib.sha1(payload).hexdigest() == inventory[DECLARATION]["blob"]
    assert b"e2b2e647" in declaration


def test_complete_support_original_state_and_both_clock_rows(records):
    p = records[1]["report"]
    geometry = p["resources"]["geometry"]
    assert p["protocol_revision"] == "e2b2e647" and p["deformation"] == 1
    assert p["form_error_bounds"] == p["phase_error_bounds"] == (ERROR,) * 6
    assert p["capacity"] == (Q(1),) * 54
    assert geometry["node_order"] == tuple(range(54))
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(6) for j in range(9)
    )
    outer = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    contact = tuple((9 * c + j, 45 + j, Q(1)) for c in range(5) for j in range(9))
    assert geometry["internal_edges"] == internal
    assert geometry["outer_edges"] == outer
    assert set(geometry["contact_edges"]) == set(contact)
    assert geometry["weighted_edges"] == internal + outer + geometry["contact_edges"]
    degrees = [0] * 54
    for left, right, weight in geometry["weighted_edges"]:
        assert weight == 1
        degrees[left] += 1
        degrees[right] += 1
    assert degrees == [5] * 45 + [7] * 9
    assert len(geometry["weighted_edges"]) == 144
    assert geometry["clock_factor"] == Q(1, 4)
    assert p["elapsed_scaled_time"] == Q(1, 1000)
    assert p["elapsed_original_time"] * Q(1023, 1024) == p["elapsed_scaled_time"]
    assert p["source_form_totals"] == (Q(0), Q(0))
    assert p["source_donor_form_mean"] == 0
    assert (
        p["joined_lifted_phase_total"] == "9*beta; beta supplied at donor preparation"
    )
    assert p["pressure_refresh"] == "continuous_current_at_actual_state"
    assert p["event_policy"] == (
        "fifteen_supplied_word_jumps_then_one_attachment_then_no_events"
    )


def test_six_actual_recoveries_keep_all_96_relative_directions(records):
    p = records[1]["report"]
    host, donor = p["host_acquisition"], p["donor_recovery"]
    assert p["resources"]["host_bounds"] == host["resources"]
    assert host["form_error_bounds"] == p["form_error_bounds"][:5]
    assert host["phase_error_bounds"] == p["phase_error_bounds"][:5]
    assert host["deformation"] == donor["deformation"] == p["deformation"] == 1
    assert host["relative_preparation_dimension"] == 80
    children = host["child_sources"] + (donor,)
    assert len(children) == 6 and p["relative_preparation_dimension"] == 96
    assert p["recovered_norm_bound"] == RADIUS
    assert p["source_host_time"] == host["elapsed_scaled_time"] == 200
    assert p["donor_formation_time"] == host["source_formation_time"] == 200
    assert (
        p["donor_recovery_duration"] == host["source_relaxation_duration"] == 2 * 10**13
    )
    for child in children:
        source = child["source"]
        assert child["deformation"] == source["deformation"] == 1
        assert child["form_error_bound"] == child["phase_error_bound"] == ERROR
        assert source["preparation_dimension"] == 16
        assert source["residual_constraints"] == (
            "sum_form_residuals_equals_zero",
            "sum_lifted_phase_residuals_equals_zero",
        )
        assert (
            source["conserved_form_mean"] == source["conserved_lifted_phase_mean"] == 0
        )
        assert source["phase_hessian_upper_bound"] == STIFFNESS
        assert source["initial_potential_deformation_upper_bound"] == 6 * ERROR**4 > 0
        assert child["formation_time"] * 2 == p["donor_formation_time"]
        assert child["relaxation_duration"] * 2 == p["donor_recovery_duration"]
        assert child["exact_decay_upper_bound"] == Q(1, 2**512)
        returned = child["lyapunov_initial_upper_bound"] / 2**512
        eta = _interval(source["eta_bounds"])
        x2 = 20 * returned / eta.lo
        z2 = 2 * returned / child["lyapunov_position_lower_coefficient"]
        assert child["returned_lyapunov_upper_bound"] == returned > 0
        assert child["endpoint_form_norm_squared_upper_bound"] == x2 < RADIUS**2
        assert child["endpoint_phase_norm_squared_upper_bound"] == z2 < RADIUS**2
        assert child["endpoint_form_margin"] == RADIUS**2 - x2 > 0
        assert child["endpoint_phase_margin"] == RADIUS**2 - z2 > 0
        assert child["source_certified"] and child["recovery_certified"]
    assert host["admitted"] and host["capture_certified"]
    assert (
        host["control_capture_certified"]
        and host["control_outer_zero_winding_retained"]
    )
    assert p["source_admission_flags"] == (True, True, True)
    assert p["source_certified"] is True


def test_carried_word_keeps_actual_drift_events_and_separate_work(records):
    p, r = records[1]["report"], records[1]["report"]["resources"]
    host = r["host_bounds"]
    assert r["recovered_norm_bound"] == RADIUS
    assert r["current_absolute_upper_bound"] == CURRENT
    assert r["current_lipschitz_upper_bound"] == STIFFNESS
    assert p["word_window_duration"] == Q(1, 100000)
    assert p["word_total_duration"] == 20 * p["word_window_duration"] == Q(1, 5000)
    assert p["word_phase_amplitude"] == Q(1, 10)
    assert p["word_event_multipliers"] == (1, -2, 1)
    assert p["word_marked_child_order"] == tuple(range(5))
    assert p["word_receiver_child_order"] == (1, 2, 3, 4, 0)
    assert p["word_reading_count"] == 25
    assert p["word_raw_readout_error_upper_bound"] == Q(1, 10**15)
    edge_sets = host["geometry"]["internal_edges"], host["geometry"]["outer_edges"]
    for child in range(5):
        command = tuple(Q(int(node // 9 == child)) - Q(1, 5) for node in range(45))
        assert sum(command) == 0
        assert all(command[a] == command[b] for a, b, _ in edge_sets[0])
        assert sum(command[a] != command[b] for a, b, _ in edge_sets[1]) == 18
        assert sum(p["word_event_multipliers"]) == 0
        cumulative = (0, 1, -1, 0)
        assert max(abs(value) for value in cumulative) == 1
    gamma = _interval(r["geometry"]["gamma_bounds"])
    duration = p["word_total_duration"]
    bo = Q(1, 100) + gamma.hi * CURRENT * duration
    drift = 2 * gamma.hi * bo * duration
    assert r["observation_form_upper_bound"] == bo
    assert r["carried_phase_drift_upper_bound"] == drift > 0
    assert r["word_edge_drift_upper_bound"] == 2 * drift
    assert r["source_form_upper_bounds"] == (
        host["primitive_node_form_radius_upper_bound"]
        + host["full_form_error_upper_bound"],
        host["full_form_error_upper_bound"],
    )
    assert dict(r["source_inclusion_margins"]) == {
        "outer_one_initial_form": Q(1, 100) - r["source_form_upper_bounds"][0],
        "outer_zero_initial_form": Q(1, 100) - r["source_form_upper_bounds"][1],
        "donor_initial_form": Q(1, 100) - RADIUS,
    }
    event_work = tuple(
        18 * CURRENT * abs(value) * p["word_phase_amplitude"]
        for value in p["word_event_multipliers"]
    )
    assert r["per_block_absolute_event_work_upper_bounds"] == event_work
    assert (
        r["total_absolute_event_work_upper_bound"]
        == 5 * sum(event_work)
        == 36 * CURRENT
    )
    assert dict(r["word_work_margins"]) == {
        "absolute_event_work": 4 - max(event_work),
        "total_absolute_word_work": 40 - 5 * sum(event_work),
    }
    assert max(event_work) > Q(36, 10) and 5 * sum(event_work) > 36


def test_source_correlations_bound_fundamental_moment_without_a_decoder(records):
    r = records[1]["report"]["resources"]
    host = r["host_bounds"]
    phase_error = (
        host["full_phase_error_upper_bound"]
        + r["carried_phase_drift_upper_bound"]
        + RADIUS
    )
    assert r["boundary_phase_error_upper_bound"] == phase_error > RADIUS
    pi = pi_interval()
    radius = host["primitive_node_phase_radius_upper_bound"]
    expected = tuple(4 * (c - 2) / pi + I(-radius, radius) for c in range(5))
    families = tuple(
        tuple(map(_interval, row)) for row in r["nominal_host_phase_offset_bounds"]
    )
    assert families == (expected, (I(0),) * 5)
    for index, phases in enumerate(families):
        cosine = sum(
            (cos(phase + I(-phase_error, phase_error)) for phase in phases), I(0)
        )
        sine = -sum(
            (sin(phase + I(-phase_error, phase_error)) for phase in phases), I(0)
        )
        amplitude = sqrt(cosine**2 + sine**2)
        assert _interval(r["fundamental_cosine_bounds"][index]) == cosine
        assert _interval(r["fundamental_sine_bounds"][index]) == sine
        assert _interval(r["fundamental_amplitude_bounds"][index]) == amplitude
        assert amplitude.lo >= 0
    assert (
        "source_identity_from_preparation_not_a_transferred_readout_inverse"
        in records[1]["report"]["scope"]
    )


def test_complete_reaction_remainder_harmonic_tail_and_each_reading_error(records):
    p, r = records[1]["report"], records[1]["report"]["resources"]
    gamma = _interval(r["geometry"]["gamma_bounds"])
    h = p["elapsed_scaled_time"]
    b = r["observation_form_upper_bound"] + Q(7, 4) * gamma.hi * CURRENT * h
    assert r["joined_form_upper_bound"] == b
    assert r["contact_edge_drift_upper_bound"] == 7 * gamma.hi * b * h
    expected = (
        Q(15, 16) * gamma.hi * CURRENT * h**2
        + Q(15, 4) * gamma.hi**2 * STIFFNESS * b * h**2
    )
    assert r["finite_response_remainder_upper_bound"] == expected > 0
    assert p["raw_readout_error_upper_bound"] == ERROR
    assert p["increment_readout_error_upper_bound"] == 2 * ERROR
    assert p["paired_difference_readout_error_upper_bound"] == 4 * ERROR
    assert r["total_increment_error_upper_bound"] == expected + 2 * ERROR
    # Forty-five actual marked contacts, normalized by nine donor nodes,
    # and both nonzero higher Fourier coefficients determine the tail.
    tail = Q(45, 9) * (Q(1, 2187) + Q(1, 4374))
    assert r["higher_harmonic_current_upper_bound"] == tail == Q(5, 1458)
    factor = gamma * h / 4
    capabilities = []
    for amplitude in map(_interval, r["fundamental_amplitude_bounds"]):
        error = expected + 2 * ERROR
        capabilities.append(
            I(
                max(Q(0), factor.lo * max(Q(0), amplitude.lo - tail) - error),
                factor.hi * (amplitude.hi + tail) + error,
            )
        )
    assert tuple(map(_interval, r["observed_capability_bounds"])) == tuple(capabilities)
    assert p["outer_one_upper_threshold"] == Q(1, 10**8)
    assert p["outer_zero_lower_threshold"] == Q(3, 10**7)
    assert (
        r["outer_one_upper_margin"]
        == p["outer_one_upper_threshold"] - capabilities[0].hi
    )
    assert (
        r["outer_zero_lower_margin"]
        == capabilities[1].lo - p["outer_zero_lower_threshold"]
    )
    assert (
        r["paired_absolute_response_gap_lower_bound"]
        == capabilities[1].lo - capabilities[0].hi
    )
    assert (
        "outer_one_every_origin_upper_and_outer_zero_per_source_witness_lower"
        in p["scope"]
    )
    assert (
        "witness_can_depend_on_hidden_state_and_supplies_no_alignment_selector"
        in p["scope"]
    )


def test_contact_event_and_current_storage_keep_actual_potential_and_prior_work(
    records,
):
    p, r = records[1]["report"], records[1]["report"]["resources"]
    host = r["host_bounds"]
    bo = r["observation_form_upper_bound"]
    work = 45 * ((bo + RADIUS) ** 2 / 2 + 2 + Q(2, 19683))
    assert r["contact_work_upper_bound"] == work > 90
    v9 = _interval(host["internal_cycle_minimum_bounds"])
    donor = v9.hi + 2 * (1 + STIFFNESS) * RADIUS**2
    assert r["donor_storage_upper_bound"] == donor
    sources = (
        _interval(host["target_storage_bounds"]).hi
        + host["endpoint_excess_upper_bound"],
        5 * v9.hi + host["control_initial_excess_upper_bound"],
    )
    assert r["host_preword_storage_upper_bounds"] == sources
    joined = tuple(value + 36 * CURRENT + donor + work for value in sources)
    assert r["joined_initial_storage_upper_bounds"] == joined
    assert p["contact_work_budget"] == 100 and p["joined_storage_budget"] == 200
    assert dict(r["contact_work_margins"]) == {
        "contact_attachment_work": 100 - work,
        "outer_one_joined_storage": 200 - joined[0],
        "outer_zero_joined_storage": 200 - joined[1],
    }
    # Acquisition work is a separate historical item, not a negative credit.
    assert p["host_acquisition"]["resources"]["attachment_nominal_work"] == 1507001760


def test_full_word_and_contact_identities_retain_actual_fine_state(records):
    r = records[1]["report"]["resources"]
    host = r["host_bounds"]
    alpha, pi = _interval(r["geometry"]["alpha_bounds"]), pi_interval()
    z = host["full_phase_error_upper_bound"]
    do, dj = r["word_edge_drift_upper_bound"], r["contact_edge_drift_upper_bound"]
    gaps = tuple(map(_interval, host["endpoint_oriented_outer_phase_gap_bounds"]))
    actual_gaps = (max(gap.abs_max for gap in gaps), 2 * z)
    assert r["source_outer_gap_upper_bounds"] == actual_gaps
    word = {
        "word_internal_positive_gap": alpha.lo - 2 * z - do,
        "word_internal_acute_gap": pi.lo / 2 - alpha.hi - 2 * z - do,
    }
    for name, angle in zip(("outer_one", "outer_zero"), actual_gaps):
        word[f"word_{name}_outer_acute_gap"] = pi.lo / 2 - angle - Q(1, 10) - do
    assert dict(r["word_identity_margins"]) == word
    expected = {}
    for child in range(6):
        error, previous_drift = (RADIUS, Q(0)) if child == 5 else (z, do)
        expected[f"contact_internal_positive_gap[{child}]"] = (
            alpha.lo - 2 * error - previous_drift - dj
        )
        expected[f"contact_internal_acute_gap[{child}]"] = (
            pi.lo / 2 - alpha.hi - 2 * error - previous_drift - dj
        )
    for name, angle in zip(("outer_one", "outer_zero"), actual_gaps):
        for slot in range(9):
            expected[f"contact_{name}_outer_acute_gap[{slot}]"] = (
                pi.lo / 2 - angle - do - dj
            )
    assert dict(r["contact_identity_margins"]) == expected
    assert len(expected) == 12 + 2 * 9
    assert (
        "all_six_internal_and_nine_host_outer_identities_only_through_contact_horizon"
        in records[1]["report"]["scope"]
    )


def test_joint_verdict_requires_all_own_law_source_history_and_contact_conditions(
    records,
):
    result, p = records[1], records[1]["report"]
    r = p["resources"]
    grouped = (
        "source_inclusion_margins",
        "word_identity_margins",
        "word_work_margins",
        "contact_work_margins",
        "contact_identity_margins",
        "response_margins",
    )
    margins = tuple(item for name in grouped for item in r[name])
    assert r["sufficient_margins"] == margins
    all_positive = all(value > 0 for _, value in margins)
    assert r["conditional_admitted"] == all_positive
    assert p["word_certified"] == (
        p["source_certified"]
        and r["word_identity_certified"]
        and r["word_work_certified"]
    )
    assert p["work_certified"] == (p["word_certified"] and r["contact_work_certified"])
    assert p["identity_certified"] == (
        p["word_certified"] and r["contact_identity_certified"]
    )
    assert p["response_certified"] == (p["word_certified"] and r["response_certified"])
    assert p["admitted"] == (all(p["source_admission_flags"]) and all_positive)
    assert result["verdict"] == (
        "alternative_contact_certified"
        if p["admitted"]
        else "alternative_contact_unavailable"
    )
    assert p["status"] == result["verdict"]
    assert bool(p["unresolved_conditions"]) == (not p["admitted"])
    if p["admitted"]:
        assert (
            _interval(r["fundamental_amplitude_bounds"][1]).lo
            > r["higher_harmonic_current_upper_bound"]
        )
        assert r["paired_absolute_response_gap_lower_bound"] > Q(29, 10**8)
