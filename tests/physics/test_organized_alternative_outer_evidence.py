"""Read-only exact audit of the single own-law outer-acquisition assessment.

No source, carrier, full certificate or frozen sine producer is evaluated.
Receipts associate retained bytes and committed source, not authenticated
execution, a trajectory or a physical observation.
"""

import hashlib
import importlib
import subprocess
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I
from tnfr.research.artifact_io import decode_exact_tree, exact_record, file_receipt
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
NAME = "organized_alternative_outer_acquisition"
EVIDENCE = ROOT / "docs/assets" / NAME
DECLARATION = "theory/nodal/ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md"
EVALUATOR = "benchmarks/" + NAME + ".py"
SOURCE = "src/tnfr/physics/_" + NAME + ".py"
ERROR, RADIUS = Q(1, 10**10), Q(1, 10**70)
CURRENT, STIFFNESS = 1 + Q(1, 1458), 1 + Q(2, 243)


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    endpoints = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(*endpoints)
    assert (result.lo, result.hi) == endpoints
    return result


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
    revision = attempt["source_revision"]
    try:
        return {
            "listing": _git(
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                revision,
                "--",
                *attempt["source_scope"],
            ),
            "src": _git("rev-parse", revision + ":src").decode().strip(),
            "declaration": _git("show", revision + ":" + DECLARATION),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.fixture(scope="module", autouse=True)
def preserve_evidence_and_exclude_execution(records, source_snapshot):
    # Git reads and imports precede the shared no-subprocess execution guard.
    targets = (
        (
            "tnfr.physics._organized_alternative_outer_acquisition",
            (
                "_assess_organized_alternative_outer_acquisition",
                "_derive_organized_alternative_outer_bounds",
                "_assess_constitutive_recovery",
            ),
        ),
        (
            "tnfr.physics._organized_outer_winding",
            (
                "_assess_organized_outer_winding",
                "_derive_organized_outer_resource_bounds",
                "certify_sine_prepared_entry",
            ),
        ),
        (
            "tnfr.physics._sine_constitutive_recovery",
            ("_assess_constitutive_recovery", "_assess_constitutive_class_pair"),
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
        pytest.fail("read-only outer evidence audit attempted scientific evaluation")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, attributes in imported:
            for attribute in attributes:
                patch.setattr(module, attribute, forbidden)
        yield
    assert tuple(file_receipt(path) for path in paths) == before


def test_single_attempt_source_receipt_and_committed_inventory(
    records, source_snapshot
):
    attempt, result, raw_attempt, _ = records
    assert attempt["schema"] == NAME + "_attempt_v1"
    assert result["schema"] == NAME + "_result_v1"
    assert result["source_revision"] == attempt["source_revision"]
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
        pytest.skip("retained source revision unavailable in this checkout")
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
    for text in (b"99bf595c", b"all 80 relative", b"direct edge functionals"):
        assert text in declaration


def test_full_product_state_identity_and_both_clock_transforms(records):
    p = records[1]["report"]
    r, geometry = p["resources"], p["resources"]["geometry"]
    assert p["protocol_revision"] == "99bf595c" and p["deformation"] == 1
    assert p["form_error_bounds"] == p["phase_error_bounds"] == (ERROR,) * 5
    assert p["recovered_child_norm_bound"] == r["recovered_child_norm_bound"] == RADIUS
    assert p["source_formation_time"] == 200
    assert p["source_relaxation_duration"] == 2 * 10**13
    assert p["source_handoff_time"] == 200 + 2 * 10**13
    assert p["elapsed_scaled_time"] == 200 == 2 * p["primitive_scaled_time"]
    assert p["elapsed_original_time"] * Q(1023, 1024) == 200
    assert geometry["clock_factor"] == Q(1, 4)
    assert geometry["phase_domain"] == "regular"
    assert geometry["node_order"] == tuple(range(45))
    assert geometry["held_capacities"] == (Q(1),) * 45
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    outer = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for j in range(9) for c in range(5)
    )
    assert geometry["internal_edges"] == internal
    assert geometry["outer_edges"] == outer
    assert geometry["weighted_edges"] == internal + outer
    assert p["initial_child_form_origins"] == tuple(4092 * (c - 2) for c in range(5))
    assert p["initial_child_phase_origins"] == (Q(0),) * 5
    turns = tuple(Q(j - 4, 9) + Q(c - 2, 5) for c in range(5) for j in range(9))
    assert p["target_phase_turns"] == turns and sum(turns) == 0
    assert p["target_form"] == (Q(0),) * 45
    assert p["target_child_windings"] == (1,) * 5
    assert p["target_outer_windings"] == (1,) * 9


def test_five_own_law_source_records_keep_all_directions_and_actual_recovery(records):
    p = records[1]["report"]
    children = p["child_sources"]
    assert len(children) == 5 and p["relative_preparation_dimension"] == 80
    assert p["recovered_source_flags"] == (True,) * 5
    source_scale = Q(2046, 9) * Q(355, 113) ** 2
    for child in children:
        s = child["source"]
        assert child["deformation"] == s["deformation"] == 1
        assert child["form_error_bound"] == child["phase_error_bound"] == ERROR
        assert s["form_error_bound"] == s["phase_error_bound"] == ERROR
        assert s["preparation_dimension"] == s["relative_mean_leaf_dimension"] == 16
        assert s["residual_constraints"] == (
            "sum_form_residuals_equals_zero",
            "sum_lifted_phase_residuals_equals_zero",
        )
        assert s["initial_forms_by_class"][0] == tuple(
            source_scale * (j - 4) for j in range(9)
        )
        assert s["initial_phases"] == (Q(0),) * 9
        assert s["phase_hessian_upper_bound"] == STIFFNESS
        assert s["initial_potential_deformation_upper_bound"] == 6 * ERROR**4 > 0
        assert child["formation_time"] * 2 == p["source_formation_time"]
        assert child["relaxation_duration"] * 2 == p["source_relaxation_duration"]
        assert child["decay_exponent"] >= child["decay_power"] == 512
        assert child["exact_decay_upper_bound"] == Q(1, 2**512)
        returned = child["lyapunov_initial_upper_bound"] / 2**512
        eta = _interval(s["eta_bounds"])
        x2 = 20 * returned / eta.lo
        z2 = 2 * returned / child["lyapunov_position_lower_coefficient"]
        assert child["returned_lyapunov_upper_bound"] == returned > 0
        assert child["endpoint_form_norm_squared_upper_bound"] == x2 < RADIUS**2
        assert child["endpoint_phase_norm_squared_upper_bound"] == z2 < RADIUS**2
        assert child["endpoint_form_margin"] == RADIUS**2 - x2
        assert child["endpoint_phase_margin"] == RADIUS**2 - z2
    assert p["original_preparation_storage_upper_bound"] == sum(
        child["source"]["initial_storage_upper_bounds"][0] for child in children
    )


def test_fixed_heat_bound_retains_own_current_and_nonzero_fine_errors(records):
    r = records[1]["report"]["resources"]
    assert r["current_absolute_upper_bound"] == CURRENT
    assert r["current_lipschitz_upper_bound"] == STIFFNESS
    assert r["primitive_gap_lower_bound"] == Q(1, 2)
    assert r["primitive_decay_upper_bound"] == (1 + Q(50, 16)) ** -16
    gamma = _interval(r["geometry"]["gamma_bounds"])
    initial = _interval(r["primitive_initial_scaled_norm_bounds"])
    forcing = _interval(r["primitive_forcing_norm_bounds"])
    a = r["primitive_decay_upper_bound"] * initial.hi + 2 * (gamma**2).hi * forcing.hi
    phase = a + 100 * (gamma**2).hi * forcing.hi
    assert r["primitive_scaled_form_radius_upper_bound"] == a
    assert r["primitive_phase_radius_upper_bound"] == phase
    assert 400 * STIFFNESS * gamma.hi**2 < Q(1, 2)
    z = 6 * RADIUS * (1 + gamma.hi)
    x = 3 * RADIUS + 24 * STIFFNESS * gamma.hi * z
    assert r["full_phase_error_upper_bound"] == z > 0
    assert r["full_form_error_upper_bound"] == x > 0
    gaps = tuple(map(_interval, r["endpoint_oriented_outer_phase_gap_bounds"]))
    assert len(gaps) == 45 and all(
        gaps[5 * i : 5 * i + 5] == gaps[:5] for i in range(9)
    )
    assert all(0 < gap.lo < gap.hi < Q(3, 2) for gap in gaps)
    assert _interval(r["initial_outer_phase_gap_bounds"]).contains(0)
    assert _interval(r["transit_internal_phase_gap_bounds"]).lo > 0


def test_complete_product_faces_and_storage_determine_the_strict_margin(records):
    r = records[1]["report"]["resources"]
    v9, v5 = map(
        _interval, (r["internal_cycle_minimum_bounds"], r["outer_cycle_minimum_bounds"])
    )
    positive9, negative9, positive5 = map(
        _interval,
        (
            r["internal_positive_face_storage_bounds"],
            r["internal_negative_face_storage_bounds"],
            r["outer_positive_face_storage_bounds"],
        ),
    )
    assert negative9.lo > positive9.hi
    internal, outer = positive9 - v9, positive5 - v5
    assert _interval(r["internal_boundary_excess_bounds"]) == internal
    assert _interval(r["outer_boundary_excess_bounds"]) == outer
    expected = tuple(
        value for _ in range(45) for value in ((negative9 - v9).lo, internal.lo)
    )
    expected += tuple(value for _ in range(45) for value in (None, outer.lo))
    assert r["signed_face_excess_lower_bounds"] == expected
    assert r["signed_face_feasible"] == tuple(value is not None for value in expected)
    assert len(expected) == 180 and expected.count(None) == 45
    boundary = min(internal.lo, outer.lo)
    assert r["boundary_energy_lower_bound"] == boundary > 0
    nominal = 9 * (r["primitive_endpoint_storage_upper_bound"] - v5.lo)
    assert r["endpoint_nominal_excess_upper_bound"] == nominal
    residual = r["endpoint_residual_excess_upper_bound"]
    assert residual > 14 * CURRENT * r["full_phase_error_upper_bound"] > 0
    assert r["endpoint_excess_upper_bound"] == nominal + residual
    assert r["capture_margin"] == boundary - nominal - residual > 0
    assert r["acute_convexity_split_margin"] > 0
    assert _interval(r["target_outer_stiffness_bounds"]).lo > 0


def test_event_work_and_paired_control_retain_actual_law_costs(records):
    p, r = records[1]["report"], records[1]["report"]["resources"]
    shifts = p["initial_child_form_origins"]
    nominal = sum(Q(9, 2) * (shifts[(c + 1) % 5] - shifts[c]) ** 2 for c in range(5))
    assert r["attachment_nominal_work"] == nominal == 1507001760
    residual = 10 * (1 + STIFFNESS) * RADIUS**2
    assert r["attachment_residual_work_upper_bound"] == residual > 20 * RADIUS**2
    assert r["attachment_work_upper_bound"] == nominal + residual
    initial = (
        5 * _interval(r["internal_cycle_minimum_bounds"]).hi + nominal + 2 * residual
    )
    assert r["initial_storage_upper_bound"] == initial
    assert r["attachment_work_margin"] == p["budget"] - nominal - residual > 0
    assert r["initial_storage_margin"] == p["budget"] - initial > 0
    assert p["budget"] == 2 * 10**9
    assert r["control_initial_excess_upper_bound"] == 2 * residual
    assert (
        r["control_capture_margin"]
        == r["control_boundary_energy_lower_bound"] - 2 * residual
        > 0
    )
    assert p["control_form_origins"] == (Q(0),) * 5
    assert p["control_target_outer_windings"] == (0,) * 9


def test_result_claims_require_source_entry_capture_and_paired_control(records):
    result, p = records[1], records[1]["report"]
    assert result["verdict"] == "alternative_outer_acquisition_certified"
    assert p["admitted"] and not p["unresolved_conditions"]
    for name in (
        "source_certified",
        "initial_outer_zero_winding_certified",
        "child_winding_retained_during_transit",
        "outer_entry_certified",
        "capture_certified",
        "outer_winding_retained",
        "child_winding_retained",
        "converges_to_target",
        "control_capture_certified",
        "control_outer_zero_winding_retained",
        "control_child_winding_retained",
        "control_converges_to_target",
    ):
        assert p[name] is True
    assert "no_autonomous_event_selection_or_physical_identification" in p["scope"]
