"""Read retained exact evidence; never rerun an assessor or instrument.

Byte associations and arithmetic checks do not authenticate execution or
replace the conditional proof's model and preparation premises.
"""

import hashlib
import subprocess
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_joint_resources"
REVISION = "0ca8630dc05a94d90b5d9fbad3fdc3b694f2e4ea"


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


def test_attempt_receipt_and_declared_corner(records):
    attempt, result = records
    assert attempt["schema"] == "organized_joint_resources_attempt_v1"
    assert result["schema"] == "organized_joint_resources_result_v1"
    raw = (EVIDENCE / "attempt.json").read_bytes()
    assert result["attempt"] == dict(
        path="attempt.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    assert result["source_revision"] == attempt["source_revision"] == REVISION
    assert attempt["parameters"] == dict(
        form_error_bounds=(Q(1, 10**6),) * 6,
        phase_error_bounds=(Q(1, 10**6),) * 6,
        quarter_dwell=Q(8 * 10**10),
        recovered_radius=Q(1, 10**8),
        decay_power=80,
        sequential_readout_error=Q(1, 10**14),
        contact_readout_error=Q(1, 10**9),
    )
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert not (EVIDENCE / "incomplete.json").exists()
    assert (
        attempt["node_order"] == tuple(range(54)) and attempt["source_directions"] == 96
    )
    for name, value in attempt["parameters"].items():
        assert result["report"][name] == value


def test_exact_frozen_source_inventory_and_declaration(records):
    attempt, result = records
    revision = attempt["source_revision"]
    try:
        _git("cat-file", "-e", revision + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("frozen Git revision is unavailable in this checkout")
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
    assert (
        _git("rev-parse", revision + ":src").decode().strip() == attempt["source_tree"]
    )
    assert set(result["imported_sources"].values()) <= set(inventory)
    declaration = _git("show", revision + ":" + attempt["declaration"]).decode("utf-8")
    for statement in (
        "[0,1e-6]",
        "D=8*10^10",
        "q=1e-8",
        "2^-80",
        "[0,1e-14]",
        "[0,1e-9]",
    ):
        assert statement in declaration


def test_stored_recovery_reconstructs_decay_and_both_endpoint_contracts(records):
    attempt, result = records
    params = attempt["parameters"]
    q, power = params["recovered_radius"], params["decay_power"]
    for recovery in result["report"]["recoveries"]:
        source = recovery["source"]
        eta = source["eta_bounds"]
        mu = eta["lo"] * recovery["class_one_phase_hessian_lower_bound"] / 25
        lower, upper = mu / 2 + Q(1, 400), 2 * eta["hi"] + Q(21, 400)
        rate = min(Q(1, 5), mu / (20 * upper))
        initial = (
            Q(3, 2) * eta["hi"] * source["endpoint_form_norm_upper_bounds"][0] ** 2
        )
        initial += 5 * upper * source["endpoint_target_phase_norm_upper_bounds"][0] ** 2
        returned = initial / 2**power
        x2, theta2 = 20 * returned / eta["lo"], 2 * returned / lower
        assert recovery["decay_exponent"] == rate * params["quarter_dwell"] / 2 >= power
        assert recovery["exact_decay_upper_bound"] == Q(1, 2**power)
        assert recovery["endpoint_form_norm_squared_upper_bound"] == x2 < q**2
        assert recovery["endpoint_phase_norm_squared_upper_bound"] == theta2 < q**2
        assert recovery["endpoint_form_margin"] == q**2 - x2
        assert recovery["endpoint_phase_margin"] == q**2 - theta2


def test_stored_contact_remainders_and_response_margins(records):
    attempt, result = records
    report, params = result["report"], attempt["parameters"]
    gamma = I(**report["outer_resources"]["geometry"]["gamma_bounds"])
    g, h, q = gamma.hi, Q(1, 1000), params["recovered_radius"]
    b_obs = Q(1, 100) + g / 5000
    b = max(b_obs, q) + Q(7, 4) * g * h
    remainder = Q(15, 16) * g * h**2 + Q(15, 4) * g**2 * b * h**2
    assert report["contact_form_upper_bound"] == b
    assert report["response_remainder_upper_bound"] == remainder
    assert report["contact_work_upper_bound"] == 90 + Q(45, 2) * (b_obs + q) ** 2 < 100
    error = remainder + 2 * params["contact_readout_error"]
    factor = gamma * h / 4
    for amplitude, capability in zip(
        report["boundary_amplitude_bounds"], report["capability_bounds"]
    ):
        response = factor * I(**amplitude)
        expected = I(max(Q(0), response.lo - error), response.hi + error)
        assert capability == dict(lo=expected.lo, hi=expected.hi)
    assert report["capability_bounds"][0]["hi"] < Q(1, 10**8)
    assert report["capability_bounds"][1]["lo"] > Q(3, 10**7)
    assert report["fixed_origin_contrast_bounds"]["lo"] > Q(1, 10**7)
    assert all(value > 0 for _, value in report["sufficient_margins"])
    assert report["admitted"] and not report["unresolved_conditions"]
    assert result["verdict"] == "joint_resource_box_admitted"
