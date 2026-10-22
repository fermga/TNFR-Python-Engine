"""Read-only audit of the frozen whole-record predictive sufficiency evidence.

The archived local Taylor witnesses retain their smooth-forcing proof premises;
this audit rebuilds radius propagation, sample association and dual arithmetic.
It evaluates no coefficient design, weight proposal, nonlinear history or source.
Receipts associate contents and declared source, not execution authentication.
"""

import hashlib
import subprocess
import zipfile
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_observed_prediction"
REVISION = "de5b500f70b1626d75f12484af0e6381fc9d93af"
DECLARATION = "theory/nodal/ORGANIZED_OBSERVED_PREDICTION.md"
EVALUATOR = "benchmarks/organized_observed_prediction.py"
CONSUMER = "src/tnfr/physics/_organized_observed_prediction.py"
KERNEL = "src/tnfr/mathematics/_validated_linear_forcing.py"
MARKER = (
    b"<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->"
)
EXPANSION_LIMIT = 128 * 1024**2


def _decode(raw):
    return decode_exact_tree(json_loads(raw))


def _interval(value):
    assert set(value) == {"lo", "hi"}
    lower, upper = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(lower, upper)
    assert (result.lo, result.hi) == (lower, upper)
    return result


def _matrix(values, rows, columns):
    assert len(values) == rows and all(len(row) == columns for row in values)
    return tuple(tuple(map(_interval, row)) for row in values)


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="module")
def source_snapshot():
    # Inspect Git before the opt-in execution guard blocks subprocesses.
    try:
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        return None
    scope = ("src/tnfr", EVALUATOR, DECLARATION, "pyproject.toml")
    return {
        "listing": _git("ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *scope),
        "src": _git("rev-parse", REVISION + ":src").decode("ascii").strip(),
        "declaration": _git("show", REVISION + ":" + DECLARATION),
    }


@pytest.fixture(scope="module")
def retained():
    receipt = _decode(read_bytes_bounded(EVIDENCE / "receipt.json", max_bytes=2**20))
    archive = EVIDENCE / "response.zip"
    assert receipt["response_archive"] == file_receipt(
        archive, label="response.zip", max_bytes=EXPANSION_LIMIT
    )
    member = receipt["result"]
    assert member["path"] == "result.json"
    verify_archive_members(
        archive, {"result.json": member["sha256"]}, max_bytes=EXPANSION_LIMIT
    )
    with zipfile.ZipFile(archive) as stream:
        raw_result = stream.read("result.json")
    assert len(raw_result) == member["bytes"]
    assert hashlib.sha256(raw_result).hexdigest() == member["sha256"]
    raw_attempt = read_bytes_bounded(EVIDENCE / "attempt.json", max_bytes=2**22)
    return _decode(raw_attempt), _decode(raw_result), raw_attempt


@pytest.fixture(scope="module", autouse=True)
def exclude_producers_and_preserve_evidence(retained, source_snapshot):
    from tnfr.mathematics import _validated_linear_forcing as kernel
    from tnfr.physics import _organized_observed_prediction as owner

    paths = tuple(
        EVIDENCE / name for name in ("attempt.json", "receipt.json", "response.zip")
    ) + tuple(
        ROOT / name
        for name in (
            EVALUATOR,
            CONSUMER,
            KERNEL,
            "docs/assets/organized_computable_prediction/attempt.json",
            "docs/assets/organized_computable_prediction/response.zip",
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
        )
    )
    before = tuple(file_receipt(path) for path in paths)

    def forbidden(*args, **kwargs):
        pytest.fail("evidence audit attempted coefficient or response regeneration")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for name in (
            "_assess_organized_observed_prediction",
            "_static_design",
            "_dual_proposal",
            "_edge_forcing",
            "validated_linear_forcing_step",
        ):
            patch.setattr(owner, name, forbidden)
        patch.setattr(kernel, "validated_linear_forcing_step", forbidden)
        yield
    assert tuple(file_receipt(path) for path in paths) == before


def test_retained_source_declaration_and_receipts(retained, source_snapshot):
    attempt, result, raw_attempt = retained
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_observed_prediction_attempt_v1"
    assert result["schema"] == "organized_observed_prediction_result_v1"
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
    assert result["attempt"] == {
        "path": "attempt.json",
        "bytes": len(raw_attempt),
        "sha256": hashlib.sha256(raw_attempt).hexdigest(),
    }
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "attempt.json",
        "receipt.json",
        "response.zip",
    }
    if source_snapshot is None:
        pytest.skip("retained source revision unavailable in this checkout")
    listing, inventory = source_snapshot["listing"], {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = {"mode": mode, "blob": blob}
    assert attempt["source_inventory"] == inventory
    assert attempt["source_inventory_sha256"] == hashlib.sha256(listing).hexdigest()
    assert attempt["source_tree"] == source_snapshot["src"]
    assert {DECLARATION, EVALUATOR, CONSUMER, KERNEL} <= set(inventory)
    for record in (attempt, result):
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert (
        result["imported_sources"]["tnfr.physics._organized_observed_prediction"]
        == CONSUMER
    )
    assert (
        result["imported_sources"]["tnfr.mathematics._validated_linear_forcing"]
        == KERNEL
    )
    frozen = source_snapshot["declaration"].replace(b"\r\n", b"\n")
    current = (ROOT / DECLARATION).read_bytes().replace(b"\r\n", b"\n")
    assert frozen.count(MARKER) == current.count(MARKER) == 1
    assert frozen.split(MARKER)[0] == current.split(MARKER)[0]


def _stages():
    elapsed = Q(200)
    stages = []
    for _ in range(15):
        stages.append((elapsed, Q(1, 100000), True))
        elapsed += Q(1, 100000)
        stages.append((elapsed, Q(1), False))
        elapsed += 1
    stages.append((elapsed, 216 - elapsed, False))
    return tuple(stages)


def test_every_compact_step_propagates_and_populates_the_same_joint_matrix(retained):
    design = retained[1]["report"]["design"]
    observation = _matrix(design["observation_matrix"], 150, 15)
    future = _matrix(design["future_matrix"], 5, 15)
    assert len(design["steps"]) == 15 * 143
    centers = [tuple(Q(i == j) for i in range(5)) for j in range(15)]
    radii = [Q(0)] * 15
    observations, times = [], []
    position = step_count = 0
    maximum_local = Q(0)

    def snapshot():
        return tuple(
            tuple(
                I(centers[j][i] - radii[j], centers[j][i] + radii[j]) for j in range(15)
            )
            for i in range(5)
        )

    for start, duration, plateau in _stages():
        time, end = start, start + duration
        if plateau:
            times.append(time - 200)
            observations.extend(snapshot())
        while time < end:
            step_duration = min(Q(1, 8), end - time)
            for column in range(15):
                witness = design["steps"][position]
                position += 1
                assert type(witness["column"]) is int and witness["column"] == column
                assert exact_record(witness["time"]) == time
                assert exact_record(witness["duration"]) == step_duration
                polynomial = tuple(map(_interval, witness["polynomial"]))
                remainder = tuple(map(_interval, witness["remainder"]))
                assert len(polynomial) == len(remainder) == 5
                local = I(
                    max(p.radius + r.abs_max for p, r in zip(polynomial, remainder))
                ).hi
                centers[column] = tuple(p.midpoint for p in polynomial)
                assert (
                    tuple(map(exact_record, witness["endpoint_center"]))
                    == centers[column]
                )
                radii[column] = I(radii[column] + local).hi
                assert exact_record(witness["endpoint_radius"]) == radii[column]
                maximum_local = max(maximum_local, local)
            step_count += 1
            time += step_duration
        if plateau:
            times.append(time - 200)
            observations.extend(snapshot())
    assert step_count == design["steps_per_column"] == 143
    assert position == len(design["steps"])
    assert tuple(times) == tuple(map(exact_record, design["observation_times"]))
    assert times[0] == 0 and times[-1] == Q(1400015, 100000)
    assert (
        len(times) == 30 and max(b - a for a, b in zip(times, times[1:] + [Q(16)])) < 2
    )
    assert tuple(observations) == observation
    assert snapshot() == future
    assert exact_record(design["maximum_local_error"]) == maximum_local
    assert exact_record(design["maximum_coefficient_radius"]) == max(radii)
    admitted = max(radii) <= Q(1, 10**12)
    assert type(design["admitted"]) is bool and design["admitted"] is admitted
    assert design["reason"] == (
        None if admitted else "static_coefficient_budget_exceeded"
    )


def test_conserved_common_form_and_zero_sum_forcing_remain_in_the_design(retained):
    design = retained[1]["report"]["design"]
    observation = _matrix(design["observation_matrix"], 150, 15)
    future = _matrix(design["future_matrix"], 5, 15)
    for row in observation + future:
        assert sum(row[:5], I(0)).contains(1)
    for offset in range(0, 150, 5):
        for column in range(15):
            assert sum(
                (observation[offset + i][column] for i in range(5)), I(0)
            ).contains(int(column < 5))
    for column in range(15):
        assert sum((future[i][column] for i in range(5)), I(0)).contains(
            int(column < 5)
        )


def _rebuilt_budget():
    g, duration = Q(1, 3000), Q(16)
    a0, b0 = Q(31, 10**9), Q(601, 10**10)
    coupling = 2 * g * duration
    a = (a0 + coupling * b0) / (1 - coupling**2)
    z = (b0 + coupling * a0) / (1 - coupling**2)
    source = duration * g * z**2 / (9 * (1 - (g * duration) ** 2))
    drift = 4 * g * Q(1, 100) + 2 * g**2 * duration
    beta = 6 * g**2 * duration
    margin = 1 - 2 * g**2 * duration
    return {
        "hidden_form_upper": a,
        "hidden_phase_upper": z,
        "source_error": source,
        "historical_pair_error": 2 * (Q(1, 10**14) + source),
        "edge_drift": drift,
        "heat_integral_upper": Q(6),
        "continuous_sample_margin": margin,
        "remainder_phasor_factor": 18 * drift + beta * 6 * (1 + drift) / margin,
        "remainder_record_factor": beta / margin,
    }


def test_frozen_primitives_rebuild_source_and_difference_remainder(retained):
    budget = _rebuilt_budget()
    actual = retained[1]["report"]["budget"]
    assert {name: exact_record(value) for name, value in actual.items()} == budget
    assert budget["source_error"] == Q(6162522987, 2846927305294951424000000000)
    assert 0 < budget["continuous_sample_margin"] < 1


def test_saved_duals_rebuild_residuals_and_independent_decision(retained):
    from tnfr.physics._organized_observed_prediction import _certify_dual, _StaticDesign

    result = retained[1]
    report, budget = result["report"], _rebuilt_budget()
    design, saved = report["design"], report["certificate"]
    if saved is None:
        assert report["admitted"] is False and report["proposal_backend"] is None
        assert (
            result["verdict"] == "specified_joint_record_prediction_bound_unavailable"
        )
        if not design["admitted"]:
            assert (
                report["reason"]
                == design["reason"]
                == "static_coefficient_budget_exceeded"
            )
        else:
            assert report["reason"].startswith("matrix_proposal_unavailable:")
        return
    assert design["admitted"] is True
    assert report["proposal_backend"].startswith("NumPy ")
    assert "binary64 proposals" in report["proposal_backend"]
    phase = tuple(tuple(map(exact_record, row)) for row in saved["phase_weights"])
    weights = tuple(tuple(map(exact_record, row)) for row in saved["future_weights"])
    assert all(
        (value * 2**40).denominator == 1 for row in phase + weights for value in row
    )
    rebuilt = _certify_dual(
        _StaticDesign(
            tuple(map(exact_record, design["observation_times"])),
            _matrix(design["observation_matrix"], 150, 15),
            _matrix(design["future_matrix"], 5, 15),
            design["steps_per_column"],
            exact_record(design["maximum_coefficient_radius"]),
            exact_record(design["maximum_local_error"]),
            False,  # A cached design verdict is not a premise of residual arithmetic.
            "deliberately_discarded_cached_status",
        ),
        phase,
        weights,
    )
    assert asdict(rebuilt) == saved
    k = max(sum(map(abs, row)) for row in phase)
    w = max(sum(map(abs, row)) for row in weights)
    a, b = budget["remainder_phasor_factor"], budget["remainder_record_factor"]
    margin = 1 - saved["inverse_phasor_defect"] - k * a
    assert saved["phase_weight_norm"] == k and saved["future_weight_norm"] == w
    assert saved["bootstrap_margin"] == margin
    admitted = False
    if margin > 0:
        gain = (k * (1 + b) + saved["inverse_initial_defect"]) / margin
        future_gain = (
            w
            + saved["future_initial_defect"]
            + saved["future_phasor_defect"] * gain
            + (1 + w) * (a * gain + b)
        )
        diameter = (
            budget["historical_pair_error"] * future_gain + 2 * budget["source_error"]
        )
        assert saved["phasor_gain"] == gain
        assert saved["carrier_future_gain"] == future_gain
        assert saved["fine_future_diameter_upper"] == diameter
        assert saved["target_margin"] == Q(2, 10**13) - diameter
        admitted = diameter <= Q(2, 10**13)
        reason = None if admitted else "future_diameter_target_not_certified"
    else:
        assert all(
            saved[name] is None
            for name in (
                "phasor_gain",
                "carrier_future_gain",
                "fine_future_diameter_upper",
                "target_margin",
            )
        )
        reason = "nonpositive_bootstrap_margin"
    assert type(report["admitted"]) is bool and report["admitted"] is admitted
    assert saved["admitted"] is admitted
    assert report["reason"] == saved["reason"] == reason
    assert result["verdict"] == (
        "conditional_joint_record_prediction_sufficient"
        if admitted
        else "specified_joint_record_prediction_bound_unavailable"
    )
