"""Read-only reconstruction of the fixed record-only software demonstration.

No proposal, candidate validation, coefficient design or trajectory is rerun.
Compact Taylor derivatives retain their frozen conditional proof premises;
the audit rebuilds source/record association, ball propagation and the final
sample-residual/error decision. Receipts do not authenticate execution.
"""

from dataclasses import asdict
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import subprocess
import zipfile

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_observed_forecast"
REVISION = "53917ac7d05c4fd9c79c8ce37ed1c95c25587f50"
DECLARATION = "theory/nodal/ORGANIZED_OBSERVED_FORECAST.md"
EVALUATOR = "benchmarks/organized_observed_forecast.py"
CONSUMER = "src/tnfr/physics/_organized_observed_forecast.py"
PROPOSAL = "src/tnfr/physics/_organized_forecast_proposal.py"
INPUTS = "src/tnfr/research/_organized_forecast_inputs.py"
MARKER = (
    b"<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->"
)
EXPANSION_LIMIT = 128 * 1024**2


def _decode(raw):
    return decode_exact_tree(json_loads(raw))


def _interval(value):
    assert set(value) == {"lo", "hi"}
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(lo, hi)
    assert (result.lo, result.hi) == (lo, hi)
    return result


def _coordinates(values, count=10):
    assert len(values) == count
    return tuple(map(exact_record, values))


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="module")
def source_snapshot():
    # Git reads precede the execution guard's complete subprocess exclusion.
    try:
        _git("cat-file", "-e", REVISION + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "listing": _git(
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            REVISION,
            "--",
            "src/tnfr",
            EVALUATOR,
            DECLARATION,
            "pyproject.toml",
        ),
        "src": _git("rev-parse", REVISION + ":src").decode("ascii").strip(),
        "declaration": _git("show", REVISION + ":" + DECLARATION),
    }


@pytest.fixture(scope="module")
def retained():
    receipt = _decode(read_bytes_bounded(EVIDENCE / "receipt.json", max_bytes=2**20))
    assert receipt == {
        "response_archive": {
            "path": "response.zip",
            "bytes": 208534,
            "sha256": "645071fe5da80881b83181be4ce228320f6e6888093c8e090762d145db5d95a0",
        },
        "result": {
            "path": "result.json",
            "bytes": 2192950,
            "sha256": "0251336e3ee633b921c0f6a5bfef59c27fa19391f9d3d9d512f9c0d2b61ba5b2",
        },
    }
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
        raw = stream.read("result.json")
    assert len(raw) == member["bytes"]
    assert hashlib.sha256(raw).hexdigest() == member["sha256"]
    attempt_raw = read_bytes_bounded(EVIDENCE / "attempt.json", max_bytes=2**22)
    return _decode(attempt_raw), _decode(raw), attempt_raw


@pytest.fixture(scope="module", autouse=True)
def exclude_producers_and_preserve_evidence(retained, source_snapshot):
    from tnfr.mathematics import _validated_linear_forcing as linear
    from tnfr.physics import _organized_computable_prediction as original
    from tnfr.physics import _organized_forecast_proposal as proposer
    from tnfr.physics import _organized_observed_forecast as candidate
    from tnfr.physics import _organized_observed_prediction as observed

    paths = tuple(
        ROOT / "docs/assets" / name / filename
        for name in (
            "organized_observed_forecast",
            "organized_computable_prediction",
            "organized_observed_prediction",
        )
        for filename in ("attempt.json", "receipt.json", "response.zip")
    ) + tuple(ROOT / path for path in (EVALUATOR, CONSUMER, PROPOSAL, INPUTS))
    before = tuple(file_receipt(path) for path in paths)

    def forbidden(*args, **kwargs):
        pytest.fail("forecast evidence audit attempted proposal or regeneration")

    targets = (
        (
            proposer,
            ("propose_candidate", "_historical_response", "_numeric_carrier_field"),
        ),
        (
            candidate,
            (
                "_assess_organized_observed_forecast",
                "validate_observed_candidate",
                "_forecast_leg",
            ),
        ),
        (original, ("_forecast_leg", "validated_metric_taylor_step")),
        (
            observed,
            (
                "_assess_organized_observed_prediction",
                "_static_design",
                "_dual_proposal",
                "_edge_forcing",
                "validated_linear_forcing_step",
            ),
        ),
        (linear, ("validated_linear_forcing_step",)),
    )
    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, names in targets:
            for name in names:
                patch.setattr(module, name, forbidden)
        yield
    assert tuple(file_receipt(path) for path in paths) == before


@pytest.fixture(scope="module")
def rebuilt_inputs(exclude_producers_and_preserve_evidence):
    from tnfr.research._organized_forecast_inputs import (
        extract_archived_form_record,
        load_archived_prediction_gain,
    )

    return extract_archived_form_record(ROOT), load_archived_prediction_gain(ROOT)


def test_source_declaration_attempt_and_archive_associations(retained, source_snapshot):
    attempt, result, raw_attempt = retained
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_observed_forecast_attempt_v1"
    assert result["schema"] == "organized_observed_forecast_result_v1"
    assert type(attempt["maximum_assessments"]) is int
    assert type(result["assessment_count"]) is int
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert attempt["arithmetic_method"] == (
        "rational_outward_dyadic128_machin_trigonometry_v1"
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
    assert {DECLARATION, EVALUATOR, CONSUMER, PROPOSAL, INPUTS} <= set(inventory)
    for record in (attempt, result):
        assert set(record["imported_sources"].values()) <= set(inventory)
    for module, path in (
        ("tnfr.physics._organized_observed_forecast", CONSUMER),
        ("tnfr.physics._organized_forecast_proposal", PROPOSAL),
        ("tnfr.research._organized_forecast_inputs", INPUTS),
    ):
        assert result["imported_sources"][module] == path
    frozen = source_snapshot["declaration"].replace(b"\r\n", b"\n")
    current = (ROOT / DECLARATION).read_bytes().replace(b"\r\n", b"\n")
    assert frozen.count(MARKER) == current.count(MARKER) == 1
    assert frozen.split(MARKER)[0] == current.split(MARKER)[0]


def test_raw_record_and_gain_are_rebuilt_from_original_primitive_evidence(
    retained, rebuilt_inputs
):
    report = retained[1]["report"]
    record, gain = rebuilt_inputs
    assert report["record"] == asdict(record)
    assert report["gain_provenance"] == asdict(gain.provenance)
    assert exact_record(report["rebuilt_gain"]) == gain.carrier_future_gain
    assert exact_record(report["containing_class_source_error"]) == gain.source_error
    assert len(record.times) == len(record.values) == 30
    assert record.times[0] == 200 and record.times[-1] == Q(21400015, 100000)
    assert all(len(row) == 5 for row in record.values)
    assert all(0 < error <= Q(1, 10**14) for error in record.extraction_errors)
    assert report["evaluation_kind"] == (
        "archived_synthetic_record_only_software_demonstration"
    )


def test_proposal_and_exact_candidate_keep_the_same_record_association(
    retained, rebuilt_inputs
):
    from tnfr.physics._organized_observed_forecast import _admit_candidate

    report = retained[1]["report"]
    proposal, forecast = report["proposal"], report["forecast"]
    assert type(proposal["optimization_success"]) is bool
    assert type(proposal["nfev"]) is int and 0 <= proposal["nfev"] <= 80
    assert type(proposal["forward_calls"]) is int and proposal["forward_calls"] >= 0
    backend = json_loads(proposal["backend"])
    assert backend["arithmetic"] == "binary64"
    for library in ("numpy", "scipy"):
        assert backend[library]["version"]
        assert set(backend[library]["linear_algebra"]) == {"blas", "lapack"}
    if proposal["initial_candidate"] is None:
        assert forecast is None and report["admitted"] is False
        assert report["reason"] == proposal["reason"]
        return
    candidate = _admit_candidate(_coordinates(proposal["initial_candidate"]))
    assert proposal["optimization_success"] is True
    assert type(proposal["optimization_status"]) is int
    assert proposal["reason"] is None and forecast is not None
    assert exact_record(proposal["maximum_proposal_residual"]) >= 0
    assert max(map(abs, candidate[:5])) <= Q(1, 100) and candidate[5] == 0
    for values in (candidate, _coordinates(proposal["proposed_start"])):
        for coordinate, value in enumerate(values):
            assert (coordinate < 5 and abs(value) == Q(1, 100)) or (
                value.denominator & (value.denominator - 1) == 0
            )
    record, _gain = rebuilt_inputs
    assert _coordinates(forecast["times"], 30) == record.times
    assert _coordinates(forecast["readings"], 150) == tuple(
        value for row in record.values for value in row
    )
    assert _coordinates(forecast["initial_candidate"]) == candidate
    assert _coordinates(forecast["trajectory"]["initial_center"]) == candidate
    assert exact_record(forecast["trajectory"]["initial_radius"]) == 0


def _edge_rates(time, tube, stage):
    gamma = Q(1, 1023) / pi_interval()
    if stage.child is None:
        command = (I(0),) * 5
    else:
        lo = (time.lo - stage.start) / stage.duration
        hi = (time.hi - stage.start) / stage.duration
        points = [lo, hi] + [r for r in (Q(0), Q(1)) if lo <= r <= hi]
        levels = [stage.base + stage.increment * (3 * r**2 - 2 * r**3) for r in points]
        level = I(min(levels), max(levels))
        command = tuple(level * (Q(int(c == stage.child)) - Q(1, 5)) for c in range(5))
    phases = tuple(value + offset for value, offset in zip(tube[5:], command))
    gradient, current = [I(0) for _ in range(5)], [I(0) for _ in range(5)]
    for tail in range(5):
        head = (tail + 1) % 5
        difference = tube[tail] - tube[head]
        flow = sin(phases[head] - phases[tail])
        gradient[tail] += difference
        gradient[head] -= difference
        current[tail] += flow
        current[head] -= flow
    return tuple((-x + gamma * j) / 4 for x, j in zip(gradient, current)) + tuple(
        gamma * x / 4 for x in gradient
    )


@pytest.fixture(scope="module")
def rebuilt_trajectory(retained, exclude_producers_and_preserve_evidence):
    from tnfr.physics._organized_computable_prediction import _forecast_stages
    from tnfr.research._organized_forecast_inputs import _audit_metric_step

    forecast = retained[1]["report"]["forecast"]
    if forecast is None:
        return None
    leg = forecast["trajectory"]
    center, radius = _coordinates(leg["initial_center"]), exact_record(
        leg["initial_radius"]
    )
    assert leg["name"] == "record_only_candidate" and radius == 0
    elapsed = exact_record(leg["start"])
    assert elapsed == 200
    stages = _forecast_stages()[1:]
    stage_index = 0
    states = {elapsed: (center, radius)}
    for step in leg["steps"]:
        while elapsed == stages[stage_index].start + stages[stage_index].duration:
            stage_index += 1
        stage = stages[stage_index]
        duration = min(Q(1, 8), stage.start + stage.duration - elapsed)
        tube = tuple(map(_interval, step["tube"]))
        initial = tuple(I(value - radius, value + radius) for value in center)
        rates = _edge_rates(I(elapsed, elapsed + duration), tube, stage)
        image = tuple(
            value + rate * I(0, duration) for value, rate in zip(initial, rates)
        )
        assert all(
            bound.lo < value.lo <= value.hi < bound.hi
            for value, bound in zip(image, tube)
        )
        center, radius = _audit_metric_step(
            step, center, radius, time=elapsed, duration=duration
        )
        elapsed += duration
        states[elapsed] = (center, radius)
    assert elapsed == exact_record(leg["end"])
    assert center == _coordinates(leg["endpoint_center"])
    assert radius == exact_record(leg["endpoint_radius"])
    assert len(leg["steps"]) <= 143
    if leg["admitted"]:
        assert type(leg["admitted"]) is bool
        assert elapsed == 216 and len(leg["steps"]) == 143
        assert leg["reason"] is None and leg["failed_tube"] is None
    else:
        assert forecast["error"] is None and forecast["admitted"] is False
        assert forecast["reason"] == leg["reason"]
    return states


def test_one_carried_candidate_rebuilds_every_sample_and_numerical_endpoint(
    retained, rebuilt_inputs, rebuilt_trajectory
):
    forecast = retained[1]["report"]["forecast"]
    if forecast is None or not forecast["trajectory"]["admitted"]:
        return
    record, _gain = rebuilt_inputs
    assert all(time in rebuilt_trajectory for time in record.times)
    assert Q(216) in rebuilt_trajectory
    error = forecast["error"]
    samples = tuple(
        I(value - radius, value + radius)
        for time in record.times
        for center, radius in (rebuilt_trajectory[time],)
        for value in center[:5]
    )
    center, radius = rebuilt_trajectory[Q(216)]
    future = tuple(I(value - radius, value + radius) for value in center[:5])
    assert tuple(map(_interval, error["candidate_samples"])) == samples
    assert tuple(map(_interval, error["candidate_future"])) == future


def test_independent_transfer_arithmetic_and_scoped_decision(
    retained, rebuilt_inputs, rebuilt_trajectory
):
    result = retained[1]
    report, record_gain = result["report"], rebuilt_inputs
    forecast = report["forecast"]
    admitted = False
    if forecast is not None and forecast["trajectory"]["admitted"]:
        record, gain = record_gain
        error = forecast["error"]
        readings = tuple(value for row in record.values for value in row)
        samples = tuple(map(_interval, error["candidate_samples"]))
        future = tuple(map(_interval, error["candidate_future"]))
        source, eta = gain.source_error, Q(1, 10**14)
        residual = max(
            max(abs(box.lo - value), abs(box.hi - value))
            for box, value in zip(samples, readings)
        )
        discrepancy = eta + source + residual
        exact_error = gain.carrier_future_gain * discrepancy + source
        centers = tuple(box.midpoint for box in future)
        radii = tuple(exact_error + box.radius for box in future)
        assert _coordinates(error["observations"], 150) == readings
        assert exact_record(error["gain"]) == gain.carrier_future_gain
        assert exact_record(error["source_error"]) == source
        assert exact_record(error["reading_error"]) == eta
        assert exact_record(error["residual_upper"]) == residual
        assert exact_record(error["carrier_sample_difference_upper"]) == discrepancy
        assert exact_record(error["candidate_exact_future_error_upper"]) == exact_error
        assert _coordinates(error["forecast_center"], 5) == centers
        assert _coordinates(error["true_output_radii"], 5) == radii
        assert exact_record(error["maximum_true_output_radius"]) == max(radii)
        assert exact_record(forecast["target_radius"]) == Q(1, 10**13)
        admitted = max(radii) <= Q(1, 10**13)
        if admitted:
            true = tuple(I(c - r, c + r) for c, r in zip(centers, radii))
            reading = tuple(I(c - r - eta, c + r + eta) for c, r in zip(centers, radii))
            assert tuple(map(_interval, forecast["predicted_true_form_bounds"])) == true
            assert (
                tuple(map(_interval, forecast["predicted_reading_bounds"])) == reading
            )
            assert forecast["reason"] is None
        else:
            assert forecast["predicted_true_form_bounds"] is None
            assert forecast["predicted_reading_bounds"] is None
            assert (
                forecast["reason"]
                == "candidate_forecast_radius_exceeds_declared_policy"
            )
        assert forecast["admitted"] is admitted
        assert forecast["scope"] == (
            "conditional_on_original_fine_class_and_rebuilt_gain_evidence"
        )
    assert type(report["admitted"]) is bool and report["admitted"] is admitted
    assert report["reason"] == (
        forecast["reason"] if forecast is not None else report["proposal"]["reason"]
    )
    assert result["verdict"] == (
        "conditional_record_only_forecast_certified"
        if admitted
        else "specified_record_only_forecast_unavailable"
    )
