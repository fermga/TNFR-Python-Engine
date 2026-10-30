"""Read-only arithmetic audit of the one frozen full-state motion attempt.

Recorded derivative jets/remainders retain their frozen producer provenance;
these tests do not independently re-prove them or evaluate another trajectory.
They re-admit the source, check Picard inclusion using its actual held field,
reconstruct endpoint arithmetic, and derive the retained claims from the rows.
Hashes associate bytes; they do not authenticate acquisition or chronology.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.research import (
    local_capacity_history,
    maintained_parent_thol_execution,
    maintained_successor_motion_certificate,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import TIMES, encode_result
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_successor_motion_certificate import prepare_motion
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/successor_nonstationary"
STEP = Q(1, 512)
COUNT = 256
ORDER = 12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def _interval(value):
    _require(isinstance(value, dict) and set(value) == {"lo", "hi"}, "interval record")
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    interval = I(lo, hi)
    _require(interval.lo == lo and interval.hi == hi, "noncanonical interval endpoints")
    return interval


def _box(values):
    _require(type(values) is tuple and len(values) == 42, "complete state required")
    return tuple(_interval(value) for value in values)


def _receipt(root, record):
    name = validate_artifact_path(record["path"])
    path = root / name
    _require(path.resolve().is_relative_to(root.resolve()), "receipt leaves root")
    _require(file_receipt(path, label=name) == record, "artifact receipt differs")


def _principal_margins(state, edges, shifts):
    pi = pi_interval()
    return tuple(
        pi.lo - (state[21 + j] - state[21 + i] - 2 * shift * pi).abs_max
        for (i, j), shift in zip(edges, shifts, strict=True)
    )


def _audit_step(record, *, index, initial, prediction, flow):
    """Reconstruct one retained step without recomputing solution jets."""
    _require(type(record["index"]) is int and record["index"] == index, "step order")
    time = index * STEP
    _require(exact_record(record["time"]) == time, "step chronology")
    step = record["step"]
    if step is None:
        _require(
            isinstance(record["reason"], str) and record["reason"], "failure reason"
        )
        if record["failed_tube"] is not None:
            tube = _box(record["failed_tube"])
            _require(
                all(x.subset_of(y) for x, y in zip(initial, tube, strict=True)),
                "failed tube omits the source",
            )
        return None, None, "enclosure_unavailable"

    _require(exact_record(step["time"]) == time, "Taylor chronology")
    _require(exact_record(step["duration"]) == STEP, "Taylor duration")
    _require(type(step["order"]) is int and step["order"] == ORDER, "Taylor order")
    _require(
        step["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1",
        "Taylor method",
    )
    source = _box(step["initial_box"])
    _require(source == initial, "endpoint carry differs")
    tube = _box(step["tube"])
    _require(
        tuple(map(exact_record, step["domain_lower_bounds"])) == (Q(1),),
        "complete smooth domain",
    )
    # A fresh interval image checks strict inclusion; no cached Picard verdict
    # or derivative coefficients substitute for this actual-law calculation.
    image = tuple(
        x + rate * I(0, STEP) for x, rate in zip(source, flow(tube), strict=True)
    )
    interior = min(
        min(x.lo - bound.lo, bound.hi - x.hi)
        for x, bound in zip(image, tube, strict=True)
    )
    _require(interior > 0, "strict Picard inclusion")
    _require(exact_record(step["picard_interior_margin"]) == interior, "Picard margin")
    series = tuple(tuple(map(_interval, row)) for row in step["series"])
    remainder = _box(step["local_remainder_bounds"])
    increment, endpoint = reconstruct_box_taylor_arithmetic(
        source, tube, series, remainder, STEP, order=ORDER
    )
    _require(increment == _box(step["increment"]), "Taylor increment")
    _require(endpoint == _box(step["endpoint"]), "Taylor endpoint")
    chamber = prediction.chamber
    tube_margins = _principal_margins(tube, chamber.edges, chamber.conditional.shifts)
    _require(
        tuple(map(exact_record, record["tube_principal_margins"])) == tube_margins,
        "cached tube margins",
    )
    combined = tuple(
        max(tube_margin, energy.lo)
        for tube_margin, energy in zip(
            tube_margins, prediction.conditional.horizon_remaining_margins, strict=True
        )
    )
    _require(
        tuple(map(exact_record, record["combined_principal_margins"])) == combined,
        "cached combined margins",
    )
    rates = flow(endpoint)
    velocity = rates[41] - rates[23]
    _require(
        _interval(record["endpoint_relative_velocity"]) == velocity, "cached velocity"
    )
    status = None if min(combined) > 0 else "principal_bound_unresolved"
    return endpoint, (combined, velocity), status


def _audit_summary(records, report, *, initial, prediction, flow):
    rates = flow(initial)
    velocity = rates[41] - rates[23]
    _require(velocity.hi < 0, "initial negative velocity")
    _require(
        _interval(report["initial_relative_velocity"]) == velocity, "initial velocity"
    )
    chamber = prediction.chamber
    minimum = min(
        _principal_margins(initial, chamber.edges, chamber.conditional.shifts)
    )
    _require(minimum > 0, "initial identity")
    _require(len(records) <= COUNT, "step budget")
    accepted, positive, terminal = 0, [], None
    carried = initial
    for index, record in enumerate(records):
        _require(terminal is None, "records after terminal result")
        endpoint, observation, terminal = _audit_step(
            record, index=index, initial=carried, prediction=prediction, flow=flow
        )
        if terminal is None:
            margins, endpoint_velocity = observation
            minimum = min(minimum, min(margins))
            accepted += 1
            carried = endpoint
            if endpoint_velocity.lo > 0:
                positive.append((index + 1) * STEP)
    elapsed = report["elapsed_seconds"]
    _require(
        type(elapsed) in (int, float) and isfinite(elapsed) and elapsed >= 0,
        "elapsed clock",
    )
    if terminal is None:
        terminal = "completed" if accepted == COUNT else "resource_budget_exhausted"
        if accepted != COUNT:
            _require(elapsed >= 900, "resource stop chronology")
    retained = accepted == COUNT
    expected = dict(
        status=terminal,
        attempted_steps=len(records),
        accepted_steps=accepted,
        certified_retention_time=accepted * STEP,
        positive_velocity_times=tuple(positive),
        minimum_retained_margin=minimum,
        full_horizon_retention=retained,
        reversal_certified=bool(positive),
        joint_target_certified=retained and bool(positive),
    )
    _check_summary(report, expected)
    return expected


def _check_summary(report, expected):
    """Read primitive summary scalars before comparing with the rebuilt prefix."""
    normalized = dict(report)
    for key in ("certified_retention_time", "minimum_retained_margin"):
        normalized[key] = exact_record(report[key])
    _require(
        type(report["positive_velocity_times"]) is tuple, "positive times sequence"
    )
    normalized["positive_velocity_times"] = tuple(
        map(exact_record, report["positive_velocity_times"])
    )
    for key, value in expected.items():
        if type(value) in (bool, int):
            _require(type(report[key]) is type(value), f"summary type: {key}")
        _require(normalized[key] == value, f"summary differs: {key}")


@pytest.fixture(scope="module", autouse=True)
def no_reserved_response_or_events():
    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence audit cannot execute a response or event")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_successor_motion_certificate, "certify_motion"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "rk4_coupled_step"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    # A missing result is a missing required record, never a passing skip or a
    # reason to launch the frozen producer from a test.
    result = _read(EVIDENCE / "result.json")
    frozen = _read(EVIDENCE / "freeze.json")
    manifest = _read(EVIDENCE / "source-manifest.json")
    protocol = _read(EVIDENCE / "protocol.json")
    attempt = _read(EVIDENCE / "attempt.json")
    source = _read(EVIDENCE / "source.json")
    before = {
        path.relative_to(EVIDENCE).as_posix(): file_receipt(path)
        for path in EVIDENCE.rglob("*")
        if path.is_file()
    }
    _require(
        result["schema"] == "tnfr.successor-nonstationary-result.v1", "result schema"
    )
    records = []
    for index, item in enumerate(result["records"]):
        _require(
            item["artifact"]["path"] == f"steps/{index:03d}.zip", "archive sequence"
        )
        _require(item["member"]["path"] == f"step-{index:03d}.json", "member sequence")
        _receipt(EVIDENCE, item["artifact"])
        path = EVIDENCE / item["artifact"]["path"]
        member = item["member"]
        verify_archive_members(path, {member["path"]: member["sha256"]})
        with ZipFile(path) as archive:
            raw = archive.read(member["path"])
        _require(len(raw) == member["bytes"], "member byte count")
        _require(sha256_bytes(raw) == member["sha256"], "member digest")
        records.append(decode_exact_tree(json_loads(raw)))
    _require(
        {path.name for path in (EVIDENCE / "steps").iterdir()}
        == {f"{index:03d}.zip" for index in range(len(records))},
        "unreported step artifacts",
    )
    graph = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    ).graph
    prediction, initial, flow = prepare_motion(graph)
    data = dict(
        result=result,
        frozen=frozen,
        manifest=manifest,
        protocol=protocol,
        attempt=attempt,
        source=source,
        records=tuple(records),
        prediction=prediction,
        initial=initial,
        flow=flow,
    )
    # Reuse this complete audit. Malformed controls below inspect just one
    # recorded step or compare with this independently reconstructed summary.
    data["audited_summary"] = _audit_summary(
        data["records"],
        result["report"],
        initial=initial,
        prediction=prediction,
        flow=flow,
    )
    yield data
    after = {
        path.relative_to(EVIDENCE).as_posix(): file_receipt(path)
        for path in EVIDENCE.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_frozen_bytes_archive_and_single_attempt_associations(retained):
    frozen, manifest = retained["frozen"], retained["manifest"]
    assert frozen["schema"] == "tnfr.successor-nonstationary-freeze.v1"
    assert frozen["evaluation_status_at_freeze"] == "not_evaluated"
    assert {item["path"] for item in frozen["artifacts"]} == {
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "source.zip",
    }
    for item in frozen["artifacts"]:
        _receipt(EVIDENCE, item)
    for item in frozen["prior_artifacts"]:
        _receipt(ROOT, item)
    files = manifest["files"]
    assert len(files) == len({item["path"] for item in files})
    expected = {item["path"]: item["sha256"] for item in files}
    assert {
        "src/tnfr/research/maintained_successor_motion.py",
        "src/tnfr/research/maintained_successor_motion_certificate.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/physics/_sine_flow.py",
        "instrument.py",
        "pyproject.toml",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {info.filename: info.file_size for info in archive.infolist()} == {
            item["path"]: item["bytes"] for item in files
        }
    attempt = retained["attempt"]
    assert attempt["schema"] == "tnfr.successor-nonstationary-attempt.v1"
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["seed"] is None
    _receipt(EVIDENCE, attempt["freeze"])
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, retained["result"]["attempt"])
    assert retained["result"]["attempt"]["path"] == "attempt.json"
    assert not (EVIDENCE / "incomplete.json").exists()


def test_static_source_is_readmitted_and_protocol_preserves_full_law(retained):
    prediction, initial = retained["prediction"], retained["initial"]
    rebuilt = decode_exact_tree(
        encode_exact_tree(
            encode_result({"prediction": prediction, "initial_box": initial})
        )
    )
    assert retained["source"] == rebuilt
    protocol = retained["protocol"]
    assert protocol["schema"] == "tnfr.successor-nonstationary-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["node_order"] == prediction.chamber.reset.nodes
    assert protocol["law"] == "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))"
    assert protocol["prior_clock"] == TIMES[-1]
    assert protocol["horizon"] == COUNT * STEP == Q(1, 2)
    assert protocol["step"] == STEP
    assert protocol["maximum_steps"] == COUNT
    assert protocol["order"] == ORDER
    assert protocol["maximum_attempts"] == 1
    assert protocol["wall_budget_seconds_between_steps"] == 900
    assert protocol["seed"] is None
    assert len(initial) == 42 and len(prediction.chamber.edges) == 53
    assert prediction.chamber.numerical_allowance == Q(1, 2**40)
    assert len(prediction.conditional.retained_edge_indices) == 41


def test_saved_prefix_and_summary_follow_actual_law_arithmetic(retained):
    _check_summary(retained["result"]["report"], retained["audited_summary"])


@pytest.mark.parametrize("mutation", ("scalar", "carry", "chronology", "cached_margin"))
def test_altered_step_cannot_borrow_valid_cached_verdicts(retained, mutation):
    record = deepcopy(retained["records"][0])
    assert record["step"] is not None
    if mutation == "scalar":
        record["step"]["initial_box"][0]["lo"] = False
    elif mutation == "carry":
        record["step"]["initial_box"][0]["lo"] -= 1
    elif mutation == "chronology":
        record["time"] = STEP
    else:
        values = record["combined_principal_margins"]
        record["combined_principal_margins"] = (values[0] + 1,) + values[1:]
    with pytest.raises(ValueError):
        _audit_step(
            record,
            index=0,
            initial=retained["initial"],
            prediction=retained["prediction"],
            flow=retained["flow"],
        )


@pytest.mark.parametrize(
    "mutation", ("accepted", "time_float", "margin_float", "positive_float")
)
def test_cached_summary_cannot_override_the_retained_prefix(retained, mutation):
    report = deepcopy(retained["result"]["report"])
    if mutation == "accepted":
        report["accepted_steps"] += 1
    elif mutation == "time_float":
        report["certified_retention_time"] = float(report["certified_retention_time"])
    elif mutation == "margin_float":
        report["minimum_retained_margin"] = float(report["minimum_retained_margin"])
    else:
        times = report["positive_velocity_times"]
        report["positive_velocity_times"] = (
            (float(times[0]),) + times[1:] if times else (False,)
        )
    with pytest.raises(ValueError):
        _check_summary(report, retained["audited_summary"])


def test_retained_interval_endpoints_cannot_be_silently_rounded():
    with pytest.raises(ValueError, match="noncanonical"):
        _interval({"lo": Q(1, 3), "hi": Q(1, 2)})
