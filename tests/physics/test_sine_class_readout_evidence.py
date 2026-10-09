"""Read-only audit of the single frozen full54 four-history outcome.

The shared arithmetic kernel rebuilds saved polynomial and remainder arithmetic;
it never regenerates a derivative enclosure or a Picard certificate. Those are
retained numerical execution premises. Byte associations and the intent ledger
do not authenticate execution, source acquisition or physical measurements.
"""

import subprocess
import zipfile
from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr._exact_time import exact_or_represented_real
from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "class-nonlinear-readout-v1"
BASE = "fa8e98a9b1bdd755709da481de7fc092b57bfe65"
RESPONSE_BYTES = 191886903
RESPONSE_SHA256 = "35c5c049dd00576f515a8ad3e63969a0bf9c12708ded9376beb7f48d7c9e9be2"
RESPONSE_MEMBER = STEM + ".json"
TRANSPORT_BYTES = 10554263
TRANSPORT_SHA256 = "b2e96205ed6f185a1193e4a936735572be8f59fd9a585c13203d53309875fbc5"
PLAN = (
    ("prefix_unprobed", None, 0, 1, Q(0)),
    ("prefix_first", None, 0, 1, Q(1, 2000)),
    ("neither", 0, 1, 2, Q(0)),
    ("first_only", 1, 1, 2, Q(0)),
    ("second_only", 0, 1, 2, Q(1, 2000)),
    ("both", 1, 1, 2, Q(1, 2000)),
)
CRITERIA = (
    "true_mixed_negative",
    "recorded_mixed_negative",
    "four_record_sets_disjoint",
    "theorem_open_interval_overlap",
    "theorem_open_interval_contains_numerical_band",
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_class_nonlinear_protocol,
        relational_sine_class_readout,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence audit must not run a response or worker")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (_sine_flow, ("_full_sine_field", "_sine_rate_evaluator")),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (relational_sine_class_mediation, ("assess_sine_class_mediation",)),
            (
                relational_sine_class_nonlinear_protocol,
                ("bound_sine_class_nonlinear_protocol", "_heat_cubic_channels"),
            ),
            (
                relational_sine_class_readout,
                (
                    "bound_sine_class_four_history_readout",
                    "_full_sine_field",
                    "validated_box_taylor_step",
                ),
            ),
            (
                relational_sine_two_port_readout,
                ("bound_sine_two_port_readout", "_full_sine_field"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _interval(record):
    assert isinstance(record, dict) and set(record) == {"lo", "hi"}
    lo, hi = (exact_record(record[key]) for key in ("lo", "hi"))
    assert lo <= hi
    value = I(lo, hi)
    # Retained endpoints already lie on the outward grid; no repair is allowed.
    assert (value.lo, value.hi) == (lo, hi)
    return value


def _box(records):
    assert len(records) == 54
    return tuple(map(_interval, records))


def _pair(value):
    return value.lo, value.hi


def _integer(value):
    assert type(value) is int and value >= 0
    return value


def _load(path):
    # A file-read resource cap, not a numerical width or scientific criterion.
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=2**30)))


def _decode_response(content):
    """Decode once, replacing large step subtrees instead of copying the tree."""
    raw = json_loads(content)
    envelope = raw.pop("producer_report")
    saved = decode_exact_tree(raw)
    if envelope is None:
        saved["producer_report"] = None
        return saved
    report = envelope.pop("report")
    segments = report.pop("segments")
    decoded_report = decode_exact_tree(report)
    for index, segment in enumerate(segments):
        steps = segment.pop("steps")
        for step_index, step in enumerate(steps):
            steps[step_index] = decode_exact_tree(step)
        decoded_segment = decode_exact_tree(segment)
        decoded_segment["steps"] = tuple(steps)
        segments[index] = decoded_segment
    decoded_report["segments"] = tuple(segments)
    saved["producer_report"] = decode_exact_tree(envelope)
    saved["producer_report"]["report"] = decoded_report
    return saved


@pytest.fixture(scope="module")
def evidence():
    paths = {
        suffix: DIRECTORY / (STEM + suffix)
        for suffix in (
            ".protocol.json",
            ".source.zip",
            ".freeze.json",
            ".attempt.json",
            ".response.zip",
        )
    }
    assert all(path.is_file() for path in paths.values()), "missing retained outcome"
    before = {suffix: file_receipt(path) for suffix, path in paths.items()}
    saved = {
        suffix: _load(path)
        for suffix, path in paths.items()
        if suffix.endswith(".json")
    }
    transport = paths[".response.zip"]
    assert (before[".response.zip"]["bytes"], before[".response.zip"]["sha256"]) == (
        TRANSPORT_BYTES,
        TRANSPORT_SHA256,
    )
    verify_archive_members(
        transport, {RESPONSE_MEMBER: RESPONSE_SHA256}, max_bytes=256 * 1024**2
    )
    with zipfile.ZipFile(transport) as archive:
        assert archive.getinfo(RESPONSE_MEMBER).file_size == RESPONSE_BYTES
        saved[".json"] = _decode_response(archive.read(RESPONSE_MEMBER))
    # The committed transport is authoritative. An optional local original is
    # compared by streamed bytes only, never preferred as a decoding fallback.
    local_raw = DIRECTORY / RESPONSE_MEMBER
    if local_raw.exists():
        local_receipt = file_receipt(local_raw)
        assert (local_receipt["bytes"], local_receipt["sha256"]) == (
            RESPONSE_BYTES,
            RESPONSE_SHA256,
        )
    yield saved, paths, before
    assert {suffix: file_receipt(path) for suffix, path in paths.items()} == before


def test_immutable_freeze_attempt_and_response_associations(evidence):
    saved, paths, receipts = evidence
    protocol, freeze, attempt, response = (
        saved[suffix]
        for suffix in (".protocol.json", ".freeze.json", ".attempt.json", ".json")
    )
    schemas = {
        ".protocol.json": "protocol",
        ".freeze.json": "freeze",
        ".attempt.json": "attempt",
        ".json": "evaluation",
    }
    for suffix, name in schemas.items():
        assert saved[suffix]["schema"] == f"tnfr.sine-class-nonlinear-readout-{name}.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert freeze["immutable_after_evaluation"] is True
    assert freeze["source_base_commit"] == protocol["source_base_commit"] == BASE
    assert freeze["runtime_overlays"] == protocol["runtime_overlays"] == ()
    assert len(freeze["artifacts"]) == 2
    for item in freeze["artifacts"]:
        path = ROOT / item["path"]
        assert path in (paths[".protocol.json"], paths[".source.zip"])
        receipt = file_receipt(path)
        assert _integer(item["bytes"]) == receipt["bytes"]
        assert item["sha256"] == receipt["sha256"]
    hashes = {
        suffix: receipts[suffix]["sha256"]
        for suffix in (".protocol.json", ".source.zip", ".freeze.json")
    }
    assert attempt["artifact_sha256"] == response["artifact_sha256"] == hashes
    assert response["attempt_sha256"] == receipts[".attempt.json"]["sha256"]
    assert _integer(attempt["producer_call_limit"]) == 1
    assert _integer(attempt["global_step_attempt_limit"]) == 384
    start = datetime.fromisoformat(attempt["started_utc"])
    finish = datetime.fromisoformat(response["finished_utc"])
    assert start.utcoffset() == finish.utcoffset() and start <= finish
    for item in protocol["source_specification"]["prior_artifact_receipts"]:
        prior = file_receipt(ROOT / item["path"])
        assert (item["bytes"], item["sha256"]) == (prior["bytes"], prior["sha256"])


def test_archive_inventory_and_maintained_prospective_prefix(evidence):
    _, paths, _ = evidence
    with zipfile.ZipFile(paths[".source.zip"]) as archive:
        manifest_bytes = archive.read("source-manifest.json")
        manifest = json_loads(manifest_bytes)
        assert manifest["source_base_commit"] == BASE
        entries = manifest["files"]
        hashes = {entry["path"]: entry["sha256"] for entry in entries}
        assert len(hashes) == len(entries)
        hashes["source-manifest.json"] = sha256_bytes(manifest_bytes)
        verify_archive_members(paths[".source.zip"], hashes)
        for entry in entries:
            content = archive.read(entry["path"])
            assert _integer(entry["bytes"]) == len(content)
            if entry["path"].startswith("src/"):
                assert (
                    sha256_bytes(content.replace(b"\r\n", b"\n"))
                    == entry["normalized_lf_sha256"]
                )
        protocol_path = paths[".protocol.json"].relative_to(ROOT).as_posix()
        assert archive.read(protocol_path) == paths[".protocol.json"].read_bytes()
        proof = "theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md"
        assert (
            (ROOT / proof)
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .startswith(archive.read(proof).replace(b"\r\n", b"\n"))
        )


def test_source_cover_and_fixed_policy_are_rebuilt_without_acquisition(evidence):
    protocol = evidence[0][".protocol.json"]
    inputs, source = protocol["producer_inputs"], protocol["source_specification"]
    epsilon = Q(1, 10**32)
    pi = _pi_bounds()
    assert source["pi_bounds"] == pi
    assert tuple(map(_integer, source["classes"])) == (1, 2, 1)
    assert source["endpoint_radius"] == epsilon
    assert inputs["initial_form_bounds"] == ((-epsilon, epsilon),) * 27
    phase = []
    for winding in (1, 2, 1):
        for j in range(9):
            ends = tuple(Q(2 * winding * (j - 4), 9) * value for value in pi)
            phase.append((min(ends) - epsilon, max(ends) + epsilon))
    assert inputs["initial_phase_bounds"] == tuple(phase)
    assert source["original_form_slope"] == Q(2046, 9) * Q(355, 113) ** 2
    assert tuple(map(exact_record, source["phase_origins"])) == (Q(0),) * 3
    assert source["formation_time"] == 100
    assert source["relaxation_duration"] == 10**13
    assert _integer(source["decay_power"]) == 512
    assert source["original_form_error_bound"] == Q(1, 10**10)
    assert source["original_phase_error_bound"] == Q(1, 10**10)
    expected = dict(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
        time_step=Q(1, 64),
        order=16,
        max_steps=384,
    )
    assert set(inputs) == set(expected) | {
        "initial_form_bounds",
        "initial_phase_bounds",
    }
    for key, value in expected.items():
        assert exact_record(inputs[key]) == value
    policy = protocol["numerical_policy"]
    assert policy["numerical_width_cutoff"] is None
    assert policy["adaptive_retries"] is False
    assert protocol["observation"]["readout_error_bound"] == Q(1, 10**30)
    assert protocol["prediction"]["open_interval"] == (Q(-32, 10**28), Q(-3, 10**27))


def _audit_step(step, state, time, width, order):
    assert exact_record(step["time"]) == time
    assert exact_record(step["duration"]) == width
    assert _integer(step["order"]) == order
    assert _box(step["initial_box"]) == state
    assert exact_record(step["picard_interior_margin"]) > 0
    assert tuple(map(exact_record, step["domain_lower_bounds"])) == (Q(1),)
    assert step["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1"
    changes, endpoint = reconstruct_box_taylor_arithmetic(
        state,
        _box(step["tube"]),
        tuple(tuple(map(_interval, row)) for row in step["series"]),
        _box(step["local_remainder_bounds"]),
        width,
        order=order,
    )
    assert _box(step["increment"]) == changes
    assert _box(step["endpoint"]) == endpoint
    return changes, endpoint


def _audit_model(report, protocol):
    edges = tuple(
        sorted(
            {
                tuple(sorted((9 * c + j, 9 * c + (j + 1) % 9)))
                for c in range(3)
                for j in range(9)
            }
            | {(4, 13), (13, 22)}
        )
    )
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))
    assert tuple(map(_integer, report["geometry"]["nodes"])) == tuple(range(27))
    assert (
        tuple(tuple(map(_integer, edge)) for edge in report["geometry"]["edges"])
        == edges
    )
    assert tuple(map(_integer, report["degrees"])) == degrees
    assert protocol["support"]["edges"] == edges
    assert protocol["support"]["degrees"] == degrees
    for key, value in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        assert exact_or_represented_real(report["reference_model"][key], key) == value
    assert report["reference_model"]["phase_domain"] == "regular"
    assert tuple(map(exact_record, report["capacity"])) == (Q(1),) * 27
    assert report["clock"] == "tau=e*t; e=1023/1024"
    assert report["state_order"] == tuple(
        f"{kind}_{i}" for kind in ("x", "theta") for i in range(27)
    )
    assert (_integer(report["donor_node"]), _integer(report["receiver_node"])) == (
        4,
        22,
    )
    assert report["history_order"] == tuple(row[0] for row in PLAN[2:])
    assert tuple(
        tuple(map(_integer, row)) for row in report["history_segment_indices"]
    ) == ((0, 2), (1, 3), (0, 4), (1, 5))
    assert tuple(map(exact_record, report["mixed_readout_coefficients"])) == (
        1,
        -1,
        -1,
        1,
    )
    assert tuple(
        tuple(map(exact_record, row)) for row in report["history_event_amplitudes"]
    ) == tuple(
        tuple(map(exact_record, row)) for row in protocol["events"]["form_amplitudes"]
    )
    inputs = protocol["producer_inputs"]
    source = tuple(
        I(*map(exact_record, pair))
        for key in ("initial_form_bounds", "initial_phase_bounds")
        for pair in inputs[key]
    )
    assert _box(report["source_box"]) == source
    assert tuple(map(_interval, report["initial_form_bounds"])) == source[:27]
    assert tuple(map(_interval, report["initial_phase_bounds"])) == source[27:]
    assert _interval(report["initial_receiver_form_bounds"]) == source[22]
    for key in inputs.keys() - {"initial_form_bounds", "initial_phase_bounds"}:
        admit = _integer if key in ("order", "max_steps") else exact_record
        assert admit(report[key]) == inputs[key]
    return source


def _audit_report(report, protocol):
    source = _audit_model(report, protocol)
    inputs = protocol["producer_inputs"]
    assert len(report["segments"]) == len(PLAN)
    finals, increments, reasons = {}, {}, []
    failure = None
    completed = attempted = 0
    for index, (segment, plan) in enumerate(zip(report["segments"], PLAN)):
        label, parent, start, end, jump = plan
        assert segment["label"] == label
        assert (
            segment["parent_segment_index"] is None
            if parent is None
            else (_integer(segment["parent_segment_index"]) == parent)
        )
        for key, value in (
            ("start_time", start),
            ("end_time", end),
            ("form_jump", jump),
        ):
            assert exact_record(segment[key]) == value
        if failure is not None:
            assert segment["status"] == "not_attempted" and not segment["steps"]
            assert all(
                segment[key] is None
                for key in (
                    "pre_event_box",
                    "initial_box",
                    "completed_time",
                    "completed_endpoint_box",
                    "completed_receiver_increment_bounds",
                    "final_state_bounds",
                    "failed_initial_box",
                    "failed_time",
                    "failed_tube",
                    "reason",
                )
            )
            continue
        if segment["initial_box"] is None:
            assert segment["status"] == "budget_exhausted"
            assert attempted == inputs["max_steps"] and not segment["steps"]
            assert segment["reason"] == "unique_step_budget_exhausted_before_event"
            assert all(
                segment[key] is None
                for key in (
                    "pre_event_box",
                    "completed_time",
                    "completed_endpoint_box",
                    "completed_receiver_increment_bounds",
                    "final_state_bounds",
                    "failed_initial_box",
                    "failed_time",
                    "failed_tube",
                )
            )
            failure = index
            reasons.append(f"{label}: {segment['reason']}")
            continue
        before = source if parent is None else finals[parent]
        assert _box(segment["pre_event_box"]) == before
        state = tuple(
            value + jump if i == 4 and jump else value for i, value in enumerate(before)
        )
        assert _box(segment["initial_box"]) == state
        time, increment = Q(start), I(0)
        for step in segment["steps"]:
            width = min(inputs["time_step"], end - time)
            assert width > 0 and attempted < inputs["max_steps"]
            changes, state = _audit_step(step, state, time, width, inputs["order"])
            increment += changes[22]
            time += width
            completed += 1
            attempted += 1
        assert exact_record(segment["completed_time"]) == time
        assert _box(segment["completed_endpoint_box"]) == state
        assert _interval(segment["completed_receiver_increment_bounds"]) == increment
        if segment["status"] == "admitted":
            assert time == end and _box(segment["final_state_bounds"]) == state
            assert all(
                segment[key] is None
                for key in (
                    "failed_initial_box",
                    "failed_time",
                    "failed_tube",
                    "reason",
                )
            )
            finals[index], increments[index] = state, increment
        else:
            assert segment["status"] in ("unavailable", "budget_exhausted")
            assert time < end and segment["final_state_bounds"] is None
            assert exact_record(segment["failed_time"]) == time
            assert _box(segment["failed_initial_box"]) == state
            assert isinstance(segment["reason"], str) and segment["reason"]
            if segment["failed_tube"] is not None:
                _box(segment["failed_tube"])
            if segment["status"] == "unavailable":
                assert attempted < inputs["max_steps"]
                attempted += 1
            else:
                assert attempted == inputs["max_steps"]
            failure = index
            reasons.append(f"{label}: {segment['reason']}")
    for key, value in (
        ("completed_step_count", completed),
        ("attempted_step_count", attempted),
        ("completed_segment_count", len(finals)),
        ("planned_unique_step_count", 384),
    ):
        assert _integer(report[key]) == value
    assert (
        report["failed_segment_index"] is None
        if failure is None
        else (_integer(report["failed_segment_index"]) == failure)
    )
    assert tuple(map(_integer, report["unattempted_segment_indices"])) == (
        () if failure is None else tuple(range(failure + 1, 6))
    )
    assert report["unavailable_reasons"] == tuple(reasons)
    readings = tuple((PLAN[i][0], finals[i][22]) for i in range(2, 6) if i in finals)
    assert (
        tuple(
            (name, _interval(value))
            for name, value in report["completed_history_readout_bounds"]
        )
        == readings
    )
    complete = len(readings) == 4
    assert report["status"] == ("admitted" if complete else "unavailable")
    result = dict(
        complete=complete,
        evidence_integrity=True,
        mixed_bounds=None,
        raw_endpoint_mixed_bounds=None,
        endpoint_bounds=None,
        suffix_increment_bounds=None,
    )
    if complete:
        endpoints = tuple(finals[i][22] for i in range(2, 6))
        suffix = tuple(increments[i] for i in range(2, 6))
        raw = endpoints[0] - endpoints[1] - endpoints[2] + endpoints[3]
        mixed = suffix[0] - suffix[1] - suffix[2] + suffix[3]
        assert tuple(map(_interval, report["endpoint_readout_bounds"])) == endpoints
        assert (
            tuple(map(_interval, report["suffix_receiver_increment_bounds"])) == suffix
        )
        assert _interval(report["raw_endpoint_mixed_bounds"]) == raw
        assert _interval(report["mixed_readout_bounds"]) == mixed
        result.update(
            mixed_bounds=_pair(mixed),
            raw_endpoint_mixed_bounds=_pair(raw),
            endpoint_bounds=tuple(map(_pair, endpoints)),
            suffix_increment_bounds=tuple(map(_pair, suffix)),
        )
    else:
        assert all(
            report[key] is None
            for key in (
                "endpoint_readout_bounds",
                "suffix_receiver_increment_bounds",
                "raw_endpoint_mixed_bounds",
                "mixed_readout_bounds",
            )
        )
    return result


@pytest.fixture(scope="module")
def reconstructed(evidence):
    saved = evidence[0]
    response = saved[".json"]
    envelope = response["producer_report"]
    if envelope is None:
        assert response["evaluation_error"] is not None
        return None
    assert envelope["schema"] == "tnfr.sine-class-four-history-readout.v1"
    return _audit_report(envelope["report"], saved[".protocol.json"])


def test_every_retained_step_and_branch_rebuilds_saved_observations(
    evidence, reconstructed
):
    response = evidence[0][".json"]
    assert reconstructed == response["reconstructed_observations"]


def test_decisions_rebuilt_from_unintersected_numerical_bounds(evidence, reconstructed):
    saved = evidence[0]
    response, protocol = saved[".json"], saved[".protocol.json"]
    bounds = None if reconstructed is None else reconstructed["mixed_bounds"]
    values = (None,) * 5
    if bounds is not None:
        lo, hi = bounds
        delta = exact_record(protocol["observation"]["readout_error_bound"])
        p_lo, p_hi = map(exact_record, protocol["prediction"]["open_interval"])
        values = (
            hi < 0,
            hi + 4 * delta < 0,
            hi + 8 * delta < 0,
            hi > p_lo and lo < p_hi,
            p_lo < lo and hi < p_hi,
        )
    expected = dict(zip(CRITERIA, values))
    assert response["criteria"] == expected
    if response["evaluation_error"] is not None:
        assert set(response["evaluation_error"]) == {"type", "message"}
        status = "evaluation_error"
    elif reconstructed is None or not reconstructed["complete"]:
        status = "unavailable"
    elif not expected["theorem_open_interval_overlap"]:
        status = "consistency_conflict"
    elif all(values[:3]):
        status = "certified_reserved_readout"
    else:
        status = "inconclusive"
    assert response["status"] == status


def test_retained_first_outcome_has_the_reconstructed_exact_negative_band(
    evidence, reconstructed
):
    """Pin the observed outcome after arithmetic reconstruction, not as a premise."""
    response = evidence[0][".json"]
    report = response["producer_report"]["report"]
    assert reconstructed["mixed_bounds"] == (
        Q(-263844792565, 2**126),
        Q(-263523400319, 2**126),
    )
    assert report["attempted_step_count"] == report["completed_step_count"] == 384
    assert report["completed_segment_count"] == 6
    assert tuple(len(segment["steps"]) for segment in report["segments"]) == (64,) * 6
    assert report["failed_segment_index"] is None
    assert report["unattempted_segment_indices"] == ()
    assert response["evaluation_error"] is None
    assert response["criteria"] == dict.fromkeys(CRITERIA, True)
    assert response["status"] == "certified_reserved_readout"


@pytest.mark.parametrize(
    "key",
    ("increment", "endpoint", "series", "time", "order", "picard_interior_margin"),
)
def test_selective_saved_step_corruption_cannot_replace_arithmetic(evidence, key):
    envelope = evidence[0][".json"]["producer_report"]
    if envelope is None:
        assert evidence[0][".json"]["evaluation_error"] is not None
        return
    steps = [
        step for segment in envelope["report"]["segments"] for step in segment["steps"]
    ]
    if not steps:
        assert envelope["report"]["completed_step_count"] == 0
        return
    original = steps[0]
    changed = dict(original)
    if key in ("time", "order", "picard_interior_margin"):
        changed[key] = True
    else:
        values = list(original[key])
        bad = {"lo": Q(100), "hi": Q(100)}
        values[0] = (bad,) + values[0][1:] if key == "series" else bad
        changed[key] = tuple(values)
    with pytest.raises((AssertionError, ValueError)):
        _audit_step(
            changed,
            _box(original["initial_box"]),
            original["time"],
            original["duration"],
            original["order"],
        )
