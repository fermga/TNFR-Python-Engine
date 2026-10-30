"""Read the first distinct-neighbor outcome without executing a response.

Every consumed step is rebuilt with the shared full-state arithmetic reader.
Stored derivative/Picard generation and the acquired common source remain
premises; byte associations authenticate neither execution nor acquisition.
The static prediction and its truncation errors never narrow the forward band.
"""

import io
import zipfile
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.physics import test_sine_class_comparison_evidence as views
from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.mathematics._rational_interval import I
from tnfr.physics._sine_class_readout_evidence import _reconstruct_class_readout
from tnfr.physics.relational_sine_class_readout import _admit_class_readout_inputs
from tnfr.research import sine_class_neighbor_forward as policy_owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "class-neighbor-forward-v1"
MEMBER = STEM + ".json"
BASE = "8af5b1932308b81e5fbb4e519de0162b8e82ccfc"
FROZEN = {
    ".protocol.json": (
        152589,
        "bcdc0160bc91e628bb1259139f5451772d2af61298f88e068d55232fef498179",
    ),
    ".source.zip": (
        149640,
        "2317c3a54dd83704d99ad1bcf9ff976d6f25e5cd7cc2754fa5e38726769733ec",
    ),
    ".freeze.json": (
        891,
        "c2dec8230791e8b5377283cd088b1414c105f010d3ada82a4e5e075d8ddc0cdf",
    ),
}
# Pins describe the first retained outcome, not a requested decision.
TRANSPORT_BYTES = 2016937
TRANSPORT_SHA256 = "c62c4edcdf8765028f4af9c9c41b7ff35bc02a9d89bdddfed8c14b34501b6c68"
OUTCOME_BYTES = 29737680
OUTCOME_SHA256 = "e308c8c9432d94372e515f68673471b1e056258ba28f829b79eb9a9048abea97"
ATTEMPT_BYTES = 762
ATTEMPT_SHA256 = "3ed3763c37a7ddc83f075eb1d2ee5cd369d5b88991091f441e6598e96b7bc537"
EXPANSION_LIMIT = 128 * 1024**2
EPS, G, H, M = Q(1, 10**32), Q(1, 3000), Q(1, 8), Q(7, 10000)
DELTA, WIDTH = Q(1, 10**30), Q(1, 10**30)
SELECTORS = dict(first_probe_node=4, second_probe_node=22, readout_node=13)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    def forbidden(*args, **kwargs):
        pytest.fail("retained audit must independently rebuild assessment decisions")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy_owner, "assess_neighbor_forward_bounds", forbidden)
        patch.setattr(policy_owner, "assess_neighbor_forward_report", forbidden)
        yield


def _same_exact_tree(actual, expected):
    # Keep integer, Boolean and exact-fraction encodings distinct.
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(expected), sort_keys=True
    )


def _pinned_bytes(path, size, digest):
    value = read_bytes_bounded(path, max_bytes=size)
    assert len(value) == size and sha256_bytes(value) == digest
    return value


def _inputs(raw):
    def cover(rows):
        assert len(rows) == 27 and all(len(pair) == 2 for pair in rows)
        result = tuple(tuple(map(exact_record, pair)) for pair in rows)
        assert all(lower <= upper for lower, upper in result)
        return result

    real = (
        "first_probe_amplitude",
        "second_probe_amplitude",
        "delay",
        "total_duration",
        "time_step",
    )
    integers = ("order", "max_steps", *SELECTORS)
    assert set(raw) == {"initial_form_bounds", "initial_phase_bounds", *real, *integers}
    return dict(
        initial_form_bounds=cover(raw["initial_form_bounds"]),
        initial_phase_bounds=cover(raw["initial_phase_bounds"]),
        **{key: exact_record(raw[key]) for key in real},
        **{key: views._integer(raw[key]) for key in integers},
    )


@pytest.fixture(scope="module")
def retained(no_scientific_regeneration):
    pins = FROZEN | {
        ".attempt.json": (ATTEMPT_BYTES, ATTEMPT_SHA256),
        ".response.zip": (TRANSPORT_BYTES, TRANSPORT_SHA256),
    }
    paths = {suffix: DIRECTORY / (STEM + suffix) for suffix in pins}
    records = {}
    admitted_bytes = {}
    for suffix, (size, digest) in pins.items():
        value = _pinned_bytes(paths[suffix], size, digest)
        if suffix.endswith(".json"):
            records[suffix] = decode_exact_tree(json_loads(value))
        else:
            admitted_bytes[suffix] = value

    # Both source and response archives are opened from the exact admitted
    # bytes. The bounded source manifest is admitted before inventory use.
    source = admitted_bytes[".source.zip"]
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        with archive.open("source-manifest.json") as stream:
            manifest_bytes = stream.read(1024**2 + 1)
    assert len(manifest_bytes) <= 1024**2
    manifest = decode_exact_tree(json_loads(manifest_bytes))
    expected = {item["path"]: item["sha256"] for item in manifest["files"]}
    assert len(expected) == len(manifest["files"]) == 30
    expected["source-manifest.json"] = sha256_bytes(manifest_bytes)
    verify_archive_members(io.BytesIO(source), expected, max_bytes=4 * 1024**2)
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        assert archive.read(
            "docs/assets/sine_formed_classes/" + STEM + ".protocol.json"
        ) == _pinned_bytes(paths[".protocol.json"], *FROZEN[".protocol.json"])
        for entry in manifest["files"]:
            assert len(archive.read(entry["path"])) == entry["bytes"]

    transport = admitted_bytes[".response.zip"]
    verify_archive_members(
        io.BytesIO(transport), {MEMBER: OUTCOME_SHA256}, max_bytes=EXPANSION_LIMIT
    )
    with zipfile.ZipFile(io.BytesIO(transport)) as archive:
        assert archive.namelist() == [MEMBER]
        assert archive.getinfo(MEMBER).file_size == OUTCOME_BYTES
        outcome = json_loads(archive.read(MEMBER))
    del admitted_bytes, source, transport
    raw_path = DIRECTORY / MEMBER
    if raw_path.exists():
        receipt = file_receipt(raw_path, max_bytes=EXPANSION_LIMIT)
        assert (receipt["bytes"], receipt["sha256"]) == (OUTCOME_BYTES, OUTCOME_SHA256)
    inputs = _inputs(outcome["producer_inputs"])
    _same_exact_tree(inputs, records[".protocol.json"]["producer_inputs"])
    primitives = {key: value for key, value in inputs.items() if key not in SELECTORS}
    admitted = _admit_class_readout_inputs(**primitives)
    evidence = (
        None
        if outcome["report"] is None
        else _reconstruct_class_readout(
            views._Record(outcome["report"]), admitted, **SELECTORS
        )
    )
    # This rebuilds only the existing static theorem; no coefficient or
    # complete-law response is computed and no cached theorem flag is used.
    prediction = policy_owner.neighbor_forward_prediction()
    yield SimpleNamespace(
        records=records,
        outcome=outcome,
        inputs=inputs,
        admitted=admitted,
        evidence=evidence,
        manifest=manifest,
        prediction=prediction,
    )
    for suffix, path in paths.items():
        receipt = file_receipt(path)
        assert (receipt["bytes"], receipt["sha256"]) == pins[suffix]


def test_exact_attempt_freeze_outcome_and_source_association(retained):
    protocol, freeze, attempt = (
        retained.records[suffix]
        for suffix in (".protocol.json", ".freeze.json", ".attempt.json")
    )
    outcome, manifest = retained.outcome, retained.manifest
    for record, kind in (
        (protocol, "protocol"),
        (freeze, "freeze"),
        (attempt, "attempt"),
        (outcome, "outcome"),
    ):
        assert record["schema"] == f"tnfr.sine-class-neighbor-forward-{kind}.v1"
    assert manifest["schema"] == "tnfr.sine-class-neighbor-forward-source-snapshot.v1"
    assert (
        protocol["source_base_commit"]
        == freeze["source_base_commit"]
        == attempt["source_base_commit"]
        == manifest["source_base_commit"]
        == BASE
    )
    assert (
        protocol["runtime_overlays"]
        == freeze["runtime_overlays"]
        == manifest["runtime_overlays"]
        == ()
    )
    assert (
        protocol["evaluation_status_at_freeze"]
        == freeze["evaluation_status_at_freeze"]
        == "not_evaluated"
    )
    assert attempt["freeze_sha256"] == FROZEN[".freeze.json"][1]
    assert datetime.fromisoformat(attempt["started_utc"]).tzinfo is not None
    assert attempt["seed"] is None
    assert (
        attempt["producer"]
        == "tnfr.physics.relational_sine_class_readout.bound_sine_class_four_history_readout"
    )
    assert attempt["attempt_policy"] == "one_fixed_attempt_no_retry_or_adaptation"
    _same_exact_tree(attempt["runtime_environment"], protocol["runtime_environment"])
    _same_exact_tree(
        decode_exact_tree(outcome["attempt"]),
        dict(
            path="docs/assets/sine_formed_classes/" + STEM + ".attempt.json",
            bytes=ATTEMPT_BYTES,
            sha256=ATTEMPT_SHA256,
        ),
    )
    assert type(outcome["elapsed_seconds"]) is float and outcome["elapsed_seconds"] > 0
    assert {
        item["path"]: (item["bytes"], item["sha256"]) for item in freeze["artifacts"]
    } == {
        "docs/assets/sine_formed_classes/" + STEM + suffix: FROZEN[suffix]
        for suffix in (".protocol.json", ".source.zip")
    }
    assert protocol["source_specification"]["prior_artifact_receipts"] == ()
    _same_exact_tree(protocol["static_prediction"], asdict(retained.prediction))


def test_canonical_absolute_source_and_selected_events_are_independent_of_prediction(
    retained,
):
    inputs = retained.inputs
    source = retained.records[".protocol.json"]["source_covers"]
    assert tuple(map(views._integer, source["classes"])) == (1, 2, 1)
    pi = _pi_bounds()
    assert tuple(map(exact_record, source["pi_bounds"])) == pi
    assert inputs["initial_form_bounds"] == ((Q(0), Q(0)),) * 27
    for node in range(27):
        factor = Q(2 * (2 if 9 <= node < 18 else 1) * (node % 9 - 4), 9)
        ends = tuple(factor * value for value in pi)
        target = min(ends), max(ends)
        assert inputs["initial_phase_bounds"][node] == target
        assert source["reference_phase_bounds"][node] == target
        assert source["reference_form_bounds"][node] == (0, 0)
        assert source["actual_phase_bounds"][node] == (target[0] - EPS, target[1] + EPS)
        assert source["actual_form_bounds"][node] == (-EPS, EPS)
    assert {key: inputs[key] for key in SELECTORS} == SELECTORS
    assert inputs["first_probe_amplitude"] == inputs["second_probe_amplitude"] == M
    assert inputs["delay"] == 0 and inputs["total_duration"] == H
    assert inputs["time_step"] == Q(1, 128)
    assert inputs["order"] == 16 and inputs["max_steps"] == 64
    assert set(inputs).isdisjoint(
        {"prediction", "readout_error_bound", "mediator_class", "source_handoff"}
    )


def _source_error():
    ell, d = 1 - 2 * G * H, 1 - 2 * G**2 * H**2
    terms = tuple(
        4 * G * H * EPS / (ell * d) * (G * a / d + EPS / ell) for a in (0, M, M, 2 * M)
    )
    assert terms[0] > 0
    return sum(terms, Q(0))


def _decisions(pair):
    lower, upper = pair
    oriented = lower if lower > 0 else -upper if upper < 0 else None
    orientation = 1 if lower > 0 else -1 if upper < 0 else 0
    sign = None if oriented is None else oriented - 4 * DELTA
    null = None if oriented is None else oriented - 8 * DELTA
    cancellation = 4 * DELTA - max(abs(lower), abs(upper))
    true_sign, recorded, excluded = (
        value is not None and value > 0 for value in (oriented, sign, null)
    )
    status = (
        "zero_contrast_record_sets_disjoint"
        if excluded
        else (
            "recorded_sign_certified"
            if recorded
            else "true_sign_certified" if true_sign else "bounds_only"
        )
    )
    return dict(
        true_bounds=pair,
        recorded_bounds=(lower - 4 * DELTA, upper + 4 * DELTA),
        orientation=orientation,
        oriented_lower=oriented,
        recorded_sign_margin=sign,
        null_separation_margin=null,
        noise_ceiling=oriented / 4 if true_sign else None,
        true_sign=true_sign,
        recorded_sign=recorded,
        null_excluded=excluded,
        cancellation_margin=cancellation,
        scalar_cancellation=cancellation >= 0,
        status=status,
    )


@pytest.fixture(scope="module")
def comparison(retained):
    evidence, prediction = retained.evidence, retained.prediction
    assert evidence is not None and evidence.complete
    source = _source_error()
    reference = evidence.mixed_bounds.lo, evidence.mixed_bounds.hi
    actual = reference[0] - source, reference[1] + source
    direct = prediction.static_cone.direct_cubic_heat_bounds
    truncation = (
        prediction.complete_cubic_correction_upper_bound
        + prediction.higher_amplitude_error_upper_bound
    )
    nominal = direct[0] - truncation, direct[1] + truncation
    predicted = nominal[0] - source, nominal[1] + source
    assert predicted == prediction.decision.true_bounds
    nominal_overlap = max(reference[0], nominal[0]) <= min(reference[1], nominal[1])
    actual_overlap = max(actual[0], predicted[0]) <= min(actual[1], predicted[1])
    narrow = actual[1] - actual[0] <= WIDTH
    decision = _decisions(actual)
    status = (
        "consistency_conflict"
        if not (nominal_overlap and actual_overlap)
        else (
            "numerically_unresolved"
            if not narrow
            else (
                "all_conditions_met"
                if decision["orientation"] == -1 and decision["null_excluded"]
                else "discrimination_unresolved"
            )
        )
    )
    return dict(
        reference_bounds=reference,
        actual_bounds=actual,
        reference_width=reference[1] - reference[0],
        actual_width=actual[1] - actual[0],
        source_error_upper_bound=source,
        nominal_prediction_bounds=nominal,
        actual_prediction_bounds=predicted,
        nominal_prediction_overlap=nominal_overlap,
        actual_prediction_overlap=actual_overlap,
        width_within_allowance=narrow,
        decision=decision,
        additive_recorded_bounds=(-4 * DELTA, 4 * DELTA),
        all_conditions_met=status == "all_conditions_met",
        status=status,
    )


def test_all_retained_steps_and_independent_assessment_match(retained, comparison):
    expected = dict(
        evidence=asdict(retained.evidence),
        comparison=comparison,
        all_conditions_met=retained.evidence.complete
        and comparison["all_conditions_met"],
        status=comparison["status"],
    )
    _same_exact_tree(decode_exact_tree(retained.outcome["assessment"]), expected)
    assert retained.outcome["status"] == expected["status"]
    evidence, raw = retained.evidence, retained.outcome["report"]
    assert (
        evidence.planned_step_count
        == evidence.attempted_step_count
        == evidence.completed_step_count
        == 64
    )
    assert tuple(len(segment["steps"]) for segment in raw["segments"]) == (
        0,
        0,
        16,
        16,
        16,
        16,
    )
    assert (
        raw["failed_segment_index"] is None and raw["unattempted_segment_indices"] == []
    )
    source = tuple(
        I(*pair)
        for pair in retained.inputs["initial_form_bounds"]
        + retained.inputs["initial_phase_bounds"]
    )
    for index, (first, second) in enumerate(((0, 0), (M, 0), (0, M), (M, M))):
        segment = views._Record(raw["segments"][index + 2])
        expected_initial = list(source)
        if first:
            expected_initial[4] += first
        if second:
            expected_initial[22] += second
        assert segment.initial_box == tuple(expected_initial)
        assert segment.initial_box[27:] == source[27:]
        assert evidence.endpoint_bounds[index] == segment.final_state_bounds[13]
    increments = evidence.suffix_increment_bounds
    assert (
        evidence.mixed_bounds
        == increments[0] - increments[1] - increments[2] + increments[3]
    )


def test_source_and_four_reading_noise_are_charged_once(retained, comparison):
    policy = retained.records[".protocol.json"]["observation_policy"]
    source = _source_error()
    for key, value in dict(
        endpoint_radius=EPS,
        gamma_upper=G,
        total_duration=H,
        source_error_upper_bound=source,
        readout_error_bound=DELTA,
        actual_width_allowance=WIDTH,
    ).items():
        assert exact_record(policy[key]) == value
    assert (
        views._integer(policy["reading_count"])
        == views._integer(policy["independent_additive_reading_count"])
        == 4
    )
    assert policy["mixed_coefficients"] == (1, -1, -1, 1)
    assert source < Q(1556, 10**45)
    assert comparison["actual_width"] == comparison["reference_width"] + 2 * source
    errors = tuple(
        sum(c * e for c, e in zip((1, -1, -1, 1), corner))
        for corner in product((-DELTA, DELTA), repeat=4)
    )
    assert min(errors) == -4 * DELTA and max(errors) == 4 * DELTA
    worst = comparison["actual_bounds"][1] + max(errors) - min(errors)
    assert -worst == comparison["decision"]["null_separation_margin"]
    assert (
        comparison["decision"]["recorded_bounds"][1]
        - comparison["additive_recorded_bounds"][0]
        == worst
    )


def test_first_retained_outcome_has_the_reconstructed_exact_band(comparison):
    assert comparison["reference_bounds"] == (
        Q(-1456691539, 2**126),
        Q(-2913382991, 2**127),
    )
    assert comparison["status"] == "all_conditions_met"
    assert comparison["all_conditions_met"] is True
    assert comparison["nominal_prediction_overlap"] is True
    assert comparison["actual_prediction_overlap"] is True
    assert comparison["width_within_allowance"] is True
    assert comparison["decision"]["true_sign"] is True
    assert comparison["decision"]["recorded_sign"] is True
    assert comparison["decision"]["null_excluded"] is True
    # Conservative descriptions of the observed first outcome, not new gates.
    assert comparison["actual_width"] < Q(1, 10**36)
    assert comparison["decision"]["null_separation_margin"] > Q(9, 10**30)


@pytest.mark.parametrize(
    "tamper", ("selector", "source", "event", "increment", "mixed")
)
def test_selective_retained_corruption_cannot_replace_evidence(retained, tamper):
    # Shallow copies preserve the one decoded tree; only one altered path is new.
    raw = retained.outcome["report"]
    altered = dict(raw)
    if tamper == "selector":
        altered["second_probe_node"] = True
    elif tamper == "source":
        altered["initial_receiver_form_bounds"] = encode_exact_tree(asdict(I(42)))
    elif tamper == "mixed":
        altered["mixed_readout_bounds"] = encode_exact_tree(asdict(I(42)))
    else:
        segments = list(raw["segments"])
        index = 4 if tamper == "event" else 2
        segment = dict(segments[index])
        if tamper == "event":
            changed = list(segment["initial_box"])
            changed[4] = encode_exact_tree(asdict(I(M)))
            changed[22] = encode_exact_tree(asdict(I(0)))
            segment["initial_box"] = changed
        else:
            steps = list(segment["steps"])
            step = dict(steps[0])
            changed = list(step["increment"])
            changed[13] = encode_exact_tree(asdict(I(42)))
            step["increment"] = changed
            steps[0] = step
            segment["steps"] = steps
        segments[index] = segment
        altered["segments"] = segments
    with pytest.raises((ValueError, TypeError)):
        _reconstruct_class_readout(
            views._Record(altered), retained.admitted, **SELECTORS
        )
