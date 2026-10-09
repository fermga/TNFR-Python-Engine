"""Reconstruct the retained single-impulse forward observation without replay.

The shared reader checks all 54 coordinates, event carry and Taylor arithmetic.
Derivative generation and the strict Picard certificate remain execution
premises. Hashes associate retained bytes; they do not authenticate acquisition
or execution. The nominal reference and the actual source family stay separate.
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
from tnfr.physics._sine_class_interface_composition import (
    _bound_cubic_interface_composition,
)
from tnfr.physics._sine_class_port_readout_evidence import _reconstruct_port_readout
from tnfr.physics.relational_sine_class_port_readout import _admit_port_readout_inputs
from tnfr.research import sine_class_collective_forward as policy_owner
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
STEM = "class-collective-forward-v1"
MEMBER = STEM + ".json"
BASE = "362124d03f34940e23b5d84ca29c6c839d6471a1"
FROZEN = {
    ".protocol.json": (
        122816,
        "e47ac667ecedf44d0863dd21a44da7deacc507a742a026cd0fe7d9bfd066f147",
    ),
    ".source.zip": (
        100793,
        "a99b8f05153ebebe4964c7512183d606fb3d1e5317f133e20af57c28833204a3",
    ),
    ".freeze.json": (
        898,
        "919b2d79ba8beddce19664437d9d0c8738142e771c062878b98af986a38da5d2",
    ),
}
# These pins describe the first retained outcome, not a requested verdict.
TRANSPORT_BYTES = 1051693
TRANSPORT_SHA256 = "edb0332fc9c60c5ed9ecec37139d380c941d2165d532a92b5f0cfcc95e84a481"
OUTCOME_BYTES = 12691468
OUTCOME_SHA256 = "ccdd701a4f708b95b4acb460fd79545827c1cc588d5dad553c2caa72a396e194"
ATTEMPT_BYTES = 761
ATTEMPT_SHA256 = "23a3a0faac763d6c964bb9875a83d88c5bd2f19388040f9c4aa37de183a70f16"
EXPANSION_LIMIT = 128 * 1024**2
EPS, G, DELTA = Q(1, 10**32), Q(1, 3000), Q(1, 10**8)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    def forbidden(*args, **kwargs):
        pytest.fail("retained forward audit must rebuild its own decisions")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        for name in (
            "assess_collective_forward_bounds",
            "assess_collective_forward_report",
        ):
            patch.setattr(policy_owner, name, forbidden)
        yield


def _same_exact_tree(actual, expected):
    # A Boolean must not pass merely because Python makes True == 1.
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(expected), sort_keys=True
    )


def _small_record(path):
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=2**20)))


def _inputs(value):
    assert set(value) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "port_impulse",
        "horizon",
        "time_step",
        "order",
        "max_steps",
    }
    return dict(
        initial_form_bounds=views._covers(value["initial_form_bounds"]),
        initial_phase_bounds=views._covers(value["initial_phase_bounds"]),
        port_impulse=tuple(map(exact_record, value["port_impulse"])),
        horizon=exact_record(value["horizon"]),
        time_step=exact_record(value["time_step"]),
        order=views._integer(value["order"]),
        max_steps=views._integer(value["max_steps"]),
    )


@pytest.fixture(scope="module")
def retained(no_scientific_regeneration):
    paths = {
        suffix: DIRECTORY / (STEM + suffix)
        for suffix in (*FROZEN, ".attempt.json", ".response.zip")
    }
    receipts = {suffix: file_receipt(path) for suffix, path in paths.items()}
    for suffix, expected in {
        **FROZEN,
        ".attempt.json": (ATTEMPT_BYTES, ATTEMPT_SHA256),
        ".response.zip": (TRANSPORT_BYTES, TRANSPORT_SHA256),
    }.items():
        assert (receipts[suffix]["bytes"], receipts[suffix]["sha256"]) == expected
    records = {
        suffix: _small_record(path)
        for suffix, path in paths.items()
        if suffix.endswith(".json")
    }
    # Admit the exact transport before opening its member or parsing JSON.
    transport = read_bytes_bounded(paths[".response.zip"], max_bytes=TRANSPORT_BYTES)
    assert len(transport) == TRANSPORT_BYTES
    assert sha256_bytes(transport) == TRANSPORT_SHA256
    verify_archive_members(
        io.BytesIO(transport), {MEMBER: OUTCOME_SHA256}, max_bytes=EXPANSION_LIMIT
    )
    with zipfile.ZipFile(io.BytesIO(transport)) as archive:
        assert archive.namelist() == [MEMBER]
        assert archive.getinfo(MEMBER).file_size == OUTCOME_BYTES
        with io.TextIOWrapper(
            archive.open(MEMBER), encoding="utf-8", newline=""
        ) as stream:
            text = stream.read()
        outcome = json_loads(text)
        del text
    del transport
    # The committed ZIP is authoritative transport; a local raw file is never
    # a fallback, and checking it does not create another decoded report tree.
    raw_path = DIRECTORY / MEMBER
    if raw_path.exists():
        raw = file_receipt(raw_path, max_bytes=EXPANSION_LIMIT)
        assert (raw["bytes"], raw["sha256"]) == (OUTCOME_BYTES, OUTCOME_SHA256)
    inputs = _inputs(outcome["producer_inputs"])
    _same_exact_tree(inputs, records[".protocol.json"]["producer_inputs"])
    admitted = _admit_port_readout_inputs(**inputs)
    # Lazy views expose one step at a time instead of duplicating its tree.
    report = outcome["report"]
    evidence = (
        None
        if report is None
        else _reconstruct_port_readout(views._Record(report), admitted)
    )
    prior = policy_owner.read_collective_prediction(ROOT)
    yield SimpleNamespace(
        records=records,
        outcome=outcome,
        receipts=receipts,
        inputs=inputs,
        evidence=evidence,
        prior=prior,
    )
    assert {suffix: file_receipt(path) for suffix, path in paths.items()} == receipts


def test_exact_attempt_freeze_response_and_prior_association(retained):
    records, outcome = retained.records, retained.outcome
    freeze, protocol, attempt = (
        records[suffix]
        for suffix in (".freeze.json", ".protocol.json", ".attempt.json")
    )
    assert freeze["schema"] == "tnfr.sine-class-collective-forward-freeze.v1"
    assert protocol["schema"] == "tnfr.sine-class-collective-forward-protocol.v1"
    assert attempt["schema"] == "tnfr.sine-class-collective-forward-attempt.v1"
    assert outcome["schema"] == "tnfr.sine-class-collective-forward-outcome.v1"
    assert (
        freeze["source_base_commit"]
        == protocol["source_base_commit"]
        == attempt["source_base_commit"]
        == BASE
    )
    assert (
        freeze["evaluation_status_at_freeze"]
        == protocol["evaluation_status_at_freeze"]
        == "not_evaluated"
    )
    assert attempt["freeze_sha256"] == FROZEN[".freeze.json"][1]
    assert datetime.fromisoformat(attempt["started_utc"]).tzinfo is not None
    assert attempt["seed"] is None
    assert (
        attempt["producer"]
        == "tnfr.physics.relational_sine_class_port_readout.bound_sine_class_port_readout"
    )
    assert attempt["attempt_policy"] == "one_fixed_attempt_no_retry_or_adaptation"
    _same_exact_tree(attempt["runtime_environment"], protocol["runtime_environment"])
    _same_exact_tree(
        decode_exact_tree(outcome["attempt"]),
        {
            "path": "docs/assets/sine_formed_classes/" + STEM + ".attempt.json",
            "bytes": ATTEMPT_BYTES,
            "sha256": ATTEMPT_SHA256,
        },
    )
    assert type(outcome["elapsed_seconds"]) is float and outcome["elapsed_seconds"] > 0
    assert {
        item["path"]: (item["bytes"], item["sha256"]) for item in freeze["artifacts"]
    } == {
        "docs/assets/sine_formed_classes/" + STEM + suffix: FROZEN[suffix]
        for suffix in (".protocol.json", ".source.zip")
    }
    (prior_receipt,) = protocol["source_specification"]["prior_artifact_receipts"]
    assert prior_receipt["path"] == policy_owner.PREDICTION_PATH
    assert prior_receipt["bytes"] == 1016399
    assert prior_receipt["sha256"] == policy_owner.PREDICTION_SHA256
    _same_exact_tree(
        protocol["inspected_prediction"], tuple(asdict(item) for item in retained.prior)
    )


def test_sources_are_nominal_absolute_targets_with_actual_family_separate(retained):
    inputs = retained.inputs
    source = retained.records[".protocol.json"]["source_covers"]
    classes = tuple(tuple(views._integer(k) for k in row) for row in source["classes"])
    assert classes == ((1, 1, 1), (1, 2, 1))
    pi = _pi_bounds()
    assert tuple(map(exact_record, source["pi_bounds"])) == pi
    for index, ks in enumerate(classes):
        assert inputs["initial_form_bounds"][index] == ((Q(0), Q(0)),) * 27
        for node in range(27):
            factor = Q(2 * ks[node // 9] * (node % 9 - 4), 9)
            endpoints = tuple(factor * bound for bound in pi)
            target = min(endpoints), max(endpoints)
            assert inputs["initial_phase_bounds"][index][node] == target
            assert (
                tuple(map(exact_record, source["reference_phase_bounds"][index][node]))
                == target
            )
            assert tuple(
                map(exact_record, source["actual_phase_bounds"][index][node])
            ) == (target[0] - EPS, target[1] + EPS)
            assert tuple(
                map(exact_record, source["actual_form_bounds"][index][node])
            ) == (-EPS, EPS)
    assert inputs["port_impulse"] == (Q(0), Q(7, 10000), Q(0))
    assert inputs["horizon"] == 1 and inputs["time_step"] == Q(1, 16)
    assert inputs["order"] == 12 and inputs["max_steps"] == 32
    assert set(inputs).isdisjoint(
        {"prediction", "mediator_class", "readout_error_bound"}
    )


def _comparison(reference, prior):
    """Independent exact scalar reconstruction; no interval intersection repair."""
    lower, upper = reference.lo, reference.hi
    source = EPS / (1 - 2 * G)
    actual = lower - source, upper + source
    radius = (upper - lower) / 2
    nominal = prior.nominal_full_bounds
    overlap = lower <= nominal[1] and nominal[0] <= upper
    window = prior.actual_full_bounds[0] - Q(1, 10**10), prior.actual_full_bounds[
        1
    ] + Q(1, 10**10)
    contained = window[0] <= actual[0] and actual[1] <= window[1]
    recorded = actual[0] - DELTA, actual[1] + DELTA
    other = (
        prior.actual_comparator_bounds[0] - DELTA,
        prior.actual_comparator_bounds[1] + DELTA,
    )
    gap = actual[0] - prior.actual_comparator_bounds[1] - 2 * DELTA
    narrow = radius <= Q(1, 10**12)
    if not overlap:
        status = "consistency_conflict"
    elif not narrow:
        status = "numerically_unresolved"
    elif not contained:
        status = "resolution_not_met"
    elif gap <= 0:
        status = "discrimination_unresolved"
    else:
        status = "all_conditions_met"
    return dict(
        reference_bounds=(lower, upper),
        actual_full_bounds=actual,
        reference_radius=radius,
        reference_radius_within_budget=narrow,
        nominal_prediction_overlap=overlap,
        actual_resolution_window=window,
        actual_prediction_resolution_met=contained,
        recorded_full_bounds=recorded,
        recorded_comparator_bounds=other,
        recorded_separation_margin=gap,
        comparator_separated=gap > 0,
        all_conditions_met=narrow and overlap and contained and gap > 0,
        status=status,
    )


@pytest.fixture(scope="module")
def reconstructed_assessment(retained):
    evidence = retained.evidence
    if evidence is None:
        return None
    if not evidence.complete:
        return dict(
            status="unavailable",
            evidence=asdict(evidence),
            comparisons=None,
            all_conditions_met=False,
        )
    comparisons = tuple(
        _comparison(band, prior)
        for band, prior in zip(evidence.endpoint_bounds, retained.prior)
    )
    passed = all(row["all_conditions_met"] for row in comparisons)
    return dict(
        status="all_conditions_met" if passed else "complete_with_unmet_conditions",
        evidence=asdict(evidence),
        comparisons=comparisons,
        all_conditions_met=passed,
    )


def test_full_state_step_reconstruction_matches_retained_assessment(
    retained, reconstructed_assessment
):
    outcome = retained.outcome
    assert reconstructed_assessment is not None
    _same_exact_tree(decode_exact_tree(outcome["assessment"]), reconstructed_assessment)
    assert outcome["status"] == reconstructed_assessment["status"]
    # These completion facts describe the observed first result, after shared
    # reconstruction has checked every source, jump and retained Taylor step.
    evidence = retained.evidence
    assert evidence.complete
    assert (
        evidence.planned_step_count
        == evidence.attempted_step_count
        == evidence.completed_step_count
        == 32
    )
    assert evidence.completed_source_count == 2
    assert (
        evidence.failed_source_index is None
        and evidence.unattempted_source_indices == ()
    )
    assert tuple(len(row["steps"]) for row in outcome["report"]["histories"]) == (
        16,
        16,
    )


def test_source_noise_and_resolution_policy_are_separate(
    retained, reconstructed_assessment
):
    policy = retained.records[".protocol.json"]["observation_policy"]
    for key, expected in (
        ("endpoint_radius", EPS),
        ("gamma_upper", G),
        ("linear_source_allowance", EPS / (1 - 2 * G)),
        ("reading_error_per_model", DELTA),
        ("reference_radius_ceiling", Q(1, 10**12)),
        ("actual_prediction_resolution_allowance", Q(1, 10**10)),
    ):
        assert exact_record(policy[key]) == expected
    assert (
        policy["nominal_prediction_comparison"]
        == "closed_interval_overlap_without_intersection"
    )
    for row, prior in zip(reconstructed_assessment["comparisons"], retained.prior):
        lo, hi = row["reference_bounds"]
        alo, ahi = row["actual_full_bounds"]
        assert alo == lo - EPS / (1 - 2 * G) and ahi == hi + EPS / (1 - 2 * G)
        assert row["recorded_full_bounds"] == (alo - DELTA, ahi + DELTA)
        differences = tuple(
            alo + forward_error - prior.actual_comparator_bounds[1] - comparator_error
            for forward_error, comparator_error in product((-DELTA, DELTA), repeat=2)
        )
        assert min(differences) == row["recorded_separation_margin"]


def test_observed_first_outcome_satisfies_reconstructed_prospective_conditions(
    reconstructed_assessment,
):
    assert reconstructed_assessment["status"] == "all_conditions_met"
    assert reconstructed_assessment["all_conditions_met"] is True
    for row in reconstructed_assessment["comparisons"]:
        assert row["reference_radius_within_budget"] is True
        assert row["nominal_prediction_overlap"] is True
        assert row["actual_prediction_resolution_met"] is True
        assert row["comparator_separated"] is True
        # Conservative factual summaries of this retained result, not new
        # stopping criteria or a guarantee about unobserved inputs.
        assert row["reference_radius"] < Q(1, 10**24)
        assert row["recorded_separation_margin"] > Q(1, 40000)


def test_component_substitution_preserves_the_reconstructed_memory_distinction(
    retained, reconstructed_assessment
):
    # Reuse the original source, law, event and observation admission. The new
    # bound changes neither this retained response nor its frozen assessment.
    inputs = retained.inputs
    policy = retained.records[".protocol.json"]["observation_policy"]
    bound = _bound_cubic_interface_composition(
        total_input_variation=sum(map(abs, inputs["port_impulse"]), Q(0)),
        horizon=inputs["horizon"],
        endpoint_radius=exact_record(policy["endpoint_radius"]),
    )
    error = bound.composition_error.form_error_upper_bound
    assert 0 < error < Q(22, 10**30)
    for row in reconstructed_assessment["comparisons"]:
        # Source transport and both recording errors are already in this gap.
        # Charge only the new lifted-response substitution defect, once.
        lo, _ = row["recorded_full_bounds"]
        _, other_hi = row["recorded_comparator_bounds"]
        substitute_gap = lo - error - other_hi
        assert substitute_gap == row["recorded_separation_margin"] - error
        assert substitute_gap > Q(249, 10**7)
