"""Read-only full54 comparison reconstruction and actual-source transfer.

Saved derivative and Picard generation remain numerical execution premises.
This audit reconstructs retained arithmetic, complete ancestry and the consumed
observation. Digests associate bytes, not acquisition or execution authenticity.
The large JSON tree is parsed once; lazy typed views avoid duplicating its
retained coefficients while reusing the shared arithmetic reader.
"""

import io
import subprocess
import zipfile
from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.physics import test_sine_class_readout_evidence as old_audit
from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.mathematics._rational_interval import I
from tnfr.physics._sine_class_readout_evidence import _reconstruct_class_readout
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
STEM = "class-comparison-readout-v1"
MEMBER = STEM + ".json"
BASE = "8e40dac013c026c0f198a2f95131512277286bbf"
FROZEN = {
    ".protocol.json": (
        121655,
        "c7178e6cbaebfead12a66a020e54d274d9c4c7199d899cea1316239c50ad6dab",
    ),
    ".source.zip": (
        1433580,
        "564642f21fea44929f53463e2391a10e569bfe81b72e1c4e964eb3c5610424f7",
    ),
    ".freeze.json": (
        1009,
        "4616665bac697157b11471f98d0dd4efb7a3055f977ed5e7535367e318295840",
    ),
}
# Filled only from the retained first outcome, never chosen to imply success.
TRANSPORT_BYTES = 42090162
TRANSPORT_SHA256 = "c77d5b48a3e77e3e2c826dfb6c850dba73769d861ddc95acbc5bcf1828d8131c"
OUTCOME_BYTES = 808798262
OUTCOME_SHA256 = "a6553e20450454a5ff191b608c35a651bd85a82c9ba9dfc5ed78c98302aff03b"
ATTEMPT_BYTES = 723
ATTEMPT_SHA256 = "0f55f35c32edde0cc5584767d84d03aaeb5113a87c55a8de9004b10c8d5077d7"


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
    )
    from tnfr.research import frozen_source, sine_class_comparison_protocol

    def forbidden(*args, **kwargs):
        pytest.fail(
            "retained comparison audit must not evaluate or regenerate evidence"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (_sine_flow, ("_full_sine_field", "_sine_rate_evaluator")),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                (
                    "bound_sine_class_cubic_response",
                    "_class_cubic_coefficients",
                    "_time_coefficients",
                ),
            ),
            (
                relational_sine_class_readout,
                (
                    "bound_sine_class_four_history_readout",
                    "validated_box_taylor_step",
                    "_full_sine_field",
                ),
            ),
            (
                relational_sine_class_comparison_readout,
                (
                    "bound_sine_class_comparison_readout",
                    "bound_sine_class_four_history_readout",
                ),
            ),
            (
                sine_class_comparison_protocol,
                (
                    "assess_reference_contrast",
                    "comparison_producer_inputs",
                    "canonical_comparison_sources",
                ),
            ),
            (frozen_source, ("restore_frozen_source",)),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _integer(value):
    assert type(value) is int and value >= 0
    return value


def _covers(value):
    assert len(value) == 2 and all(len(cover) == 27 for cover in value)
    result = []
    for cover in value:
        rows = []
        for row in cover:
            assert len(row) == 2
            lower, upper = map(exact_record, row)
            assert lower <= upper
            rows.append((lower, upper))
        result.append(tuple(rows))
    return tuple(result)


def _interval(value):
    assert isinstance(value, dict) and set(value) == {"lo", "hi"}
    lower, upper = (exact_record(value[key]) for key in ("lo", "hi"))
    assert lower <= upper
    interval = I(lower, upper)
    assert (interval.lo, interval.hi) == (lower, upper)
    return interval


def _typed(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return exact_record(value)
        if set(value) == {"lo", "hi"}:
            return _interval(value)
        return _Record(value)
    if isinstance(value, (list, tuple)):
        return tuple(map(_typed, value))
    return value


class _Record:
    """A lazy attribute view; accessing one step does not copy either tree."""

    def __init__(self, mapping):
        self.mapping = mapping

    def __getattr__(self, key):
        return _typed(self.mapping[key])


def _small_record(path):
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=2**20)))


@pytest.fixture(scope="module")
def retained():
    paths = {
        suffix: DIRECTORY / (STEM + suffix)
        for suffix in (*FROZEN, ".attempt.json", ".response.zip")
    }
    assert all(
        path.is_file() for path in paths.values()
    ), "missing retained first outcome"
    receipts = {suffix: file_receipt(path) for suffix, path in paths.items()}
    for suffix, expected in {
        **FROZEN,
        ".attempt.json": (ATTEMPT_BYTES, ATTEMPT_SHA256),
        ".response.zip": (TRANSPORT_BYTES, TRANSPORT_SHA256),
    }.items():
        actual = receipts[suffix]
        assert (actual["bytes"], actual["sha256"]) == expected
    records = {
        suffix: _small_record(path)
        for suffix, path in paths.items()
        if suffix.endswith(".json")
    }
    protocol = records[".protocol.json"]
    limit = _integer(protocol["response_audit_expansion_limit_bytes"])
    assert limit == 2**31
    verify_archive_members(
        paths[".response.zip"], {MEMBER: OUTCOME_SHA256}, max_bytes=limit
    )
    with zipfile.ZipFile(paths[".response.zip"]) as archive:
        assert archive.getinfo(MEMBER).file_size == OUTCOME_BYTES
        # Keep tagged exact scalars in the large parse tree. They are admitted
        # only as consumed; no second decoded copy of all time series is made.
        # Materialize text before parsing, avoiding concurrent raw bytes plus
        # their equally large decoded string during JSON object construction.
        with io.TextIOWrapper(
            archive.open(MEMBER), encoding="utf-8", newline=""
        ) as stream:
            content = stream.read()
        outcome = json_loads(content)
        del content
    raw_path = DIRECTORY / MEMBER
    if raw_path.exists():
        raw_receipt = file_receipt(raw_path, max_bytes=limit)
        assert (raw_receipt["bytes"], raw_receipt["sha256"]) == (
            OUTCOME_BYTES,
            OUTCOME_SHA256,
        )
    yield records, outcome, receipts, paths
    assert {suffix: file_receipt(path) for suffix, path in paths.items()} == receipts


def test_frozen_attempt_outcome_and_archive_association(retained):
    records, outcome, receipts, paths = retained
    freeze, protocol, attempt = (
        records[key] for key in (".freeze.json", ".protocol.json", ".attempt.json")
    )
    assert freeze["schema"] == "tnfr.sine-class-comparison-freeze.v1"
    assert protocol["schema"] == "tnfr.sine-class-comparison-protocol.v1"
    assert freeze["source_base_commit"] == protocol["source_base_commit"] == BASE
    assert (
        freeze["evaluation_status_at_freeze"]
        == protocol["evaluation_status_at_freeze"]
        == "not_evaluated"
    )
    assert freeze["runtime_overlays"] == protocol["runtime_overlays"] == ()
    for item, suffix in zip(freeze["artifacts"], (".protocol.json", ".source.zip")):
        assert item["path"] == f"docs/assets/sine_formed_classes/{STEM}{suffix}"
        assert (_integer(item["bytes"]), item["sha256"]) == FROZEN[suffix]
    assert attempt["schema"] == "tnfr.sine-class-comparison-attempt.v1"
    assert attempt["source_base_commit"] == BASE
    assert (
        _integer(attempt["freeze"]["bytes"]),
        attempt["freeze"]["sha256"],
    ) == FROZEN[".freeze.json"]
    assert outcome["schema"] == "tnfr.sine-class-comparison-outcome.v1"
    assert _integer(outcome["attempt"]["bytes"]) == ATTEMPT_BYTES
    assert outcome["attempt"]["sha256"] == ATTEMPT_SHA256
    started, completed = (
        datetime.fromisoformat(value)
        for value in (attempt["started_utc"], outcome["completed_utc"])
    )
    assert (
        started.tzinfo is not None
        and completed.tzinfo is not None
        and started <= completed
    )
    runtime = attempt["runtime"]
    assert runtime["python"] == protocol["runtime_python"]
    assert runtime["implementation"] == protocol["runtime_implementation"]
    assert runtime["dependencies"] == protocol["runtime_dependencies"]
    assert runtime["platform"] and runtime["randomness"] == "none"
    with zipfile.ZipFile(paths[".source.zip"]) as archive:
        manifest = json_loads(archive.read("source-manifest.json"))
        assert manifest["schema"] == "tnfr.sine-class-comparison-source-snapshot.v1"
        assert (
            manifest["source_base_commit"] == BASE
            and manifest["runtime_overlays"] == []
        )
        inventory = {item["path"]: item["sha256"] for item in manifest["files"]}
        inventory["source-manifest.json"] = sha256_bytes(
            archive.read("source-manifest.json")
        )
        for item in manifest["files"]:
            assert archive.getinfo(item["path"]).file_size == _integer(item["bytes"])
        proof = "theory/nodal/SINE_CLASS_COMPARISON_PROTOCOL.md"
        assert (
            (ROOT / proof)
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .startswith(archive.read(proof).replace(b"\r\n", b"\n"))
        )
    verify_archive_members(paths[".source.zip"], inventory, max_bytes=2**27)
    assert receipts[".source.zip"]["sha256"] == FROZEN[".source.zip"][1]


@pytest.fixture(scope="module")
def admitted(retained):
    protocol = retained[0][".protocol.json"]
    specification = protocol["source_specification"]
    covers = specification["covers"]
    eps, pi = Q(1, 10**32), _pi_bounds()
    assert tuple(map(exact_record, covers["pi_bounds"])) == pi
    classes = tuple(tuple(map(_integer, row)) for row in covers["classes"])
    assert classes == ((1, 1, 1), (1, 2, 1))
    target = []
    for row in classes:
        phase = []
        for winding in row:
            for local in range(9):
                products = tuple(
                    Q(2 * winding * (local - 4), 9) * value for value in pi
                )
                phase.append((min(products), max(products)))
        target.append(tuple(phase))
    references = (((Q(0), Q(0)),) * 27,) * 2
    assert _covers(covers["reference_form_bounds"]) == references
    assert _covers(covers["reference_phase_bounds"]) == tuple(target)
    assert _covers(covers["actual_form_bounds"]) == (((-eps, eps),) * 27,) * 2
    assert _covers(covers["actual_phase_bounds"]) == tuple(
        tuple((lo - eps, hi + eps) for lo, hi in phase) for phase in target
    )
    assert exact_record(specification["reached_component_error_radius"]) == eps
    assert exact_record(specification["formation_source_error_radius"]) == Q(1, 10**10)
    assert specification["reference_is_not_a_preparation_or_hidden_state_reset"] is True
    prior = specification["prior_artifact_receipts"]
    assert len(prior) == 1
    prior_name = "docs/assets/sine_formed_classes/class-cubic-response-v1.evidence.zip"
    assert prior[0]["path"] == prior_name
    prior_receipt = file_receipt(ROOT / prior_name, max_bytes=2**21)
    assert (_integer(prior[0]["bytes"]), prior[0]["sha256"]) == (
        prior_receipt["bytes"],
        prior_receipt["sha256"],
    )
    assert 3000 < 1023 * pi[0]  # The transfer's declared g exceeds actual gamma.
    inputs = protocol["producer_inputs"]
    assert _covers(inputs["initial_form_bounds"]) == references
    assert _covers(inputs["initial_phase_bounds"]) == tuple(target)
    expected = dict(
        first_probe_amplitude=Q(7, 10000),
        second_probe_amplitude=Q(7, 10000),
        delay=Q(1),
        total_duration=Q(2),
        time_step=Q(1, 128),
        order=16,
        max_steps=1536,
    )
    assert set(inputs) == set(expected) | {
        "initial_form_bounds",
        "initial_phase_bounds",
    }
    for key, value in expected.items():
        actual = (
            _integer(inputs[key])
            if key in ("order", "max_steps")
            else exact_record(inputs[key])
        )
        assert actual == value
    model = protocol["complete_model"]
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
    assert tuple(map(_integer, model["nodes"])) == tuple(range(27))
    assert tuple(tuple(map(_integer, row)) for row in model["edges"]) == edges
    assert tuple(map(exact_record, model["capacity"])) == (Q(1),) * 27
    assert model["clock"] == "tau=e*t; e=1023/1024"
    assert model["form_row"] == "x'=-D^-1 L x + gamma D^-1 S(theta)"
    assert model["phase_row"] == "theta'=gamma D^-1 L x"
    assert model["gamma"] == "1/(1023*pi)"
    assert exact_record(model["storage_scale"]) == 1
    return inputs


@pytest.fixture(scope="module")
def reconstructed(retained, admitted):
    outcome = retained[1]
    if outcome["report"] is None:
        assert outcome["error"] is not None
        return None
    envelope = outcome["report"]
    assert envelope["schema"] == "tnfr.sine-class-comparison-readout.v1"
    raw = envelope["report"]
    report = _Record(raw)
    sources = tuple(
        tuple(tuple(I(*pair) for pair in cover) for cover in admitted[key])
        for key in ("initial_form_bounds", "initial_phase_bounds")
    )
    assert report.initial_form_bounds == sources[0]
    assert report.initial_phase_bounds == sources[1]
    for key in admitted.keys() - {"initial_form_bounds", "initial_phase_bounds"}:
        admit = _integer if key in ("order", "max_steps") else exact_record
        assert admit(getattr(report, key)) == admitted[key]
    assert tuple(map(_integer, report.source_order)) == (0, 1)
    assert tuple(map(exact_record, report.contrast_coefficients)) == (1, -1)
    assert report.method == "sequential_two_source_full54_readout_reconstruction_v1"
    assert report.clock == "tau=e*t; e=1023/1024"
    assert len(raw["readout_reports"]) == len(raw["reconstructed_observations"]) == 2
    rebuilt = []
    attempted = completed = completed_sources = 0
    failure = None
    reasons = []
    for index, child in enumerate(raw["readout_reports"]):
        remaining = admitted["max_steps"] - attempted
        if failure is not None or remaining == 0:
            assert child is None and raw["reconstructed_observations"][index] is None
            if failure is None:
                failure = index
                reasons.append(
                    f"source_{index}: total_step_budget_exhausted_before_source"
                )
            rebuilt.append(None)
            continue
        assert child is not None
        inputs = (
            sources[0][index],
            sources[1][index],
            *(
                admitted[key]
                for key in (
                    "first_probe_amplitude",
                    "second_probe_amplitude",
                    "delay",
                    "total_duration",
                    "time_step",
                    "order",
                )
            ),
            min(4096, remaining),
        )
        evidence = _reconstruct_class_readout(_Record(child), inputs)
        saved = _Record(raw["reconstructed_observations"][index])
        assert saved.complete is evidence.complete
        for key in (
            "planned_step_count",
            "attempted_step_count",
            "completed_step_count",
        ):
            assert _integer(getattr(saved, key)) == getattr(evidence, key)
        for key in (
            "endpoint_bounds",
            "suffix_increment_bounds",
            "mixed_bounds",
            "raw_mixed_bounds",
        ):
            assert getattr(saved, key) == getattr(evidence, key)
        rebuilt.append(evidence)
        attempted += evidence.attempted_step_count
        completed += evidence.completed_step_count
        if evidence.complete:
            completed_sources += 1
        else:
            failure = index
            reasons.append(f"source_{index}: incomplete_four_history_readout")
    assert _integer(report.planned_step_count) == 1536
    assert _integer(report.attempted_step_count) == attempted <= 1536
    assert _integer(report.completed_step_count) == completed
    assert _integer(report.completed_source_count) == completed_sources
    if report.failed_source_index is not None:
        _integer(report.failed_source_index)
    assert report.failed_source_index == failure
    assert tuple(map(_integer, report.unattempted_source_indices)) == tuple(
        i for i, child in enumerate(raw["readout_reports"]) if child is None
    )
    assert report.unavailable_reasons == tuple(reasons)
    assert report.status == ("admitted" if completed_sources == 2 else "unavailable")
    primary = raw_mixed = None
    if completed_sources == 2:
        bands = tuple(item.mixed_bounds for item in rebuilt)
        assert report.class_mixed_readout_bounds == bands
        primary = bands[0] - bands[1]
        raw_mixed = rebuilt[0].raw_mixed_bounds - rebuilt[1].raw_mixed_bounds
    else:
        assert report.class_mixed_readout_bounds is None
    assert report.contrast_readout_bounds == primary
    assert report.raw_endpoint_contrast_bounds == raw_mixed
    return primary, raw_mixed, tuple(rebuilt)


def test_every_child_step_and_comparison_reconstructs(retained, reconstructed):
    if reconstructed is None:
        assert retained[1]["error"] is not None
        return
    primary, raw, children = reconstructed
    if primary is not None:
        assert len(children) == 2 and all(child.complete for child in children)
        assert primary == children[0].mixed_bounds - children[1].mixed_bounds
        assert raw == children[0].raw_mixed_bounds - children[1].raw_mixed_bounds


def _assessment(reference, policy):
    eps, g, T, delta, allowance = (
        exact_record(policy[key])
        for key in (
            "endpoint_radius",
            "gamma_upper",
            "total_duration",
            "readout_error_bound",
            "width_allowance",
        )
    )
    assert (eps, g, T, delta, allowance) == (
        Q(1, 10**32),
        Q(1, 3000),
        Q(2),
        Q(1, 10**30),
        Q(1, 10**30),
    )
    rp, ap = (
        policy["reference_prediction_open_bounds"],
        policy["prediction_open_bounds"],
    )
    assert rp == (Q(-19254, 10**33), Q(-19237, 10**33))
    assert ap == (Q(-19334, 10**33), Q(-19157, 10**33))
    source = 8 * eps / (1 - 2 * g * T)
    result = dict(
        reference_bounds=None,
        actual_bounds=None,
        reference_width=None,
        actual_width=None,
        source_error_upper_bound=source,
        decision=None,
        width_within_allowance=None,
        reference_theorem_open_interval_overlap=None,
        reference_theorem_open_interval_contains_band=None,
        theorem_open_interval_overlap=None,
        theorem_open_interval_contains_actual_band=None,
        status="unavailable",
    )
    if reference is None:
        return result
    lo, hi = reference.lo - source, reference.hi + source
    orientation = 1 if lo > 0 else -1 if hi < 0 else 0
    lower = lo if orientation > 0 else -hi if orientation < 0 else None
    true_sign = lower is not None and lower > 0
    sign_margin = None if lower is None else lower - 8 * delta
    null_margin = None if lower is None else lower - 16 * delta
    recorded_sign = sign_margin is not None and sign_margin > 0
    null_excluded = null_margin is not None and null_margin > 0
    cancellation = 8 * delta - max(abs(lo), abs(hi))
    decision = dict(
        true_bounds=(lo, hi),
        recorded_bounds=(lo - 8 * delta, hi + 8 * delta),
        orientation=orientation,
        oriented_lower=lower,
        recorded_sign_margin=sign_margin,
        null_separation_margin=null_margin,
        noise_ceiling=lower / 8 if true_sign else None,
        true_sign=true_sign,
        recorded_sign=recorded_sign,
        null_excluded=null_excluded,
        cancellation_margin=cancellation,
        scalar_cancellation=cancellation >= 0,
        status=(
            "zero_contrast_record_sets_disjoint"
            if null_excluded
            else (
                "recorded_sign_certified"
                if recorded_sign
                else "true_sign_certified" if true_sign else "bounds_only"
            )
        ),
    )
    reference_overlap = reference.hi > rp[0] and reference.lo < rp[1]
    actual_overlap = hi > ap[0] and lo < ap[1]
    within = hi - lo <= allowance
    status = (
        "consistency_conflict"
        if not reference_overlap or not actual_overlap
        else (
            "resolution_not_certified"
            if not within
            else (
                "discrimination_not_certified"
                if not null_excluded
                else "discrimination_certified"
            )
        )
    )
    result.update(
        reference_bounds=(reference.lo, reference.hi),
        actual_bounds=(lo, hi),
        reference_width=reference.hi - reference.lo,
        actual_width=hi - lo,
        decision=decision,
        width_within_allowance=within,
        reference_theorem_open_interval_overlap=reference_overlap,
        reference_theorem_open_interval_contains_band=rp[0] < reference.lo
        and reference.hi < rp[1],
        theorem_open_interval_overlap=actual_overlap,
        theorem_open_interval_contains_actual_band=ap[0] < lo and hi < ap[1],
        status=status,
    )
    return result


def test_actual_source_transfer_and_decisions_use_unintersected_bounds(
    retained, reconstructed
):
    records, outcome, _, _ = retained
    for index, key in enumerate(("assessment", "raw_endpoint_assessment")):
        if outcome[key] is None:
            assert outcome["error"] is not None
            continue
        reference = None if reconstructed is None else reconstructed[index]
        expected = _assessment(
            reference, records[".protocol.json"]["observation_policy"]
        )
        actual = decode_exact_tree(outcome[key])
        assert actual == expected
        for field in (
            "width_within_allowance",
            "reference_theorem_open_interval_overlap",
            "reference_theorem_open_interval_contains_band",
            "theorem_open_interval_overlap",
            "theorem_open_interval_contains_actual_band",
        ):
            assert actual[field] is expected[field]
        if actual["decision"] is not None:
            assert type(actual["decision"]["orientation"]) is int
            for field in (
                "true_sign",
                "recorded_sign",
                "null_excluded",
                "scalar_cancellation",
            ):
                assert actual["decision"][field] is expected["decision"][field]


@pytest.mark.parametrize(
    "key", ("increment", "series", "order", "picard_interior_margin")
)
def test_selected_saved_step_corruption_is_rejected(retained, key):
    envelope = retained[1]["report"]
    if envelope is None:
        assert retained[1]["error"] is not None
        return
    original = next(
        (
            step
            for child in envelope["report"]["readout_reports"]
            if child is not None
            for segment in child["segments"]
            for step in segment["steps"]
        ),
        None,
    )
    if original is None:
        assert envelope["report"]["completed_step_count"] == 0
        return
    changed = dict(original)
    if key in ("order", "picard_interior_margin"):
        changed[key] = True
    else:
        values = list(original[key])
        bad = {"lo": Q(100), "hi": Q(100)}
        values[0] = (bad, *values[0][1:]) if key == "series" else bad
        changed[key] = values
    with pytest.raises((AssertionError, ValueError)):
        old_audit._audit_step(
            changed,
            tuple(map(_interval, original["initial_box"])),
            exact_record(original["time"]),
            exact_record(original["duration"]),
            _integer(original["order"]),
        )


def test_retained_first_outcome_has_the_reconstructed_negative_bands(
    retained, reconstructed
):
    """Pin the observed result only after both source trees are rebuilt."""
    outcome = retained[1]
    primary, raw, children = reconstructed
    assert (primary.lo, primary.hi) == (Q(-6549026775, 2**128), Q(-204655195, 2**123))
    assert (raw.lo, raw.hi) == (Q(-1637257735, 2**126), Q(-6548962075, 2**128))
    assert primary.hi - primary.lo == Q(60535, 2**128)
    assert raw.hi - raw.lo == Q(68865, 2**128)
    assert outcome["error"] is None
    report = _Record(outcome["report"]["report"])
    assert report.attempted_step_count == report.completed_step_count == 1536
    assert report.completed_source_count == 2
    assert (
        report.failed_source_index is None and report.unattempted_source_indices == ()
    )
    for child, evidence in zip(report.readout_reports, children):
        assert evidence.complete
        assert evidence.attempted_step_count == evidence.completed_step_count == 768
        assert tuple(len(segment.steps) for segment in child.segments) == (128,) * 6
        assert (
            child.failed_segment_index is None
            and child.unattempted_segment_indices == ()
        )
    for key in ("assessment", "raw_endpoint_assessment"):
        saved = decode_exact_tree(outcome[key])
        assert saved["status"] == "discrimination_certified"
        assert saved["width_within_allowance"] is True
        assert saved["reference_theorem_open_interval_contains_band"] is True
        assert saved["theorem_open_interval_contains_actual_band"] is True
        assert saved["decision"]["true_sign"] is True
        assert saved["decision"]["recorded_sign"] is True
        assert saved["decision"]["null_excluded"] is True
        assert saved["decision"]["scalar_cancellation"] is False
