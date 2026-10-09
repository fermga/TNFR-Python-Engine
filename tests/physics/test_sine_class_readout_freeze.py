"""Freeze associations and synthetic metadata checks without reserved evolution.

Only selected pure definitions are compiled from the source archive. Synthetic
constant-field Taylor metadata test wiring, not the complete sine derivative
premise. No evaluator, source acquisition, flow, or archived main is run.
"""

import ast
import copy
import hashlib
import os
import subprocess
import zipfile
from fractions import Fraction as Q
from pathlib import Path, PurePosixPath

import pytest

from tnfr._exact_time import exact_or_represented_real
from tnfr.mathematics._comparison_flow import _exact
from tnfr.mathematics._phase_midpoint import _pi_bounds
from tnfr.mathematics._rational_interval import I
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "class-nonlinear-readout-v1"
HELPERS = "build/class-nonlinear-readout-freeze"
BASE = "fa8e98a9b1bdd755709da481de7fc092b57bfe65"


def _source(name):
    path = f"{HELPERS}/{name}.py"
    archive = DIRECTORY / (STEM + ".source.zip")
    with zipfile.ZipFile(archive) as source:
        return source.read(path)


def _definitions(name, selected, namespace, assignments=()):
    parsed = ast.parse(_source(name))
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in selected
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets)
    ]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == set(
        selected
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_class_nonlinear_protocol,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("freeze checks must not run a response, acquisition or worker")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
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
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def pure():
    namespace = {
        "Q": Q,
        "I": I,
        "_exact": _exact,
        "exact_or_represented_real": exact_or_represented_real,
        "_pi_bounds": _pi_bounds,
        "BASE": BASE,
        "Path": Path,
        "os": os,
        "json_dumps": json_dumps,
    }
    _definitions(
        "experiment_support",
        {
            "encode",
            "decode",
            "canonical_source",
            "producer_inputs",
            "interval_criteria",
            "_interval",
            "_box",
            "_pair",
            "reconstruct_observations",
            "normalized_text",
            "write_once",
        },
        namespace,
        ("INPUT_KEYS", "PLAN"),
    )
    _definitions("prepare_protocol", {"make_protocol"}, namespace)
    _definitions("evaluate_readout", {"outcome_status"}, namespace)
    return namespace


def _record(value):
    return {"lo": value.lo, "hi": value.hi}


def _records(values):
    return tuple(map(_record, values))


@pytest.fixture(scope="module")
def synthetic(pure):
    """A constant zero field metadata fixture, not a reserved sine response."""
    protocol = pure["make_protocol"]([], {})
    inputs = protocol["producer_inputs"]
    unrelated = ((Q(-1, 8), Q(1, 8)),) * 27
    inputs["initial_form_bounds"] = unrelated
    inputs["initial_phase_bounds"] = unrelated
    source = (I(*unrelated[0]),) * 54
    zero = I(0)
    zero_box = _records((zero,) * 54)
    tube = _records((I(-1, 1),) * 54)
    segments, finals = [], {}
    for index, (label, parent, start, end, jump) in enumerate(pure["PLAN"]):
        before = source if parent is None else finals[parent]
        state = tuple(
            value + jump if i == 4 and jump else value for i, value in enumerate(before)
        )
        box = _records(state)
        series = tuple((_record(value),) + (_record(zero),) * 16 for value in state)
        steps = tuple(
            dict(
                time=start + Q(i, 64),
                duration=Q(1, 64),
                order=16,
                initial_box=box,
                tube=tube,
                series=series,
                local_remainder_bounds=zero_box,
                increment=zero_box,
                endpoint=box,
                picard_interior_margin=Q(1, 10),
                domain_lower_bounds=(Q(1),),
            )
            for i in range(64)
        )
        segments.append(
            dict(
                label=label,
                parent_segment_index=parent,
                start_time=start,
                end_time=end,
                form_jump=jump,
                pre_event_box=_records(before),
                initial_box=box,
                steps=steps,
                completed_time=end,
                completed_endpoint_box=box,
                completed_receiver_increment_bounds=_record(zero),
                final_state_bounds=box,
                failed_initial_box=None,
                failed_time=None,
                failed_tube=None,
                status="admitted",
                reason=None,
            )
        )
        finals[index] = state
    endpoints = tuple(finals[i][22] for i in range(2, 6))
    report = dict(inputs)
    report.update(
        initial_form_bounds=_records(source[:27]),
        initial_phase_bounds=_records(source[27:]),
        source_box=_records(source),
        geometry=protocol["support"],
        degrees=protocol["support"]["degrees"],
        reference_model=dict(
            storage_scale=1.0,
            epi_weight=1023 / 1024,
            phase_weight=1 / 1024,
            phase_domain="regular",
        ),
        capacity=(Q(1),) * 27,
        clock="tau=e*t; e=1023/1024",
        state_order=tuple(
            f"{channel}_{i}" for channel in ("x", "theta") for i in range(27)
        ),
        donor_node=4,
        receiver_node=22,
        segments=tuple(segments),
        completed_step_count=384,
        attempted_step_count=384,
        completed_segment_count=6,
        planned_unique_step_count=384,
        failed_segment_index=None,
        unattempted_segment_indices=(),
        endpoint_readout_bounds=_records(endpoints),
        suffix_receiver_increment_bounds=_records((zero,) * 4),
        raw_endpoint_mixed_bounds=_record(
            endpoints[0] - endpoints[1] - endpoints[2] + endpoints[3]
        ),
        mixed_readout_bounds=_record(zero),
    )
    return report, protocol


def test_complete_carried_metadata_reconstructs_increment_without_claiming_sine(
    pure, synthetic
):
    report, protocol = synthetic
    result = pure["reconstruct_observations"](report, protocol)
    assert result["complete"] and result["evidence_integrity"]
    assert result["mixed_bounds"] == (0, 0)
    assert result["raw_endpoint_mixed_bounds"] == (-Q(1, 2), Q(1, 2))
    assert (
        pure["decode"](json_loads(__import__("json").dumps(pure["encode"](result))))
        == result
    )


@pytest.mark.parametrize(
    "section,key,value",
    (
        ("report", "order", True),
        ("report", "completed_step_count", True),
        ("model", "storage_scale", True),
        ("report", "delay", True),
        ("segment", "parent_segment_index", False),
        ("segment", "start_time", False),
        ("segment", "form_jump", False),
        ("segment", "completed_time", True),
        ("step", "time", False),
        ("step", "duration", float("inf")),
        ("step", "order", True),
        ("step", "picard_interior_margin", True),
        ("step", "domain_lower_bounds", (True,)),
    ),
)
def test_evidence_admits_primitives_before_equality(
    pure, synthetic, section, key, value
):
    report = copy.deepcopy(synthetic[0])
    target = {
        "report": report,
        "model": report["reference_model"],
        "segment": report["segments"][0],
        "step": report["segments"][0]["steps"][0],
    }[section]
    target[key] = value
    with pytest.raises((TypeError, ValueError)):
        pure["reconstruct_observations"](report, synthetic[1])


@pytest.mark.parametrize("key", ("increment", "endpoint", "series"))
def test_rebuild_rejects_cached_step_disagreement(pure, synthetic, key):
    report = copy.deepcopy(synthetic[0])
    step = report["segments"][0]["steps"][0]
    values = list(step[key])
    values[0] = (_record(I(1)),) * 17 if key == "series" else _record(I(1))
    step[key] = tuple(values)
    with pytest.raises(ValueError):
        pure["reconstruct_observations"](report, synthetic[1])


def test_cached_mixed_cannot_replace_primitive_increment_evidence(pure, synthetic):
    report = copy.deepcopy(synthetic[0])
    report["mixed_readout_bounds"] = _record(I(-1))
    with pytest.raises(ValueError, match="cached mixed"):
        pure["reconstruct_observations"](report, synthetic[1])


@pytest.fixture(scope="module")
def partial(synthetic):
    report = copy.deepcopy(synthetic[0])
    first = report["segments"][0]
    first.update(
        steps=(),
        completed_time=Q(0),
        completed_endpoint_box=first["initial_box"],
        final_state_bounds=None,
        failed_initial_box=first["initial_box"],
        failed_time=Q(0),
        failed_tube=first["initial_box"],
        status="unavailable",
        reason="synthetic_failed_Picard_inclusion",
    )
    for segment in report["segments"][1:]:
        segment.update(steps=(), status="not_attempted")
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
        ):
            segment[key] = None
    report.update(
        completed_step_count=0,
        completed_segment_count=0,
        attempted_step_count=1,
        failed_segment_index=0,
        unattempted_segment_indices=(1, 2, 3, 4, 5),
        endpoint_readout_bounds=None,
        suffix_receiver_increment_bounds=None,
        raw_endpoint_mixed_bounds=None,
        mixed_readout_bounds=None,
    )
    return report


def test_partial_first_failure_retains_no_fabricated_observations(
    pure, partial, synthetic
):
    result = pure["reconstruct_observations"](partial, synthetic[1])
    assert result["complete"] is False and result["evidence_integrity"] is True
    assert result["mixed_bounds"] is result["endpoint_bounds"] is None


@pytest.mark.parametrize(
    "key,value",
    (
        ("attempted_step_count", 0),
        ("attempted_step_count", 2),
        ("failed_segment_index", False),
        ("failed_segment_index", 1),
        ("unattempted_segment_indices", (2, 3, 4, 5)),
        ("raw_endpoint_mixed_bounds", {"lo": Q(0), "hi": Q(0)}),
    ),
)
def test_partial_count_ancestry_and_null_evidence_cannot_be_forged(
    pure, partial, synthetic, key, value
):
    report = copy.deepcopy(partial)
    report[key] = value
    with pytest.raises(ValueError):
        pure["reconstruct_observations"](report, synthetic[1])


def test_no_future_event_or_box_after_failed_prefix(pure, partial, synthetic):
    report = copy.deepcopy(partial)
    report["segments"][1]["initial_box"] = report["source_box"]
    with pytest.raises(ValueError, match="invented event"):
        pure["reconstruct_observations"](report, synthetic[1])


def test_missing_and_conflicting_outcomes_stay_distinct(pure):
    classify = pure["interval_criteria"]
    status = pure["outcome_status"]
    prediction, delta = (Q(-10), Q(-9)), Q(1)
    assert status(None, classify(None, delta, prediction), None) == "unavailable"
    assert (
        status({"complete": True}, classify((Q(-20), Q(-10)), delta, prediction), None)
        == "consistency_conflict"
    )
    assert (
        status({"complete": True}, classify((Q(-10), Q(1)), delta, prediction), None)
        == "inconclusive"
    )
    assert (
        status({"complete": True}, classify((Q(-10), Q(-9)), delta, prediction), None)
        == "certified_reserved_readout"
    )
    assert (
        status(None, classify(None, delta, prediction), {"type": "Synthetic"})
        == "evaluation_error"
    )


def test_exclusive_write_preserves_first_json_bytes(pure, tmp_path):
    target = tmp_path / "ordinary-control.json"
    pure["write_once"](target, {"value": Q(2, 7), "scope": "synthetic serialization"})
    original = target.read_bytes()
    assert pure["decode"](json_loads(original))["value"] == Q(2, 7)
    with pytest.raises(FileExistsError):
        pure["write_once"](target, {"value": Q(3, 7)})
    assert target.read_bytes() == original


@pytest.fixture(scope="module")
def frozen():
    paths = tuple(
        DIRECTORY / (STEM + suffix)
        for suffix in (".protocol.json", ".source.zip", ".freeze.json")
    )
    assert all(path.exists() for path in paths), "missing immutable freeze artifact"
    original = {path: path.read_bytes() for path in paths}
    yield paths, original
    assert all(path.read_bytes() == data for path, data in original.items())


def test_freeze_receipt_exact_inventory_and_historical_not_evaluated(frozen, pure):
    paths, original = frozen
    protocol = pure["decode"](json_loads(original[paths[0]]))
    receipt = json_loads(original[paths[2]])
    assert protocol["schema"] == "tnfr.sine-class-nonlinear-readout-protocol.v1"
    assert receipt["schema"] == "tnfr.sine-class-nonlinear-readout-freeze.v1"
    assert (
        receipt["evaluation_status_at_freeze"]
        == protocol["evaluation_status_at_freeze"]
        == "not_evaluated"
    )
    assert receipt["immutable_after_evaluation"] is True
    assert receipt["source_base_commit"] == protocol["source_base_commit"] == BASE
    assert (
        tuple(receipt["runtime_overlays"]) == tuple(protocol["runtime_overlays"]) == ()
    )
    assert not {"passed", "status", "response", "criteria"} & receipt.keys()
    entries = receipt["artifacts"]
    assert {item["path"] for item in entries} == {
        path.relative_to(ROOT).as_posix() for path in paths[:2]
    }
    assert len(entries) == 2
    for item in entries:
        data = original[ROOT / item["path"]]
        assert item["bytes"] == len(data)
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
    assert protocol["producer_inputs"] == pure["producer_inputs"]()
    assert protocol["numerical_policy"]["numerical_width_cutoff"] is None
    for item in protocol["source_specification"]["prior_artifact_receipts"]:
        data = (ROOT / item["path"]).read_bytes()
        assert (
            item["bytes"] == len(data)
            and item["sha256"] == hashlib.sha256(data).hexdigest()
        )


def test_archive_inventory_prospective_prefix_and_dormant_single_attempt(frozen, pure):
    paths, original = frozen
    with zipfile.ZipFile(paths[1]) as archive:
        manifest = json_loads(archive.read("source-manifest.json"))
        entries = manifest["files"]
        names = [item["path"] for item in entries]
        assert len(names) == len(set(names))
        assert all(
            not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            for name in names
        )
        hashes = {item["path"]: item["sha256"] for item in entries}
        hashes["source-manifest.json"] = hashlib.sha256(
            archive.read("source-manifest.json")
        ).hexdigest()
        _verify_archive(paths[1], hashes)
        assert set(archive.namelist()) == set(hashes)
        assert archive.read(paths[0].relative_to(ROOT).as_posix()) == original[paths[0]]
        for item in entries:
            data = archive.read(item["path"])
            assert len(data) == item["bytes"]
            if item["path"].startswith("src/"):
                assert (
                    hashlib.sha256(pure["normalized_text"](data)).hexdigest()
                    == item["normalized_lf_sha256"]
                )
        proof = "theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md"
        assert pure["normalized_text"]((ROOT / proof).read_bytes()).startswith(
            pure["normalized_text"](archive.read(proof))
        )
        assert b"sine-nonlinear-protocol-frozen-evaluation" in archive.read(proof)
        evaluator = ast.parse(archive.read(f"{HELPERS}/evaluate_readout.py"))
        calls = [node for node in ast.walk(evaluator) if isinstance(node, ast.Call)]
        producer_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "bound_sine_class_four_history_readout"
        ]
        assert len(producer_calls) == 1
        assert not any(
            isinstance(node, (ast.For, ast.While))
            and producer_calls[0] in list(ast.walk(node))
            for node in ast.walk(evaluator)
        )
        evaluate = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
        )
        ledger_call = next(
            node
            for node in ast.walk(evaluate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_once"
        )
        assert ledger_call.lineno < producer_calls[0].lineno


@pytest.mark.parametrize("newline", (b"\n", b"\r\n"))
def test_live_prefix_normalizes_checkout_newlines_only(pure, newline):
    prefix = b"proof\nexact line\n"
    live = prefix.replace(b"\n", newline) + b"later result\n"
    assert pure["normalized_text"](live).startswith(prefix)
    assert (
        hashlib.sha256(prefix).digest()
        != hashlib.sha256(prefix.replace(b"\n", b"\r\n")).digest()
    )
