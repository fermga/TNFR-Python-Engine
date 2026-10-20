"""Fixed-cell re-admission controls without selecting or producing a response."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.research import maintained_child_joint_source as owner
from tnfr.research import maintained_child_joint_window, maintained_sha_motion
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
)
from tnfr.research.maintained_parent_thol import _signature
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_joint_window"


def _read(name):
    return decode_exact_tree(
        json_loads(
            read_bytes_bounded(EVIDENCE / f"{name}.json", max_bytes=32 * 1024**2)
        )
    )


@pytest.fixture(scope="module", autouse=True)
def no_selection_trajectory_or_event():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "fixed-cell admission cannot execute a selection, trajectory or event"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_child_joint_window, "inspect_child_joint_window"),
            (maintained_sha_motion, "certify_sha_retention"),
            (maintained_sha_motion, "validated_box_taylor_step"),
            (_validated_taylor, "validated_box_taylor_step"),
            (network_stage, "execute_dissonance_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def source():
    original_load, original_bound = (
        owner.load_sha_retention,
        owner.bound_node_history_box,
    )
    loads, boxes = [], []

    def load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        loads.append(value)
        return value

    def bound(**kwargs):
        boxes.append(kwargs)
        return original_bound(**kwargs)

    before = {
        path.name: file_receipt(path) for path in EVIDENCE.iterdir() if path.is_file()
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "load_sha_retention", load)
        patch.setattr(owner, "bound_node_history_box", bound)
        admitted = owner.load_child_joint_window(EVIDENCE, repository_root=ROOT)
    signature = _signature(admitted.retention.admitted.graph)
    yield admitted, _read("protocol"), _read("result")["report"], loads, boxes
    assert _signature(admitted.retention.admitted.graph) == signature
    assert before == {
        path.name: file_receipt(path) for path in EVIDENCE.iterdir() if path.is_file()
    }


def test_loader_reads_retention_once_and_rebuilds_only_the_fixed_whole_cell(source):
    admitted, _, _, loads, boxes = source
    assert loads == [admitted.retention]
    assert len(boxes) == 1
    assert admitted.step is admitted.retention.steps[12]
    assert boxes[0]["epi"] == admitted.step.tube[:21]
    assert boxes[0]["phases"] == admitted.step.tube[21:]
    assert boxes[0]["target_index"] == 20
    for name in ("capacities", "edges", "conductances", "shifts"):
        assert boxes[0][name] == getattr(admitted.retention.prediction, name)
    assert admitted.cell.index == 12
    assert admitted.cell.elapsed_start == Q(3, 128)
    assert admitted.cell.elapsed_end == Q(13, 512)
    assert admitted.cell.conditions.joint
    assert admitted.cell.form == admitted.bounds.form == admitted.step.tube[20]
    assert admitted.minimum_form == Q.from_float(0.2)
    assert admitted.threshold == Q.from_float(0.1)
    assert {row["path"] for row in admitted.receipts} == {
        "protocol.json",
        "freeze.json",
        "attempt.json",
        "result.json",
        "source-manifest.json",
        "source.zip",
    }
    for row in admitted.receipts:
        assert file_receipt(EVIDENCE / row["path"], label=row["path"]) == row
    assert not hasattr(admitted, "acquired_history")
    with pytest.raises(FrozenInstanceError):
        admitted.threshold = Q(0)


def test_rebuilt_child_rows_match_independent_degree_one_equations(source):
    admitted = source[0]
    prediction, tube = admitted.retention.prediction, admitted.step.tube
    x, phase, pi = tube[:21], tube[21:], pi_interval()

    def omega(node):
        neighbors = tuple(
            (right if left == node else left, weight)
            for (left, right), weight in zip(
                prediction.edges, prediction.conductances, strict=True
            )
            if node in (left, right)
        )
        return (
            prediction.capacities[node]
            * sum(
                (weight * (x[node] - x[neighbor]) for neighbor, weight in neighbors),
                I(0),
            )
            / (pi * len(neighbors))
        )

    gap = phase[20] - phase[2]
    pressure = -sin(gap) / pi
    acceleration = -prediction.capacities[20] * cos(gap) * (omega(20) - omega(2)) / pi
    for observed, independent in (
        (admitted.cell.pressure, pressure),
        (admitted.cell.acceleration, acceleration),
    ):
        assert max(observed.lo, independent.lo) <= min(observed.hi, independent.hi)
        assert max(
            abs(observed.lo - independent.lo), abs(observed.hi - independent.hi)
        ) < Q(1, 2**85)


@pytest.mark.parametrize(
    "field",
    (
        "form",
        "pressure",
        "acceleration",
        "minimum_principal_margin",
        "elapsed_start",
        "elapsed_end",
        "condition",
        "index",
    ),
)
def test_changed_selected_record_cannot_supply_a_scientific_premise(source, field):
    admitted, protocol, original = source[:3]
    report = deepcopy(original)
    selected = report["selected_cell"]
    if field in ("form", "pressure", "acceleration"):
        selected[field]["lo"] += Q(1, 1024)
    elif field == "condition":
        selected["conditions"]["joint"] = 1
    elif field == "index":
        selected[field] = 12.0
    else:
        selected[field] += Q(1, 1024)
    report["inspected_cells"] = report["inspected_cells"][:-1] + (deepcopy(selected),)
    with pytest.raises(ValueError):
        owner._admit_fixed_cell(admitted.retention, protocol, report)


@pytest.mark.parametrize(
    "field,value",
    (
        ("minimum_form", Q(0)),
        ("threshold", False),
        ("source_time", Q(0)),
        ("parent", True),
        ("retained_cells", 256.0),
        ("executed_event", True),
        ("acquired_history", 0),
    ),
)
def test_changed_report_policy_time_or_scope_rejects_before_box_rebuild(
    source, monkeypatch, field, value
):
    admitted, protocol, original = source[:3]
    report = deepcopy(original)
    report[field] = value

    def forbidden(**kwargs):
        pytest.fail("invalid report must reject before scientific box evaluation")

    monkeypatch.setattr(owner, "bound_node_history_box", forbidden)
    with pytest.raises(ValueError, match="joint report"):
        owner._admit_fixed_cell(admitted.retention, protocol, report)


@pytest.mark.parametrize(
    "field,value",
    (
        ("minimum_form", Q(0)),
        ("threshold", False),
        ("cell_width", Q(1, 1024)),
        ("source_time", Q(0)),
        ("state_dimension", 42.0),
        ("forcing", False),
        ("law", "same equilibrium"),
    ),
)
def test_changed_protocol_cannot_borrow_the_fixed_cell(source, field, value):
    admitted, original, report = source[:3]
    protocol = deepcopy(original)
    protocol[field] = value
    with pytest.raises(ValueError, match="joint protocol"):
        owner._admit_fixed_cell(admitted.retention, protocol, report)


@pytest.mark.parametrize("association", ("protocol", "report"))
def test_changed_retention_byte_association_is_rejected(source, association):
    admitted, original_protocol, original_report = source[:3]
    protocol, report = deepcopy(original_protocol), deepcopy(original_report)
    if association == "protocol":
        protocol["source_result"]["sha256"] = "0" * 64
    else:
        report["source_receipts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="association|source"):
        owner._admit_fixed_cell(admitted.retention, protocol, report)


def test_favorable_cached_flag_cannot_override_changed_whole_cell(source):
    admitted, protocol, report = source[:3]
    changed_tube = list(admitted.step.tube)
    changed_tube[20] = I(0)
    step = replace(admitted.step, tube=tuple(changed_tube))
    steps = list(admitted.retention.steps)
    steps[12] = step
    changed = replace(admitted.retention, steps=tuple(steps))
    with pytest.raises(ValueError, match="does not rebuild joint admission"):
        owner._admit_fixed_cell(changed, protocol, report)


def test_fixed_cell_reader_does_not_reclassify_the_earlier_recorded_prefix(source):
    admitted, protocol, original = source[:3]
    report = deepcopy(original)
    # This private arithmetic check consumes only the selected cell. The public
    # loader separately binds the original bytes; first-selection history is
    # inherited and remains the earlier observer/audit's responsibility.
    report["inspected_cells"][0]["form"] = {"not_consumed": True}
    _, _, cell, minimum, threshold = owner._admit_fixed_cell(
        admitted.retention, protocol, report
    )
    assert cell == admitted.cell
    assert minimum == admitted.minimum_form and threshold == admitted.threshold


def test_duplicate_json_fields_reject_before_any_retention_load(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(
        '{"schema": 1, "schema": 2}', encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("malformed observation record must reject before retention load")

    monkeypatch.setattr(owner, "load_sha_retention", forbidden)
    with pytest.raises(ValueError):
        owner.load_child_joint_window(tmp_path, repository_root=ROOT)
