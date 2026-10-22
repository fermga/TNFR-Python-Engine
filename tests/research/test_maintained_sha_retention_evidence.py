"""Read-only arithmetic audit of the actual post-SHA retention attempt.

Derivative jets and remainders retain their frozen producer provenance; this
audit does not regenerate them or evaluate another trajectory. It re-admits
the actual endpoint, checks Picard inclusion under its changed-capacity law,
reconstructs Taylor arithmetic and derives the retained prefix from the rows.
Byte associations do not authenticate acquisition or chronology.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I
from tnfr.operators import al_sha_stage_proposals, network_stage, word_execution
from tnfr.research import (
    local_capacity_history,
    maintained_parent_sha_execution,
    maintained_parent_thol_execution,
    maintained_parent_um_execution,
    maintained_principal_retention,
    maintained_sha_motion,
)
from tnfr.research import maintained_sha_retention_evidence as owner
from tnfr.research import maintained_successor_motion_certificate
from tnfr.research.artifact_io import file_receipt, verify_archive_members
from tnfr.research.local_capacity_history import TIMES
from tnfr.research.maintained_parent_sha_source import _same
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.retained_graph_state import restore_recorded_graph_state

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_sha_retention"
STEP, COUNT, ORDER = Q(1, 512), 256, 12


_audit_step = owner._audit_step
_check_summary = owner._check_summary
_interval = owner._interval
_receipt = owner._receipt


@pytest.fixture(scope="module", autouse=True)
def no_reserved_response_events_or_refresh():
    def forbidden(*args, **kwargs):
        pytest.fail("SHA retention audit cannot execute another response or event")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_sha_motion, "certify_sha_retention"),
            (maintained_sha_motion, "certify_retention"),
            (maintained_sha_motion, "validated_box_taylor_step"),
            (maintained_principal_retention, "certify_retention"),
            (maintained_successor_motion_certificate, "certify_motion"),
            (maintained_parent_sha_execution, "evaluate_sha_event"),
            (maintained_parent_sha_execution, "refresh_pressure"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "refresh_pressure"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "refresh_pressure"),
            (local_capacity_history, "rk4_coupled_step"),
            (al_sha_stage_proposals, "propose_silence_stage"),
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
    # Missing evidence fails explicitly; tests never launch its producer.
    before = {
        path.relative_to(EVIDENCE).as_posix(): file_receipt(path)
        for path in EVIDENCE.rglob("*")
        if path.is_file()
    }
    evidence = owner.load_sha_retention(EVIDENCE, repository_root=ROOT)
    signature = _signature(evidence.admitted.graph)
    data = dict(evidence.documents)
    data.update(
        evidence=evidence,
        admitted=evidence.admitted,
        records=evidence.records,
        prediction=evidence.prediction,
        initial=evidence.initial,
        flow=evidence.flow,
        audited_summary=evidence.summary,
    )
    yield data
    assert _signature(evidence.admitted.graph) == signature
    assert before == {
        path.relative_to(EVIDENCE).as_posix(): file_receipt(path)
        for path in EVIDENCE.rglob("*")
        if path.is_file()
    }


def test_frozen_archive_prior_evidence_and_single_attempt_bindings(retained):
    frozen, manifest = retained["freeze"], retained["source-manifest"]
    assert frozen["schema"] == "tnfr.maintained-sha-retention-freeze.v1"
    assert frozen["evaluation_status_at_freeze"] == "not_evaluated"
    assert len(frozen["artifacts"]) == 4
    assert {row["path"] for row in frozen["artifacts"]} == {
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "source.zip",
    }
    for row in frozen["artifacts"]:
        _receipt(EVIDENCE, row)
    for row in frozen["prior_artifacts"]:
        _receipt(ROOT, row)
    assert manifest["schema"] == "tnfr.source-manifest.v1"
    members = manifest["files"]
    assert len(members) == len({row["path"] for row in members})
    expected = {row["path"]: row["sha256"] for row in members}
    assert {
        "src/tnfr/research/maintained_sha_motion_source.py",
        "src/tnfr/research/maintained_sha_motion.py",
        "src/tnfr/research/maintained_principal_retention.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/physics/_sine_flow.py",
        "instrument.py",
        "pyproject.toml",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert _same(
            {item.filename: item.file_size for item in archive.infolist()},
            {row["path"]: row["bytes"] for row in members},
        )
    attempt = retained["attempt"]
    assert attempt["schema"] == "tnfr.maintained-sha-retention-attempt.v1"
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["seed"] is None
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, attempt["freeze"])
    assert retained["result"]["attempt"]["path"] == "attempt.json"
    _receipt(EVIDENCE, retained["result"]["attempt"])
    assert not (EVIDENCE / "incomplete.json").exists()


def test_source_rebuild_uses_actual_post_sha_point_and_all_42_coordinates(retained):
    source, admitted = retained["source"], retained["admitted"]
    prediction, initial = retained["prediction"], retained["initial"]
    assert source["schema"] == "tnfr.maintained-sha-retention-source.v1"
    restored = restore_recorded_graph_state(source["endpoint"]).graph
    assert _signature(restored) == _signature(admitted.graph)
    assert _same(source["retained_source_receipts"], admitted.receipts)
    assert _same(source["prediction"], prediction)
    assert _same(source["initial_box"], initial)
    assert len(initial) == 42 and all(value.lo == value.hi for value in initial)
    fields = admitted.assessment["actual_fields"]
    assert prediction.nodes == fields["nodes"]
    assert prediction.epi == fields["epi"] and prediction.phases == fields["phase"]
    assert prediction.capacities == fields["capacity"]
    assert prediction.capacities == admitted.prediction.represented_capacity_after
    assert prediction.capacities[2] < admitted.prediction.capacity_before[2]
    assert not admitted.prediction.source_um_parent_capacity.contains(
        prediction.capacities[2]
    )
    assert all(
        value.lo == value.hi == capacity
        for value, capacity in zip(
            prediction.bounds.capacities, prediction.capacities, strict=True
        )
    )
    assert all(value > 0 for value in prediction.capacities + prediction.conductances)
    assert len(prediction.edges) == len(prediction.shifts) == 53
    assert all(type(shift) is int for shift in prediction.shifts)
    assert tuple(admitted.graph.nodes[2]["glyph_history"]) == (
        "VAL",
        "THOL",
        "UM",
        "SHA",
    )
    assert "epi_time_history" not in admitted.graph.nodes["2_sub_0"]
    assert len(prediction.bounds.retained_edge_indices) == 41
    unresolved = tuple(
        edge
        for edge, margin in zip(
            prediction.edges, prediction.bounds.horizon_remaining_margins, strict=True
        )
        if margin.lo <= 0
    )
    assert len(unresolved) == 12 and all(2 in edge for edge in unresolved)


def test_frozen_protocol_keeps_complete_law_clock_and_retention_only_budget(retained):
    protocol = retained["protocol"]
    assert protocol["schema"] == "tnfr.maintained-sha-retention-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    expected = {
        "node_order": retained["prediction"].nodes,
        "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
        "prior_clock": TIMES[-1],
        "horizon": Q(1, 2),
        "step": STEP,
        "maximum_steps": COUNT,
        "order": ORDER,
        "maximum_attempts": 1,
        "maximum_events": 0,
        "wall_budget_seconds_between_steps": 900,
        "seed": None,
    }
    assert all(_same(protocol[key], value) for key, value in expected.items())
    assert retained["prediction"].horizon == COUNT * STEP


def test_retained_prefix_and_summary_follow_actual_law_arithmetic(retained):
    _check_summary(retained["result"]["report"], retained["audited_summary"])


def test_reusable_reader_returns_the_complete_admitted_chronological_prefix(retained):
    evidence = retained["evidence"]
    assert len(evidence.steps) == COUNT
    carried = evidence.initial
    for index, step in enumerate(evidence.steps):
        assert step.index == index and type(step.index) is int
        assert step.time == index * STEP and step.duration == STEP
        assert step.initial == carried
        assert len(step.initial) == len(step.tube) == len(step.endpoint) == 42
        assert all(type(value) is I for value in step.tube)
        assert all(
            value.subset_of(tube)
            for value, tube in zip(step.endpoint, step.tube, strict=True)
        )
        assert (
            step.principal_margins
            == maintained_principal_retention.principal_margins(
                step.tube, evidence.prediction.edges, evidence.prediction.shifts
            )
        )
        assert min(step.combined_principal_margins) > 0
        carried = step.endpoint
    assert evidence.summary["accepted_steps"] == len(evidence.steps)
    assert evidence.summary["full_horizon_retention"]
    assert len(evidence.receipts) == COUNT + 7
    assert len({row["path"] for row in evidence.receipts}) == len(evidence.receipts)
    for row in evidence.receipts:
        _receipt(EVIDENCE, row)
    with pytest.raises(FrozenInstanceError):
        evidence.steps[0].time = Q(1)


@pytest.mark.parametrize(
    "key,value",
    (
        ("maximum_events", True),
        ("maximum_steps", 256.0),
        ("horizon", 0.5),
        ("step", Q(1, 256)),
        ("prior_clock", Q(0)),
        ("law", "same equilibria"),
        ("seed", False),
    ),
)
def test_reader_rejects_changed_protocol_before_admitting_cells(retained, key, value):
    protocol = deepcopy(retained["protocol"])
    protocol[key] = value
    with pytest.raises(ValueError, match="retention protocol"):
        owner._validate_protocol(protocol, retained["prediction"])


def test_failed_attempt_returns_only_its_admitted_prefix_without_inventing_motion(
    retained,
):
    initial = retained["initial"]
    minimum = min(
        maintained_principal_retention.principal_margins(
            initial, retained["prediction"].edges, retained["prediction"].shifts
        )
    )
    record = {
        "index": 0,
        "time": Q(0),
        "step": None,
        "failed_tube": None,
        "reason": "enclosure did not close",
    }
    report = dict(
        status="enclosure_unavailable",
        attempted_steps=1,
        accepted_steps=0,
        certified_retention_time=Q(0),
        minimum_retained_margin=minimum,
        full_horizon_retention=False,
        elapsed_seconds=1.0,
    )
    summary, steps = owner._audit_summary(
        (record,),
        report,
        initial=initial,
        prediction=retained["prediction"],
        flow=retained["flow"],
    )
    assert steps == ()
    assert summary["status"] == "enclosure_unavailable"
    assert summary["certified_retention_time"] == 0
    with pytest.raises(ValueError, match="records after terminal"):
        owner._audit_summary(
            (record, retained["records"][0]),
            report,
            initial=initial,
            prediction=retained["prediction"],
            flow=retained["flow"],
        )


@pytest.mark.parametrize(
    "mutation", ("scalar", "carry", "chronology", "cached_margin", "endpoint", "order")
)
def test_altered_step_cannot_borrow_cached_validity(retained, mutation):
    record = deepcopy(retained["records"][0])
    assert record["step"] is not None
    if mutation == "scalar":
        record["step"]["initial_box"][0]["lo"] = False
    elif mutation == "carry":
        record["step"]["initial_box"][0]["lo"] -= 1
    elif mutation == "chronology":
        record["time"] = STEP
    elif mutation == "cached_margin":
        values = record["combined_principal_margins"]
        record["combined_principal_margins"] = (values[0] + 1,) + values[1:]
    elif mutation == "endpoint":
        record["step"]["endpoint"][0]["hi"] += 1
    else:
        record["step"]["order"] = float(ORDER)
    with pytest.raises(ValueError):
        _audit_step(
            record,
            index=0,
            initial=retained["initial"],
            prediction=retained["prediction"],
            flow=retained["flow"],
        )


@pytest.mark.parametrize(
    "mutation",
    ("accepted", "time_float", "margin_float", "retained_integer", "turning"),
)
def test_summary_cannot_override_retained_prefix_or_add_unreserved_readout(
    retained, mutation
):
    report = deepcopy(retained["result"]["report"])
    if mutation == "accepted":
        report["accepted_steps"] += 1
    elif mutation == "time_float":
        report["certified_retention_time"] = float(report["certified_retention_time"])
    elif mutation == "margin_float":
        report["minimum_retained_margin"] = float(report["minimum_retained_margin"])
    elif mutation == "retained_integer":
        report["full_horizon_retention"] = int(report["full_horizon_retention"])
    else:
        report["reversal_certified"] = True
    with pytest.raises(ValueError):
        _check_summary(report, retained["audited_summary"])


def test_recorded_interval_cannot_be_silently_rounded():
    with pytest.raises(ValueError, match="noncanonical"):
        _interval({"lo": Q(1, 3), "hi": Q(1, 2)})
