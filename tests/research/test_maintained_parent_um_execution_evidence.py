"""Audit the failed reserved UM attempt without replaying an event or flow.

The pressure-handoff guard rejected the attempt and the stage rolled back.
The retained raw intermediate permits detached reset/state-family checks; it
is not an accepted endpoint and does not establish runtime motion retention.
The inherited half-unit theorem has its separate motion-evidence audit owner.
Byte associations do not independently authenticate acquisition chronology.
"""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.grammar_debt import node_debt
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_parent_um_execution as owner
from tnfr.research import maintained_successor_motion_certificate
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import MODEL, TIMES, encode_result
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_parent_um_work import _bound_state, predict_parent_um_work
from tnfr.research.maintained_successor_motion import predict_successor_motion
from tnfr.research.retained_graph_state import (
    encode_recorded_container,
    restore_recorded_graph_state,
)
from tnfr.utils._structural_signature import structural_proof_signature
from tnfr.utils.cache import _graph_cache_manager
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_um_execution"
MOTION = ROOT / "docs/assets/successor_nonstationary"
PARENT = 2
CHILD = "2_sub_0"


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def _same(left, right):
    def canonical(value):
        return json_dumps(encode_exact_tree(encode_result(value)), sort_keys=True)

    return canonical(left) == canonical(right)


def _same_state(left, right):
    def canonical(value):
        return json_dumps(encode_exact_tree(encode_recorded_container(value)))

    return canonical(left) == canonical(right)


def _receipt(root, record):
    name = validate_artifact_path(record["path"])
    path = root / name
    assert path.resolve().is_relative_to(root.resolve())
    assert file_receipt(path, label=name) == record


def _values(graph, alias):
    return tuple(
        _represented(
            get_attr(graph.nodes[node], alias, strict=True, conv=lambda value: value),
            f"retained coordinate at {node!r}",
        )
        for node in graph
    )


def _farthest(value, interval):
    return max(abs(value - interval.lo), abs(value - interval.hi))


@pytest.fixture(scope="module", autouse=True)
def no_event_or_motion_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("retained UM audit cannot execute an operator or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (owner, "evaluate_um_event"),
            (owner, "refresh_pressure"),
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
    # Missing reserved records fail explicitly; tests never run a producer.
    names = (
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "freeze.json",
        "attempt.json",
        "raw-after.json",
        "failure.json",
        "failure-endpoint.json",
        "diagnosis.json",
        "detached-refreshed.json",
    )
    records = {name: _read(EVIDENCE / name) for name in names}
    paths = tuple(sorted(path for path in EVIDENCE.rglob("*") if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    admitted = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    graphs = tuple(
        restore_recorded_graph_state(record).graph
        for record in (
            records["source.json"]["before"],
            records["raw-after.json"],
            records["failure-endpoint.json"],
        )
    )
    signatures = tuple(map(_signature, graphs))
    detached = restore_recorded_graph_state(records["detached-refreshed.json"]).graph
    detached_signature = _signature(detached)
    prediction = predict_successor_motion(graphs[0])
    work = predict_parent_um_work(graphs[0])
    raw, nodes = graphs[1], prediction.chamber.reset.nodes
    raw_state = _bound_state(
        _values(raw, ALIAS_EPI),
        tuple(map(I, _values(raw, ALIAS_THETA))),
        tuple(map(I, _values(raw, ALIAS_VF))),
        prediction.chamber.edges,
        tuple(
            I(_represented(raw[nodes[i]][nodes[j]]["weight"], "conductance"))
            for i, j in prediction.chamber.edges
        ),
    )
    yield dict(
        records=records,
        admitted=admitted,
        graphs=graphs,
        prediction=prediction,
        work=work,
        detached=detached,
        raw_state=raw_state,
    )
    assert tuple(map(_signature, graphs)) == signatures
    assert _signature(detached) == detached_signature
    assert paths == tuple(
        sorted(path for path in EVIDENCE.rglob("*") if path.is_file())
    )
    assert receipts == tuple(file_receipt(path) for path in paths)


def test_freeze_archive_failure_and_inherited_motion_associations(retained):
    records = retained["records"]
    freeze, attempt, failure = (
        records[name] for name in ("freeze.json", "attempt.json", "failure.json")
    )
    assert freeze["schema"] == "tnfr.maintained-parent-um-execution-freeze.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert {entry["path"] for entry in freeze["artifacts"]} == {
        "protocol.json",
        "source.json",
        "source-manifest.json",
        "source.zip",
    }
    for item in freeze["artifacts"]:
        _receipt(EVIDENCE, item)
    for item in freeze["prior_artifacts"]:
        _receipt(ROOT, item)
    entries = records["source-manifest.json"]["files"]
    assert len(entries) == len({row["path"] for row in entries})
    expected = {row["path"]: row["sha256"] for row in entries}
    assert {
        "src/tnfr/research/maintained_parent_um_execution.py",
        "src/tnfr/research/maintained_successor_motion.py",
        "src/tnfr/operators/_coupling_stage_kernel.py",
        "src/tnfr/operators/network_stage.py",
        "pyproject.toml",
        "instrument.py",
    } <= expected.keys()
    verify_archive_members(EVIDENCE / "source.zip", expected)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {row.filename: row.file_size for row in archive.infolist()} == {
            row["path"]: row["bytes"] for row in entries
        }
    _receipt(EVIDENCE, attempt["freeze"])
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, failure["attempt"])
    assert failure["attempt"]["path"] == "attempt.json"
    assert failure["schema"] == "tnfr.maintained-parent-um-execution-failure.v1"
    assert attempt["schema"] == "tnfr.maintained-parent-um-execution-attempt.v1"
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert type(attempt["seed"]) is int and attempt["seed"] == 17
    source = records["source.json"]
    assert {row["path"] for row in source["existing_motion_receipts"]} == {
        f"docs/assets/successor_nonstationary/{name}"
        for name in ("freeze.json", "source.json", "attempt.json", "result.json")
    }
    for row in source["existing_motion_receipts"]:
        _receipt(ROOT, row)
    assert not (EVIDENCE / "response.json").exists()
    assert not (EVIDENCE / "report.json").exists()


def test_before_source_and_cached_predictions_are_independently_rebuilt(retained):
    source = retained["records"]["source.json"]
    before = retained["graphs"][0]
    assert source["schema"] == "tnfr.maintained-parent-um-execution-source.v1"
    assert _signature(before) == _signature(retained["admitted"].graph)
    assert _signature(
        restore_recorded_graph_state(source["before"]).graph
    ) == _signature(before)
    assert _same(source["motion_prediction"], retained["prediction"])
    assert _same(source["work_prediction"], retained["work"])
    assert _same(source["retained_source_receipts"], retained["admitted"].receipts)
    motion_source = _read(MOTION / "source.json")
    assert _same(motion_source["prediction"], retained["prediction"])
    conditional = retained["prediction"].conditional
    assert _same(motion_source["initial_box"], conditional.epi + conditional.phases)


def test_guard_failure_and_exact_persistent_rollback_remain_negative_evidence(retained):
    before, raw, rolled_back = retained["graphs"]
    diagnostic = retained["records"]["failure.json"]["diagnostic"]
    assert type(diagnostic) is str
    assert diagnostic.endswith("ValueError: UM handoff changed non-pressure state\n")
    assert "compute_delta_nfr(graph)" in diagnostic
    assert "_pressure_preservation_signature(successor, runtime=True)" in diagnostic
    assert _signature(rolled_back) == _signature(before)
    assert _signature(raw) != _signature(before)
    assert tuple(rolled_back.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL")
    assert rolled_back.degree(CHILD) == 0
    assert rolled_back.number_of_edges() == before.number_of_edges() == 45
    assert raw.number_of_edges() == 53


def test_protocol_keeps_one_hybrid_stage_and_no_new_motion(retained):
    protocol = retained["records"]["protocol.json"]
    assert protocol["schema"] == "tnfr.maintained-parent-um-execution-protocol.v1"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["word_context"] == ("VAL", "THOL", "UM", "SHA")
    assert protocol["target_order"] == (PARENT,)
    assert all(
        type(protocol[key]) is int
        for key in (
            "step_index",
            "seed",
            "maximum_events",
            "maximum_attempts",
            "maximum_trajectories",
        )
    )
    assert protocol["step_index"] == 2
    assert protocol["seed"] == 17
    assert protocol["structural_time"] == TIMES[-1]
    assert protocol["event_metadata_timestamp"] == "2030-01-02T03:04:07+00:00"
    assert protocol["maximum_events"] == protocol["maximum_attempts"] == 1
    assert protocol["maximum_trajectories"] == 0
    assert protocol["reset_error_allowance"] == Q(1, 2**40)
    assert protocol["pressure_refresh_error_allowance"] == Q(1, 2**46)


def _audit_raw_coordinates(before, raw, prediction):
    chamber, conditional = prediction.chamber, prediction.conditional
    reset, epsilon = chamber.reset, chamber.numerical_allowance
    nodes = reset.nodes
    assert tuple(before) == tuple(raw) == nodes
    assert _values(raw, ALIAS_EPI) == reset.epi
    for index in reset.phase_changed_indices:
        data = raw.nodes[nodes[index]]
        assert all(key in data for key in ALIAS_THETA)
        phases = tuple(
            _represented(data[key], "changed phase alias") for key in ALIAS_THETA
        )
        assert len(set(phases)) == 1
    for alias, bounds, ideals, changed in (
        (
            ALIAS_THETA,
            conditional.phases,
            chamber.ideal.phases,
            reset.phase_changed_indices,
        ),
        (
            ALIAS_VF,
            conditional.capacities,
            reset.capacity_after,
            reset.capacity_changed_indices,
        ),
    ):
        values, old = _values(raw, alias), _values(before, alias)
        assert all(
            I(value).subset_of(bound)
            for value, bound in zip(values, bounds, strict=True)
        )
        assert all(
            _farthest(values[index], ideals[index]) <= epsilon for index in changed
        )
        assert all(
            values[index] == old[index] for index in range(21) if index not in changed
        )
    old_edges = {(nodes[i], nodes[j]) for i, j in reset.old_edges}
    new_edges = {(nodes[i], nodes[j]) for i, j in reset.new_edges}
    rank = {node: index for index, node in enumerate(nodes)}
    observed = {tuple(sorted(edge, key=rank.__getitem__)) for edge in raw.edges}
    assert (
        observed == old_edges | new_edges
        and len(new_edges) == 8
        and len(observed) == 53
    )
    weights = tuple(
        _represented(raw[nodes[i]][nodes[j]]["weight"], "edge weight")
        for i, j in chamber.edges
    )
    assert all(
        I(value).subset_of(bound)
        for value, bound in zip(weights, conditional.conductances, strict=True)
    )
    for (i, j), ideal in zip(reset.new_edges, reset.new_conductances, strict=True):
        assert set(raw[nodes[i]][nodes[j]]) == {"weight"}
        assert (
            _farthest(
                _represented(raw[nodes[i]][nodes[j]]["weight"], "new weight"), ideal
            )
            <= epsilon
        )
    for left, right in old_edges:
        assert _same(raw[left][right], before[left][right])
    phases, pi = _values(raw, ALIAS_THETA), pi_interval()
    assert all(
        (I(phases[j]) - phases[i] - 2 * shift * pi).abs_max < pi.lo
        for (i, j), shift in zip(chamber.edges, chamber.conditional.shifts, strict=True)
    )


def test_raw_intermediate_only_enters_the_frozen_mathematical_family(retained):
    before, raw, _ = retained["graphs"]
    _audit_raw_coordinates(before, raw, retained["prediction"])


def _audit_raw_history(before, raw):
    assert tuple(before.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL")
    assert tuple(raw.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL", "UM")
    assert type(raw.nodes[PARENT]["_operator_step"]) is int
    assert raw.nodes[PARENT]["_operator_step"] == 3
    assert type(raw.nodes[PARENT]["_grammar_u2_debt"]) is int
    assert node_debt(raw.nodes[PARENT]) == 0
    assert _same_state(
        raw.nodes[PARENT]["epi_time_history"], before.nodes[PARENT]["epi_time_history"]
    )
    assert _same_state(raw.graph["hierarchy"], before.graph["hierarchy"])
    assert _represented(raw.graph["_t"], "structural clock") == TIMES[-1]
    assert (
        raw.nodes[CHILD]["glyph_history"] == before.nodes[CHILD]["glyph_history"] == []
    )
    assert _same_state(raw.nodes[CHILD], before.nodes[CHILD])
    assert not any(key in raw.nodes[CHILD] for key in ALIAS_DEPI)
    assert "epi_time_history" not in raw.nodes[CHILD]


def test_raw_prefix_preserves_parent_history_and_child_availability(retained):
    before, raw, _ = retained["graphs"]
    _audit_raw_history(before, raw)


def _audit_raw_pressure(before, raw, prediction):
    reset = prediction.chamber.reset
    values = _values(raw, ALIAS_DNFR)
    assert _farthest(values[PARENT], reset.raw_pressure_after[PARENT]) <= Q(1, 2**40)
    assert all(
        value == reset.raw_pressure_before[index]
        for index, value in enumerate(values)
        if index != PARENT
    )
    # Canonical pressure writes preserve the first-present key and every shadow.
    for node in raw:
        key = next(key for key in ALIAS_DNFR if key in before.nodes[node])
        assert next(key for key in ALIAS_DNFR if key in raw.nodes[node]) == key
        for shadow in ALIAS_DNFR:
            if shadow != key:
                assert (shadow in raw.nodes[node]) == (shadow in before.nodes[node])
                if shadow in before.nodes[node]:
                    assert _same(raw.nodes[node][shadow], before.nodes[node][shadow])
    for key in ("_last_network_stage_schedule", "_last_network_stage_contract"):
        assert _same(raw.graph[key], before.graph[key])


def test_raw_record_precedes_pressure_refresh_and_final_stage_metadata(retained):
    before, raw, _ = retained["graphs"]
    _audit_raw_pressure(before, raw, retained["prediction"])
    # There is no successful callback endpoint to compare or certify.
    assert not (EVIDENCE / "response.json").exists()


def test_detached_raw_work_and_law_rows_rebuild_without_pressure_refresh(retained):
    before, raw, _ = retained["graphs"]
    prediction = retained["prediction"]
    nodes, reset = prediction.chamber.reset.nodes, prediction.chamber.reset
    x, phase_after = _values(raw, ALIAS_EPI), _values(raw, ALIAS_THETA)
    phase_before = _values(before, ALIAS_THETA)
    old_phase = sum(
        (
            cos(I(phase_before[j]) - phase_before[i])
            - cos(I(phase_after[j]) - phase_after[i])
            for i, j in reset.old_edges
        ),
        I(0),
    )
    added = sum(
        (
            I(_represented(raw[nodes[i]][nodes[j]]["weight"], "new weight"))
            * (x[j] - x[i]) ** 2
            / 2
            + 1
            - cos(I(phase_after[j]) - phase_after[i])
            for i, j in reset.new_edges
        ),
        I(0),
    )
    work = old_phase + added
    assert work.lo > 0
    assert work.subset_of(retained["work"].conditional.work.total)
    fields = bound_relational_sine_support_exchange(raw, reference_model=MODEL)
    assert fields.nodes == nodes
    actual_state = retained["raw_state"]
    # The stored raw pressure is not the independently specified refreshed map.
    assert any(
        _farthest(value, exact) > Q(1, 2**46)
        for value, exact in zip(_values(raw, ALIAS_DNFR), fields.pressure, strict=True)
    )
    pi, capacity = pi_interval(), _values(raw, ALIAS_VF)
    positions = {node: index for index, node in enumerate(nodes)}
    for index, node in enumerate(nodes):
        degree = raw.degree[node]
        current = sum(
            (
                sin(I(phase_after[positions[neighbor]]) - phase_after[index])
                for neighbor in raw[node]
            ),
            I(0),
        )
        gradient = sum(
            (
                _represented(data["weight"], "conductance")
                * (x[index] - x[positions[neighbor]])
                for neighbor, data in raw[node].items()
            ),
            Q(0),
        )
        expected = (
            capacity[index] * current / (pi * degree),
            capacity[index] * gradient / (pi * degree),
        )
        for actual, independent, bound in zip(
            (actual_state.form_rate[index], actual_state.phase_rate[index]),
            expected,
            (
                prediction.conditional.form_rate[index],
                prediction.conditional.phase_rate[index],
            ),
            strict=True,
        ):
            defect = actual - independent
            assert defect.contains(0) and defect.abs_max < Q(1, 10**30)
            assert actual.subset_of(bound)


def test_separate_diagnosis_binds_the_frozen_guard_and_keeps_failure_status(retained):
    records = retained["records"]
    diagnosis = records["diagnosis.json"]
    assert diagnosis["schema"] == "tnfr.maintained-parent-um-guard-diagnosis.v1"
    expected_inputs = {
        "freeze.json",
        "attempt.json",
        "raw-after.json",
        "failure.json",
        "failure-endpoint.json",
        "diagnose.py",
    }
    assert {row["path"] for row in diagnosis["inputs"]} == expected_inputs
    for row in diagnosis["inputs"]:
        _receipt(EVIDENCE, row)
    _receipt(EVIDENCE, diagnosis["detached_refreshed"])
    assert diagnosis["detached_refreshed"]["path"] == "detached-refreshed.json"
    frozen_guard = next(
        row
        for row in records["source-manifest.json"]["files"]
        if row["path"] == "src/tnfr/research/maintained_parent_um_execution.py"
    )
    assert _same(diagnosis["guard_source"], frozen_guard)
    assert diagnosis["original_attempt_status"] == "failed_and_rolled_back"
    assert diagnosis["runtime_guard_unchanged"] is False
    assert diagnosis["runtime_successor_committed"] is False
    assert diagnosis["motion_certificate_transferred_to_runtime"] is False
    for key in (
        "persistent_pressure_only_guard_unchanged",
        "cache_owned_payload_unchanged_with_opaque_owner",
        "restoring_only_allowed_maxima_restores_frozen_guard",
        "rollback_matches_frozen_before",
    ):
        assert diagnosis[key] is True


def test_detached_diagnostic_primitives_explain_the_recursive_guard_rejection(retained):
    _, raw, _ = retained["graphs"]
    detached = retained["detached"]
    diagnosis = retained["records"]["diagnosis.json"]
    maxima = ("_dnfrmax", "_dnfrmax_node")
    assert _same(
        {key: raw.graph[key] for key in maxima}, diagnosis["allowed_maxima_before"]
    )
    assert _same(
        {key: detached.graph[key] for key in maxima}, diagnosis["allowed_maxima_after"]
    )
    # Verify the disposable refreshed record using an independently specified
    # pressure map. No refresh, operator or trajectory is executed by this reader.
    for value, bound in zip(
        _values(detached, ALIAS_DNFR), retained["raw_state"].pressure, strict=True
    ):
        assert _farthest(value, bound) <= Q(1, 2**46)
    stripped = []
    for graph in (raw, detached):
        graph = deepcopy(graph)
        for node in graph:
            key = next(key for key in ALIAS_DNFR if key in raw.nodes[node])
            assert next(key for key in ALIAS_DNFR if key in graph.nodes[node]) == key
            del graph.nodes[node][key]
        for key in maxima:
            del graph.graph[key]
        stripped.append(_signature(graph))
    assert stripped[0] == stripped[1]
    # Reconstruct only a fresh graph-owned cache manager, then substitute the
    # two recorded maxima on this disposable copy. Recursive owner traversal
    # reintroduces those expressly allowed values; its own payload stays fixed.
    graph = deepcopy(raw)
    manager = _graph_cache_manager(graph.graph)
    assert manager._graph_owner is graph.graph
    own_payload = structural_proof_signature(manager, opaque_references=(graph.graph,))
    recursive = structural_proof_signature(manager)
    graph.graph.update({key: detached.graph[key] for key in maxima})
    assert (
        structural_proof_signature(manager, opaque_references=(graph.graph,))
        == own_payload
    )
    assert structural_proof_signature(manager) != recursive
    graph.graph.update({key: raw.graph[key] for key in maxima})
    assert structural_proof_signature(manager) == recursive


@pytest.mark.parametrize(
    "mutation",
    (
        "form",
        "capacity",
        "phase",
        "weight",
        "history",
        "hierarchy_type",
        "phase_shadow",
        "pressure_shadow",
        "boolean",
    ),
)
def test_altered_raw_primitives_cannot_borrow_the_frozen_prediction(retained, mutation):
    before, raw, _ = deepcopy(retained["graphs"])
    if mutation == "form":
        set_attr(raw.nodes[PARENT], ALIAS_EPI, 0.25)
    elif mutation == "capacity":
        set_attr(
            raw.nodes[PARENT],
            ALIAS_VF,
            float(get_attr(raw.nodes[PARENT], ALIAS_VF)) + 1,
        )
    elif mutation == "phase":
        set_attr(
            raw.nodes[PARENT],
            ALIAS_THETA,
            float(get_attr(raw.nodes[PARENT], ALIAS_THETA)) + 0.01,
        )
    elif mutation == "weight":
        raw[PARENT][CHILD]["weight"] += 0.01
    elif mutation == "history":
        raw.nodes[PARENT]["glyph_history"] = []
    elif mutation == "hierarchy_type":
        assert (
            type(raw.nodes[CHILD]["hierarchy_level"]) is int
            and raw.nodes[CHILD]["hierarchy_level"] == 1
        )
        raw.nodes[CHILD]["hierarchy_level"] = True
    elif mutation == "phase_shadow":
        raw.nodes[PARENT][ALIAS_THETA[-1]] = "invalid shadow alias"
    elif mutation == "pressure_shadow":
        raw.nodes[PARENT][ALIAS_DNFR[-1]] = "invalid shadow alias"
    else:
        raw.nodes[PARENT][ALIAS_VF[0]] = True
    with pytest.raises((AssertionError, TypeError, ValueError)):
        _audit_raw_coordinates(before, raw, retained["prediction"])
        _audit_raw_history(before, raw)
        _audit_raw_pressure(before, raw, retained["prediction"])


def test_equivalent_boolean_metadata_cannot_disappear_in_rollback_comparison(retained):
    before, _, rolled_back = deepcopy(retained["graphs"])
    assert type(rolled_back.nodes[CHILD]["hierarchy_level"]) is int
    assert rolled_back.nodes[CHILD]["hierarchy_level"] == 1
    rolled_back.nodes[CHILD]["hierarchy_level"] = True
    assert _signature(rolled_back) != _signature(before)
