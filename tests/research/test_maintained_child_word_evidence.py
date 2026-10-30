"""Independent read-only audit of the frozen child THOL/SHA continuation.

The retained source is admitted once. No event, proposal, pressure refresh,
trajectory or new temporal sample is evaluated. Byte receipts associate the
preserved evidence; they do not authenticate acquisition or occurrence.
"""

from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import mpmath
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from tnfr.dynamics import integrators, relational
from tnfr.operators import network_stage, word_execution
from tnfr.operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.research import local_capacity_history, maintained_child_history_acquisition
from tnfr.research import maintained_child_word_execution as producer
from tnfr.research import maintained_parent_thol_execution, maintained_sha_motion
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.local_capacity_history import encode_result
from tnfr.research.maintained_child_history_source import load_child_history_source
from tnfr.research.maintained_child_word_check import (
    EVENT_TIMESTAMP,
    check_child_sha_endpoint,
    check_child_thol_endpoint,
)
from tnfr.research.maintained_child_word_prediction import _predict_child_word
from tnfr.research.maintained_parent_sha_source import _same
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.retained_graph_state import (
    encode_recorded_container,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_word"
SOURCE = ROOT / "docs/assets/maintained_child_history"
ACTOR, CHILD = "2_sub_0", "2_sub_0_sub_0"
OLD_NODES = (*range(20), ACTOR)
NEW_NODES = (*OLD_NODES, CHILD)
TIME = Q(3683329, 524288)
MAX_BYTES = 128 * 1024**2
FROZEN = {
    "protocol.json",
    "preparation.json",
    "prediction.json",
    "source-manifest.json",
    "source.zip",
}
RECORDS = {
    "source.json",
    "thol_raw.json",
    "thol_after.json",
    "sha_raw.json",
    "sha_after.json",
    "response.json",
}


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=MAX_BYTES))
    )


def _receipt(base, row):
    assert type(row) is dict and set(row) == {"path", "bytes", "sha256"}
    name = validate_artifact_path(row["path"])
    assert type(row["bytes"]) is int and row["bytes"] >= 0
    path = (base / name).resolve()
    assert path.is_relative_to(base.resolve())
    assert file_receipt(path, label=name, max_bytes=MAX_BYTES) == row


def _value(graph, node, aliases):
    return _represented(
        get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value),
        "retained child-word coordinate",
    )


def _values(graph, aliases, nodes=OLD_NODES):
    return tuple(_value(graph, node, aliases) for node in nodes)


def _mp(value):
    value = Q(value)
    return mpmath.mpf(value.numerator) / value.denominator


def _contains(bounds, value):
    return _mp(bounds.lo) <= value <= _mp(bounds.hi)


@pytest.fixture(scope="module", autouse=True)
def no_event_proposal_refresh_or_response():
    def forbidden(*args, **kwargs):
        pytest.fail("retained word audit cannot execute a proposal, event or flow")

    with pytest.MonkeyPatch.context() as patch:
        for owner, names in (
            (
                producer,
                (
                    "evaluate_child_word",
                    "_execute_child_word",
                    "execute_network_operator_stage",
                    "refresh_frozen_successor",
                ),
            ),
            (
                maintained_parent_thol_execution,
                ("evaluate_thol_event", "refresh_frozen_successor", "refresh_pressure"),
            ),
            (
                maintained_child_history_acquisition,
                (
                    "evaluate_child_history",
                    "_acquire_child_history",
                    "_advance",
                    "rk4_coupled_step",
                    "refresh_pressure",
                    "numerical_field",
                ),
            ),
            (local_capacity_history, ("numerical_field", "refresh_pressure")),
            (integrators, ("rk4_coupled_step",)),
            (relational, ("step_relational_exchange",)),
            (word_execution, ("execute_network_operator_stage",)),
            (
                network_stage,
                ("execute_self_organization_stage", "execute_pointwise_stage"),
            ),
            (SelfOrganization, ("_prepare_execution", "_prepare_bifurcation")),
            (
                maintained_sha_motion,
                ("certify_sha_retention", "validated_box_taylor_step"),
            ),
        ):
            for name in names:
                patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def evidence():
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    assert {path.name for path in paths} == FROZEN | RECORDS | {
        "freeze.json",
        "attempt.json",
        "result.json",
    }
    receipts = tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)
    documents = {
        path.stem: _read(path.name) for path in paths if path.suffix == ".json"
    }
    yield documents
    assert paths == tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    assert receipts == tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)


@pytest.fixture(scope="module")
def admitted():
    result = load_child_history_source(SOURCE, repository_root=ROOT)
    original = _signature(result.graph)
    yield result
    assert _signature(result.graph) == original


@pytest.fixture(scope="module")
def graphs(evidence):
    response = evidence["response"]
    return tuple(
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "thol_raw", "thol_after", "sha_raw", "endpoint")
    )


@pytest.fixture(scope="module")
def prediction(admitted):
    return _predict_child_word(admitted)


def test_frozen_protocol_archive_and_prior_bytes_keep_exact_associations(evidence):
    freeze, attempt, result = (evidence[key] for key in ("freeze", "attempt", "result"))
    assert freeze["schema"] == "tnfr.maintained-child-word-freeze.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert {row["path"] for row in freeze["artifacts"]} == FROZEN
    assert len(freeze["artifacts"]) == len(FROZEN)
    for row in freeze["artifacts"]:
        _receipt(EVIDENCE, row)
    assert len(freeze["prior_artifacts"]) == len(
        {row["path"] for row in freeze["prior_artifacts"]}
    )
    for row in freeze["prior_artifacts"]:
        _receipt(ROOT, row)
    members = evidence["source-manifest"]["files"]
    assert len(members) == len({row["path"] for row in members})
    verify_archive_members(
        EVIDENCE / "source.zip",
        {row["path"]: row["sha256"] for row in members},
        max_bytes=MAX_BYTES,
    )
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {item.filename: item.file_size for item in archive.infolist()} == {
            row["path"]: row["bytes"] for row in members
        }
    owners = {
        "src/tnfr/research/maintained_child_history_source.py",
        "src/tnfr/research/maintained_child_word_prediction.py",
        "src/tnfr/research/maintained_child_word_check.py",
        "src/tnfr/research/maintained_child_word_execution.py",
        "src/tnfr/operators/network_stage.py",
        "src/tnfr/operators/word_execution.py",
        "instrument.py",
        "pyproject.toml",
    }
    assert owners <= {row["path"] for row in members}
    # The instrument verified live source bytes against this archive before
    # evaluation. Preserve that archive, rather than locking unrelated future
    # source changes; current consumers rebuild the primitive checks below.
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    _receipt(EVIDENCE, attempt["freeze"])
    assert attempt["freeze"]["path"] == "freeze.json"
    _receipt(EVIDENCE, result["attempt"])
    assert result["attempt"]["path"] == "attempt.json"
    assert {row["path"] for row in result["artifacts"]} == RECORDS
    assert len(result["artifacts"]) == len(RECORDS)
    for row in result["artifacts"]:
        _receipt(EVIDENCE, row)
    assert datetime.fromisoformat(freeze["created_utc"]) <= datetime.fromisoformat(
        attempt["created_utc"]
    )
    assert result["status"] == "completed" and result["failure"] is None
    assert result["committed_event_records"] == ("thol_after", "sha_after")
    assert result["numerical_steps"] == result["new_history_samples"] == 0


def test_protocol_scope_and_fresh_prediction_precede_the_events(
    evidence, admitted, graphs, prediction
):
    protocol, response = evidence["protocol"], evidence["response"]
    assert protocol["source_node_order"] == OLD_NODES
    assert protocol["successor_node_order"] == NEW_NODES
    assert protocol["reserved_events"] == ("THOL", "SHA")
    assert protocol["context_word"] == ("OZ", "THOL", "SHA")
    assert protocol["already_executed_prefix"] == ("OZ",)
    assert protocol["target_order_each_stage"] == (ACTOR,)
    assert protocol["structural_time"] == TIME
    assert protocol["sha_metadata_timestamp"] == EVENT_TIMESTAMP
    assert protocol["maximum_reserved_evaluations"] == 1
    assert protocol["maximum_events"] == 2
    assert protocol["numerical_steps"] == protocol["new_history_samples"] == 0
    assert protocol["new_edges"] == protocol["forcing"] == protocol["loss"] == 0
    assert protocol["event_comparison_allowance"] == Q(1, 2**44)
    assert protocol["represented_point_pressure_error_allowance"] == Q(1, 2**46)
    assert protocol["physical_identification"] is False
    assert _signature(graphs[0]) == _signature(admitted.graph)
    for name in ("preparation", "source"):
        assert _signature(
            restore_recorded_graph_state(evidence[name]["endpoint"]).graph
        ) == _signature(admitted.graph)
        assert _same(evidence[name]["source_receipts"], admitted.receipts)
    assert evidence["prediction"]["actual_event_evaluated"] is False
    for cached in (
        evidence["prediction"]["prediction"],
        evidence["source"]["prediction"],
        response["prediction"],
    ):
        assert _same(cached, encode_result(prediction))
    assert _same(response["source_receipts"], admitted.receipts)
    assert prediction.events_executed == 0 and not prediction.complete_word_executed
    assert not prediction.future_from_acquired_point_certified
    assert not prediction.old_retention_transferred_after_capacity_reset


def test_durable_callbacks_identify_raw_and_committed_boundaries(evidence, graphs):
    before, thol_raw, thol, sha_raw, after = graphs
    for name, graph, prefix in (
        ("thol_raw", thol_raw, ()),
        ("thol_after", thol, ("THOL",)),
        ("sha_raw", sha_raw, ("THOL",)),
        ("sha_after", after, ("THOL", "SHA")),
    ):
        record = evidence[name]
        assert _signature(
            restore_recorded_graph_state(record["endpoint"]).graph
        ) == _signature(graph)
        assert record["completed_stage_prefix"] == prefix
    assert tuple(before) == OLD_NODES
    assert all(tuple(graph) == NEW_NODES for graph in graphs[1:])
    for name, glyph in (("thol_after", "THOL"), ("sha_after", "SHA")):
        stage = evidence[name]["stage"]
        assert stage["glyph"] == glyph and stage["nodes_processed"] == 1
        assert stage["schedule"] == "two_phase_jacobi"
        assert _same(stage, evidence["response"]["stages"][glyph.lower()])
    response = evidence["response"]
    assert response["completed_stage_prefix"] == ("THOL", "SHA")
    assert response["complete_word_executed"] is True and response["new_births"] == 1
    assert response["continuous_steps"] == response["new_temporal_samples"] == 0
    assert not response["future_retention_certified"]
    assert not response["autonomous_occurrence_derived"]
    assert not response["physical_identification"]


def test_full_primitive_footprints_and_law_checks_are_rebuilt(
    evidence, admitted, graphs, prediction
):
    _, thol_raw, thol, sha_raw, after = graphs
    response = evidence["response"]
    thol_check = check_child_thol_endpoint(
        admitted,
        thol_raw,
        thol,
        prediction=prediction,
        stage=response["stages"]["thol"],
    )
    sha_check = check_child_sha_endpoint(
        admitted,
        thol,
        sha_raw,
        after,
        prediction=prediction,
        stage=response["stages"]["sha"],
    )
    for name, rebuilt in (("thol", thol_check), ("sha", sha_check)):
        assert rebuilt["all_checks_passed"] is True
        assert _same(rebuilt, response["assessments"][name])
        assert rebuilt["edge_storage_jump"] == 0
        assert not rebuilt["newborn_history_available"]
        assert not rebuilt["future_retention_certified"]
        assert not rebuilt["numerical_u5_observation_available"]
        assert not rebuilt["cross_birth_u6_observation_available"]


def test_independent_binary64_reset_products_and_degree_one_metabolism(
    graphs, prediction
):
    before, raw, thol, sha_raw, after = graphs
    observation = observe_structural_acceleration(before, ACTOR)
    assert observation.available and observation.current_endpoint_matches_state
    acceleration = _represented(observation.value, "observed acceleration")
    pressure = _value(before, ACTOR, ALIAS_DNFR)
    increment = Q(float(Q(COUPLING_GENTLE) * acceleration))
    raw_pressure = Q(float(pressure + increment))
    assert _value(raw, ACTOR, ALIAS_DNFR) == raw_pressure
    capacity = _value(before, ACTOR, ALIAS_VF)
    child_capacity = Q(float(capacity * Q(THOL_CHILD_VF_DAMPING)))
    closed_capacity = Q(float(capacity * prediction.sha_factor))
    assert all(_value(graph, CHILD, ALIAS_VF) == child_capacity for graph in graphs[1:])
    assert _value(thol, ACTOR, ALIAS_VF) == capacity
    assert (
        _value(sha_raw, ACTOR, ALIAS_VF)
        == _value(after, ACTOR, ALIAS_VF)
        == closed_capacity
    )
    assert 0 < closed_capacity < capacity
    assert tuple(before.neighbors(ACTOR)) == (2,)
    x, neighbor = _value(before, ACTOR, ALIAS_EPI), _value(before, 2, ALIAS_EPI)
    ideal = Q(THOL_SUB_EPI_SCALING) * x + Q(COUPLING_MODERATE) * (neighbor - x)
    child_form = _value(raw, CHILD, ALIAS_EPI)
    assert abs(child_form - ideal) <= Q(1, 2**44)
    assert 0 < child_form < Q(0.2)
    assert all(
        _value(graph, CHILD, ALIAS_THETA) == _value(before, ACTOR, ALIAS_THETA)
        for graph in graphs[1:]
    )
    record = raw.nodes[ACTOR]["sub_epis"][0]
    signals = record["network_signals"]
    assert type(signals["neighbor_count"]) is int and signals["neighbor_count"] == 1
    assert abs(Q(signals["mean_neighbor_epi"]) - neighbor) <= Q(1, 2**44)
    assert abs(Q(signals["epi_gradient"]) - (neighbor - x)) <= Q(1, 2**44)
    assert abs(Q(signals["phase_variance"])) <= Q(1, 2**44)
    assert record["timestamp"] == 2 and Q(record["d2_epi"]) == acceleration
    assert Q(record["tau"]) == Q(0.1)


def test_independent_unchanged_edge_storage_and_changed_mean_weights(
    graphs, prediction
):
    before, _, thol, _, after = graphs
    edges = tuple((a, b, dict(data)) for a, b, data in before.edges(data=True))
    for graph in graphs[1:]:
        assert (
            tuple((a, b, dict(data)) for a, b, data in graph.edges(data=True)) == edges
        )
        assert graph.degree(CHILD) == 0
        assert _values(graph, ALIAS_EPI) == _values(before, ALIAS_EPI)
        assert _values(graph, ALIAS_THETA) == _values(before, ALIAS_THETA)
        assert Q(graph.graph["_t"]) == TIME
    # Every argument of the edge-only storage remains identical. No subtraction
    # of overlapping marginal energy intervals is used as a zero-jump proof.
    source_arguments = tuple(
        (
            Q(data["weight"]),
            _value(before, a, ALIAS_EPI) - _value(before, b, ALIAS_EPI),
            _value(before, a, ALIAS_THETA) - _value(before, b, ALIAS_THETA),
        )
        for a, b, data in edges
    )
    for graph in (thol, after):
        assert source_arguments == tuple(
            (
                Q(data["weight"]),
                _value(graph, a, ALIAS_EPI) - _value(graph, b, ALIAS_EPI),
                _value(graph, a, ALIAS_THETA) - _value(graph, b, ALIAS_THETA),
            )
            for a, b, data in edges
        )
    means = []
    for graph in (before, after):
        weights = tuple(
            Q(graph.degree(node)) / _value(graph, node, ALIAS_VF) for node in OLD_NODES
        )
        means.append(
            tuple(
                sum(
                    w * x for w, x in zip(weights, _values(graph, aliases), strict=True)
                )
                / sum(weights)
                for aliases in (ALIAS_EPI, ALIAS_THETA)
            )
        )
    assert means[1][0] - means[0][0] == prediction.form_mean_jump
    assert means[1][1] - means[0][1] == prediction.phase_mean_jump
    assert prediction.form_mean_jump != 0 and prediction.phase_mean_jump != 0
    assert not prediction.global_positive_mean_metric_available


def test_independent_high_precision_complete_rows_and_refresh_error(graphs, prediction):
    before, _, thol, _, after = graphs
    with mpmath.workdps(150):
        for graph, expected in (
            (before, prediction.before),
            (thol, prediction.before),
            (after, prediction.after),
        ):
            for index, node in enumerate(OLD_NODES):
                neighbors = tuple(graph[node])
                phase = _mp(_value(graph, node, ALIAS_THETA))
                form = _mp(_value(graph, node, ALIAS_EPI))
                pressure = sum(
                    mpmath.sin(_mp(_value(graph, other, ALIAS_THETA)) - phase)
                    for other in neighbors
                ) / (mpmath.pi * len(neighbors))
                gradient = sum(
                    _mp(Q(graph[node][other]["weight"]))
                    * (form - _mp(_value(graph, other, ALIAS_EPI)))
                    for other in neighbors
                )
                capacity = _mp(_value(graph, node, ALIAS_VF))
                assert _contains(expected.pressure[index], pressure)
                assert _contains(expected.form_rate[index], capacity * pressure)
                assert _contains(
                    expected.phase_rate[index],
                    capacity * gradient / (mpmath.pi * len(neighbors)),
                )
                assert abs(_mp(_value(graph, node, ALIAS_DNFR)) - pressure) <= _mp(
                    Q(1, 2**46)
                )
        assert (
            prediction.successor.form_rate[-1].lo
            == prediction.successor.form_rate[-1].hi
            == 0
        )
        assert (
            prediction.successor.phase_rate[-1].lo
            == prediction.successor.phase_rate[-1].hi
            == 0
        )


def test_associated_histories_context_and_nested_identity_are_not_inherited(graphs):
    before, raw, thol, sha_raw, after = graphs
    original_history = encode_recorded_container(
        before.nodes[ACTOR]["epi_time_history"]
    )
    for graph in graphs[1:]:
        assert (
            encode_recorded_container(graph.nodes[ACTOR]["epi_time_history"])
            == original_history
        )
        assert graph.nodes[2]["sub_nodes"] == [ACTOR]
        assert graph.nodes[ACTOR]["sub_nodes"] == [CHILD]
        assert graph.graph["hierarchy"] == {2: [ACTOR], ACTOR: [CHILD]}
        child = graph.nodes[CHILD]
        assert child["parent_node"] == ACTOR
        assert child["hierarchy_level"] == child["_bifurcation_level"] == 2
        assert child["_hierarchy_path"] == [2, ACTOR]
        assert child["glyph_history"] == []
        assert child["epi_history"] == [float(_value(graph, CHILD, ALIAS_EPI))]
        assert "epi_time_history" not in child
        assert not observe_structural_acceleration(graph, CHILD).available
        assert node_debt(graph.nodes[ACTOR]) == 0
        for node in range(20):
            assert encode_recorded_container(
                graph.nodes[node]
            ) == encode_recorded_container(before.nodes[node])
    assert (
        tuple(raw.nodes[ACTOR]["glyph_history"])
        == tuple(thol.nodes[ACTOR]["glyph_history"])
        == ("OZ", "THOL")
    )
    for graph in (sha_raw, after):
        actor = graph.nodes[ACTOR]
        assert tuple(actor["glyph_history"]) == ("OZ", "THOL", "SHA")
        assert actor["_operator_step"] == 3 and actor["latent"] is True
        assert actor["latency_start_time"] == EVENT_TIMESTAMP
        assert Q(actor["preserved_epi"]) == _value(before, ACTOR, ALIAS_EPI)
        assert actor["silence_duration"] == 0
        assert actor["was_initial_on_silence"] is False
