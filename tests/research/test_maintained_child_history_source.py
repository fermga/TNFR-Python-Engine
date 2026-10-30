"""Fresh child source admission and hostile retained-record controls."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.dynamics import integrators
from tnfr.operators import network_stage, word_execution
from tnfr.research import maintained_child_history_acquisition as producer
from tnfr.research import maintained_child_history_source as owner
from tnfr.research import maintained_sha_motion
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_history"


@pytest.fixture(scope="module", autouse=True)
def no_event_response_or_trace_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("child source admission cannot execute or replay a response")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                producer,
                (
                    "evaluate_child_history",
                    "_acquire_child_history",
                    "_advance",
                    "numerical_field",
                    "refresh_pressure",
                    "rk4_coupled_step",
                ),
            ),
            (integrators, ("rk4_coupled_step",)),
            (word_execution, ("execute_network_operator_stage",)),
            (
                network_stage,
                ("execute_dissonance_stage", "execute_self_organization_stage"),
            ),
            (
                maintained_sha_motion,
                ("certify_sha_retention", "validated_box_taylor_step"),
            ),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def admitted():
    documents, receipts = owner._read_child_history_artifacts(EVIDENCE, ROOT)
    certificate = owner.prepare_child_sample_certificate(
        ROOT / owner._JOINT_PATH, repository_root=ROOT
    )
    signature = owner._signature(certificate.source.retention.admitted.graph)
    report = owner._admit_child_history(documents, certificate)
    yield documents, certificate, receipts, report
    assert owner._signature(certificate.source.retention.admitted.graph) == signature


def _mutate_graph(documents, name, mutate):
    record = (
        documents[name]
        if name in ("event_before", "event_raw")
        else documents[name]["endpoint"]
    )
    graph = restore_recorded_graph_state(record).graph
    mutate(graph)
    updated = record_graph_state(graph)
    if name in ("event_before", "event_raw"):
        documents[name] = updated
    else:
        documents[name]["endpoint"] = updated


def test_retained_endpoint_exposes_actual_history_and_fresh_law(admitted):
    documents, certificate, receipts, result = admitted
    assert len(receipts) == 15
    assert result.certificate is certificate
    assert result.fields.nodes == tuple(range(20)) + (owner.CHILD,)
    assert len(result.fields.edges) == 53
    assert result.assessment.passed
    assert (
        result.observation.available
        and result.observation.current_endpoint_matches_state
    )
    assert (
        result.observation.source == "epi_time_history"
        and result.observation.history_length == 3
    )
    assert result.observation.value == 120.04361511376919
    assert result.threshold == Q.from_float(0.1)
    assert tuple(result.graph.nodes[owner.CHILD]["glyph_history"]) == ("OZ",)
    assert owner.node_debt(result.graph.nodes[owner.CHILD]) == 1
    assert result.sequence.names == ("dissonance", "self_organization", "silence")
    assert owner._signature(result.graph) == owner._signature(
        restore_recorded_graph_state(documents["sample_2"]["endpoint"]).graph
    )


def test_public_loader_reloads_exactly_once_and_keeps_receipts(admitted, monkeypatch):
    documents, certificate, receipts, _ = admitted
    calls = []
    monkeypatch.setattr(
        owner, "_read_child_history_artifacts", lambda *args: (documents, receipts)
    )

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return certificate

    monkeypatch.setattr(owner, "prepare_child_sample_certificate", load)
    result = owner.load_child_history_source(EVIDENCE, repository_root=ROOT)
    assert len(calls) == 1
    assert calls[0][0] == (ROOT / owner._JOINT_PATH,)
    assert result.receipts == receipts


@pytest.mark.parametrize(
    "key,value",
    (
        ("maximum_steps", True),
        ("maximum_events", 2),
        ("threshold", Q(0)),
        ("step", Q(1, 2**13)),
        ("held", ("support",)),
        ("forcing", 1),
        ("context_word", ("OZ", "SHA")),
        ("full_state_error_allowance", Q(1)),
        ("represented_point_pressure_error_allowance", Q(1)),
        ("random_seed", 18),
    ),
)
def test_changed_protocol_cannot_inherit_cached_success(admitted, key, value):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    altered["protocol"][key] = value
    with pytest.raises(ValueError, match="protocol differs"):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "key,value",
    (
        ("numerical_steps", True),
        ("outstanding_u2_debt", True),
        ("complete_word_executed", True),
        ("second_birth_executed", True),
        ("executed_prefix", ("OZ", "THOL")),
    ),
)
def test_changed_response_scope_is_not_scalar_equality(admitted, key, value):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    altered["response"][key] = value
    with pytest.raises(ValueError, match="scope or chronology"):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize("field", ("state", "pressure"))
def test_favorable_cached_assessment_cannot_cover_bad_primitives(admitted, field):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    key, coordinate = ("states", 20) if field == "state" else ("pressures", 20)
    values = list(altered["response"][key][0])
    values[coordinate] += Q(1, 1024)
    altered["response"][key] = (tuple(values),) + altered["response"][key][1:]
    assert altered["response"]["assessment"]["passed"] is True
    with pytest.raises(ValueError, match="sample certificate failed"):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "key", ("assessment", "observation", "grammar", "timing_admission")
)
def test_cached_derived_outputs_are_compared_only_after_rebuilding(admitted, key):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    if key == "assessment":
        altered["response"][key]["max_state_error_bounds"] = (Q(0),) * 3
    elif key == "observation":
        altered["response"][key]["value"] = 999.0
    elif key == "grammar":
        altered["response"][key]["violations"] = ({"rule": "forged"},)
    else:
        altered["source"][key]["conditional_storage_jump"] = Q(1)
    with pytest.raises(ValueError, match="cached"):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "name",
    (
        "source",
        "event_before",
        "event_raw",
        "event_after",
        "sample_0",
        "sample_2",
        "response",
    ),
)
def test_undeclared_configuration_changes_are_rejected_at_every_boundary(
    admitted, name
):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    _mutate_graph(altered, name, lambda graph: graph.graph.update(THOL_MIN_EPI=0.0))
    with pytest.raises(ValueError):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "change",
    (
        "capacity",
        "support",
        "hierarchy",
        "identity",
        "operator_clock",
        "debt",
        "history",
    ),
)
def test_hidden_endpoint_changes_do_not_inherit_the_retained_source(admitted, change):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)

    def mutate(graph):
        if change == "capacity":
            owner.set_attr(graph.nodes[owner.CHILD], owner.ALIAS_VF, 122.0)
        elif change == "support":
            graph.add_edge(0, owner.CHILD, weight=1.0)
        elif change == "hierarchy":
            graph.graph["hierarchy"] = {2: []}
        elif change == "identity":
            graph.nodes[owner.CHILD]["parent_node"] = 3
        elif change == "operator_clock":
            graph.nodes[owner.CHILD]["_operator_step"] = 2
        elif change == "debt":
            graph.nodes[owner.CHILD]["_grammar_u2_debt"] = True
        else:
            graph.nodes[owner.CHILD]["glyph_history"] = ["OZ", "THOL"]

    _mutate_graph(altered, "sample_2", mutate)
    altered["response"]["endpoint"] = deepcopy(altered["sample_2"]["endpoint"])
    with pytest.raises(ValueError):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "change",
    ("raw_pressure", "neighbor_pressure", "precondition", "propagation", "phase"),
)
def test_oz_raw_context_and_pressure_footprint_are_rebuilt(admitted, change):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)

    def mutate(graph):
        if change == "raw_pressure":
            owner.set_dnfr(graph, owner.CHILD, 0.2)
        elif change == "neighbor_pressure":
            owner.set_dnfr(graph, 2, 0.2)
        elif change == "precondition":
            graph.nodes[owner.CHILD]["_oz_precondition_context"]["degree"] = True
        elif change == "propagation":
            graph.graph["_oz_propagation_events"][-1]["affected_nodes"] = [2]
        else:
            owner.set_attr(graph.nodes[owner.CHILD], owner.ALIAS_THETA, 0.0)

    _mutate_graph(altered, "event_raw", mutate)
    with pytest.raises(ValueError):
        owner._admit_child_history(altered, certificate)


@pytest.mark.parametrize(
    "change", ("stale_time", "own_value", "borrowed", "other_node")
)
def test_all_node_sample_footprints_and_child_owned_live_history_are_checked(
    admitted, change
):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)

    def mutate(graph):
        if change == "stale_time":
            graph.graph["_t"] -= 1.0
        elif change == "own_value":
            graph.nodes[owner.CHILD]["epi_time_history"][-1] = (graph.graph["_t"], 1.0)
        elif change == "borrowed":
            graph.nodes[owner.CHILD]["epi_time_history"] = deepcopy(
                graph.nodes[2]["epi_time_history"]
            )
        else:
            graph.nodes[0]["epi_time_history"] = []

    _mutate_graph(altered, "sample_2", mutate)
    altered["response"]["endpoint"] = deepcopy(altered["sample_2"]["endpoint"])
    with pytest.raises(ValueError):
        owner._admit_child_history(altered, certificate)


def test_source_receipt_and_sample_associations_cannot_be_relabelled(admitted):
    documents, certificate, _, _ = admitted
    altered = deepcopy(documents)
    altered["response"]["source_receipts"] = ()
    with pytest.raises(ValueError, match="source associations"):
        owner._admit_child_history(altered, certificate)
    altered = deepcopy(documents)
    altered["sample_1"]["index"] = True
    with pytest.raises(ValueError, match="sample index"):
        owner._admit_child_history(altered, certificate)


def test_replaceable_certificate_boxes_do_not_supply_bounds(admitted):
    documents, certificate, _, result = admitted
    altered = replace(certificate, boxes=(), sample_times=certificate.sample_times)
    rebuilt = owner._admit_child_history(documents, altered)
    assert rebuilt.assessment == result.assessment


@pytest.mark.parametrize(
    "artifact", ("protocol.json", "response.json", "source.zip", "steps.zip")
)
def test_artifact_byte_alteration_rejects_before_scientific_admission(
    monkeypatch, artifact
):
    original = owner.read_bytes_bounded

    def read(path, **kwargs):
        data = original(path, **kwargs)
        return data + b" " if path.name == artifact else data

    monkeypatch.setattr(owner, "read_bytes_bounded", read)
    with pytest.raises(ValueError, match="receipt differs"):
        owner._read_child_history_artifacts(EVIDENCE, ROOT)


def test_duplicate_json_keys_are_rejected_before_receipts(monkeypatch):
    original = owner.read_bytes_bounded

    def read(path, **kwargs):
        return (
            b'{"schema":"a","schema":"b"}'
            if path.name == "protocol.json"
            else original(path, **kwargs)
        )

    monkeypatch.setattr(owner, "read_bytes_bounded", read)
    with pytest.raises(ValueError):
        owner._read_child_history_artifacts(EVIDENCE, ROOT)
