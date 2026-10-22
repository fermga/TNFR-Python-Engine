"""Primitive retained-source admission without THOL/UM or producer replay."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tnfr.alias import set_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_parent_um_source as owner
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"


@pytest.fixture(scope="module", autouse=True)
def no_response():
    def forbidden(*args, **kwargs):
        pytest.fail("admission must not execute a producer, proposal or event")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (SelfOrganization, "_prepare_execution"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    return owner.load_thol_successor(EVIDENCE, repository_root=ROOT), decode_exact_tree(
        json_loads((EVIDENCE / "response.json").read_bytes())
    )


def test_associated_source_rebuilds_independent_history_and_live_um_context(retained):
    admitted, _ = retained
    assert tuple(admitted.graph) == tuple(range(20)) + ("2_sub_0",)
    assert admitted.fields.isolate_indices == (20,)
    assert admitted.prebirth.certificate.trigger_lower_bound > 0.1
    assert tuple(admitted.graph.nodes[2]["glyph_history"]) == ("VAL", "THOL")
    assert admitted.graph.nodes[2]["_grammar_u2_debt"] == 0
    assert "Si" not in admitted.graph.nodes["2_sub_0"]
    assert len(admitted.receipts) == 9


def test_cached_verdicts_fields_and_prediction_are_not_mathematical_premises(retained):
    admitted, response = retained
    changed = deepcopy(response)
    changed.update(
        checks={"fabricated": False},
        all_checks_passed=False,
        successor_fields={"not": "state"},
        prediction={"not": "premises"},
        edge_storage_jump=999,
    )
    rebuilt = owner.admit_thol_successor(changed, prebirth=admitted.prebirth)
    assert _signature(rebuilt.graph) == _signature(admitted.graph)
    assert rebuilt.fields == admitted.fields


@pytest.mark.parametrize(
    "case",
    (
        "old_form",
        "child_form",
        "child_capacity",
        "child_parent",
        "level_boolean",
        "history",
        "clock",
        "old_sense",
        "invented_child_sense",
        "debt",
        "negative_debt",
        "seed",
        "strict_policy",
        "operator_step",
        "fresh_pressure",
    ),
)
def test_changed_authoritative_endpoint_rejects(retained, case):
    admitted, response = retained
    changed = deepcopy(response)
    graph = restore_recorded_graph_state(changed["endpoint"]).graph
    if case == "old_form":
        set_attr(graph.nodes[4], ALIAS_EPI, 0.8)
    elif case == "child_form":
        set_attr(graph.nodes["2_sub_0"], ALIAS_EPI, 0.8)
    elif case == "child_capacity":
        set_attr(graph.nodes["2_sub_0"], ALIAS_VF, 1.0)
    elif case == "child_parent":
        graph.nodes["2_sub_0"]["parent_node"] = 1
    elif case == "level_boolean":
        graph.nodes["2_sub_0"]["hierarchy_level"] = True
    elif case == "history":
        graph.nodes[2]["epi_time_history"] = ()
    elif case == "clock":
        graph.graph["_t"] = 8.0
    elif case == "old_sense":
        graph.nodes[0]["Si"] = 0.0
    elif case == "invented_child_sense":
        graph.nodes["2_sub_0"]["Si"] = 0.5
    elif case == "debt":
        graph.nodes[2]["_grammar_u2_debt"] = 1
    elif case == "negative_debt":
        graph.nodes[2]["_grammar_u2_debt"] = -1
    elif case == "seed":
        graph.graph["RANDOM_SEED"] = 18
    elif case == "strict_policy":
        graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    elif case == "operator_step":
        graph.nodes[2]["_operator_step"] = 7
    else:
        set_dnfr(graph, 2, 0.2)
    changed["endpoint"] = record_graph_state(graph)
    with pytest.raises(ValueError):
        owner.admit_thol_successor(changed, prebirth=admitted.prebirth)


def test_raw_pressure_is_separate_from_constitutive_refresh(retained):
    admitted, response = retained
    changed = deepcopy(response)
    raw = restore_recorded_graph_state(changed["raw_after"]).graph
    set_dnfr(raw, 2, 0.3)
    changed["raw_after"] = record_graph_state(raw)
    with pytest.raises(ValueError, match="raw pressure"):
        owner.admit_thol_successor(changed, prebirth=admitted.prebirth)


def test_prebirth_cached_certificate_is_rebuilt(retained):
    admitted, response = retained
    forged = replace(
        admitted.prebirth,
        certificate=replace(admitted.prebirth.certificate, trigger_lower_bound=999),
    )
    fresh = owner.admit_thol_successor(response, prebirth=forged)
    assert fresh.prebirth.certificate == admitted.prebirth.certificate


def test_typed_prebirth_context_association_cannot_be_replaced(retained):
    admitted, response = retained
    changed = deepcopy(response)
    before = restore_recorded_graph_state(changed["before"]).graph
    before.graph["RANDOM_SEED"] = 18
    changed["before"] = record_graph_state(before)
    with pytest.raises(ValueError, match="birth source"):
        owner.admit_thol_successor(changed, prebirth=admitted.prebirth)
