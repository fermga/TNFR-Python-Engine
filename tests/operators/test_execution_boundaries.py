"""Execution must validate before mutation and honor the selected operator."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import (
    Dissonance, Emission, Reception, SelfOrganization, Silence, Transition,
)
from tnfr.operators.grammar_debt import PRIOR_COHERENCE_KEY, U2_DEBT_KEY
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph():
    graph = nx.Graph()
    graph.add_node(0, **{
        ALIAS_EPI[0]: 0.5, ALIAS_VF[0]: 1.0,
        ALIAS_DNFR[0]: 0.2, ALIAS_THETA[0]: 0.0,
        "glyph_history": ["AL"], U2_DEBT_KEY: 0, PRIOR_COHERENCE_KEY: False,
    })
    return graph


@pytest.mark.parametrize("window", [-1, 1.5, True])
@pytest.mark.parametrize("use_operator", [False, True])
def test_invalid_window_rejected_before_state_history_metadata_or_cache_changes(window, use_operator):
    graph = _graph()
    before_node = deepcopy(graph.nodes[0])
    before_graph = deepcopy(graph.graph)
    with pytest.raises((TypeError, ValueError)):
        if use_operator:
            Emission()(graph, 0, window=window)
        else:
            apply_glyph(graph, 0, "AL", window=window)
    assert graph.nodes[0] == before_node
    assert graph.graph == before_graph


@pytest.mark.parametrize("window", [1.5, True])
def test_invalid_default_window_is_not_silently_coerced(window):
    graph = _graph()
    graph.graph["GLYPH_HYSTERESIS_WINDOW"] = window
    before = deepcopy(graph.nodes[0])
    with pytest.raises(TypeError, match="window"):
        apply_glyph(graph, 0, "AL")
    assert graph.nodes[0] == before
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize("operator", [Emission(), Reception(), Silence(), Transition()])
def test_failed_precondition_keeps_latency_metadata_and_history(operator):
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.nodes[0].update(latent=True, preserved_epi=0.5)
    if isinstance(operator, (Emission, Reception)):
        graph.nodes[0][ALIAS_EPI[0]] = 1.0
    else:
        graph.nodes[0][ALIAS_VF[0]] = 0.0
    before_node = deepcopy(graph.nodes[0])
    before_graph = deepcopy(graph.graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        operator(graph, 0)
    assert graph.nodes[0] == before_node
    assert graph.graph == before_graph


def test_self_organization_fallback_cannot_create_unrecorded_sub_structure():
    graph = _graph()
    graph.nodes[0]["epi_history"] = [0.0, 0.0, 1.0]
    graph.graph["THOL_METABOLIC_ENABLED"] = False
    SelfOrganization()(graph, 0, collect_metrics=True)
    assert list(graph) == [0]
    assert not graph.nodes[0].get("sub_epis")
    assert list(graph.nodes[0]["glyph_history"])[-1] == "IL"
    assert graph.nodes[0][PRIOR_COHERENCE_KEY] is True
    assert graph.graph["operator_metrics"][-1]["operator"] == "Coherence"


def test_contract_audit_cannot_certify_a_fallback_as_the_requested_operator(monkeypatch):
    from tnfr.operators.definitions import Coherence
    from tnfr.physics.integrity import audit_operator_contracts

    def execute_fallback(self, graph, node, **kwargs):
        Coherence()(graph, node, **kwargs)

    monkeypatch.setattr(Dissonance, "_execute", execute_fallback)
    audit = audit_operator_contracts(n_nodes=6)
    result = next(item for item in audit.results if item.glyph == "OZ")
    assert not result.satisfied
    assert "executed" in result.detail


def test_dissonance_fallback_does_not_propagate_stabilization_as_dissonance():
    graph = _graph()
    graph.add_node(1, **{ALIAS_DNFR[0]: 0.1, ALIAS_EPI[0]: 0.5,
                        ALIAS_THETA[0]: 0.0, ALIAS_VF[0]: 1.0})
    graph.add_edge(0, 1)
    graph.nodes[0]["glyph_history"] = ["VAL", "VAL"]
    graph.nodes[0][U2_DEBT_KEY] = 2
    Dissonance()(graph, 0)
    assert list(graph.nodes[0]["glyph_history"])[-1] == "IL"
    assert graph.nodes[0][U2_DEBT_KEY] == 1
    assert graph.nodes[1][ALIAS_DNFR[0]] == 0.1
    assert "_oz_propagation_events" not in graph.graph


def test_silence_fallback_does_not_mark_activated_node_as_latent():
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.0
    graph.nodes[0]["glyph_history"] = []
    Silence()(graph, 0)
    assert list(graph.nodes[0]["glyph_history"])[-1] == "NAV"
    assert not graph.nodes[0].get("latent", False)
    assert graph.graph["_nav_transitions"][-1]["node"] == 0


def test_transition_fallback_does_not_perform_unrecorded_regime_shift():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["VAL", "VAL", "VAL"]
    graph.nodes[0][U2_DEBT_KEY] = 3
    Transition()(graph, 0)
    assert list(graph.nodes[0]["glyph_history"])[-1] == "IL"
    assert "_nav_transitions" not in graph.graph
    assert "_regime_before" not in graph.nodes[0]


@pytest.mark.parametrize("name", ["coherence", "Coherence", "il", "Glyph.IL"])
def test_low_level_dispatch_uses_the_shared_canonical_name_mapping(name):
    graph = _graph()
    apply_glyph(graph, 0, name)
    assert list(graph.nodes[0]["glyph_history"])[-1] == "IL"


def test_accepted_operator_runs_grammar_selection_once(monkeypatch):
    from tnfr.operators.definitions import Coherence
    from tnfr.operators import grammar_application

    original = grammar_application.enforce_canonical_grammar
    calls = []

    def record(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(grammar_application, "enforce_canonical_grammar", record)
    Coherence()(_graph(), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("node_collection", [tuple, frozenset, iter])
def test_grammar_application_accepts_general_node_iterables(node_collection):
    from tnfr.operators.grammar_application import apply_glyph_with_grammar

    graph = _graph()
    graph.add_node(1, **deepcopy(graph.nodes[0]))
    apply_glyph_with_grammar(graph, node_collection([0, 1]), "IL")
    for node in graph:
        assert list(graph.nodes[node]["glyph_history"])[-1] == "IL"


def test_missing_batch_target_is_rejected_before_applying_to_valid_targets():
    from tnfr.operators.grammar_application import apply_glyph_with_grammar

    graph = _graph()
    before = deepcopy(graph.nodes[0])
    with pytest.raises(KeyError):
        apply_glyph_with_grammar(graph, [0, 99], "IL")
    assert graph.nodes[0] == before
    assert graph.graph == {}


@pytest.mark.parametrize("use_operator", [False, True])
def test_one_shot_history_rejected_without_consumption_or_mutation(use_operator):
    graph = _graph()
    history = iter(["IL", "VAL"])
    graph.nodes[0]["glyph_history"] = history
    before = dict(graph.nodes[0])
    with pytest.raises(ValueError, match="replayable"):
        if use_operator:
            Emission()(graph, 0)
        else:
            apply_glyph(graph, 0, "AL")
    assert graph.nodes[0] == before
    assert graph.graph == {}
    assert list(history) == ["IL", "VAL"]


def test_unknown_glyph_does_not_create_runtime_or_history_caches():
    graph = _graph()
    before = deepcopy(graph.nodes[0])
    with pytest.raises(ValueError, match="unknown glyph"):
        apply_glyph(graph, 0, "invalid_operator")
    assert graph.nodes[0] == before
    assert graph.graph == {}


def test_existing_tuple_node_is_not_expanded_into_separate_targets():
    from tnfr.operators.grammar_application import apply_glyph_with_grammar

    graph = nx.relabel_nodes(_graph(), {0: (0, 1)})
    apply_glyph_with_grammar(graph, (0, 1), "IL")
    assert list(graph.nodes[(0, 1)]["glyph_history"])[-1] == "IL"


def test_object_dispatch_validates_window_before_changing_state():
    from tnfr.node import NodeNX
    from tnfr.operators import apply_glyph_obj

    graph = _graph()
    node = NodeNX(graph, 0)
    before = deepcopy(graph.nodes[0])
    with pytest.raises(ValueError, match="window"):
        apply_glyph_obj(node, "AL", window=-1)
    assert graph.nodes[0] == before
