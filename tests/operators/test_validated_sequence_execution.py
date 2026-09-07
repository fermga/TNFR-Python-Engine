"""Validated future handlers must not bypass causal live grammar checks."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import (
    Coherence, Coupling, Dissonance, Emission, Expansion, Mutation, Recursivity, Silence,
)
from tnfr.operators.grammar_debt import PRIOR_COHERENCE_KEY, U2_DEBT_KEY


def _graph():
    graph = nx.Graph()
    graph.add_node(0, **{
        ALIAS_EPI[0]: 0.5, ALIAS_VF[0]: 1.0,
        ALIAS_DNFR[0]: 0.2, ALIAS_THETA[0]: 0.0,
        "epi_history": [0.0, 0.2],
        "glyph_history": [], U2_DEBT_KEY: 0, PRIOR_COHERENCE_KEY: False,
    })
    return graph


def _run(graph, operators, driver):
    if driver == "sdk":
        from tnfr.sdk.simple import _run_network_sequence
        _run_network_sequence(graph, [operator.name for operator in operators])
    elif driver == "fluent":
        from tnfr.sdk.fluent import TNFRNetwork
        network = TNFRNetwork()
        network._graph = graph
        network.apply_sequence([operator.name for operator in operators])
    else:
        from tnfr.structural import run_sequence
        run_sequence(graph, 0, operators)


@pytest.mark.parametrize("driver", ["sdk", "fluent", "structural"])
def test_validated_future_handler_preserves_the_requested_word(driver):
    graph = _graph()
    _run(graph, [Emission(), Dissonance(), Coherence(), Silence()], driver)
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "OZ", "IL", "SHA"]


@pytest.mark.parametrize("driver", ["sdk", "fluent", "structural"])
def test_invalid_debt_prefix_is_rejected_before_sequence_mutation(driver):
    graph = _graph()
    before = deepcopy(graph.nodes[0])
    operators = [Emission(), Expansion(), Expansion(), Expansion(), Coherence(), Silence()]
    with pytest.raises(ValueError, match="debt"):
        _run(graph, operators, driver)
    assert graph.nodes[0] == before
    assert graph.graph == {}


@pytest.mark.parametrize("driver", ["sdk", "fluent", "structural"])
def test_future_handler_does_not_allow_new_live_debt_above_capacity(driver):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["VAL", "VAL"]
    graph.nodes[0][U2_DEBT_KEY] = 2
    with pytest.raises(RuntimeError, match="debt"):
        _run(graph, [Emission(), Dissonance(), Coherence(), Silence()], driver)
    # The accepted prefix remains committed; the blocked step is not recorded.
    assert list(graph.nodes[0]["glyph_history"]) == ["VAL", "VAL", "AL"]
    assert graph.nodes[0][U2_DEBT_KEY] == 2


def test_standalone_dissonance_still_requires_a_past_handler():
    graph = _graph()
    Dissonance()(graph, 0)
    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]


def test_future_context_does_not_supply_missing_prior_coherence():
    from tnfr.operators.grammar_execution import ValidatedSequence

    operators = [Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
    context = ValidatedSequence(operators).step(3)
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["OZ"]
    graph.nodes[0][U2_DEBT_KEY] = 1
    before = deepcopy(graph.nodes[0])
    with pytest.raises(RuntimeError, match="prior IL"):
        Mutation()(graph, 0, sequence_context=context)
    assert graph.nodes[0] == before


def test_step_context_rejects_a_different_requested_operator():
    from tnfr.operators.grammar_execution import ValidatedSequence

    context = ValidatedSequence([Emission(), Dissonance(), Coherence(), Silence()]).step(1)
    graph = _graph()
    before = deepcopy(graph.nodes[0])
    with pytest.raises(ValueError, match="context"):
        Expansion()(graph, 0, sequence_context=context)
    assert graph.nodes[0] == before


def test_context_does_not_replace_the_live_destabilizer_window():
    from tnfr.operators.grammar_execution import ValidatedSequence

    operators = [Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["IL", "EN", "EN", "EN", "EN"]
    graph.nodes[0][PRIOR_COHERENCE_KEY] = True
    before = deepcopy(graph.nodes[0])
    with pytest.raises(RuntimeError, match="recent destabilizer"):
        Mutation()(graph, 0, sequence_context=ValidatedSequence(operators).step(3))
    assert graph.nodes[0] == before


def test_context_does_not_bypass_the_live_phase_gate():
    import math

    from tnfr.operators.grammar_execution import ValidatedSequence
    from tnfr.operators.preconditions import OperatorPreconditionError

    context = ValidatedSequence([Emission(), Coupling(), Coherence(), Silence()]).step(1)
    graph = _graph()
    graph.add_node(1, **deepcopy(graph.nodes[0]))
    graph.nodes[1][ALIAS_THETA[0]] = math.pi
    graph.add_edge(0, 1)
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(OperatorPreconditionError):
        Coupling()(graph, 0, sequence_context=context)
    assert dict(graph.nodes(data=True)) == before


def test_validated_word_checks_actual_recursivity_depth():
    from tnfr.operators.grammar_execution import ValidatedSequence

    with pytest.raises(ValueError, match="U5"):
        ValidatedSequence([Emission(), Recursivity(depth=2), Silence()])


def test_failed_handler_cannot_leave_future_permission_in_the_graph(monkeypatch):
    graph = _graph()

    def fail_handler(self, graph, node, **kwargs):
        raise RuntimeError("handler execution failed")

    with monkeypatch.context() as patch:
        patch.setattr(Coherence, "_execute", fail_handler)
        with pytest.raises(RuntimeError, match="handler execution failed"):
            _run(graph, [Emission(), Dissonance(), Coherence(), Silence()], "sdk")
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "OZ"]
    # Outside the failed word no future handler is available: OZ falls back.
    Dissonance()(graph, 0)
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "OZ", "IL"]


@pytest.mark.parametrize("driver", ["sdk", "fluent"])
def test_failed_legacy_validation_result_is_not_ignored(driver):
    graph = _graph()
    before = deepcopy(graph.nodes[0])
    # U2 permits debt two, but the retained compatibility layer rejects VAL->VAL.
    with pytest.raises(ValueError, match="Invalid sequence"):
        _run(graph, [Emission(), Expansion(), Expansion(), Coherence(), Silence()], driver)
    assert graph.nodes[0] == before
    assert graph.graph == {}


def test_executor_without_word_validation_retains_incremental_fallback():
    from tnfr.sdk.simple import _run_network_sequence

    graph = _graph()
    with pytest.warns(UserWarning, match="Anti-pattern"):
        _run_network_sequence(
            graph, ["emission", "dissonance", "coherence", "silence"], validate=False,
        )
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "IL", "IL", "SHA"]
