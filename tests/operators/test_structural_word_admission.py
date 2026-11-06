"""Structural runner and compatibility reports share actual word admission."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_debt import U2_DEBT_KEY
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.structural import run_sequence
from tnfr.types import serialize_bepi
from tnfr.validation import SequenceSemanticValidator, validation_config


def graph_with_form(epi):
    graph = nx.Graph()
    graph.add_node(0, EPI=epi, nu_f=1.0, theta=0.0, delta_nfr=0.2, glyph_history=[])
    return graph


@pytest.fixture(autouse=True)
def no_optional_invariant_diagnostics(monkeypatch):
    monkeypatch.setattr(validation_config, "validate_invariants", False)


@pytest.mark.parametrize(
    "value", [-0.5, BEPIElement((-0.5, -0.5), (-0.5, -0.5), (0.0, 1.0))]
)
def test_actual_signed_form_admits_initialized_word(value):
    graph = graph_with_form(value)
    run_sequence(graph, 0, [Coherence(), Silence()])
    assert list(graph.nodes[0]["glyph_history"]) == ["IL", "SHA"]
    assert graph._last_operator_applied == "silence"


@pytest.mark.parametrize(
    "value",
    [
        True,
        "0.5",
        None,
        float("inf"),
        1j,
        Fraction(1, 10**400),
        BEPIElement((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    ],
)
def test_invalid_authoritative_form_rejects_before_any_word_effect(value):
    graph = graph_with_form(value)
    graph.nodes[0][ALIAS_EPI[1]] = 0.5
    before = deepcopy(graph.nodes[0])
    with pytest.raises(ValueError):
        run_sequence(
            graph, 0, [Emission(), Silence()], context={"initial_epi_nonzero": True}
        )
    assert graph.nodes[0]["EPI"] is value
    assert {key: item for key, item in graph.nodes[0].items() if key != "EPI"} == {
        key: item for key, item in before.items() if key != "EPI"
    }
    assert graph.graph == {}
    assert not hasattr(graph, "_last_operator_applied")


def test_zero_embedding_cannot_borrow_initialization_from_caller_flag():
    graph = graph_with_form(serialize_bepi(0.0))
    before = deepcopy(graph.nodes[0])
    with pytest.raises(ValueError, match="U1a"):
        run_sequence(
            graph, 0, [Coherence(), Silence()], context={"initial_epi_nonzero": True}
        )
    assert graph.nodes[0] == before


def test_actual_word_admission_occurs_once_and_obsolete_semantics_do_not_run(
    monkeypatch,
):
    from tnfr.operators.grammar_core import GrammarValidator

    calls = []
    original = GrammarValidator.validate_checks

    def checked(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GrammarValidator, "validate_checks", checked)
    graph = graph_with_form(0.5)
    run_sequence(graph, 0, [Emission(), Silence()])
    assert calls == [1]
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "SHA"]


def test_blocked_candidate_does_not_replace_last_completed_operator():
    graph = graph_with_form(0.5)
    graph.nodes[0][U2_DEBT_KEY] = 2
    graph.nodes[0]["glyph_history"] = ["VAL", "VAL"]
    with pytest.raises(RuntimeError, match="debt"):
        run_sequence(graph, 0, [Emission(), Dissonance(), Coherence(), Silence()])
    assert graph._last_operator_applied == "emission"
    assert list(graph.nodes[0]["glyph_history"]) == ["VAL", "VAL", "AL"]


@pytest.mark.parametrize("context", [True, {"compatibility_profile": "unknown"}])
def test_even_identity_execution_admits_supplied_context(context):
    with pytest.raises((TypeError, ValueError)):
        run_sequence(nx.Graph(), 0, [], context=context)


@pytest.mark.parametrize("profile", ["core", "legacy"])
@pytest.mark.parametrize(
    "word",
    [
        ["emission", "silence"],
        ["emission", "coherence", "silence"],
        ["mutation", "silence"],
    ],
)
def test_semantic_compatibility_report_reuses_shared_verdict(word, profile):
    context = {"compatibility_profile": profile}
    outcome = validate_sequence(word, context=context)
    report = SequenceSemanticValidator().validate_semantic_sequence(
        word, context=context
    )
    assert bool(report) is not outcome.passed
    if report:
        assert outcome.summary["message"] in report[0].description
