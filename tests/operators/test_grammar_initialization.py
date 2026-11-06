"""U1a distinguishes finite signed form from the zero scalar coordinate."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import Coherence, Emission, Silence
from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_dynamics import (
    enforce_grammar_on_glyph,
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_memoization import (
    create_sequence_signature,
    validate_sequence_optimized,
)
from tnfr.types import serialize_bepi


def _embedding(value):
    return BEPIElement((value, value), (value, value), (0.0, 1.0))


def _graph(epi, history=()):
    graph = nx.Graph()
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: epi,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": list(history),
        },
    )
    return graph


@pytest.mark.parametrize(
    "epi",
    (
        -0.5,
        0.5,
        -5e-324,
        5e-324,
        _embedding(-0.5),
        serialize_bepi(_embedding(-0.5)),
    ),
)
def test_nonzero_signed_form_has_the_same_initialization_status_in_all_paths(epi):
    sequence = [Coherence(), Silence()]
    assert GrammarValidator().validate(sequence, epi_initial=epi)[0]
    assert validate_sequence_optimized(sequence, epi_initial=epi)[0]
    assert not create_sequence_signature(sequence, epi_initial=epi).epi_zero_start
    assert validate_candidate(_graph(epi), 0, "IL").allowed


@pytest.mark.parametrize(
    "epi", (0.0, -0.0, _embedding(0.0), serialize_bepi(_embedding(0.0)))
)
def test_zero_still_requires_a_generator_without_prior_history(epi):
    sequence = [Coherence(), Silence()]
    assert not GrammarValidator().validate(sequence, epi_initial=epi)[0]
    assert not validate_sequence_optimized(sequence, epi_initial=epi)[0]
    assert create_sequence_signature(sequence, epi_initial=epi).epi_zero_start
    result = validate_candidate(_graph(epi), 0, "IL")
    assert not result.allowed
    assert any(violation.rule == "U1a" for violation in result.violations)
    assert validate_candidate(_graph(epi), 0, "AL").allowed


def test_existing_history_remains_a_separate_initialization_witness():
    assert validate_candidate(_graph(0.0, history=("AL",)), 0, "IL").allowed


@pytest.mark.parametrize(
    "epi",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        None,
        "0.5",
        1.0j,
        BEPIElement((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    ),
)
def test_invalid_provided_epi_cannot_become_initialization_permission(epi):
    sequence = [Emission(), Silence()]
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        GrammarValidator().validate(sequence, epi_initial=epi)
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        validate_sequence_optimized(sequence, epi_initial=epi)
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        create_sequence_signature(sequence, epi_initial=epi)
    # Prior history does not make malformed live coordinates admissible.
    for history in ((), ("AL",)):
        graph = _graph(epi, history=history)
        with pytest.raises(TNFRValueError, match="finite"):
            validate_candidate(graph, 0, "IL")
        with pytest.raises(TNFRValueError, match="finite"):
            enforce_grammar_on_glyph(graph, 0, "IL")


def test_first_present_epi_alias_cannot_be_hidden_by_a_valid_later_alias():
    graph = _graph("invalid")
    graph.nodes[0][ALIAS_EPI[1]] = 0.5
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        validate_candidate(graph, 0, "IL")


def test_absent_epi_retains_the_provisional_selection_compatibility_default():
    graph = _graph(0.0)
    del graph.nodes[0][ALIAS_EPI[0]]
    assert validate_candidate(graph, 0, "IL").allowed
    assert not any(alias in graph.nodes[0] for alias in ALIAS_EPI)


def test_shadow_validation_restores_state_after_invalid_coordinate_failure():
    graph = _graph("invalid", history=("AL",))
    before = deepcopy(graph.nodes[0])
    history = graph.nodes[0]["glyph_history"]
    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        validate_sequence_incremental(graph, 0, ("IL", "SHA"))
    assert graph.nodes[0] == before
    assert graph.nodes[0]["glyph_history"] is history


def test_validated_word_executes_coherence_on_negative_form_without_a_generator():
    graph = _graph(-0.5)
    operators = [Coherence(), Silence()]
    word = ValidatedSequence(operators, context={"initial_epi_nonzero": True})
    operators[0](graph, 0, sequence_context=word.step(0))
    assert graph.nodes[0][ALIAS_EPI[0]] == -0.5
    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]
    assert abs(graph.nodes[0][ALIAS_DNFR[0]]) < 0.2
