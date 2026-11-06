"""Activation provenance and causal ordering of strict OZ admission."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EMISSION_TIMESTAMP, ALIAS_EPI
from tnfr.errors import TNFRValueError
from tnfr.operators.al_sha_stage_proposals import propose_emission_stage
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.dissonance import validate_dissonance_strict


def _graph():
    graph = nx.path_graph(2)
    graph.graph.update(RANDOM_SEED=17, GLYPH_FACTORS={"AL_boost": 0.125})
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            DeltaNFR=0.125,
            glyph_history=["AL", "IL"],
        )
    return graph


def _emit(graph, path):
    if path == "proposal":
        return propose_emission_stage(
            graph, 0, {"AL_boost": 0.125}, timestamp="2030-01-01T00:00:00+00:00"
        )
    if path == "public":
        return Emission()(graph, 0)
    return execute_pointwise_stage(graph, Emission(), (0,))


@pytest.mark.parametrize("path", ("public", "atomic"))
def test_false_activation_flag_initializes_once_without_losing_other_origin(path):
    graph = _graph()
    graph.nodes[0].update(_emission_activated=False, structural_origin="other-event")

    _emit(graph, path)

    assert float(get_attr(graph.nodes[0], ALIAS_EPI)) == 0.375
    assert graph.nodes[0]["_emission_activated"] is True
    assert graph.nodes[0]["_structural_lineage"]["activation_count"] == 1
    assert graph.nodes[0]["structural_origin"] == "other-event"
    origin = graph.nodes[0]["_emission_origin"]

    _emit(graph, path)

    assert graph.nodes[0]["_emission_origin"] == origin
    assert graph.nodes[0][ALIAS_EMISSION_TIMESTAMP[0]] == origin
    assert graph.nodes[0]["_structural_lineage"]["activation_count"] == 2


@pytest.mark.parametrize("path", ("proposal", "public", "atomic"))
@pytest.mark.parametrize("flag", (0, 1, "false", "true", None, []))
def test_invalid_activation_flag_rejects_before_form_or_metadata_writes(path, flag):
    graph = _graph()
    graph.nodes[0]["_emission_activated"] = flag
    before = deepcopy(graph)

    with pytest.raises(OperatorPreconditionError, match="_emission_activated"):
        _emit(graph, path)

    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("path", ("proposal", "public", "atomic"))
@pytest.mark.parametrize("present_flag", (False, True))
@pytest.mark.parametrize(
    "key", ("_emission_origin", "_structural_lineage", *ALIAS_EMISSION_TIMESTAMP)
)
def test_inactive_flag_with_retained_emission_records_never_erases_history(
    path, present_flag, key
):
    graph = _graph()
    if present_flag:
        graph.nodes[0]["_emission_activated"] = False
    graph.nodes[0][key] = {"retained": "provenance"}
    before = deepcopy(graph)

    with pytest.raises(TNFRValueError, match="emission provenance"):
        _emit(graph, path)

    assert nx.utils.graphs_equal(graph, before)


def _dissonance(graph, path):
    if path == "validator":
        return validate_dissonance_strict(graph, 0)
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    context = ValidatedSequence(
        [Emission(), Dissonance(), Coherence(), Silence()]
    ).step(1)
    if path == "public":
        return Dissonance()(graph, 0, sequence_context=context, propagate=False)
    return execute_dissonance_stage(
        graph, Dissonance(), (0,), sequence_context=context, propagate=False
    )


@pytest.mark.parametrize("path", ("validator", "public", "atomic"))
@pytest.mark.parametrize("resolver", ("IL", "THOL", "NUL"))
def test_resolver_before_oz_cannot_resolve_that_later_perturbation(path, resolver):
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.nodes[0]["glyph_history"] = ["AL", resolver, "OZ"]
    before = deepcopy(graph)

    with pytest.raises((TNFRValueError, OperatorPreconditionError), match="overload"):
        _dissonance(graph, path)

    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("resolver", ("IL", "THOL", "NUL"))
def test_later_resolver_discharges_the_selected_oz_policy(resolver):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["AL", "OZ", resolver, "EN"]

    validate_dissonance_strict(graph, 0)

    assert graph.nodes[0]["_oz_precondition_context"]["validation_passed"] is True


def test_a_new_oz_after_resolution_creates_an_unresolved_suffix():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["AL", "OZ", "IL", "OZ", "EN"]

    with pytest.raises(TNFRValueError, match="Found 1 OZ"):
        validate_dissonance_strict(graph, 0)


def test_overload_window_remains_bounded_and_does_not_reinterpret_u2_debt():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["OZ", "EN", "EN", "EN", "EN", "EN"]

    validate_dissonance_strict(graph, 0)

    assert graph.nodes[0]["_oz_precondition_context"]["validation_passed"] is True
