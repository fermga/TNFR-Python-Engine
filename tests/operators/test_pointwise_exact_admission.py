"""Exact pointwise requests reject fallback without requesting a proof model."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Expansion
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    TWO_PHASE_JACOBI,
    execute_pointwise_stage,
)


def _graph(*, blocked):
    graph = nx.Graph(
        GRAMMAR_REJECTION_MODE="fallback",
        EDGE_AWARE_ENABLED=False,
        GLYPH_FACTORS={"VAL_scale": 1.125},
    )
    graph.add_node(
        0,
        EPI=0.25,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.05,
        glyph_history=["IL", "OZ", "VAL", "VAL"] if blocked else ["AL", "IL"],
        _grammar_u2_debt=3 if blocked else 0,
        _grammar_prior_coherence=True,
    )
    return graph


def _state(graph, *, without_wrapper_cache=False):
    metadata = dict(graph.graph)
    if without_wrapper_cache:
        # Distinct reference graphs own different NodeNX wrapper identities;
        # their authoritative node data is compared separately below.
        metadata.pop("_node_cache", None)
    return deepcopy(
        (metadata, dict(graph.nodes(data=True)), list(graph.edges(data=True)))
    )


def test_ordinary_pointwise_fallback_remains_the_default():
    graph = _graph(blocked=True)
    result = execute_pointwise_stage(graph, Expansion(), (0,))
    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.nodes[0]["glyph_history"][-1] == "IL"
    assert graph.nodes[0]["nu_f"] == 1.0


@pytest.mark.parametrize("targets", [(0,), (1, 0), (0, 1)])
def test_exact_request_rejects_all_members_before_fallback_or_refresh(targets):
    graph = _graph(blocked=True)
    graph.add_node(1, **deepcopy(_graph(blocked=False).nodes[0]))
    before = _state(graph)
    callbacks = []
    with pytest.raises(TNFRValueError, match="exact two-phase"):
        execute_pointwise_stage(
            graph,
            Expansion(),
            targets,
            require_exact_operator=True,
            compute_delta_nfr=lambda graph: callbacks.append(graph),
        )
    assert _state(graph) == before
    assert callbacks == []
    assert not hasattr(graph, "_last_operator_applied")


def test_admitted_exact_request_reuses_stage_without_auxiliary_certificate():
    ordinary = _graph(blocked=False)
    guarded = deepcopy(ordinary)
    reference = execute_pointwise_stage(ordinary, Expansion(), (0,))
    result = execute_pointwise_stage(
        guarded, Expansion(), (0,), require_exact_operator=True
    )
    assert result == reference
    assert result.schedule == TWO_PHASE_JACOBI
    assert result.epi_jump_certificate is None
    assert _state(guarded, without_wrapper_cache=True) == _state(
        ordinary, without_wrapper_cache=True
    )
    assert guarded.nodes[0]["nu_f"] == 1.125


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_exact_guard_requires_an_actual_boolean_before_any_write(invalid):
    graph = _graph(blocked=False)
    before = _state(graph)
    with pytest.raises(TypeError, match="require_exact_operator must be a bool"):
        execute_pointwise_stage(
            graph, Expansion(), (0,), require_exact_operator=invalid
        )
    assert _state(graph) == before
