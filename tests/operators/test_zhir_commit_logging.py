"""ZHIR commit logging cannot invalidate an accepted network stage."""

from __future__ import annotations

from copy import deepcopy
import logging

import networkx as nx

from tnfr.constants.aliases import ALIAS_THETA
from tnfr.operators.definitions import Mutation
from tnfr.operators.network_stage import TWO_PHASE_JACOBI, execute_pointwise_stage
from tnfr.operators.preconditions.mutation import diagnose_mutation_readiness


class _FailingHandler(logging.Handler):
    """Count accepted context records and fail after each irreversible emit."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.calls += 1
        raise RuntimeError("injected logging failure")


def _mutation_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        RANDOM_SEED=17,
        ZHIR_THRESHOLD_XI=0.1,
        GLYPH_FACTORS={"ZHIR_theta_shift_factor": 0.5},
    )
    for node in graph:
        epi = 0.3 + 0.1 * node
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            DeltaNFR=0.2,
            theta=0.1 * node,
            EPI_kind="wave",
            epi_history=[0.0, 0.1, epi],
            glyph_history=["IL", "OZ"],
        )
    return graph


def test_failing_log_handler_cannot_rollback_or_interrupt_accepted_targets() -> None:
    graph = _mutation_graph()
    theta_before = {
        node: float(graph.nodes[node][ALIAS_THETA[0]]) for node in graph
    }
    logger = logging.getLogger("tnfr.operators.preconditions.mutation")
    handler = _FailingHandler()
    old_level = logger.level
    old_propagate = logger.propagate
    old_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
    try:
        result = execute_pointwise_stage(graph, Mutation(), (0, 1))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        logger.disabled = old_disabled

    assert result.schedule == TWO_PHASE_JACOBI
    assert result.nodes_processed == 2
    assert handler.calls == 2
    for node in graph:
        assert graph.nodes[node][ALIAS_THETA[0]] != theta_before[node]
        assert graph.nodes[node]["glyph_history"][-1] == "ZHIR"
        assert graph.nodes[node]["_mutation_context"][
            "destabilizer_operator"
        ] == "dissonance"


def test_readiness_diagnostic_is_observationally_pure() -> None:
    graph = _mutation_graph()
    graph_before = deepcopy(graph.graph)
    nodes_before = {node: deepcopy(attrs) for node, attrs in graph.nodes(data=True)}
    logger = logging.getLogger("tnfr.operators.preconditions.mutation")
    handler = _FailingHandler()
    old_level = logger.level
    old_propagate = logger.propagate
    old_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
    try:
        report = diagnose_mutation_readiness(graph, 0)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        logger.disabled = old_disabled

    assert report["ready"] is True
    assert handler.calls == 0
    assert graph.graph == graph_before
    assert {
        node: dict(attrs) for node, attrs in graph.nodes(data=True)
    } == nodes_before
