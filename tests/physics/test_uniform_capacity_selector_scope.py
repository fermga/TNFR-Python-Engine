"""Configured default selection is distinct from Mutation admissibility.

These controls evaluate diagnostics, selection and read-only grammar/gates.
Declared histories are portable inputs, not claims of prior native execution.
No glyph, integration step, pressure refresh or new phase law is executed.
"""

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import get_graph_param
from tnfr.constants.aliases import ALIAS_SI, ALIAS_VF
from tnfr.dynamics.selectors import DefaultGlyphSelector, _choose_glyph
from tnfr.metrics import sense_index
from tnfr.operators._mutation_gate import validate_mutation_runtime_gate
from tnfr.operators.grammar_dynamics import enforce_grammar_on_glyph, validate_candidate
from tnfr.types import Glyph


def _pair(capacity=1.0):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.5,
            theta=node * math.pi,
            delta_nfr=1.0,
            glyph_history=["IL", "OZ"],
            epi_time_history=[(0.0, 0.4), (0.25, 0.5)],
        )
        # Use a supported alternate spelling, without a competing first alias.
        graph.nodes[node][ALIAS_VF[-1]] = capacity
    return graph


@pytest.mark.parametrize("capacity", [float.fromhex("0x0.0000000000001p-1022"), 2.0])
@pytest.mark.parametrize("scalar", [False, True])
def test_fresh_default_si_has_a_sharp_uniform_capacity_floor(
    capacity, scalar, monkeypatch
):
    if scalar:
        monkeypatch.setattr(sense_index, "np", None)
    graph = _pair(capacity)
    graph.graph.update(_vfmax=1000.0, _dnfrmax=1000.0)
    alpha, beta, gamma = sense_index.get_Si_weights(graph)
    sense_index.compute_Si(graph, inplace=True)
    values = tuple(get_attr(graph.nodes[node], ALIAS_SI) for node in graph)
    assert alpha > 0.5 and beta >= 0 and gamma >= 0
    assert alpha + beta + gamma == pytest.approx(1.0, abs=2e-16, rel=0)
    assert graph.graph["_vfmax"] == capacity
    assert graph.graph["_dnfrmax"] == 1.0
    # Antipodal singleton neighbors and maximal pressure make both remaining
    # diagnostic contributions zero. This attains, rather than fits, the bound.
    assert all(value >= alpha for value in values)
    assert values == pytest.approx((alpha, alpha), abs=2e-16, rel=0)
    selector = DefaultGlyphSelector()
    assert tuple(selector(graph, node) for node in graph) == ("IL", "IL")
    selector.prepare(graph, tuple(graph))
    assert tuple(selector(graph, node) for node in graph) == ("IL", "IL")


def test_valid_mutation_history_and_grammar_do_not_make_default_selector_request_it():
    graph = _pair()
    # This input has the same strict phase/pressure interval bounds as the
    # current comparison; this P2 input does not inherit the prism theorem.
    graph.nodes[1]["theta"] = math.pi / 6
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 0.2
    sense_index.compute_Si(graph, inplace=True)
    selector = DefaultGlyphSelector()
    selector.prepare(graph, tuple(graph))
    for node in graph:
        gate = validate_mutation_runtime_gate(graph.nodes[node], graph.graph)
        assert gate.threshold.crossed and gate.threshold.physical_time_resolved
        assert gate.threshold.depi_dt > gate.threshold.xi
        assert validate_candidate(graph, node, "ZHIR").allowed
        assert selector(graph, node) == "IL"
        assert enforce_grammar_on_glyph(graph, node, "IL") == "IL"


def test_force_lag_and_initialization_gates_do_not_create_a_mutation_request():
    graph = _pair()
    sense_index.compute_Si(graph, inplace=True)
    selector = DefaultGlyphSelector()
    al_max = get_graph_param(graph, "AL_MAX_LAG", int)
    en_max = get_graph_param(graph, "EN_MAX_LAG", int)

    def choose(al_lag, en_lag):
        return _choose_glyph(
            graph, 0, selector, True, {0: al_lag}, {0: en_lag}, al_max, en_max
        )

    # These are the shared chooser's supplied counter values; incrementing
    # native counters and executing the chosen operator are separate actions.
    assert choose(al_max, en_max) == "IL"
    assert choose(al_max, en_max + 1) is Glyph.EN
    assert choose(al_max + 1, en_max + 1) is Glyph.AL
    graph.nodes[0]["EPI"] = 0.0
    graph.nodes[0]["glyph_history"] = []
    assert not validate_candidate(graph, 0, "IL").allowed
    replacement = enforce_grammar_on_glyph(graph, 0, "IL")
    assert replacement in {"AL", "NAV", "REMESH"}
    assert replacement != "ZHIR"


def test_remote_larger_capacity_removes_the_uniform_global_capacity_premise():
    graph = _pair()
    graph.nodes[0]["delta_nfr"] = 0.0
    sense_index.compute_Si(graph, inplace=True)
    selector = DefaultGlyphSelector()
    assert selector(graph, 0) == "IL"
    original_neighbors = tuple(graph.neighbors(0))
    # Construct the second declared input before any graph-owned cache exists;
    # this is not a live topology update or a cache-invalidation experiment.
    graph = _pair()
    graph.nodes[0]["delta_nfr"] = 0.0
    graph.add_node(2, EPI=0.5, nu_f=100.0, theta=0.0, delta_nfr=1.0)
    values = sense_index.compute_Si(graph, inplace=True)
    assert tuple(graph.neighbors(0)) == original_neighbors
    assert values[0] < 0.25
    assert DefaultGlyphSelector()(graph, 0) == "ZHIR"
    # This is a diagnostic normalization counterexample to an unconditional
    # no-Mutation claim, not an executed Mutation or a derived physical law.
