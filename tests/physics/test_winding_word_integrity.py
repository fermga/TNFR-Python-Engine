"""Live admission, configured gates and support identity in winding words."""

from copy import deepcopy
import math

import networkx as nx
import pytest

from tnfr.operators.definitions import Coherence, Coupling, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.emergent_particles import winding_ring
from tnfr.physics.winding_certificates import (
    certify_phase_winding,
    observe_winding_word,
)


def _graph():
    graph = winding_ring(8, 1)
    for _, data in graph.nodes(data=True):
        data["νf"] = float(data["nu_f"])
        data["vf"] = float(data["nu_f"])
        data["ΔNFR"] = float(data["dnfr"])
        data["glyph_history"] = []
    return graph


def _observe_hook(graph, change):
    calls = []

    def hook(current):
        if not calls:
            change(current)
        calls.append(len(calls))

    graph.graph["compute_delta_nfr"] = hook
    result = observe_winding_word(
        graph, range(8), 0, [Emission(), Coherence(), Silence()]
    )
    assert calls == [0, 1, 2]
    assert result.actual_history == ("AL", "IL", "SHA")
    assert result.history_preserved
    assert not result.steps[1].topology_changed
    return result.steps[0]


@pytest.mark.parametrize("missing", [False, True])
def test_non_generator_word_cannot_invent_initial_epi_admission(missing):
    graph = _graph()
    if missing:
        graph.nodes[0].pop("EPI")
    else:
        graph.nodes[0]["EPI"] = 0.0
    before = deepcopy(list(graph.nodes(data=True)))

    with pytest.raises(ValueError, match="U1a"):
        observe_winding_word(
            graph, range(8), 0, [Coherence(), Silence()]
        )

    assert list(graph.nodes(data=True)) == before
    assert all(graph.nodes[node]["EPI"] != 0.0 for node in range(1, 8))


@pytest.mark.parametrize("initial", [-1.0, 1.0])
def test_live_nonzero_primary_epi_admits_non_generator_start(initial):
    graph = _graph()
    graph.nodes[0]["EPI"] = initial
    # A retained generator also satisfies the separate incremental live gate.
    graph.nodes[0]["glyph_history"] = ["AL"]

    result = observe_winding_word(
        graph, range(8), 0, [Coherence(), Silence()]
    )

    assert result.history_preserved
    assert result.actual_history == ("IL", "SHA")


def test_generator_word_still_starts_from_zero_primary_epi():
    graph = _graph()
    graph.nodes[0]["EPI"] = 0.0

    result = observe_winding_word(
        graph, range(8), 0, [Emission(), Coherence(), Silence()]
    )

    assert result.history_preserved
    assert result.actual_history == ("AL", "IL", "SHA")


@pytest.mark.parametrize(
    "operator_types",
    [(Coupling, Coupling, Silence), (Emission, Silence, Silence)],
)
def test_instance_valid_words_still_require_shared_adjacency_admission(
    operator_types,
):
    graph = _graph()
    word = [operator_type() for operator_type in operator_types]
    ValidatedSequence(word, context={"initial_epi_nonzero": True})
    before = deepcopy(
        (graph.graph, list(graph.nodes(data=True)), list(graph.edges(data=True)))
    )

    with pytest.raises(ValueError, match="Invalid sequence"):
        observe_winding_word(graph, range(8), 0, word)

    after = (graph.graph, list(graph.nodes(data=True)), list(graph.edges(data=True)))
    assert after == before


def test_word_uses_runtime_graph_gate_without_changing_standalone_default():
    graph = _graph()
    graph.graph["DELTA_PHI_MAX"] = math.pi / 8.0
    direct = certify_phase_winding(graph, range(8))

    result = observe_winding_word(
        graph, range(8), 0, [Emission(), Coherence(), Silence()]
    )

    assert direct.u3_admissible
    for certificate in (result.initial, *(step.certificate for step in result.steps)):
        assert certificate.is_defined
        assert certificate.winding == 1
        assert not certificate.u3_admissible
        assert certificate.minimum_u3_margin == pytest.approx(-math.pi / 8.0)


def test_post_step_gate_is_read_after_the_configured_hook():
    graph = _graph()

    step = _observe_hook(
        graph, lambda current: current.graph.update(DELTA_PHI_MAX=math.pi / 8.0)
    )

    assert not step.certificate.u3_admissible
    assert step.certificate.minimum_u3_margin == pytest.approx(-math.pi / 8.0)
    assert not step.topology_changed


def test_coupling_step_reports_its_effective_tightened_gate():
    graph = _graph()
    graph.graph["UM_MAX_PHASE_DIFF"] = math.pi / 3.0

    result = observe_winding_word(
        graph, range(8), 0, [Emission(), Coupling(), Silence()]
    )

    assert result.initial.minimum_u3_margin == pytest.approx(math.pi / 4.0)
    coupled = result.steps[1].certificate
    expected = certify_phase_winding(graph, range(8), phase_gate=math.pi / 3.0)
    assert coupled.minimum_u3_margin == expected.minimum_u3_margin
    assert coupled.u3_admissible == expected.u3_admissible
    assert result.steps[2].certificate.minimum_u3_margin == pytest.approx(
        coupled.minimum_u3_margin + math.pi / 6.0
    )


def test_invalid_graph_gate_rejects_before_the_first_operator():
    graph = _graph()
    graph.graph["DELTA_PHI_MAX"] = math.pi

    with pytest.raises(ValueError, match="canonical interval"):
        observe_winding_word(
            graph, range(8), 0, [Emission(), Coherence(), Silence()]
        )

    assert graph.nodes[0]["glyph_history"] == []


def test_later_coupling_failure_keeps_the_executed_prefix():
    graph = _graph()
    graph.graph["DELTA_PHI_MAX"] = math.pi / 8.0

    with pytest.raises(OperatorPreconditionError, match="U3 phase gate"):
        observe_winding_word(
            graph, range(8), 0, [Emission(), Coupling(), Silence()]
        )

    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL",)


def test_equal_count_rewiring_is_a_topology_change():
    graph = _graph()
    graph.add_edge(0, 2)

    def rewire(current):
        current.remove_edge(0, 2)
        current.add_edge(0, 3)

    step = _observe_hook(graph, rewire)

    assert step.edge_count_before == step.edge_count_after
    assert step.topology_changed
    assert step.certificate.cycle_exists


def test_node_replacement_is_detected_without_an_edge_count_change():
    graph = _graph()
    graph.add_node("old", theta=0.0)

    def replace(current):
        current.remove_node("old")
        current.add_node("new", theta=0.0)

    step = _observe_hook(graph, replace)

    assert step.edge_count_before == step.edge_count_after
    assert step.topology_changed
    assert step.certificate.cycle_exists


def test_directed_edge_reversal_is_detected_at_fixed_edge_count():
    graph = nx.DiGraph(_graph())
    graph.add_edge(0, 2)

    def reverse(current):
        current.remove_edge(0, 2)
        current.add_edge(2, 0)

    step = _observe_hook(graph, reverse)

    assert step.edge_count_before == step.edge_count_after
    assert step.topology_changed


def test_parallel_edge_key_replacement_is_detected():
    graph = nx.MultiGraph(_graph())
    graph.add_edge(0, 2, key="old")

    def replace_key(current):
        current.remove_edge(0, 2, key="old")
        current.add_edge(0, 2, key="new")

    step = _observe_hook(graph, replace_key)

    assert step.edge_count_before == step.edge_count_after
    assert step.topology_changed


def test_undirected_reinsertion_and_attribute_changes_preserve_topology():
    graph = _graph()
    graph.add_node("extra", theta=0.0)

    def reinsert(current):
        current.remove_edge(0, 1)
        current.add_edge(1, 0, weight=2.0)
        current.remove_node("extra")
        current.add_node("extra", theta=0.0)

    step = _observe_hook(graph, reinsert)

    assert step.edge_count_before == step.edge_count_after
    assert not step.topology_changed
    assert step.certificate.cycle_exists
