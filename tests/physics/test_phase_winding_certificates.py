"""Branch-aware phase-winding telemetry and U3 boundary checks."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    Silence,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.emergent_particles import winding_ring
from tnfr.physics.winding_certificates import (
    certify_phase_winding,
    observe_winding_word,
)


def _executable_ring(nodes: int = 8, winding: float = 1.0) -> nx.Graph:
    graph = winding_ring(nodes, winding)
    for _, data in graph.nodes(data=True):
        data["νf"] = float(data["nu_f"])
        data["vf"] = float(data["nu_f"])
        data["ΔNFR"] = float(data["dnfr"])
        data["glyph_history"] = []
    graph.nodes[0]["epi_history"] = [0.0, 0.2, 1.0]
    return graph


def test_orientation_reversal_flips_only_signed_winding():
    graph = _executable_ring()
    forward = certify_phase_winding(graph, range(8))
    reverse = certify_phase_winding(graph, reversed(range(8)))
    assert forward.winding == 1
    assert reverse.winding == -1
    assert forward.absolute_winding == reverse.absolute_winding == 1


def test_relabeling_preserves_declared_oriented_winding():
    graph = _executable_ring()
    mapping = {node: f"node-{(3 * node) % 8}" for node in graph}
    relabelled = nx.relabel_nodes(graph, mapping)
    original = certify_phase_winding(graph, range(8))
    transformed = certify_phase_winding(
        relabelled, [mapping[node] for node in range(8)]
    )
    assert transformed.winding == original.winding
    assert transformed.raw_winding == pytest.approx(original.raw_winding)


def test_phase_perturbation_below_margin_preserves_winding():
    graph = _executable_ring()
    before = certify_phase_winding(graph, range(8))
    margin = min(before.minimum_u3_margin, before.minimum_branch_margin)
    perturbation = margin / 8
    for node in graph:
        graph.nodes[node]["theta"] += perturbation * math.sin(
            2 * math.pi * node / 8
        )
    after = certify_phase_winding(graph, range(8))
    assert after.winding == before.winding == 1
    assert after.u3_admissible
    assert after.minimum_branch_margin > 0.0


def test_branch_boundary_is_undefined_instead_of_rounded():
    graph = _executable_ring()
    epsilon = 1e-5
    states = []
    for phase in (math.pi - epsilon, math.pi, math.pi + epsilon):
        graph.nodes[1]["theta"] = phase
        states.append(certify_phase_winding(graph, range(8)))
    assert states[0].is_defined and states[0].winding == 1
    assert not states[1].is_defined and states[1].winding is None
    assert states[1].cycle_exists
    assert "branch boundary" in states[1].reason
    assert states[2].is_defined and states[2].winding == 0


def test_missing_cycle_is_an_undefined_observation():
    graph = _executable_ring()
    graph.remove_edge(3, 4)
    certificate = certify_phase_winding(graph, range(8))
    assert not certificate.is_defined
    assert not certificate.cycle_exists
    assert certificate.winding is None


def test_u3_rejects_inadmissible_high_winding_word():
    graph = _executable_ring(winding=3.0)
    certificate = certify_phase_winding(graph, range(8))
    assert certificate.winding == 3
    assert certificate.u3_admissible is False
    with pytest.raises(OperatorPreconditionError, match="U3 phase gate"):
        observe_winding_word(
            graph,
            range(8),
            0,
            [Emission(), Coupling(), Coherence(), Silence()],
        )
    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL",)


def test_validated_words_record_actual_history_and_preserve_cycle():
    graph = _executable_ring()
    no_phase_change = observe_winding_word(
        graph, range(8), 0, [Emission(), Coherence(), Silence()]
    )
    assert no_phase_change.history_preserved
    assert no_phase_change.actual_history == ("AL", "IL", "SHA")
    assert all(not step.phase_changes for step in no_phase_change.steps)
    assert all(step.certificate.winding == 1 for step in no_phase_change.steps)

    graph = _executable_ring()
    coupled = observe_winding_word(
        graph,
        range(8),
        0,
        [Emission(), Coupling(), Coherence(), Silence()],
    )
    assert coupled.history_preserved
    assert coupled.steps[1].operator == "UM"
    assert coupled.steps[1].phase_changes
    assert coupled.steps[1].certificate.u3_admissible
    assert coupled.steps[-1].certificate.winding == 1

    graph = _executable_ring()
    mutated = observe_winding_word(
        graph,
        range(8),
        0,
        [
            Emission(),
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
            Silence(),
        ],
    )
    assert mutated.history_preserved
    assert mutated.actual_history == ("AL", "IL", "OZ", "ZHIR", "IL", "SHA")
    assert mutated.steps[3].phase_changes
    assert mutated.steps[3].certificate.winding == 1
    assert mutated.steps[3].certificate.minimum_branch_margin > 0.0
