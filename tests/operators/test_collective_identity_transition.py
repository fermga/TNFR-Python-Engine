"""Synthetic marked-contact controls for the complete NUL/IL/SHA transition.

These tests use no retained research preparation or response producer. They
exercise the existing runtime and show information lost by tempting smaller
descriptions; they do not certify a surrounding family or future maintenance.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import atan2, cos, pi, sin

import networkx as nx
import pytest

from tnfr._exact_time import finite_represented_real
from tnfr.alias import set_dnfr
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.operators import network_stage
from tnfr.operators._coherence_stage_kernel import propose_coherence_phase
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.definitions import Coherence, Contraction, Silence
from tnfr.operators.grammar_debt import U2_DEBT_KEY, node_debt
from tnfr.operators.grammar_dynamics import (
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution
from tnfr.types import Glyph, require_finite_real_scalar_epi

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PERMUTATION = (0, 1) + tuple(node ^ 1 for node in range(2, 20))
PORTS = (0, 1)
SCALE = 1 - 1 / (4 * pi)
NOW = datetime(2030, 1, 2, 3, 4, 6, tzinfo=timezone.utc)


class _SuppliedClock:
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def _contact():
    graph = nx.Graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"SHA_vf_factor": 0.5},
        SHA_MIN_VF=0.125,
        EPI_LATENT_MAX=1.0,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        EDGE_AWARE_ENABLED=True,
        _t=7,
    )
    for node in range(20):
        history = ("AL", "UM", "SHA", "IL", "OZ")
        if node == 1:
            history = ("AL", "AL") + history
        offset = 1 / 64 if node == 1 else 1 / 16 if node in (10, 11) else 0
        graph.add_node(
            node,
            EPI=5 / 16 + (1 / 1024 if node == 0 else 0),
            theta=2 * pi * (node % 10 // 2) / 5 + offset,
            nu_f=0.5 if node < 2 else 1.0,
            delta_nfr=0.0,
            glyph_history=deque(history, maxlen=8),
            _grammar_u2_debt=1,
            _grammar_prior_coherence=True,
            constituent_record={"member": node},
        )
    for i in graph:
        for j in range(i + 1, 20):
            if i // 10 == j // 10 and (i % 10 // 2 - j % 10 // 2) % 5 in (1, 4):
                graph.add_edge(i, j, weight=1.0)
    for i, j, weight in (
        (0, 1, 7 / 8),
        (0, 10, 15 / 16),
        (0, 11, 15 / 16),
        (1, 10, 13 / 16),
        (1, 11, 13 / 16),
    ):
        graph.add_edge(i, j, weight=weight)
    return graph


def _fields(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _refresh(graph):
    fields = _fields(graph)
    for node, pressure in zip(fields.nodes, fields.pressure, strict=True):
        set_dnfr(
            graph,
            node,
            finite_represented_real(pressure.midpoint, "synthetic pressure")[0],
        )


def _physical(graph):
    return tuple(
        tuple(graph.nodes[node][key] for key in ("EPI", "theta", "nu_f", "delta_nfr"))
        for node in graph
    )


def _execute(graph):
    operators = (Contraction(), Coherence(), Silence())
    sequence = ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    _refresh(graph)
    before = deepcopy(graph)
    stages, raw_pressures, observations, states = [], [], [], []

    def refresh(subject):
        raw_pressures.append(tuple(subject.nodes[node]["delta_nfr"] for node in PORTS))
        _refresh(subject)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            assert all(
                validate_candidate(graph, node, operator.glyph.value).allowed
                for node in PORTS
            )
            potential = compute_structural_potential(graph)
            stages.append(
                execute_network_operator_stage(
                    graph,
                    operator,
                    PORTS,
                    sequence_context=sequence.step(index),
                    compute_delta_nfr=refresh,
                )
            )
            observations.append(
                observe_structural_potential_confinement(
                    graph,
                    potential,
                    compute_structural_potential(graph),
                    reference=f"synthetic_before_{operator.glyph.value}",
                )
            )
            states.append(deepcopy(graph))
    return (
        before,
        graph,
        tuple(stages),
        tuple(raw_pressures),
        tuple(observations),
        tuple(states),
    )


@pytest.fixture(scope="module")
def complete_word():
    return _execute(_contact())


def test_complete_word_preserves_marked_symmetry_and_associated_context(complete_word):
    before, after, stages, _, observations, states = complete_word
    assert before.number_of_edges() == after.number_of_edges() == 45
    assert tuple(before.edges(data=True)) == tuple(after.edges(data=True))
    assert tuple(before) == tuple(after) == tuple(range(20))
    assert all(
        tuple(before.neighbors(node)) == tuple(after.neighbors(node)) for node in before
    )
    assert tuple(stage.glyph for stage in stages) == ("NUL", "IL", "SHA")
    assert all(
        stage.schedule == "two_phase_jacobi" and stage.nodes_processed == 2
        for stage in stages
    )
    assert all(observation.confined for observation in observations)
    assert [
        tuple(node_debt(graph.nodes[node]) for node in PORTS) for graph in states
    ] == [(1, 1), (0, 0), (0, 0)]
    for graph in (before,) + states:
        reduced = assess_sine_support_involution(
            _fields(graph), permutation_indices=PERMUTATION
        )
        assert reduced.source_trajectory_reduction_certified
        assert reduced.multiplicities == (1, 1) + (2,) * 9
        for node in range(2, 20, 2):
            assert tuple(
                graph.nodes[node][key] for key in ("EPI", "theta", "nu_f")
            ) == tuple(graph.nodes[node + 1][key] for key in ("EPI", "theta", "nu_f"))
        for left, right in graph.edges:
            gap = (graph.nodes[right]["theta"] - graph.nodes[left]["theta"] + pi) % (
                2 * pi
            ) - pi
            assert abs(gap) < pi / 2
        for origin in (0, 10):
            cycle = tuple(origin + 2 * index for index in range(5))
            gaps = tuple(
                (
                    graph.nodes[cycle[(index + 1) % 5]]["theta"]
                    - graph.nodes[node]["theta"]
                    + pi
                )
                % (2 * pi)
                - pi
                for index, node in enumerate(cycle)
            )
            assert sum(gaps) == pytest.approx(2 * pi, abs=1e-14)
    for node in after:
        assert after.nodes[node]["constituent_record"] == {"member": node}
        if node not in PORTS:
            assert (
                after.nodes[node]["glyph_history"]
                == before.nodes[node]["glyph_history"]
            )
            assert node_debt(after.nodes[node]) == 1
    for node in PORTS:
        assert tuple(after.nodes[node]["glyph_history"])[-3:] == ("NUL", "IL", "SHA")
        assert after.nodes[node]["latent"] is True
        assert after.nodes[node]["latency_start_time"] == NOW.isoformat()
        assert after.nodes[node]["preserved_epi"] == require_finite_real_scalar_epi(
            after.nodes[node]["EPI"]
        )
    assert before.graph["_t"] == after.graph["_t"] == 7


def test_primary_maps_and_refresh_are_distinct_at_actual_stage_boundaries(
    complete_word,
):
    before, after, _, raw_pressures, _, states = complete_word
    for node in PORTS:
        assert (
            require_finite_real_scalar_epi(after.nodes[node]["EPI"])
            == SCALE * before.nodes[node]["EPI"]
        )
        assert after.nodes[node]["nu_f"] == (SCALE * before.nodes[node]["nu_f"]) * 0.5
        assert states[0].nodes[node]["theta"] == before.nodes[node]["theta"]
        assert states[0].nodes[node]["delta_nfr"] == before.nodes[node]["delta_nfr"]
        assert raw_pressures[0][node] == before.nodes[node]["delta_nfr"] * (1 / SCALE)
        assert raw_pressures[0][node] != states[0].nodes[node]["delta_nfr"]
        neighbors = tuple(before.neighbors(node))
        mean = atan2(
            sum(sin(before.nodes[n]["theta"]) for n in neighbors),
            sum(cos(before.nodes[n]["theta"]) for n in neighbors),
        )
        angle = before.nodes[node]["theta"]
        delta = (mean - angle + pi) % (2 * pi) - pi
        expected = (angle + 0.3 * delta) % (2 * pi)
        assert after.nodes[node]["theta"] == pytest.approx(expected, abs=1e-15)


def test_eleven_cell_chart_is_not_an_eleven_node_operator_graph():
    fine = _contact()
    reduced = assess_sine_support_involution(
        _fields(fine), permutation_indices=PERMUTATION
    )
    coarse = nx.Graph()
    for cell, representative in enumerate(reduced.representative_indices):
        coarse.add_node(cell, **fine.nodes[representative])
    for i, j in fine.edges:
        coarse.add_edge(reduced.node_orbits[i], reduced.node_orbits[j])
    assert fine.degree(0) == reduced.reduced_degrees[0] == 7
    assert coarse.degree(0) == 4
    assert reduced.bare_neighbor_counts[0][reduced.node_orbits[10]] == 2
    fine_phase = propose_coherence_phase(fine, 0, 0.3)
    coarse_phase = propose_coherence_phase(coarse, 0, 0.3)
    assert abs(fine_phase.theta_after - coarse_phase.theta_after) > 1e-4


def test_same_physical_successor_does_not_erase_inherited_grammar_obligations(
    complete_word,
):
    before, after, *_ = complete_word
    other = _contact()
    for node in PORTS:
        other.nodes[node][U2_DEBT_KEY] = 2
    other_before, other_after, *_ = _execute(other)
    assert _physical(before) == _physical(other_before)
    assert _physical(after) == _physical(other_after)
    for node in PORTS:
        assert (
            before.nodes[node]["glyph_history"]
            == other_before.nodes[node]["glyph_history"]
        )
        assert node_debt(after.nodes[node]) == 0
        assert node_debt(other_after.nodes[node]) == 1
        clear = validate_sequence_incremental(after, node, ("VAL", "VAL"))
        owing = validate_sequence_incremental(other_after, node, ("VAL", "VAL"))
        assert tuple(item.allowed for item in clear) == (True, True)
        assert tuple(item.allowed for item in owing) == (True, False)
        assert {violation.rule for violation in owing[-1].violations} == {"U2"}


def test_centered_form_loses_the_origin_consumed_by_local_contraction():
    def centered(values):
        mean = sum(values) / len(values)
        return tuple(value - mean for value in values)

    original = (Q(5, 16),) * 20
    shifted = tuple(value + Q(1, 32) for value in original)
    assert centered(original) == centered(shifted)

    def apply(values):
        output = list(values)
        for node in PORTS:
            proposal = propose_scale_operator(
                glyph=Glyph.NUL,
                factor=0.75,
                vf_before=0.5,
                dnfr_before=0.0,
                edge_aware_enabled=True,
                epi_before=values[node],
            )
            output[node] = Q(proposal.epi_after)
        return centered(output)

    original_after, shifted_after = apply(original), apply(shifted)
    expected = tuple(
        Q(-1, 4) * Q(1, 32) * (Q(node in PORTS) - Q(1, 10)) for node in range(20)
    )
    assert tuple(b - a for a, b in zip(original_after, shifted_after)) == expected
    assert any(expected)
