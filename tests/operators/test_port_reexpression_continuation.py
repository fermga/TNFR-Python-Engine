"""Acquired unequal port refinement and actual marked EN/RA continuation.

Formation uses current operators. Input occasions and the UM candidate policy
are supplied. Split edge weights belong only to the labelled diagnostic control.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.operators._port_reexpression_source import (
    A,
    B,
    R,
    U,
    V,
    W,
    _emit,
    _form,
    _held_channels,
    _receive,
    _source,
)
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.node import add_edge
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import capture_retained_organization
from tnfr.operators.retained_reception import (
    execute_retained_output_reception_stage,
    retained_output_ports,
)


def _input(graph, group, child, u, v):
    _emit(graph, U, u - Q(1, 8))
    _emit(graph, V, v - Q(1, 8))
    held = _held_channels(graph)
    a_input = _receive(graph, group, (U,))
    b_input = _receive(graph, (B,), (V,))
    assert tuple(row.node for row in a_input.reception_observations) == group
    assert all(row.source_nodes == (U,) for row in a_input.reception_observations)
    assert b_input.reception_observations[0].source_nodes == (V,)
    x1, y1 = (Q(25, 64) + u) / 2, (Q(3, 8) + v) / 2
    assert sum((_form(graph, n) for n in group), Q(0)) / len(group) == x1
    assert _form(graph, B) == y1
    if child is not None:
        assert (_form(graph, A) - _form(graph, child)) / 2 == Q(35, 512)
    assert _held_channels(graph) == held
    return x1, y1


def _output(graph, record):
    ports = retained_output_ports(record)
    (port,) = tuple(port for port in ports if port.receiver == R)
    assert port.sources == record.members
    protected = {node: deepcopy(graph.nodes[node]) for node in record.members}
    values = tuple(float(_form(graph, node)) for node in record.members)
    before = float(_form(graph, R))
    result = execute_retained_output_reception_stage(graph, record, port.ordinal)
    assert result.port == port
    assert result.before.members == result.after.members == record.members
    assert (
        result.stage_result.glyph == "EN" and result.stage_result.nodes_processed == 1
    )
    (observed,) = result.stage_result.reception_observations
    assert observed.node == R and observed.source_nodes == record.members
    assert observed.neighbors == record.members
    assert observed.neighbor_epi_values == values
    assert observed.target_epi_before == before
    assert observed.target_epi_after == float(_form(graph, R))
    assert not observed.source_tracking_enabled
    assert all(graph.nodes[node] == data for node, data in protected.items())
    return result


def _check_contexts(graph, child, *, grouped):
    for node, count in (
        (A, 13 if child is not None else 10),
        (B, 7),
        (U, 2),
        (V, 2),
        (R, 3 if grouped else 2),
        (W, 2),
    ):
        state = read_grammar_state(graph.nodes[node])
        assert state.operator_step == count and state.prior_coherence
    assert graph.nodes[A]["_structural_lineage"]["activation_count"] == (
        3 if child is not None else 2
    )
    assert graph.nodes[B]["_structural_lineage"]["activation_count"] == 2
    if child is not None:
        state = read_grammar_state(graph.nodes[child])
        assert state.operator_step == 4 and state.history == ("IL", "UM", "EN", "EN")
        assert state.prior_coherence and "_structural_lineage" not in graph.nodes[child]
    assert read_grammar_state(graph.nodes[R]).history == (
        ("IL", "EN", "EN") if grouped else ("IL", "EN")
    )
    assert read_grammar_state(graph.nodes[W]).history == ("IL", "RA")
    assert graph.graph["_t"] == 0.25 and tuple(graph.neighbors(W)) == (R,)
    assert get_attr(graph.nodes[W], ALIAS_VF) == 9 / 8


@pytest.mark.parametrize(
    "u,v,z1,z2,w1",
    [
        (Q(1, 4), Q(1, 2), Q(73, 256), Q(185, 512), Q(377, 1024)),
        (Q(1, 4), Q(3, 4), Q(73, 256), Q(217, 512), Q(409, 1024)),
        (Q(1, 2), Q(1, 2), Q(89, 256), Q(201, 512), Q(393, 1024)),
        (Q(1, 2), Q(3, 4), Q(89, 256), Q(233, 512), Q(425, 1024)),
    ],
)
def test_actual_unequal_refinement_preserves_marked_port_prefixes(u, v, z1, z2, w1):
    observations = {}
    for refined in (False, True):
        graph, group, child = _source(refined=refined)
        _input(graph, group, child, u, v)
        a_record = capture_retained_organization(graph, group)
        b_record = capture_retained_organization(graph, (B,))
        _output(graph, a_record)
        assert _form(graph, R) == z1
        first = _form(graph, R)
        # B's source record stays valid while the shared external receiver changes.
        _output(graph, b_record)
        assert _form(graph, R) == z2
        second = _form(graph, R)
        source_before_ra = {
            node: deepcopy(graph.nodes[node]) for node in (*group, B, R)
        }
        Resonance()(graph, W, validate_preconditions=True)
        assert _form(graph, W) == w1
        assert all(graph.nodes[node] == data for node, data in source_before_ra.items())
        _check_contexts(graph, child, grouped=True)
        observations[refined] = (first, second, _form(graph, W))
    assert observations[False] == observations[True] == (z1, z2, w1)


@pytest.mark.parametrize("split_weights", [False, True])
def test_flattened_ports_change_response_even_when_edge_weights_are_split(
    split_weights,
):
    outcomes = {}
    for refined in (False, True):
        graph, group, child = _source(refined=refined)
        _input(graph, group, child, Q(1, 4), Q(1, 2))
        if refined and split_weights:
            # Explicit noncanonical diagnostic intervention; recapture afterwards.
            add_edge(graph, A, R, 0.5, overwrite=True)
            add_edge(graph, child, R, 0.5, overwrite=True)
            assert graph.edges[A, R]["weight"] + graph.edges[child, R]["weight"] == 1.0
        organization = (*group, B)
        record = capture_retained_organization(graph, organization)
        assert tuple(port.receiver for port in retained_output_ports(record)) == (R,)
        _output(graph, record)
        expected_receiver = Q(156, 512) if refined else Q(161, 512)
        assert _form(graph, R) == expected_receiver
        Resonance()(graph, W, validate_preconditions=True)
        expected_observer = Q(348, 1024) if refined else Q(353, 1024)
        assert _form(graph, W) == expected_observer
        _check_contexts(graph, child, grouped=False)
        outcomes[refined] = _form(graph, W)
    assert outcomes[True] - outcomes[False] == -Q(5, 1024)
