"""Actual member AL realizes a supplied port gain within a marked EN/RA word.

Member selection and invocation are declared. The shared source is acquired by
actual operators; its own histories and unequal members are never reconstructed
from the mean. These finite cases do not assert repeated collective AL descent.
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
    _support,
)
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI_KIND, ALIAS_VF
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_emission import execute_retained_emission_stage
from tnfr.operators.retained_organization import capture_retained_organization
from tnfr.operators.retained_reception import (
    execute_retained_output_reception_stage,
    retained_output_ports,
)
from tnfr.utils._structural_signature import structural_proof_signature


def _mean(graph, group):
    return sum((_form(graph, node) for node in group), Q(0)) / len(group)


def _checkpoint(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _realize_gain(graph, group, child, gain):
    index = 0 if child is None else 1
    target = group[index]
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = float(len(group) * gain)
    factors = deepcopy(graph.graph["GLYPH_FACTORS"])
    record = capture_retained_organization(graph, group)
    protected = {
        node: deepcopy(data) for node, data in graph.nodes(data=True) if node != target
    }
    held = _held_channels(graph)
    support = _support(graph)
    prior = read_grammar_state(graph.nodes[target])
    kind = get_attr(graph.nodes[target], ALIAS_EPI_KIND, strict=True, conv=str)
    target_before = _form(graph, target)
    mean_before = _mean(graph, group)
    origin_before = graph.nodes[target].get("_emission_origin")
    assert prior.u2_debt == 0 and prior.prior_coherence
    assert target_before < Q(1, 2)
    assert get_attr(graph.nodes[target], ALIAS_VF) >= 0.5
    if child is not None:
        assert _form(graph, A) == Q(135, 256) > Q(1, 2)
        assert "_structural_lineage" not in graph.nodes[child]
        assert prior.history == ("IL", "UM", "EN")

    dispatch = execute_retained_emission_stage(graph, record, index, mean_gain=gain)

    assert dispatch.member_index == index and dispatch.member == target
    assert dispatch.mean_gain == dispatch.realized_mean_gain == gain
    assert dispatch.local_boost == float(len(group) * gain)
    assert dispatch.before.members == dispatch.after.members == group
    assert dispatch.stage_result.glyph == "AL"
    assert dispatch.stage_result.nodes_processed == 1
    assert _form(graph, target) == target_before + len(group) * gain
    assert _mean(graph, group) == mean_before + gain == Q(25, 64) + gain
    assert _held_channels(graph) == held and _support(graph) == support
    assert graph.graph["GLYPH_FACTORS"] == factors and graph.graph["_t"] == 0.25
    assert all(graph.nodes[node] == data for node, data in protected.items())
    assert get_attr(graph.nodes[target], ALIAS_EPI_KIND, strict=True, conv=str) == kind
    after = read_grammar_state(graph.nodes[target])
    assert after.operator_step == prior.operator_step + 1
    assert after.prior_coherence and after.u2_debt == prior.u2_debt
    assert after.history[-1] == "AL"
    lineage = graph.nodes[target]["_structural_lineage"]
    assert graph.nodes[target]["_emission_activated"] is True
    assert lineage["origin"] == graph.nodes[target]["_emission_origin"]
    if child is None:
        assert lineage["activation_count"] == 3
        assert graph.nodes[target]["_emission_origin"] == origin_before
    else:
        assert lineage["activation_count"] == 1
        assert lineage["parent_emission"] is None and lineage["derived_nodes"] == []
        assert after.history == ("IL", "UM", "EN", "AL")
        assert "epi_time_history" not in graph.nodes[child]
        assert (_form(graph, A) - _form(graph, child)) / 2 == Q(35, 256) - gain


def _output(graph, record):
    (port,) = tuple(
        port for port in retained_output_ports(record) if port.receiver == R
    )
    before = {node: deepcopy(graph.nodes[node]) for node in record.members}
    values = tuple(float(_form(graph, node)) for node in record.members)
    dispatch = execute_retained_output_reception_stage(graph, record, port.ordinal)
    (observation,) = dispatch.stage_result.reception_observations
    assert observation.source_nodes == observation.neighbors == record.members
    assert observation.neighbor_epi_values == values
    assert observation.node == R
    assert all(graph.nodes[node] == data for node, data in before.items())


def _final_contexts(graph, child):
    for node, count in (
        (A, 11 if child is None else 13),
        (B, 7),
        (U, 2),
        (V, 2),
        (R, 3),
        (W, 2),
    ):
        state = read_grammar_state(graph.nodes[node])
        assert state.operator_step == count and state.prior_coherence
    assert graph.nodes[A]["_structural_lineage"]["activation_count"] == 3
    assert graph.nodes[B]["_structural_lineage"]["activation_count"] == 2
    if child is not None:
        state = read_grammar_state(graph.nodes[child])
        assert state.operator_step == 5
        assert state.history == ("IL", "UM", "EN", "AL", "EN")
        assert state.prior_coherence and state.u2_debt == 0
        assert graph.nodes[child]["_structural_lineage"]["activation_count"] == 1
    assert read_grammar_state(graph.nodes[R]).history == ("IL", "EN", "EN")
    assert read_grammar_state(graph.nodes[W]).history == ("IL", "RA")
    assert get_attr(graph.nodes[W], ALIAS_VF) == 9 / 8
    assert tuple(graph.neighbors(W)) == (R,) and graph.graph["_t"] == 0.25


@pytest.mark.parametrize(
    "gain,u,v,z1,z2,w1",
    [
        (Q(1, 16), Q(1, 4), Q(1, 2), Q(77, 256), Q(189, 512), Q(381, 1024)),
        (Q(1, 16), Q(1, 4), Q(3, 4), Q(77, 256), Q(221, 512), Q(413, 1024)),
        (Q(1, 16), Q(1, 2), Q(1, 2), Q(93, 256), Q(205, 512), Q(397, 1024)),
        (Q(1, 16), Q(1, 2), Q(3, 4), Q(93, 256), Q(237, 512), Q(429, 1024)),
        (Q(1, 32), Q(1, 4), Q(1, 2), Q(75, 256), Q(187, 512), Q(379, 1024)),
    ],
)
def test_actual_selected_member_emission_preserves_the_declared_port_word(
    gain, u, v, z1, z2, w1
):
    observed = {}
    for refined in (False, True):
        graph, group, child = _source(refined=refined)
        _realize_gain(graph, group, child, gain)
        _emit(graph, U, u - Q(1, 8))
        _emit(graph, V, v - Q(1, 8))
        held = _held_channels(graph)
        _receive(graph, group, (U,))
        _receive(graph, (B,), (V,))
        assert _mean(graph, group) == (Q(25, 64) + gain + u) / 2
        assert _form(graph, B) == (Q(3, 8) + v) / 2
        assert _held_channels(graph) == held
        if child is not None:
            assert (_form(graph, A) - _form(graph, child)) / 2 == (
                Q(35, 256) - gain
            ) / 2
        a_record = capture_retained_organization(graph, group)
        b_record = capture_retained_organization(graph, (B,))
        _output(graph, a_record)
        assert _form(graph, R) == z1
        first = _form(graph, R)
        _output(graph, b_record)
        assert _form(graph, R) == z2
        second = _form(graph, R)
        protected = {node: deepcopy(graph.nodes[node]) for node in (*group, B, R)}
        Resonance()(graph, W, validate_preconditions=True)
        assert _form(graph, W) == w1
        assert all(graph.nodes[node] == data for node, data in protected.items())
        _final_contexts(graph, child)
        observed[refined] = (first, second, _form(graph, W))
    assert observed[False] == observed[True] == (z1, z2, w1)


def test_uniform_broadcast_rejects_without_partial_child_activation():
    graph, group, child = _source(refined=True)
    assert group == (A, child)
    assert _form(graph, child) < Q(1, 2) < _form(graph, A)
    assert "_structural_lineage" not in graph.nodes[child]
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 1 / 16
    before = _checkpoint(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)) as caught:
        execute_pointwise_stage(
            graph,
            Emission(),
            (child, A),
            require_exact_operator=True,
            validate_preconditions=True,
        )
    assert "latent" in str(caught.value).lower()
    assert _checkpoint(graph) == before
    assert "_structural_lineage" not in graph.nodes[child]
    assert read_grammar_state(graph.nodes[child]).history == ("IL", "UM", "EN")
