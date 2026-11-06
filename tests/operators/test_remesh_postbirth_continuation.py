"""Actual post-birth delayed REMESH contribution to selected EN and RA.

The same advisory and sampling occasions are supplied to both arms. Delay is
a sample-index offset, with no elapsed time inferred from these two snapshots.
"""

from copy import deepcopy
from fractions import Fraction as Q

from tests.operators._remesh_postbirth_source import build_source, form
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.remesh_history import append_remesh_epi_history_snapshot
from tnfr.operators.emission import Emission
from tnfr.operators.event_remesh_runtime import execute_event_remesh_cycle
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.types import Glyph, require_finite_real_scalar_epi


def _history_vectors(graph):
    return tuple(
        tuple(Q.from_float(require_finite_real_scalar_epi(row[node])) for node in graph)
        for row in graph.graph["_epi_hist"]
    )


def _held_channels(graph):
    return tuple(
        tuple(
            get_attr(data, aliases) for aliases in (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA)
        )
        for _node, data in graph.nodes(data=True)
    )


def test_actual_delayed_remesh_changes_postbirth_en_ra_beyond_common_advisory():
    responses = {}
    delayed_histories = {}
    for arm in ("advisory", "delayed"):
        graph, parent, peer, child = build_source()
        assert "_epi_hist" not in graph.graph
        nodes = tuple(graph)
        first = append_remesh_epi_history_snapshot(graph)
        assert first.history_length_before == 0 and first.history_length_after == 1
        assert first.history_maxlen == 64
        h0 = (Q(25, 64), Q(3, 8), Q(15, 128))
        assert _history_vectors(graph) == (h0,)

        Emission()(graph, child, validate_preconditions=True)
        assert form(graph, child) == Q(31, 128)
        assert read_grammar_state(graph.nodes[child]).history == ("IL", "AL")
        assert graph.nodes[child]["_structural_lineage"]["activation_count"] == 1
        assert graph.graph["_t"] == 0.25
        assert _history_vectors(graph) == (h0,)
        h1 = (Q(25, 64), Q(3, 8), Q(31, 128))
        schedule = build_operator_event_schedule(
            ("recursivity",), start_time=0.25, flow_durations=(0.0, 0.0)
        )
        held = _held_channels(graph)
        edges = deepcopy(tuple(graph.edges(data=True)))
        counts = tuple(
            read_grammar_state(graph.nodes[node]).operator_step for node in nodes
        )
        if arm == "advisory":
            execution = execute_operator_event_schedule(graph, schedule)
            assert tuple(form(graph, node) for node in nodes) == h1
            appended = append_remesh_epi_history_snapshot(graph)
            assert appended.history_length_before == 1
            assert appended.history_length_after == 2
        else:
            cycle = execute_event_remesh_cycle(
                graph, schedule, refresh_pressure_after_remesh=False
            )
            execution = cycle.event_execution
            assert cycle.remesh_applied and cycle.remesh.status == "applied"
            assert cycle.history_length_before_cycle == 1
            assert cycle.history_length_after_append == 2
            transition = cycle.history_transition
            assert transition.incoming_exact_history == (h0,)
            assert transition.outgoing_exact_history == (h0, h1)
            assert transition.selected_local_delayed_epi == h0
            assert transition.selected_global_delayed_epi == h0
            assert cycle.pre_remesh_epi.epi_values == tuple(map(float, h1))
            assert cycle.post_remesh_epi.epi_values == (25 / 64, 3 / 8, 19 / 128)
            assert all(
                not item.clipping_intervened for item in cycle.remesh.plan.proposals
            )
            for node, value in zip(nodes, cycle.post_remesh_epi.epi_values):
                assert graph.nodes[node]["epi_time_history"][-1] == (0.25, value)
        assert execution.target_nodes == nodes
        assert execution.final_time == 0.25
        assert len(execution.events) == 1
        assert execution.events[0].glyph is Glyph.REMESH
        assert execution.events[0].nodes_processed == 3
        assert execution.pressure_refresh_callback_invocations == 0
        assert tuple(graph) == nodes
        assert tuple(graph.edges(data=True)) == edges
        assert _held_channels(graph) == held
        assert _history_vectors(graph) == (h0, h1)
        for node, count in zip(nodes, counts):
            grammar = read_grammar_state(graph.nodes[node])
            assert grammar.operator_step == count + 1
            assert grammar.history[-1] == "REMESH"
        expected_child = Q(31, 128) if arm == "advisory" else Q(19, 128)
        assert form(graph, child) == expected_child
        assert form(graph, parent) == Q(25, 64) and form(graph, peer) == Q(3, 8)

        child_before = deepcopy(graph.nodes[child])
        reception = execute_neighbor_stage(
            graph,
            Reception(),
            (parent,),
            source_nodes=(child,),
            track_sources=False,
            validate_preconditions=True,
        )
        (observed,) = reception.reception_observations
        expected_parent = Q(81, 256) if arm == "advisory" else Q(69, 256)
        assert observed.source_nodes == (child,) and observed.neighbors == (child,)
        assert observed.neighbor_epi_values == (float(expected_child),)
        assert observed.target_epi_before == 25 / 64
        assert observed.target_epi_after == float(expected_parent)
        assert form(graph, parent) == expected_parent
        Resonance()(graph, peer, validate_preconditions=True)
        expected_peer = Q(177, 512) if arm == "advisory" else Q(165, 512)
        assert form(graph, peer) == expected_peer
        assert get_attr(graph.nodes[peer], ALIAS_VF) == 9 / 8
        assert graph.nodes[child] == child_before
        assert tuple(get_attr(graph.nodes[n], ALIAS_DNFR) for n in nodes) == (
            9 / 128,
            1 / 8,
            0.0,
        )
        assert all(get_attr(graph.nodes[n], ALIAS_THETA) == 0.0 for n in nodes)
        for node, count, word in (
            (parent, 13, ("THOL", "UM", "IL", "IL", "IL", "REMESH", "EN")),
            (peer, 8, ("OZ", "AL", "AL", "IL", "IL", "REMESH", "RA")),
            (child, 3, ("IL", "AL", "REMESH")),
        ):
            grammar = read_grammar_state(graph.nodes[node])
            assert grammar.operator_step == count and grammar.history == word
            assert grammar.prior_coherence
        assert graph.graph["_t"] == 0.25 and tuple(graph) == nodes
        assert tuple(graph.edges(data=True)) == edges
        assert _history_vectors(graph) == (h0, h1)
        delayed_histories[arm] = _history_vectors(graph)
        responses[arm] = form(graph, peer)
    assert delayed_histories["advisory"] == delayed_histories["delayed"]
    assert responses["delayed"] - responses["advisory"] == -Q(3, 128)
