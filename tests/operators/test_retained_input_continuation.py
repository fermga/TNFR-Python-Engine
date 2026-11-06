"""Actual addressed EN and RA distinguish retained interior from live input.

The four arms are declared in the prospective retained-input-port protocol.
Initial organization, address and action occasions are supplied; these finite
controls neither delete constituents nor infer an autonomous occurrence law.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.coherence import Coherence
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.operators.retained_reception import (
    execute_retained_reception_stage,
    retained_input_ports,
)
from tnfr.operators.retained_synthesis import (
    capture_retained_pair,
    execute_retained_synthesis_stage,
)
from tnfr.types import require_finite_real_scalar_epi

CORE, CHILD, DONOR, PEER = "core", "child", "donor", "peer"
MEMBERS = (CORE, CHILD)


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
    )


def _edges(graph):
    return tuple(
        (left, right, deepcopy(data)) for left, right, data in graph.edges(data=True)
    )


def _source(initial_core, initial_child):
    graph = nx.DiGraph()
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=71,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        NUL_scale=0.5,
        EN_mix=0.5,
        AL_boost=0.125,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    # The injected reciprocal would contradict this new scale and reject it.
    graph.graph["GLYPH_FACTORS"].pop("NUL_densification_factor")
    for node, form in (
        (CORE, initial_core),
        (CHILD, initial_child),
        (DONOR, Q(1, 4)),
        (PEER, Q(3, 8)),
    ):
        graph.add_node(
            node,
            EPI=float(form),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5 if node == DONOR else 0.0,
            Si=0.0,
            epi_kind="form",
        )
        assert read_grammar_state(graph.nodes[node]).history == ()
    for left, right in (
        (CORE, CHILD),
        (CHILD, CORE),
        (DONOR, CHILD),
        (PEER, CORE),
    ):
        graph.add_edge(left, right, weight=1.0)
    execute_pointwise_stage(
        graph,
        Coherence(),
        (CORE, CHILD, PEER),
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    for node in (CORE, CHILD, PEER):
        assert read_grammar_state(graph.nodes[node]).history == ("IL",)
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
    pair = capture_retained_pair(graph, MEMBERS)
    source = execute_retained_synthesis_stage(
        graph,
        pair,
        contraction_kwargs={"validate_preconditions": True},
        reception_kwargs={"validate_preconditions": True},
    )
    assert source.contraction_result.stage_result.nodes_processed == 2
    observed = source.reception_result.reception_observations[0]
    assert observed.node == CORE and observed.source_nodes == (CHILD,)
    assert not observed.source_tracking_enabled
    assert _form(graph, CORE) == Q(1, 4)
    assert read_grammar_state(graph.nodes[CORE]).history == ("IL", "NUL", "EN")
    assert read_grammar_state(graph.nodes[CHILD]).history == ("IL", "NUL")
    for node in MEMBERS:
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.5
    return graph


def test_addressed_input_preserves_interior_and_reads_actual_donor_al_live():
    """One fixed four-arm contract separates the two causal contributions."""
    outcomes = {}
    for interior, initial_core, initial_child, child_before, emitted, expected in (
        ("A", Q(3, 4), Q(1, 4), Q(1, 8), False, (Q(3, 16), Q(7, 32), Q(19, 64))),
        ("B", Q(1, 2), Q(1, 2), Q(1, 4), False, (Q(1, 4), Q(1, 4), Q(5, 16))),
        ("A", Q(3, 4), Q(1, 4), Q(1, 8), True, (Q(1, 4), Q(1, 4), Q(5, 16))),
        ("B", Q(1, 2), Q(1, 2), Q(1, 4), True, (Q(5, 16), Q(9, 32), Q(21, 64))),
    ):
        graph = _source(initial_core, initial_child)
        assert _form(graph, CHILD) == child_before
        assert _form(graph, DONOR) == Q(1, 4)
        assert _form(graph, PEER) == Q(3, 8)
        support = _edges(graph)
        record = capture_retained_organization(graph, MEMBERS)
        assert tuple(
            (port.ordinal, port.member, port.source)
            for port in retained_input_ports(record)
        ) == ((0, CORE, PEER), (1, CHILD, DONOR))
        retained_before = {node: deepcopy(graph.nodes[node]) for node in MEMBERS}
        if emitted:
            Emission()(graph, DONOR, validate_preconditions=True)
            assert graph.nodes[DONOR]["_structural_lineage"]["activation_count"] == 1
        else:
            assert "_structural_lineage" not in graph.nodes[DONOR]
        donor_form = Q(3, 8) if emitted else Q(1, 4)
        assert _form(graph, DONOR) == donor_form
        assert all(graph.nodes[node] == retained_before[node] for node in MEMBERS)
        assert retained_organization_matches_graph(graph, record)
        assert graph.graph["_t"] == 0.0

        nonreceiver = {
            node: deepcopy(graph.nodes[node]) for node in (CORE, DONOR, PEER)
        }
        dispatched = execute_retained_reception_stage(
            graph,
            record,
            1,
            execution_kwargs={"validate_preconditions": True},
        )
        assert (
            dispatched.port.ordinal,
            dispatched.port.member,
            dispatched.port.source,
        ) == (1, CHILD, DONOR)
        assert dispatched.before.members == dispatched.after.members == MEMBERS
        assert dispatched.stage_result.nodes_processed == 1
        (observation,) = dispatched.stage_result.reception_observations
        assert observation.node == CHILD
        assert observation.source_nodes == observation.neighbors == (DONOR,)
        assert observation.source_selection_origin == "explicit_request"
        assert not observation.source_tracking_enabled
        assert observation.target_epi_before == float(child_before)
        assert observation.target_epi_after == float(expected[0])
        assert observation.neighbor_epi_values == (float(donor_form),)
        assert _form(graph, CHILD) == expected[0]
        assert all(graph.nodes[node] == old for node, old in nonreceiver.items())
        assert get_attr(graph.nodes[CHILD], ALIAS_DNFR) == 0.0
        assert get_attr(graph.nodes[CHILD], ALIAS_VF) == 0.5
        assert retained_organization_matches_graph(graph, dispatched.after)
        assert not retained_organization_matches_graph(graph, record)
        assert read_grammar_state(graph.nodes[CHILD]).history == ("IL", "NUL", "EN")

        child_after = deepcopy(graph.nodes[CHILD])
        donor_after = deepcopy(graph.nodes[DONOR])
        Resonance()(graph, CORE, validate_preconditions=True)
        assert _form(graph, CORE) == expected[1]
        assert _form(graph, PEER) == Q(3, 8)
        Resonance()(graph, PEER, validate_preconditions=True)
        assert _form(graph, PEER) == expected[2]
        assert graph.nodes[CHILD] == child_after
        assert graph.nodes[DONOR] == donor_after
        assert get_attr(graph.nodes[CORE], ALIAS_VF) == 9 / 16
        assert get_attr(graph.nodes[PEER], ALIAS_VF) == 9 / 8
        assert get_attr(graph.nodes[CHILD], ALIAS_VF) == 0.5
        assert get_attr(graph.nodes[DONOR], ALIAS_VF) == 1.0
        assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in graph)
        assert all(
            get_attr(graph.nodes[node], ALIAS_DNFR) == (0.5 if node == DONOR else 0.0)
            for node in graph
        )
        for node, word in (
            (CORE, ("IL", "NUL", "EN", "RA")),
            (CHILD, ("IL", "NUL", "EN")),
            (PEER, ("IL", "RA")),
            (DONOR, ("AL",) if emitted else ()),
        ):
            grammar = read_grammar_state(graph.nodes[node])
            assert grammar.history == word and grammar.operator_step == len(word)
            assert grammar.prior_coherence == (node != DONOR)
        assert tuple(graph) == (CORE, CHILD, DONOR, PEER)
        assert _edges(graph) == support and graph.graph["_t"] == 0.0
        assert all("epi_time_history" not in graph.nodes[node] for node in graph)
        outcomes[interior, emitted] = _form(graph, PEER)

    assert outcomes["B", False] - outcomes["A", False] == Q(1, 64)
    assert outcomes["B", True] - outcomes["A", True] == Q(1, 64)
    assert outcomes["A", True] - outcomes["A", False] == Q(1, 64)
    assert outcomes["B", True] - outcomes["B", False] == Q(1, 64)
