"""Finite selected-pair AL/SHA runtime controls, without a flow trajectory.

The supplied doubled-C5 state is a static intervention fixture. These checks
reuse the actual shared stage, lifecycle and word admission owners; they do
not prove collective grammar descent, post-event trapping or event occurrence.
"""

from copy import deepcopy
from datetime import datetime, timezone

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.glyph_history import push_glyph
from tnfr.operators import network_stage
from tnfr.operators._mutation_gate import validate_mutation_runtime_gate
from tnfr.operators.definitions import Emission, Silence
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import (
    execute_network_operator_stage,
    run_network_sequence,
)

_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_BOOST = 1 / 16
_ATTENUATION = 1 / 2
_TARGETS = (0, 1)
_CORE = {"compatibility_profile": "core"}


@pytest.fixture
def supplied_stage_clock(monkeypatch):
    """Equal event inputs permit comparisons of complete nodal metadata."""

    class SuppliedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return _NOW if tz is None else _NOW.astimezone(tz)

    monkeypatch.setattr(network_stage, "datetime", SuppliedClock)


@pytest.fixture
def pair_source():
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        following = (block + 1) % 5
        for member in (2 * block, 2 * block + 1):
            for neighbor in (2 * following, 2 * following + 1):
                graph.add_edge(member, neighbor, weight=1, length=1)
    graph.graph.update(
        _t=7,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1,
        VF_BASAL_THRESHOLD=1 / 8,
        SHA_MIN_VF=1 / 8,
        EPI_MIN=-1,
        EPI_MAX=1,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"AL_boost": _BOOST, "SHA_vf_factor": _ATTENUATION},
    )
    for node in graph:
        block, member = divmod(node, 2)
        sign = 1 if member == 0 else -1
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 1 / 4 + block / 64 + sign / 32,
                ALIAS_THETA[0]: block / 32 + sign / 64,
                ALIAS_VF[0]: 1 + block / 4,
                ALIAS_DNFR[0]: sign / 16,
                "glyph_history": [],
            }
        )
    # Supplied asymmetric provenance must move with its constituent. This
    # metadata is not asserted to have arisen from an earlier trajectory.
    origin = "2029-01-01T00:00:00+00:00"
    graph.nodes[0].update(
        emission_timestamp=origin,
        _emission_activated=True,
        _emission_origin=origin,
        _structural_lineage={
            "origin": origin,
            "activation_count": 4,
            "derived_nodes": [],
            "parent_emission": None,
        },
    )
    return graph


def _physical(graph, node):
    return tuple(
        float(get_attr(graph.nodes[node], alias))
        for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR)
    )


def _word():
    operators = (Emission(), Silence())
    outcome = validate_sequence(
        [operator.name for operator in operators], context=_CORE
    )
    assert outcome.passed
    assert outcome.metadata["compatibility_profile"] == "core"
    return operators, ValidatedSequence(operators, context=_CORE)


def _execute_word(graph, targets=_TARGETS, *, certificates=False):
    operators, sequence = _word()
    return tuple(
        execute_network_operator_stage(
            graph,
            operator,
            targets,
            sequence_context=sequence.step(index),
            include_epi_jump_certificate=certificates,
        )
        for index, operator in enumerate(operators)
    )


def test_selected_pair_word_preserves_channels_and_correlated_lifecycle(
    pair_source, supplied_stage_clock
):
    graph = pair_source
    source = deepcopy(graph)
    operators, sequence = _word()
    result = execute_network_operator_stage(
        graph, operators[0], _TARGETS, sequence_context=sequence.step(0)
    )
    assert result.schedule == TWO_PHASE_JACOBI
    for node in _TARGETS:
        epi, theta, capacity, pressure = _physical(source, node)
        assert _physical(graph, node) == (epi + _BOOST, theta, capacity, pressure)
        assert tuple(graph.nodes[node]["glyph_history"]) == ("AL",)
        assert "latent" not in graph.nodes[node]

    result = execute_network_operator_stage(
        graph, operators[1], _TARGETS, sequence_context=sequence.step(1)
    )
    assert result.schedule == TWO_PHASE_JACOBI
    for node in _TARGETS:
        epi, theta, capacity, pressure = _physical(source, node)
        assert _physical(graph, node) == (
            epi + _BOOST,
            theta,
            _ATTENUATION * capacity,
            pressure,
        )
        data = graph.nodes[node]
        assert tuple(data["glyph_history"]) == ("AL", "SHA")
        assert data["_operator_step"] == 2
        assert data["latent"] is True
        assert data["latency_start_time"] == _NOW.isoformat()
        assert data["preserved_epi"] == epi + _BOOST
        assert data["silence_duration"] == 0
        assert data["was_initial_on_silence"] is False
    assert graph.nodes[0]["_structural_lineage"]["activation_count"] == 5
    assert graph.nodes[0]["emission_timestamp"] == source.nodes[0]["emission_timestamp"]
    assert graph.nodes[1]["_structural_lineage"]["activation_count"] == 1
    assert graph.nodes[1]["emission_timestamp"] == _NOW.isoformat()
    assert graph.graph["_t"] == source.graph["_t"]
    for node in set(graph) - set(_TARGETS):
        assert graph.nodes[node] == source.nodes[node]
    assert dict(graph.edges) == dict(source.edges)


def test_pair_swap_transports_complete_nodal_context_with_its_state(
    pair_source, supplied_stage_clock
):
    forward = pair_source
    swapped = deepcopy(forward)
    permutation = {node: 1 - node if node in _TARGETS else node for node in forward}
    for node in forward:
        swapped.nodes[node].clear()
        swapped.nodes[node].update(deepcopy(forward.nodes[permutation[node]]))
    _execute_word(forward)
    _execute_word(swapped, tuple(permutation[node] for node in _TARGETS))
    for node in forward:
        assert swapped.nodes[permutation[node]] == forward.nodes[node]
    assert swapped.graph["_t"] == forward.graph["_t"]
    assert dict(swapped.edges) == dict(forward.edges)


def test_pair_target_order_preserves_all_nodal_metadata(
    pair_source, supplied_stage_clock
):
    reverse = deepcopy(pair_source)
    forward_results = _execute_word(pair_source)
    reverse_results = _execute_word(reverse, tuple(reversed(_TARGETS)))
    assert forward_results == reverse_results
    assert dict(pair_source.nodes) == dict(reverse.nodes)
    assert dict(pair_source.edges) == dict(reverse.edges)


def test_default_core_and_explicit_legacy_reach_existing_word_executor(pair_source):
    source = deepcopy(pair_source)
    with pytest.raises(TNFRValueError, match="Invalid legacy sequence"):
        run_network_sequence(
            pair_source,
            ["emission", "silence"],
            cycles=0,
            context={"compatibility_profile": "legacy"},
        )
    # Zero cycles tests profile forwarding without turning the all-node runner
    # into a selected-pair executor or installing a different intervention.
    run_network_sequence(pair_source, ["emission", "silence"], cycles=0, context=_CORE)
    run_network_sequence(pair_source, ["emission", "silence"], cycles=0)
    assert pair_source.graph == source.graph
    assert dict(pair_source.nodes) == dict(source.nodes)
    assert dict(pair_source.edges) == dict(source.edges)


def test_selected_pair_does_not_inherit_all_node_epi_gain_certificate(
    pair_source, supplied_stage_clock
):
    results = _execute_word(pair_source, certificates=True)
    for result in results:
        certificate = result.epi_jump_certificate
        assert certificate is not None
        assert not certificate.runtime_epi_realization_certified
        assert not certificate.supports_hybrid_common_metric_bridge
        assert "all_graph_nodes_targeted_once" in (
            certificate.failed_runtime_epi_realization_conditions
        )
        assert result.schedule == TWO_PHASE_JACOBI
        assert result.nodes_processed == 2


def test_history_multiset_needs_its_constituent_state_association(pair_source):
    """Equal independent multisets lose evidence freshness under reassociation."""

    matched = pair_source
    matched.graph["ZHIR_THRESHOLD_XI"] = 1 / 8
    for node in _TARGETS:
        for glyph in ("AL", "IL", "OZ"):
            push_glyph(matched.nodes[node], glyph, window=3)
        matched.nodes[node]["epi_time_history"] = (
            (6, 0),
            (7, _physical(matched, node)[0]),
        )
    reassociated = deepcopy(matched)
    for node in _TARGETS:
        reassociated.nodes[node]["epi_time_history"] = matched.nodes[1 - node][
            "epi_time_history"
        ]
    assert tuple(_physical(matched, node) for node in matched) == tuple(
        _physical(reassociated, node) for node in reassociated
    )
    assert sorted(
        matched.nodes[node]["epi_time_history"] for node in _TARGETS
    ) == sorted(reassociated.nodes[node]["epi_time_history"] for node in _TARGETS)
    before = deepcopy((dict(matched.nodes), dict(reassociated.nodes)))
    for node in _TARGETS:
        assert validate_candidate(matched, node, "ZHIR").allowed
        assert validate_candidate(reassociated, node, "ZHIR").allowed
        gate = validate_mutation_runtime_gate(matched.nodes[node], matched.graph)
        assert gate.threshold.crossed
        assert gate.threshold.certificate.evidence.current_endpoint_matches_state
        with pytest.raises(OperatorPreconditionError, match="stale_physical_endpoint"):
            validate_mutation_runtime_gate(reassociated.nodes[node], reassociated.graph)
    assert (dict(matched.nodes), dict(reassociated.nodes)) == before
