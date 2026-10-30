"""Collective flow coordinates do not determine registered operator admission.

These are read-only domain witnesses on the exact doubled-C5 pair state. The
supplied histories are runtime context, not a reconstructed past trajectory.
Neither grammar shadow validation nor a fresh measured secant executes an
operator or proves descent of its complete state transition.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.glyph_history import push_glyph
from tnfr.operators._mutation_gate import (
    mutation_threshold_sample,
    validate_mutation_runtime_gate,
)
from tnfr.operators.grammar_debt import PRIOR_COHERENCE_KEY, U2_DEBT_KEY
from tnfr.operators.grammar_dynamics import (
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.relational_sine_pair import derive_sine_global_pair_state


@pytest.fixture(scope="module")
def collective_source():
    """One exactly realizable source; each consumer receives its own copy."""
    graph = nx.Graph()
    for block in range(5):
        following = (block + 1) % 5
        for member in (2 * block, 2 * block + 1):
            for neighbor in (2 * following, 2 * following + 1):
                graph.add_edge(member, neighbor, weight=1)
    graph.graph.update(_t=1, ZHIR_THRESHOLD_XI=Q(1, 8))
    for node in graph:
        graph.nodes[node].update(
            EPI=Q(1, 4), theta=0, nu_f=1, delta_nfr=0, glyph_history=[]
        )
    return graph, _collective_state(graph)


def _collective_state(graph):
    # This control deliberately uses the exact zero-phase stratum. The phasors
    # are its exact circular representation, not a mean-angle replacement.
    assert all(graph.nodes[node]["theta"] == 0 for node in range(10))
    return derive_sine_global_pair_state(
        tuple(graph.nodes[node]["EPI"] for node in range(10)),
        ((Q(1), Q(0)),) * 10,
    )


def _physical_state(graph):
    return (
        tuple(
            tuple(
                graph.nodes[node][field]
                for field in ("EPI", "theta", "nu_f", "delta_nfr")
            )
            for node in range(10)
        ),
        tuple(
            sorted(
                (min(a, b), max(a, b), data["weight"])
                for a, b, data in graph.edges(data=True)
            )
        ),
    )


def _context(source, tokens, *, window):
    graph = deepcopy(source)
    for token in tokens:
        # Use the shared bookkeeping owner, without claiming that recording
        # these supplied tokens executes their associated physical maps.
        push_glyph(graph.nodes[0], token, window)
    return graph


def _snapshot(graph):
    return deepcopy(
        (graph.graph, dict(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


def test_same_pair_state_and_compact_history_can_have_different_u2_admission(
    collective_source,
):
    source, pair_state = collective_source
    unpaid = _context(source, ("AL", "IL", "VAL", "VAL", "EN"), window=1)
    paid = _context(source, ("AL", "IL", "VAL", "IL", "EN"), window=1)
    assert _physical_state(unpaid) == _physical_state(paid)
    assert _collective_state(unpaid) == _collective_state(paid) == pair_state
    assert unpaid.nodes[0]["glyph_history"] == paid.nodes[0]["glyph_history"]
    assert tuple(unpaid.nodes[0]["glyph_history"]) == ("EN",)
    assert (
        unpaid.nodes[0][PRIOR_COHERENCE_KEY]
        is paid.nodes[0][PRIOR_COHERENCE_KEY]
        is True
    )
    assert (unpaid.nodes[0][U2_DEBT_KEY], paid.nodes[0][U2_DEBT_KEY]) == (2, 0)
    before = (_snapshot(unpaid), _snapshot(paid))

    blocked = validate_candidate(unpaid, 0, "VAL")
    admitted = validate_candidate(paid, 0, "VAL")

    assert not blocked.allowed
    assert {violation.rule for violation in blocked.violations} == {"U2"}
    assert admitted.allowed
    assert (_snapshot(unpaid), _snapshot(paid)) == before


def test_same_pair_state_debt_and_recent_history_can_omit_prior_coherence(
    collective_source,
):
    source, pair_state = collective_source
    stable = _context(source, ("AL", "IL", "VAL", "THOL", "OZ"), window=2)
    missing = _context(source, ("AL", "EN", "VAL", "THOL", "OZ"), window=2)
    assert _physical_state(stable) == _physical_state(missing)
    assert _collective_state(stable) == _collective_state(missing) == pair_state
    assert stable.nodes[0]["glyph_history"] == missing.nodes[0]["glyph_history"]
    assert tuple(stable.nodes[0]["glyph_history"]) == ("THOL", "OZ")
    assert stable.nodes[0][U2_DEBT_KEY] == missing.nodes[0][U2_DEBT_KEY] == 1
    assert stable.nodes[0][PRIOR_COHERENCE_KEY] is True
    assert missing.nodes[0][PRIOR_COHERENCE_KEY] is False
    before = (_snapshot(stable), _snapshot(missing))

    admitted = validate_candidate(stable, 0, "ZHIR")
    blocked = validate_candidate(missing, 0, "ZHIR")

    assert admitted.allowed
    assert not blocked.allowed
    assert {violation.rule for violation in blocked.violations} == {"U4b"}
    assert (_snapshot(stable), _snapshot(missing)) == before


def test_same_collective_state_and_grammar_need_independent_measured_secants(
    collective_source,
):
    source, pair_state = collective_source
    growing = _context(source, ("AL", "IL", "OZ"), window=3)
    flat = deepcopy(growing)
    growing.nodes[0]["epi_time_history"] = [(0, 0), (1, Q(1, 4))]
    flat.nodes[0]["epi_time_history"] = [(0, Q(1, 4)), (1, Q(1, 4))]
    assert _physical_state(growing) == _physical_state(flat)
    assert _collective_state(growing) == _collective_state(flat) == pair_state
    before = (_snapshot(growing), _snapshot(flat))

    for graph in (growing, flat):
        assert validate_candidate(graph, 0, "ZHIR").allowed
        sample = mutation_threshold_sample(graph.nodes[0], graph.graph)
        assert sample.physical_time_resolved
        assert sample.certificate.evidence_valid
        assert sample.certificate.evidence.current_endpoint_matches_state
        assert sample.sample_interval == 1
        # The current continuous law predicts zero at this source. It cannot
        # replace either supplied observation over the preceding interval.
        assert sample.certificate.predicted_depi_dt == 0
    accepted = validate_mutation_runtime_gate(growing.nodes[0], growing.graph)
    assert accepted.threshold.depi_dt == Q(1, 4)
    assert accepted.threshold.xi == Q(1, 8)
    assert accepted.threshold.crossed
    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        validate_mutation_runtime_gate(flat.nodes[0], flat.graph)
    assert (_snapshot(growing), _snapshot(flat)) == before


def test_ordered_prefix_admission_does_not_bank_later_stabilization(
    collective_source,
):
    source, pair_state = collective_source
    graph = _context(source, ("AL", "IL", "VAL", "VAL"), window=2)
    before = _snapshot(graph)

    repair_first = validate_sequence_incremental(graph, 0, ("IL", "VAL"))
    repair_later = validate_sequence_incremental(graph, 0, ("VAL", "IL"))

    assert tuple(item.allowed for item in repair_first) == (True, True)
    assert tuple(item.allowed for item in repair_later) == (False, True)
    assert {violation.rule for violation in repair_later[0].violations} == {"U2"}
    assert _snapshot(graph) == before
    assert _collective_state(graph) == pair_state
    # Shadow admission updates context only. It has not changed capacity,
    # pressure, support or any later live-state precondition.
    assert _physical_state(graph) == _physical_state(source)


def test_al_sha_word_uses_core_admission_not_runtime_descent():
    word = ("emission", "silence")
    canonical = validate_sequence(word, compatibility_profile="core")
    legacy = validate_sequence(word, compatibility_profile="legacy")

    assert canonical.passed
    assert validate_sequence(word) == canonical
    assert canonical.metadata["canonical_word_checked"]
    assert not canonical.metadata["diagnostic_probe"]
    assert not legacy.passed
    assert legacy.metadata["compatibility_profile"] == "legacy"
