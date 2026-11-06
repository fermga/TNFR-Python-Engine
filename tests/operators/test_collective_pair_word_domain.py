"""Static admission controls for one declared collective AL/SHA word.

The sources, actions and comparison budget are fixed below before evaluating
the stages. Supplied histories are admission context, not past trajectories.
These controls exercise the complete selected-target runtime and one retained
sine-law barrier; they select no event and integrate no continuous flow.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import nextafter

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.glyph_history import push_glyph
from tnfr.operators import network_stage
from tnfr.operators.al_sha_stage_proposals import emission_epi_proposal
from tnfr.operators.definitions import Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_scale import assess_sine_replica_capacity
from tnfr.types import require_finite_real_scalar_epi

_PAIRS = tuple((2 * index, 2 * index + 1) for index in range(5))
_TARGETS = _PAIRS[0]
_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_CORE = {"compatibility_profile": "core"}
_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
_RADIUS = Q(1, 16)
_SOURCE_RADIUS = Q(1, 512)
_EXCESS_CEILING = Q(1, 16384)


@dataclass(frozen=True)
class _Preparation:
    common_form: Q
    internal_form: Q
    internal_phase: Q
    capacities: tuple[Q, ...]
    boost: Q
    attenuation: Q


# A finite family of prespecified probes, not a parameter search or proof that
# every point of a real neighborhood survives binary64 materialization.
_PREPARATIONS = (
    _Preparation(Q(5, 16), Q(1, 8192), Q(1, 32768), (Q(1),) * 5, Q(1, 1024), Q(1, 2)),
    _Preparation(
        Q(-1, 4),
        Q(-1, 8192),
        Q(1, 32768),
        (Q(1), Q(5, 4), Q(3, 2), Q(7, 4), Q(2)),
        Q(1, 2048),
        Q(3, 4),
    ),
    _Preparation(
        Q(3, 8),
        Q(1, 16384),
        Q(-1, 32768),
        (Q(3, 2), Q(1), Q(2), Q(5, 4), Q(7, 4)),
        Q(1, 1024),
        Q(1, 4),
    ),
    _Preparation(
        Q(1, 8),
        Q(0),
        Q(1, 65536),
        (Q(2), Q(3, 2), Q(1), Q(7, 4), Q(5, 4)),
        Q(1, 2048),
        Q(1, 2),
    ),
)


def _source(preparation=_PREPARATIONS[0], *, supplied_context=True):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for index, pair in enumerate(_PAIRS):
        graph.add_edges_from(
            (left, right, {"weight": 1, "length": 1})
            for left in pair
            for right in _PAIRS[(index + 1) % 5]
        )
        for sign, node in zip((1, -1), pair):
            graph.nodes[node].update(
                EPI=float(preparation.common_form + sign * preparation.internal_form),
                theta=float(
                    Q(674651885 * index, 536870912) + sign * preparation.internal_phase
                ),
                nu_f=float(preparation.capacities[index]),
                delta_nfr=sign / 16,
                glyph_history=[],
            )
    graph.graph.update(
        _t=7,
        RANDOM_SEED=17,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1,
        VF_BASAL_THRESHOLD=1 / 8,
        SHA_MIN_VF=1 / 8,
        MIN_NETWORK_DEGREE_COUPLING=1,
        EPI_MIN=-1,
        EPI_MAX=1,
        CLIP_MODE="hard",
        GLYPH_HYSTERESIS_WINDOW=8,
        GLYPH_FACTORS={
            "AL_boost": float(preparation.boost),
            "SHA_vf_factor": float(preparation.attenuation),
        },
    )
    if not supplied_context:
        return graph
    for glyph in ("OZ", "EN"):
        push_glyph(graph.nodes[0], glyph, window=8)
    push_glyph(graph.nodes[1], "EN", window=8)
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
    graph.nodes[1].update(
        latent=True,
        preserved_epi=graph.nodes[1]["EPI"],
        silence_duration=0,
        latency_start_time=origin,
        was_initial_on_silence=False,
    )
    return graph


def _frozen_word():
    operators = (Emission(), Silence())
    admission = validate_sequence([op.name for op in operators], context=_CORE)
    assert admission.passed
    assert admission.metadata["compatibility_profile"] == "core"
    return operators, ValidatedSequence(operators, context=_CORE)


def _stored_state(graph):
    return deepcopy((_graph_state(graph), dict(graph.nodes), dict(graph.edges)))


def _graph_state(graph):
    state = dict(graph.graph)
    if "_node_cache" in state:
        # NodeNX identity is a rebuildable adapter, not a physical coordinate.
        # Check its live graph/node association instead of comparing objects
        # created by two different executions or deep copies.
        adapters = state["_node_cache"]
        for node, adapter in adapters.items():
            assert adapter.G is graph and adapter.n == node
        state["_node_cache"] = tuple(sorted(adapters))
    return state


def _assess(graph):
    return assess_sine_replica_capacity(
        graph,
        reference_model=_MODEL,
        pairs=_PAIRS,
        radius=_RADIUS,
        excess_ceiling=_EXCESS_CEILING,
        form_mean_bounds=(Q(1, 4), Q(1, 2)),
    )


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


@pytest.fixture
def supplied_clock(monkeypatch):
    monkeypatch.setattr(network_stage, "datetime", _SuppliedClock)


@pytest.mark.parametrize("preparation", _PREPARATIONS)
def test_complete_word_descends_under_correlated_pair_swap(preparation, supplied_clock):
    source = _source(preparation)
    forward, swapped = deepcopy(source), deepcopy(source)
    permutation = {node: 1 - node if node in _TARGETS else node for node in source}
    for node in source:
        swapped.nodes[permutation[node]].clear()
        swapped.nodes[permutation[node]].update(deepcopy(source.nodes[node]))
    operators, sequence = _frozen_word()
    frozen_configuration = deepcopy(source.graph)
    # Freeze the action and both constituent histories before either response.
    assert source.nodes[0]["_grammar_u2_debt"] == 1
    assert source.nodes[1]["_grammar_u2_debt"] == 0
    for index, operator in enumerate(operators):
        left = execute_network_operator_stage(
            forward, operator, _TARGETS, sequence_context=sequence.step(index)
        )
        right = execute_network_operator_stage(
            swapped,
            operator,
            tuple(permutation[node] for node in _TARGETS),
            sequence_context=sequence.step(index),
        )
        assert left == right
        assert left.schedule == TWO_PHASE_JACOBI
        for node in source:
            assert forward.nodes[node] == swapped.nodes[permutation[node]]
        assert _graph_state(forward) == _graph_state(swapped)
        assert dict(forward.edges) == dict(swapped.edges) == dict(source.edges)
        for key, value in frozen_configuration.items():
            assert forward.graph[key] == value
        for node in set(source) - set(_TARGETS):
            assert forward.nodes[node] == source.nodes[node]

    for node in _TARGETS:
        before, after = source.nodes[node], forward.nodes[node]
        before_form = require_finite_real_scalar_epi(before["EPI"], "source form")
        after_form = require_finite_real_scalar_epi(after["EPI"], "response form")
        assert Q(after_form) - Q(before_form) == preparation.boost
        assert Q(after["nu_f"]) == preparation.attenuation * Q(before["nu_f"])
        assert after["theta"] == before["theta"]
        assert after["delta_nfr"] == before["delta_nfr"]
        assert tuple(after["glyph_history"]) == (*before["glyph_history"], "AL", "SHA")
        assert after["_grammar_u2_debt"] == before["_grammar_u2_debt"]
        assert after["_operator_step"] == before["_operator_step"] + 2
        assert after["latent"] is True
        assert after["preserved_epi"] == after_form
        assert after["latency_start_time"] == _NOW.isoformat()
    assert forward.nodes[0]["_structural_lineage"]["activation_count"] == 5
    assert forward.nodes[1]["_structural_lineage"]["activation_count"] == 1
    assert (
        forward.nodes[0]["emission_timestamp"] == source.nodes[0]["emission_timestamp"]
    )
    assert forward.nodes[1]["emission_timestamp"] == _NOW.isoformat()


@pytest.mark.parametrize("failed_member", _TARGETS)
@pytest.mark.parametrize("condition", ("debt", "latent", "basal"))
def test_invalid_first_prefix_rejects_the_whole_pair_without_writes(
    failed_member, condition, supplied_clock
):
    graph = _source()
    if condition == "debt":
        graph.nodes[failed_member]["_grammar_u2_debt"] = 3
        error_type, message = StructuralGrammarError, "Convergence violation"
    elif condition == "latent":
        graph.nodes[failed_member]["EPI"] = graph.graph["EPI_LATENT_MAX"]
        error_type, message = TNFRValueError, "AL precondition failed: EPI"
    else:
        graph.nodes[failed_member]["nu_f"] = graph.graph["VF_BASAL_THRESHOLD"] / 2
        error_type, message = TNFRValueError, "AL precondition failed: νf"
    operators, sequence = _frozen_word()
    before = _stored_state(graph)
    with pytest.raises(error_type, match=message) as failure:
        execute_network_operator_stage(
            graph, operators[0], _TARGETS, sequence_context=sequence.step(0)
        )
    if condition == "debt":
        assert failure.value.rule == "U2"
    assert _stored_state(graph) == before


def test_late_silence_failure_retains_exactly_the_committed_emission_prefix(
    supplied_clock,
):
    graph = _source()
    # This higher SHA threshold is part of preparation, not a mid-word change.
    graph.graph["SHA_MIN_VF"] = 2
    operators, sequence = _frozen_word()
    source = _stored_state(graph)
    execute_network_operator_stage(
        graph, operators[0], _TARGETS, sequence_context=sequence.step(0)
    )
    prefix = _stored_state(graph)
    assert prefix != source
    with pytest.raises(OperatorPreconditionError, match="already minimal"):
        execute_network_operator_stage(
            graph, operators[1], _TARGETS, sequence_context=sequence.step(1)
        )
    assert _stored_state(graph) == prefix
    for node in _TARGETS:
        assert tuple(graph.nodes[node]["glyph_history"])[-1] == "AL"
        assert graph.nodes[node]["nu_f"] == 1
        assert "latent" not in graph.nodes[node]


def test_common_positive_boost_need_not_give_equal_represented_increments():
    graph_attributes = {"EPI_MIN": -1, "EPI_MAX": 1, "CLIP_MODE": "hard"}
    boost = 2.0**-55
    source = (nextafter(0.25, 0.0), 0.25)
    proposals = tuple(emission_epi_proposal(graph_attributes, x, boost) for x in source)
    increments = tuple(Q(after) - Q(before) for before, after in proposals)
    # At 1/4 the boost is a half-ULP tie; the preceding float reaches 1/4
    # exactly. Both calls are valid AL maps, with no clipping or projection.
    assert increments == (Q(1, 2**55), Q(0))
    assert increments[0] != increments[1]
    before_internal = (Q(source[0]) - Q(source[1])) ** 2 / 4
    after_internal = (Q(proposals[0][1]) - Q(proposals[1][1])) ** 2 / 4
    assert before_internal > 0 and after_internal == 0


@pytest.fixture(scope="module")
def admitted_word_barrier():
    graph = _source(supplied_context=False)
    source = deepcopy(graph)
    operators, sequence = _frozen_word()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            execute_network_operator_stage(
                graph, operator, _TARGETS, sequence_context=sequence.step(index)
            )
    return source, graph, _assess(source), _assess(graph)


def test_actual_complete_word_reuses_the_declared_sine_identity_barrier(
    admitted_word_barrier,
):
    source, graph, before, after = admitted_word_barrier
    assert before.family.norm_squared_upper_bound < _SOURCE_RADIUS**2
    assert before.family.source_set_trapping_certified
    assert after.family.source_set_trapping_certified
    assert after.family.source_family_membership == "certified_inside"
    assert after.pair_capacity_means == (Q(1, 2), Q(1), Q(1), Q(1), Q(1))
    assert after.pair_capacity_half_differences == (Q(0),) * 5
    assert after.weighted_form_mean == Q(5, 16) + Q(1, 3072)
    assert after.weighted_mean_rate_residual_bounds.contains(0)
    for row in after.capacity_aware_rate_residuals:
        assert all(residual.contains(0) for residual in row)
    for node in _TARGETS:
        # The stage did not install or refresh the continuous pressure law.
        assert graph.nodes[node]["delta_nfr"] == source.nodes[node]["delta_nfr"]
        assert tuple(graph.nodes[node]["glyph_history"])[-2:] == ("AL", "SHA")
    assert after.all_time_internal_activity_status == "not_certified_by_this_reader"
