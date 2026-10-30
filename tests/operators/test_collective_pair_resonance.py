"""Static full-word Resonance controls on the declared doubled-C5 pair.

The source, word, factors, clock and observation budget below precede response
evaluation. These controls install no continuous flow or endogenous selector.
Exact real collective formulas and represented runtime observations retain
separate scopes, including the phase reduction's finite arithmetic.
"""

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import cos, pi

import networkx as nx
import pytest

from tnfr.alias import get_attr_str
from tnfr.constants.aliases import ALIAS_EPI_KIND
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.glyph_history import push_glyph
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._rational_interval import cos as interval_cos
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.operators import network_stage
from tnfr.operators.definitions import Emission, Resonance, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_scale import assess_sine_replica_capacity
from tnfr.types import require_finite_real_scalar_epi

_PAIRS = tuple((2 * index, 2 * index + 1) for index in range(5))
_TARGETS = _PAIRS[0]
_CORE = {"compatibility_profile": "core"}
_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
_FORM, _INTERNAL_FORM = Q(5, 16), Q(1, 8192)
_PHASE_OFFSET, _INTERNAL_PHASE = Q(1, 16), Q(1, 32768)
_TWIST = Q(674651885, 536870912)
_BOOST, _MIX, _PHASE_COUPLING = Q(1, 1024), Q(1, 8), Q(1, 16)
_AMPLIFICATION, _ATTENUATION = Q(1, 32), Q(1, 2)
_RADIUS, _SOURCE_RADIUS = Q(1, 16), Q(1, 512)
_EXCESS_CEILING = Q(1, 16384)


def _source(*, wider_left_neighbor=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for index, pair in enumerate(_PAIRS):
        graph.add_edges_from(
            (left, right, {"weight": 1, "length": 1})
            for left in pair
            for right in _PAIRS[(index + 1) % 5]
        )
        width = _INTERNAL_PHASE * (3 if wider_left_neighbor and index == 4 else 1)
        for sign, node in zip((1, -1), pair):
            graph.nodes[node].update(
                EPI=float(_FORM + sign * _INTERNAL_FORM),
                theta=float(_PHASE_OFFSET + index * _TWIST + sign * width),
                nu_f=1.0,
                # Supplied policy input; the detached law reader reevaluates
                # its own pressure rather than interpreting this as its law.
                delta_nfr=sign / 16,
                glyph_history=[],
            )
            graph.nodes[node][ALIAS_EPI_KIND[0]] = "pair-pattern"
    graph.graph.update(
        _t=7,
        RANDOM_SEED=17,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1,
        VF_BASAL_THRESHOLD=1 / 8,
        SHA_MIN_VF=1 / 8,
        RA_MIN_SOURCE_EPI=1 / 8,
        RA_MIN_VF=1 / 8,
        RA_MAX_DISSONANCE=1 / 2,
        RA_MAX_PHASE_DIFF=pi / 2,
        DELTA_PHI_MAX=pi / 2,
        MIN_NETWORK_DEGREE_COUPLING=1,
        EPI_MIN=-1,
        EPI_MAX=1,
        CLIP_MODE="hard",
        GLYPH_HYSTERESIS_WINDOW=8,
        GLYPH_FACTORS={
            "AL_boost": float(_BOOST),
            "RA_epi_diff": float(_MIX),
            "RA_phase_coupling": float(_PHASE_COUPLING),
            "RA_vf_amplification": float(_AMPLIFICATION),
            "SHA_vf_factor": float(_ATTENUATION),
        },
    )
    return graph


def _word():
    operators = (Emission(), Resonance(), Silence())
    admission = validate_sequence(
        [operator.name for operator in operators], context=_CORE
    )
    assert admission.passed
    assert admission.metadata["compatibility_profile"] == "core"
    return operators, ValidatedSequence(operators, context=_CORE)


def _execute(graph, targets=_TARGETS):
    operators, sequence = _word()
    return tuple(
        execute_network_operator_stage(
            graph, operator, targets, sequence_context=sequence.step(index)
        )
        for index, operator in enumerate(operators)
    )


def _graph_state(graph):
    state = dict(graph.graph)
    if "_node_cache" in state:
        for node, adapter in state["_node_cache"].items():
            assert adapter.G is graph and adapter.n == node
        state["_node_cache"] = tuple(sorted(state["_node_cache"]))
    if "_vfmax_node" in state:
        # A tied argmax witness is rebuildable implementation state. Validate
        # its meaning without making its selected label part of pair identity.
        maximum = max(abs(data["nu_f"]) for _, data in graph.nodes(data=True))
        assert state["_vfmax"] == maximum
        assert abs(graph.nodes[state["_vfmax_node"]]["nu_f"]) == maximum
        state["_vfmax_node"] = "valid_argmax_witness"
    return state


def _stored_state(graph):
    return deepcopy(
        (
            _graph_state(graph),
            dict(graph.nodes),
            dict(graph.edges),
            getattr(graph, "_last_operator_applied", None),
        )
    )


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


@pytest.fixture
def supplied_clock(monkeypatch):
    monkeypatch.setattr(network_stage, "datetime", _SuppliedClock)


def test_complete_word_matches_independent_collective_rows_and_context(supplied_clock):
    graph = _source()
    source = deepcopy(graph)
    operators, sequence = _word()
    # Equal-width neighbor pairs have resultant direction equal to the midpoint
    # of their center lifts about the target: gamma + 5*twist/2 - pi. Rational
    # pi bounds test the real formula independently of libm's phasor reduction.
    resultant_angle = I(_PHASE_OFFSET + 5 * _TWIST / 2) - pi_interval()
    for index, operator in enumerate(operators):
        result = execute_network_operator_stage(
            graph, operator, _TARGETS, sequence_context=sequence.step(index)
        )
        assert result.schedule == TWO_PHASE_JACOBI
        assert result.nodes_processed == 2
        assert result.epi_jump_certificate is None
        assert result.neighbor_epi_jump_certificate is None
        assert graph.graph["_t"] == 7
        assert graph.graph.get("_trig_version", 0) == (0 if index == 0 else 2)
        assert graph._last_operator_applied == operator.name
        assert "recognized_coherence_patterns" not in graph.graph
        assert dict(graph.edges) == dict(source.edges)
        for node in set(graph) - set(_TARGETS):
            assert graph.nodes[node] == source.nodes[node]
        for sign, node in zip((1, -1), _TARGETS):
            data = graph.nodes[node]
            expected_form = _FORM + _BOOST + sign * _INTERNAL_FORM
            expected_capacity = Q(1)
            expected_phase = I(Q(source.nodes[node]["theta"]))
            if index >= 1:
                expected_form = _FORM + (1 - _MIX) * (_BOOST + sign * _INTERNAL_FORM)
                expected_capacity *= 1 + _AMPLIFICATION
                expected_phase += _PHASE_COUPLING * (resultant_angle - expected_phase)
            if index == 2:
                expected_capacity *= _ATTENUATION
            actual_form = require_finite_real_scalar_epi(data["EPI"], "word response")
            assert Q(actual_form) == expected_form
            assert Q(data["nu_f"]) == expected_capacity
            # The exact real phase formula is checked to finite arithmetic
            # accuracy; it is not asserted to equal the rounded formula.
            phase_error = Q(data["theta"]) - expected_phase
            assert max(abs(phase_error.lo), abs(phase_error.hi)) < Q(1, 2**48)
            if index >= 1:
                # A small, explicitly bounded represented defect remains;
                # passing this control does not turn libm into exact dynamics.
                assert not phase_error.contains(0)
                assert data["phase"] == data["theta"]
            assert data["delta_nfr"] == source.nodes[node]["delta_nfr"]
            assert get_attr_str(data, ALIAS_EPI_KIND) == "pair-pattern"
            assert tuple(data["glyph_history"]) == ("AL", "RA", "SHA")[: index + 1]
            assert data["_operator_step"] == index + 1
            assert data["_grammar_u2_debt"] == 0
            assert data["_structural_lineage"]["activation_count"] == 1
            assert data["emission_timestamp"] == _NOW.isoformat()
            if index < 2:
                assert "latent" not in data
            else:
                assert data["latent"] is True
                assert data["preserved_epi"] == actual_form
                assert data["latency_start_time"] == _NOW.isoformat()
                assert data["silence_duration"] == 0
                assert data["was_initial_on_silence"] is False
        _graph_state(graph)


@pytest.mark.parametrize("swap_constituents", (False, True))
def test_target_pair_permutation_transports_complete_associated_context(
    swap_constituents, supplied_clock
):
    source = _source()
    origin = "2029-01-01T00:00:00+00:00"
    source.nodes[0].update(
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
    for _ in range(4):
        push_glyph(source.nodes[0], "AL", window=8)
    assert source.nodes[0]["_operator_step"] == 4
    assert source.nodes[0]["_grammar_u2_debt"] == 0
    forward, reverse = deepcopy(source), deepcopy(source)
    permutation = {
        node: 1 - node if swap_constituents and node in _TARGETS else node
        for node in source
    }
    for node in source:
        reverse.nodes[permutation[node]].clear()
        reverse.nodes[permutation[node]].update(deepcopy(source.nodes[node]))
    left = _execute(forward)
    right = _execute(reverse, tuple(reversed(_TARGETS)))
    assert left == right
    for node in source:
        assert forward.nodes[node] == reverse.nodes[permutation[node]]
    assert _graph_state(forward) == _graph_state(reverse)
    assert dict(forward.edges) == dict(reverse.edges)
    assert forward._last_operator_applied == reverse._last_operator_applied


def test_neighbor_internal_phase_is_required_beyond_collective_means(supplied_clock):
    regular, wider = _source(), _source(wider_left_neighbor=True)
    for pair in _PAIRS:
        for field in ("EPI", "theta"):
            assert sum(Q(regular.nodes[node][field]) for node in pair) == sum(
                Q(wider.nodes[node][field]) for node in pair
            )
    assert cos(float(3 * _INTERNAL_PHASE)) < cos(float(_INTERNAL_PHASE))
    _execute(regular)
    _execute(wider)
    # Reducing the left neighbor's resultant magnitude shifts the aggregate
    # direction toward the right neighbor; equal means alone miss that shift.
    for node in _TARGETS:
        assert wider.nodes[node]["theta"] > regular.nodes[node]["theta"]
        for field in ("EPI", "nu_f", "delta_nfr"):
            assert wider.nodes[node][field] == regular.nodes[node][field]


def _fine_storage(graph):
    forms = {
        node: Q(require_finite_real_scalar_epi(data["EPI"], "storage form"))
        for node, data in graph.nodes(data=True)
    }
    phases = {node: Q(data["theta"]) for node, data in graph.nodes(data=True)}
    form_storage = sum(
        ((forms[right] - forms[left]) ** 2 / 2 for left, right in graph.edges), Q(0)
    )
    phase_storage = sum(
        (
            1 - interval_cos(I(phases[right] - phases[left]))
            for left, right in graph.edges
        ),
        I(0),
    )
    return forms, form_storage, phase_storage


def test_resonance_work_matches_graph_algebra_and_does_not_erase_emission_work(
    supplied_clock,
):
    graph = _source()
    operators, sequence = _word()
    source = _fine_storage(graph)
    states = []
    for index, operator in enumerate(operators):
        execute_network_operator_stage(
            graph, operator, _TARGETS, sequence_context=sequence.step(index)
        )
        states.append(_fine_storage(graph))
    (forms, emitted_form, emitted_phase), resonated, silenced = states
    fine_laplacian_row = {
        node: sum(forms[node] - forms[neighbor] for neighbor in graph.neighbors(node))
        for node in _TARGETS
    }
    expected_form_work = (
        -_MIX * (2 - _MIX) / 8 * sum(value**2 for value in fine_laplacian_row.values())
    )
    observed_form_work = resonated[1] - emitted_form
    assert observed_form_work == expected_form_work < 0

    # The exact-real collective phase formula uses the same fixed represented
    # source phases as the graph observation; no runtime reducer is reused.
    mean = I(_PHASE_OFFSET + 5 * _TWIST / 2) - pi_interval()
    magnitude = -interval_cos(I(_INTERNAL_PHASE)) * interval_cos(I(3 * _TWIST / 2))
    expected_phase_work = (
        4
        * magnitude
        * sum(
            (
                interval_cos(I(_PHASE_OFFSET + sign * _INTERNAL_PHASE) - mean)
                - interval_cos(
                    (1 - _PHASE_COUPLING)
                    * (I(_PHASE_OFFSET + sign * _INTERNAL_PHASE) - mean)
                )
                for sign in (1, -1)
            ),
            I(0),
        )
    )
    observed_phase_work = resonated[2] - emitted_phase
    assert expected_phase_work.hi < 0
    assert observed_phase_work.hi < 0
    represented_work_defect = observed_phase_work - expected_phase_work
    assert represented_work_defect.abs_max < Q(1, 2**46)
    assert (observed_form_work + observed_phase_work).hi < 0
    # SHA changes mobility only. AL's supplied input still makes this entire
    # word's storage gain positive; RA passivity supplies no event reservoir.
    assert silenced[1:] == resonated[1:]
    assert (silenced[1] + silenced[2] - source[1] - source[2]).lo > 0


@pytest.mark.parametrize("failed_member", _TARGETS)
@pytest.mark.parametrize("condition", ("phase", "kind", "pressure"))
def test_rejected_resonance_preserves_exactly_the_committed_emission_prefix(
    condition, failed_member, supplied_clock
):
    graph = _source()
    if condition == "phase":
        graph.nodes[failed_member]["theta"] = float(_PHASE_OFFSET) + pi
        error_type, message = OperatorPreconditionError, "U3 phase gate"
    elif condition == "kind":
        graph.nodes[failed_member][ALIAS_EPI_KIND[0]] = "different-pattern"
        # Keep form below its admitted neighbors after AL so kind selection
        # really proposes the dominant neighbor label and reaches the gate.
        graph.nodes[failed_member]["EPI"] = 1 / 4
        error_type, message = TNFRValueError, "established_epi_kind_would_change"
    else:
        graph.nodes[failed_member]["delta_nfr"] = 3 / 4
        error_type, message = TNFRValueError, "dissonance"
    operators, sequence = _word()
    execute_network_operator_stage(
        graph, operators[0], _TARGETS, sequence_context=sequence.step(0)
    )
    prefix = _stored_state(graph)
    with pytest.raises(error_type, match=message):
        execute_network_operator_stage(
            graph, operators[1], _TARGETS, sequence_context=sequence.step(1)
        )
    assert _stored_state(graph) == prefix
    for node in _TARGETS:
        assert tuple(graph.nodes[node]["glyph_history"]) == ("AL",)
        assert graph.nodes[node]["nu_f"] == 1
        assert "latent" not in graph.nodes[node]


@pytest.fixture(scope="module")
def retained_resonance_word():
    graph = _source()
    source = deepcopy(graph)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        _execute(graph)
    reports = tuple(
        assess_sine_replica_capacity(
            state,
            reference_model=_MODEL,
            pairs=_PAIRS,
            radius=_RADIUS,
            excess_ceiling=_EXCESS_CEILING,
            form_mean_bounds=(Q(1, 4), Q(1, 2)),
        )
        for state in (source, graph)
    )
    return source, graph, reports


def test_complete_word_retains_the_same_declared_sine_identity_barrier(
    retained_resonance_word,
):
    source, graph, (before, after) = retained_resonance_word
    assert before.family.norm_squared_upper_bound < _SOURCE_RADIUS**2
    assert before.family.source_set_trapping_certified
    assert after.family.source_set_trapping_certified
    assert after.family.source_family_membership == "certified_inside"
    assert after.pair_capacity_means == (Q(33, 64), Q(1), Q(1), Q(1), Q(1))
    assert after.pair_capacity_half_differences == (Q(0),) * 5
    assert after.internal_form_squared[0] == ((1 - _MIX) * _INTERNAL_FORM) ** 2
    assert after.weighted_mean_rate_residual_bounds.contains(0)
    for row in after.capacity_aware_rate_residuals:
        assert all(residual.contains(0) for residual in row)
    for node in _TARGETS:
        assert graph.nodes[node]["delta_nfr"] == source.nodes[node]["delta_nfr"]
    assert after.all_time_internal_activity_status == "not_certified_by_this_reader"
