"""Actual bounded operator sources for the opt-in hybrid Mutation policy.

The supplied word and clock do not establish autonomous occurrence. Every
positive source below executes its own IL/OZ context; no history is supplied
as growth or grammar evidence. The held zero driver between events is explicit.
"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators._mutation_gate import mutation_trigger_certificate
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.mutation import Mutation
from tnfr.operators.operator_growth import observe_operator_epi_growth
from tnfr.operators.operator_growth_mutation import execute_operator_growth_mutation
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature
from tnfr.validation import StructuralGrammarError

CORE, DONOR = "core", "donor"
NODES = (CORE, DONOR)
INITIAL = (Fraction(3, 8), Fraction(1, 8))
ZERO = (Fraction(0), Fraction(0))
WORD = ("emission", "coherence", "dissonance", "coherence", "silence")
RECEIPT = "_operator_growth_mutation_end"


def _source(*, threshold=0.0625, shift=0.25):
    graph = nx.Graph()
    graph.add_edge(CORE, DONOR, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        EDGE_AWARE_ENABLED=False,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        OZ_NOISE_MODE=False,
        RANDOM_SEED=31,
        ZHIR_THRESHOLD_XI=threshold,
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        IL_dnfr_factor=0.0,
        SHA_vf_factor=0.5,
        ZHIR_theta_shift=shift,
    )
    for node, form in zip(NODES, INITIAL, strict=True):
        graph.nodes[node].update(
            EPI=float(form),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            epi_kind="form",
        )
        assert "epi_history" not in graph.nodes[node]
        assert "epi_time_history" not in graph.nodes[node]
        assert "glyph_history" not in graph.nodes[node]
    assert "compute_delta_nfr" not in graph.graph
    return graph


def _schedule(word=WORD, *, start=0.0):
    # AL is inside the window; all subsequent glyphs are at its terminal time.
    durations = (0.5, 0.5) + (0.0,) * (len(word) - 1)
    return build_operator_event_schedule(
        word, start_time=start, flow_durations=durations
    )


def _source_only(graph, word=WORD, *, start=0.0):
    return execute_operator_event_schedule(
        graph,
        _schedule(word, start=start),
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
        include_stage_certificates=True,
    )


def _execute(graph, targets=NODES, *, word=WORD, start=0.0):
    return execute_operator_growth_mutation(
        graph,
        _schedule(word, start=start),
        targets,
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
    )


def _forms(graph):
    return tuple(
        Fraction.from_float(
            require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))
        )
        for node in NODES
    )


def _state(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        ),
        opaque_references=(graph,),
    )


def test_actual_jump_growth_executes_phase_only_mutation_with_own_il_oz_context():
    graph = _source()
    result = _execute(graph)
    growth = result.growth
    assert result.targets == result.source_result.target_nodes == NODES
    assert result.source_result.pressure_refresh_callback_invocations == 0
    assert growth.available
    assert growth.exact_start_time == 0 and growth.exact_end_time == 1
    assert growth.exact_epi_before == INITIAL
    assert (
        growth.exact_epi_after
        == _forms(graph)
        == (
            Fraction(1, 2),
            Fraction(1, 4),
        )
    )
    assert growth.exact_flow_increment == ZERO
    assert (
        growth.exact_jump_increment
        == growth.exact_net_growth_rate
        == (Fraction(1, 8),) * 2
    )
    assert tuple(graph.edges) == ((CORE, DONOR),)
    assert graph.edges[CORE, DONOR]["weight"] == 1.0
    assert graph.graph["_t"] == 1.0
    assert len(result.stage_result.mutation_decision_observations) == 2
    for node, observation in zip(
        NODES, result.stage_result.mutation_decision_observations, strict=True
    ):
        assert get_attr(graph.nodes[node], ALIAS_THETA) == 0.25
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.5
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0
        assert graph.nodes[node][RECEIPT] == Fraction(1)
        assert type(graph.nodes[node][RECEIPT]) is Fraction
        assert "_native_zhir_consumed_intervals" not in graph.nodes[node]
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == ("AL", "IL", "OZ", "IL", "SHA", "ZHIR")
        assert grammar.prior_coherence and grammar.u2_debt == 1
        assert observation.node == node
        assert observation.recent_history[-3:] == (
            "dissonance",
            "coherence",
            "silence",
        )
        assert observation.destabilizer_operator == "dissonance"
        certificate = observation.trigger_certificate
        assert certificate.source == "operator_schedule_net_growth"
        assert certificate.time_basis == "operator_schedule_time"
        assert certificate.exact_observed_rate == Fraction(1, 8)
        assert certificate.exact_xi == Fraction(1, 16)
        # The successful alternative does not manufacture continuous growth.
        default = mutation_trigger_certificate(graph.nodes[node], graph.graph)
        assert not default.threshold_gate_satisfied


def test_selected_target_mutates_without_changing_peer_phase_or_consuming_peer():
    graph = _source()
    result = _execute(graph, (CORE,))
    assert result.targets == (CORE,)
    assert result.growth.nodes == NODES
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.25
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.0
    assert graph.nodes[CORE][RECEIPT] == 1
    assert RECEIPT not in graph.nodes[DONOR]
    assert read_grammar_state(graph.nodes[DONOR]).history == (
        "AL",
        "IL",
        "OZ",
        "IL",
        "SHA",
    )
    assert _forms(graph) == (Fraction(1, 2), Fraction(1, 4))


def test_same_actual_source_leaves_ordinary_mutation_continuous_gate_unsatisfied():
    graph = _source()
    source = _source_only(graph)
    assert (
        observe_operator_epi_growth(source).exact_net_growth_rate
        == (Fraction(1, 8),) * 2
    )
    context = read_grammar_state(graph.nodes[CORE])
    assert context.prior_coherence and context.history[-3:] == ("OZ", "IL", "SHA")
    default = mutation_trigger_certificate(graph.nodes[CORE], graph.graph)
    assert default.source == "epi_time_history"
    assert default.observed_depi_dt == 0.0
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="dEPI/dt|threshold"):
        Mutation()(graph, CORE)
    assert _state(graph) == before
    assert all(RECEIPT not in graph.nodes[node] for node in NODES)


CANCELLING_WORD = (
    "emission",
    "reception",
    "coherence",
    "dissonance",
    "coherence",
    "silence",
)


@pytest.mark.parametrize("targets", [(CORE,), NODES])
def test_cancelling_core_rejects_and_rolls_back_entire_source_and_mutation(targets):
    reference = _source()
    source = _source_only(reference, CANCELLING_WORD)
    growth = observe_operator_epi_growth(source)
    assert growth.available
    assert growth.exact_net_growth_rate == (Fraction(0), Fraction(1, 4))
    assert growth.exact_jump_variation == (Fraction(1, 4),) * 2
    assert read_grammar_state(reference.nodes[CORE]).history[-3:] == (
        "OZ",
        "IL",
        "SHA",
    )
    graph = _source()
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="growth|threshold|dEPI/dt"):
        _execute(graph, targets, word=CANCELLING_WORD)
    assert _state(graph) == before
    assert _forms(graph) == INITIAL and graph.graph["_t"] == 0.0


def test_cancelling_core_does_not_prevent_own_positive_donor_admission():
    graph = _source()
    result = _execute(graph, (DONOR,), word=CANCELLING_WORD)
    assert result.growth.exact_net_growth_rate == (Fraction(0), Fraction(1, 4))
    assert result.growth.exact_jump_variation == (Fraction(1, 4),) * 2
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.0
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.25
    assert RECEIPT not in graph.nodes[CORE]
    assert graph.nodes[DONOR][RECEIPT] == 1


@pytest.mark.parametrize(
    "word,reason",
    [
        # Both IL and a destabilizer are absent; this is not an isolated IL test.
        (("emission", "silence"), "destabilizer|prior IL|U4b"),
        (("emission", "coherence", "silence"), "destabilizer|U4b"),
        (WORD + ("silence",), "destabilizer|U4b"),
    ],
)
def test_actual_positive_source_cannot_replace_missing_live_grammar(word, reason):
    reference = _source()
    source = _source_only(reference, word)
    growth = observe_operator_epi_growth(source)
    assert growth.available and growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    graph = _source()
    before = _state(graph)
    with pytest.raises(
        (StructuralGrammarError, OperatorPreconditionError), match=reason
    ):
        _execute(graph, word=word)
    assert _state(graph) == before


def test_disjoint_successor_source_uses_new_growth_and_advances_own_watermark():
    graph = _source()
    first = _execute(graph)
    # Repeating AL on the core would violate its latent-state prerequisite.
    # Actual EN instead supplies the donor's next positive growth interval.
    second = _execute(
        graph,
        (DONOR,),
        word=("reception",) + WORD[1:],
        start=1.0,
    )
    assert first.growth.exact_end_time == second.growth.exact_start_time == 1
    assert second.growth.exact_end_time == 2
    assert second.growth.exact_flow_increment == ZERO
    assert second.growth.exact_net_growth_rate == (Fraction(-1, 8), Fraction(1, 8))
    assert _forms(graph) == (Fraction(3, 8),) * 2
    assert get_attr(graph.nodes[CORE], ALIAS_THETA) == 0.25
    assert get_attr(graph.nodes[DONOR], ALIAS_THETA) == 0.5
    assert graph.nodes[CORE][RECEIPT] == Fraction(1)
    assert graph.nodes[DONOR][RECEIPT] == Fraction(2)
    for node in NODES:
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.25
        assert get_attr(graph.nodes[node], ALIAS_DNFR) == 0.0


def test_nonoverlapping_clock_does_not_override_emission_latent_state_admission():
    graph = _source()
    _execute(graph)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="AL precondition failed|latent state"):
        _execute(graph, start=1.0)
    assert _state(graph) == before
    assert graph.graph["_t"] == 1.0
    assert all(graph.nodes[node][RECEIPT] == 1 for node in NODES)


@pytest.mark.parametrize("start", [0.0, 0.5])
def test_clock_reset_cannot_reuse_or_overlap_an_already_consumed_window(start):
    graph = _source()
    _execute(graph)
    # Explicit hostile clock reset: the retained receipt must still prevent reuse.
    graph.graph["_t"] = start
    before = _state(graph)
    with pytest.raises(ValueError, match="overlap|consum|window|watermark"):
        _execute(graph, start=start)
    assert _state(graph) == before
    assert all(graph.nodes[node][RECEIPT] == 1 for node in NODES)


def test_invalid_zero_shift_rolls_back_real_source_without_advancing_watermark():
    graph = _source(shift=0.0)
    before = _state(graph)
    with pytest.raises(GlyphFactorValidationError, match="ZHIR_theta_shift"):
        _execute(graph)
    assert _state(graph) == before
    assert _forms(graph) == INITIAL
    assert all(RECEIPT not in graph.nodes[node] for node in NODES)


@pytest.mark.parametrize("threshold", [0.125, 0.25])
def test_equality_and_larger_threshold_reject_actual_positive_growth(threshold):
    graph = _source(threshold=threshold)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="growth|threshold|dEPI/dt"):
        _execute(graph)
    assert _state(graph) == before
