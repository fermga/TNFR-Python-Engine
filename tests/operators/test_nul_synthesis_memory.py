"""Actual own IL preserves a distinct successor domain after equal NUL/EN forms.

The initial pair is a declared boundary state, not an acquired THOL structure.
Every successful glyph history below is produced by the actual operator path.
Equal primitive channels/support are not claimed equal complete runtime state.
"""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.retained_synthesis import (
    capture_retained_pair,
    execute_retained_synthesis_stage,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

CORE, MEMBER = "core", "member"
MEMBERS = (CORE, MEMBER)


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _primitive_support(graph):
    """The explicitly compared projection, excluding acquired own memory."""
    return (
        tuple(
            (
                node,
                _form(graph, node),
                get_attr(graph.nodes[node], ALIAS_VF),
                get_attr(graph.nodes[node], ALIAS_THETA),
                get_attr(graph.nodes[node], ALIAS_DNFR),
            )
            for node in MEMBERS
        ),
        tuple((u, v, deepcopy(data)) for u, v, data in graph.edges(data=True)),
    )


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


def _cohere(graph, node):
    result = execute_pointwise_stage(
        graph,
        Coherence(),
        (node,),
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    assert result.glyph == "IL" and result.nodes_processed == 1


def _equalized_pair():
    graph = nx.Graph()
    graph.add_edge(CORE, MEMBER, weight=1.0, origin="declared-interface")
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=43,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        OZ_NOISE_MODE=False,
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        NUL_scale=0.5,
        EN_mix=1.0,
        OZ_dnfr_factor=2.0,
    )
    # Derive the reciprocal from this scale, not the injected default scale.
    graph.graph["GLYPH_FACTORS"].pop("NUL_densification_factor")
    for node in MEMBERS:
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            epi_kind="shared-form",
        )
        assert read_grammar_state(graph.nodes[node]).history == ()

    _cohere(graph, CORE)
    assert read_grammar_state(graph.nodes[CORE]).history == ("IL",)
    assert read_grammar_state(graph.nodes[MEMBER]).history == ()
    pair = capture_retained_pair(graph, MEMBERS)
    dispatch = execute_retained_synthesis_stage(
        graph,
        pair,
        contraction_kwargs={"validate_preconditions": True},
        reception_kwargs={"validate_preconditions": True},
    )
    assert dispatch.after.members == MEMBERS
    assert _primitive_support(graph)[0] == (
        (CORE, 0.25, 0.5, 0.0, 0.0),
        (MEMBER, 0.25, 0.5, 0.0, 0.0),
    )
    assert read_grammar_state(graph.nodes[CORE]).history == ("IL", "NUL", "EN")
    member_grammar = read_grammar_state(graph.nodes[MEMBER])
    assert member_grammar.history == ("NUL",)
    assert member_grammar.operator_step == 1
    assert not member_grammar.prior_coherence and member_grammar.u2_debt == 0
    assert graph.graph["_t"] == 0.0
    return graph


@pytest.mark.parametrize("acquire_own_il", [False, True])
def test_equal_nul_forms_do_not_replace_own_acquired_handler_context(acquire_own_il):
    graph = _equalized_pair()
    equal_primitive_support = _primitive_support(graph)
    if acquire_own_il:
        _cohere(graph, MEMBER)
        assert _primitive_support(graph) == equal_primitive_support
        grammar = read_grammar_state(graph.nodes[MEMBER])
        assert grammar.history == ("NUL", "IL")
        assert grammar.prior_coherence and grammar.u2_debt == 0

    core_before = structural_proof_signature(graph.nodes[CORE])
    before = _checkpoint(graph)
    if not acquire_own_il:
        with pytest.raises(StructuralGrammarError) as caught:
            execute_dissonance_stage(
                graph,
                Dissonance(),
                (MEMBER,),
                propagate_to_network=False,
                validate_preconditions=True,
            )
        assert caught.value.rule == "U4a"
        assert _checkpoint(graph) == before
        assert read_grammar_state(graph.nodes[MEMBER]).history == ("NUL",)
        assert get_attr(graph.nodes[MEMBER], ALIAS_DNFR) == 0.0
    else:
        result = execute_dissonance_stage(
            graph,
            Dissonance(),
            (MEMBER,),
            propagate_to_network=False,
            validate_preconditions=True,
        )
        assert result.glyph == "OZ" and result.nodes_processed == 1
        # This is the existing deterministic near-zero seed, not factor * zero.
        assert get_attr(graph.nodes[MEMBER], ALIAS_DNFR) == 0.1
        grammar = read_grammar_state(graph.nodes[MEMBER])
        assert grammar.history == ("NUL", "IL", "OZ")
        assert grammar.operator_step == 3 and grammar.u2_debt == 1
        assert grammar.prior_coherence

    assert structural_proof_signature(graph.nodes[CORE]) == core_before
    assert (_form(graph, CORE), _form(graph, MEMBER)) == (0.25, 0.25)
    assert all(get_attr(graph.nodes[node], ALIAS_VF) == 0.5 for node in MEMBERS)
    assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in MEMBERS)
    assert _primitive_support(graph)[1] == equal_primitive_support[1]
    assert all("epi_time_history" not in graph.nodes[node] for node in MEMBERS)
    assert graph.graph["_t"] == 0.0
