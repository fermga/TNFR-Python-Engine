"""Actual IL/NUL preparation and one connected UM admission discriminator.

The explicit support-only comparator is a diagnostic graph intervention, not
a canonical substitute for UM. All occasions and the contact pair are supplied.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.node import add_edge
from tnfr.operators.coherence import Coherence
from tnfr.operators.contraction import Contraction
from tnfr.operators.coupling import Coupling
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    execute_retained_scale_stage,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

A, INNER_A, B, INNER_B = "A", "inner-A", "B", "inner-B"
ORDER = (A, INNER_A, B, INNER_B)


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
    )


def _capacity(graph, node):
    return Q.from_float(get_attr(graph.nodes[node], ALIAS_VF))


def _pressure(graph, node):
    return Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR))


def _edges(graph):
    return {
        frozenset((left, right)): deepcopy(data)
        for left, right, data in graph.edges(data=True)
    }


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


def _source():
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=71,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        NUL_scale=0.5,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        AL_boost=0.125,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    # The old injected reciprocal would contradict the newly declared scale.
    graph.graph["GLYPH_FACTORS"].pop("NUL_densification_factor")
    for node, form, capacity, pressure in (
        (A, Q(1, 2), 1.0, Q(3, 16)),
        (INNER_A, Q(1, 4), 1.0, Q(0)),
        (B, Q(1, 2), 2.0, Q(0)),
        (INNER_B, Q(1, 2), 2.0, Q(0)),
    ):
        graph.add_node(
            node,
            EPI=float(form),
            nu_f=capacity,
            theta=0.0,
            delta_nfr=float(pressure),
            Si=0.0,
            epi_kind="form",
        )
        assert read_grammar_state(graph.nodes[node]).history == ()
    graph.add_edge(A, INNER_A, weight=1.0, role="left-interior")
    graph.add_edge(B, INNER_B, weight=1.0, role="right-interior")
    execute_pointwise_stage(
        graph,
        Coherence(),
        ORDER,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    for members in ((A, INNER_A), (B, INNER_B)):
        result = execute_retained_scale_stage(
            graph,
            capture_retained_organization(graph, members),
            Contraction(),
            execution_kwargs={"validate_preconditions": True},
        )
        assert result.stage_result.nodes_processed == 2
    assert tuple(_form(graph, node) for node in ORDER) == (
        Q(1, 4),
        Q(1, 8),
        Q(1, 4),
        Q(1, 4),
    )
    assert tuple(_capacity(graph, node) for node in ORDER) == (
        Q(1, 2),
        Q(1, 2),
        Q(1),
        Q(1),
    )
    assert tuple(_pressure(graph, node) for node in ORDER) == (
        Q(3, 16),
        Q(0),
        Q(0),
        Q(0),
    )
    assert {frozenset(component) for component in nx.connected_components(graph)} == {
        frozenset((A, INNER_A)),
        frozenset((B, INNER_B)),
    }
    for node in ORDER:
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == ("IL", "NUL") and grammar.operator_step == 2
        assert grammar.prior_coherence
    return graph


def test_connected_um_changes_admitted_response_beyond_the_same_added_edge():
    outcomes = {}
    contact_support = {}
    for arm in ("um", "support-only", "no-contact"):
        graph = _source()
        old_edges = _edges(graph)
        own_contexts = {node: read_grammar_state(graph.nodes[node]) for node in ORDER}
        if arm == "um":
            result = execute_coupling_stage(
                graph,
                Coupling(),
                (A,),
                potential_partner=B,
                potential_scope="disjoint_components",
                validate_preconditions=True,
            )
            assert result.glyph == "UM" and result.nodes_processed == 1
            pair = graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"]
            assert pair["target"] == A and pair["partner"] == B
            assert pair["origin"] == "explicit_request"
            assert pair["scope"] == "disjoint_components"
            assert _capacity(graph, A) == Q(9, 16)
            assert _pressure(graph, A) == Q(9, 64)
            assert read_grammar_state(graph.nodes[A]).history == ("IL", "NUL", "UM")
        elif arm == "support-only":
            # Explicit noncanonical control, with ordinary support-cache invalidation.
            add_edge(graph, A, B, 1.0)
        if arm != "um":
            assert _capacity(graph, A) == Q(1, 2)
            assert _pressure(graph, A) == Q(3, 16)
            assert read_grammar_state(graph.nodes[A]) == own_contexts[A]
        for node in (INNER_A, B, INNER_B):
            assert read_grammar_state(graph.nodes[node]) == own_contexts[node]
            assert _pressure(graph, node) == 0
        assert tuple(_form(graph, node) for node in ORDER) == (
            Q(1, 4),
            Q(1, 8),
            Q(1, 4),
            Q(1, 4),
        )
        assert tuple(_capacity(graph, node) for node in (INNER_A, B, INNER_B)) == (
            Q(1, 2),
            Q(1),
            Q(1),
        )
        assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in ORDER)
        for edge, attrs in old_edges.items():
            assert _edges(graph)[edge] == attrs
        assert graph.has_edge(A, B) == (arm != "no-contact")
        if arm != "no-contact":
            assert graph.edges[A, B] == {"weight": 1.0}
            contact_support[arm] = _edges(graph)
            assert nx.is_connected(graph)
        else:
            assert nx.number_connected_components(graph) == 2

        Emission()(graph, B, validate_preconditions=True)
        assert _form(graph, B) == Q(3, 8)
        assert _capacity(graph, B) == 1 and _pressure(graph, B) == 0
        assert graph.nodes[B]["_structural_lineage"]["activation_count"] == 1
        assert read_grammar_state(graph.nodes[B]).history == ("IL", "NUL", "AL")
        before_reception = _checkpoint(graph)
        if arm == "um":
            received = execute_neighbor_stage(
                graph,
                Reception(),
                (A,),
                source_nodes=(B,),
                track_sources=False,
                validate_preconditions=True,
            )
            (observation,) = received.reception_observations
            assert observation.node == A and observation.source_nodes == (B,)
            assert observation.neighbors == (
                B,
            ) and observation.neighbor_epi_values == (3 / 8,)
            assert not observation.source_tracking_enabled
            assert observation.target_epi_before == 1 / 4
            assert observation.target_epi_after == 5 / 16
            assert _form(graph, A) == Q(5, 16)
        else:
            with pytest.raises((ValueError, OperatorPreconditionError)) as rejected:
                execute_neighbor_stage(
                    graph,
                    Reception(),
                    (A,),
                    source_nodes=(B,),
                    track_sources=False,
                    validate_preconditions=True,
                )
            reason = str(rejected.value)
            if arm == "support-only":
                assert "DNFR" in reason
            else:
                assert "DNFR" in reason or "realized incoming neighbors" in reason
            # No order claim: no-contact has both missing support and high pressure.
            assert _checkpoint(graph) == before_reception
            assert _form(graph, A) == Q(1, 4)
        protected = {node: deepcopy(graph.nodes[node]) for node in (A, B, INNER_B)}
        Resonance()(graph, INNER_A, validate_preconditions=True)
        expected_response = Q(7, 32) if arm == "um" else Q(3, 16)
        assert _form(graph, INNER_A) == expected_response
        assert _capacity(graph, INNER_A) == Q(9, 16)
        assert _pressure(graph, INNER_A) == 0
        assert all(graph.nodes[node] == value for node, value in protected.items())
        for node, word in (
            (A, ("IL", "NUL", "UM", "EN") if arm == "um" else ("IL", "NUL")),
            (INNER_A, ("IL", "NUL", "RA")),
            (B, ("IL", "NUL", "AL")),
            (INNER_B, ("IL", "NUL")),
        ):
            grammar = read_grammar_state(graph.nodes[node])
            assert grammar.history == word and grammar.operator_step == len(word)
            assert grammar.prior_coherence
        assert tuple(graph) == ORDER and graph.graph["_t"] == 0.0
        assert all("epi_time_history" not in graph.nodes[node] for node in ORDER)
        assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in ORDER)
        outcomes[arm] = _form(graph, INNER_A)

    assert contact_support["um"] == contact_support["support-only"]
    assert outcomes["um"] - outcomes["support-only"] == Q(1, 32)
    assert outcomes["um"] - outcomes["no-contact"] == Q(1, 32)
