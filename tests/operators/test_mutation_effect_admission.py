"""ZHIR admits an actual represented circular change before any commit."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators._mutation_stage_kernel import (
    mutation_phase_changed,
    propose_mutation_stage,
)
from tnfr.operators.definitions import Mutation
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.postconditions import OperatorContractViolation
from tnfr.operators.postconditions.mutation import verify_phase_transformed


def _source(*, magnitude, sign, fixed, theta=1.0):
    factors = (
        {"ZHIR_theta_shift": sign * magnitude}
        if fixed
        else {"ZHIR_theta_shift_factor": magnitude}
    )
    graph = nx.Graph(
        ZHIR_THRESHOLD_XI=0.1,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS=factors,
    )
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.2,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: sign * 0.3,
            ALIAS_THETA[0]: theta,
            "epi_kind": "wave",
            "epi_history": [0.0, 0.2],
            "glyph_history": ["IL", "OZ"],
        },
    )
    return graph


def _apply(graph, path):
    if path == "glyph":
        apply_glyph(graph, 0, "ZHIR")
    elif path == "public":
        Mutation()(graph, 0, validate_postconditions=True)
    else:
        execute_pointwise_stage(graph, Mutation(), (0,))


def _snapshot(graph):
    # apply_glyph may cache its NodeNX adapter even when admission rejects.
    # It must not commit physical state, history or lifecycle/decision metadata.
    return deepcopy(
        (
            {key: value for key, value in graph.graph.items() if key != "_node_cache"},
            dict(graph.nodes[0]),
            list(graph.edges),
        )
    )


@pytest.mark.parametrize("path", ("glyph", "public", "stage"))
@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_lost_phase_effect_rejects_before_state_history_or_metadata(path, fixed, sign):
    graph = _source(magnitude=1e-20, sign=sign, fixed=fixed)
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="no represented circular effect"):
        _apply(graph, path)

    assert _snapshot(graph) == before


@pytest.mark.parametrize("path", ("glyph", "public", "stage"))
@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_small_represented_change_is_accepted_without_size_cutoff(path, fixed, sign):
    graph = _source(magnitude=1e-7, sign=sign, fixed=fixed)
    before = deepcopy(dict(graph.nodes[0]))

    _apply(graph, path)

    after = graph.nodes[0]
    assert mutation_phase_changed(before[ALIAS_THETA[0]], after[ALIAS_THETA[0]])
    assert 0.0 < abs(after[ALIAS_THETA[0]] - before[ALIAS_THETA[0]]) < 1e-6
    assert list(after["glyph_history"])[-1] == "ZHIR"
    for field in (ALIAS_EPI[0], ALIAS_VF[0], ALIAS_DNFR[0], "epi_kind"):
        assert after[field] == before[field]
    verify_phase_transformed(graph, 0, before[ALIAS_THETA[0]])


@pytest.mark.parametrize("before", (0.0, math.tau, -math.tau, 4.0 * math.tau))
def test_chart_normalization_is_not_a_mutation(before):
    assert not mutation_phase_changed(before, 0.0)
    graph = nx.Graph()
    graph.add_node(0, **{ALIAS_THETA[0]: 0.0})
    with pytest.raises(OperatorContractViolation, match="not transformed"):
        verify_phase_transformed(graph, 0, before)
    with pytest.raises(TNFRValueError, match="no represented circular effect"):
        propose_mutation_stage(before, fixed_shift=-1e-20)


@pytest.mark.parametrize("fixed", (False, True))
def test_no_arbitrary_floor_for_change_representable_near_zero(fixed):
    proposal = propose_mutation_stage(
        0.0,
        1.0,
        fixed_shift=1e-20 if fixed else None,
        theta_shift_factor=1e-20,
    )
    assert 0.0 < proposal.theta_after < 1e-19
    assert mutation_phase_changed(0.0, proposal.theta_after)


@pytest.mark.parametrize("shift", (0.0, math.tau, -math.tau))
def test_pure_proposal_rejects_zero_circular_effect(shift):
    with pytest.raises(TNFRValueError, match="no represented circular effect"):
        propose_mutation_stage(0.0, fixed_shift=shift)


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_endpoint_comparison_retains_small_signed_displacement(sign):
    assert mutation_phase_changed(sign * 1e-20, 0.0)
    assert mutation_phase_changed(0.0, sign * 1e-20)
    graph = nx.Graph()
    graph.add_node(0, **{ALIAS_THETA[0]: 0.0})
    verify_phase_transformed(graph, 0, sign * 1e-20)


def test_normalization_loss_alone_does_not_make_proposal_effective():
    with pytest.raises(TNFRValueError, match="no represented circular effect"):
        propose_mutation_stage(-1e-20, fixed_shift=-1e-20)
