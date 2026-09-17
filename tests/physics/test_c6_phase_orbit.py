"""Exact supplied-orbit and shared default phase-projection checks."""

from dataclasses import FrozenInstanceError
import math

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF
from tnfr.operators._coherence_stage_kernel import (
    DEFAULT_PHASE_LOCKING_COEFFICIENT, propose_coherence_phase,
)
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage
from tnfr.operators._phase_gate import resolve_u3_phase_limits
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.physics import c6_phase_orbit as owner
from tnfr.types import Glyph
from tnfr.utils import angle_diff
from tnfr.utils.cache import ensure_node_offset_map


BASE = tuple(i * math.pi / 3 for i in range(6))
FIRST = (float.fromhex("0x1.5f38ce8af5756p-57"), *BASE[1:])
SECOND = (float.fromhex("0x1.25977c4ff39d6p-56"), *BASE[1:])
TERMINAL = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))


def _bits(values):
    return tuple(value.hex() for value in values)


def _graph(phase, variant=False):
    graph = nx.cycle_graph(6)
    inject_defaults(graph)
    graph.graph["RANDOM_SEED"] = 17
    for i in graph:
        graph.nodes[i].update({
            ALIAS_EPI[0]: 0.2 + 0.1 * i if variant else 0.5,
            ALIAS_VF[0]: 0.7 + 0.1 * i if variant else 1.0,
            ALIAS_DNFR[0]: (-1.0)**i * 0.03 if variant else 0.0,
            ALIAS_SI[0]: 0.1 + 0.1 * i if variant else 0.5,
            ALIAS_THETA[0]: phase[i], "glyph_history": [],
        })
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    if variant:
        graph.graph["_node_sample"] = tuple(reversed(tuple(graph)))
    return graph


def _source_projection(graph):
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    stage = propose_coupling_stage(
        graph, tuple(graph), factors, resolved_seed=17,
        node_offsets=dict(ensure_node_offset_map(graph)),
    )
    assert graph.graph.get("UM_FUNCTIONAL_LINKS", True)
    assert stage.edges == ()
    coupled = tuple(update.theta_after for update in stage.node_updates)
    for node, value in zip(graph, coupled, strict=True):
        graph.nodes[node][ALIAS_THETA[0]] = value
    proposals = tuple(propose_coherence_phase(graph, node, DEFAULT_PHASE_LOCKING_COEFFICIENT) for node in graph)
    return coupled, tuple(proposal.theta_after for proposal in proposals), stage


@pytest.fixture(scope="module")
def terminal_step():
    return owner.observe_c6_coupling_coherence_phase_step(phase=TERMINAL)


def test_retained_first_two_transitions_reproduce_exact_binary64_coordinates():
    first = owner.observe_c6_coupling_coherence_phase_step(phase=BASE)
    second = owner.observe_c6_coupling_coherence_phase_step(phase=FIRST)
    assert first.phase_after_coupling[0].hex() == "0x1.5555555555555p-56"
    assert _bits(first.phase_after_coherence) == _bits(FIRST)
    assert second.phase_after_coupling[0].hex() == "0x1.fde097f694500p-56"
    assert _bits(second.phase_after_coherence) == _bits(SECOND)
    assert _bits(first.phase_after_coupling[1:]) == _bits(BASE[1:])
    assert _bits(second.phase_after_coherence[1:]) == _bits(BASE[1:])


def test_terminal_step_closes_only_after_both_real_phase_operators(terminal_step):
    assert _bits(terminal_step.phase_before) == _bits(TERMINAL)
    assert terminal_step.phase_after_coupling[0].hex() == "0x1.ab75f42bdf52ep-55"
    assert _bits(terminal_step.phase_after_coupling) != _bits(TERMINAL)
    assert _bits(terminal_step.phase_after_coherence) == _bits(TERMINAL)
    assert terminal_step.coherence_methods == ("exact_two_neighbor_midpoint",) * 6
    with pytest.raises(FrozenInstanceError):
        terminal_step.phase_before = BASE


@pytest.mark.parametrize("phase", (BASE, FIRST, TERMINAL, (math.tau, *BASE[1:]), (-0.0, *BASE[1:])))
def test_projected_phase_matches_existing_owners_with_unrelated_primary_fields_changed(phase):
    actual = owner.observe_c6_coupling_coherence_phase_step(phase=phase)
    plain = _graph(phase)
    varied = _graph(phase, variant=True)
    expected_um, expected_il, plain_stage = _source_projection(plain)
    varied_um, varied_il, varied_stage = _source_projection(varied)
    assert _bits(actual.phase_after_coupling) == _bits(expected_um) == _bits(varied_um)
    assert _bits(actual.phase_after_coherence) == _bits(expected_il) == _bits(varied_il)
    assert plain_stage.node_updates != varied_stage.node_updates


def test_default_factors_and_all_three_gate_boundaries_are_exposed(terminal_step):
    graph = _graph(TERMINAL)
    defaults = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    _, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    assert terminal_step.um_phase_factor == defaults["UM_theta_push"]
    assert terminal_step.il_phase_factor == DEFAULT_PHASE_LOCKING_COEFFICIENT
    assert terminal_step.effective_phase_limit == gate
    boundaries = (terminal_step.phase_before, terminal_step.phase_after_coupling, terminal_step.phase_after_coherence)
    assert len(terminal_step.edge_margins) == len(terminal_step.nonedge_margins) == 3
    for phase, edge_margins, nonedge_margins in zip(
        boundaries, terminal_step.edge_margins, terminal_step.nonedge_margins, strict=True,
    ):
        expected_edges = tuple(gate - abs(angle_diff(phase[i], phase[j])) for i, j in graph.edges)
        expected_nonedges = tuple(abs(angle_diff(phase[i], phase[j])) - gate for i, j in nx.non_edges(graph))
        assert tuple(edge_margins) == expected_edges
        assert tuple(nonedge_margins) == expected_nonedges
        assert len(edge_margins) == 6 and min(edge_margins) > 0
        assert len(nonedge_margins) == 9 and min(nonedge_margins) > 0


def test_exact_two_state_fixed_phase_orbit_has_period_one():
    states = (TERMINAL, TERMINAL)
    result = owner.derive_c6_coupling_coherence_phase_orbit(phase_states=states, cycle_start=0)
    assert result.phase_states == states
    assert result.preperiod == 0
    assert result.period == 1
    assert len(result.steps) == 1
    assert result.conditional_phase_periodic
    assert result.future_runtime_certified is False


def test_a_valid_nonminimal_supplied_loop_does_not_require_search():
    result = owner.derive_c6_coupling_coherence_phase_orbit(
        phase_states=(TERMINAL, TERMINAL, TERMINAL), cycle_start=0,
    )
    assert result.preperiod == 0
    assert result.period == 2
    assert len(result.steps) == 2
    assert result.conditional_phase_periodic


@pytest.mark.parametrize("states,start", (
    ((BASE, FIRST), 0),
    ((BASE, SECOND, SECOND), 1),
    ((TERMINAL, (math.nextafter(TERMINAL[0], math.inf), *TERMINAL[1:])), 0),
    ((TERMINAL, tuple((value + 0.25) % math.tau for value in TERMINAL)), 0),
))
def test_nonclosing_skipped_or_approximately_matching_transitions_are_rejected(states, start):
    with pytest.raises((ValueError, RuntimeError)):
        owner.derive_c6_coupling_coherence_phase_orbit(phase_states=states, cycle_start=start)


@pytest.mark.parametrize("states,start,error", (
    ((), 0, ValueError), ((TERMINAL,), 0, ValueError),
    ([TERMINAL, TERMINAL], 0, ValueError), ((list(TERMINAL), TERMINAL), 0, TypeError),
    ((TERMINAL, TERMINAL), -1, ValueError), ((TERMINAL, TERMINAL), 1, ValueError),
    ((TERMINAL, TERMINAL), True, ValueError), ((TERMINAL, TERMINAL), 0.0, ValueError),
))
def test_invalid_orbit_shape_and_declared_cycle_index_are_rejected(states, start, error):
    with pytest.raises(error):
        owner.derive_c6_coupling_coherence_phase_orbit(phase_states=states, cycle_start=start)


@pytest.mark.parametrize("phase,error", (
    ([0.0] * 6, TypeError), (BASE[:5], ValueError), ((*BASE, 0.0), ValueError),
    ((True, *BASE[1:]), TypeError), ((math.nan, *BASE[1:]), ValueError),
    ((math.inf, *BASE[1:]), ValueError), ((-0.01, *BASE[1:]), ValueError),
    ((math.nextafter(math.tau, math.inf), *BASE[1:]), ValueError),
    ((0.0,) * 6, ValueError), ((math.pi, *BASE[1:]), ValueError),
))
def test_input_phase_and_fixed_support_domain_are_enforced(phase, error):
    with pytest.raises(error):
        owner.observe_c6_coupling_coherence_phase_step(phase=phase)
