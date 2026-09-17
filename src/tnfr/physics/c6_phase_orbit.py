"""Exact finite closure of the default C6 UM/IL phase projection.

The production proposal owners are evaluated on detached read fixtures.
Their phase outputs define this numerical map. No operator is committed,
no EPI flow is run, and grammar or future runtime admission is not inferred.
Fixed support is justified by checking every edge and nonedge phase gate.
"""

from dataclasses import dataclass
import math

import networkx as nx

from .._binary64 import uses_ieee_binary64_rounding
from ..config import inject_defaults
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF
from ..dynamics._euler_kernel import _binary64_tuple
from ..operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT, propose_coherence_phase
from ..operators._coupling_stage_kernel import propose_coupling_stage
from ..operators._phase_gate import resolve_u3_phase_limits
from ..operators.factor_contracts import resolve_runtime_operator_factors
from ..types import Glyph
from ..utils import angle_diff

__all__ = [
    "C6CouplingCoherencePhaseStep", "C6CouplingCoherencePhaseOrbit",
    "observe_c6_coupling_coherence_phase_step", "derive_c6_coupling_coherence_phase_orbit",
]


def _phases(values):
    values = _binary64_tuple(values, "phase")
    if len(values) != 6 or any(not 0.0 <= value <= math.tau for value in values):
        raise ValueError("phase must contain six stored C6 coordinates in [0,tau]")
    return values


def _signature(values):
    # Exact stored-state identity includes the sign of zero. There is no
    # centered, circular-tolerance or common-rotation identification here.
    return tuple(value.hex() for value in values)


def _phase_graph(phase):
    graph = nx.cycle_graph(6)
    inject_defaults(graph)
    graph.graph["RANDOM_SEED"] = 17
    for node, value in enumerate(phase):
        graph.nodes[node].update({
            ALIAS_EPI[0]: .5, ALIAS_VF[0]: 1.0, ALIAS_THETA[0]: value,
            ALIAS_DNFR[0]: 0.0, ALIAS_SI[0]: .5,
        })
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    return graph


def _gate_margins(graph, phase, limit):
    edges = tuple(limit - abs(angle_diff(phase[i], phase[j])) for i, j in graph.edges)
    nonedges = tuple(abs(angle_diff(phase[i], phase[j])) - limit
                     for i in range(6) for j in range(i + 1, 6) if not graph.has_edge(i, j))
    if min(edges) < 0 or min(nonedges) <= 0:
        raise ValueError("the phase projection requires all C6 edges admitted and all nonedges excluded")
    return edges, nonedges


@dataclass(frozen=True, slots=True)
class C6CouplingCoherencePhaseStep:
    """Detached shared-kernel phase transition with gates at all boundaries."""

    phase_before: tuple[float, ...]
    phase_after_coupling: tuple[float, ...]
    phase_after_coherence: tuple[float, ...]
    um_phase_factor: float
    il_phase_factor: float
    effective_phase_limit: float
    edge_margins: tuple[tuple[float, ...], ...]
    nonedge_margins: tuple[tuple[float, ...], ...]
    coherence_methods: tuple[str, ...]


def observe_c6_coupling_coherence_phase_step(*, phase: tuple[float, ...]) -> C6CouplingCoherencePhaseStep:
    """Evaluate default all-target UM followed by simultaneous IL phase.

    A fresh ordered unit C6 supplies canonical default factors and topology.
    Other nodal fields are read fixtures, not evolving EPI states. All edges
    must pass U3 and every nonedge must be strictly excluded before UM, after
    its merge and after IL. Functional links remain enabled: their phase
    gate excludes every candidate independently of EPI/SI or RNG selection.
    UM's phase proposal/merge reads only phases, ordered support and factors;
    IL's phase proposal reads only phases/support and its default coefficient.
    Unit capacity is preserved by the actual UM capacity proposal.

    This proves a closed numerical phase component conditionally on fixed
    configuration, admitted stages and unchanged support. The input/return
    values do not authenticate a live graph, a history, pressure refresh,
    grammar, full operator postconditions or future EPI-band membership.
    The host's actual transcendental outputs are retained, not replaced by
    an assumed correctly rounded ideal sine/atan2 implementation.
    """
    before = _phases(phase)
    if not uses_ieee_binary64_rounding():
        raise RuntimeError("phase orbit replay requires the declared IEEE binary64 environment")
    graph = _phase_graph(before)
    if not all(graph.graph.get(key, True) for key in ("UM_BIDIRECTIONAL", "UM_FUNCTIONAL_LINKS", "UM_SYNC_VF")):
        raise ValueError("the projection requires the declared bidirectional/default-link/unit-capacity policy")
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    _, limit = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    initial_edges, initial_nonedges = _gate_margins(graph, before, limit)
    stage = propose_coupling_stage(
        graph, tuple(graph), factors, resolved_seed=17, node_offsets={i: i for i in graph},
    )
    if (stage.edges or tuple(update.node for update in stage.node_updates) != tuple(graph)
            or any(p.compatible_neighbors != tuple(graph.neighbors(p.node)) or p.link_candidates
                   for p in stage.target_proposals)
            or any(update.vf_after != 1.0 for update in stage.node_updates)):
        raise RuntimeError("the default phase proposal changed the declared fixed C6 support or capacity")
    after_um = _phases(tuple(update.theta_after for update in stage.node_updates))
    um_edges, um_nonedges = _gate_margins(graph, after_um, limit)
    # This is construction of a second detached proposal input. No operator
    # or numerical EPI update is represented by these phase-only fixtures.
    il_graph = _phase_graph(after_um)
    proposals = tuple(propose_coherence_phase(il_graph, i, DEFAULT_PHASE_LOCKING_COEFFICIENT)
                      for i in il_graph)
    after_il = _phases(tuple(proposal.theta_after for proposal in proposals))
    il_edges, il_nonedges = _gate_margins(graph, after_il, limit)
    return C6CouplingCoherencePhaseStep(
        before, after_um, after_il, float(factors["UM_theta_push"]),
        DEFAULT_PHASE_LOCKING_COEFFICIENT, limit,
        (initial_edges, um_edges, il_edges), (initial_nonedges, um_nonedges, il_nonedges),
        tuple(proposal.method for proposal in proposals),
    )


@dataclass(frozen=True, slots=True)
class C6CouplingCoherencePhaseOrbit:
    """Replayed finite path to an exact stored-state cycle of the phase map.

    Preperiod and period are declared indices, not necessarily the smallest
    possible ones. The verified cycle permits indefinite repetition of this
    deterministic projection in the same fixed numerical environment. It
    does not prove that the full engine can keep admitting those stages.
    """

    phase_states: tuple[tuple[float, ...], ...]
    steps: tuple[C6CouplingCoherencePhaseStep, ...]
    preperiod: int
    period: int

    @property
    def conditional_phase_periodic(self):
        return True

    @property
    def future_runtime_certified(self):
        return False


def derive_c6_coupling_coherence_phase_orbit(
    *, phase_states: tuple[tuple[float, ...], ...], cycle_start: int,
) -> C6CouplingCoherencePhaseOrbit:
    """Replay a caller-supplied path and verify its exact terminal closure.

    The path includes the final repeated state. All transitions are derived
    again from the production phase proposal owners; caller fields, rounded
    tolerances and earlier observations are not accepted as substitutes.
    This function does not search for cycles. Bitwise state identity and
    deterministic transition imply conditional phase periodicity after the
    declared preperiod. Source pressure also depends on EPI: it is not made
    periodic by this result. Public dataclasses remain detached records.
    """
    if type(phase_states) is not tuple or len(phase_states) < 2:
        raise ValueError("a phase orbit requires an ordered path including its repeated endpoint")
    if type(cycle_start) is not int or not 0 <= cycle_start < len(phase_states) - 1:
        raise ValueError("cycle_start must index a nonempty terminal cycle")
    states = tuple(_phases(phase) for phase in phase_states)
    if _signature(states[-1]) != _signature(states[cycle_start]):
        raise ValueError("the terminal phase tuple must exactly repeat the declared cycle start")
    steps = []
    for before, expected in zip(states[:-1], states[1:], strict=True):
        step = observe_c6_coupling_coherence_phase_step(phase=before)
        if _signature(step.phase_after_coherence) != _signature(expected):
            raise ValueError("supplied phase transition differs from the production proposal replay")
        steps.append(step)
    return C6CouplingCoherencePhaseOrbit(states, tuple(steps), cycle_start, len(steps) - cycle_start)
