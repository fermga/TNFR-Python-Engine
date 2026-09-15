"""Branch-aware winding observations for declared oriented cycles.

Missing cycles, absent or invalid phase values, and phase differences on the
wrap branch are reported as undefined rather than defaulted or rounded into an
apparent invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable

from ..constants import EPI_PRIMARY
from ..constants.aliases import ALIAS_THETA
from ..utils.numeric import angle_diff

__all__ = [
    "WindingCertificate",
    "WindingStepObservation",
    "WindingWordObservation",
    "certify_phase_winding",
    "observe_winding_word",
]


@dataclass(frozen=True)
class WindingCertificate:
    """Winding and admissibility certificate for one oriented cycle."""

    status: str
    winding: int | None
    absolute_winding: int | None
    raw_winding: float | None
    quantization_residual: float | None
    cycle_nodes: tuple[Any, ...]
    cycle_exists: bool
    orientation: str
    branch_convention: str
    minimum_branch_margin: float | None
    minimum_u3_margin: float | None
    u3_admissible: bool | None
    reason: str

    @property
    def is_defined(self) -> bool:
        """Whether winding is defined on a non-ambiguous closed cycle."""
        return self.status == "defined"


@dataclass(frozen=True)
class WindingStepObservation:
    """One executed operator step and its measured structural changes."""

    operator: str
    certificate: WindingCertificate
    phase_changes: tuple[tuple[Any, float], ...]
    edge_count_before: int
    edge_count_after: int
    topology_changed: bool


@dataclass(frozen=True)
class WindingWordObservation:
    """Actual history and winding trace for one validated operator word."""

    initial: WindingCertificate
    steps: tuple[WindingStepObservation, ...]
    requested_history: tuple[str, ...]
    actual_history: tuple[str, ...]
    history_preserved: bool


def _has_oriented_edge(graph: Any, source: Any, target: Any) -> bool:
    """Return whether the declared oriented traversal exists."""
    if graph.is_directed():
        return bool(graph.has_edge(source, target))
    return bool(graph.has_edge(source, target))


def _topology_state(graph: Any) -> tuple[Any, ...]:
    """Read node and edge identities, independent of insertion order or data."""
    directed = graph.is_directed()
    multiple = graph.is_multigraph()
    edges = graph.edges(keys=True) if multiple else graph.edges()
    identities = frozenset(
        (
            (edge[0], edge[1]) if directed else frozenset(edge[:2]),
            edge[2] if multiple else None,
        )
        for edge in edges
    )
    return directed, multiple, frozenset(graph.nodes()), identities


def _runtime_phase_gate(graph: Any, operator_code: str = "") -> float:
    """Use the shared hard graph limit and any current UM-only tightening."""
    from ..operators._phase_gate import resolve_u3_phase_limits

    _, effective_limit = resolve_u3_phase_limits(
        graph.graph, operator_code=operator_code
    )
    return effective_limit


def _finite_cycle_phase(graph: Any, node: Any) -> tuple[float | None, str | None]:
    """Read one explicit finite phase without a zero-valued fallback."""
    data = graph.nodes[node]
    alias = next((key for key in ALIAS_THETA if key in data), None)
    if alias is None:
        return None, f"phase is missing at cycle node {node!r}"
    raw = data[alias]
    if isinstance(raw, bool) or not isinstance(raw, Real):
        return None, f"phase at cycle node {node!r} must be a finite real number"
    try:
        phase = float(raw)
    except (OverflowError, ValueError):
        return None, f"phase at cycle node {node!r} must be finite"
    if not math.isfinite(phase):
        return None, f"phase at cycle node {node!r} must be finite"
    # Reduce each operand before subtraction.  Two individually finite values
    # near opposite float limits can otherwise overflow their phase difference
    # and fabricate NaN winding telemetry.
    return math.remainder(phase, math.tau), None


def _finite_nonnegative_parameter(value: Any, name: str) -> float:
    """Normalize a finite nonnegative certificate parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


def _required_phase(graph: Any, node: Any) -> float:
    """Read an explicit phase or fail before an observation is fabricated."""
    phase, error = _finite_cycle_phase(graph, node)
    if error is not None:
        raise ValueError(error)
    assert phase is not None
    return phase


def certify_phase_winding(
    graph: Any,
    cycle_nodes: Iterable[Any],
    *,
    branch_tolerance: float | None = None,
    phase_gate: float = math.pi / 2.0,
) -> WindingCertificate:
    r"""Certify signed winding on a declared oriented cycle.

    Wrapped differences use the half-open branch ``[-π, π)``.  A difference
    within ``branch_tolerance`` of the branch boundary makes the result
    undefined.  U3 admissibility is reported separately and does not determine
    whether the topological winding itself is defined. Every declared cycle
    node must carry an explicit finite real phase; no zero fallback is used.
    """
    nodes = tuple(cycle_nodes)
    convention = "wrapped phase differences in [-pi, pi)"
    if branch_tolerance is None:
        branch_tolerance = math.sqrt(float.fromhex("0x1.0p-52")) * math.pi
    branch_tolerance = _finite_nonnegative_parameter(
        branch_tolerance, "branch_tolerance"
    )
    phase_gate = _finite_nonnegative_parameter(phase_gate, "phase_gate")

    valid_nodes = len(nodes) >= 3 and len(set(nodes)) == len(nodes)
    cycle_exists = valid_nodes and all(node in graph for node in nodes)
    if cycle_exists:
        cycle_exists = all(
            _has_oriented_edge(graph, source, target)
            for source, target in zip(nodes, nodes[1:] + nodes[:1])
        )
    if not cycle_exists:
        return WindingCertificate(
            "undefined", None, None, None, None, nodes, False, "declared",
            convention, None, None, None, "declared oriented cycle is absent",
        )

    phases: dict[Any, float] = {}
    for node in nodes:
        phase, phase_issue = _finite_cycle_phase(graph, node)
        if phase_issue is not None:
            return WindingCertificate(
                status="undefined",
                winding=None,
                absolute_winding=None,
                raw_winding=None,
                quantization_residual=None,
                cycle_nodes=nodes,
                cycle_exists=True,
                orientation="declared",
                branch_convention=convention,
                minimum_branch_margin=None,
                minimum_u3_margin=None,
                u3_admissible=None,
                reason=phase_issue,
            )
        assert phase is not None
        phases[node] = phase

    differences = []
    for source, target in zip(nodes, nodes[1:] + nodes[:1]):
        source_phase = phases[source]
        target_phase = phases[target]
        differences.append(float(angle_diff(target_phase, source_phase)))
    branch_margin = min(math.pi - abs(value) for value in differences)
    gate_margin = min(phase_gate - abs(value) for value in differences)
    gate_admissible = gate_margin >= 0.0
    if branch_margin <= branch_tolerance:
        return WindingCertificate(
            "undefined", None, None, None, None, nodes, True, "declared",
            convention, branch_margin, gate_margin, gate_admissible,
            "a phase difference lies on the wrap branch boundary",
        )

    raw = math.fsum(differences) / (2.0 * math.pi)
    winding = int(round(raw))
    residual = abs(raw - winding)
    return WindingCertificate(
        "defined", winding, abs(winding), raw, residual, nodes, True,
        "declared", convention, branch_margin, gate_margin, gate_admissible,
        "winding is defined on the declared non-ambiguous cycle",
    )


def observe_winding_word(
    graph: Any,
    cycle_nodes: Iterable[Any],
    node: Any,
    operators: Iterable[Any],
) -> WindingWordObservation:
    """Execute a validated word and record each structural change.

    The shared instance and named-sequence validators run before execution,
    matching the structural sequence runner's grammar and adjacency checks.
    Every operator receives its actual :class:`ValidatedSequenceStep`; live
    preconditions, including the hard U3 gate for Coupling and Resonance,
    remain authoritative.  A configured DeltaNFR hook is called exactly as in
    the structural sequence runner. U1a admission uses the target's primary
    EPI truth value, matching that runner. Cycle U3 margins use the current
    configured hard graph gate, tightened by UM_MAX_PHASE_DIFF for a Coupling
    step; cycle-wide admissibility remains separate from a local operator's
    compatible-neighbor check. Topology changes compare node and keyed edge
    identities, including direction, rather than only edge counts.
    """
    from ..errors import TNFRValueError
    from ..operators.grammar_execution import ValidatedSequence
    from ..validation import validate_sequence

    word = tuple(operators)
    context = {
        "initial_epi_nonzero": bool(graph.nodes[node].get(EPI_PRIMARY, 0.0))
    }
    validated = ValidatedSequence(word, context=context)
    outcome = validate_sequence(validated.names, context=context)
    if not outcome.passed:
        message = outcome.summary.get("message", "validation failed")
        raise TNFRValueError(
            f"Invalid sequence: {message}",
            context={"sequence": validated.names, "outcome": outcome.summary},
        )
    cycle = tuple(cycle_nodes)
    initial = certify_phase_winding(
        graph, cycle, phase_gate=_runtime_phase_gate(graph)
    )
    if not initial.is_defined:
        raise ValueError(
            "winding word requires a defined initial cycle: " + initial.reason
        )
    for item in graph.nodes():
        _required_phase(graph, item)
    history_before = tuple(graph.nodes[node].get("glyph_history", ()))
    compute = graph.graph.get("compute_delta_nfr")
    steps = []
    for index, operator in enumerate(word):
        phases_before = {
            item: _required_phase(graph, item) for item in graph.nodes()
        }
        edges_before = graph.number_of_edges()
        topology_before = _topology_state(graph)
        operator(graph, node, sequence_context=validated.step(index))
        if callable(compute):
            compute(graph)
        phases_after = {
            item: _required_phase(graph, item) for item in graph.nodes()
        }
        phase_changes = tuple(
            (item, float(angle_diff(phases_after[item], phases_before[item])))
            for item in phases_after
            if item in phases_before
            if phases_after[item] != phases_before[item]
        )
        edges_after = graph.number_of_edges()
        steps.append(
            WindingStepObservation(
                operator=operator.glyph.value,
                certificate=certify_phase_winding(
                    graph,
                    cycle,
                    phase_gate=_runtime_phase_gate(graph, operator.glyph.value),
                ),
                phase_changes=phase_changes,
                edge_count_before=edges_before,
                edge_count_after=edges_after,
                topology_changed=topology_before != _topology_state(graph),
            )
        )
    requested = tuple(operator.glyph.value for operator in word)
    actual = tuple(graph.nodes[node].get("glyph_history", ()))
    actual = actual[len(history_before):]
    return WindingWordObservation(
        initial, tuple(steps), requested, actual, requested == actual
    )
