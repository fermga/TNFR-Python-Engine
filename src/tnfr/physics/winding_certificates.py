"""Branch-aware winding observations for declared oriented cycles.

The legacy integer winding helper remains unchanged. Missing cycles and phase
differences on the wrap branch are reported as undefined rather than rounded
into an apparent invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from ..alias import get_attr
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
    whether the topological winding itself is defined.
    """
    nodes = tuple(cycle_nodes)
    convention = "wrapped phase differences in [-pi, pi)"
    if branch_tolerance is None:
        branch_tolerance = math.sqrt(float.fromhex("0x1.0p-52")) * math.pi
    if not math.isfinite(branch_tolerance) or branch_tolerance < 0.0:
        raise ValueError("branch_tolerance must be finite and nonnegative")
    if not math.isfinite(phase_gate) or phase_gate < 0.0:
        raise ValueError("phase_gate must be finite and nonnegative")

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

    differences = []
    for source, target in zip(nodes, nodes[1:] + nodes[:1]):
        source_phase = float(get_attr(graph.nodes[source], ALIAS_THETA, 0.0))
        target_phase = float(get_attr(graph.nodes[target], ALIAS_THETA, 0.0))
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

    Every operator receives its actual :class:`ValidatedSequenceStep`; live
    preconditions, including the hard U3 gate for Coupling and Resonance,
    remain authoritative.  A configured DeltaNFR hook is called exactly as in
    the structural sequence runner.
    """
    from ..operators.grammar_execution import ValidatedSequence

    word = tuple(operators)
    context = {"initial_epi_nonzero": True}
    validated = ValidatedSequence(word, context=context)
    cycle = tuple(cycle_nodes)
    initial = certify_phase_winding(graph, cycle)
    history_before = tuple(graph.nodes[node].get("glyph_history", ()))
    compute = graph.graph.get("compute_delta_nfr")
    steps = []
    for index, operator in enumerate(word):
        phases_before = {
            item: float(get_attr(graph.nodes[item], ALIAS_THETA, 0.0))
            for item in graph.nodes()
        }
        edges_before = graph.number_of_edges()
        operator(graph, node, sequence_context=validated.step(index))
        if callable(compute):
            compute(graph)
        phase_changes = tuple(
            (
                item,
                float(
                    angle_diff(
                        float(get_attr(graph.nodes[item], ALIAS_THETA, 0.0)),
                        phases_before[item],
                    )
                ),
            )
            for item in graph.nodes()
            if float(get_attr(graph.nodes[item], ALIAS_THETA, 0.0))
            != phases_before[item]
        )
        edges_after = graph.number_of_edges()
        steps.append(
            WindingStepObservation(
                operator=operator.glyph.value,
                certificate=certify_phase_winding(graph, cycle),
                phase_changes=phase_changes,
                edge_count_before=edges_before,
                edge_count_after=edges_after,
                topology_changed=edges_before != edges_after,
            )
        )
    requested = tuple(operator.glyph.value for operator in word)
    actual = tuple(graph.nodes[node].get("glyph_history", ()))
    actual = actual[len(history_before):]
    return WindingWordObservation(
        initial, tuple(steps), requested, actual, requested == actual
    )
