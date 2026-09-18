"""Declared structural-metabolism words for a caller-selected TNFR node.

These helpers compose public operators into complete grammar-validated words.
They read existing neighboring state, use configured threshold policies and
retain each operator's live checks. They neither select eligible targets nor
derive autonomous routing, future stability or guaranteed sub-EPI creation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "StructuralMetabolism",
    "digest_stimulus",
    "adaptive_metabolism",
    "cascading_reorganization",
]

_OPERATOR = "Structural metabolism"
_CASCADE_THRESHOLD_DECAY_DEFAULT = 0.6


def _node_scalar_epi(G: TNFRGraph, node: NodeId) -> float:
    """Read the shared signed scalar EPI chart before building a word."""

    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI
    from ..operators._argument_validation import finite_real, reject_operator_argument
    from ..operators._epi_domain import require_real_scalar_epi

    raw = get_attr(
        G.nodes[node],
        ALIAS_EPI,
        0.0,
        strict=True,
        conv=lambda value: value,
    )
    try:
        value = require_real_scalar_epi(raw, operator=_OPERATOR, label="node EPI")
    except Exception:
        reject_operator_argument(
            _OPERATOR, "node EPI must be a valid signed scalar embedding"
        )
    return finite_real(value, operator=_OPERATOR, label="node EPI")


def _configured_tau(G: TNFRGraph, requested: float | None) -> float:
    """Resolve THOL's threshold with the canonical configuration precedence."""

    from ..operators._thol_config import resolve_thol_bifurcation_threshold

    return resolve_thol_bifurcation_threshold(G.graph, requested, operator=_OPERATOR)


def _effective_tau(base_tau: float, metabolic_rate: Any) -> float:
    """Scale a configured threshold by a validated positive metabolic rate."""

    from ..operators._argument_validation import finite_real

    rate = finite_real(
        metabolic_rate,
        operator=_OPERATOR,
        label="metabolic_rate",
        lower=math.nextafter(0.0, math.inf),
    )
    return finite_real(
        base_tau / rate,
        operator=_OPERATOR,
        label="effective tau",
        lower=0.0,
    )


def _metabolic_steps(
    G: TNFRGraph,
    node: NodeId,
    thresholds: list[float],
    *,
    receive: bool,
) -> list[tuple[Any, dict[str, Any]]]:
    """Build one complete U1-U5 word with explicit U4b contexts."""

    from ..operators.definitions import (
        Coherence,
        Dissonance,
        Emission,
        Reception,
        SelfOrganization,
        Transition,
    )

    steps: list[tuple[Any, dict[str, Any]]] = []
    if _node_scalar_epi(G, node) == 0.0:
        steps.append((Emission(), {}))
    if receive:
        steps.append((Reception(), {}))
    for tau in thresholds:
        # OZ supplies THOL's explicit U4b perturbation context; THOL and IL
        # both absorb the opened U2 debt before the next cascade level.
        steps.extend(
            (
                (Dissonance(), {}),
                (SelfOrganization(), {"tau": tau}),
                (Coherence(), {}),
            )
        )
    # NAV declares a non-freezing U1b handoff. SHA would erase the capacity
    # required for subsequent metabolism.
    steps.append((Transition(), {}))
    return steps


def _execute_atomic_word(
    G: TNFRGraph,
    node: NodeId,
    steps: list[tuple[Any, dict[str, Any]]],
) -> None:
    """Execute an exact validated word and roll back every owned graph write."""

    from ..operators.grammar_execution import ValidatedSequence
    from ..operators.self_organization import _GraphSnapshot

    operators = [operator for operator, _ in steps]
    context = {"initial_epi_nonzero": _node_scalar_epi(G, node) != 0.0}
    word = ValidatedSequence(operators, context=context)
    snapshot = _GraphSnapshot(G)
    try:
        for index, (operator, kwargs) in enumerate(steps):
            operator(
                G,
                node,
                sequence_context=word.step(index),
                **kwargs,
            )
    except BaseException as failure:
        monitor = G.graph.get("integrity_monitor")
        discard_pending = getattr(monitor, "discard_pending_operator", None)
        if callable(discard_pending):
            try:
                discard_pending()
            except Exception:
                pass
        snapshot.restore_after_failure(G, failure)
        raise


class StructuralMetabolism:
    """Execute prescribed T'HOL-based words on an explicitly selected node.

    The caller chooses the node, method, stress or threshold inputs. The
    resulting word is a configured execution policy; canonical operators
    determine its admitted effects from the current graph and history.

    **Metabolic Characteristics:**

    - **Reception (EN)**: Reads incoming form from current neighbors
    - **Reorganization (THOL)**: Applies pressure action and any admitted birth
    - **Stabilization (IL)**: Applies the public coherence operation

    Each public method executes one complete validated word. The metabolic core
    is ``EN? -> OZ -> THOL -> IL`` and NAV supplies a non-freezing closure.

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the metabolizing node
    node : NodeId
        Identifier of the node performing metabolism

    Attributes
    ----------
    G : TNFRGraph
        Reference to the graph
    node : NodeId
        Reference to the node identifier
    metabolic_rate : float
        Positive rate that inversely scales THOL's bifurcation threshold.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import StructuralMetabolism
    >>> G, node = create_nfr("cell", epi=0.5, vf=1.0)
    >>> metabolism = StructuralMetabolism(G, node)
    >>> # Request the declared word with an explicit acceleration threshold.
    >>> metabolism.digest(tau=0.3)  # doctest: +SKIP
    >>> # The argument is a threshold, not a measured stimulus value.
    """

    def __init__(self, graph: TNFRGraph, node: NodeId) -> None:
        """Initialize structural metabolism for a node.

        Parameters
        ----------
        graph : TNFRGraph
            Graph containing the node
        node : NodeId
            Node identifier
        """
        self.G = graph
        self.node = node
        self.metabolic_rate = 1.0

    def digest(self, tau: float | None = None) -> None:
        """Execute a complete metabolic word using existing neighbor state.

        Requests the declared word: EN → OZ → THOL → IL → NAV.
        AL is prepended only when the initial scalar EPI is zero.

        1. Reception (EN): receives external stimulus from neighbors.
        2. Dissonance (OZ): opens explicit perturbation context.
        3. Self-organization (THOL): applies pressure and admitted sub-EPI effects.
        4. Coherence (IL): absorbs destabilizer debt.
        5. Transition (NAV): closes the word without suppressing capacity.

        Parameters
        ----------
        tau : float, optional
            Nonnegative THOL threshold. When omitted, the graph's canonical
            THOL threshold configuration is used.

        Notes
        -----
        The metabolic rate only rescales the THOL admission threshold. It
        does not rescale the gains or effects of the other operators. Lower
        effective tau increases configured bifurcation sensitivity. This
        policy neither derives tau from the nodal equation nor guarantees a
        birth; THOL retains its history, hierarchy and numerical checks.
        """
        threshold = _effective_tau(
            _configured_tau(self.G, tau), self.metabolic_rate
        )
        _execute_atomic_word(
            self.G,
            self.node,
            _metabolic_steps(self.G, self.node, [threshold], receive=True),
        )

    def adaptive_metabolism(self, stress_level: float) -> None:
        """Adapt metabolic response to stress level.

        Every response carries explicit OZ → THOL → IL context. Stress lowers
        the configured THOL threshold continuously, increasing bifurcation
        sensitivity without an unrelated hard stress boundary.

        Parameters
        ----------
        stress_level : float
            Level of structural stress/dissonance (0.0 to 1.0+)
            Higher nonnegative values lower THOL's threshold continuously.

        Notes
        -----
        The caller-supplied stress sets the declared policy
        ``tau / (1 + stress_level) / metabolic_rate``. It is not an inferred
        pressure measurement or a derived law for selecting a response.
        """
        from ..operators._argument_validation import finite_real

        stress = finite_real(
            stress_level,
            operator=_OPERATOR,
            label="stress_level",
            lower=0.0,
        )
        base_tau = _configured_tau(self.G, None)
        stress_adjusted_tau = finite_real(
            base_tau / (1.0 + stress),
            operator=_OPERATOR,
            label="stress-adjusted tau",
            lower=0.0,
        )
        threshold = _effective_tau(stress_adjusted_tau, self.metabolic_rate)
        _execute_atomic_word(
            self.G,
            self.node,
            _metabolic_steps(self.G, self.node, [threshold], receive=False),
        )

    def cascading_reorganization(self, depth: int = 3) -> None:
        """Repeat the declared T'HOL word on the same selected parent.

        Applies repeated OZ → THOL → IL levels with progressively decreasing
        bifurcation thresholds. A level creates a sub-EPI only when its
        observed acceleration crosses that level's threshold.

        The resulting THOL sub-EPIs retain their canonical independent-node
        identity. Repeated root applications can create sibling levels; this
        helper does not fabricate evidence to force a nested child cascade.

        Parameters
        ----------
        depth : int
            Number of cascade levels (default 3)

        Notes
        -----
        Each level uses tau = base_tau * (level_decay ^ level), creating progressively
        more sensitive bifurcation at deeper levels.

        ``depth`` counts requested repetitions, not realized hierarchy depth.
        The decay factor is configured policy, and no nested child traversal
        or refreshed physical-flow evidence is supplied by this helper.
        """
        from ..operators._argument_validation import finite_real, nonnegative_integer

        levels = nonnegative_integer(depth, operator=_OPERATOR, label="depth")
        if levels == 0:
            return
        base_tau = _configured_tau(self.G, None)
        decay = finite_real(
            self.G.graph.get(
                "THOL_CASCADE_THRESHOLD_DECAY",
                _CASCADE_THRESHOLD_DECAY_DEFAULT,
            ),
            operator=_OPERATOR,
            label="THOL_CASCADE_THRESHOLD_DECAY",
            lower=math.nextafter(0.0, math.inf),
            upper=1.0,
        )
        thresholds = [
            _effective_tau(base_tau * decay**level, self.metabolic_rate)
            for level in range(levels)
        ]
        _execute_atomic_word(
            self.G,
            self.node,
            _metabolic_steps(self.G, self.node, thresholds, receive=False),
        )


def digest_stimulus(
    G: TNFRGraph, node: NodeId, tau: float | None = None
) -> None:
    """Functional interface for single metabolic cycle.

    Equivalent to `StructuralMetabolism(G, node).digest(tau)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    tau : float, optional
        Bifurcation threshold; graph configuration is used when omitted.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import digest_stimulus
    >>> G, node = create_nfr("neuron", epi=0.4, vf=1.2)
    >>> digest_stimulus(G, node, tau=0.1)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.digest(tau)


def adaptive_metabolism(G: TNFRGraph, node: NodeId, stress: float) -> None:
    """Functional interface for adaptive metabolic response.

    Equivalent to `StructuralMetabolism(G, node).adaptive_metabolism(stress)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    stress : float
        Stress level (0.0 to 1.0+)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import adaptive_metabolism
    >>> G, node = create_nfr("organism", epi=0.6, vf=1.0)
    >>> adaptive_metabolism(G, node, stress=0.7)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.adaptive_metabolism(stress)


def cascading_reorganization(G: TNFRGraph, node: NodeId, depth: int = 3) -> None:
    """Functional interface for cascading reorganization.

    Equivalent to `StructuralMetabolism(G, node).cascading_reorganization(depth)`.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    depth : int
        Cascade depth

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.metabolism import cascading_reorganization
    >>> G, node = create_nfr("system", epi=0.7, vf=1.1)
    >>> cascading_reorganization(G, node, depth=3)  # doctest: +SKIP
    """
    metabolism = StructuralMetabolism(G, node)
    metabolism.cascading_reorganization(depth)
