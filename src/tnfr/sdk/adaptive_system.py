"""Facade over TNFR feedback, selection, homeostasis, learning and metabolism.

The default evolution loop runs homeostasis and feedback. Learning and
metabolism are exposed as components for explicit invocation; their presence
does not imply that every adaptive mechanism runs in each cycle.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..dynamics.adaptive_sequences import AdaptiveSequenceSelector
from ..dynamics.feedback import StructuralFeedbackLoop
from ..dynamics.homeostasis import StructuralHomeostasis
from ..dynamics.learning import AdaptiveLearningSystem
from ..dynamics.metabolism import StructuralMetabolism

__all__ = ["TNFRAdaptiveSystem"]


def _finite_pressure(value: object) -> float:
    """Return one finite real DeltaNFR value without coercing booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError("DeltaNFR must be a finite real")
    return float(value)


class TNFRAdaptiveSystem:
    """High-level facade for TNFR adaptive components.

    The default cycle orchestrates homeostasis and feedback. Sequence
    selection, learning and metabolism remain available through their
    component attributes for caller-controlled execution.

    **Integrated Components:**

    - **Feedback Loop**: Regulates coherence via operator selection
    - **Sequence Selector**: Scores a catalogue of grammar-valid words
    - **Homeostasis**: Maintains parameter equilibrium
    - **Learning System**: Implements AL + T'HOL learning cycles
    - **Metabolism**: Digests stimuli into structure

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the evolving node
    node : NodeId
        Identifier of the adaptive node
    stress_normalization : float, default=0.1
        Positive pressure magnitude mapped to the saturated stress value 1.0.
    random_seed : int, optional
        Seed for the adaptive sequence selector. The graph RANDOM_SEED is used
        when omitted.

    Attributes
    ----------
    G : TNFRGraph
        Graph reference
    node : NodeId
        Node identifier
    feedback : StructuralFeedbackLoop
        Feedback regulation component
    sequence_selector : AdaptiveSequenceSelector
        Adaptive sequence selection component
    homeostasis : StructuralHomeostasis
        Homeostatic regulation component
    learning : AdaptiveLearningSystem
        Adaptive learning component
    metabolism : StructuralMetabolism
        Structural metabolism component
    STRESS_NORM : float
        Normalization factor for stress measurement

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.sdk.adaptive_system import TNFRAdaptiveSystem
    >>> G, node = create_nfr("adaptive_node")
    >>> system = TNFRAdaptiveSystem(G, node)
    >>> system.autonomous_evolution(num_cycles=20)

    Notes
    -----
    The default loop integrates homeostatic regulation and feedback-driven
    operator selection. Metabolic and learning operations require explicit
    calls because they have their own runtime preconditions.
    """

    # ΔNFR normalization constant: 0.1 (operational dynamic threshold).
    # Maximum stress before structural reorganization is required.
    STRESS_NORM = 0.1

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        stress_normalization: float = STRESS_NORM,
        *,
        random_seed: int | None = None,
    ) -> None:
        if (
            isinstance(stress_normalization, bool)
            or not isinstance(stress_normalization, Real)
            or not math.isfinite(float(stress_normalization))
            or float(stress_normalization) <= 0.0
        ):
            raise ValueError("stress_normalization must be a positive finite real")
        self.G = graph
        self.node = node
        self.STRESS_NORM = float(stress_normalization)

        # Initialize all components
        self.feedback = StructuralFeedbackLoop(graph, node)
        self.sequence_selector = AdaptiveSequenceSelector(
            graph, node, seed=random_seed
        )
        self.homeostasis = StructuralHomeostasis(graph, node)
        self.learning = AdaptiveLearningSystem(graph, node)
        self.metabolism = StructuralMetabolism(graph, node)

    def autonomous_evolution(self, num_cycles: int = 20) -> None:
        """Execute the default homeostasis-and-feedback cycles.

        Each cycle integrates adaptive components:

        1. **Homeostasis**: Correct out-of-range parameters
        2. **Feedback**: Regulate coherence via operator selection

        Parameters
        ----------
        num_cycles : int, default=20
            Number of evolution cycles to execute

        Notes
        -----
        The integration follows TNFR principles:

        - **Homeostasis first**: Ensure safe operating parameters
        - **Feedback loops**: Maintain target coherence

        This creates robust, adaptive, self-regulating dynamics.

        **Advanced Usage:**

        For full metabolic and learning cycles, use the component systems
        directly:

        - ``system.metabolism.adaptive_metabolism(stress)``
        - ``system.learning.consolidate_memory()``

        These require careful sequence design to comply with TNFR grammar.
        """
        if (
            isinstance(num_cycles, bool)
            or not isinstance(num_cycles, Integral)
            or num_cycles < 0
        ):
            raise ValueError("num_cycles must be a nonnegative integer")
        for _ in range(num_cycles):
            # 1. Homeostatic regulation: maintain parameter equilibrium
            self.homeostasis.maintain_equilibrium()

            # 2. Feedback loop: regulate coherence
            self.feedback.homeostatic_cycle(num_steps=3)

    def _measure_stress(self) -> float:
        """Measure structural stress level from ΔNFR.

        Stress is proportional to reorganization pressure. High ΔNFR
        indicates high stress requiring metabolic response.

        Returns
        -------
        float
            Stress level normalized to [0, 1]

        Notes
        -----
        Stress mapping (selected operational normalization):

        - ΔNFR = 0.0 → stress = 0.0 (no pressure)
        - ΔNFR = STRESS_NORM → stress = 1.0 (maximum pressure threshold)
        - Linear interpolation between

        This normalization allows consistent stress response across
        different system scales.
        """
        dnfr = get_attr(
            self.G.nodes[self.node],
            ALIAS_DNFR,
            0.0,
            strict=True,
            conv=_finite_pressure,
        )
        return min(1.0, abs(float(dnfr)) / self.STRESS_NORM)
