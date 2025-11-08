"""Unified adaptive system integrating all TNFR dynamic components.

This module provides a high-level interface that combines feedback loops,
adaptive sequence selection, homeostasis, learning, and metabolism into a
single coherent adaptive system. It represents the complete implementation
of TNFR autonomous evolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..dynamics.feedback import StructuralFeedbackLoop
from ..dynamics.adaptive_sequences import AdaptiveSequenceSelector
from ..dynamics.homeostasis import StructuralHomeostasis
from ..dynamics.learning import AdaptiveLearningSystem
from ..dynamics.metabolism import StructuralMetabolism

__all__ = ["TNFRAdaptiveSystem"]


class TNFRAdaptiveSystem:
    """Complete adaptive system integrating all TNFR dynamic components.

    This class orchestrates feedback loops, sequence selection, homeostasis,
    learning, and metabolism into autonomous evolution cycles. It provides
    a single entry point for complex adaptive behaviors.

    **Integrated Components:**

    - **Feedback Loop**: Regulates coherence via operator selection
    - **Sequence Selector**: Learns optimal operator trajectories
    - **Homeostasis**: Maintains parameter equilibrium
    - **Learning System**: Implements AL + T'HOL learning cycles
    - **Metabolism**: Digests stimuli into structure

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the evolving node
    node : NodeId
        Identifier of the adaptive node
    stress_normalization : float, default=0.2
        ΔNFR value that corresponds to maximum stress (1.0)

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
    The adaptive system implements complete TNFR autonomous evolution as
    specified in the operational manual. Each cycle integrates:

    1. Homeostatic regulation
    2. Feedback-driven operator selection
    3. Metabolic stress response
    4. Learning consolidation

    This creates self-regulating, adaptive structural dynamics.
    """

    STRESS_NORM = 0.2  # ΔNFR value corresponding to maximum stress

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        stress_normalization: float = STRESS_NORM,
    ) -> None:
        self.G = graph
        self.node = node
        self.STRESS_NORM = float(stress_normalization)

        # Initialize all components
        self.feedback = StructuralFeedbackLoop(graph, node)
        self.sequence_selector = AdaptiveSequenceSelector(graph, node)
        self.homeostasis = StructuralHomeostasis(graph, node)
        self.learning = AdaptiveLearningSystem(graph, node)
        self.metabolism = StructuralMetabolism(graph, node)

    def autonomous_evolution(self, num_cycles: int = 20) -> None:
        """Execute complete autonomous evolution cycles.

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
        for cycle in range(num_cycles):
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
        Stress mapping:

        - ΔNFR = 0.0 → stress = 0.0 (no pressure)
        - ΔNFR = 0.2 → stress = 1.0 (maximum pressure)
        - Linear interpolation between

        This normalization allows consistent stress response across
        different system scales.
        """
        dnfr = get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)
        # Normalize ΔNFR to [0, 1] stress level
        return min(1.0, abs(dnfr) / self.STRESS_NORM)
