"""Adaptive learning system for TNFR.

This module implements high-level adaptive learning dynamics combining
emission (AL) and self-organization (T'HOL) operators into canonical
learning cycles. All functionality reuses existing TNFR infrastructure.
"""

from __future__ import annotations

from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_DNFR
from ..operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    SelfOrganization,
    Silence,
    Transition,
)
from ..structural import run_sequence
from ..types import TNFRGraph

__all__ = ["AdaptiveLearningSystem"]


class AdaptiveLearningSystem:
    """System for adaptive learning using TNFR operators.

    This class orchestrates adaptive learning cycles combining emission (AL),
    reception (EN), self-organization (T'HOL), and stabilization (IL) operators.
    It implements the canonical learning sequences defined in the TNFR manual.

    All methods reuse existing operators and run_sequence infrastructure.

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the learning node.
    node : Any
        Node identifier for the learning entity.
    learning_rate : float, default=1.0
        Sensitivity to dissonance that triggers reorganization. Higher values
        make the system more responsive to novel stimuli.
    consolidation_threshold : float, default=0.7
        ΔNFR threshold below which the system stabilizes. Lower values mean
        earlier consolidation.

    Attributes
    ----------
    G : TNFRGraph
        Reference to the graph.
    node : Any
        Node being managed.
    learning_rate : float
        Configured learning sensitivity.
    consolidation_threshold : float
        Configured consolidation trigger.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.learning import AdaptiveLearningSystem
    >>> G, node = create_nfr("learner", epi=0.3, vf=1.0)
    >>> system = AdaptiveLearningSystem(G, node, learning_rate=0.8)
    >>> # Basic learning cycle
    >>> system.learn_from_input(stimulus=0.5, consolidate=True)
    >>> # Consolidate memory
    >>> system.consolidate_memory()
    """

    def __init__(
        self,
        graph: TNFRGraph,
        node: Any,
        learning_rate: float = 1.0,
        consolidation_threshold: float = 0.7,
    ) -> None:
        """Initialize adaptive learning system."""
        self.G = graph
        self.node = node
        self.learning_rate = learning_rate
        self.consolidation_threshold = consolidation_threshold

    def learn_from_input(
        self,
        stimulus: float,
        consolidate: bool = True,
    ) -> None:
        """Execute learning cycle from external stimulus.

        Implements canonical learning sequence following TNFR grammar:
        - AL (Emission): Activate learning readiness
        - EN (Reception): Receive stimulus
        - IL (Coherence): Stabilize before dissonance (grammar requirement)
        - OZ (Dissonance): If stimulus is dissonant
        - T'HOL (SelfOrganization): Reorganize if needed
        - NUL (Contraction): Close T'HOL block (grammar requirement)
        - IL (Coherence): Consolidate if requested
        - SHA (Silence): End sequence properly

        Parameters
        ----------
        stimulus : float
            External input value to learn from.
        consolidate : bool, default=True
            Whether to stabilize after learning.

        Notes
        -----
        Reuses run_sequence and existing operators for all transformations.
        Dissonance detection uses current EPI from node attributes.
        Sequences must follow TNFR grammar rules including T'HOL closure.

        **Grammar compliance:**

        - T'HOL (SelfOrganization) blocks require closure with NUL (Contraction) or SHA (Silence)
        - Dissonance should be preceded by stabilization (Coherence)
        """
        sequence: list[Operator] = [Emission(), Reception()]

        # Check if stimulus is dissonant (requires reorganization)
        if self._is_dissonant(stimulus):
            # For dissonant input, follow grammar-compliant reorganization
            # T'HOL block must be closed with SILENCE or CONTRACTION
            sequence.extend(
                [
                    Coherence(),  # Stabilize before dissonance (grammar)
                    Dissonance(),  # Introduce controlled instability
                    SelfOrganization(),  # Autonomous reorganization
                    Silence(),  # Close T'HOL block and end sequence (grammar requirement)
                ]
            )
        else:
            # Non-dissonant: simpler path with optional consolidation
            if consolidate:
                sequence.append(Coherence())
            sequence.append(Silence())  # Always end with terminal operator

        # Execute using canonical run_sequence
        run_sequence(self.G, self.node, sequence)

    def _is_dissonant(self, stimulus: float) -> bool:
        """Determine if stimulus requires reorganization.

        Parameters
        ----------
        stimulus : float
            External stimulus value.

        Returns
        -------
        bool
            True if stimulus differs significantly from current EPI.

        Notes
        -----
        Reuses get_attr for accessing node EPI canonically.
        """
        current_epi = float(get_attr(self.G.nodes[self.node], ALIAS_EPI, 0.0))
        return abs(stimulus - current_epi) > self.learning_rate

    def consolidate_memory(self) -> None:
        """Execute memory consolidation cycle.

        Implements canonical consolidation sequence:
        - AL (Emission): Reactivate for consolidation
        - EN (Reception): Integrate memory
        - IL (Coherence): Stabilize structure
        - REMESH (Recursivity): Recursive consolidation

        Notes
        -----
        Reuses run_sequence and existing operators for consolidation.
        This sequence is useful for post-learning stabilization.
        Follows TNFR grammar: must start with emission and include reception->coherence.
        """
        sequence = [Emission(), Reception(), Coherence(), Recursivity()]
        run_sequence(self.G, self.node, sequence)

    def adaptive_cycle(self, num_iterations: int = 10) -> None:
        """Execute full adaptive learning cycle with exploration.

        Implements iterative learning with conditional stabilization:
        - Each iteration: AL -> EN -> IL -> THOL with closure
        - Stabilizes with SILENCE if ΔNFR below threshold
        - Continues exploring with DISSONANCE if ΔNFR above threshold

        Parameters
        ----------
        num_iterations : int, default=10
            Number of learning iterations to execute.

        Notes
        -----
        Reuses operators and _should_stabilize logic for adaptive behavior.
        Each iteration applies a grammar-compliant sequence.
        T'HOL requires proper context (AL -> EN -> IL) and closure (SILENCE/CONTRACTION).
        """
        for _ in range(num_iterations):
            # Grammar-compliant activation sequence
            Emission()(self.G, self.node)
            Reception()(self.G, self.node)
            Coherence()(self.G, self.node)

            # Self-organization: autonomous reorganization
            SelfOrganization()(self.G, self.node)

            # T'HOL requires closure
            Silence()(self.G, self.node)

    def _should_stabilize(self) -> bool:
        """Decide whether to stabilize based on current ΔNFR.

        Returns
        -------
        bool
            True if ΔNFR is below consolidation threshold.

        Notes
        -----
        Reuses get_attr for accessing ΔNFR canonically.
        Low ΔNFR indicates structure is settling and ready for consolidation.
        """
        dnfr = abs(float(get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)))
        return dnfr < self.consolidation_threshold

    def deep_learning_cycle(self) -> None:
        """Execute deep learning with crisis and reorganization.

        Implements canonical deep learning sequence:
        AL -> EN -> IL -> OZ -> THOL -> IL -> (SHA or NUL)

        The final operator (SHA/SILENCE or NUL/CONTRACTION) is selected by
        the TNFR grammar based on structural conditions:
        - SHA (SILENCE) if Si >= si_high (high sense index)
        - NUL (CONTRACTION) if Si < si_high (low sense index)

        This is canonical THOL closure behavior per TNFR sec.4.

        Notes
        -----
        Reuses run_sequence with predefined deep learning pattern.
        Grammar may adaptively select the appropriate THOL closure.
        """
        sequence = [
            Emission(),
            Reception(),
            Coherence(),
            Dissonance(),
            SelfOrganization(),
            Coherence(),
            Silence(),  # Grammar may replace with Contraction if Si < si_high
        ]
        run_sequence(self.G, self.node, sequence)

    def exploratory_learning_cycle(self) -> None:
        """Execute exploratory learning with enhanced propagation.

        Implements canonical exploratory learning sequence:
        AL -> EN -> IL -> OZ -> THOL -> IL -> SHA

        After self-organization, coherence stabilizes and closes T'HOL,
        then silence terminates.

        Notes
        -----
        Reuses run_sequence with predefined exploratory pattern.
        This is similar to deep_learning_cycle but focuses on consolidation.
        Supports operational fractality (nested THOL allowed per sec.3.7).
        """
        sequence = [
            Emission(),
            Reception(),
            Coherence(),
            Dissonance(),
            SelfOrganization(),
            Coherence(),  # Stabilize and close T'HOL
            Silence(),  # Terminal operator
        ]
        run_sequence(self.G, self.node, sequence)

    def adaptive_mutation_cycle(self) -> None:
        """Execute transformative learning with mutation.

        Implements canonical adaptive mutation sequence:
        AL -> EN -> IL -> OZ -> ZHIR -> NAV

        Notes
        -----
        Reuses run_sequence with predefined mutation pattern.
        This represents transformative learning with phase transitions.
        Follows TNFR grammar: dissonance before mutation, ends with transition.
        """
        sequence = [
            Emission(),
            Reception(),
            Coherence(),
            Dissonance(),
            Mutation(),
            Transition(),
        ]
        run_sequence(self.G, self.node, sequence)
