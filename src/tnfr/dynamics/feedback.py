"""Structural feedback loops for TNFR adaptive dynamics.

This module implements feedback loops that automatically adjust nodal parameters
based on current structural state. Feedback loops enable autonomous regulation
and homeostatic cycles as specified in TNFR dynamics theory.

The core principle: ΔNFR → operator selection → application → measure effect →
adjust thresholds, creating closed-loop structural regulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..operators.registry import get_operator_class
from ..config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    SELF_ORGANIZATION,
    SILENCE,
)

__all__ = ["StructuralFeedbackLoop"]


class StructuralFeedbackLoop:
    """Feedback loop that adapts nodal dynamics based on structural state.

    This class implements closed-loop regulation where the system measures its
    current coherence state and selects appropriate operators to maintain
    target coherence levels. The feedback loop adjusts thresholds adaptively
    based on performance.

    **Feedback Cycle:**

    1. **Measure**: Compute current coherence from ΔNFR and local state
    2. **Decide**: Select operator based on deviation from target
    3. **Act**: Apply selected operator
    4. **Learn**: Adjust thresholds based on achieved coherence

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the regulated node
    node : NodeId
        Identifier of the node to regulate
    target_coherence : float, default=0.7
        Target coherence level (C_target)
    tau_adaptive : float, default=0.1
        Initial bifurcation threshold (adaptive)
    learning_rate : float, default=0.05
        Rate of threshold adaptation

    Attributes
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    target_coherence : float
        Target C(t) for homeostasis
    tau_adaptive : float
        Adaptive bifurcation threshold
    learning_rate : float
        Threshold adjustment rate

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.feedback import StructuralFeedbackLoop
    >>> G, node = create_nfr("test_node")
    >>> loop = StructuralFeedbackLoop(G, node, target_coherence=0.7)
    >>> operator_name = loop.regulate()
    >>> loop.homeostatic_cycle(num_steps=5)
    """

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        target_coherence: float = 0.7,
        tau_adaptive: float = 0.1,
        learning_rate: float = 0.05,
    ) -> None:
        self.G = graph
        self.node = node
        self.target_coherence = float(target_coherence)
        self.tau_adaptive = float(tau_adaptive)
        self.learning_rate = float(learning_rate)

    def regulate(self) -> str:
        """Select appropriate operator based on current structural state.

        Decision logic follows TNFR canonical regulation principles:

        - **Low coherence**: Stabilize with IL (Coherence)
        - **High coherence**: Explore with OZ (Dissonance)
        - **High ΔNFR**: Self-organize with THOL
        - **Low EPI**: Activate with AL (Emission)
        - **Stable**: Consolidate with SHA (Silence)

        Returns
        -------
        str
            Operator name to apply

        Notes
        -----
        The regulation logic implements structural decision-making based on
        current node state. It avoids arbitrary choices by following TNFR
        coherence principles.
        """
        dnfr = get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)
        epi = get_attr(self.G.nodes[self.node], ALIAS_EPI, 0.0)

        # Compute local coherence estimate
        coherence = self._compute_local_coherence()

        # Structural decision tree
        if coherence < self.target_coherence - 0.2:
            # Very low coherence → stabilize
            return COHERENCE
        elif coherence > self.target_coherence + 0.1:
            # High coherence → explore
            return DISSONANCE
        elif dnfr > 0.15:
            # High reorganization pressure → self-organize
            return SELF_ORGANIZATION
        elif epi < 0.3:
            # Low activation → emit
            return EMISSION
        else:
            # Stable state → consolidate
            return SILENCE

    def _compute_local_coherence(self) -> float:
        """Estimate local coherence from ΔNFR.

        Coherence is inversely proportional to reorganization pressure.
        When ΔNFR is low, coherence is high (structure is stable).

        Returns
        -------
        float
            Estimated coherence in [0, 1]
        """
        dnfr = get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)
        # Coherence inversely proportional to |ΔNFR|
        return max(0.0, min(1.0, 1.0 - abs(dnfr)))

    def adapt_thresholds(self, performance_metric: float) -> None:
        """Adapt thresholds based on achieved performance.

        Uses proportional feedback control to adjust tau_adaptive toward
        target coherence. This implements learning in the feedback loop.

        Parameters
        ----------
        performance_metric : float
            Achieved coherence or other performance measure

        Notes
        -----
        Threshold adaptation follows:

        .. math::

            \\tau_{t+1} = \\tau_t + \\alpha (C_{target} - C_{achieved})

        where α is the learning rate.
        """
        error = self.target_coherence - performance_metric

        # Proportional adjustment
        self.tau_adaptive += self.learning_rate * error

        # Clamp to valid range
        self.tau_adaptive = max(0.05, min(0.25, self.tau_adaptive))

    def homeostatic_cycle(self, num_steps: int = 10) -> None:
        """Execute homeostatic regulation cycle.

        Maintains target coherence through repeated sense-decide-act-learn cycles.

        Parameters
        ----------
        num_steps : int, default=10
            Number of regulation steps

        Notes
        -----
        Each step:

        1. Measures current coherence
        2. Selects operator via regulate()
        3. Applies operator
        4. Measures new coherence
        5. Adapts thresholds

        This implements autonomous structural homeostasis.
        """
        for step in range(num_steps):
            # Measure state before
            coherence_before = self._compute_local_coherence()

            # Select and apply operator
            operator_name = self.regulate()
            operator_class = get_operator_class(operator_name)
            operator = operator_class()
            operator(self.G, self.node, tau=self.tau_adaptive)

            # Measure state after
            coherence_after = self._compute_local_coherence()

            # Adapt thresholds based on performance
            self.adapt_thresholds(coherence_after)
