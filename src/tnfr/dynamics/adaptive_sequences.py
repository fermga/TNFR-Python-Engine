"""Adaptive sequence selection for TNFR operator trajectories.

This module implements learning-based selection of operator sequences.
Rather than executing fixed sequences, the system learns which sequences
work best for given contexts and adapts its selection over time.

The approach combines predefined canonical sequences with epsilon-greedy
exploration to balance exploitation (use known good sequences) with
exploration (try new patterns).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ..config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    SILENCE,
    TRANSITION,
)

__all__ = ["AdaptiveSequenceSelector"]


class AdaptiveSequenceSelector:
    """Learns and selects optimal operator sequences based on context.

    This class maintains a pool of canonical operator sequences and tracks
    their performance over time. It uses epsilon-greedy selection to balance
    exploitation of known-good sequences with exploration of alternatives.

    **Selection Strategy:**

    - **Exploitation (80%)**: Choose sequence with best historical performance
    - **Exploration (20%)**: Random selection to discover new patterns

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the node
    node : NodeId
        Identifier of the node

    Attributes
    ----------
    G : TNFRGraph
        Graph reference
    node : NodeId
        Node identifier
    sequences : dict[str, list[str]]
        Pool of canonical operator sequences
    performance : dict[str, list[float]]
        Historical performance for each sequence

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.adaptive_sequences import AdaptiveSequenceSelector
    >>> G, node = create_nfr("test_node")
    >>> selector = AdaptiveSequenceSelector(G, node)
    >>> context = {"goal": "stability", "urgency": 0.5}
    >>> sequence = selector.select_sequence(context)
    >>> selector.record_performance("basic_activation", 0.85)
    """

    def __init__(self, graph: TNFRGraph, node: NodeId) -> None:
        self.G = graph
        self.node = node

        # Canonical operator sequences
        # Note: Sequences are designed to comply with TNFR grammar rules
        self.sequences: Dict[str, List[str]] = {
            "basic_activation": [EMISSION, COHERENCE],
            "deep_learning": [EMISSION, RECEPTION, COHERENCE],
            "exploration": [EMISSION, DISSONANCE, COHERENCE],
            "consolidation": [COHERENCE, SILENCE, RECURSIVITY],
            "mutation": [COHERENCE, MUTATION, TRANSITION, COHERENCE],
        }

        # Performance history: sequence_name -> [coherence_gains]
        self.performance: Dict[str, List[float]] = {k: [] for k in self.sequences.keys()}

    def select_sequence(self, context: Dict[str, Any]) -> List[str]:
        """Select optimal sequence based on context and historical performance.

        Uses goal-based filtering and epsilon-greedy selection:

        1. Filter sequences appropriate for goal
        2. With probability 0.8: select best-performing sequence
        3. With probability 0.2: select random sequence (exploration)

        Parameters
        ----------
        context : dict
            Context with keys:

            - **goal** (str): "stability", "growth", or "adaptation"
            - **urgency** (float): Urgency level (0-1), currently unused

        Returns
        -------
        list[str]
            Sequence of operator names to execute

        Notes
        -----
        Goal-to-sequence mapping follows TNFR principles:

        - **stability**: Sequences emphasizing IL (Coherence) and SHA (Silence)
        - **growth**: Sequences with AL (Emission) and THOL (Self-organization)
        - **adaptation**: Sequences with ZHIR (Mutation) and learning cycles
        """
        goal = context.get("goal", "stability")

        # Map goals to appropriate sequence candidates
        if goal == "stability":
            candidates = ["basic_activation", "consolidation"]
        elif goal == "growth":
            candidates = ["deep_learning", "exploration"]
        elif goal == "adaptation":
            candidates = ["mutation", "deep_learning"]
        else:
            candidates = list(self.sequences.keys())

        # Epsilon-greedy selection (20% exploration, 80% exploitation)
        epsilon = 0.2

        if np is not None:
            random_val = np.random.random()
        else:
            random_val = random.random()

        if random_val < epsilon:
            # Exploration: random selection
            if np is not None:
                selected = str(np.random.choice(candidates))
            else:
                selected = random.choice(candidates)
        else:
            # Exploitation: select best-performing sequence
            avg_perf = {
                k: (
                    sum(self.performance[k]) / len(self.performance[k])
                    if self.performance[k]
                    else 0.0
                )
                for k in candidates
            }
            selected = max(avg_perf, key=avg_perf.get)  # type: ignore[arg-type]

        return self.sequences[selected]

    def record_performance(self, sequence_name: str, coherence_gain: float) -> None:
        """Record performance metric for a sequence to enable learning.

        Parameters
        ----------
        sequence_name : str
            Name of the sequence that was executed
        coherence_gain : float
            Achieved coherence improvement or other performance metric

        Notes
        -----
        Maintains a sliding window of the last 20 executions to adapt to
        changing dynamics. Older performance data is discarded.
        """
        if sequence_name in self.performance:
            self.performance[sequence_name].append(float(coherence_gain))
            # Keep only last 20 executions (sliding window)
            self.performance[sequence_name] = self.performance[sequence_name][-20:]
