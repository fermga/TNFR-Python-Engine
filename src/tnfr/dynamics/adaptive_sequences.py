"""Reproducible adaptive selection among grammar-valid TNFR words.

The selector applies an epsilon-greedy heuristic to a fixed catalogue of
standalone U1-U6 words and bounded history. It does not prove that the selected
word is optimal for the current graph state; runtime preconditions still apply.
"""

from __future__ import annotations

import math
import random
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..rng import resolve_graph_seed, validate_seed

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
    """Select grammar-valid operator words from context and observed scores.

    The epsilon-greedy policy balances the greatest recorded mean score with a
    selected exploration probability. It is a reproducible policy heuristic,
    not an optimality certificate.

    **Selection Strategy:**

    - **Exploitation (80%)**: Choose sequence with best historical performance
    - **Exploration (20%)**: Random selection to discover new patterns

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the node
    node : NodeId
        Identifier of the node.
    seed : int, optional
        Explicit selector seed. When omitted, the graph RANDOM_SEED is resolved
        once and reused.

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

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        seed: int | None = None,
    ) -> None:
        self.G = graph
        self.node = node
        self.seed = (
            resolve_graph_seed(graph)
            if seed is None
            else validate_seed(seed, allow_none=False)
        )
        self._rng = random.Random(self.seed)

        # Every entry is a standalone grammar-valid U1-U6 word. Runtime
        # preconditions, including U3 phase checks and ZHIR evidence, remain
        # the executor's responsibility.
        self.sequences: dict[str, list[str]] = {
            "basic_activation": [EMISSION, COHERENCE, SILENCE],
            "deep_learning": [
                EMISSION,
                RECEPTION,
                COHERENCE,
                DISSONANCE,
                COHERENCE,
                SILENCE,
            ],
            "exploration": [
                EMISSION,
                DISSONANCE,
                COHERENCE,
                SILENCE,
            ],
            "consolidation": [EMISSION, COHERENCE, RECURSIVITY],
            "mutation": [
                EMISSION,
                COHERENCE,
                DISSONANCE,
                MUTATION,
                COHERENCE,
                SILENCE,
            ],
        }

        self.performance_scores: dict[str, list[float]] = {
            key: [] for key in self.sequences
        }
        # Compatibility alias. Values are generic observed scores, not
        # necessarily changes in canonical C(t).
        self.performance = self.performance_scores

    def select_sequence(self, context: dict[str, Any]) -> list[str]:
        """Select a sequence from context and historical performance.

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
        - **growth**: Emission-led reception and controlled exploration words
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

        if self._rng.random() < epsilon:
            # Exploration: random selection from the instance-local stream.
            selected = self._rng.choice(candidates)
        else:
            # Exploitation: select best-performing sequence
            avg_perf = {
                k: (
                    math.fsum(
                        value / len(self.performance_scores[k])
                        for value in self.performance_scores[k]
                    )
                    if self.performance_scores[k]
                    else 0.0
                )
                for k in candidates
            }
            selected = max(avg_perf, key=avg_perf.get)  # type: ignore[arg-type]

        return list(self.sequences[selected])

    def record_performance(
        self,
        sequence_name: str,
        coherence_gain: float | None = None,
        *,
        performance_score: float | None = None,
    ) -> None:
        """Record one finite observed score in a 20-sample sliding window.

        The coherence_gain name is retained as a compatibility keyword. The
        value is a generic selection score unless the caller explicitly
        supplies a measured change in canonical C(t).
        """
        if coherence_gain is not None and performance_score is not None:
            raise ValueError(
                "coherence_gain and performance_score are aliases; provide one"
            )
        value = (
            performance_score
            if performance_score is not None
            else coherence_gain
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError("performance score must be a finite real")
        if sequence_name not in self.performance_scores:
            raise KeyError(f"unknown sequence name: {sequence_name}")
        history = self.performance_scores[sequence_name]
        history.append(float(value))
        del history[:-20]
